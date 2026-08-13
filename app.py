"""trnslt — распознавание речи из аудио в текст.

Окно с перетаскиванием файлов: бросаешь аудио — получаешь текст, при желании
с таймингами и разделением по говорящим. Всё считается локально.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Разметку голосов считает отдельный процесс — этот же файл с ключом --diarize.
# Ключ проверяем до тяжёлых импортов: дочернему процессу не нужны ни Whisper,
# ни CUDA, а их загрузка отнимала бы около минуты на каждую запись.
if "--diarize" in sys.argv:
    import diarize

    raise SystemExit(diarize.main(sys.argv))

import exports  # noqa: E402
import ui  # noqa: E402
from common import format_time, lower_priority  # noqa: E402
from engine import (AUDIO_EXTENSIONS, LANGUAGES, MODELS, SPEAKER_COUNTS, Cancelled,  # noqa: E402
                    Engine, Line, Progress, Result, diarization_available)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except Exception:  # без tkinterdnd2 остаётся кнопка «Выбрать файлы»
    DND_AVAILABLE = False

SETTINGS_PATH = Path(os.environ.get("APPDATA", Path.home())) / "trnslt" / "settings.json"


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.engine = Engine()
        self.events: queue.Queue[tuple] = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.files: list[Path] = []
        self.results: list[Result] = []
        self.started = 0.0
        self.can_diarize = diarization_available()

        root.title("trnslt — аудио в текст")
        root.geometry("1000x760")
        root.minsize(760, 560)
        root.configure(bg=ui.BG)
        ui.apply_theme(root)

        self._build_ui()
        self._load_settings()
        self._announce_device()
        self.root.after(80, self._drain_events)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── интерфейс ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg=ui.BG, padx=20, pady=18)
        outer.pack(fill="both", expand=True)

        self._build_drop_zone(outer)
        self._build_settings(outer)
        self._build_progress(outer)
        # Кнопки прижимаем к низу до текста: иначе поле расшифровки съедает всё
        # свободное место и нижний ряд просто обрезается за краем окна.
        self._build_actions(outer)
        self._build_text(outer)

    def _build_drop_zone(self, parent) -> None:
        self.drop = tk.Label(
            parent, text="Перетащи сюда аудио или видео",
            bg=ui.CARD, fg=ui.FG, font=(ui.FONT, 13), justify="center",
            relief="flat", bd=0, pady=22, cursor="hand2",
        )
        self.drop.pack(fill="x")
        self.hint = tk.Label(parent, text="mp3 · wav · m4a · mp4 · ogg · flac — или нажми, чтобы выбрать",
                             bg=ui.BG, fg=ui.FG_DIM, font=(ui.FONT, 9))
        self.hint.pack(fill="x", pady=(6, 0))

        self.drop.bind("<Button-1>", lambda _e: self._choose_files())
        self.drop.bind("<Enter>", lambda _e: self.drop.configure(bg=ui.CARD_HOVER))
        self.drop.bind("<Leave>", lambda _e: self.drop.configure(bg=ui.CARD))

        if DND_AVAILABLE:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)
            self.drop.dnd_bind("<<DropEnter>>", lambda _e: self.drop.configure(bg=ui.CARD_HOVER))
            self.drop.dnd_bind("<<DropLeave>>", lambda _e: self.drop.configure(bg=ui.CARD))
        else:
            self.drop.configure(text="Нажми, чтобы выбрать аудио или видео")

    def _build_settings(self, parent) -> None:
        panel = tk.Frame(parent, bg=ui.BG, pady=14)
        panel.pack(fill="x")

        left = tk.Frame(panel, bg=ui.BG)
        left.pack(side="left", fill="x", expand=True)

        # Первый ряд: модель и язык с подписями сверху.
        row = tk.Frame(left, bg=ui.BG)
        row.pack(fill="x")

        model_box = tk.Frame(row, bg=ui.BG)
        model_box.pack(side="left", padx=(0, 18))
        ui.caption(model_box, "Модель распознавания").pack(anchor="w")
        self.model_var = tk.StringVar(value=MODELS[2][1])
        self.model_box = ttk.Combobox(model_box, textvariable=self.model_var, state="readonly",
                                      values=[label for _, label in MODELS], width=26,
                                      style="trnslt.TCombobox", font=(ui.FONT, 10))
        self.model_box.pack(anchor="w", pady=(3, 0))

        lang_box = tk.Frame(row, bg=ui.BG)
        lang_box.pack(side="left")
        ui.caption(lang_box, "Язык записи").pack(anchor="w")
        self.lang_var = tk.StringVar(value=LANGUAGES[0][1])
        self.lang_box = ttk.Combobox(lang_box, textvariable=self.lang_var, state="readonly",
                                     values=[label for _, label in LANGUAGES], width=20,
                                     style="trnslt.TCombobox", font=(ui.FONT, 10))
        self.lang_box.pack(anchor="w", pady=(3, 0))

        # Второй ряд: что показывать в расшифровке.
        toggles = tk.Frame(left, bg=ui.BG, pady=12)
        toggles.pack(fill="x")

        self.time_var = tk.BooleanVar(value=True)
        self.time_box = ui.Checkbox(toggles, "Тайминги", self.time_var, self._render)
        self.time_box.pack(side="left", padx=(0, 20))

        # Разделение включено сразу, число голосов — «определить самому».
        self.diar_var = tk.BooleanVar(value=self.can_diarize)
        self.diar_box = ui.Checkbox(toggles, "Разделять говорящих", self.diar_var,
                                    self._on_diar_toggle)
        self.diar_box.pack(side="left", padx=(0, 10))

        self.speakers_var = tk.StringVar(value=SPEAKER_COUNTS[0][1])
        self.speakers_box = ttk.Combobox(
            toggles, textvariable=self.speakers_var,
            state="readonly" if self.diar_var.get() else "disabled",
            values=[label for _, label in SPEAKER_COUNTS], width=18,
            style="trnslt.TCombobox", font=(ui.FONT, 10))
        self.speakers_box.pack(side="left")

        if not self.can_diarize:
            self.diar_box.set_enabled(False)
            tk.Label(toggles, text="модели разделения не найдены", bg=ui.BG, fg=ui.FG_DIM,
                     font=(ui.FONT, 9)).pack(side="left", padx=(10, 0))

        self.run_btn = ui.Button(panel, "Распознать", self._start, primary=True, padx=28, pady=12)
        self.run_btn.pack(side="right", padx=(18, 0))
        self.run_btn.set_enabled(False)

    def _build_progress(self, parent) -> None:
        self.progress = ttk.Progressbar(parent, style="trnslt.Horizontal.TProgressbar",
                                        mode="determinate", maximum=1000)
        self.progress.pack(fill="x")

        state = tk.Frame(parent, bg=ui.BG)
        state.pack(fill="x", pady=(8, 10))
        self.status = tk.Label(state, text="Файл не выбран", bg=ui.BG, fg=ui.FG_DIM,
                               font=(ui.FONT, 10), anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        # Часы работы: пока они идут, видно, что программа считает, а не повисла.
        self.elapsed_label = tk.Label(state, text="", bg=ui.BG, fg=ui.FG_DIM, font=(ui.MONO, 10))
        self.elapsed_label.pack(side="right")

    def _build_text(self, parent) -> None:
        frame = tk.Frame(parent, bg=ui.CARD)
        frame.pack(fill="both", expand=True)

        self.text = tk.Text(frame, wrap="word", bg=ui.CARD, fg=ui.FG, insertbackground=ui.FG,
                            relief="flat", bd=0, padx=18, pady=14, font=(ui.FONT, 11),
                            spacing1=3, spacing2=2, spacing3=8, undo=True, height=12,
                            selectbackground=ui.ACCENT_DIM, selectforeground=ui.FG)
        scroll = ttk.Scrollbar(frame, command=self.text.yview, style="trnslt.Vertical.TScrollbar")
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        self.text.tag_configure("header", foreground=ui.ACCENT, font=(ui.FONT, 10, "bold"),
                                spacing1=14)
        self.text.tag_configure("time", foreground=ui.FG_DIM, font=(ui.MONO, 9))
        self.text.tag_configure("body", lmargin1=0, lmargin2=0)
        for index, color in enumerate(ui.SPEAKER_COLORS):
            self.text.tag_configure(f"speaker{index}", foreground=color,
                                    font=(ui.FONT, 10, "bold"))
        self.text.tag_configure("placeholder", foreground=ui.FG_DIM, font=(ui.FONT, 11))
        self._show_placeholder()

    def _build_actions(self, parent) -> None:
        actions = tk.Frame(parent, bg=ui.BG, pady=14)
        actions.pack(fill="x", side="bottom")

        self.open_btn = ui.Button(actions, "Открыть в Word", self._open_in_word, primary=True)
        self.open_btn.pack(side="left", padx=(0, 10))
        for text, command in (("Сохранить .docx", self._save_docx),
                              ("Сохранить .txt", self._save_txt),
                              ("Копировать", self._copy),
                              ("Очистить", self._clear)):
            ui.Button(actions, text, command).pack(side="left", padx=(0, 8))

        self.device_label = tk.Label(actions, text="", bg=ui.BG, fg=ui.FG_DIM, font=(ui.FONT, 9))
        self.device_label.pack(side="right")

    def _on_diar_toggle(self) -> None:
        self.speakers_box.configure(state="readonly" if self.diar_var.get() else "disabled")
        if self.diar_var.get():
            self._set_status("Голоса размечает процессор — на часовой записи это добавит минут "
                             "пятнадцать. Знаешь число участников — укажи, будет точнее.")
        self._render()

    # ── выбор файлов ─────────────────────────────────────────────────────────

    def _on_drop(self, event) -> None:
        self.drop.configure(bg=ui.CARD)
        self._accept([Path(p) for p in self.root.tk.splitlist(event.data)])

    def _choose_files(self) -> None:
        picked = filedialog.askopenfilenames(
            title="Выбери аудио или видео",
            filetypes=[("Аудио и видео", " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)),
                       ("Все файлы", "*.*")],
        )
        if picked:
            self._accept([Path(p) for p in picked])

    def _accept(self, paths: list[Path]) -> None:
        if self._busy():
            return
        files = [p for p in paths if p.is_file()]
        skipped = [p for p in files if p.suffix.lower() not in AUDIO_EXTENSIONS]
        files = [p for p in files if p.suffix.lower() in AUDIO_EXTENSIONS]
        if not files:
            names = ", ".join(p.name for p in skipped) or "ничего подходящего"
            self._set_status(f"Не могу открыть: {names}")
            return

        self.files = files
        self.run_btn.set_enabled(True)
        if len(files) == 1:
            size = files[0].stat().st_size / 1024 / 1024
            self.drop.configure(text=files[0].name)
            self.hint.configure(text=f"{size:.1f} МБ — готов к распознаванию")
            self._set_status("Нажми «Распознать»")
        else:
            self.drop.configure(text=f"Выбрано файлов: {len(files)}")
            self.hint.configure(text="будут обработаны по очереди")
            self._set_status(f"Нажми «Распознать» — в очереди {len(files)} файлов")

    # ── запуск и остановка ───────────────────────────────────────────────────

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _pick(self, options: list[tuple], variable: tk.StringVar):
        """Возвращает значение из списка (код, подпись) по выбранной подписи."""
        labels = [label for _, label in options]
        return options[labels.index(variable.get())][0]

    def _start(self) -> None:
        if self._busy():
            self.cancel.set()
            self._set_status("Останавливаю…")
            return
        if not self.files:
            self._choose_files()
            return

        self._save_settings()
        self.cancel = threading.Event()
        self.run_btn.set_text("Остановить")
        for widget in (self.model_box, self.lang_box, self.speakers_box):
            widget.configure(state="disabled")
        self.diar_box.set_enabled(False)
        self.progress.configure(value=0)
        if not any(result.lines for result in self.results):
            self.text.delete("1.0", "end")

        model = self._pick(MODELS, self.model_var)
        language = self._pick(LANGUAGES, self.lang_var)
        speakers = self._pick(SPEAKER_COUNTS, self.speakers_var) if self.diar_var.get() else None

        self.started = time.monotonic()
        self.worker = threading.Thread(target=self._work,
                                       args=(list(self.files), model, language, speakers),
                                       daemon=True)
        self.worker.start()
        self._tick()

    def _tick(self) -> None:
        """Раз в полсекунды обновляет часы работы, пока идёт распознавание."""
        if not self._busy():
            return
        self.elapsed_label.configure(text="прошло " + format_time(time.monotonic() - self.started))
        self.root.after(500, self._tick)

    def _work(self, files: list[Path], model: str, language: str, speakers: int | None) -> None:
        """Выполняется в фоновом потоке; общение с окном — только через self.events."""
        lower_priority()
        try:
            for index, path in enumerate(files):
                if self.cancel.is_set():
                    raise Cancelled

                prefix = f"[{index + 1}/{len(files)}] " if len(files) > 1 else ""
                self.events.put(("file_start", str(path)))

                def report(progress: Progress, prefix: str = prefix) -> None:
                    self.events.put(("progress", prefix + progress.stage,
                                     progress.fraction, progress.line))

                result = self.engine.transcribe(str(path), model, language, self.cancel,
                                                report, speakers=speakers)
                self.events.put(("file_done", result))
            self.events.put(("finished", None))
        except Cancelled:
            self.events.put(("finished", "Остановлено"))
        except Exception as exc:
            self.events.put(("finished", f"Ошибка: {exc}"))

    # ── обработка событий из фонового потока ─────────────────────────────────

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]

                if kind == "file_start":
                    self.results.append(Result(path=event[1]))
                    if len(self.results) > 1:
                        self._insert_header(Path(event[1]).name)

                elif kind == "progress":
                    _, stage, fraction, line = event
                    self._set_status(stage)
                    self._set_progress(fraction)
                    if line is not None and self.results:
                        self.results[-1].lines.append(line)
                        self._insert_line(line)

                elif kind == "file_done":
                    result: Result = event[1]
                    if self.results:  # берём итоговые язык и число голосов
                        self.results[-1].language = result.language
                        self.results[-1].speakers = result.speakers
                    self._render()  # перерисовываем: реплики склеиваются по говорящим

                elif kind == "finished":
                    self._finish(event[1])

        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _finish(self, note: str | None) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.run_btn.set_text("Распознать")
        if self.started:
            self.elapsed_label.configure(
                text="заняло " + format_time(time.monotonic() - self.started))
        for widget in (self.model_box, self.lang_box):
            widget.configure(state="readonly")
        if self.can_diarize:
            self.diar_box.set_enabled(True)
            self.speakers_box.configure(state="readonly" if self.diar_var.get() else "disabled")

        if note:
            self._set_status(note)
            if note.startswith("Ошибка"):
                self.progress.configure(value=0)
                messagebox.showerror("trnslt", note)
            return

        self.progress.configure(value=1000)
        words = sum(len(line.text.split()) for r in self.results for line in r.lines)
        speakers = max((r.speakers for r in self.results), default=0)
        facts = [f"слов: {words}"]
        if speakers:
            facts.append(f"говорящих: {speakers}")
        self._set_status("Готово · " + " · ".join(facts) + " · можно открыть в Word")

    # ── вывод текста ─────────────────────────────────────────────────────────

    def _set_progress(self, fraction: float) -> None:
        if fraction < 0:
            if self.progress["mode"] != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(18)
        else:
            if self.progress["mode"] != "determinate":
                self.progress.stop()
                self.progress.configure(mode="determinate")
            self.progress.configure(value=int(fraction * 1000))

    def _show_placeholder(self) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("end", "Здесь появится расшифровка.\n\n"
                                "Тайминги и говорящих можно включать и выключать "
                                "уже после распознавания — текст перестроится сразу.",
                         "placeholder")

    def _speaker_tag(self, speaker: str | None) -> str:
        """Свой цвет каждому говорящему, по номеру из подписи «Спикер N»."""
        if not speaker:
            return "speaker0"
        digits = "".join(ch for ch in speaker if ch.isdigit())
        number = int(digits) - 1 if digits else 0
        return f"speaker{number % len(ui.SPEAKER_COLORS)}"

    def _insert_header(self, name: str) -> None:
        at_bottom = self.text.yview()[1] > 0.98
        if self.text.get("1.0", "end").strip():
            self.text.insert("end", "\n")
        self.text.insert("end", f"{name}\n", "header")
        if at_bottom:
            self.text.see("end")

    def _insert_line(self, line: Line, text: str | None = None) -> None:
        at_bottom = self.text.yview()[1] > 0.98
        show_speaker = self.diar_var.get() and line.speaker

        if show_speaker:
            self.text.insert("end", line.speaker, self._speaker_tag(line.speaker))
        if self.time_var.get():
            stamp = ("   " if show_speaker else "") + format_time(line.start)
            self.text.insert("end", stamp, "time")
        if show_speaker or self.time_var.get():
            self.text.insert("end", "\n")

        self.text.insert("end", (text if text is not None else line.text) + "\n", "body")
        if at_bottom:
            self.text.see("end")

    def _render(self) -> None:
        """Собирает текст заново — после смены переключателей или конца файла."""
        if not any(result.lines for result in self.results):
            if not self._busy():
                self._show_placeholder()
            return

        position = self.text.yview()[0]
        self.text.delete("1.0", "end")

        for result in self.results:
            if len(self.results) > 1:
                self._insert_header(Path(result.path).name)
            for turn in exports.group_turns(result.lines):
                self._insert_line(Line(turn.start, turn.end, turn.text, turn.speaker), turn.text)

        self.text.yview_moveto(position)

    # ── сохранение ───────────────────────────────────────────────────────────

    def _has_text(self) -> bool:
        if any(result.lines for result in self.results):
            return True
        self._set_status("Пока нечего сохранять")
        return False

    def _free_path(self, base: Path) -> Path:
        """Подбирает свободное имя: файл может быть уже открыт в Word."""
        candidate, index = base, 2
        while candidate.exists():
            candidate = base.with_name(f"{base.stem} ({index}){base.suffix}")
            index += 1
        return candidate

    def _open_in_word(self) -> None:
        """Складывает документ рядом с записью и сразу открывает его."""
        if not self._has_text():
            return

        source = Path(self.results[0].path)
        target = self._free_path(source.with_suffix(".docx"))
        try:
            exports.save_docx(target, self.results, self.time_var.get(), self.diar_var.get())
        except OSError:
            # Папка с записью может быть недоступна для записи — уходим в «Документы».
            target = self._free_path(Path.home() / "Documents" / f"{source.stem}.docx")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                exports.save_docx(target, self.results, self.time_var.get(), self.diar_var.get())
            except OSError as exc:
                messagebox.showerror("trnslt", f"Не удалось сохранить документ:\n{exc}")
                return

        self._set_status(f"Открываю: {target}")
        try:
            os.startfile(str(target))  # noqa: S606 — открываем свой же файл
        except OSError:
            # Word не установлен или .docx ни к чему не привязан — покажем папку.
            subprocess.Popen(["explorer", "/select,", str(target)])
            self._set_status(f"Сохранено: {target} (Word на этом компьютере не найден)")

    def _ask_path(self, extension: str, kind: str) -> Path | None:
        default = self.files[0].with_suffix(extension).name if self.files else f"расшифровка{extension}"
        chosen = filedialog.asksaveasfilename(
            title="Сохранить расшифровку", defaultextension=extension, initialfile=default,
            initialdir=str(self.files[0].parent) if self.files else None,
            filetypes=[(kind, f"*{extension}")],
        )
        return Path(chosen) if chosen else None

    def _save_docx(self) -> None:
        if not self._has_text():
            return
        path = self._ask_path(".docx", "Документ Word")
        if not path:
            return
        try:
            exports.save_docx(path, self.results, self.time_var.get(), self.diar_var.get())
        except PermissionError:
            messagebox.showerror("trnslt", f"Файл занят другой программой:\n{path}\n\n"
                                           "Закрой его в Word и сохрани ещё раз.")
            return
        self._set_status(f"Сохранено: {path}")

    def _save_txt(self) -> None:
        if not self._has_text():
            return
        path = self._ask_path(".txt", "Текстовый файл")
        if not path:
            return
        exports.save_txt(path, self.results, self.time_var.get(), self.diar_var.get())
        self._set_status(f"Сохранено: {path}")

    def _copy(self) -> None:
        if not self._has_text():
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(
            exports.render_text(self.results, self.time_var.get(), self.diar_var.get()))
        self._set_status("Текст скопирован в буфер обмена")

    def _clear(self) -> None:
        if self._busy():
            return
        self.files.clear()
        self.results.clear()
        self.progress.configure(value=0)
        self.run_btn.set_enabled(False)
        self.elapsed_label.configure(text="")
        self.drop.configure(text="Перетащи сюда аудио или видео")
        self.hint.configure(text="mp3 · wav · m4a · mp4 · ogg · flac — или нажми, чтобы выбрать")
        self._show_placeholder()
        self._set_status("Файл не выбран")

    def _set_status(self, message: str) -> None:
        self.status.configure(text=message)

    def _announce_device(self) -> None:
        device, compute = Engine.probe_device()
        human = "видеокарта" if device == "cuda" else "процессор"
        self.device_label.configure(text=f"считает {human} · {compute}")

    # ── настройки ────────────────────────────────────────────────────────────

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        # Разделение по говорящим намеренно не запоминаем: окно всегда открывается
        # с включённой галочкой и автоопределением числа голосов.
        for key, variable, options in (("model", self.model_var, MODELS),
                                       ("language", self.lang_var, LANGUAGES)):
            if data.get(key) in [label for _, label in options]:
                variable.set(data[key])
        self.time_var.set(bool(data.get("timecodes", True)))
        self.time_box.refresh()

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(
                json.dumps({"model": self.model_var.get(), "language": self.lang_var.get(),
                            "timecodes": self.time_var.get()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass  # настройки — приятный бонус, из-за них падать не стоит

    def _on_close(self) -> None:
        self.cancel.set()
        self.root.destroy()


def make_shortcut() -> int:
    """Кладёт ярлык приложения на рабочий стол.

    Запускается двойным кликом по «Ярлык на рабочий стол.bat» рядом с программой.
    Ярлык создаём через PowerShell, но команду передаём отдельным аргументом:
    так русские буквы в имени не превращаются в кракозябры по дороге через cmd.
    """
    target = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve()
    folder = target.parent
    script = (
        "$shell = New-Object -ComObject WScript.Shell; "
        "$path = Join-Path ([Environment]::GetFolderPath('Desktop')) "
        "'trnslt — аудио в текст.lnk'; "
        "$link = $shell.CreateShortcut($path); "
        f"$link.TargetPath = '{target}'; "
        f"$link.WorkingDirectory = '{folder}'; "
        f"$link.IconLocation = '{target},0'; "
        "$link.Description = 'Расшифровка аудио в текст'; "
        "$link.Save(); "
        "Write-Output $path"
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        where = (done.stdout or "").strip()
        if done.returncode != 0 or not where:
            raise RuntimeError((done.stderr or "").strip() or "PowerShell не отчитался")
    except Exception as exc:
        _tell("trnslt", f"Не получилось создать ярлык:\n{exc}\n\n"
                        "Можно сделать вручную: правой кнопкой по trnslt.exe → "
                        "«Отправить» → «Рабочий стол (создать ярлык)».")
        return 1

    _tell("trnslt", f"Ярлык на рабочем столе готов:\n{where}\n\n"
                    "Папку с программой не переименовывай и не удаляй — "
                    "ярлык ссылается на неё.")
    return 0


def _tell(title: str, message: str) -> None:
    """Показывает сообщение окошком: у собранного приложения консоли нет."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()


def selftest(audio: str, speakers: int | None = None) -> int:
    """Проверка сборки: распознаёт файл без окна и пишет отчёт в trnslt-selftest.log.

    Запуск: trnslt.exe --selftest путь\\к\\файлу.wav [число говорящих]
    У собранного приложения нет консоли, поэтому отчёт идёт в файл.
    """
    import traceback

    from engine import bundled_models

    lines: list[str] = []

    def say(message: str) -> None:
        lines.append(message)
        if sys.stdout is not None:
            print(message, flush=True)

    code = 1
    try:
        engine = Engine()
        say(f"устройство: {engine.probe_device()}")
        say(f"модели в комплекте: {', '.join(sorted(bundled_models())) or 'нет'}")
        say(f"разделение по говорящим доступно: {diarization_available()}")

        result = engine.transcribe(audio, "small", "", threading.Event(),
                                   lambda p: say("  " + p.stage), speakers=speakers)
        say(f"считало на: {engine.device} / {engine.compute_type}")
        say(f"голосов найдено: {result.speakers}")
        say("--- расшифровка ---")
        say(exports.render_text([result], show_time=True, show_speaker=speakers is not None))
        code = 0 if result.text else 1
    except Exception:
        say("ОШИБКА:\n" + traceback.format_exc())
    finally:
        Path("trnslt-selftest.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return code


def main() -> None:
    if "--shortcut" in sys.argv:
        raise SystemExit(make_shortcut())

    if "--selftest" in sys.argv:
        rest = sys.argv[sys.argv.index("--selftest") + 1:]
        speakers = int(rest[1]) if len(rest) > 1 else None
        raise SystemExit(selftest(rest[0], speakers))

    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
