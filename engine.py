"""Движок распознавания речи: обёртка над faster-whisper и sherpa-onnx.

Аудио декодируется через PyAV (входит в faster-whisper), поэтому ffmpeg
устанавливать не нужно — mp3, m4a, mp4, ogg, flac, wav читаются напрямую.
Разделение по говорящим считается на ONNX-моделях, без torch.
"""

from __future__ import annotations

import os
import site
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from common import (SAMPLE_RATE, SUSPICIOUS_SPEAKERS, app_dir, diarization_models, format_time,
                    lower_priority, models_dir, worker_threads)


def _register_cuda_dlls() -> None:
    """Показывает Windows, где лежат cuBLAS и cuDNN.

    В обычном запуске они стоят колёсами в site-packages/nvidia/*/bin, в собранном
    приложении — рядом с exe. ctranslate2 ищет cublas64_12.dll обычным поиском
    по PATH; без этой подсказки GPU отваливается с «Library ... is not found».
    """
    if sys.platform != "win32":
        return

    # В собранном приложении библиотеки лежат рядом с exe (и в подпапке cuda,
    # если кто-то доложил их туда вручную), при обычном запуске — в пакетах nvidia-*.
    candidates: list[Path] = [app_dir(), app_dir() / "cuda"]
    roots = {Path(p) for p in site.getsitepackages() if p}
    roots.add(Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages")
    for root in roots:
        candidates.extend(sorted((root / "nvidia").glob("*/bin")))

    for bin_dir in candidates:
        if not bin_dir.is_dir():
            continue
        path = str(bin_dir)
        if path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(path)
        except OSError:
            pass


_register_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402  (импорт после настройки путей к DLL)
from faster_whisper.audio import decode_audio  # noqa: E402

# Модели по возрастанию качества и требований к памяти.
# Подписи видит пользователь в выпадающем списке — держим их короткими,
# иначе строка не помещается и обрезается на середине слова.
MODELS: list[tuple[str, str]] = [
    ("tiny", "tiny — быстро, грубо"),
    ("base", "base — быстро"),
    ("small", "small — рекомендую"),
    ("medium", "medium — точнее, дольше"),
    ("large-v3", "large-v3 — лучшее качество"),
]

# Языки для выпадающего списка. Пустая строка = автоопределение.
LANGUAGES: list[tuple[str, str]] = [
    ("", "определить сам"),
    ("ru", "русский"),
    ("en", "английский"),
    ("uk", "украинский"),
    ("de", "немецкий"),
    ("fr", "французский"),
    ("es", "испанский"),
    ("it", "итальянский"),
    ("pl", "польский"),
    ("tr", "турецкий"),
    ("zh", "китайский"),
    ("ja", "японский"),
]

# Сколько говорящих искать. None = определить самостоятельно.
SPEAKER_COUNTS: list[tuple[int, str]] = [
    (0, "определить самому"),
    *[(n, f"{n} говорящих") for n in range(2, 11)],
]

# Сколько кусков речи уходит в видеокарту за раз. Замер на 17 минутах:
# по одному — 87 секунд, пачками по 8 — 46. Шестнадцать уже хуже: не хватает
# видеопамяти на 4 ГБ и начинается своп.
BATCH_SIZE = 8

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".oga", ".opus", ".flac",
    ".aac", ".wma", ".webm", ".mkv", ".avi", ".mov", ".m4b", ".amr",
)


class Cancelled(Exception):
    """Пользователь остановил распознавание."""


@dataclass
class Line:
    """Реплика: кусок расшифровки со временем и, если известно, говорящим."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Progress:
    """Состояние работы, которое показывается в окне."""

    stage: str                       # текст для строки состояния
    fraction: float                  # 0.0 .. 1.0, отрицательное = прогресс неизвестен
    line: Line | None = None         # готовая реплика, если она появилась


@dataclass
class Result:
    """Итог по одному файлу."""

    path: str
    lines: list[Line] = field(default_factory=list)
    language: str = ""
    speakers: int = 0

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines).strip()


def model_source(name: str) -> str:
    """Путь к вложенной в сборку модели, иначе имя для скачивания с Hugging Face."""
    local = models_dir() / name
    return str(local) if (local / "model.bin").is_file() else name


def bundled_models() -> set[str]:
    """Имена моделей, которые лежат внутри приложения и не требуют интернета."""
    folder = models_dir()
    if not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if (p / "model.bin").is_file()}


def diarization_available() -> bool:
    """Можно ли вообще делить запись по говорящим."""
    if diarization_models() is None:
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        return False
    return True


class Engine:
    """Держит загруженные модели, чтобы не перечитывать их с диска каждый раз."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self._key: tuple[str, str, str] | None = None
        self._batched = None
        self._lock = threading.Lock()
        self.device: str = "cpu"
        self.compute_type: str = "int8"

    @staticmethod
    def probe_device() -> tuple[str, str]:
        """Возвращает (device, compute_type): GPU, если он доступен, иначе CPU.

        На GPU берём int8_float16, а не float16: на картах без тензорных ядер
        (GTX 16xx) float16 считается вшестеро-всемеро медленнее при том же тексте.
        Если карта умеет float16 быстро, int8 тоже не проигрывает.
        """
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                supported = ctranslate2.get_supported_compute_types("cuda")
                for candidate in ("int8_float16", "float16", "int8"):
                    if candidate in supported:
                        return "cuda", candidate
        except Exception:
            pass
        return "cpu", "int8"

    # ── модели ───────────────────────────────────────────────────────────────

    def load(self, model_name: str, on_progress: Callable[[Progress], None]) -> WhisperModel:
        device, compute_type = self.probe_device()
        key = (model_name, device, compute_type)
        with self._lock:
            if self._model is not None and self._key == key:
                return self._model

            source = model_source(model_name)
            if source == model_name:
                on_progress(Progress(f"Скачиваю модель {model_name} (только в первый раз)…", -1.0))
            else:
                on_progress(Progress(f"Загружаю модель {model_name} ({device})…", -1.0))
            try:
                model = WhisperModel(source, device=device, compute_type=compute_type)
                if device == "cuda":
                    _warm_up(model)
            except Exception as exc:
                if device != "cuda":
                    raise
                # Нет видеопамяти, старый драйвер, не нашлись cuBLAS или cuDNN —
                # молча уходим на процессор, лишь бы человек получил свой текст.
                on_progress(Progress(f"GPU недоступен ({exc}); считаю на процессоре…", -1.0))
                device, compute_type = "cpu", "int8"
                key = (model_name, device, compute_type)
                model = WhisperModel(source, device=device, compute_type=compute_type)

            self._model = model
            self._key = key
            self.device = device
            self.compute_type = compute_type
            return model

    # ── работа ───────────────────────────────────────────────────────────────

    def transcribe(
        self,
        path: str,
        model_name: str,
        language: str,
        cancel: threading.Event,
        on_progress: Callable[[Progress], None],
        speakers: int | None = None,
    ) -> Result:
        """Распознаёт файл целиком.

        speakers: None — не делить по говорящим, 0 — определить число самому,
        N — искать ровно N голосов. Реплики отдаются по мере готовности
        через on_progress, а целиком возвращаются в Result.
        """
        model = self.load(model_name, on_progress)
        if cancel.is_set():
            raise Cancelled

        on_progress(Progress("Читаю аудио…", -1.0))
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE

        # Разметку голосов запускаем сразу и не ждём: она занимает процессор,
        # расшифровка — видеокарту, так что пусть работают одновременно.
        job = self._start_diarization(path, speakers, duration, on_progress)
        result = Result(path=path)
        collected = []

        segments, info = self._transcribe_segments(model, audio, language,
                                                   word_timestamps=job is not None)
        result.language = info.language
        detected = f"язык: {info.language} ({info.language_probability:.0%})"
        tail = 0.12 if job else 0.0  # хвост шкалы оставляем на ожидание разметки

        for segment in _iter_cancellable(segments, cancel, job):
            collected.append(segment)
            text = segment.text.strip()
            if not text:
                continue
            line = Line(start=segment.start, end=segment.end, text=text)
            result.lines.append(line)
            share = min(segment.end / duration, 1.0) if duration else 0.0
            note = f", голоса размечаются: {job.share():.0%}" if job and job.running else ""
            on_progress(Progress(
                f"Распознаю… {format_time(segment.end)} / {format_time(duration)}, "
                f"{detected}{note}",
                (1 - tail) * share, line=line,
            ))

        if job is not None:
            turns = self._collect_diarization(job, cancel, on_progress, tail)
            if turns:
                result.lines = [line for segment in collected
                                for line in _split_by_speaker(segment, turns)]

        result.speakers = len({line.speaker for line in result.lines if line.speaker})
        return result

    def _transcribe_segments(self, model, audio, language: str, word_timestamps: bool):
        """Расшифровка пачками, если получится: на длинной записи это вдвое быстрее.

        Пакетный режим гоняет через видеокарту сразу несколько кусков речи между
        паузами. Замер на 17 минутах: 87 секунд обычным ходом против 46 пачками.
        """
        options = dict(
            language=language or None,
            beam_size=5,
            vad_filter=True,  # пропускаем тишину: быстрее и меньше выдуманных фраз
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=word_timestamps,  # нужны, чтобы резать реплики по смене голоса
        )
        try:
            from faster_whisper import BatchedInferencePipeline

            if self._batched is None:
                self._batched = BatchedInferencePipeline(model=model)
            return self._batched.transcribe(audio, batch_size=BATCH_SIZE, **options)
        except Exception:
            # Старая версия faster-whisper или не хватило видеопамяти — идём обычным ходом.
            return model.transcribe(audio, condition_on_previous_text=False, **options)

    def _start_diarization(self, path: str, speakers: int | None, duration: float,
                           on_progress: Callable[[Progress], None]) -> Diarization | None:
        """Запускает разметку голосов фоном; при неудаче возвращает None."""
        if speakers is None:
            return None
        try:
            job = Diarization(path, speakers)
        except Exception as exc:
            on_progress(Progress(f"Разделить по говорящим не вышло ({exc}); "
                                 "расшифровываю без этого", -1.0))
            return None
        on_progress(Progress(
            f"Разбираю, кто говорит… (запись {format_time(duration)}, считает процессор)", 0.0))
        return job

    def _collect_diarization(self, job: Diarization, cancel: threading.Event,
                             on_progress: Callable[[Progress], None],
                             tail: float) -> list[Line]:
        """Дожидается разметку голосов и проверяет, похож ли результат на правду."""
        try:
            while job.running:
                if cancel.is_set():
                    raise Cancelled
                share = job.share()
                on_progress(Progress(f"Доразмечаю голоса… {share:.0%}",
                                     (1 - tail) + tail * share))
                time.sleep(0.2)
            turns = job.turns()
        except Cancelled:
            raise
        except Exception as exc:
            on_progress(Progress(f"Разделить по говорящим не вышло ({exc}); "
                                 "оставляю текст без имён", -1.0))
            return []
        finally:
            job.close()

        found = len({turn.speaker for turn in turns})
        if not found:
            message = "Отдельных голосов не нашлось"
        elif job.wanted == 0 and found > SUSPICIOUS_SPEAKERS:
            # Автоопределение легко дробит одного человека на несколько «спикеров»,
            # если запись шумная или голос меняет громкость. Честно предупреждаем.
            message = (f"Нашёл голосов: {found} — похоже на ошибку разметки. "
                       "Укажи число участников в списке и запусти снова")
        else:
            message = f"Нашёл голосов: {found}"
        on_progress(Progress(message, 1.0))
        return turns


def _split_by_speaker(segment, turns: list[Line]) -> Iterator[Line]:
    """Режет кусок расшифровки там, где меняется говорящий.

    Whisper нарезает текст по паузам и запросто склеивает вопрос и ответ в одну
    фразу. Поэтому при разделении голосов идём по словам: пока говорит один и тот
    же человек — копим слова, сменился — начинаем новую реплику.
    """
    text = segment.text.strip()
    if not text:
        return

    if not turns:
        yield Line(start=segment.start, end=segment.end, text=text)
        return

    words = getattr(segment, "words", None) or []
    if not words:
        yield Line(start=segment.start, end=segment.end, text=text,
                   speaker=_speaker_at(segment.start, segment.end, turns))
        return

    current: list[str] = []
    speaker: str | None = None
    start = words[0].start
    previous_end = words[0].end

    for word in words:
        who = _speaker_at(word.start, word.end, turns)
        if current and who != speaker:
            yield Line(start=start, end=previous_end, text="".join(current).strip(),
                       speaker=speaker)
            current, start = [], word.start
        speaker = who
        current.append(word.word)
        previous_end = word.end

    if current:
        yield Line(start=start, end=words[-1].end, text="".join(current).strip(),
                   speaker=speaker)


def _speaker_at(start: float, end: float, turns: list[Line]) -> str | None:
    """Чей это отрезок: берём голос с наибольшим перекрытием по времени."""
    best, best_overlap = None, 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best, best_overlap = turn.speaker, overlap
    if best is not None:
        return best

    # Слово попало в паузу разметки — отдаём ближайшему по времени голосу.
    middle = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(t.start - middle), abs(t.end - middle)),
                  default=None)
    return nearest.speaker if nearest else None


class Diarization:
    """Разметка голосов, которая считается своим процессом параллельно расшифровке.

    Отдельный процесс нужен по двум причинам. Во-первых, sherpa-onnx держит
    блокировку интерпретатора всё время расчёта — в одном процессе с окном оно
    замирало на минуты. Во-вторых, разметка занимает процессор, а расшифровка —
    видеокарту, и запускать их по очереди значит держать половину машины
    без дела: на записи в 40 минут это лишние три с лишним минуты ожидания.
    """

    def __init__(self, path: str, speakers: int) -> None:
        import subprocess
        import tempfile

        self.wanted = speakers  # 0 — число голосов ищем сами
        self._folder = tempfile.mkdtemp(prefix="trnslt-")
        self.report = Path(self._folder) / "diarization.txt"
        self._process = subprocess.Popen(
            _child_command(["--diarize", path, str(speakers), str(self.report)]),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    @property
    def running(self) -> bool:
        return self._process.poll() is None

    def share(self) -> float:
        """Насколько разметка продвинулась, 0.0 … 1.0."""
        done = 0.0
        for line in _read_lines(self.report):
            if line.startswith("p "):
                done = float(line[2:])
        return done

    def turns(self) -> list[Line]:
        """Готовые куски «кто когда говорит». Пустой список — разметка не удалась."""
        import json

        for line in _read_lines(self.report):
            if line.startswith("{"):
                payload = json.loads(line)
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return [Line(start=s[0], end=s[1], text="", speaker=f"Спикер {s[2] + 1}")
                        for s in payload["segments"]]
        raise RuntimeError("разметка голосов не вернула результата")

    def close(self) -> None:
        import shutil

        if self.running:
            self._process.kill()
        self._process.wait(timeout=10)
        shutil.rmtree(self._folder, ignore_errors=True)


def _child_command(arguments: list[str]) -> list[str]:
    """Команда запуска самого себя: собранный exe или python с app.py."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    return [sys.executable, str(Path(__file__).resolve().parent / "app.py"), *arguments]


def _read_lines(path: Path) -> list[str]:
    """Читает отчёт дочернего процесса, не спотыкаясь о недописанную строку."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [line for line in content.splitlines() if line]


def _warm_up(model: WhisperModel) -> None:
    """Прогоняет секунду тишины, чтобы CUDA-библиотеки подгрузились прямо сейчас.

    cuBLAS и cuDNN подхватываются лениво, уже во время счёта, поэтому без прогрева
    поломка GPU вылезала бы на середине расшифровки — когда откатываться поздно.
    """
    import numpy as np

    segments, _ = model.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32), language="ru", beam_size=1, vad_filter=False
    )
    list(segments)


def _iter_cancellable(segments: Iterator, cancel: threading.Event, job=None) -> Iterator:
    for segment in segments:
        if cancel.is_set():
            if job is not None:
                job.close()  # чтобы дочерний процесс не остался считать в пустоту
            raise Cancelled
        yield segment
