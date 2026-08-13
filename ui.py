"""Внешний вид окна: цвета, стили ttk и кнопки, которые откликаются на мышь."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# Тёмная палитра с заметной разницей между фоном, карточками и текстом.
BG = "#15171c"          # фон окна
CARD = "#1d2027"        # карточки: настройки, поле текста
CARD_HOVER = "#242833"
BORDER = "#2f333d"
FG = "#eceef2"          # основной текст
FG_DIM = "#98a1b0"      # подписи и второстепенное
ACCENT = "#4c9f70"      # главные кнопки
ACCENT_HOVER = "#59b47f"
ACCENT_DIM = "#37694f"
DANGER = "#c76b6b"

# Цвета меток говорящих: по одному на голос, дальше по кругу.
SPEAKER_COLORS = ["#6fb3e0", "#e3a765", "#bb92e6", "#63c9a4", "#e58080", "#cfc86e"]

FONT = "Segoe UI"
MONO = "Consolas"


def apply_theme(root: tk.Misc) -> ttk.Style:
    """Настраивает ttk и выпадающие списки под тёмный фон.

    Список внутри Combobox — обычный Listbox, до которого стили ttk не достают:
    без option_add он рисуется системными цветами, и строки на тёмном фоне
    становятся нечитаемыми.
    """
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("trnslt.TCombobox", fieldbackground=CARD, background=CARD, foreground=FG,
                    arrowcolor=FG_DIM, bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD,
                    borderwidth=1, padding=6, relief="flat")
    style.map("trnslt.TCombobox",
              fieldbackground=[("readonly", CARD), ("disabled", BG)],
              foreground=[("readonly", FG), ("disabled", FG_DIM)],
              # Без этого выбранная строка подсвечивается системным синим,
              # и текст на ней не читается.
              selectbackground=[("readonly", CARD)],
              selectforeground=[("readonly", FG)],
              bordercolor=[("focus", ACCENT)],
              arrowcolor=[("disabled", BORDER)])

    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", (FONT, 10))
    root.option_add("*TCombobox*Listbox.borderWidth", 0)

    # У clam полоса прогресса и полоса прокрутки берут светлую рамку из темы,
    # поэтому гасим ещё и bordercolor с light/darkcolor — иначе на тёмном фоне
    # остаётся белая обводка.
    style.configure("trnslt.Horizontal.TProgressbar", background=ACCENT, troughcolor=CARD,
                    bordercolor=CARD, lightcolor=ACCENT, darkcolor=ACCENT,
                    borderwidth=0, thickness=8)
    style.configure("trnslt.Vertical.TScrollbar", background=BORDER, troughcolor=CARD,
                    bordercolor=CARD, lightcolor=BORDER, darkcolor=BORDER,
                    arrowcolor=FG_DIM, borderwidth=0, relief="flat", arrowsize=12, width=12)
    style.map("trnslt.Vertical.TScrollbar",
              background=[("pressed", ACCENT), ("active", FG_DIM)],
              arrowcolor=[("pressed", ACCENT), ("active", FG)])
    return style


class Button(tk.Label):
    """Кнопка на основе Label: у обычной tk.Button на Windows не убрать серую рамку."""

    def __init__(self, parent, text: str, command, primary: bool = False, **kwargs) -> None:
        self.command = command
        self.primary = primary
        self._enabled = True
        colors = self._colors()
        super().__init__(parent, text=text, bg=colors[0], fg=colors[1],
                         font=(FONT, 10, "bold" if primary else "normal"),
                         padx=kwargs.pop("padx", 16), pady=kwargs.pop("pady", 7),
                         cursor="hand2", **kwargs)
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)

    def _colors(self) -> tuple[str, str]:
        if not self._enabled:
            return (ACCENT_DIM, "#93a89b") if self.primary else (CARD, FG_DIM)
        return (ACCENT, "#ffffff") if self.primary else (CARD, FG)

    def _click(self, _event) -> None:
        if self._enabled:
            self.command()

    def _enter(self, _event) -> None:
        if self._enabled:
            self.configure(bg=ACCENT_HOVER if self.primary else CARD_HOVER)

    def _leave(self, _event) -> None:
        self.configure(bg=self._colors()[0])

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        background, foreground = self._colors()
        self.configure(bg=background, fg=foreground, cursor="hand2" if enabled else "arrow")

    def set_text(self, text: str) -> None:
        self.configure(text=text)


class Checkbox(tk.Frame):
    """Своя галочка: у системной на тёмном фоне остаётся светлый квадрат."""

    def __init__(self, parent, text: str, variable: tk.BooleanVar, command=None) -> None:
        super().__init__(parent, bg=BG)
        self.variable = variable
        self.command = command
        self._enabled = True

        self.mark = tk.Label(self, width=2, font=(FONT, 11, "bold"), bg=CARD, fg=ACCENT,
                             padx=2, pady=1, cursor="hand2")
        self.mark.pack(side="left")
        self.caption = tk.Label(self, text=text, bg=BG, fg=FG, font=(FONT, 10),
                                padx=8, cursor="hand2")
        self.caption.pack(side="left")

        for widget in (self.mark, self.caption):
            widget.bind("<Button-1>", self._toggle)
        self.refresh()

    def _toggle(self, _event=None) -> None:
        if not self._enabled:
            return
        self.variable.set(not self.variable.get())
        self.refresh()
        if self.command:
            self.command()

    def refresh(self) -> None:
        checked = self.variable.get()
        self.mark.configure(text="✓" if checked else " ",
                            fg=ACCENT if self._enabled else BORDER,
                            bg=CARD if self._enabled else BG)
        self.caption.configure(fg=FG if self._enabled else FG_DIM)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        cursor = "hand2" if enabled else "arrow"
        self.mark.configure(cursor=cursor)
        self.caption.configure(cursor=cursor)
        self.refresh()


def card(parent, **kwargs) -> tk.Frame:
    """Панель с фоном карточки и внутренними отступами."""
    return tk.Frame(parent, bg=CARD, padx=kwargs.pop("padx", 14),
                    pady=kwargs.pop("pady", 12), **kwargs)


def caption(parent, text: str) -> tk.Label:
    """Мелкая подпись над полем."""
    return tk.Label(parent, text=text, bg=BG, fg=FG_DIM, font=(FONT, 9))
