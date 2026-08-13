"""Мелочи, нужные и основному приложению, и отдельному процессу разметки голосов.

Здесь намеренно нет ничего тяжёлого: этот модуль подключает дочерний процесс,
которому Whisper и CUDA не нужны совсем.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SAMPLE_RATE = 16000

# Голоса объединяются, пока среднее сходство отпечатков выше этого порога.
# Подобран замером на пяти записях с известным числом говорящих: с моделью
# campplus значения от 0.35 до 0.55 дают верный ответ на всех пяти, взята середина.
MERGE_THRESHOLD = 0.45

# Куски короче — на подсчёт голосов не влияют: у «ага» и «да» отпечаток случайный,
# и раньше из таких огрызков вырастали лишние спикеры.
MIN_RELIABLE_SECONDS = 1.0

# Верхняя граница для режима «определить самому».
MAX_SPEAKERS = 12

# Больше этого числа голосов — стоит предупредить, что разметка могла рассыпаться.
SUSPICIOUS_SPEAKERS = 10


def app_dir() -> Path:
    """Папка, рядом с которой лежат ресурсы: исходники или распакованная сборка."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def models_dir() -> Path:
    """Папка с моделями: в сборке — models/, при обычном запуске — build_payload/models/."""
    inside = app_dir() / "models"
    if inside.is_dir():
        return inside
    return app_dir() / "build_payload" / "models"


def diarization_models() -> tuple[Path, Path] | None:
    """Пути к моделям разделения по говорящим, если они на месте."""
    folder = models_dir() / "diarization"
    segmentation, embedding = folder / "segmentation.onnx", folder / "embedding.onnx"
    if segmentation.is_file() and embedding.is_file():
        return segmentation, embedding
    return None


def worker_threads() -> int:
    """Сколько ядер отдавать счёту на процессоре.

    Оставляем машине два ядра: если занять все, поток окна перестаёт получать
    время, окно белеет и Windows пишет «Не отвечает» — хотя работа идёт.
    """
    return max(1, min(4, (os.cpu_count() or 4) - 2))


def lower_priority() -> None:
    """Понижает приоритет текущего потока, чтобы окно оставалось живым."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        THREAD_PRIORITY_BELOW_NORMAL = -1
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:
        pass  # не вышло — не страшно, просто окно будет отзываться хуже


def format_time(seconds: float) -> str:
    """00:07 для коротких записей, 1:03:12 — для длинных."""
    seconds = max(int(seconds or 0), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
