"""Пережимает модель распознавания в 8-битный формат для сборки.

Зачем: модель с Hugging Face лежит в 16-битном виде (461 МБ), а считаем мы её
всё равно в 8-битном — CTranslate2 пережимает её при каждом запуске. Если сделать
это один раз здесь, файл станет вдвое меньше и приложение будет стартовать быстрее.
Качество не меняется: вычисления те же самые.

Запуск (нужны transformers и torch, только для сборки — в приложение они не попадают):
    .venv\\Scripts\\pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
    .venv\\Scripts\\python shrink_model.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "build_payload" / "models" / "small"
SOURCE_REPO = "openai/whisper-small"


def main() -> int:
    before = 0
    if (TARGET / "model.bin").is_file():
        before = (TARGET / "model.bin").stat().st_size
        print(f"сейчас: {before / 1024 / 1024:.0f} МБ")

    staging = ROOT / "build_payload" / "models" / "small-int8"
    shutil.rmtree(staging, ignore_errors=True)

    print(f"пережимаю {SOURCE_REPO} (первый раз скачает ~1 ГБ)…", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "ctranslate2.converters.transformers",
         "--model", SOURCE_REPO, "--output_dir", str(staging),
         "--quantization", "int8_float16",
         "--copy_files", "tokenizer.json", "preprocessor_config.json"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("не вышло — оставляю прежнюю модель")
        return result.returncode

    if not (staging / "model.bin").is_file():
        print("пережатая модель не появилась — оставляю прежнюю")
        return 1

    shutil.rmtree(TARGET, ignore_errors=True)
    staging.rename(TARGET)
    after = (TARGET / "model.bin").stat().st_size
    print(f"стало: {after / 1024 / 1024:.0f} МБ"
          + (f" (было {before / 1024 / 1024:.0f})" if before else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
