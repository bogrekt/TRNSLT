"""Собирает trnslt в папку, которую можно просто переслать другу.

Запуск:  .venv\\Scripts\\python.exe build.py
Итог:    dist\\trnslt\\trnslt.exe  и  dist\\trnslt-windows.zip
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / "build_payload"
# Папку можно задать: --dist <путь>. Пригождается, когда прошлая сборка
# ещё запущена и держит свои файлы.
DIST = ROOT / (sys.argv[sys.argv.index("--dist") + 1] if "--dist" in sys.argv else "dist")
MODEL_REPO = "Systran/faster-whisper-small"
MODEL_NAME = "small"

# Модели для разделения по говорящим (ONNX, работают через sherpa-onnx без torch).
# Сегментация — pyannote 3.0, отпечатки голоса — campplus, обученный на китайском
# и английском.
#
# Модель выбрана замером на пяти записях с известным числом участников, уже со
# своей кластеризацией: campplus угадывает все пять при любом пороге от 0.35 до
# 0.55, eres2net — тоже пять, но весит в полтора раза больше. Стоявший здесь
# раньше resnet34 на разговоре четверых сливал всех в один голос — из-за него
# на живых записях и появлялись «четыре минуты одного спикера».
RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_URL = f"{RELEASES}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"

# Библиотеки CUDA, которые кладём в сборку — только cuBLAS.
#
# cuDNN (вся, ~1.1 ГБ) и nvrtc проверены опытом: сборка без них считает на GPU
# ровно так же, ctranslate2 их для Whisper не трогает. cublasLt, наоборот,
# обязателен — без него не грузится сам cublas64_12.dll и всё уходит на процессор.
# Проверять просто: удалить библиотеку из dist и запустить trnslt.exe --selftest,
# в отчёте будет видно, осталось ли «считало на: cuda».
CUDA_DLLS = [
    "cublas64_12.dll",
    "cublasLt64_12.dll",
]

READ_ME = """trnslt — распознавание речи из аудио в текст

Куда положить
  Распакуй архив и перетащи папку целиком куда удобно — например, в «Документы».
  Установка не нужна, права администратора не нужны, в реестр ничего не пишется.
  Переносить нужно всю папку: trnslt.exe без соседней папки _internal не запустится.

Ярлык на рабочем столе
  Двойной клик по «Ярлык на рабочий стол.bat» — и ярлык появится на рабочем столе.
  Дальше запускать можно с него. Папку с программой после этого не переименовывай
  и не переноси, иначе ярлык перестанет её находить (тогда просто сделай заново).

Как пользоваться
  1. Запусти trnslt.exe
  2. Перетащи в окно аудио или видео (mp3, wav, m4a, mp4, ogg, flac и другие)
  3. Нажми «Распознать», дождись текста
  4. Нажми «Открыть в Word» — документ ляжет рядом с записью и сразу откроется

Кнопки «Сохранить .docx» и «Сохранить .txt» рядом — если нужно выбрать папку
самому. Повторное нажатие не затирает прошлый документ, рядом появится «(2)».

Тайминги
  Галочка «Тайминги» подписывает время начала каждой реплики. Её можно включать
  и выключать уже после распознавания — текст перерисуется сразу.

Разные говорящие
  Галочка «Разделять говорящих» подписывает реплики: Спикер 1, Спикер 2 и так далее.

  ВАЖНО: если знаешь, сколько человек на записи, обязательно выбери число в списке
  рядом. Режим «определить самому» ошибается — на шумной записи он дробит одного
  человека на десяток «спикеров», а иногда наоборот сливает всех в одного.

  Считает процессор, не видеокарта. Прикидка: 16 минут записи — около 4 минут
  разметки, час записи — около 15 минут. Окно при этом остаётся живым, внизу
  справа идут часы работы.

Всё считается на твоём компьютере. Интернет не нужен, файлы никуда не отправляются.

Модель small уже внутри — она хорошо распознаёт русскую речь.
Модели medium и large-v3 в списке точнее, но их придётся скачать при первом выборе
(нужен интернет), и на видеокарте меньше 6 ГБ памяти large-v3 может не поместиться.

Если есть видеокарта NVIDIA, программа сама её найдёт: час записи распознаётся
примерно за 6 минут. Без неё будет считать процессор — медленнее, но так же точно.

При первом запуске Windows может показать синее окно «Защита Windows».
Нажми «Подробнее» → «Выполнить в любом случае»: так система реагирует
на любую программу без платной подписи разработчика.
"""


def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def prepare_model() -> None:
    """Кладёт модель распознавания в payload; при отсутствии — скачивает."""
    # Кэш Hugging Face иногда оседает прямо здесь и уезжает в сборку целиком:
    # так однажды в архив попала лишняя копия модели на 461 МБ.
    for stray in (PAYLOAD / "models").glob("models--*"):
        shutil.rmtree(stray, ignore_errors=True)
        log(f"  - убрал кэш {stray.name}")

    target = PAYLOAD / "models" / MODEL_NAME
    if (target / "model.bin").is_file():
        log(f"модель уже готова: {target}")
        return

    # Качаем файлы поимённо в обычную папку: snapshot_download кладёт их в кэш
    # через символьные ссылки, а на Windows без прав администратора это падает.
    from huggingface_hub import hf_hub_download

    log(f"беру модель {MODEL_REPO} (~460 МБ, если ещё не скачана)…")
    target.mkdir(parents=True, exist_ok=True)
    required = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
    optional = ("vocabulary.json", "preprocessor_config.json")
    for name in required + optional:
        try:
            path = Path(hf_hub_download(MODEL_REPO, name, local_dir=str(target)))
        except Exception as exc:
            if name in required:
                raise SystemExit(f"Не удалось получить {name}: {exc}")
            continue
        log(f"  + {name} ({path.stat().st_size / 1024 / 1024:.1f} МБ)")


def prepare_diarization() -> None:
    """Кладёт в payload модели для разделения по говорящим."""
    import tarfile
    import tempfile
    import urllib.request

    target = PAYLOAD / "models" / "diarization"
    target.mkdir(parents=True, exist_ok=True)
    segmentation, embedding = target / "segmentation.onnx", target / "embedding.onnx"

    if not embedding.is_file():
        log("качаю модель голосовых отпечатков (~28 МБ)…")
        urllib.request.urlretrieve(EMBEDDING_URL, embedding)

    if not segmentation.is_file():
        log("качаю модель сегментации речи (~7 МБ)…")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "segmentation.tar.bz2"
            urllib.request.urlretrieve(SEGMENTATION_URL, archive)
            with tarfile.open(archive) as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith("model.onnx"))
                member.name = Path(member.name).name
                tar.extract(member, path=tmp, filter="data")
            shutil.move(str(Path(tmp) / "model.onnx"), segmentation)

    for path in (segmentation, embedding):
        log(f"  = {path.name} ({path.stat().st_size / 1024 / 1024:.1f} МБ)")


def prepare_cuda() -> None:
    """Копирует нужные CUDA-библиотеки из пакетов nvidia-* в payload/cuda."""
    target = PAYLOAD / "cuda"
    target.mkdir(parents=True, exist_ok=True)

    sources = {dll.name: dll for dll in (ROOT / ".venv" / "Lib" / "site-packages" / "nvidia").glob("*/bin/*.dll")}
    missing = [name for name in CUDA_DLLS if name not in sources]
    if missing:
        raise SystemExit(
            "Не нашлись библиотеки: " + ", ".join(missing) +
            "\nПоставь их: .venv\\Scripts\\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
        )

    # Лишнее из прошлых сборок убираем: spec берёт из этой папки всё подряд.
    for stale in target.glob("*.dll"):
        if stale.name not in CUDA_DLLS:
            stale.unlink()
            log(f"  - {stale.name} (больше не нужна)")

    total = 0
    for name in CUDA_DLLS:
        destination = target / name
        if not destination.is_file():
            shutil.copy2(sources[name], destination)
        total += destination.stat().st_size
    log(f"CUDA-библиотек: {len(CUDA_DLLS)} шт., {total / 1024 ** 3:.2f} ГБ")


def run_pyinstaller() -> None:
    log("запускаю PyInstaller (несколько минут)…")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "trnslt.spec", "--noconfirm",
         "--distpath", str(DIST), "--workpath", str(ROOT / "build")],
        cwd=ROOT, check=True,
    )


def finish() -> Path:
    app = DIST / "trnslt"
    (app / "Как пользоваться.txt").write_text(READ_ME, encoding="utf-8")

    # Сам файл держим в ASCII и в кодировке консоли: всё русское показывает
    # уже приложение своим окном, а не cmd — так ничего не рассыпается.
    (app / "Ярлык на рабочий стол.bat").write_text(
        '@echo off\r\n'
        'rem Kladet yarlyk na rabochiy stol. Zapusti dvoynym klikom.\r\n'
        'start "" "%~dp0trnslt.exe" --shortcut\r\n',
        encoding="cp866")

    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file())
    log(f"папка приложения: {size / 1024 ** 3:.2f} ГБ")

    log("упаковываю в zip (это дольше всего)…")
    started = time.time()
    archive = Path(shutil.make_archive(str(DIST / "trnslt-windows"), "zip", DIST, "trnslt"))
    log(f"архив: {archive.stat().st_size / 1024 ** 3:.2f} ГБ за {time.time() - started:.0f} с")
    return archive


def main() -> None:
    prepare_model()
    prepare_diarization()
    prepare_cuda()
    run_pyinstaller()
    archive = finish()
    log(f"готово: {archive}")
    log(f"проверка: {DIST / 'trnslt' / 'trnslt.exe'} --selftest путь\\к\\аудио.wav")


if __name__ == "__main__":
    main()
