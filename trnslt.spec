# PyInstaller-сборка trnslt. Запускать через build.py, а не напрямую:
# build.py готовит папку payload (модель + CUDA-библиотеки), на которую этот файл ссылается.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PAYLOAD = Path("build_payload").resolve()

datas = []
binaries = []
hiddenimports = []

# Пакеты, которые тащат за собой данные или бинарники: без collect_all
# PyInstaller теряет то ассеты VAD, то tcl-часть drag & drop, то кодеки PyAV.
for package in ("faster_whisper", "ctranslate2", "av", "onnxruntime", "tokenizers", "tkinterdnd2",
                "sherpa_onnx", "docx"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Модели кладём поимённо, а не всю папку разом: в build_payload могут лежать
# и другие, скачанные для замеров, — однажды так в архив уехала лишняя копия
# модели на 461 МБ. engine.py ищет их в models/ рядом с приложением.
for folder in ("small", "diarization"):
    datas.append((str(PAYLOAD / "models" / folder), f"models/{folder}"))

# CUDA-библиотеки кладём в корень _internal, рядом с ctranslate2.dll. В подпапке
# PyInstaller всё равно продублирует их в корень как найденные зависимости —
# и сборка потяжелеет на лишние 800 МБ.
binaries += [(str(dll), ".") for dll in sorted((PAYLOAD / "cuda").glob("*.dll"))]

# Видеокодировщики из поставки PyAV (libx265, libx264, SvtAv1 — вместе 27 МБ)
# выбросить нельзя, хотя мы пишем только звук: они прописаны в таблице импорта
# avcodec, и без них приложение не стартует вовсе. Проверено удалением из готовой
# сборки — запуск падает ещё до первой строки лога.

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Тяжёлый научный стек и сам PyInstaller в готовом приложении не нужны.
    excludes=["matplotlib", "scipy", "pandas", "IPython", "pytest", "PyInstaller",
              "torch", "transformers", "setuptools", "pip"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="trnslt",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # обычное окно без чёрной консоли
    disable_windowed_traceback=False,
    icon=str(PAYLOAD / "trnslt.ico"),  # её же видно на ярлыке рабочего стола
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="trnslt",
)
