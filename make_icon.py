"""Рисует иконку приложения: звуковая волна на тёмном скруглённом квадрате.

Нужна только при сборке (требует pillow), готовый .ico лежит в build_payload
и в приложение попадает уже картинкой:
    .venv\\Scripts\\python make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "build_payload" / "trnslt.ico"

BACKGROUND = (29, 32, 39, 255)   # тот же тёмный фон, что и в окне
ACCENT = (76, 159, 112, 255)     # зелёный акцент
QUIET = (152, 161, 176, 255)     # приглушённый для крайних полос

# Высоты полос в долях от высоты картинки — силуэт звуковой дорожки.
BARS = [0.22, 0.42, 0.66, 0.92, 0.66, 0.34, 0.52, 0.80, 0.44, 0.24]


def draw(size: int) -> Image.Image:
    # Рисуем крупно и уменьшаем: края получаются гладкими без возни со сглаживанием.
    scale = 8
    side = size * scale
    image = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    painter = ImageDraw.Draw(image)
    painter.rounded_rectangle([0, 0, side - 1, side - 1], radius=int(side * 0.22),
                              fill=BACKGROUND)

    count = len(BARS)
    gap = side * 0.030
    width = (side * 0.78 - gap * (count - 1)) / count
    left = (side - (width * count + gap * (count - 1))) / 2
    middle = side / 2

    for index, height in enumerate(BARS):
        tall = side * 0.62 * height
        x = left + index * (width + gap)
        colour = ACCENT if 1 <= index <= count - 2 else QUIET
        painter.rounded_rectangle(
            [x, middle - tall / 2, x + width, middle + tall / 2],
            radius=width / 2, fill=colour)

    return image.resize((size, size), Image.LANCZOS)


def main() -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [draw(size) for size in sizes]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    images[-1].save(TARGET, format="ICO",
                    sizes=[(size, size) for size in sizes],
                    append_images=images[:-1])
    print(f"иконка: {TARGET} ({TARGET.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
