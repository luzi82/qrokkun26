"""Version-local image helpers used by the rich-observation renderer."""

from pathlib import Path

from PIL import Image, ImageDraw

BULLET_COLORS = {
    0: (220, 80, 80),
    1: (80, 200, 120),
    2: (120, 160, 255),
    3: (240, 200, 60),
}


def load_sprite(path: Path, fallback_r: int, color: tuple[int, int, int]) -> Image.Image:
    if path.is_file():
        return Image.open(path).convert("RGBA")
    size = max(fallback_r * 2, 8)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, size - 1, size - 1), fill=color + (255,))
    return image


def paste_centered(
    canvas: Image.Image,
    sprite: Image.Image,
    x: float,
    y: float,
    scale: int,
) -> None:
    width, height = sprite.size
    scaled_width, scaled_height = width * scale, height * scale
    if scale != 1:
        sprite = sprite.resize((scaled_width, scaled_height), Image.NEAREST)
    canvas.alpha_composite(
        sprite,
        (int(round(x * scale - scaled_width / 2)), int(round(y * scale - scaled_height / 2))),
    )
