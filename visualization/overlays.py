"""Deterministic colors, instance boundaries and side-by-side PNG previews."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from contracts import Annotation
from storage import load_mask, save_png


def category_color(category_id: int) -> tuple[int, int, int]:
    rgb = colorsys.hsv_to_rgb((category_id * 0.61803398875) % 1.0, 0.72, 0.95)
    return tuple(round(v * 255) for v in rgb)


def render_preview(
    image: Image.Image,
    annotations: list[Annotation],
    job_dir: Path,
    destination: Path,
    opacity: float = 0.45,
) -> None:
    if not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between 0 and 1")
    pixels = np.array(image, dtype=np.float32, copy=True)
    labels = []
    for annotation in annotations:
        mask = load_mask(job_dir, annotation.mask_ref, annotation.mask_sha256, image.size)
        color = np.array(category_color(annotation.category_id))
        pixels[mask] = pixels[mask] * (1 - opacity) + color * opacity
        padded = np.pad(mask, 1)
        interior = (
            mask & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
        pixels[mask & ~interior] = color
        x, y, _, _ = annotation.bbox
        labels.append((x, y, f"{annotation.category_name} #{annotation.id}"))
    overlay = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for x, y, text in labels:
        bounds = draw.textbbox((x, y), text)
        draw.rectangle(bounds, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
    canvas = Image.new("RGB", (image.width * 2, image.height + 24), (24, 24, 24))
    canvas.paste(image, (0, 24))
    canvas.paste(overlay, (image.width, 24))
    title = ImageDraw.Draw(canvas)
    title.text((5, 5), "Original (upright)", fill="white")
    title.text((image.width + 5, 5), "Model predictions - review required", fill="white")
    save_png(destination, canvas)
