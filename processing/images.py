"""Recursive image registration without altering source files."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from PIL import Image, ImageOps

from config import Settings
from contracts import ImageRecord
from storage import safe_path, sha256, stable_id

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FORMATS = {"JPEG", "PNG", "WEBP"}


def discover(root: Path, settings: Settings) -> list[str]:
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    found = []
    for directory, folders, names in os.walk(root, followlinks=False):
        folders[:] = sorted(f for f in folders if not Path(directory, f).is_symlink())
        for name in sorted(names):
            candidate = Path(directory, name)
            if candidate.suffix.lower() not in EXTENSIONS:
                continue
            relative = candidate.relative_to(root).as_posix()
            safe_path(root, relative)
            if candidate.is_symlink():
                raise ValueError(f"Use regular input files, not symlinks: {relative}")
            found.append(relative)
            if len(found) > settings.max_images:
                raise ValueError(
                    f"Input exceeds max_images={settings.max_images}; nothing truncated"
                )
    if not found:
        raise ValueError("No JPG, JPEG, PNG or WebP images found")
    return sorted(found)


def prepare(path: Path, max_pixels: int) -> tuple[Image.Image, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as original:
            if original.format not in FORMATS or getattr(original, "n_frames", 1) != 1:
                raise ValueError("Expected a single-frame JPEG, PNG or WebP image")
            width, height = original.size
            if width * height > max_pixels:
                raise ValueError(f"Image has {width * height} pixels; limit is {max_pixels}")
            exif = original.getexif()
            orientation = int(exif.get(274, 1))
            if orientation not in range(1, 9):
                raise ValueError(f"Invalid EXIF orientation: {orientation}")
            metadata = {
                key: str(exif[tag])
                for tag, key in (
                    (36867, "captured_at"),
                    (306, "modified_at"),
                    (271, "camera_make"),
                    (272, "camera_model"),
                )
                if tag in exif
            }
            original.load()
            prepared = ImageOps.exif_transpose(original).convert("RGB")
            return prepared, {
                "original_width": width,
                "original_height": height,
                "exif_orientation": orientation,
                "metadata": metadata,
            }


def inspect_image(root: Path, relative: str, settings: Settings) -> ImageRecord:
    path = safe_path(root, relative)
    before = path.stat()
    if before.st_size > settings.max_file_bytes:
        raise ValueError(f"File exceeds {settings.max_file_bytes} bytes")
    digest = sha256(path)
    image, metadata = prepare(path, settings.max_pixels)
    try:
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed while being registered")
        return ImageRecord(
            id=stable_id(relative),
            relative_path=relative,
            sha256=digest,
            width=image.width,
            height=image.height,
            size_bytes=before.st_size,
            mtime_ns=before.st_mtime_ns,
            **metadata,
        )
    finally:
        image.close()


def open_registered(root: Path, record: ImageRecord, max_pixels: int) -> Image.Image:
    path = safe_path(root, record.relative_path)
    if sha256(path) != record.sha256:
        raise ValueError(f"Source changed: {record.relative_path}; start a new job")
    image, _ = prepare(path, max_pixels)
    if image.size != (record.width, record.height):
        image.close()
        raise ValueError(f"Image dimensions changed: {record.relative_path}")
    return image
