"""Atomic local writes, path containment, and a process-scoped job lock."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


def now() -> str:
    return datetime.now(UTC).isoformat()


def stable_id(value: str) -> int:
    # Stay within exact JSON/JavaScript integer range.
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:13], 16) or 1


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if (
        not relative
        or not part.parts
        or part.is_absolute()
        or ".." in part.parts
        or "\\" in relative
        or ":" in relative
        or "\x00" in relative
        or part.as_posix() != relative
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    result = root.joinpath(*part.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {relative!r}")
    return result


@contextmanager
def atomic_target(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value) -> None:
    with atomic_target(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def read_json(path: Path, default=None):
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_png(path: Path, image: Image.Image) -> None:
    with atomic_target(path) as temporary:
        image.save(temporary, format="PNG")


def load_mask(root: Path, relative: str, expected_hash: str, size: tuple[int, int]) -> np.ndarray:
    path = safe_path(root, relative)
    if sha256(path) != expected_hash:
        raise ValueError(f"Mask changed: {relative}")
    with Image.open(path) as image:
        if image.mode != "L" or image.size != size:
            raise ValueError(f"Mask mode/dimensions invalid: {relative}")
        array = np.asarray(image)
        if not np.isin(array, [0, 255]).all():
            raise ValueError(f"Non-binary mask: {relative}")
        return array == 255


@contextmanager
def job_lock(directory: Path):
    """OS advisory lock, released even if the process crashes."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is using {directory}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
