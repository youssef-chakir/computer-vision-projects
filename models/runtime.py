"""CUDA preflight, memory logging and explicit resource release."""

from __future__ import annotations

import gc
import importlib.metadata
import logging
import sys

from config import Settings

LOG = logging.getLogger(__name__)


def check_runtime(settings: Settings) -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The supported inference environment is Python 3.12")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real inference has no CPU or mock fallback")
    device = torch.device(settings.device)
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    if settings.expected_gpu and settings.expected_gpu.casefold() not in name.casefold():
        raise RuntimeError(f"Expected {settings.expected_gpu!r}, found {name!r}")
    if settings.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 12 and (
        torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8)
    ):
        raise RuntimeError("RTX 5090 requires a Blackwell-capable build; use CUDA 12.8 wheels")
    packages = {}
    for name in ("torch", "torchvision", "transformers", "sam3", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    missing = [key for key, value in packages.items() if value == "MISSING"]
    if missing:
        raise RuntimeError(f"Missing inference packages: {', '.join(missing)}")
    report = {
        "python": sys.version.split()[0],
        "packages": packages,
        "gpu": torch.cuda.get_device_name(device),
        "capability": [major, minor],
        "cuda_runtime": torch.version.cuda,
        "vram_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
    LOG.info("Runtime: %s", report)
    return report


def log_memory(device: str, stage: str) -> None:
    import torch

    torch.cuda.synchronize(device)
    LOG.info(
        "%s GPU MiB: allocated=%.1f reserved=%.1f peak=%.1f",
        stage,
        torch.cuda.memory_allocated(device) / 2**20,
        torch.cuda.memory_reserved(device) / 2**20,
        torch.cuda.max_memory_allocated(device) / 2**20,
    )


def release(device: str) -> None:
    # Call only AFTER deleting model, processor and per-image tensor references.
    import torch

    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    log_memory(device, "released")
