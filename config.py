"""Validated settings and immutable official model pins."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
SAM_CODE_REVISION = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
SAM_WEIGHTS_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    qwen_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_revision: str = QWEN_REVISION
    qwen_max_pixels: int = Field(default=1_048_576, ge=4096)
    qwen_max_tokens: int = Field(default=384, ge=32, le=2048)
    max_classes: int = Field(default=64, ge=1, le=200)
    sam_checkpoint: Path = Path("checkpoints/sam3.pt")
    sam_sha256: str = Field(default=SAM_SHA256, pattern=r"^[0-9a-f]{64}$")
    device: str = Field(default="cuda:0", pattern=r"^cuda(?::\d+)?$")
    expected_gpu: str = "RTX 5090"
    precision: Literal["bfloat16", "float16"] = "bfloat16"
    local_files_only: bool = False
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    min_area: int = Field(default=32, ge=1)
    duplicate_iou: float = Field(default=0.95, gt=0, le=1)
    max_images: int = Field(default=10_000, ge=1)
    max_pixels: int = Field(default=40_000_000, ge=1)
    max_file_bytes: int = Field(default=100_000_000, ge=1)
    overlay_opacity: float = Field(default=0.45, ge=0, le=1)
    storage_root: Path = Path("storage")
    synonyms: dict[str, str] = Field(
        default_factory=lambda: {
            "cars": "car",
            "people": "person",
            "persons": "person",
            "bicycles": "bicycle",
            "bike": "bicycle",
            "motorbike": "motorcycle",
        }
    )
    stuff_classes: list[str] = Field(
        default_factory=lambda: [
            "road",
            "sidewalk",
            "sky",
            "grass",
            "vegetation",
            "wall",
            "water",
        ]
    )

    @field_validator("device")
    @classmethod
    def explicit_device(cls, value: str) -> str:
        return "cuda:0" if value == "cuda" else value

    @model_validator(mode="after")
    def official_qwen(self):
        if not self.qwen_model.startswith("Qwen/Qwen3-VL-"):
            raise ValueError("Use an explicit official Qwen3-VL checkpoint ID")
        if len(self.qwen_revision) != 40 or any(
            c not in "0123456789abcdef" for c in self.qwen_revision
        ):
            raise ValueError("qwen_revision must be an immutable 40-character commit")
        return self

    def fingerprint(self) -> str:
        data = self.model_dump(
            mode="json",
            exclude={
                "sam_checkpoint",
                "storage_root",
                "expected_gpu",
                "overlay_opacity",
                "max_images",
                "max_pixels",
                "max_file_bytes",
                "local_files_only",
            },
        )
        data["sam_code_revision"] = SAM_CODE_REVISION
        data["pipeline_version"] = 1
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_settings(
    path: Path | None = None, overrides: dict | None = None, defaults: dict | None = None
) -> Settings:
    """JSON config < AUTO_ environment variables < explicit CLI options."""
    values = dict(defaults or {})
    if path:
        values.update(json.loads(path.read_text(encoding="utf-8-sig")))
    for name in Settings.model_fields:
        value = os.environ.get("AUTO_" + name.upper())
        if value is not None:
            try:
                values[name] = json.loads(value)
            except json.JSONDecodeError:
                values[name] = value
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return Settings.model_validate(values)
