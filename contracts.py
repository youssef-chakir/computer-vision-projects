"""Shared contracts; inference tensors never enter persisted JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ReviewStatus = Literal["pending", "accepted", "rejected"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageRecord(Record):
    id: int = Field(gt=0)
    relative_path: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    exif_orientation: int = 1
    size_bytes: int
    mtime_ns: int
    metadata: dict = Field(default_factory=dict)


class Category(Record):
    id: int = Field(gt=0)
    name: str
    isthing: Literal[0, 1] = 1
    supercategory: str = ""


class Prediction(Record):
    image_id: int
    status: Literal["succeeded", "empty", "failed"]
    classes: list[str] = Field(default_factory=list)
    raw_response: str = ""
    error: str | None = None
    seconds: float = 0


class Annotation(Record):
    id: int = Field(gt=0)
    image_id: int = Field(gt=0)
    category_id: int = Field(gt=0)
    category_name: str
    score: float | None = Field(default=None, ge=0, le=1)
    bbox: tuple[int, int, int, int]
    area: int = Field(gt=0)
    mask_ref: str
    mask_sha256: str
    review_status: Literal["pending"] = "pending"


@dataclass
class RawPrediction:
    class_name: str
    mask: np.ndarray
    box: list[float] | None = None  # Original SAM xyxy prediction.
    score: float | None = None


@dataclass
class ValidatedInstance:
    class_name: str
    category_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    score: float | None
