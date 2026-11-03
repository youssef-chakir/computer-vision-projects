"""Validate geometry and conservatively suppress same-class duplicates."""

from __future__ import annotations

import logging
import math

import numpy as np

from config import Settings
from contracts import RawPrediction, ValidatedInstance

LOG = logging.getLogger(__name__)


def mask_geometry(mask: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        raise ValueError("Empty mask")
    x, y = int(cols.min()), int(rows.min())
    return (x, y, int(cols.max()) - x + 1, int(rows.max()) - y + 1), int(mask.sum())


def validate_instances(
    raw: list[RawPrediction], categories: dict[str, int], size: tuple[int, int], settings: Settings
) -> tuple[list[ValidatedInstance], list[dict]]:
    width, height = size
    candidates, issues = [], []

    def issue(index, name, reason):
        item = {"prediction_index": index, "class_name": name, "reason": reason}
        issues.append(item)
        LOG.warning("Mask validation: %s", item)

    for index, prediction in enumerate(raw):
        mask = np.asarray(prediction.mask)
        name, score = prediction.class_name, prediction.score
        if name not in categories:
            raise ValueError(f"SAM returned an unregistered prompt: {name}")
        if mask.shape != (height, width):
            issue(index, name, f"invalid shape {mask.shape}")
            continue
        if mask.dtype.kind not in "buif" or not np.isin(mask, [0, 1]).all():
            issue(index, name, "mask must be binary, with finite values")
            continue
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            issue(index, name, "invalid score")
            continue
        if score is not None and score < settings.min_confidence:
            issue(index, name, "below confidence threshold")
            continue
        mask = mask.astype(bool, copy=False)
        area = int(mask.sum())
        if area < settings.min_area:
            issue(index, name, f"area {area} below minimum {settings.min_area}")
            continue
        bbox, area = mask_geometry(mask)
        if prediction.box is not None:
            box = np.asarray(prediction.box, dtype=float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or not (0 <= box[0] < box[2] <= width)
                or not (0 <= box[1] < box[3] <= height)
            ):
                issue(index, name, "invalid predicted box; recomputed from mask")
        candidates.append(
            (
                index,
                ValidatedInstance(
                    name,
                    categories[name],
                    mask,
                    bbox,
                    area,
                    score,
                ),
            )
        )

    # Suppress near-identical masks of the SAME category only.
    # Cross-class conflicts remain visible for review.
    candidates.sort(
        key=lambda pair: (-(pair[1].score if pair[1].score is not None else -1), pair[0])
    )
    kept = []
    for index, candidate in candidates:
        duplicate = False
        for previous in kept:
            if candidate.category_id != previous.category_id:
                continue
            if min(candidate.area, previous.area) / max(candidate.area, previous.area) < (
                settings.duplicate_iou
            ):
                continue
            ax, ay, aw, ah = candidate.bbox
            bx, by, bw, bh = previous.bbox
            x0, y0 = max(ax, bx), max(ay, by)
            x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = int(
                np.count_nonzero(candidate.mask[y0:y1, x0:x1] & previous.mask[y0:y1, x0:x1])
            )
            union = candidate.area + previous.area - intersection
            if intersection / union >= settings.duplicate_iou:
                duplicate = True
                break
        if duplicate:
            issue(index, candidate.class_name, "near-identical same-class duplicate")
        else:
            kept.append(candidate)
    return kept, issues
