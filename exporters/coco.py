"""Lossless COCO RLE and panoptic export with the documented CVAT ZIP layout."""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from contracts import Annotation, Category, ImageRecord
from processing.images import open_registered
from processing.validate import mask_geometry
from storage import atomic_json, atomic_target, load_mask, now, safe_path, save_png, sha256
from visualization.overlays import category_color

LOG = logging.getLogger(__name__)


def encode_rle(mask: np.ndarray) -> dict:
    """Standard uncompressed COCO RLE, in column-major order, starting with zeros."""
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    if not flat.size:
        raise ValueError("Cannot encode a zero-sized image")
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], transitions, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def decode_rle(rle: dict) -> np.ndarray:
    height, width = rle["size"]
    counts = rle["counts"]
    if (
        height <= 0
        or width <= 0
        or not isinstance(counts, list)
        or not all(type(v) is int and v >= 0 for v in counts)
        or sum(counts) != height * width
    ):
        raise ValueError("Invalid uncompressed COCO RLE")
    values = np.arange(len(counts), dtype=np.uint8) % 2
    return np.repeat(values, counts).reshape((height, width), order="F").astype(bool)


def compose_panoptic(
    annotations: list[Annotation],
    masks: dict[int, np.ndarray],
    categories: dict[int, Category],
    size: tuple[int, int],
) -> tuple[Image.Image, list[dict]]:
    width, height = size
    ids = np.zeros((height, width), dtype=np.uint32)
    segments, stuff_ids = {}, {}
    order = sorted(
        annotations,
        key=lambda a: (
            -(a.score if a.score is not None else -1),
            a.area,
            a.id,
        ),
    )
    next_id = 1
    for annotation in order:
        visible = masks[annotation.id] & (ids == 0)
        if not visible.any():
            LOG.warning(
                "Panoptic: annotation %s fully occluded by higher-priority masks", annotation.id
            )
            continue
        category = categories[annotation.category_id]
        segment_id = stuff_ids.get(category.id) if not category.isthing else None
        if segment_id is None:
            segment_id, next_id = next_id, next_id + 1
            if segment_id >= 2**24:
                raise ValueError("Panoptic segment ID exceeds RGB capacity")
            segments[segment_id] = {"id": segment_id, "category_id": category.id, "iscrowd": 0}
            if not category.isthing:
                stuff_ids[category.id] = segment_id
        ids[visible] = segment_id
    for segment_id, segment in segments.items():
        bbox, area = mask_geometry(ids == segment_id)
        segment.update(bbox=list(bbox), area=area)
    rgb = np.stack((ids & 255, (ids >> 8) & 255, (ids >> 16) & 255), axis=-1).astype(np.uint8)
    return Image.fromarray(rgb), list(segments.values())


def export_dataset(
    job_dir: Path,
    source_root: Path,
    images: list[ImageRecord],
    annotations: list[Annotation],
    categories: list[Category],
    reviews: dict,
    max_pixels: int,
    include_pending: bool = False,
) -> dict:
    image_map = {image.id: image for image in images}
    category_map = {category.id: category for category in categories}
    if len(image_map) != len(images) or len(category_map) != len(categories):
        raise ValueError("Duplicate image/category IDs")
    annotation_map = {a.id: a for a in annotations}
    if len(annotation_map) != len(annotations):
        raise ValueError("Duplicate annotation IDs")
    if set(reviews) - {str(key) for key in annotation_map}:
        raise ValueError("Review records reference unknown annotations")
    counts = Counter()
    selected = {image.id: [] for image in images}
    for annotation in annotations:
        status = reviews.get(str(annotation.id), {}).get("status", "pending")
        if status not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"Invalid review status for {annotation.id}")
        counts[status] += 1
        if annotation.image_id not in image_map or annotation.category_id not in category_map:
            raise ValueError(f"Invalid annotation references: {annotation.id}")
        if annotation.category_name != category_map[annotation.category_id].name:
            raise ValueError("Annotation name and category registry disagree")
        if status == "accepted" or (include_pending and status == "pending"):
            selected[annotation.image_id].append(annotation)
    if counts["pending"]:
        LOG.warning(
            "%d pending annotations; %s",
            counts["pending"],
            "explicitly included for CVAT correction"
            if include_pending
            else "excluded from accepted-only exports",
        )
    document = {
        "info": {
            "description": "Qwen3-VL + SAM 3 assisted annotations",
            "date_created": now(),
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [c.model_dump(exclude={"isthing"}) for c in categories],
    }
    panoptic = {
        **document,
        "images": document["images"],
        "annotations": [],
        "categories": [{**c.model_dump(), "color": list(category_color(c.id))} for c in categories],
    }
    mapping, names = [], set()
    with (
        atomic_target(job_dir / "cvat_export.zip") as instance_zip,
        atomic_target(job_dir / "panoptic_export.zip") as panoptic_zip,
    ):
        with (
            zipfile.ZipFile(instance_zip, "w", compression=zipfile.ZIP_DEFLATED) as izip,
            zipfile.ZipFile(panoptic_zip, "w", compression=zipfile.ZIP_DEFLATED) as pzip,
        ):
            for record in images:
                source = safe_path(source_root, record.relative_path)
                with open_registered(source_root, record, max_pixels) as image:
                    # Orientation is unambiguous for every decoder. Preserve the original
                    # bytes separately only when an upright derived image is necessary.
                    oriented = record.exif_orientation != 1
                    filename = (
                        f"_oriented/{record.id}/{Path(record.relative_path).name}.png"
                        if oriented
                        else record.relative_path
                    )
                    safe_path(job_dir, filename)
                    if filename.casefold() in names:
                        raise ValueError(f"Export filename collision: {filename}")
                    names.add(filename.casefold())
                    entry = {
                        "id": record.id,
                        "file_name": filename,
                        "width": record.width,
                        "height": record.height,
                    }
                    document["images"].append(entry)
                    for archive in (izip, pzip):
                        if oriented:
                            buffer = io.BytesIO()
                            image.save(buffer, format="PNG")
                            archive.writestr("images/default/" + filename, buffer.getvalue())
                            archive.write(source, "originals/" + record.relative_path)
                        else:
                            archive.write(source, "images/default/" + filename)
                    # Detect source changes during the ZIP write as well.
                    if sha256(source) != record.sha256:
                        raise ValueError(f"Source changed during export: {source}")
                    mapping.append(
                        {
                            "image_id": record.id,
                            "source": record.relative_path,
                            "export_file": filename,
                            "sha256": record.sha256,
                            "exif_orientation": record.exif_orientation,
                        }
                    )
                    masks = {}
                    for annotation in selected[record.id]:
                        mask = load_mask(
                            job_dir, annotation.mask_ref, annotation.mask_sha256, image.size
                        )
                        bbox, area = mask_geometry(mask)
                        if bbox != annotation.bbox or area != annotation.area:
                            raise ValueError(f"Mask geometry mismatch: {annotation.id}")
                        masks[annotation.id] = mask
                        rle = encode_rle(mask)
                        if not np.array_equal(decode_rle(rle), mask):
                            raise ValueError("RLE round-trip mismatch")
                        status = reviews.get(str(annotation.id), {}).get("status", "pending")
                        item = {
                            "id": annotation.id,
                            "image_id": record.id,
                            "category_id": annotation.category_id,
                            "segmentation": rle,
                            "bbox": list(bbox),
                            "area": area,
                            "iscrowd": 0,
                            "attributes": {"review_status": status, "source": "SAM 3"},
                        }
                        if annotation.score is not None:
                            item["score"] = annotation.score
                        document["annotations"].append(item)
                    rgb, segments = compose_panoptic(
                        selected[record.id],
                        masks,
                        category_map,
                        image.size,
                    )
                    png_name = f"{record.id}.png"
                    png_path = job_dir / "panoptic" / png_name
                    save_png(png_path, rgb)
                    pzip.write(png_path, "annotations/panoptic_default/" + png_name)
                    panoptic["annotations"].append(
                        {
                            "image_id": record.id,
                            "file_name": png_name,
                            "segments_info": segments,
                        }
                    )
            json_text = json.dumps(document, ensure_ascii=False, allow_nan=False)
            panoptic_text = json.dumps(panoptic, ensure_ascii=False, allow_nan=False)
            izip.writestr("annotations/instances_default.json", json_text)
            pzip.writestr("annotations/panoptic_default.json", panoptic_text)
            provenance = json.dumps(
                {
                    "images": mapping,
                    "include_pending": include_pending,
                    "review_counts": dict(counts),
                },
                ensure_ascii=False,
            )
            for archive in (izip, pzip):
                archive.writestr("provenance.json", provenance)
        atomic_json(job_dir / "instances_default.json", document)
        atomic_json(job_dir / "panoptic_default.json", panoptic)
    result = {
        "created_at": now(),
        "images": len(images),
        "annotations": len(document["annotations"]),
        "review_counts": dict(counts),
        "include_pending": include_pending,
        "files": {
            name: sha256(job_dir / name)
            for name in (
                "cvat_export.zip",
                "panoptic_export.zip",
                "instances_default.json",
                "panoptic_default.json",
            )
        },
    }
    atomic_json(job_dir / "export_manifest.json", result)
    return result
