"""Staged, resumable workflow. Only this layer coordinates the adapters."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from config import Settings
from contracts import Annotation, ImageRecord, Prediction
from exporters.coco import export_dataset
from processing.categories import CategoryRegistry
from processing.images import discover, inspect_image, open_registered
from processing.validate import validate_instances
from storage import (
    atomic_json,
    job_lock,
    load_mask,
    now,
    read_json,
    safe_path,
    save_png,
    sha256,
    stable_id,
)
from visualization.overlays import render_preview

LOG = logging.getLogger(__name__)


def read_job(directory: Path) -> tuple[dict, Settings, list[ImageRecord]]:
    job = read_json(directory / "job.json")
    settings = Settings.model_validate(job["settings"])
    if job.get("schema_version") != 1 or job.get("fingerprint") != settings.fingerprint():
        raise ValueError("Job schema/settings fingerprint mismatch")
    images = [ImageRecord.model_validate(x) for x in job["images"]]
    if len({x.id for x in images}) != len(images):
        raise ValueError("Image ID collision")
    return job, settings, images


def annotations_from(directory: Path, fingerprint: str) -> list[Annotation]:
    state = read_json(
        directory / "annotations_SAM3.json", {"fingerprint": fingerprint, "images": {}}
    )
    if state["fingerprint"] != fingerprint:
        raise ValueError("SAM cache belongs to different inference settings")
    annotations = []
    for image_id, result in state["images"].items():
        for value in result.get("annotations", []):
            annotation = Annotation.model_validate(value)
            if str(annotation.image_id) != image_id:
                raise ValueError("SAM cache image association is invalid")
            annotations.append(annotation)
    return annotations


def export_job(directory: Path, include_pending: bool = False) -> dict:
    job, settings, images = read_job(directory)
    job.update(exports_stale=True, updated_at=now())
    atomic_json(directory / "job.json", job)
    registry = CategoryRegistry(
        directory / "categories.json", settings.synonyms, settings.stuff_classes
    )
    annotations = annotations_from(directory, job["fingerprint"])
    reviews = read_json(directory / "reviews.json", {"decisions": {}})["decisions"]
    result = export_dataset(
        directory,
        Path(job["input_root"]),
        images,
        annotations,
        registry.categories,
        reviews,
        settings.max_pixels,
        include_pending,
    )
    job.update(exports_stale=False, last_export=result, updated_at=now())
    atomic_json(directory / "job.json", job)
    return result


def review_job(
    directory: Path,
    status: str,
    ids: list[int] | None = None,
    image_id: int | None = None,
    all_annotations: bool = False,
) -> int:
    if status not in {"pending", "accepted", "rejected"}:
        raise ValueError("Invalid review status")
    if sum((bool(ids), image_id is not None, all_annotations)) != 1:
        raise ValueError("Select annotation IDs, one image ID, or --all")
    job, _, _ = read_job(directory)
    annotations = annotations_from(directory, job["fingerprint"])
    available = {a.id: a for a in annotations}
    if ids:
        unknown = set(ids) - available.keys()
        if unknown:
            raise ValueError(f"Unknown annotation IDs: {sorted(unknown)}")
        chosen = sorted(set(ids))
    else:
        chosen = [a.id for a in annotations if all_annotations or a.image_id == image_id]
    if not chosen:
        raise ValueError("No annotations matched the review selection")
    reviews = read_json(directory / "reviews.json", {"decisions": {}, "history": []})
    timestamp = now()
    for annotation_id in chosen:
        key = str(annotation_id)
        old = reviews["decisions"].get(key, {}).get("status", "pending")
        reviews["decisions"][key] = {"status": status, "updated_at": timestamp}
        reviews["history"].append(
            {"annotation_id": annotation_id, "before": old, "after": status, "at": timestamp}
        )
    atomic_json(directory / "reviews.json", reviews)
    job.update(exports_stale=True, updated_at=timestamp)
    atomic_json(directory / "job.json", job)
    return len(chosen)


def preview_job(
    directory: Path,
    opacity: float | None = None,
    annotation_ids: list[int] | None = None,
    image_id: int | None = None,
) -> int:
    job, settings, images = read_job(directory)
    annotations = annotations_from(directory, job["fingerprint"])
    known = {a.id for a in annotations}
    if annotation_ids and set(annotation_ids) - known:
        raise ValueError("Unknown annotation IDs in preview selection")
    if image_id is not None and image_id not in {image.id for image in images}:
        raise ValueError("Unknown image ID")
    count = 0
    for record in images:
        if image_id is not None and record.id != image_id:
            continue
        selected = [
            a
            for a in annotations
            if a.image_id == record.id and (not annotation_ids or a.id in annotation_ids)
        ]
        if annotation_ids and not selected:
            continue
        suffix = "_selected" if annotation_ids else ""
        with open_registered(Path(job["input_root"]), record, settings.max_pixels) as image:
            render_preview(
                image,
                selected,
                directory,
                directory / "previews_visual" / f"{record.id}{suffix}.png",
                settings.overlay_opacity if opacity is None else opacity,
            )
        count += 1
    return count


class Annotator:
    def __init__(self, settings: Settings):
        self.settings = settings.model_copy(
            update={
                "sam_checkpoint": settings.sam_checkpoint.expanduser().resolve(),
                "storage_root": settings.storage_root.expanduser().resolve(),
            }
        )
        self.runtime = None

    def _ensure_runtime(self, directory: Path, job: dict) -> None:
        if self.runtime is None:
            from models.runtime import check_runtime

            self.runtime = check_runtime(self.settings)
            job["runtime"] = self.runtime
            atomic_json(directory / "job.json", job)

    def run(
        self,
        input_root: Path,
        directory: Path,
        resume: bool = False,
        retry_failed: bool = False,
        stop_after: str = "all",
    ) -> dict:
        input_root, directory = input_root.resolve(), directory.resolve()
        storage_root = self.settings.storage_root.resolve()
        if directory.is_relative_to(input_root) or input_root.is_relative_to(directory):
            raise ValueError("Input and output directories must be separate, with no nesting")
        if storage_root.is_relative_to(input_root):
            raise ValueError("storage_root must not be inside the input directory")
        if retry_failed and not resume:
            raise ValueError("--retry-failed requires --resume")
        if resume and not (directory / "job.json").is_file():
            raise ValueError("--resume requires an existing job.json")
        if not resume and directory.exists() and any(directory.iterdir()):
            raise ValueError("Output is not empty; use --resume or choose a new output")
        with job_lock(directory):
            # Recheck after acquiring the lock to prevent two new-job writers racing.
            if not resume and (directory / "job.json").exists():
                raise ValueError("Job already exists; use --resume")
            handler = logging.FileHandler(directory / "job.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(handler)
            job = None
            try:
                job = self._register(input_root, directory, resume, retry_failed)
                registry = CategoryRegistry(
                    directory / "categories.json",
                    self.settings.synonyms,
                    self.settings.stuff_classes,
                )
                self._qwen(input_root, directory, job, registry, retry_failed)
                if stop_after == "qwen":
                    job["status"] = "qwen_with_errors" if job["errors"] else "qwen_complete"
                else:
                    self._sam(input_root, directory, job, registry, retry_failed)
                    job["status"] = "rendering"
                    atomic_json(directory / "job.json", job)
                    preview_job(directory)
                    job["status"] = "exporting"
                    atomic_json(directory / "job.json", job)
                    result = export_job(directory)
                    job.update(last_export=result, exports_stale=False)
                    job["status"] = "ready_with_errors" if job["errors"] else "ready_for_review"
                job["updated_at"] = now()
                atomic_json(directory / "job.json", job)
                return job
            except BaseException as exc:
                if job is not None:
                    job.update(
                        status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        last_error=f"{type(exc).__name__}: {exc}",
                        updated_at=now(),
                    )
                    atomic_json(directory / "job.json", job)
                raise
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()

    def _register(self, root: Path, directory: Path, resume: bool, retry_failed: bool) -> dict:
        paths = discover(root, self.settings)
        if resume:
            job, _, old_images = read_job(directory)
            if job["input_root"] != str(root) or job["fingerprint"] != self.settings.fingerprint():
                raise ValueError("Input/model settings changed; create a new job")
            if job["input_paths"] != paths:
                raise ValueError("Input file list changed; create a new job")
            old = {record.relative_path: record for record in old_images}
        else:
            job = {
                "schema_version": 1,
                "job_id": uuid.uuid4().hex,
                "import_id": uuid.uuid4().hex,
                "created_at": now(),
                "input_root": str(root),
                "input_paths": paths,
                "fingerprint": self.settings.fingerprint(),
                "settings": self.settings.model_dump(mode="json"),
                "images": [],
                "errors": {},
                "exports_stale": True,
            }
            old = {}
        job["status"] = "registering"
        records, failures = [], {}
        old_errors = job.get("errors", {}).get("input", {})
        for relative in paths:
            if relative in old:
                record = old[relative]
                if sha256(safe_path(root, relative)) != record.sha256:
                    raise ValueError(f"Source changed: {relative}; create a new job")
                records.append(record)
            elif relative in old_errors and not retry_failed:
                failures[relative] = old_errors[relative]
            else:
                try:
                    records.append(inspect_image(root, relative, self.settings))
                except Exception as exc:
                    failures[relative] = f"{type(exc).__name__}: {exc}"
                    LOG.error("Input %s: %s", relative, exc)
        if len({record.id for record in records}) != len(records):
            raise ValueError("Image ID collision")
        job["images"] = [record.model_dump() for record in records]
        job["settings"] = self.settings.model_dump(mode="json")
        job["errors"] = {"input": failures} if failures else {}
        job.update(updated_at=now(), exports_stale=True)
        atomic_json(directory / "job.json", job)
        manifest = {
            key: job[key]
            for key in (
                "import_id",
                "created_at",
                "input_root",
                "input_paths",
                "images",
            )
        }
        manifest["errors"] = failures
        atomic_json(self.settings.storage_root / "imports" / f"{job['import_id']}.json", manifest)
        if not records:
            job.update(status="failed", last_error="No decodable input images", updated_at=now())
            atomic_json(directory / "job.json", job)
            raise ValueError("No decodable input images; see job.json errors")
        LOG.info("Registered %d images; %d input errors", len(records), len(failures))
        return job

    def _qwen(
        self, root: Path, directory: Path, job: dict, registry: CategoryRegistry, retry: bool
    ) -> None:
        from models.qwen import Qwen

        state = read_json(
            directory / "qwen_predictions.json",
            {"fingerprint": job["fingerprint"], "predictions": {}},
        )
        if state["fingerprint"] != job["fingerprint"]:
            raise ValueError("Qwen cache fingerprint mismatch")
        predictions = {
            key: Prediction.model_validate(value) for key, value in state["predictions"].items()
        }
        records = [ImageRecord.model_validate(x) for x in job["images"]]
        pending = []
        for record in records:
            previous = predictions.get(str(record.id))
            if previous is not None and previous.image_id != record.id:
                raise ValueError("Qwen cache image association mismatch")
            if previous is None or (previous.status == "failed" and retry):
                pending.append(record)
        job["status"] = "qwen"
        atomic_json(directory / "job.json", job)
        if pending:
            self._ensure_runtime(directory, job)
            adapter = Qwen(self.settings)
            try:
                adapter.load()
                for index, record in enumerate(pending, 1):
                    started = time.perf_counter()
                    try:
                        with open_registered(root, record, self.settings.max_pixels) as image:
                            classes, response = adapter.discover(image)
                        classes = registry.names(classes)
                        result = Prediction(
                            image_id=record.id,
                            classes=classes,
                            raw_response=response,
                            status="succeeded" if classes else "empty",
                        )
                    except Exception as exc:
                        result = Prediction(
                            image_id=record.id,
                            status="failed",
                            raw_response=getattr(exc, "raw_response", ""),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        LOG.error("Qwen %s: %s", record.relative_path, exc)
                    result.seconds = time.perf_counter() - started
                    predictions[str(record.id)] = result
                    state["predictions"] = {
                        key: value.model_dump() for key, value in predictions.items()
                    }
                    atomic_json(directory / "qwen_predictions.json", state)
                    from models.runtime import log_memory

                    log_memory(self.settings.device, "Qwen image")
                    LOG.info(
                        "Qwen %d/%d %s %s %.2fs",
                        index,
                        len(pending),
                        record.relative_path,
                        result.status,
                        result.seconds,
                    )
            finally:
                adapter.close()
        registry.register(
            [
                name
                for prediction in predictions.values()
                if prediction.status != "failed"
                for name in prediction.classes
            ]
        )
        failures = {
            key: value.error for key, value in predictions.items() if value.status == "failed"
        }
        if failures:
            job["errors"]["qwen"] = failures
        atomic_json(directory / "job.json", job)

    def _sam(
        self, root: Path, directory: Path, job: dict, registry: CategoryRegistry, retry: bool
    ) -> None:
        from models.sam3 import Sam3

        qwen = read_json(directory / "qwen_predictions.json")["predictions"]
        state = read_json(
            directory / "annotations_SAM3.json", {"fingerprint": job["fingerprint"], "images": {}}
        )
        if state["fingerprint"] != job["fingerprint"]:
            raise ValueError("SAM cache fingerprint mismatch")
        records = [ImageRecord.model_validate(x) for x in job["images"]]
        pending = []
        for record in records:
            previous = state["images"].get(str(record.id))
            if previous and previous["status"] in {"succeeded", "empty"}:
                for value in previous["annotations"]:
                    annotation = Annotation.model_validate(value)
                    load_mask(
                        directory,
                        annotation.mask_ref,
                        annotation.mask_sha256,
                        (record.width, record.height),
                    )
                continue
            if previous and previous["status"] == "failed" and not retry:
                continue
            prediction = Prediction.model_validate(qwen[str(record.id)])
            if prediction.status == "failed":
                continue
            if not prediction.classes:
                state["images"][str(record.id)] = {
                    "status": "empty",
                    "annotations": [],
                    "issues": [],
                    "seconds": 0,
                }
            else:
                pending.append((record, registry.names(prediction.classes)))
        job["status"] = "sam3"
        atomic_json(directory / "job.json", job)
        atomic_json(directory / "annotations_SAM3.json", state)
        if pending:
            self._ensure_runtime(directory, job)
            adapter = Sam3(self.settings)
            try:
                adapter.load()
                for index, (record, classes) in enumerate(pending, 1):
                    started = time.perf_counter()
                    try:
                        with open_registered(root, record, self.settings.max_pixels) as image:
                            raw = adapter.segment(image, classes)
                            valid, issues = validate_instances(
                                raw,
                                registry.by_name,
                                image.size,
                                self.settings,
                            )
                        saved = []
                        for instance in valid:
                            digest = hashlib.sha256(instance.mask.tobytes()).hexdigest()
                            identifier = stable_id(f"{record.id}:{instance.category_id}:{digest}")
                            relative = f"masks/{record.id}/{identifier}.png"
                            path = safe_path(directory, relative)
                            save_png(path, Image.fromarray(instance.mask.astype(np.uint8) * 255))
                            saved.append(
                                Annotation(
                                    id=identifier,
                                    image_id=record.id,
                                    category_id=instance.category_id,
                                    category_name=instance.class_name,
                                    score=instance.score,
                                    bbox=instance.bbox,
                                    area=instance.area,
                                    mask_ref=relative,
                                    mask_sha256=sha256(path),
                                ).model_dump(mode="json")
                            )
                        result = {
                            "status": "succeeded" if saved else "empty",
                            "annotations": saved,
                            "issues": issues,
                        }
                        del raw, valid
                    except Exception as exc:
                        result = {
                            "status": "failed",
                            "annotations": [],
                            "issues": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        LOG.error("SAM 3 %s: %s", record.relative_path, exc)
                    result["seconds"] = time.perf_counter() - started
                    state["images"][str(record.id)] = result
                    atomic_json(directory / "annotations_SAM3.json", state)
                    from models.runtime import log_memory

                    log_memory(self.settings.device, "SAM image")
                    LOG.info(
                        "SAM 3 %d/%d %s %s %.2fs",
                        index,
                        len(pending),
                        record.relative_path,
                        result["status"],
                        result["seconds"],
                    )
            finally:
                adapter.close()
        failures = {
            key: value["error"]
            for key, value in state["images"].items()
            if value["status"] == "failed"
        }
        if failures:
            job["errors"]["sam3"] = failures
        atomic_json(directory / "job.json", job)
