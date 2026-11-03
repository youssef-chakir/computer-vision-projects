"""Terminal entry point; all workflow and inference live in dedicated modules."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections import Counter
from pathlib import Path

from config import Settings, load_settings
from pipeline.annotator import (
    Annotator,
    annotations_from,
    export_job,
    preview_job,
    read_job,
    review_job,
)
from storage import job_lock, read_json


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="Local Qwen3-VL + official SAM 3 annotation")
    app.add_argument("--verbose", action="store_true")
    commands = app.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Register images, discover classes, segment and export")
    run.add_argument("--input", type=Path, required=True, help="Recursively scanned source folder")
    run.add_argument("--output", type=Path, help="New job folder; required for --resume")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--stop-after", choices=["qwen", "all"], default="all")
    run.add_argument("--config", type=Path, help="Settings JSON; unknown fields are rejected")
    for name in ("sam-checkpoint", "storage-root"):
        run.add_argument("--" + name, type=Path)
    for name in ("qwen-model", "qwen-revision", "device", "expected-gpu"):
        run.add_argument("--" + name)
    run.add_argument("--precision", choices=["bfloat16", "float16"])
    for name in ("min-area", "max-images", "max-pixels", "max-file-bytes", "qwen-max-pixels"):
        run.add_argument("--" + name, type=int)
    for name in ("min-confidence", "duplicate-iou", "overlay-opacity"):
        run.add_argument("--" + name, type=float)
    run.add_argument(
        "--offline",
        dest="local_files_only",
        action="store_const",
        const=True,
        default=None,
        help="Require an existing local Qwen cache",
    )
    for name, help_text in (
        ("status", "Read progress and review counts without loading models"),
        ("export", "Rebuild COCO and CVAT exports without inference"),
        ("review", "Explicitly set review decisions, retaining history"),
        ("preview", "Regenerate PNGs or display a selected mask in its PNG"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--output", type=Path, required=True)
        if name == "export":
            sub.add_argument(
                "--include-pending",
                action="store_true",
                help="Include pending predictions for correction; never accepts them",
            )
        if name == "review":
            sub.add_argument("--status", choices=["pending", "accepted", "rejected"], required=True)
            selection = sub.add_mutually_exclusive_group(required=True)
            selection.add_argument("--annotation-id", type=int, action="append")
            selection.add_argument("--image-id", type=int)
            selection.add_argument("--all", dest="all_annotations", action="store_true")
        if name == "preview":
            sub.add_argument("--opacity", type=float)
            sub.add_argument("--annotation-id", type=int, action="append")
            sub.add_argument("--image-id", type=int)
    doctor = commands.add_parser(
        "doctor", help="Check the target CUDA runtime and SAM installation"
    )
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--sam-checkpoint", type=Path)
    return app


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.command == "run":
            if args.resume and args.output is None:
                raise ValueError("--resume requires --output")
            defaults = None
            if args.resume:
                _, saved, _ = read_job(args.output.resolve())
                defaults = saved.model_dump(mode="json")
            overrides = {
                key: value
                for key, value in vars(args).items()
                if key in Settings.model_fields and value is not None
            }
            settings = load_settings(args.config, overrides, defaults)
            output = args.output or settings.storage_root / "output" / uuid.uuid4().hex
            result = Annotator(settings).run(
                args.input, output, args.resume, args.retry_failed, args.stop_after
            )
            print(
                json.dumps(
                    {
                        "job_id": result["job_id"],
                        "status": result["status"],
                        "images": len(result["images"]),
                        "errors": result["errors"],
                        "output": str(output.resolve()),
                    },
                    indent=2,
                )
            )
            return 2 if result["errors"] else 0
        if args.command == "doctor":
            from models.runtime import check_runtime
            from models.sam3 import verify_checkpoint, verify_source

            settings = load_settings(args.config, {"sam_checkpoint": args.sam_checkpoint})
            report = check_runtime(settings)
            verify_source()
            report["sam_checkpoint"] = verify_checkpoint(settings)
            print(json.dumps(report, indent=2))
            return 0
        directory = args.output.resolve()
        if not (directory / "job.json").is_file():
            raise ValueError(f"No job.json in {directory}")
        if args.command == "status":
            job, _, images = read_job(directory)
            annotations = annotations_from(directory, job["fingerprint"])
            reviews = read_json(directory / "reviews.json", {"decisions": {}})["decisions"]
            counts = Counter(
                reviews.get(str(a.id), {}).get("status", "pending") for a in annotations
            )
            print(
                json.dumps(
                    {
                        "job_id": job["job_id"],
                        "status": job["status"],
                        "images": len(images),
                        "annotations": len(annotations),
                        "reviews": dict(counts),
                        "errors": job["errors"],
                        "exports_stale": job.get("exports_stale", True),
                    },
                    indent=2,
                )
            )
            return 0
        with job_lock(directory):
            if args.command == "export":
                result = export_job(directory, args.include_pending)
            elif args.command == "review":
                result = {
                    "updated": review_job(
                        directory,
                        args.status,
                        args.annotation_id,
                        args.image_id,
                        args.all_annotations,
                    )
                }
            else:
                result = {
                    "previews": preview_job(
                        directory, args.opacity, args.annotation_id, args.image_id
                    )
                }
            print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted. Resume the same output directory to reuse completed work.")
        return 130
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
