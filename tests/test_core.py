"""Synthetic fixtures test storage/geometry only, never model quality or real inference."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import Settings
from contracts import Annotation, Category, RawPrediction
from exporters.coco import compose_panoptic, decode_rle, encode_rle, export_dataset
from models.qwen import QwenResponseError, parse_classes
from pipeline.annotator import Annotator, export_job, review_job
from processing.categories import CategoryRegistry
from processing.images import discover, inspect_image, open_registered
from processing.validate import mask_geometry, validate_instances
from storage import atomic_json, job_lock, read_json, safe_path, save_png, sha256


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="auto-annotator-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.job = self.root / "job"
        self.settings = Settings(storage_root=self.root / "storage", min_area=1)
        Image.new("RGB", (8, 6), (80, 100, 120)).save(self.source / "image.png")
        self.record = inspect_image(self.source, "image.png", self.settings)

    def seed_job(self, empty=False):
        job = Annotator(self.settings)._register(self.source, self.job, False, False)
        registry = CategoryRegistry(self.job / "categories.json", {}, ["road"])
        registry.register(["car", "road"])
        atomic_json(
            self.job / "qwen_predictions.json",
            {
                "fingerprint": self.settings.fingerprint(),
                "predictions": {
                    str(self.record.id): {
                        "image_id": self.record.id,
                        "status": "empty" if empty else "succeeded",
                        "classes": [] if empty else ["car"],
                        "raw_response": "",
                    }
                },
            },
        )
        annotations = []
        if not empty:
            mask = np.zeros((6, 8), dtype=bool)
            mask[1:5, 2:6] = True
            mask[2, 3] = False  # Preserve a hole: no polygon approximation.
            annotation = self.annotation(mask)
            annotations.append(annotation)
            atomic_json(
                self.job / "annotations_SAM3.json",
                {
                    "fingerprint": self.settings.fingerprint(),
                    "images": {
                        str(self.record.id): {
                            "status": "succeeded",
                            "annotations": [annotation.model_dump()],
                            "issues": [],
                            "seconds": 0,
                        }
                    },
                },
            )
        return job, registry, annotations

    def annotation(self, mask, identifier=101, category=1, name="car", score=0.9):
        relative = f"masks/{identifier}.png"
        save_png(self.job / relative, Image.fromarray(mask.astype(np.uint8) * 255))
        bbox, area = mask_geometry(mask)
        return Annotation(
            id=identifier,
            image_id=self.record.id,
            category_id=category,
            category_name=name,
            score=score,
            bbox=bbox,
            area=area,
            mask_ref=relative,
            mask_sha256=sha256(self.job / relative),
        )

    def test_duplicate_filenames_have_distinct_stable_ids(self):
        nested = self.source / "nested"
        nested.mkdir()
        Image.new("RGB", (2, 3)).save(nested / "image.png")
        (nested / "notes.txt").write_text("ignored")
        paths = discover(self.source, self.settings)
        self.assertEqual(paths, ["image.png", "nested/image.png"])
        records = [inspect_image(self.source, path, self.settings) for path in paths]
        self.assertNotEqual(records[0].id, records[1].id)
        self.assertEqual(records[0].id, inspect_image(self.source, paths[0], self.settings).id)

    def test_actual_content_and_resolution_are_validated(self):
        (self.source / "bad.jpg").write_bytes(b"not an image")
        with self.assertRaises(UnidentifiedImageError):
            inspect_image(self.source, "bad.jpg", self.settings)
        with self.assertRaises(ValueError):
            inspect_image(self.source, "image.png", Settings(max_pixels=1))

    def test_count_limit_does_not_silently_truncate(self):
        Image.new("RGB", (1, 1)).save(self.source / "another.png")
        with self.assertRaisesRegex(ValueError, "nothing truncated"):
            discover(self.source, Settings(max_images=1))

    def test_unsafe_paths_are_rejected(self):
        for relative in ("../x", "/x", "C:/x", "a/../../x", "a\\x", "a/./x", ""):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_path(self.source, relative)

    def test_originals_preserved_and_exif_orientation_corrected(self):
        path = self.source / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (8, 6), "red").save(path, exif=exif)
        before = path.read_bytes()
        record = inspect_image(self.source, "rotated.jpg", self.settings)
        self.assertEqual((record.width, record.height), (6, 8))
        with open_registered(self.source, record, self.settings.max_pixels) as image:
            self.assertEqual(image.size, (6, 8))
        self.assertEqual(path.read_bytes(), before)

    def test_categories_append_without_reassigning(self):
        path = self.root / "categories.json"
        registry = CategoryRegistry(path, {"cars": "car"}, [])
        registry.register(["Person", "cars", "car"])
        self.assertEqual(registry.by_name, {"car": 1, "person": 2})
        resumed = CategoryRegistry(path, {"cars": "car"}, [])
        resumed.register(["bicycle", "person"])
        self.assertEqual(resumed.by_name, {"car": 1, "person": 2, "bicycle": 3})

    def test_synonym_cycles_fail(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            CategoryRegistry(self.root / "categories.json", {"a": "b", "b": "a"}, [])

    def test_qwen_parser_empty_and_deduplicated_results(self):
        self.assertEqual(parse_classes('{"classes":[]}', 5), [])
        self.assertEqual(
            parse_classes('{"classes":["Car"," car ","person"]}', 5), ["car", "person"]
        )
        fenced = chr(96) * 3 + 'json\n{"classes":["car"]}\n' + chr(96) * 3
        self.assertEqual(parse_classes(fenced, 5), ["car"])

    def test_qwen_parser_rejects_hallucinated_format_fallbacks(self):
        for text in (
            "car, person",
            '{"classes":"car"}',
            '{"classes":[1]}',
            '{"classes":["car"],"description":"hello"}',
        ):
            with self.subTest(text=text), self.assertRaises(QwenResponseError):
                parse_classes(text, 5)

    def test_validation_keeps_overlapping_instances_and_class_association(self):
        a = np.zeros((6, 8), dtype=bool)
        b = a.copy()
        a[1:5, 1:5] = True
        b[1:5, 2:6] = True
        raw = [
            RawPrediction("car", a, [float("nan"), 0, 0, 0], 0.99),
            RawPrediction("car", a.copy(), score=0.8),
            RawPrediction("car", b, score=0.9),
            RawPrediction("road", a.copy(), score=0.7),
        ]
        valid, issues = validate_instances(raw, {"car": 1, "road": 2}, (8, 6), self.settings)
        self.assertEqual(len(valid), 3)
        self.assertEqual(valid[0].bbox, (1, 1, 4, 4))
        self.assertEqual(valid[0].area, 16)
        self.assertEqual([v.category_id for v in valid], [1, 1, 2])
        self.assertTrue(any("duplicate" in issue["reason"] for issue in issues))
        self.assertTrue(any("box" in issue["reason"] for issue in issues))

    def test_invalid_low_score_empty_and_wrong_shape_masks_are_reported(self):
        raw = [
            RawPrediction("car", np.zeros((6, 8)), score=0.9),
            RawPrediction("car", np.ones((1, 1)), score=0.9),
            RawPrediction("car", np.ones((6, 8)), score=0.1),
            RawPrediction("car", np.full((6, 8), np.nan), score=0.9),
        ]
        valid, issues = validate_instances(raw, {"car": 1}, (8, 6), self.settings)
        self.assertEqual(valid, [])
        self.assertEqual(len(issues), 4)

    def test_uncompressed_coco_rle_has_reference_column_major_counts(self):
        mask = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
        rle = encode_rle(mask)
        self.assertEqual(rle, {"size": [2, 3], "counts": [0, 1, 2, 1, 1, 1]})
        np.testing.assert_array_equal(mask, decode_rle(rle))
        self.assertEqual(encode_rle(np.zeros((2, 3), dtype=bool))["counts"], [6])
        self.assertEqual(encode_rle(np.ones((2, 3), dtype=bool))["counts"], [0, 6])

    @unittest.skipUnless(
        importlib.util.find_spec("pycocotools"), "pycocotools not installed locally"
    )
    def test_official_coco_decoder_accepts_our_rle(self):
        from pycocotools import mask as mask_tools

        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        encoded = mask_tools.frPyObjects(encode_rle(mask), 2, 3)
        np.testing.assert_array_equal(mask, mask_tools.decode(encoded).astype(bool))

    def test_invalid_rle_rejected(self):
        for counts in ([1], [-1, 7], [6.0]):
            with self.assertRaises(ValueError):
                decode_rle({"size": [2, 3], "counts": counts})

    def test_accepted_only_export_and_explicit_pending_export(self):
        _, _, annotations = self.seed_job()
        result = export_job(self.job)
        self.assertEqual(result["annotations"], 0)
        self.assertEqual(result["images"], 1)
        result = export_job(self.job, include_pending=True)
        self.assertEqual(result["annotations"], 1)
        with zipfile.ZipFile(self.job / "cvat_export.zip") as archive:
            self.assertIn("annotations/instances_default.json", archive.namelist())
            self.assertEqual(
                archive.read("images/default/image.png"), (self.source / "image.png").read_bytes()
            )
            document = json.loads(archive.read("annotations/instances_default.json"))
            exported = document["annotations"][0]
            self.assertEqual(exported["attributes"]["review_status"], "pending")
            self.assertEqual(exported["iscrowd"], 0)
            self.assertEqual(int(decode_rle(exported["segmentation"]).sum()), annotations[0].area)
        self.assertFalse((self.job / "reviews.json").exists())

    def test_review_is_separate_and_rejection_preserves_masks(self):
        _, _, annotations = self.seed_job()
        original = sha256(self.job / "annotations_SAM3.json")
        review_job(self.job, "accepted", ids=[101])
        self.assertEqual(export_job(self.job)["annotations"], 1)
        review_job(self.job, "rejected", ids=[101])
        self.assertEqual(export_job(self.job)["annotations"], 0)
        self.assertEqual(sha256(self.job / "annotations_SAM3.json"), original)
        self.assertTrue((self.job / annotations[0].mask_ref).exists())
        self.assertEqual(len(read_json(self.job / "reviews.json")["history"]), 2)

    def test_panoptic_resolves_overlap_and_keeps_void(self):
        a = np.zeros((6, 8), dtype=bool)
        b = a.copy()
        a[1:4, 1:4] = True
        b[2:5, 2:5] = True
        first = self.annotation(a, score=0.9)
        second = self.annotation(b, identifier=102, score=0.8)
        rgb, segments = compose_panoptic(
            [second, first], {101: a, 102: b}, {1: Category(id=1, name="car")}, (8, 6)
        )
        ids = np.asarray(rgb).astype(np.uint32)
        ids = ids[..., 0] + (ids[..., 1] << 8) + (ids[..., 2] << 16)
        self.assertEqual([s["area"] for s in segments], [9, 5])
        self.assertEqual(ids[2, 2], 1)
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(int((ids > 0).sum()), int((a | b).sum()))

    def test_panoptic_merges_stuff_of_same_category(self):
        a = np.zeros((6, 8), dtype=bool)
        b = a.copy()
        a[0, 0] = True
        b[5, 7] = True
        first = self.annotation(a, name="road")
        second = self.annotation(b, identifier=102, name="road")
        _, segments = compose_panoptic(
            [first, second], {101: a, 102: b}, {1: Category(id=1, name="road", isthing=0)}, (8, 6)
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["area"], 2)

    def test_oriented_export_contains_upright_pixels_and_original_bytes(self):
        path = self.source / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (8, 6), "green").save(path, exif=exif)
        record = inspect_image(self.source, "rotated.jpg", self.settings)
        export_dataset(self.job, self.source, [record], [], [], {}, self.settings.max_pixels)
        with zipfile.ZipFile(self.job / "cvat_export.zip") as archive:
            document = json.loads(archive.read("annotations/instances_default.json"))
            filename = document["images"][0]["file_name"]
            with Image.open(io.BytesIO(archive.read("images/default/" + filename))) as upright:
                self.assertEqual(upright.size, (6, 8))
                self.assertIsNone(upright.getexif().get(274))
            self.assertEqual(archive.read("originals/rotated.jpg"), path.read_bytes())

    def test_changed_source_is_never_exported(self):
        self.seed_job()
        Image.new("RGB", (8, 6), "black").save(self.source / "image.png")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            export_job(self.job, include_pending=True)

    def test_corrupt_mask_is_never_exported(self):
        self.seed_job()
        (self.job / "masks/101.png").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Mask changed"):
            export_job(self.job, include_pending=True)

    def test_resume_reuses_valid_caches_without_loading_either_model(self):
        self.seed_job()
        review_job(self.job, "accepted", ids=[101])
        before = sha256(self.job / "qwen_predictions.json")
        with (
            patch("models.qwen.Qwen.load", side_effect=AssertionError("must reuse Qwen")) as qwen,
            patch("models.sam3.Sam3.load", side_effect=AssertionError("must reuse SAM")) as sam,
        ):
            result = Annotator(self.settings).run(self.source, self.job, resume=True)
        qwen.assert_not_called()
        sam.assert_not_called()
        self.assertEqual(result["status"], "ready_for_review")
        self.assertEqual(result["last_export"]["annotations"], 1)
        self.assertEqual(sha256(self.job / "qwen_predictions.json"), before)
        self.assertTrue((self.job / "previews_visual" / f"{self.record.id}.png").exists())

    def test_empty_qwen_result_never_invents_classes_or_loads_sam(self):
        self.seed_job(empty=True)
        with (
            patch("models.qwen.Qwen.load", side_effect=AssertionError("cached")),
            patch("models.sam3.Sam3.load", side_effect=AssertionError("no classes")),
        ):
            result = Annotator(self.settings).run(self.source, self.job, resume=True)
        self.assertEqual(result["last_export"]["annotations"], 0)
        state = read_json(self.job / "annotations_SAM3.json")
        self.assertEqual(state["images"][str(self.record.id)]["status"], "empty")

    def test_changed_config_rejects_resume(self):
        self.seed_job()
        changed = self.settings.model_copy(update={"min_confidence": 0.8})
        with self.assertRaisesRegex(ValueError, "settings changed"):
            Annotator(changed).run(self.source, self.job, resume=True)

    def test_cuda_device_name_is_explicit(self):
        self.assertEqual(Settings(device="cuda").device, "cuda:0")
        self.assertEqual(Settings(device="cuda:1").device, "cuda:1")

    def test_atomic_write_failure_keeps_previous_json(self):
        target = self.root / "atomic.json"
        atomic_json(target, {"valid": 1})
        with self.assertRaises(ValueError):
            atomic_json(target, {"invalid": float("nan")})
        self.assertEqual(read_json(target), {"valid": 1})
        self.assertFalse(list(self.root.glob(".tmp-*")))

    def test_output_cannot_be_inside_input(self):
        with self.assertRaisesRegex(ValueError, "no nesting"):
            Annotator(self.settings).run(self.source, self.source / "output")

    def test_job_lock_rejects_second_writer(self):
        with job_lock(self.job), self.assertRaises(RuntimeError), job_lock(self.job):
            pass

    def test_failed_model_load_closes_adapter_and_records_failure(self):
        with (
            patch.object(Annotator, "_ensure_runtime"),
            patch("models.qwen.Qwen.load", side_effect=RuntimeError("checkpoint unavailable")),
            patch("models.qwen.Qwen.close") as close,
            self.assertRaisesRegex(RuntimeError, "checkpoint unavailable"),
        ):
            Annotator(self.settings).run(self.source, self.job)
        close.assert_called_once()
        self.assertEqual(read_json(self.job / "job.json")["status"], "failed")

    def test_all_invalid_images_record_a_failed_job(self):
        (self.source / "image.png").write_bytes(b"corrupt image fixture")
        with self.assertRaisesRegex(ValueError, "No decodable"):
            Annotator(self.settings).run(self.source, self.job)
        self.assertEqual(read_json(self.job / "job.json")["status"], "failed")

    def test_failed_predictions_are_reused_until_retry_is_requested(self):
        Annotator(self.settings)._register(self.source, self.job, False, False)
        atomic_json(
            self.job / "qwen_predictions.json",
            {
                "fingerprint": self.settings.fingerprint(),
                "predictions": {
                    str(self.record.id): {
                        "image_id": self.record.id,
                        "status": "failed",
                        "classes": [],
                        "error": "invalid JSON fixture",
                        "raw_response": "invalid",
                    }
                },
            },
        )
        with (
            patch("models.qwen.Qwen.load", side_effect=AssertionError("must not retry")),
            patch("models.sam3.Sam3.load", side_effect=AssertionError("no classes")),
        ):
            result = Annotator(self.settings).run(self.source, self.job, resume=True)
        self.assertEqual(result["status"], "ready_with_errors")
        with (
            patch.object(Annotator, "_ensure_runtime"),
            patch("models.qwen.Qwen.load", side_effect=RuntimeError("retry attempted")),
            patch("models.qwen.Qwen.close"),
            self.assertRaisesRegex(RuntimeError, "retry attempted"),
        ):
            Annotator(self.settings).run(self.source, self.job, resume=True, retry_failed=True)


if __name__ == "__main__":
    unittest.main()
