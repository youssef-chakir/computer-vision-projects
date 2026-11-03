# Auto Annotator

Local, terminal-driven annotation with **Qwen3-VL-4B-Instruct** for class
discovery and **official SAM 3** for text-prompted instance segmentation.
The application has no HTTP server, web interface, cloud inference, mock
backend, or manual geometry editor. Review uses PNG previews and explicit
CLI decisions; CVAT handles mask correction.

The brief mentions both instance and panoptic output, so both are exported.
The explicit CLI-only requirement takes precedence over its frontend references.

## Install on your RTX 5090 server

These commands target a Linux/Anaconda server with Python 3.12 and the supplied
RTX 5090 specifications. They have **not** been executed on your server.
No server connection is required to develop or inspect this repository.

From the copied `auto-annotator` directory:

```bash
conda create -n auto-annotator python=3.12 pip -y
conda activate auto-annotator

python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[models]"
python -m pip check
python -m unittest discover -s tests -v
```

The CUDA 12.8 PyTorch wheels support Blackwell. A driver reporting CUDA 13.0
does not require a CUDA 13.0 wheel or a separate CUDA toolkit for this setup.
Compilation and optional FlashAttention extensions are disabled. The install
recipe follows [Meta's prerequisites](https://github.com/facebookresearch/sam3#installation)
and the [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/);
it is an inspected integration recipe, not a GPU-tested lockfile.

Request access to [facebook/sam3](https://huggingface.co/facebook/sam3), accept
Meta's license yourself, then download the pinned checkpoint:

```bash
hf auth login
hf download facebook/sam3 sam3.pt --revision 3c879f39826c281e95690f02c7821c4de09afae7 --local-dir checkpoints
hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
python main.py doctor --sam-checkpoint checkpoints/sam3.pt
```

Qwen is public and its model card declares Apache-2.0. SAM 3 access is gated;
its custom terms apply separately. Tokens belong in Hugging Face's credential
store or your environment, never in source files.

Start with a small existing image folder for the real GPU smoke test:

```bash
python main.py run --input "/data/pilot-images" --output storage/output/pilot --sam-checkpoint checkpoints/sam3.pt
python main.py status --output storage/output/pilot
python main.py export --output storage/output/pilot --include-pending
```

Afterward, inspect `job.log`, `qwen_predictions.json`,
`annotations_SAM3.json`, and the side-by-side PNGs in `previews_visual/`.
Then point the same command at the full collection with a new output directory.
`--max-images` is a hard safety limit, not a sampling option: exceeding it
raises an error instead of silently dropping input files.

## Model provenance and memory strategy

| Component | Pinned source |
| --- | --- |
| Qwen checkpoint | `Qwen/Qwen3-VL-4B-Instruct` |
| Qwen revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Transformers | `4.57.6`; official Qwen instructions require at least 4.57.0 |
| PyTorch / torchvision | `2.10.0` / `0.25.0`, CUDA 12.8 |
| SAM source commit | `f1647e8d0c1d280991f4d1b7d974c2704537fe66` |
| SAM checkpoint | `facebook/sam3`, file `sam3.pt` |
| SAM weights revision | `3c879f39826c281e95690f02c7821c4de09afae7` |
| SAM checkpoint SHA-256 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

Sources were inspected on 2026-11-03:
[Qwen's official usage](https://github.com/QwenLM/Qwen3-VL#quickstart),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
[SAM image builder](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model_builder.py),
[SAM image processor](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model/sam3_image_processor.py).

Qwen uses `Qwen3VLForConditionalGeneration`, `AutoProcessor`,
`apply_chat_template`, deterministic generation and prompt-token trimming.
Its response must contain exactly a JSON `classes` list. Invalid responses
are saved as failures; valid empty lists remain empty. No fallback classes
are invented. Prompt constraints cannot guarantee absence of hallucinations,
so every prediction still requires review.

SAM uses `build_sam3_image_model`, `Sam3Processor.set_image`, and
`set_text_prompt`. It encodes each image once, resets prompt state between
classes, and returns the actual masks, xyxy boxes and scores. There is no SAM 2
or SAM 3.1 substitution. The adapter verifies the checkpoint digest and the
inspected builder/processor source hashes. It loads detector weights strictly,
rather than accepting the upstream loader's non-strict partial match.

The stages are sequential: load Qwen once, process the collection, save
predictions, remove model/processor references, collect garbage, release the
CUDA cache, then load SAM once. Both adapters use inference mode and mixed
precision, BF16 by default. Each image's time and GPU allocated/reserved/peak
memory are logged. An image or model failure is recorded, not replaced by
synthetic output.

The Qwen repository metadata lists about 4.44 billion BF16 parameters, roughly
8.27 GiB of weight storage before activations and generation buffers. SAM's
download is about 3.45 GB. These are storage estimates, **not measured peak
VRAM requirements**. Staging and a default Qwen input limit of 1,048,576 pixels
reduce pressure; confirm actual peaks using the pilot on your 32 GB GPU.
Large original masks can still consume substantial memory.

## Commands and configuration

```bash
# Inspect class discovery before starting segmentation.
python main.py run --input "/data/images" --output storage/output/rabat --stop-after qwen

# Continue the same immutable image collection, reusing successful intermediates.
python main.py run --input "/data/images" --output storage/output/rabat --resume

# Retry failed images only; successful predictions and masks stay cached.
python main.py run --input "/data/images" --output storage/output/rabat --resume --retry-failed

# Re-render without inference; IDs come from annotations_SAM3.json.
python main.py preview --output storage/output/rabat --opacity 0.35
python main.py preview --output storage/output/rabat --annotation-id 123

# Explicit decisions after visual inspection; replace 123 with a real ID.
python main.py review --output storage/output/rabat --annotation-id 123 --status accepted
python main.py review --output storage/output/rabat --annotation-id 123 --status rejected

# Default exports contain accepted annotations only.
python main.py export --output storage/output/rabat

# Draft export for correction in CVAT; pending is NOT relabeled accepted.
python main.py export --output storage/output/rabat --include-pending
```

`review` also supports `--image-id` or an explicit `--all`. It stores
decisions and history separately in `reviews.json`; rejected masks and
prediction records remain intact. Corrected CVAT geometry stays in CVAT and
its exported dataset. This tool does not re-import CVAT edits into predictions.

Use `python main.py run --help` for available options. All settings are in
`config.py`; use `--config settings.json` for persistent overrides, for example:

```json
{
  "min_confidence": 0.5,
  "min_area": 32,
  "duplicate_iou": 0.95,
  "max_images": 10000,
  "max_pixels": 40000000,
  "qwen_max_pixels": 1048576,
  "synonyms": {"cars": "car", "people": "person", "motorbike": "motorcycle"},
  "stuff_classes": ["road", "sidewalk", "sky", "vegetation", "wall"]
}
```

A supplied dictionary/list replaces that default field. Synonyms are explicit
and cycle-checked; unknown classes are not silently merged. New normalized
classes receive append-only IDs in sorted order. Unlisted classes are things;
adjust `stuff_classes` deliberately before starting a job.

Precedence is saved resume settings, JSON configuration, `AUTO_` environment
variables, then explicit CLI flags. `.env.example` documents the variables;
the application does not execute or automatically load .env files.
`--offline` requires Qwen files already cached. Export, review, preview, status
and the deterministic tests do not import PyTorch or load models.

Resume requires unchanged source bytes, file list and inference configuration.
Changing model/threshold/ontology settings requires a new job; rendering
opacity and local checkpoint locations can change. Corrupt or missing cached
masks cause a clear failure instead of silently treating them as valid.
The OS job lock prevents simultaneous writers and is released on process exit.

Exit codes: 0 for successful work, 2 for completed processing with per-image
errors, 1 for a fatal/configuration error, and 130 for interruption.

## COCO, panoptic and CVAT

All valid registered images are exported, including images with zero accepted
annotations or a recorded model failure. Read job errors before treating such
images as negative examples.

Instance output uses lossless **uncompressed COCO RLE** in column-major order,
with mask-derived area and xywh bounding boxes. Individual masks retain
`iscrowd: 0`; overlapping instances and holes are preserved. IDs and references
are validated, RLE is decoded back for a round-trip check, and mask hashes
are checked before export.

The CVAT target is **COCO 1.0**. Its
[documented COCO format](https://docs.cvat.ai/docs/dataset_management/formats/format-coco/)
supports masks and this archive structure:

```text
cvat_export.zip
├── images/default/<safe-relative-image-path>
├── annotations/instances_default.json
├── provenance.json
└── originals/<source-path>              # only for EXIF-oriented derivatives
```

For an existing CVAT task, upload the images from `images/default/`, configure
labels with the names in `categories.json`, then choose **Upload annotations →
COCO 1.0** and upload the JSON (or annotation archive). For a project dataset
import, select COCO 1.0 and the complete ZIP. Use `--include-pending` when
sending unreviewed predictions for correction; the ordinary export may
intentionally have no annotations until you accept some.

For EXIF orientation other than 1, the dataset uses an upright, lossless PNG
under `images/default/_oriented/<image-id>/`. The original file and metadata
remain byte-for-byte intact under `originals/`. Ordinary images are copied
byte-for-byte directly into `images/default/`. The provenance mapping records
both paths. Use the images packaged in the ZIP when importing annotations;
this avoids depending on CVAT's handling of EXIF rotation.

A **real CVAT import has not been performed**. On your pilot, verify image
count, labels, editable masks, a mask containing a hole, overlapping instances,
and an EXIF-oriented image; export back from CVAT and inspect the results.
The local format checks are not a substitute for that integration test.

`panoptic_export.zip` additionally follows the COCO panoptic JSON/RGB-PNG
convention. Higher-scoring masks own overlap pixels; ties favor smaller area,
then annotation ID. Stuff regions of the same category merge, things remain
separate, and uncovered pixels are void (ID 0). Area/bbox are recomputed after
composition. Instance masks are never changed. This provides a valid panoptic
representation of predicted coverage, not a guarantee of exhaustive scene
labeling or ground truth.

Exports are rebuilt without inference. Files are replaced atomically and
`export_manifest.json` is written last with output hashes. If an export fails,
`exports_stale` remains true and existing archives are not claimed as current.

## Project layout and contracts

```text
auto-annotator/
├── models/
│   ├── qwen.py                 # Official class-discovery adapter and strict parsing.
│   ├── sam3.py                 # Official text-prompted segmentation and provenance checks.
│   └── runtime.py              # CUDA preflight, memory logging and resource cleanup.
├── processing/
│   ├── images.py               # Recursive discovery, content checks and EXIF preparation.
│   ├── categories.py           # Normalization, synonyms and append-only category IDs.
│   └── validate.py             # Mask validation, conservative duplicate filtering and geometry.
├── pipeline/annotator.py        # Stage orchestration, resume, review and export coordination.
├── visualization/overlays.py   # Side-by-side PNGs, consistent colors, boundaries and labels.
├── exporters/coco.py           # Lossless instance RLE, panoptic PNGs and CVAT ZIPs.
├── main.py                     # argparse commands and exit status.
├── config.py                   # Pydantic settings and immutable model pins.
├── contracts.py                # Typed image, prediction, category and annotation records.
├── storage.py                  # Atomic writes, safe paths, hashes and job locking.
├── tests/test_core.py           # Synthetic deterministic regression fixtures; no model backend.
├── pyproject.toml              # Dependencies, installation metadata and lint configuration.
├── .env.example                # Optional shell environment settings without secrets.
├── .gitignore                  # Excludes runtime data, caches, checkpoints and credentials.
└── README.md
```

Each module directory includes a small `__init__.py`. The two shared files
`contracts.py` and `storage.py` keep schemas and filesystem operations out of
the model layer without adding another package hierarchy.

Runtime data is created only when needed:

```text
storage/
├── imports/<import-id>.json     # Source references, content hashes, dimensions and metadata.
├── uploads/                    # Optional user-owned source folders; external inputs also work.
└── output/<job-id-or-name>/
    ├── job.json
    ├── job.log
    ├── qwen_predictions.json
    ├── categories.json
    ├── annotations_SAM3.json
    ├── reviews.json
    ├── masks/<image-id>/<annotation-id>.png
    ├── previews_visual/<image-id>.png
    ├── panoptic/<image-id>.png
    ├── instances_default.json
    ├── panoptic_default.json
    ├── cvat_export.zip
    ├── panoptic_export.zip
    └── export_manifest.json
```

`--output` can use another directory such as `outputs/rabat`; image input can
be `uploads/rabat` or an external folder. Output must be outside the input
tree. Imported originals are referenced, not copied into working storage.

Image IDs derive from safe relative paths. Annotation IDs derive from image
ID, category and mask content; IDs remain exact in JSON/JavaScript numbers.
Binary masks are grayscale PNGs containing only 0 and 255. Their dimensions
match the upright RGB image. `mask_ref` always points to this binary geometry,
never to a colored preview. Annotation records retain source scores, category,
actual pixel area, bbox, hash and initial pending state; decisions live in
`reviews.json`. Validation exclusions and reasons are stored per image.

## Validation and troubleshooting

The local tests exercise image preservation and orientation, path limits,
category stability, malformed Qwen responses, mask validation, RLE counts,
overlap composition, accepted/pending/rejected export, cache reuse, failure
cleanup, atomic writes, and locks. Test fixtures are synthetic and do not
measure Qwen accuracy, SAM quality, GPU compatibility, runtime or memory use.
The independent `pycocotools` decoder check runs when that package is installed.

Local validation uses an existing Python 3.12 environment; no server, CVAT
instance, or real checkpoints were used. Actual GPU inference, peak VRAM,
segmentation quality and CVAT import remain to be validated by you.

- Missing/gated SAM weights: obtain access, log in, download the pinned file.
- SHA/source mismatch: use the specified checkpoint and source commit. If
  deliberately upgrading, inspect the new API and update pins/hashes together.
- Out of memory: reduce `qwen_max_pixels` in a new job and close other GPU
  workloads. Large original image masks require additional SAM memory.
- Invalid Qwen JSON: inspect the saved response, then retry failed images;
  no classes are fabricated to hide a failure.
- Changed input or settings: start a separate output directory.
- Empty default export: predictions are pending. Review them or explicitly
  export pending masks for CVAT correction.
- Windows protected-folder write errors: use a writable working/output
  directory. The deployment instructions target your Linux server.

