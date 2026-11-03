"""Official SAM 3 image/text adapter with checkpoint and API integrity checks."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from config import Settings
from contracts import RawPrediction
from models.runtime import log_memory, release
from storage import sha256

SOURCE_HASHES = {
    "model_builder.py": "60cef72b1f7e4a2162d1c3f1ad2c0451bbbdc10e3484433a9aa2d016fed6b448",
    "model/sam3_image_processor.py": (
        "d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b"
    ),
}


def verify_checkpoint(settings: Settings) -> str:
    path = settings.sam_checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SAM 3 checkpoint missing: {path}; see README setup")
    actual = sha256(path)
    if actual != settings.sam_sha256:
        raise ValueError("SAM checkpoint SHA-256 mismatch; no unverified model will run")
    return str(path)


def verify_source() -> None:
    spec = importlib.util.find_spec("sam3")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the pinned official facebookresearch/sam3 package")
    root = Path(spec.origin).parent
    for relative, expected in SOURCE_HASHES.items():
        data = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"SAM code differs from the pinned official API: {relative}")


class Sam3:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self.processor = None

    def load(self) -> None:
        import torch

        path = verify_checkpoint(self.settings)
        verify_source()
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        # The official builder's checkpoint loader uses strict=False. Build the
        # official architecture, then require every detector tensor to match.
        with torch.cuda.device(self.settings.device):
            self.model = build_sam3_image_model(
                device="cpu",
                checkpoint_path=None,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
                checkpoint = checkpoint["model"]
            detector = {
                key.removeprefix("detector."): value
                for key, value in checkpoint.items()
                if key.startswith("detector.")
            }
            if not detector:
                raise ValueError("Checkpoint has no official SAM 3 detector state")
            self.model.load_state_dict(detector, strict=True)
            del checkpoint, detector
            self.model = self.model.to(self.settings.device).eval()
            self.processor = Sam3Processor(
                self.model,
                device=self.settings.device,
                confidence_threshold=self.settings.min_confidence,
            )
        torch.cuda.reset_peak_memory_stats(self.settings.device)
        log_memory(self.settings.device, "SAM 3 loaded")

    def segment(self, image: Image.Image, classes: list[str]) -> list[RawPrediction]:
        import torch

        if self.processor is None:
            raise RuntimeError("Sam3.load() must be called first")
        predictions = []
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=getattr(torch, self.settings.precision)),
        ):
            state = self.processor.set_image(image)
            try:
                for name in classes:
                    self.processor.reset_all_prompts(state)
                    output = self.processor.set_text_prompt(state=state, prompt=name)
                    masks = output["masks"].detach().cpu().numpy()
                    boxes = output["boxes"].detach().float().cpu().numpy()
                    scores = output["scores"].detach().float().cpu().numpy()
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    if masks.ndim != 3 or masks.shape[1:] != (image.height, image.width):
                        raise ValueError(f"Unexpected SAM mask shape: {masks.shape}")
                    if len(masks) != len(scores) or boxes.shape != (len(masks), 4):
                        raise ValueError("SAM mask, box and score counts disagree")
                    for mask, box, score in zip(masks, boxes, scores, strict=True):
                        predictions.append(
                            RawPrediction(
                                name,
                                np.array(mask, copy=True),
                                box.tolist(),
                                float(score),
                            )
                        )
            finally:
                state.clear()
        return predictions

    def close(self) -> None:
        self.processor = None
        self.model = None
        release(self.settings.device)
