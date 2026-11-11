"""Masking stage: class-agnostic 2D Masks for every View with SAM2 (ADR 0003).

No text prompts are used, so which objects can become Instances does not
depend on any vocabulary.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from scenetiq.config import SAM2Options
from scenetiq.pipeline.gpu_memory import free_gpu_memory
from scenetiq.pipeline.view_data import ViewData

MAX_MASKS_PER_VIEW = 255
"""MV3DIS stores per-pixel mask ranks as uint8, so at most 255 Masks per View."""


@dataclass
class ViewMasks:
    """The Masks of one View.

    Attributes:
        masks: ``(K, H, W)`` uint8 binary Masks at the depth resolution.
        scores: ``(K,)`` float32 confidence of each Mask.
    """

    masks: npt.NDArray[np.uint8]
    scores: npt.NDArray[np.float32]


def select_masks(
    masks: npt.NDArray[np.bool_],
    scores: npt.ArrayLike,
    max_masks: int = MAX_MASKS_PER_VIEW,
    min_area: int = 0,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Keep the ``max_masks`` best-scored Masks whose area is at least ``min_area`` pixels.

    Returns:
        ``(masks, scores)`` sorted by descending score.
    """
    s = np.asarray(scores, dtype=np.float32)
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    candidates = np.flatnonzero(areas >= min_area)
    order = candidates[np.argsort(-s[candidates], kind="stable")][:max_masks]
    return masks[order].astype(np.uint8), s[order]


class SAM2Masker:
    """SAM2 automatic mask generation; the model is on the GPU only during :meth:`masks_for`."""

    def __init__(self, options: SAM2Options | None = None, device: str = "cuda") -> None:
        """Configure the stage; nothing is loaded until :meth:`masks_for`."""
        self.options = options or SAM2Options()
        self.device = device
        self._generator: Any = None

    def _load(self) -> Any:
        if self._generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2_hf

            opts = self.options
            model = build_sam2_hf(opts.model_id, device=self.device)
            self._generator = SAM2AutomaticMaskGenerator(
                model,
                points_per_side=opts.points_per_side,
                pred_iou_thresh=opts.pred_iou_thresh,
                stability_score_thresh=opts.stability_score_thresh,
            )
        return self._generator

    def masks_for(
        self, view_data: ViewData, checkpoint: Callable[[], None] | None = None
    ) -> list[ViewMasks]:
        """Generate Masks for every View, in View order (a View may get none).

        Args:
            view_data: Every View, at the depth-map resolution.
            checkpoint: Called before each View; it raises to stop the stage.
        """
        try:
            return self._generate(view_data, checkpoint)
        finally:
            self._generator = None
            free_gpu_memory()

    def _generate(
        self, view_data: ViewData, checkpoint: Callable[[], None] | None
    ) -> list[ViewMasks]:
        import torch

        generator = self._load()
        _, height, width, _ = view_data.images.shape
        min_area = int(self.options.min_area_fraction * height * width)
        result: list[ViewMasks] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for image in view_data.images:
                if checkpoint is not None:
                    checkpoint()
                records = generator.generate(np.ascontiguousarray(image))
                if not records:
                    result.append(
                        ViewMasks(
                            masks=np.zeros((0, height, width), dtype=np.uint8),
                            scores=np.zeros(0, dtype=np.float32),
                        )
                    )
                    continue
                masks = np.stack([r["segmentation"] for r in records]).astype(bool)
                scores = np.array([r["predicted_iou"] for r in records], dtype=np.float32)
                kept, kept_scores = select_masks(masks, scores, min_area=min_area)
                result.append(ViewMasks(masks=kept, scores=kept_scores))
        return result
