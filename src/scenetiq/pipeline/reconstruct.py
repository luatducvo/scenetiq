"""Reconstructing stage: metric depth and cameras for every View with Depth Anything 3.

ADR 0001: ``DA3NESTED-GIANT-LARGE`` gives depth directly in metres (CC BY-NC weights).
The model sits behind :class:`Reconstructor` so it can be swapped.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from scenetiq.domain import View
from scenetiq.pipeline.adapters import camera_from_da3, confident_depth
from scenetiq.pipeline.frames import Frames

DEFAULT_DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"


class Reconstructor(Protocol):
    """Estimates depth and cameras for a set of Views."""

    def reconstruct(self, views: list[View]) -> Frames:
        """Return depth, cameras and resized images for ``views``, in order."""
        ...


@dataclass(frozen=True)
class DA3Options:
    """Settings of the DA3 Reconstructing stage.

    Attributes:
        model_id: Hugging Face id of the DA3 weights.
        process_res: Long-side resolution DA3 processes at.
        conf_percentile: Pixels below this confidence percentile (over the whole
            Capture) get no depth.
        max_depth: Depth beyond this many metres is dropped.
    """

    model_id: str = DEFAULT_DA3_MODEL
    process_res: int = 504
    conf_percentile: float = 30.0
    max_depth: float = 12.0


class DA3Reconstructor:
    """Depth Anything 3 behind the :class:`Reconstructor` interface.

    The model loads on first use and stays on the GPU between jobs.
    """

    def __init__(self, options: DA3Options | None = None, device: str = "cuda") -> None:
        """Configure the stage; nothing is loaded until :meth:`reconstruct`."""
        self.options = options or DA3Options()
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from depth_anything_3.api import DepthAnything3

            self._model = DepthAnything3.from_pretrained(self.options.model_id).to(self.device)
        return self._model

    def reconstruct(self, views: list[View]) -> Frames:
        """Run DA3 on every View and return metric depth, cameras and resized images."""
        import torch

        model = self._load()
        with torch.inference_mode():
            pred = model.inference(
                image=[str(v.path) for v in views],
                process_res=self.options.process_res,
            )
        depth = np.asarray(pred.depth, dtype=np.float32)
        conf = np.asarray(pred.conf, dtype=np.float32)
        images = np.asarray(pred.processed_images, dtype=np.uint8)
        _, height, width = depth.shape
        min_conf = float(np.percentile(conf, self.options.conf_percentile))
        cameras = [
            camera_from_da3(pred.extrinsics[i], pred.intrinsics[i], height=height, width=width)
            for i in range(len(views))
        ]
        filtered = np.stack(
            [
                confident_depth(depth[i], conf[i], min_conf, self.options.max_depth)
                for i in range(len(views))
            ]
        )
        return Frames(images=images, depth=filtered, cameras=cameras)
