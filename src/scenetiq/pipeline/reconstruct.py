from typing import Any, Protocol

import numpy as np

from scenetiq.config import DA3Options
from scenetiq.domain import View
from scenetiq.pipeline.adapters import camera_from_da3, confident_depth
from scenetiq.pipeline.gpu_memory import free_gpu_memory
from scenetiq.pipeline.view_data import ViewData


class Reconstructor(Protocol):
    def reconstruct(self, views: list[View]) -> ViewData: ...


class DA3Reconstructor:
    def __init__(self, options: DA3Options | None = None, device: str = "cuda") -> None:
        self.options = options or DA3Options()
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from depth_anything_3.api import DepthAnything3

            self._model = DepthAnything3.from_pretrained(self.options.model_id).to(self.device)
        return self._model

    def _predict(self, views: list[View]) -> Any:
        import torch

        model = self._load()
        with torch.inference_mode():
            return model.inference(
                image=[str(v.path) for v in views],
                process_res=self.options.process_res,
                process_res_method=self.options.process_res_method,
                ref_view_strategy=self.options.ref_view_strategy,
                use_ray_pose=self.options.use_ray_pose,
            )

    def reconstruct(self, views: list[View]) -> ViewData:
        try:
            pred = self._predict(views)
        finally:
            self._model = None
            free_gpu_memory()
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
        return ViewData(images=images, depth=filtered, cameras=cameras)
