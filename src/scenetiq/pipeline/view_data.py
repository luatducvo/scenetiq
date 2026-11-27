"""Per-View data passed between the stages of the real Pipeline."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scenetiq.domain import Camera, FloatArray


@dataclass
class ViewData:
    """Every View of a Capture at the depth-map resolution.

    Attributes:
        images: ``(N, H, W, 3)`` uint8 RGB, resized to the depth resolution.
        depth: ``(N, H, W)`` float32 metres, 0 where depth is unreliable.
        cameras: Camera of every View; intrinsics refer to ``H x W``.
    """

    images: npt.NDArray[np.uint8]
    depth: FloatArray
    cameras: list[Camera]

    def __len__(self) -> int:
        """Number of Views."""
        return len(self.cameras)


@dataclass
class FusedGeometry:
    """The TSDF-fused surface of the Scene.

    Attributes:
        vertices: ``(N, 3)`` float32 mesh vertices in metres: the Reconstruction's points.
        colors: ``(N, 3)`` uint8 vertex colours.
        triangles: ``(M, 3)`` int32 vertex indices of the mesh faces.
    """

    vertices: FloatArray
    colors: npt.NDArray[np.uint8]
    triangles: npt.NDArray[np.int32]
