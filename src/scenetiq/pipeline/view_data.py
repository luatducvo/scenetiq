from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scenetiq.domain import Camera, FloatArray


@dataclass
class ViewData:
    images: npt.NDArray[np.uint8]
    depth: FloatArray
    cameras: list[Camera]

    def __len__(self) -> int:
        return len(self.cameras)


@dataclass
class FusedGeometry:
    vertices: FloatArray
    colors: npt.NDArray[np.uint8]
    triangles: npt.NDArray[np.int32]
