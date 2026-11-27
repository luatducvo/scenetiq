import numpy as np
import numpy.typing as npt

from scenetiq.config import SuperpointOptions
from scenetiq.domain.scene import FloatArray
from scenetiq.vendor.segmentator.segmentator import segment


def mesh_superpoints(
    vertices: FloatArray,
    triangles: npt.ArrayLike,
    k_thresh: float = SuperpointOptions.k_thresh,
    seg_min_verts: int = SuperpointOptions.min_verts,
) -> npt.NDArray[np.int64]:
    return segment(vertices, triangles, kthr=k_thresh, seg_min_verts=seg_min_verts)
