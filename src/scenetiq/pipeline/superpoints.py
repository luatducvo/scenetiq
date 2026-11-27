"""Superpoints stage: ScanNet Segmentator on the TSDF-fused mesh (ADR 0002)."""

import numpy as np
import numpy.typing as npt

from scenetiq.config import SuperpointOptions
from scenetiq.domain import FloatArray
from scenetiq.vendor.segmentator.segmentator import segment


def mesh_superpoints(
    vertices: FloatArray,
    triangles: npt.ArrayLike,
    k_thresh: float = SuperpointOptions.k_thresh,
    seg_min_verts: int = SuperpointOptions.min_verts,
) -> npt.NDArray[np.int64]:
    """Group mesh vertices into superpoints by normal similarity.

    Args:
        vertices: ``(N, 3)`` mesh vertices, which are the Reconstruction's points.
        triangles: ``(M, 3)`` vertex indices of the faces.
        k_thresh: Segmentation threshold; larger values give larger superpoints.
        seg_min_verts: Smaller superpoints are merged into a neighbour.

    Returns:
        ``(N,)`` superpoint id per vertex, consecutive from 0, in vertex order.
    """
    return segment(vertices, triangles, kthr=k_thresh, seg_min_verts=seg_min_verts)
