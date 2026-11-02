"""Choosing which ScanNet frames become the Views of an evaluation Capture."""

from collections.abc import Sequence

import numpy as np


def sample_evenly[T](frames: Sequence[T], max_views: int) -> list[T]:
    """Pick at most ``max_views`` frames spread evenly over ``frames``, keeping order."""
    if len(frames) <= max_views:
        return list(frames)
    step = len(frames) / max_views
    return [frames[int(np.floor(i * step))] for i in range(max_views)]
