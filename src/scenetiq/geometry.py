"""Geometry helpers on point sets."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scenetiq.domain import Caption, FloatArray, Instance


def build_instances(
    points: FloatArray,
    instance_ids: npt.NDArray[np.int32],
    embeddings: dict[int, FloatArray] | None = None,
    captions: dict[int, tuple[Caption, FloatArray]] | None = None,
) -> list[Instance]:
    """Build an Instance for every non-negative id in ``instance_ids``.

    Args:
        points: ``(N, 3)`` positions in metres.
        instance_ids: ``(N,)`` Instance id per point, ``-1`` for none.
        embeddings: Optional image embedding per Instance id.
        captions: Optional Caption and Caption embedding per Instance id.

    Returns:
        Instances sorted by id, each with its centroid and axis-aligned bounding box.
    """
    embeddings = embeddings or {}
    captions = captions or {}
    instances: list[Instance] = []
    for iid in np.unique(instance_ids):
        if iid < 0:
            continue
        member = points[instance_ids == iid]
        caption, caption_embedding = captions.get(int(iid), (None, None))
        instances.append(
            Instance(
                instance_id=int(iid),
                centroid=member.mean(axis=0).astype(np.float32),
                bbox_min=member.min(axis=0).astype(np.float32),
                bbox_max=member.max(axis=0).astype(np.float32),
                embedding=embeddings.get(int(iid)),
                caption=caption,
                caption_embedding=caption_embedding,
            )
        )
    return instances


def l2_normalize(vector: npt.ArrayLike) -> FloatArray:
    """Return ``vector`` scaled to unit length as float32 (zero vectors stay zero)."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else (arr / norm).astype(np.float32)
