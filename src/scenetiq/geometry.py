from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scenetiq.domain import EmbeddedCaption, FloatArray, Instance


def build_instances(
    points: FloatArray,
    instance_ids: npt.NDArray[np.int32],
    embeddings: dict[int, FloatArray] | None = None,
    captions: dict[int, EmbeddedCaption] | None = None,
) -> list[Instance]:
    embeddings = embeddings or {}
    captions = captions or {}
    instances: list[Instance] = []
    for iid in np.unique(instance_ids):
        if iid < 0:
            continue
        member = points[instance_ids == iid]
        embedded = captions.get(int(iid))
        instances.append(
            Instance(
                instance_id=int(iid),
                centroid=member.mean(axis=0).astype(np.float32),
                bbox_min=member.min(axis=0).astype(np.float32),
                bbox_max=member.max(axis=0).astype(np.float32),
                embedding=embeddings.get(int(iid)),
                caption=None if embedded is None else embedded.caption,
                caption_embeddings={} if embedded is None else dict(embedded.embeddings),
            )
        )
    return instances


def l2_normalize(vector: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else (arr / norm).astype(np.float32)
