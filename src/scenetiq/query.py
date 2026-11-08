"""Open-vocabulary Query over the Instances of a Reconstruction (ADR 0008).

A Query is ranked against every Instance's image embedding and against the Caption embedding
of every Instance that has one. The rankings are fused with Reciprocal Rank Fusion and the
top K Instances are the Matches.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scenetiq.domain import FloatArray, Instance


class MissingEmbeddingsError(ValueError):
    """Raised when a Query targets Instances that have no embeddings yet."""


@dataclass(frozen=True)
class Match:
    """An Instance returned for a Query.

    Attributes:
        instance_id: The Matched Instance.
        score: Fused rank score normalised to [0, 1]; relative to this Query only.
        image_score: Cosine similarity of the Query to the Instance's image embedding.
        caption_score: Cosine similarity of the Query to the Instance's Caption, or
            ``None`` when it has none.
    """

    instance_id: int
    score: float
    image_score: float
    caption_score: float | None = None


def rank_by_similarity(ids: Sequence[int], similarities: Sequence[float]) -> list[int]:
    """Order ``ids`` by descending similarity, breaking ties by ascending id."""
    return [i for _, i in sorted(zip(similarities, ids, strict=True), key=lambda p: (-p[0], p[1]))]


def rrf_scores(rankings: Sequence[Sequence[int]], rrf_k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion of several rankings.

    Args:
        rankings: Instance ids, best first; an Instance may be missing from some.
        rrf_k: The constant added to every rank (ranks start at 1).

    Returns:
        Instance id -> sum over the rankings it appears in of ``1 / (rrf_k + rank)``.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, iid in enumerate(ranking, start=1):
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def run_query(
    instances: list[Instance],
    image_query: FloatArray,
    top_k: int,
    rrf_k: int,
    caption_query: FloatArray | None = None,
) -> list[Match]:
    """Rank every Instance against a Query and return the top ``top_k``.

    Args:
        instances: Instances with L2-normalised image embeddings.
        image_query: L2-normalised embedding of the Query text in the image space.
        top_k: How many Matches to return at most.
        rrf_k: The Reciprocal Rank Fusion constant.
        caption_query: L2-normalised embedding of the Query text in the Caption space;
            the Caption ranking is skipped when omitted.

    Returns:
        Up to ``top_k`` Matches, best first, ties broken by image similarity then
        Instance id. An Instance without a Caption gets only its image rank. Scores are
        normalised by the best possible fused value over the rankings that exist. A
        Reconstruction without Instances yields no Matches.

    Raises:
        MissingEmbeddingsError: If any Instance has no image embedding.
    """
    if not instances:
        return []
    embeddings = [inst.embedding for inst in instances]
    if any(e is None for e in embeddings):
        raise MissingEmbeddingsError("Scene Instances have no embeddings yet")
    ids = [inst.instance_id for inst in instances]
    matrix = np.stack([e for e in embeddings if e is not None]).astype(np.float32)
    image = dict(zip(ids, (matrix @ np.asarray(image_query, np.float32)).tolist(), strict=True))

    rankings = [rank_by_similarity(ids, [image[i] for i in ids])]
    caption: dict[int, float] = {}
    if caption_query is not None:
        query = np.asarray(caption_query, np.float32)
        caption = {
            inst.instance_id: float(inst.caption_embedding @ query)
            for inst in instances
            if inst.caption_embedding is not None
        }
        if caption:
            rankings.append(rank_by_similarity(list(caption), list(caption.values())))
    fused = rrf_scores(rankings, rrf_k)
    best = len(rankings) / (rrf_k + 1)
    order = sorted(ids, key=lambda i: (-fused[i], -image[i], i))
    return [Match(i, fused[i] / best, image[i], caption.get(i)) for i in order[:top_k]]
