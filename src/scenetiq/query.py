"""Open-vocabulary Query over the Instances of a Reconstruction (ADR 0008).

A Query is ranked against every Instance's image embedding and, for every Instance with a
Caption, against its best-matching Description. The rankings are fused with Reciprocal Rank
Fusion and the top K Instances are the Matches.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from scenetiq.domain import Aspect, FloatArray, Instance


class MissingEmbeddingsError(ValueError):
    """Raised when a Query targets Instances that have no embeddings yet."""


@dataclass(frozen=True)
class Anchor:
    """An Instance a Match was measured against, for a Query with a Relation.

    Attributes:
        instance_id: The Instance standing for the Anchor object.
        phrase: The phrase of the object it stands for.
    """

    instance_id: int
    phrase: str


@dataclass(frozen=True)
class Match:
    """An Instance returned for a Query.

    Attributes:
        instance_id: The Matched Instance.
        score: Fused rank score normalised to [0, 1], or for a Query with a Relation the
            score of its assignment (ADR 0009); relative to this Query only.
        image_score: Cosine similarity of the Query to the Instance's image embedding.
        caption_score: Cosine similarity of the Query to the Instance's best-matching
            Description, or ``None`` when it has no Caption.
        caption_aspect: The Aspect of that Description, or ``None`` without a Caption.
        anchors: The Anchors of its assignment, empty for a Query without a Relation.
    """

    instance_id: int
    score: float
    image_score: float
    caption_score: float | None = None
    caption_aspect: Aspect | None = None
    anchors: tuple[Anchor, ...] = ()


def best_description(
    embeddings: Mapping[Aspect, FloatArray], query: FloatArray
) -> tuple[float, Aspect] | None:
    """The Description most similar to a Query.

    Args:
        embeddings: L2-normalised embedding of each Description an Instance has, by Aspect.
        query: L2-normalised Query embedding in the Caption space.

    Returns:
        The best cosine similarity and its Aspect, ties going to the earlier Aspect, or
        ``None`` when there are no Descriptions.
    """
    best: tuple[float, Aspect] | None = None
    for aspect in Aspect:
        if aspect in embeddings:
            similarity = float(np.asarray(embeddings[aspect], np.float32) @ query)
            if best is None or similarity > best[0]:
                best = (similarity, aspect)
    return best


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
            the Caption ranking, by each Instance's best Description, is skipped when
            omitted.

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
    caption: dict[int, tuple[float, Aspect]] = {}
    if caption_query is not None:
        query = np.asarray(caption_query, np.float32)
        for inst in instances:
            found = best_description(inst.caption_embeddings, query)
            if found is not None:
                caption[inst.instance_id] = found
        if caption:
            rankings.append(rank_by_similarity(list(caption), [c[0] for c in caption.values()]))
    fused = rrf_scores(rankings, rrf_k)
    best = len(rankings) / (rrf_k + 1)
    order = sorted(ids, key=lambda i: (-fused[i], -image[i], i))
    matches = []
    for i in order[:top_k]:
        caption_score, caption_aspect = caption.get(i, (None, None))
        matches.append(Match(i, fused[i] / best, image[i], caption_score, caption_aspect))
    return matches
