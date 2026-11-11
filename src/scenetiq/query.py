"""Open-vocabulary Query over the Instances of a Reconstruction (ADR 0011).

A Query is compared with every Instance's image embedding and, for every Instance with a
Caption, with its best-matching Description. Each channel's similarities are standardized
over the Reconstruction's Instances and averaged into one fused value. The Matches are the
Instances whose fused value is close enough to the best one, at most K of them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from scenetiq.domain import Aspect, FloatArray, Instance

QUERY_LOGGER = "scenetiq.query"
"""Logger of Query-time notes; ``colab_logging.json`` lets it through."""


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
        score: Fused value as a fraction of the best Instance's, so the best Match scores
            1, or for a Query with a Relation the score of its assignment (ADR 0009);
            relative to this Query only.
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
    embeddings: Mapping[Aspect, FloatArray],
    query: FloatArray,
    aspects: Sequence[Aspect] = (),
) -> tuple[float, Aspect] | None:
    """The Description most similar to a Query, among the Aspects the Query speaks about.

    Args:
        embeddings: L2-normalised embedding of each Description an Instance has, by Aspect.
        query: L2-normalised Query embedding in the Caption space.
        aspects: The Aspects the Query speaks about; empty means every Aspect. When the
            Instance has no Description of any of them, all its Descriptions are compared.

    Returns:
        The best cosine similarity and its Aspect, ties going to the earlier Aspect, or
        ``None`` when there are no Descriptions.
    """
    asked = [a for a in Aspect if a in aspects and a in embeddings]
    best: tuple[float, Aspect] | None = None
    for aspect in asked or [a for a in Aspect if a in embeddings]:
        similarity = float(np.asarray(embeddings[aspect], np.float32) @ query)
        if best is None or similarity > best[0]:
            best = (similarity, aspect)
    return best


def standardize(values: Sequence[float]) -> list[float]:
    """Each value as its distance from the mean, in population standard deviations.

    Returns:
        One standardized value per input, all ``0.0`` when the values do not differ.
    """
    if not len(values):
        return []
    array = np.asarray(values, dtype=np.float64)
    deviation = float(array.std())
    if deviation < 1e-12:
        return [0.0] * len(array)
    return ((array - array.mean()) / deviation).tolist()


def run_query(
    instances: list[Instance],
    image_query: FloatArray,
    top_k: int,
    score_ratio: float | None,
    caption_query: FloatArray | None = None,
    aspects: Sequence[Aspect] = (),
) -> list[Match]:
    """Rank every Instance against a Query and return its Matches.

    Each channel's similarities are standardized over the Instances that have it, so the
    two channels are on one scale and keep how far an Instance stands out. An Instance's
    fused value is the mean of its image value and its Caption value; one without a Caption
    takes 0 on the Caption channel, the average of the captioned ones. A Match's score is
    its fused value divided by the best one.

    Args:
        instances: Instances with L2-normalised image embeddings.
        image_query: L2-normalised embedding of the Query text in the image space.
        top_k: How many Matches to return at most.
        score_ratio: An Instance is a Match only when its score is at least this fraction
            of the best one's, which is 1; ``None`` keeps every Instance, for a ranking.
        caption_query: L2-normalised embedding of the Query text in the Caption space;
            the Caption channel, by each Instance's best Description, is skipped when
            omitted.
        aspects: The Aspects the Query speaks about (see :func:`best_description`).

    Returns:
        Up to ``top_k`` Matches, best first, ties broken by image similarity then
        Instance id; never empty for a Reconstruction with Instances, since the best
        Instance always scores 1. When no Instance stands out, every score is 1. A
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
    fused = dict(zip(ids, standardize([image[i] for i in ids]), strict=True))

    caption: dict[int, tuple[float, Aspect]] = {}
    if caption_query is not None:
        query = np.asarray(caption_query, np.float32)
        for inst in instances:
            found = best_description(inst.caption_embeddings, query, aspects)
            if found is not None:
                caption[inst.instance_id] = found
    if caption:
        standardized = dict(
            zip(caption, standardize([c[0] for c in caption.values()]), strict=True)
        )
        fused = {i: (fused[i] + standardized.get(i, 0.0)) / 2 for i in ids}

    best = max(fused.values())
    order = sorted(ids, key=lambda i: (-fused[i], -image[i], i))
    matches = []
    for i in order:
        score = fused[i] / best if best > 1e-12 else 1.0
        if len(matches) == top_k or (score_ratio is not None and score < score_ratio):
            break
        caption_score, caption_aspect = caption.get(i, (None, None))
        matches.append(Match(i, score, image[i], caption_score, caption_aspect))
    return matches
