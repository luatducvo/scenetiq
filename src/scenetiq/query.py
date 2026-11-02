"""Open-vocabulary Query over the Instances of a Reconstruction."""

from dataclasses import dataclass

import numpy as np

from scenetiq.domain import FloatArray, Instance


class MissingEmbeddingsError(ValueError):
    """Raised when a Query targets Instances that have no embeddings yet."""


@dataclass(frozen=True)
class Match:
    """An Instance returned for a Query, with its relevance score."""

    instance_id: int
    score: float


@dataclass(frozen=True)
class QueryResult:
    """The answer to a Query.

    Attributes:
        matches: Matches at or above the threshold, best first; or, when
            ``low_confidence`` is true, the single closest Instance.
        low_confidence: Whether no Instance cleared the threshold.
    """

    matches: list[Match]
    low_confidence: bool


def run_query(
    instances: list[Instance], query_embedding: FloatArray, threshold: float
) -> QueryResult:
    """Score every Instance against ``query_embedding`` by cosine similarity.

    Args:
        instances: Instances with L2-normalised embeddings.
        query_embedding: L2-normalised embedding of the Query text.
        threshold: Minimum score for a Match.

    Returns:
        Every Instance scoring at least ``threshold``, best first. If there is
        none, the single best Instance flagged as a Low-confidence Match. A
        Reconstruction without Instances yields no Matches.

    Raises:
        MissingEmbeddingsError: If any Instance has no embedding.
    """
    if not instances:
        return QueryResult(matches=[], low_confidence=False)
    embeddings = [inst.embedding for inst in instances]
    if any(e is None for e in embeddings):
        raise MissingEmbeddingsError("Scene Instances have no embeddings yet")
    matrix = np.stack([e for e in embeddings if e is not None]).astype(np.float32)
    scores = matrix @ np.asarray(query_embedding, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    ranked = [Match(instances[i].instance_id, float(scores[i])) for i in order]
    confident = [m for m in ranked if m.score >= threshold]
    if confident:
        return QueryResult(matches=confident, low_confidence=False)
    return QueryResult(matches=ranked[:1], low_confidence=True)
