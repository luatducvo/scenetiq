"""Open-vocabulary Query over the Instances of a Reconstruction (ADR 0011, ADR 0012).

A Query is compared with every Instance's image embedding and, for every Instance with a
Caption, with its best-matching Description. Semantic Labels (Caption Name, Synonyms, and
Descriptions) filter out unrelated Instances, and a negative Query with no semantic matches
returns an empty list (ADR 0012). Each channel's similarities are standardized over the
Reconstruction's Instances and averaged into one fused value. The Matches are the Instances
whose fused value is close enough to the best one among semantic matches.
"""

import re
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


def _tokenize_label(text: str) -> set[str]:
    """Extract lowercase significant words and stems from a label or query phrase."""
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    stopwords = {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "and",
        "or",
        "is",
        "it",
        "this",
        "that",
        "thing",
        "object",
        "kind",
        "type",
        "some",
    }
    words = {w for w in tokens if w not in stopwords and len(w) > 1}
    stems = set(words)
    for w in words:
        if w.endswith("es") and len(w) > 3:
            stems.add(w[:-2])
        elif w.endswith("s") and len(w) > 2:
            stems.add(w[:-1])
    return stems


def is_semantic_match(
    phrase: str,
    instance: Instance,
    caption_score: float | None = None,
    aspects: Sequence[Aspect] = (),
    min_confidence: float = 0.35,
    image_score: float | None = None,
) -> bool:
    """Whether an Instance matches a query phrase semantically (ADR 0012).

    An Instance is a semantic match when its Caption's Name, Synonyms, or relevant Aspect
    Descriptions share key words with ``phrase``, or when its Caption similarity or image
    similarity is high enough.

    Args:
        phrase: Short English phrase naming or describing the wanted object.
        instance: The Instance with its Caption and embeddings.
        caption_score: Cosine similarity of the query to the Instance's best Description.
        aspects: The Aspects the Query speaks about.
        min_confidence: Minimum similarity required when no lexical match exists.
        image_score: Cosine similarity of the query to the Instance's image embedding.

    Returns:
        ``True`` if the Instance is semantically relevant to the query phrase.
    """
    words = _tokenize_label(phrase)
    if not words:
        return True
    if instance.caption is not None:
        name_words = _tokenize_label(instance.caption.name)
        synonym_words = (
            set().union(*(_tokenize_label(s) for s in instance.caption.synonyms))
            if instance.caption.synonyms
            else set()
        )
        if words & (name_words | synonym_words):
            return True
        relevant_aspects = aspects if aspects else Aspect
        for a in relevant_aspects:
            desc = instance.caption.description(a)
            if desc and (words & _tokenize_label(desc)):
                return True
        return bool(caption_score is not None and caption_score >= min_confidence)
    return bool(image_score is not None and image_score >= min_confidence)


def run_query(
    instances: list[Instance],
    image_query: FloatArray,
    top_k: int | None,
    score_ratio: float | None,
    caption_query: FloatArray | None = None,
    aspects: Sequence[Aspect] = (),
    phrase: str | None = None,
    negative_query_threshold: float = 0.35,
) -> list[Match]:
    """Rank every Instance against a Query and return its Matches (ADR 0012).

    Each channel's similarities are standardized over the Instances that have it, so the
    two channels are on one scale and keep how far an Instance stands out. An Instance's
    fused value is the mean of its image value and its Caption value; one without a Caption
    takes 0 on the Caption channel, the average of the captioned ones. A Match's score is
    its fused value divided by the best one among semantic matches.

    When ``phrase`` is given, Instances are filtered by Semantic Label matching: unrelated
    Instances are excluded, and if no Instance matches, the query is treated as a negative
    Query and returns an empty list.

    Args:
        instances: Instances with L2-normalised image embeddings.
        image_query: L2-normalised embedding of the Query text in the image space.
        top_k: How many Matches to return at most; ``None`` returns all Matches passing
            the score ratio.
        score_ratio: An Instance is a Match only when its score is at least this fraction
            of the best one's, which is 1; ``None`` keeps every Instance, for a ranking.
        caption_query: L2-normalised embedding of the Query text in the Caption space;
            the Caption channel, by each Instance's best Description, is skipped when
            omitted.
        aspects: The Aspects the Query speaks about (see :func:`best_description`).
        phrase: English phrase of the wanted object; when supplied, enables Semantic Label
            filtering and negative Query detection (ADR 0012).
        negative_query_threshold: Minimum confidence required for semantic matching when no
            lexical overlap exists.

    Returns:
        Matches, best first, ties broken by image similarity then Instance id. Empty when
        the Reconstruction has no Instances or when no Instance matches a negative Query.

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

    order = sorted(ids, key=lambda i: (-fused[i], -image[i], i))

    has_captions = any(inst.caption is not None for inst in instances)
    if phrase is not None and has_captions:
        inst_by_id = {inst.instance_id: inst for inst in instances}
        matching_ids = {
            i
            for i in ids
            if is_semantic_match(
                phrase,
                inst_by_id[i],
                caption.get(i, (None, None))[0],
                aspects,
                negative_query_threshold,
                image.get(i),
            )
        }
        if not matching_ids:
            return []
        candidates = [i for i in order if i in matching_ids]
    else:
        candidates = order

    best = max(fused[i] for i in candidates)
    matches = []
    for i in candidates:
        score = fused[i] / best if best > 1e-12 else 1.0
        if (top_k is not None and len(matches) == top_k) or (
            score_ratio is not None and score < score_ratio
        ):
            break
        caption_score, caption_aspect = caption.get(i, (None, None))
        matches.append(Match(i, score, image[i], caption_score, caption_aspect))
    return matches
