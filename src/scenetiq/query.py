import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from scenetiq.domain import Aspect, FloatArray, Instance

QUERY_LOGGER = "scenetiq.query"


class MissingEmbeddingsError(ValueError): ...


@dataclass(frozen=True)
class Anchor:
    instance_id: int
    phrase: str


@dataclass(frozen=True)
class Match:
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
    asked = [a for a in Aspect if a in aspects and a in embeddings]
    best: tuple[float, Aspect] | None = None
    for aspect in asked or [a for a in Aspect if a in embeddings]:
        similarity = float(np.asarray(embeddings[aspect], np.float32) @ query)
        if best is None or similarity > best[0]:
            best = (similarity, aspect)
    return best


def standardize(values: Sequence[float]) -> list[float]:
    if not len(values):
        return []
    array = np.asarray(values, dtype=np.float64)
    deviation = float(array.std())
    if deviation < 1e-12:
        return [0.0] * len(array)
    return ((array - array.mean()) / deviation).tolist()


def _tokenize_label(text: str) -> set[str]:
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
