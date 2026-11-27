from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from scenetiq.config import SpatialOptions
from scenetiq.domain import (
    Aspect,
    Constraint,
    Instance,
    ParsedQuery,
    QueryObject,
    Relation,
)
from scenetiq.query.ranking import Anchor, Match


class UnsupportedRelationError(NotImplementedError): ...


def scene_extent(instances: Sequence[Instance]) -> float:
    if not instances:
        return 0.0
    low = np.min([inst.bbox_min for inst in instances], axis=0)
    high = np.max([inst.bbox_max for inst in instances], axis=0)
    return float(np.linalg.norm(high - low))


def box_gap(a: Instance, b: Instance) -> float:
    per_axis = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(per_axis))


def falloff_score(distance: float, scale: float, falloff: float) -> float:
    if distance <= 0.0:
        return 1.0
    return math.exp(-distance / (falloff * scale))


def near(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    return falloff_score(box_gap(subject, anchor), extent, falloff)


def far_from(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    return 1.0 - near(subject, anchor, extent, falloff)


_UP = 1

_FOOTPRINT = [0, 2]


def _footprint_gap(a: Instance, b: Instance) -> float:
    lo_a, hi_a = a.bbox_min[_FOOTPRINT], a.bbox_max[_FOOTPRINT]
    lo_b, hi_b = b.bbox_min[_FOOTPRINT], b.bbox_max[_FOOTPRINT]
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo_a - hi_b, lo_b - hi_a))))


def _footprint_share(a: Instance, b: Instance) -> float:
    lo_a, hi_a = a.bbox_min[_FOOTPRINT], a.bbox_max[_FOOTPRINT]
    lo_b, hi_b = b.bbox_min[_FOOTPRINT], b.bbox_max[_FOOTPRINT]
    smaller = min(float(np.prod(hi_a - lo_a)), float(np.prod(hi_b - lo_b)))
    if smaller <= 0.0:
        return 1.0 if _footprint_gap(a, b) == 0.0 else 0.0
    shared = np.maximum(0.0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))
    return min(1.0, float(np.prod(shared)) / smaller)


def above(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    if subject.centroid[_UP] >= anchor.centroid[_UP]:
        return 0.0
    return falloff_score(_footprint_gap(subject, anchor), extent, falloff)


def below(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    return above(anchor, subject, extent, falloff)


def on(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    gap = abs(float(subject.bbox_max[_UP]) - float(anchor.bbox_min[_UP]))
    return (
        above(subject, anchor, extent, falloff)
        * _footprint_share(subject, anchor)
        * falloff_score(gap, extent, falloff)
    )


def under(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    return on(anchor, subject, extent, falloff)


def between(subject: Instance, first: Instance, second: Instance, falloff: float) -> float:
    start = np.asarray(first.centroid, dtype=np.float64)
    segment = np.asarray(second.centroid, dtype=np.float64) - start
    length = float(np.linalg.norm(segment))
    if length <= 0.0:
        return 0.0
    point = np.asarray(subject.centroid, dtype=np.float64)
    t = min(1.0, max(0.0, float(np.dot(point - start, segment)) / length**2))
    distance = float(np.linalg.norm(point - (start + t * segment)))
    return falloff_score(distance, length, falloff)


def ordering_score(
    subject: Instance,
    candidates: Sequence[Instance],
    measure: Callable[[Instance], float],
    *,
    highest_first: bool,
) -> float:
    value = measure(subject)
    values = sorted({measure(c) for c in candidates} | {value}, reverse=highest_first)
    if len(values) == 1:
        return 1.0
    return 1.0 - values.index(value) / (len(values) - 1)


def box_volume(instance: Instance) -> float:
    return float(np.prod(instance.bbox_max - instance.bbox_min))


def vertical_extent(instance: Instance) -> float:
    return float(instance.bbox_max[_UP] - instance.bbox_min[_UP])


def bottom_height(instance: Instance) -> float:
    return -float(instance.bbox_max[_UP])


def nearest_to(subject: Instance, anchor: Instance, candidates: Sequence[Instance]) -> float:
    return ordering_score(subject, candidates, lambda c: box_gap(c, anchor), highest_first=False)


def farthest_from(subject: Instance, anchor: Instance, candidates: Sequence[Instance]) -> float:
    return ordering_score(subject, candidates, lambda c: box_gap(c, anchor), highest_first=True)


def largest(subject: Instance, candidates: Sequence[Instance]) -> float:
    return ordering_score(subject, candidates, box_volume, highest_first=True)


def smallest(subject: Instance, candidates: Sequence[Instance]) -> float:
    return ordering_score(subject, candidates, box_volume, highest_first=False)


def tallest(subject: Instance, candidates: Sequence[Instance]) -> float:
    return ordering_score(subject, candidates, vertical_extent, highest_first=True)


def lowest(subject: Instance, candidates: Sequence[Instance]) -> float:
    return ordering_score(subject, candidates, bottom_height, highest_first=False)


@dataclass(frozen=True)
class RelationContext:
    subject: Instance
    anchors: tuple[Instance, ...]
    subject_candidates: Sequence[Instance]
    extent: float
    falloff: float


RelationScore = Callable[[RelationContext], float]

RELATION_SCORES: dict[Relation, RelationScore] = {
    Relation.NEAR: lambda c: near(c.subject, c.anchors[0], c.extent, c.falloff),
    Relation.FAR_FROM: lambda c: far_from(c.subject, c.anchors[0], c.extent, c.falloff),
    Relation.ABOVE: lambda c: above(c.subject, c.anchors[0], c.extent, c.falloff),
    Relation.BELOW: lambda c: below(c.subject, c.anchors[0], c.extent, c.falloff),
    Relation.ON: lambda c: on(c.subject, c.anchors[0], c.extent, c.falloff),
    Relation.UNDER: lambda c: under(c.subject, c.anchors[0], c.extent, c.falloff),
    Relation.BETWEEN: lambda c: between(c.subject, c.anchors[0], c.anchors[1], c.falloff),
    Relation.NEAREST_TO: lambda c: nearest_to(c.subject, c.anchors[0], c.subject_candidates),
    Relation.FARTHEST_FROM: lambda c: farthest_from(c.subject, c.anchors[0], c.subject_candidates),
    Relation.LARGEST: lambda c: largest(c.subject, c.subject_candidates),
    Relation.SMALLEST: lambda c: smallest(c.subject, c.subject_candidates),
    Relation.TALLEST: lambda c: tallest(c.subject, c.subject_candidates),
    Relation.LOWEST: lambda c: lowest(c.subject, c.subject_candidates),
}


@dataclass(frozen=True)
class Assignment:
    score: float
    instances: Mapping[str, Instance]


def solve(
    candidates: Mapping[str, Sequence[Instance]],
    constraints: Sequence[Constraint],
    target: str,
    extent: float,
    falloff: float,
) -> list[Assignment]:
    missing = sorted({c.relation.value for c in constraints if c.relation not in RELATION_SCORES})
    if missing:
        raise UnsupportedRelationError(f"Relations without a scoring function: {missing}")
    ids = list(candidates)
    best: dict[int, Assignment] = {}
    for combo in itertools.product(*(candidates[i] for i in ids)):
        if len({inst.instance_id for inst in combo}) < len(combo):
            continue
        chosen = dict(zip(ids, combo, strict=True))
        score = 1.0
        for c in constraints:
            score *= RELATION_SCORES[c.relation](
                RelationContext(
                    subject=chosen[c.subject],
                    anchors=tuple(chosen[a] for a in c.anchors),
                    subject_candidates=candidates[c.subject],
                    extent=extent,
                    falloff=falloff,
                )
            )
        key = chosen[target].instance_id
        if key not in best or score > best[key].score:
            best[key] = Assignment(score, chosen)
    position = {inst.instance_id: p for p, inst in enumerate(candidates[target])}
    return sorted(best.values(), key=lambda a: (-a.score, position[_id(a, target)], _id(a, target)))


def _id(assignment: Assignment, object_id: str) -> int:
    return assignment.instances[object_id].instance_id


def apply_candidate_margin(
    shortlist: Sequence[Match], caption_margin: float, image_margin: float
) -> list[Match]:
    if not shortlist:
        return []
    image_floor = max(m.image_score for m in shortlist) - image_margin
    captions = [m.caption_score for m in shortlist if m.caption_score is not None]
    caption_floor = max(captions) - caption_margin if captions else None

    def passes(m: Match) -> bool:
        if m.image_score >= image_floor:
            return True
        caption = m.caption_score
        return caption_floor is not None and caption is not None and caption >= caption_floor

    return [m for m in shortlist if passes(m)]


PhraseRanking = Callable[[QueryObject, int], list[Match]]


def run_spatial_query(
    instances: Sequence[Instance],
    parsed: ParsedQuery,
    rank_phrase: PhraseRanking,
    top_k: int,
    options: SpatialOptions,
) -> list[Match] | None:
    by_id = {inst.instance_id: inst for inst in instances}
    wanted = {
        o.object_id: max(options.candidates_per_object, top_k)
        if o.object_id == parsed.target
        else options.candidates_per_object
        for o in parsed.objects
    }
    # Objects that read the same are ranked once, to the deepest shortlist any of them needs.
    first: dict[tuple[str, tuple[Aspect, ...]], QueryObject] = {}
    depth: dict[tuple[str, tuple[Aspect, ...]], int] = {}
    for o in parsed.objects:
        key = (o.phrase, o.aspects)
        first.setdefault(key, o)
        depth[key] = max(depth.get(key, 0), wanted[o.object_id])
    ranked = {key: rank_phrase(o, depth[key]) for key, o in first.items()}
    candidates = {
        o.object_id: apply_candidate_margin(
            ranked[o.phrase, o.aspects][: wanted[o.object_id]],
            options.caption_margin,
            options.image_margin,
        )
        for o in parsed.objects
    }
    assignments = solve(
        {i: [by_id[m.instance_id] for m in ms] for i, ms in candidates.items()},
        parsed.constraints,
        parsed.target,
        scene_extent(instances),
        options.falloff,
    )
    if not assignments:
        return None
    phrase_match = {m.instance_id: m for m in candidates[parsed.target]}
    matches = []
    for a in assignments[:top_k]:
        m = phrase_match[_id(a, parsed.target)]
        anchors = tuple(
            Anchor(_id(a, o.object_id), o.phrase)
            for o in parsed.objects
            if o.object_id != parsed.target
        )
        matches.append(
            Match(m.instance_id, a.score, m.image_score, m.caption_score, m.caption_aspect, anchors)
        )
    return matches
