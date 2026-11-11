"""Spatial reasoning: Relations scored on stored geometry and solved over candidates (ADR 0009).

Each object of a parsed Query is ranked like a Query of its own, by its English phrase. The
first Instances of that ranking are shortlisted, and those within the candidate margin of the
best are its candidates. The solver scores every assignment of one candidate to each object,
with no Instance used twice, as the product of its constraints' scores. Every Relation is a
pure function of axis-aligned bounding boxes and centroids in the Reconstruction's frame, where
up is -Y (ADR 0007).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from scenetiq.config import SpatialOptions
from scenetiq.domain import Constraint, Instance, ParsedQuery, Relation
from scenetiq.query import Anchor, Match


class UnsupportedRelationError(NotImplementedError):
    """Raised when a parsed Query uses a Relation that has no scoring function yet."""


def scene_extent(instances: Sequence[Instance]) -> float:
    """The Scene extent: the diagonal of the box enclosing every Instance's bounding box.

    Returns:
        The diagonal in metres, ``0.0`` without Instances.
    """
    if not instances:
        return 0.0
    low = np.min([inst.bbox_min for inst in instances], axis=0)
    high = np.max([inst.bbox_max for inst in instances], axis=0)
    return float(np.linalg.norm(high - low))


def box_gap(a: Instance, b: Instance) -> float:
    """The distance between two bounding boxes, ``0.0`` when they touch or overlap."""
    per_axis = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    return float(np.linalg.norm(per_axis))


def falloff_score(distance: float, scale: float, falloff: float) -> float:
    """The exponential falloff ``exp(-distance / (falloff * scale))``.

    Args:
        distance: A non-negative distance in metres.
        scale: The length the distance is measured against, in metres.
        falloff: The fraction of ``scale`` at which the score has dropped to ``1/e``.

    Returns:
        A score in [0, 1]: 1 at distance zero, whatever the scale.
    """
    if distance <= 0.0:
        return 1.0
    return math.exp(-distance / (falloff * scale))


def near(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    """How near ``subject`` is to ``anchor``: the falloff of the gap between their boxes.

    The gap, not the distance between centroids, so two large objects that touch are fully
    near.

    Args:
        subject: The Instance the Relation is about.
        anchor: The Instance it is measured against.
        extent: The Scene extent, in metres (see :func:`scene_extent`).
        falloff: The fraction of the extent at which the score drops to ``1/e``.

    Returns:
        1 when the boxes touch or overlap, falling towards 0 as they move apart.
    """
    return falloff_score(box_gap(subject, anchor), extent, falloff)


def far_from(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    """How far ``subject`` is from ``anchor``: one minus :func:`near`.

    Returns:
        0 when the boxes touch or overlap, rising towards 1 as they move apart.
    """
    return 1.0 - near(subject, anchor, extent, falloff)


_UP = 1
"""Index of the vertical axis. Up is -Y (ADR 0007): a smaller y is higher."""

_FOOTPRINT = [0, 2]
"""Indices of the horizontal axes: the footprint of a box is its X-Z rectangle."""


def _footprint_gap(a: Instance, b: Instance) -> float:
    lo_a, hi_a = a.bbox_min[_FOOTPRINT], a.bbox_max[_FOOTPRINT]
    lo_b, hi_b = b.bbox_min[_FOOTPRINT], b.bbox_max[_FOOTPRINT]
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo_a - hi_b, lo_b - hi_a))))


def _footprint_share(a: Instance, b: Instance) -> float:
    """The share of the smaller of two footprints that the two footprints share.

    A footprint without area (a box flat along X or Z) has nothing to share: it counts as
    fully shared when it touches the other footprint, and as not shared otherwise.

    Returns:
        A share in [0, 1]: 1 when the smaller footprint lies inside the larger one.
    """
    lo_a, hi_a = a.bbox_min[_FOOTPRINT], a.bbox_max[_FOOTPRINT]
    lo_b, hi_b = b.bbox_min[_FOOTPRINT], b.bbox_max[_FOOTPRINT]
    smaller = min(float(np.prod(hi_a - lo_a)), float(np.prod(hi_b - lo_b)))
    if smaller <= 0.0:
        return 1.0 if _footprint_gap(a, b) == 0.0 else 0.0
    shared = np.maximum(0.0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))
    return min(1.0, float(np.prod(shared)) / smaller)


def above(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    """How far ``subject`` is above ``anchor``, contact or not.

    Zero unless the subject's centroid is higher (a smaller y, ADR 0007) than the Anchor's;
    otherwise the falloff of the horizontal gap between the two footprints.

    Args:
        subject: The Instance the Relation is about.
        anchor: The Instance it is measured against.
        extent: The Scene extent, in metres (see :func:`scene_extent`).
        falloff: The fraction of the extent at which the score drops to ``1/e``.

    Returns:
        1 when the subject is higher and the footprints touch or overlap, falling towards 0
        as the footprints move apart; 0 when the subject is not higher.
    """
    if subject.centroid[_UP] >= anchor.centroid[_UP]:
        return 0.0
    return falloff_score(_footprint_gap(subject, anchor), extent, falloff)


def below(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    """How far ``subject`` is below ``anchor``: :func:`above` with the two swapped."""
    return above(anchor, subject, extent, falloff)


def on(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    """How well ``subject`` rests on ``anchor``.

    :func:`above`, times the share of the smaller footprint that the two footprints share,
    times the falloff of the vertical distance between the subject's bottom (its maximum y)
    and the Anchor's top (its minimum y). The distance counts both ways, so a subject sunk
    into the Anchor loses as much as one hovering over it.

    Args:
        subject: The Instance the Relation is about.
        anchor: The Instance it is measured against.
        extent: The Scene extent, in metres (see :func:`scene_extent`).
        falloff: The fraction of the extent at which the score drops to ``1/e``.

    Returns:
        1 when the subject is higher, its bottom meets the Anchor's top and the smaller
        footprint lies inside the other; towards 0 otherwise.
    """
    gap = abs(float(subject.bbox_max[_UP]) - float(anchor.bbox_min[_UP]))
    return (
        above(subject, anchor, extent, falloff)
        * _footprint_share(subject, anchor)
        * falloff_score(gap, extent, falloff)
    )


def under(subject: Instance, anchor: Instance, extent: float, falloff: float) -> float:
    """How well ``subject`` sits under ``anchor``: :func:`on` with the two swapped."""
    return on(anchor, subject, extent, falloff)


def between(subject: Instance, first: Instance, second: Instance, falloff: float) -> float:
    """How well ``subject`` lies between two Anchors.

    The falloff of the distance from the subject's centroid to the segment joining the
    Anchors' centroids, scaled by that segment's length instead of the Scene extent.

    Args:
        subject: The Instance the Relation is about.
        first: One Anchor.
        second: The other Anchor; their order does not matter.
        falloff: The fraction of the segment's length at which the score drops to ``1/e``.

    Returns:
        1 on the segment, falling towards 0 away from it; 0 when the Anchors' centroids
        coincide, since nothing lies between one point.
    """
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
    """Where ``subject`` stands among ``candidates`` once they are ordered by ``measure``.

    The distinct values of the measure are ordered; the first scores 1, the last 0, and the
    ones between are evenly spaced. Tied candidates (exactly equal values) therefore share a
    score, and a tie takes one step, not one per candidate: values 27, 8, 8, 1, highest
    first, score 1, 1/2, 1/2, 0. Every candidate scores 1 when all values are equal, so a
    single candidate scores 1.

    Args:
        subject: The candidate to score. It counts among the candidates even if missing.
        candidates: The candidates it is ordered against, the Anchors of the assignment
            included when they are among them.
        measure: The value candidates are ordered by.
        highest_first: Whether the highest value comes first.

    Returns:
        A score in [0, 1].
    """
    value = measure(subject)
    values = sorted({measure(c) for c in candidates} | {value}, reverse=highest_first)
    if len(values) == 1:
        return 1.0
    return 1.0 - values.index(value) / (len(values) - 1)


def box_volume(instance: Instance) -> float:
    """The volume of an Instance's bounding box, in cubic metres."""
    return float(np.prod(instance.bbox_max - instance.bbox_min))


def vertical_extent(instance: Instance) -> float:
    """The height of an Instance's bounding box from bottom to top, in metres."""
    return float(instance.bbox_max[_UP] - instance.bbox_min[_UP])


def bottom_height(instance: Instance) -> float:
    """How high the bottom of an Instance's bounding box is: its maximum y, negated.

    Up is -Y (ADR 0007), so the bottom is the maximum y, and a larger y is lower.
    """
    return -float(instance.bbox_max[_UP])


def nearest_to(subject: Instance, anchor: Instance, candidates: Sequence[Instance]) -> float:
    """Score ``subject`` as the candidate nearest to ``anchor``.

    The candidates are ordered by the gap between their box and the Anchor's, smallest
    first (see :func:`ordering_score`). The order is taken for this Anchor, so a candidate
    scores differently against another one. An Anchor that is itself among the candidates
    (two objects with the same phrase) competes too, with a gap of 0.

    Args:
        subject: The candidate standing for the constraint's subject.
        anchor: The candidate standing for its Anchor.
        candidates: Every candidate of the subject object.

    Returns:
        1 for the nearest candidate, 0 for the farthest.
    """
    return ordering_score(subject, candidates, lambda c: box_gap(c, anchor), highest_first=False)


def farthest_from(subject: Instance, anchor: Instance, candidates: Sequence[Instance]) -> float:
    """Score ``subject`` as the candidate farthest from ``anchor``: :func:`nearest_to` reversed."""
    return ordering_score(subject, candidates, lambda c: box_gap(c, anchor), highest_first=True)


def largest(subject: Instance, candidates: Sequence[Instance]) -> float:
    """Score ``subject`` as the largest of ``candidates``, by bounding-box volume.

    Returns:
        1 for the largest box, 0 for the smallest (see :func:`ordering_score`).
    """
    return ordering_score(subject, candidates, box_volume, highest_first=True)


def smallest(subject: Instance, candidates: Sequence[Instance]) -> float:
    """Score ``subject`` as the smallest of ``candidates``: :func:`largest` reversed."""
    return ordering_score(subject, candidates, box_volume, highest_first=False)


def tallest(subject: Instance, candidates: Sequence[Instance]) -> float:
    """Score ``subject`` as the tallest of ``candidates``, by the vertical extent of its box.

    Returns:
        1 for the tallest box, 0 for the shortest (see :func:`ordering_score`).
    """
    return ordering_score(subject, candidates, vertical_extent, highest_first=True)


def lowest(subject: Instance, candidates: Sequence[Instance]) -> float:
    """Score ``subject`` as the lowest of ``candidates``, by the height of its box's bottom.

    Up is -Y (ADR 0007): the lowest candidate is the one whose bottom has the largest y.

    Returns:
        1 for the lowest bottom, 0 for the highest (see :func:`ordering_score`).
    """
    return ordering_score(subject, candidates, bottom_height, highest_first=False)


@dataclass(frozen=True)
class RelationContext:
    """What a Relation's scoring function sees of one assignment.

    Attributes:
        subject: The candidate standing for the constraint's subject.
        anchors: The candidates standing for its Anchor objects, in the constraint's order.
        subject_candidates: Every candidate of the subject object, in the order of its
            phrase ranking, for Relations that compare a candidate with the others.
        extent: The Scene extent, in metres.
        falloff: The falloff fraction from the settings.
    """

    subject: Instance
    anchors: tuple[Instance, ...]
    subject_candidates: Sequence[Instance]
    extent: float
    falloff: float


RelationScore = Callable[[RelationContext], float]
"""Scores one constraint of one assignment, between 0 and 1."""

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
"""The scoring function of every Relation. A Relation missing here fails loudly."""


@dataclass(frozen=True)
class Assignment:
    """One candidate for each object of a parsed Query, and how well they satisfy it.

    Attributes:
        score: The product of the constraints' scores, between 0 and 1.
        instances: The Instance standing for each object, by object id.
    """

    score: float
    instances: Mapping[str, Instance]


def solve(
    candidates: Mapping[str, Sequence[Instance]],
    constraints: Sequence[Constraint],
    target: str,
    extent: float,
    falloff: float,
) -> list[Assignment]:
    """Rank the candidates of the target object by their best assignment.

    Every assignment of one candidate to each object, with no Instance used twice, is
    scored as the product of its constraints' scores.

    Args:
        candidates: The candidates of every object, by object id, each list in the order
            of the object's phrase ranking.
        constraints: The Relations to satisfy.
        target: Id of the object the user wants.
        extent: The Scene extent, in metres.
        falloff: The falloff fraction of the distance-based Relations.

    Returns:
        The best assignment of each target candidate that has a valid one, best first. Equal
        scores keep the order of the target's candidates; a target candidate's equally good
        assignments keep the one whose other candidates come first in their rankings. Empty
        when no assignment is valid.

    Raises:
        UnsupportedRelationError: If a constraint uses a Relation without a scoring function.
    """
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
    """Drop the shortlisted Instances that fit an object's phrase much worse than the best.

    On each channel, an Instance passes when its similarity is no more than the channel's
    margin below the highest similarity on that channel among ``shortlist``. The channels
    are the cosine of the best-matching Description, among the Instances that have a
    Caption, and the image similarity. An Instance stays when it passes on at least one
    channel, so one without a Caption is judged on the image channel alone. The margin is
    relative: the best Instance of each channel always stays, and there is no absolute
    similarity threshold. It limits which Instances may stand in for an object of a spatial
    Query; it is not a threshold on Matches (ADR 0009).

    Args:
        shortlist: The shortlisted Instances of one object, in the fused order of its
            phrase ranking.
        caption_margin: How far below the best Caption similarity an Instance may be.
        image_margin: How far below the best image similarity an Instance may be.

    Returns:
        The Instances that stay candidates, in their fused order; empty only when
        ``shortlist`` is.
    """
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


PhraseRanking = Callable[[str, int], list[Match]]
"""Ranks every Instance against a phrase and returns the first ``k`` Matches, best first."""


def run_spatial_query(
    instances: Sequence[Instance],
    parsed: ParsedQuery,
    rank_phrase: PhraseRanking,
    top_k: int,
    options: SpatialOptions,
) -> list[Match] | None:
    """Answer a parsed Query that has constraints.

    Each object shortlists the first ``options.candidates_per_object`` Instances of its
    phrase's fused ranking; the target shortlists ``top_k`` when that is larger. The
    candidate margin (:func:`apply_candidate_margin`) then drops the shortlisted Instances
    that fit the phrase much worse than the best, and the rest are the object's candidates.

    Args:
        instances: Every Instance of the Reconstruction.
        parsed: The parsed Query; its constraints are solved.
        rank_phrase: The plain Query ranking, applied to one object's phrase.
        top_k: How many Matches to return at most.
        options: The candidate count, the candidate margins and the falloff fraction.

    Returns:
        Up to ``top_k`` Matches, ordered by assignment score: at most one per candidate of
        the target, so possibly fewer than ``top_k``. A Match's ``score`` is that
        score; its similarities and Aspect are those of the target's phrase, and its Anchors
        are the other objects of its assignment. ``None`` when no assignment is valid.

    Raises:
        UnsupportedRelationError: If a constraint uses a Relation without a scoring function.
    """
    by_id = {inst.instance_id: inst for inst in instances}
    wanted = {
        o.object_id: max(options.candidates_per_object, top_k)
        if o.object_id == parsed.target
        else options.candidates_per_object
        for o in parsed.objects
    }
    depth: dict[str, int] = {}
    for o in parsed.objects:
        depth[o.phrase] = max(depth.get(o.phrase, 0), wanted[o.object_id])
    ranked = {phrase: rank_phrase(phrase, k) for phrase, k in depth.items()}
    candidates = {
        o.object_id: apply_candidate_margin(
            ranked[o.phrase][: wanted[o.object_id]], options.caption_margin, options.image_margin
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
