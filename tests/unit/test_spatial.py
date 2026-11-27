import numpy as np
import pytest

from scenetiq.domain import Constraint, Instance, Relation
from scenetiq.spatial import (
    RELATION_SCORES,
    RelationContext,
    UnsupportedRelationError,
    above,
    below,
    between,
    far_from,
    farthest_from,
    largest,
    lowest,
    near,
    nearest_to,
    on,
    scene_extent,
    smallest,
    solve,
    tallest,
    under,
)


def _box(iid: int, low: list[float], high: list[float]) -> Instance:
    lo = np.asarray(low, dtype=np.float32)
    hi = np.asarray(high, dtype=np.float32)
    return Instance(instance_id=iid, centroid=(lo + hi) / 2, bbox_min=lo, bbox_max=hi)


_UNIT = _box(0, [0, 0, 0], [1, 1, 1])


def test_the_scene_extent_is_the_diagonal_of_the_box_around_every_instance():
    # Enclosing box [0,0,0]-[2,3,6]: sqrt(4 + 9 + 36) = 7.
    instances = [_UNIT, _box(1, [1, 2, 5], [2, 3, 6])]
    assert scene_extent(instances) == pytest.approx(7.0)


def test_a_scene_without_instances_has_no_extent():
    assert scene_extent([]) == 0.0


def test_boxes_that_touch_are_fully_near():
    # Faces meet at x = 1: the gap is 0 and exp(0) = 1, however far apart the centroids.
    assert near(_UNIT, _box(1, [1, 0, 0], [5, 1, 1]), extent=10.0, falloff=0.1) == 1.0


def test_boxes_that_overlap_are_fully_near():
    assert near(_UNIT, _box(1, [0.5, 0.5, 0.5], [2, 2, 2]), extent=10.0, falloff=0.1) == 1.0


def test_near_falls_off_with_the_gap_between_boxes():
    # Gap 2 along x, scale 0.1 * 10 = 1: exp(-2) = 0.135335.
    apart = _box(1, [3, 0, 0], [4, 1, 1])
    assert near(_UNIT, apart, extent=10.0, falloff=0.1) == pytest.approx(0.135335, abs=1e-6)


def test_near_measures_the_gap_across_every_axis():
    # Gaps 3 along x and 4 along y: distance 5. Scale 0.5 * 10 = 5: exp(-1) = 0.367879.
    corner = _box(1, [4, 5, 0], [6, 6, 1])
    assert near(_UNIT, corner, extent=10.0, falloff=0.5) == pytest.approx(0.367879, abs=1e-6)


def test_near_is_symmetric():
    apart = _box(1, [3, 0, 0], [4, 1, 1])
    assert near(apart, _UNIT, extent=10.0, falloff=0.1) == pytest.approx(0.135335, abs=1e-6)


def test_boxes_that_touch_are_not_far_from_each_other():
    assert far_from(_UNIT, _box(1, [1, 0, 0], [5, 1, 1]), extent=10.0, falloff=0.1) == 0.0


def test_far_from_grows_with_the_gap_between_boxes():
    # Gap 2 along x, scale 1: 1 - exp(-2) = 1 - 0.135335 = 0.864665.
    apart = _box(1, [3, 0, 0], [4, 1, 1])
    assert far_from(_UNIT, apart, extent=10.0, falloff=0.1) == pytest.approx(0.864665, abs=1e-6)


# Vertical Relations. Up is -Y (ADR 0007): a smaller y is higher, a box's top is its minimum
# y and its bottom its maximum y. Extent 10 and falloff 0.1 give a scale of 1: exp(-d).

_DESK = _box(0, [0, 0, 0], [2, 1, 1])  # top at y = 0, footprint x 0-2, z 0-1


def test_a_box_with_a_smaller_y_over_the_anchor_is_fully_above_it():
    lamp = _box(1, [0.5, -3, 0], [1.5, -2, 1])  # centroid y -2.5 < 0.5; footprints overlap
    assert above(lamp, _DESK, extent=10.0, falloff=0.1) == 1.0


def test_a_box_with_a_larger_y_is_not_above_although_it_would_be_with_up_as_plus_y():
    # The lamp's centroid y is 2.5 > 0.5: lower, under -Y up. With +Y up it would score 1.
    lamp = _box(1, [0.5, 2, 0], [1.5, 3, 1])
    assert above(lamp, _DESK, extent=10.0, falloff=0.1) == 0.0


def test_above_falls_off_with_the_horizontal_gap_between_footprints():
    # Footprint x 4-5 against 0-2: gap 2, z overlaps. The vertical 2 m does not count: exp(-2).
    lamp = _box(1, [4, -3, 0], [5, -2, 1])
    assert above(lamp, _DESK, extent=10.0, falloff=0.1) == pytest.approx(0.135335, abs=1e-6)


def test_above_measures_the_footprint_gap_across_x_and_z():
    # Gaps 3 along x (2 to 5) and 4 along z (1 to 5): 5. Scale 0.5 * 10 = 5: exp(-1).
    lamp = _box(1, [5, -3, 5], [6, -2, 6])
    assert above(lamp, _DESK, extent=10.0, falloff=0.5) == pytest.approx(0.367879, abs=1e-6)


def test_side_by_side_boxes_at_one_height_are_not_above_each_other():
    beside = _box(1, [2, 0, 0], [3, 1, 1])  # touches the desk, both centroids at y 0.5
    assert above(beside, _DESK, extent=10.0, falloff=0.1) == 0.0
    assert above(_DESK, beside, extent=10.0, falloff=0.1) == 0.0


def test_below_is_the_mirror_of_above():
    lamp = _box(1, [4, -3, 0], [5, -2, 1])  # higher than the desk, footprints 2 apart
    assert below(_DESK, lamp, extent=10.0, falloff=0.1) == pytest.approx(0.135335, abs=1e-6)
    assert below(lamp, _DESK, extent=10.0, falloff=0.1) == 0.0


_SHELF = _box(0, [0, 0, 0], [4, 1, 2])  # top at y = 0, footprint x 0-4, z 0-2 (area 8)


def test_a_box_resting_on_the_anchor_is_fully_on_it():
    # Bottom y 0 meets the shelf's top y 0; the 1x1 footprint lies inside the shelf's.
    box = _box(1, [1, -1, 0.5], [2, 0, 1.5])
    assert on(box, _SHELF, extent=10.0, falloff=0.1) == 1.0


def test_a_box_hovering_over_the_anchor_is_less_on_it_than_above_it():
    # Bottom y -2, shelf top y 0: vertical distance 2, exp(-2). Above alone gives 1.
    box = _box(1, [1, -3, 0.5], [2, -2, 1.5])
    assert above(box, _SHELF, extent=10.0, falloff=0.1) == 1.0
    assert on(box, _SHELF, extent=10.0, falloff=0.1) == pytest.approx(0.135335, abs=1e-6)


def test_on_counts_the_share_of_the_smaller_footprint_the_two_share():
    # Footprint x 3.5-4.5, z 0.5-1.5 (area 1); x 3.5-4 lies on the shelf: 0.5 * 1 / 1 = 0.5.
    box = _box(1, [3.5, -1, 0.5], [4.5, 0, 1.5])
    assert on(box, _SHELF, extent=10.0, falloff=0.1) == pytest.approx(0.5)


def test_on_takes_the_share_of_the_smaller_footprint_when_the_anchor_is_smaller():
    # A tray (x 0-4, z 0-4, area 16) on a stool (x 3-5, z 1-2, area 2); they share x 3-4,
    # z 1-2: 1 / 2 = 0.5. The tray's bottom y 0 meets the stool's top y 0.
    tray = _box(1, [0, -0.2, 0], [4, 0, 4])
    stool = _box(2, [3, 0, 1], [5, 1, 2])
    assert on(tray, stool, extent=10.0, falloff=0.1) == pytest.approx(0.5)


def test_a_box_sunk_into_the_anchor_loses_by_how_far_it_sinks():
    # Bottom y 0.5 is 0.5 below the shelf's top: exp(-0.5) = 0.606531. Its centroid y -0.25
    # is still higher than the shelf's 0.5.
    box = _box(1, [1, -1, 0.5], [2, 0.5, 1.5])
    assert on(box, _SHELF, extent=10.0, falloff=0.1) == pytest.approx(0.606531, abs=1e-6)


def test_a_box_beside_the_anchor_is_not_on_it():
    beside = _box(1, [4, 0, 0], [5, 1, 1])  # touches the shelf's side, centroid y 0.5 = 0.5
    assert on(beside, _SHELF, extent=10.0, falloff=0.1) == 0.0


def test_a_box_hanging_below_the_anchor_is_under_it_and_not_on_it():
    # Its top y 1 meets the shelf's bottom y 1; its footprint lies inside the shelf's.
    box = _box(1, [1, 1, 0.5], [2, 2, 1.5])
    assert on(box, _SHELF, extent=10.0, falloff=0.1) == 0.0
    assert under(box, _SHELF, extent=10.0, falloff=0.1) == 1.0


def test_under_falls_off_with_the_space_between_the_anchors_bottom_and_the_subjects_top():
    # A table top y -1 to -0.8 over a bag y 0 to 0.5: 0.8 apart, exp(-0.8) = 0.449329. The
    # bag's footprint (0.5 x 0.4) lies inside the table's.
    table = _box(0, [0, -1, 0], [2, -0.8, 1])
    bag = _box(1, [0.5, 0, 0.2], [1, 0.5, 0.6])
    assert under(bag, table, extent=10.0, falloff=0.1) == pytest.approx(0.449329, abs=1e-6)
    assert under(table, bag, extent=10.0, falloff=0.1) == 0.0


def test_a_flat_footprint_touching_the_anchor_counts_as_fully_shared():
    # No width along x: no area to share. Touching the shelf's footprint counts as 1.
    sheet = _box(1, [1, -1, 0.5], [1, 0, 1.5])
    assert on(sheet, _SHELF, extent=10.0, falloff=0.1) == 1.0


def test_a_flat_footprint_away_from_the_anchor_shares_nothing():
    # x 5, 1 m from the shelf's footprint: above would give exp(-1), but nothing is shared.
    sheet = _box(1, [5, -1, 0.5], [5, 0, 1.5])
    assert on(sheet, _SHELF, extent=10.0, falloff=0.1) == 0.0


# between: the segment joins the Anchors' centroids (1, 1, 1) and (9, 1, 1): length 8, so
# with falloff 0.1 the scale is 0.8.
_LEFT = _box(1, [0, 0, 0], [2, 2, 2])
_RIGHT = _box(2, [8, 0, 0], [10, 2, 2])


def _cube_at(iid: int, x: float, y: float, z: float) -> Instance:
    return _box(iid, [x - 0.5, y - 0.5, z - 0.5], [x + 0.5, y + 0.5, z + 0.5])


def test_a_box_centred_on_the_segment_is_fully_between():
    assert between(_cube_at(0, 5, 1, 1), _LEFT, _RIGHT, falloff=0.1) == 1.0


def test_between_falls_off_with_the_distance_to_the_segment_scaled_by_its_length():
    # Centroid (5, 1, 3) is 2 from the segment: exp(-2 / 0.8) = exp(-2.5) = 0.082085.
    assert between(_cube_at(0, 5, 1, 3), _LEFT, _RIGHT, falloff=0.1) == pytest.approx(
        0.082085, abs=1e-6
    )


def test_between_scales_with_the_segment_not_the_scene():
    # Anchors at x 1 and 17: length 16, scale 1.6. Centroid (9, 1, 3) is 2 away:
    # exp(-2 / 1.6) = exp(-1.25) = 0.286505.
    farther = _box(2, [16, 0, 0], [18, 2, 2])
    assert between(_cube_at(0, 9, 1, 3), _LEFT, farther, falloff=0.1) == pytest.approx(
        0.286505, abs=1e-6
    )


def test_a_box_past_the_end_of_the_segment_is_measured_to_the_nearer_anchor():
    # Centroid (12, 1, 1) lies on the line but 3 past (9, 1, 1): exp(-3 / 0.8) = exp(-3.75)
    # = 0.023518.
    assert between(_cube_at(0, 12, 1, 1), _LEFT, _RIGHT, falloff=0.1) == pytest.approx(
        0.023518, abs=1e-6
    )


def test_between_does_not_depend_on_the_order_of_the_anchors():
    assert between(_cube_at(0, 5, 1, 3), _RIGHT, _LEFT, falloff=0.1) == pytest.approx(
        0.082085, abs=1e-6
    )


def test_nothing_is_between_two_anchors_at_one_point():
    twin = _box(2, [0, 0, 0], [2, 2, 2])  # same centroid as _LEFT: the segment has no length
    assert between(_cube_at(0, 1, 1, 1), _LEFT, twin, falloff=0.1) == 0.0
    assert between(_cube_at(0, 5, 1, 1), _LEFT, twin, falloff=0.1) == 0.0


def test_between_receives_both_anchors_of_its_constraint_through_the_solver():
    # Target candidates: one on the segment (1) and one 2 off it (exp(-2.5) = 0.082085).
    off = _cube_at(3, 5, 1, 3)
    on_segment = _cube_at(4, 5, 1, 1)
    ranked = solve(
        {"a": [off, on_segment], "b": [_LEFT], "c": [_RIGHT]},
        [Constraint(Relation.BETWEEN, "a", ("b", "c"))],
        "a",
        10.0,
        0.1,
    )
    assert [r.instances["a"].instance_id for r in ranked] == [4, 3]
    assert [r.score for r in ranked] == pytest.approx([1.0, 0.082085], abs=1e-6)


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        # The hovering box of the on test: above 1, on exp(-2) = 0.135335.
        (Relation.ABOVE, 1.0),
        (Relation.ON, 0.135335),
        (Relation.BELOW, 0.0),
        (Relation.UNDER, 0.0),
        # Gap 2 between the box (y -3 to -2) and the shelf (y 0 to 1): near exp(-2).
        (Relation.FAR_FROM, 0.864665),
    ],
)
def test_each_vertical_relation_and_far_from_is_registered(relation: Relation, expected: float):
    box = _box(1, [1, -3, 0.5], [2, -2, 1.5])
    [best] = solve({"a": [box], "b": [_SHELF]}, [Constraint(relation, "a", ("b",))], "a", 10.0, 0.1)
    assert best.score == pytest.approx(expected, abs=1e-6)


def _near(subject: str, anchor: str) -> Constraint:
    return Constraint(Relation.NEAR, subject, (anchor,))


def test_an_assignment_scores_the_product_of_its_constraints():
    a = _UNIT
    b = _box(1, [2, 0, 0], [3, 1, 1])  # gap 1 to a
    c = _box(2, [1, 3, 0], [2, 4, 1])  # gap 0 along x, 2 along y: 2
    # Scale 0.1 * 10 = 1: exp(-1) * exp(-2) = exp(-3) = 0.049787.
    [best] = solve(
        {"a": [a], "b": [b], "c": [c]},
        [_near("a", "b"), _near("a", "c")],
        target="a",
        extent=10.0,
        falloff=0.1,
    )
    assert best.score == pytest.approx(0.049787, abs=1e-6)
    assert {k: v.instance_id for k, v in best.instances.items()} == {"a": 0, "b": 1, "c": 2}


def test_no_instance_stands_for_two_objects():
    x = _UNIT
    y = _box(1, [3, 0, 0], [4, 1, 1])  # gap 2 to x
    # Pairing x with itself would score exp(0) = 1; only (x, y) and (y, x) are valid,
    # each exp(-2) = 0.135335. Equal scores keep the order of the target's candidates.
    ranked = solve({"a": [x, y], "b": [x, y]}, [_near("a", "b")], "a", 10.0, 0.1)
    assert [(r.instances["a"].instance_id, r.instances["b"].instance_id) for r in ranked] == [
        (0, 1),
        (1, 0),
    ]
    assert [r.score for r in ranked] == pytest.approx([0.135335, 0.135335], abs=1e-6)


def test_tied_scores_follow_the_position_in_the_phrase_ranking_before_the_instance_id():
    anchor = _box(1, [1, 0, 0], [2, 1, 1])
    first = _box(5, [0, 0, 0], [1, 1, 1])  # touches the anchor: 1
    second = _box(3, [2, 0, 0], [3, 1, 1])  # touches the anchor: 1
    ranked = solve({"a": [first, second], "b": [anchor]}, [_near("a", "b")], "a", 10.0, 0.1)
    assert [r.instances["a"].instance_id for r in ranked] == [5, 3]


def test_a_better_assignment_wins_over_a_better_ranked_one():
    close = _box(4, [3, 0, 0], [4, 1, 1])
    anchor = _box(9, [4, 0, 0], [5, 1, 1])
    # The first candidate is 3 from the anchor (exp(-3) = 0.049787); the second touches it (1).
    ranked = solve({"a": [_UNIT, close], "b": [anchor]}, [_near("a", "b")], "a", 10.0, 0.1)
    assert [r.instances["a"].instance_id for r in ranked] == [4, 0]
    assert [r.score for r in ranked] == pytest.approx([1.0, 0.049787], abs=1e-6)


def test_each_target_candidate_keeps_its_best_assignment_and_the_earlier_anchor_on_a_tie():
    p = _box(1, [1, 0, 0], [2, 1, 1])  # touches the target: 1
    q = _box(2, [-1, 0, 0], [0, 1, 1])  # touches the target too: 1
    far = _box(3, [5, 0, 0], [6, 1, 1])
    ranked = solve({"a": [_UNIT], "b": [far, p, q]}, [_near("a", "b")], "a", 10.0, 0.1)
    assert len(ranked) == 1
    assert ranked[0].instances["b"].instance_id == 1


def test_objects_whose_only_candidate_is_one_instance_have_no_valid_assignment():
    assert solve({"a": [_UNIT], "b": [_UNIT]}, [_near("a", "b")], "a", 10.0, 0.1) == []


def test_a_relation_receives_the_whole_candidate_set_of_its_subject(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[RelationContext] = []

    def probe(context: RelationContext) -> float:
        seen.append(context)
        return 1.0

    monkeypatch.setitem(RELATION_SCORES, Relation.LARGEST, probe)
    other = _box(1, [3, 0, 0], [4, 1, 1])
    solve({"a": [_UNIT, other]}, [Constraint(Relation.LARGEST, "a")], "a", 10.0, 0.1)
    assert [[c.instance_id for c in ctx.subject_candidates] for ctx in seen] == [[0, 1], [0, 1]]
    assert [ctx.subject.instance_id for ctx in seen] == [0, 1]
    assert [ctx.anchors for ctx in seen] == [(), ()]


def test_every_relation_has_a_scoring_function():
    assert set(RELATION_SCORES) == set(Relation)


def test_a_relation_without_a_scoring_function_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(RELATION_SCORES, Relation.TALLEST)
    with pytest.raises(UnsupportedRelationError, match="tallest"):
        solve({"a": [_UNIT]}, [Constraint(Relation.TALLEST, "a")], "a", 10.0, 0.1)


# Ordering Relations score a candidate against the subject's other candidates: ordered by the
# measure, the first distinct value scores 1, the last 0, and the distinct values between are
# evenly spaced, so tied candidates share one score.


def _cube(iid: int, x: float, side: float) -> Instance:
    """A cube of ``side`` metres standing on y = 0 (its bottom), at ``x`` along X."""
    return _box(iid, [x, -side, 0], [x + side, 0, side])


_SMALL = _cube(0, 0, 1)  # volume 1
_MEDIUM = _cube(1, 2, 2)  # volume 8
_LARGE = _cube(2, 5, 3)  # volume 27
_SLAB = _box(3, [9, -1, 0], [11, 0, 1])  # 2 x 1 x 1: volume 2


def test_a_single_candidate_is_the_largest_and_the_smallest():
    assert largest(_MEDIUM, [_MEDIUM]) == 1.0
    assert smallest(_MEDIUM, [_MEDIUM]) == 1.0


def test_of_two_candidates_the_bigger_box_is_the_largest():
    assert largest(_MEDIUM, [_SMALL, _MEDIUM]) == 1.0
    assert largest(_SMALL, [_SMALL, _MEDIUM]) == 0.0


def test_several_candidates_are_evenly_spaced_by_volume():
    # Volumes 1, 8, 27, 2. Largest first: 27 -> 1, 8 -> 2/3, 2 -> 1/3, 1 -> 0.
    candidates = [_SMALL, _MEDIUM, _LARGE, _SLAB]
    assert [largest(c, candidates) for c in candidates] == pytest.approx([0, 2 / 3, 1, 1 / 3])
    # Smallest first: 1 -> 1, 2 -> 2/3, 8 -> 1/3, 27 -> 0.
    assert [smallest(c, candidates) for c in candidates] == pytest.approx([1, 1 / 3, 0, 2 / 3])


def test_tied_volumes_share_a_score_and_the_spacing_counts_distinct_volumes():
    # Volumes 1, 8, 8, 27: three distinct values, so steps of 1/2. 27 -> 1, both 8s -> 1/2,
    # 1 -> 0.
    twin = _cube(4, 20, 2)
    candidates = [_SMALL, _MEDIUM, twin, _LARGE]
    assert [largest(c, candidates) for c in candidates] == pytest.approx([0, 0.5, 0.5, 1])


def test_candidates_tied_first_both_score_one():
    # Volumes 8, 8, 1: 8 -> 1 for both, 1 -> 0.
    twin = _cube(4, 20, 2)
    candidates = [_MEDIUM, twin, _SMALL]
    assert [largest(c, candidates) for c in candidates] == [1.0, 1.0, 0.0]


def test_candidates_that_all_tie_all_score_one():
    twin = _cube(4, 20, 2)
    assert largest(_MEDIUM, [_MEDIUM, twin]) == 1.0
    assert smallest(twin, [_MEDIUM, twin]) == 1.0


# tallest: vertical extent, along Y. A post 0.5 x 3 x 0.5 (volume 0.75) is taller than the
# 2 x 2 x 2 cube (volume 8) although it is smaller.
_POST = _box(5, [30, -3, 0], [30.5, 0, 0.5])


def test_tallest_orders_by_vertical_extent_not_volume():
    # Extents along Y: post 3, medium cube 2, slab 1. Steps of 1/2.
    candidates = [_MEDIUM, _POST, _SLAB]
    assert [tallest(c, candidates) for c in candidates] == pytest.approx([0.5, 1, 0])
    assert largest(_POST, candidates) == 0.0


def test_a_single_candidate_is_the_tallest_and_two_are_one_and_zero():
    assert tallest(_POST, [_POST]) == 1.0
    assert [tallest(c, [_POST, _MEDIUM]) for c in (_POST, _MEDIUM)] == [1.0, 0.0]


def test_tied_heights_share_the_tallest_score():
    # Extents 2 (medium), 2 (a 2 m tall slab), 1: 2 -> 1 for both, 1 -> 0.
    tall_slab = _box(6, [40, -2, 0], [41, 0, 1])
    candidates = [_MEDIUM, tall_slab, _SLAB]
    assert [tallest(c, candidates) for c in candidates] == [1.0, 1.0, 0.0]


# lowest: the height of the box's bottom. Up is -Y (ADR 0007), so a bottom (maximum y) at
# y = 3 is lower than one at y = 1.
def _shelf(iid: int, bottom_y: float) -> Instance:
    return _box(iid, [0, bottom_y - 0.2, 0], [1, bottom_y, 1])


def test_the_box_whose_bottom_has_the_largest_y_is_the_lowest():
    # Bottoms at y 1, 3, 2: lowest first 3 -> 1, 2 -> 1/2, 1 -> 0. With up as +Y the order
    # would be the reverse.
    candidates = [_shelf(0, 1), _shelf(1, 3), _shelf(2, 2)]
    assert [lowest(c, candidates) for c in candidates] == pytest.approx([0, 1, 0.5])


def test_lowest_reads_the_bottom_of_the_box_not_its_top_or_centroid():
    # A tall cabinet y -2 to 1 (bottom 1, centroid -0.5) and a small box y 0.4 to 0.6 (bottom
    # 0.6, centroid 0.5): the cabinet's bottom is lower, though its centroid and top are higher.
    cabinet = _box(0, [0, -2, 0], [1, 1, 1])
    small = _box(1, [2, 0.4, 0], [3, 0.6, 1])
    assert lowest(cabinet, [cabinet, small]) == 1.0
    assert lowest(small, [cabinet, small]) == 0.0


def test_a_single_candidate_is_the_lowest_and_tied_bottoms_share_a_score():
    assert lowest(_shelf(0, 1), [_shelf(0, 1)]) == 1.0
    # Bottoms 2, 2, 1: 2 -> 1 for both, 1 -> 0.
    candidates = [_shelf(0, 2), _shelf(1, 2), _shelf(2, 1)]
    assert [lowest(c, candidates) for c in candidates] == [1.0, 1.0, 0.0]


# nearest to, farthest from: by the gap between the candidate's box and the Anchor's box.
_WINDOW = _box(9, [0, 0, 0], [1, 1, 1])
_TOUCHING = _box(0, [1, 0, 0], [2, 1, 1])  # gap 0 to the window
_TWO_OFF = _box(1, [3, 0, 0], [4, 1, 1])  # gap 2
_FIVE_OFF = _box(2, [6, 0, 0], [7, 1, 1])  # gap 5
_CHAIRS = [_FIVE_OFF, _TOUCHING, _TWO_OFF]


def test_a_single_candidate_is_the_nearest_and_the_farthest():
    assert nearest_to(_TWO_OFF, _WINDOW, [_TWO_OFF]) == 1.0
    assert farthest_from(_TWO_OFF, _WINDOW, [_TWO_OFF]) == 1.0


def test_of_two_candidates_the_closer_box_is_the_nearest():
    assert nearest_to(_TOUCHING, _WINDOW, [_TOUCHING, _TWO_OFF]) == 1.0
    assert nearest_to(_TWO_OFF, _WINDOW, [_TOUCHING, _TWO_OFF]) == 0.0
    assert farthest_from(_TWO_OFF, _WINDOW, [_TOUCHING, _TWO_OFF]) == 1.0


def test_several_candidates_are_evenly_spaced_by_their_gap_to_the_anchor():
    # Gaps 5, 0, 2. Nearest first: 0 -> 1, 2 -> 1/2, 5 -> 0. Farthest is the reverse.
    assert [nearest_to(c, _WINDOW, _CHAIRS) for c in _CHAIRS] == [0.0, 1.0, 0.5]
    assert [farthest_from(c, _WINDOW, _CHAIRS) for c in _CHAIRS] == [1.0, 0.0, 0.5]


def test_nearest_uses_the_gap_between_boxes_not_between_centroids():
    # A long table x 1-10 touches the window (gap 0), though its centroid (5.5, 0.5, 0.5) is
    # 5 m from the window's (0.5, 0.5, 0.5); a cube at z 2-3 is 1 m away along z, its centroid
    # (0.5, 0.5, 2.5) only 2 m from the window's. The table is the nearer.
    table = _box(0, [1, 0, 0], [10, 1, 1])
    cube = _box(1, [0, 0, 2], [1, 1, 3])
    assert nearest_to(table, _WINDOW, [table, cube]) == 1.0
    assert nearest_to(cube, _WINDOW, [table, cube]) == 0.0


def test_tied_gaps_share_the_nearest_score():
    # Gaps 0, 0, 2: 0 -> 1 for both (touching on either side), 2 -> 0.
    other_side = _box(3, [-1, 0, 0], [0, 1, 1])
    candidates = [_TOUCHING, other_side, _TWO_OFF]
    assert [nearest_to(c, _WINDOW, candidates) for c in candidates] == [1.0, 1.0, 0.0]


def test_the_order_is_taken_for_the_anchor_being_scored():
    # A door at x 7-8 touches the five-off chair: gaps to the door 0 (five-off), 5 (touching
    # the window, x 2 to 7), 3 (two-off, x 4 to 7). Nearest the door: five-off 1, two-off 1/2,
    # touching 0, where nearest the window it was 0, 1/2, 1.
    door = _box(8, [7, 0, 0], [8, 1, 1])
    assert [nearest_to(c, door, _CHAIRS) for c in _CHAIRS] == [1.0, 0.0, 0.5]


def test_an_anchor_among_the_subjects_candidates_competes_with_them():
    # "the chair nearest another chair": the Anchor chair is also a candidate of the subject,
    # with gap 0 to itself, so it ranks first. Gaps to the touching chair: itself 0, two-off 1
    # (x 2 to 3), five-off 4 (x 2 to 6). Two-off scores 1/2, five-off 0.
    assert nearest_to(_TWO_OFF, _TOUCHING, _CHAIRS) == 0.5
    assert nearest_to(_FIVE_OFF, _TOUCHING, _CHAIRS) == 0.0


def test_a_size_relation_alone_ranks_the_target_through_the_solver():
    # Volumes 1, 8, 27: largest first 27 -> 1, 8 -> 1/2, 1 -> 0. No Anchor is assigned.
    ranked = solve(
        {"a": [_SMALL, _MEDIUM, _LARGE]}, [Constraint(Relation.LARGEST, "a")], "a", 10.0, 0.1
    )
    assert [r.instances["a"].instance_id for r in ranked] == [2, 1, 0]
    assert [r.score for r in ranked] == pytest.approx([1.0, 0.5, 0.0])
    assert [set(r.instances) for r in ranked] == [{"a"}, {"a"}, {"a"}]


def test_nearest_receives_the_anchor_of_each_assignment_through_the_solver():
    # Chairs [five-off, touching, two-off]; Anchor candidates [door, window]. Each chair keeps
    # its best Anchor: five-off nearest the door (1), touching nearest the window (1), two-off
    # 1/2 for both, keeping the earlier Anchor, the door. Ties keep the chairs' order.
    door = _box(8, [7, 0, 0], [8, 1, 1])
    ranked = solve(
        {"a": _CHAIRS, "b": [door, _WINDOW]},
        [Constraint(Relation.NEAREST_TO, "a", ("b",))],
        "a",
        10.0,
        0.1,
    )
    assert [(r.instances["a"].instance_id, r.instances["b"].instance_id) for r in ranked] == [
        (2, 8),
        (0, 9),
        (1, 8),
    ]
    assert [r.score for r in ranked] == pytest.approx([1.0, 1.0, 0.5])


@pytest.mark.parametrize(
    ("relation", "touching", "big"),
    [
        # The touching chair: gap 0 to the window, volume 1, extent 1, bottom y 1.
        # The big box x 3-5, y -1 to 2, z 0-2: gap 2, volume 2 * 3 * 2 = 12, extent 3, bottom
        # y 2 (lower, up is -Y).
        (Relation.NEAREST_TO, 1.0, 0.0),
        (Relation.FARTHEST_FROM, 0.0, 1.0),
        (Relation.LARGEST, 0.0, 1.0),
        (Relation.SMALLEST, 1.0, 0.0),
        (Relation.TALLEST, 0.0, 1.0),
        (Relation.LOWEST, 0.0, 1.0),
    ],
)
def test_each_ordering_relation_is_registered(relation: Relation, touching: float, big: float):
    big_box = _box(1, [3, -1, 0], [5, 2, 2])
    anchors = ("b",) if relation.anchor_count else ()
    candidates = {"a": [_TOUCHING, big_box]} | ({"b": [_WINDOW]} if anchors else {})
    ranked = solve(candidates, [Constraint(relation, "a", anchors)], "a", 10.0, 0.1)
    assert {r.instances["a"].instance_id: r.score for r in ranked} == {0: touching, 1: big}


def test_an_ordering_relation_combines_with_another_by_the_product_of_scores():
    # Near the window and largest. Extent 10, falloff 0.1: near = exp(-gap).
    # Small: touches the window (near 1), volume 1 (largest 0): 1 * 0 = 0.
    # Big: x 3-5, gap 2 (near exp(-2) = 0.135335), volume 8 (largest 1): 0.135335.
    small = _box(0, [1, 0, 0], [2, 1, 1])
    big = _box(1, [3, 0, 0], [5, 2, 2])
    ranked = solve(
        {"a": [small, big], "b": [_WINDOW]},
        [_near("a", "b"), Constraint(Relation.LARGEST, "a")],
        "a",
        10.0,
        0.1,
    )
    assert [r.instances["a"].instance_id for r in ranked] == [1, 0]
    assert [r.score for r in ranked] == pytest.approx([0.135335, 0.0], abs=1e-6)
