import numpy as np
import pytest

from scenetiq.domain import Constraint, Instance, Relation
from scenetiq.spatial import (
    RELATION_SCORES,
    RelationContext,
    UnsupportedRelationError,
    near,
    scene_extent,
    solve,
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


@pytest.mark.parametrize("relation", [r for r in Relation if r not in RELATION_SCORES])
def test_a_relation_without_a_scoring_function_fails_loudly(relation: Relation):
    anchors = tuple("bc"[: relation.anchor_count])
    candidates = {"a": [_UNIT]} | {
        k: [_box(i + 1, [5, 5, 5], [6, 6, 6])] for i, k in enumerate(anchors)
    }
    with pytest.raises(UnsupportedRelationError, match=relation.value):
        solve(candidates, [Constraint(relation, "a", anchors)], "a", 10.0, 0.1)
