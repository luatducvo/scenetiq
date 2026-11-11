import pytest

from scenetiq.domain import Constraint, ParsedQuery, QueryObject, Relation


def test_every_relation_takes_a_fixed_number_of_anchors():
    counts = {relation.value: relation.anchor_count for relation in Relation}
    assert counts == {
        "above": 1,
        "below": 1,
        "on": 1,
        "under": 1,
        "near": 1,
        "far_from": 1,
        "between": 2,
        "nearest_to": 1,
        "farthest_from": 1,
        "largest": 0,
        "smallest": 0,
        "tallest": 0,
        "lowest": 0,
    }


def test_a_query_without_relations_is_one_object_for_the_whole_text():
    parsed = ParsedQuery.without_relations("red chair")
    assert parsed.objects == (QueryObject("a", "red chair"),)
    assert parsed.target == "a"
    assert parsed.constraints == ()
    assert parsed.unsupported == ()


def test_a_consistent_near_query_is_accepted():
    parsed = ParsedQuery(
        objects=(QueryObject("a", "chair"), QueryObject("b", "window")),
        target="a",
        constraints=(Constraint(Relation.NEAR, "a", ("b",)),),
    )
    assert parsed.constraints[0].anchors == ("b",)


_TWO = (QueryObject("a", "chair"), QueryObject("b", "window"))


@pytest.mark.parametrize(
    ("objects", "target", "constraints", "reason"),
    [
        ((), "a", (), "at least one object"),
        (tuple(QueryObject(i, "x") for i in "abcde"), "a", (), "at most 4 objects"),
        ((QueryObject("a", "chair"), QueryObject("a", "window")), "a", (), "repeats"),
        (_TWO, "c", (), "target"),
        (_TWO, "a", (Constraint(Relation.NEAR, "c", ("b",)),), "not listed"),
        (_TWO, "a", (Constraint(Relation.NEAR, "a", ("c",)),), "not listed"),
        (_TWO, "a", (Constraint(Relation.NEAR, "a", ()),), "1 Anchor"),
        (_TWO, "a", (Constraint(Relation.LARGEST, "a", ("b",)),), "0 Anchor"),
        (_TWO, "a", (Constraint(Relation.BETWEEN, "a", ("b",)),), "2 Anchor"),
        (_TWO, "a", (Constraint(Relation.NEAR, "a", ("a",)),), "own Anchor"),
        (_TWO, "a", (Constraint(Relation.NEAR, "a", ("b",)),) * 5, "at most 4 constraints"),
    ],
)
def test_an_inconsistent_parsed_query_is_refused(objects, target, constraints, reason):
    with pytest.raises(ValueError, match=reason):
        ParsedQuery(objects=objects, target=target, constraints=constraints)
