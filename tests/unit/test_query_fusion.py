import numpy as np
import pytest

from scenetiq.domain import Aspect, Caption, Instance
from scenetiq.query import best_description, is_semantic_match, run_query, standardize


def _instance(iid: int, image: list[float], caption: list[float] | None = None) -> Instance:
    zero = np.zeros(3, dtype=np.float32)
    return Instance(
        instance_id=iid,
        centroid=zero,
        bbox_min=zero,
        bbox_max=zero,
        embedding=np.asarray(image, dtype=np.float32),
        caption=None if caption is None else Caption(f"thing {iid}"),
        caption_embeddings=(
            {} if caption is None else {Aspect.IDENTITY: np.asarray(caption, dtype=np.float32)}
        ),
    )


def _vectors(**by_aspect: list[float]) -> dict[Aspect, np.ndarray]:
    return {Aspect(a): np.asarray(v, dtype=np.float32) for a, v in by_aspect.items()}


def test_the_best_description_is_the_one_most_similar_to_the_query():
    embeddings = _vectors(identity=[1.0, 0.0], appearance=[0.6, 0.8], use=[0.0, -1.0])
    # Query (0, 1): identity 0, appearance 0.8, use -1.
    assert best_description(embeddings, np.array([0.0, 1.0])) == (
        pytest.approx(0.8),
        Aspect.APPEARANCE,
    )


def test_an_aspect_without_a_description_cannot_be_the_best():
    # Query (0.6, 0.8): identity -1.0, material -0.6. Appearance and use have no
    # Description, so they do not count as similarity 0 and the best stays negative.
    embeddings = _vectors(identity=[-0.6, -0.8], material=[-1.0, 0.0])
    assert best_description(embeddings, np.array([0.6, 0.8])) == (
        pytest.approx(-0.6),
        Aspect.MATERIAL,
    )


def test_tied_descriptions_go_to_the_first_aspect_in_order():
    embeddings = _vectors(use=[1.0, 0.0], appearance=[1.0, 0.0])
    assert best_description(embeddings, np.array([1.0, 0.0])) == (
        pytest.approx(1.0),
        Aspect.APPEARANCE,
    )


def test_no_descriptions_have_no_best():
    assert best_description({}, np.array([1.0, 0.0])) is None


def test_only_the_aspects_a_query_speaks_about_are_compared():
    embeddings = _vectors(identity=[0.6, 0.8], use=[1.0, 0.0])
    # Query (1, 0): identity 0.6, use 1.0. A Query about Identity never reads the Use.
    assert best_description(embeddings, np.array([1.0, 0.0]), (Aspect.IDENTITY,)) == (
        pytest.approx(0.6),
        Aspect.IDENTITY,
    )


def test_an_instance_lacking_every_aspect_asked_about_is_compared_on_all_it_has():
    embeddings = _vectors(identity=[0.6, 0.8], use=[1.0, 0.0])
    assert best_description(embeddings, np.array([1.0, 0.0]), (Aspect.MATERIAL,)) == (
        pytest.approx(1.0),
        Aspect.USE,
    )


def test_standardize_measures_each_value_in_deviations_from_the_mean():
    # Mean 5, population standard deviation 2.
    assert standardize([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(
        [-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]
    )


def test_standardize_gives_zero_when_nothing_stands_out():
    assert standardize([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]
    assert standardize([0.7]) == [0.0]
    assert standardize([]) == []


def _three() -> list[Instance]:
    # Image similarities to the Query (1, 0): 0.6, 0.3, 0.0 -> mean 0.3, deviation
    # sqrt(0.06) = 0.244949 -> standardized 1.224745, 0, -1.224745.
    # Caption similarities: 0.2, 0.8, 0.2 -> mean 0.4, deviation sqrt(0.08) = 0.282843
    # -> standardized -0.707107, 1.414214, -0.707107.
    # Fused (the mean of the two): 0.258819, 0.707107, -0.965926.
    return [
        _instance(1, [0.6, 0.0], [0.2, 0.0]),
        _instance(2, [0.3, 0.0], [0.8, 0.0]),
        _instance(3, [0.0, 0.0], [0.2, 0.0]),
    ]


def test_matches_are_scored_against_the_best_fused_value():
    query = np.array([1.0, 0.0])
    matches = run_query(_three(), query, 3, None, query)
    assert [m.instance_id for m in matches] == [2, 1, 3]
    # 0.707107 / 0.707107, 0.258819 / 0.707107, -0.965926 / 0.707107.
    assert [m.score for m in matches] == pytest.approx([1.0, 0.366025, -1.366025], abs=1e-6)


def test_an_instance_scoring_too_far_below_the_best_is_not_a_match():
    query = np.array([1.0, 0.0])
    assert [m.instance_id for m in run_query(_three(), query, 3, 0.3, query)] == [2, 1]
    assert [m.instance_id for m in run_query(_three(), query, 3, 0.5, query)] == [2]


def test_at_most_top_k_matches_are_returned():
    query = np.array([1.0, 0.0])
    assert [m.instance_id for m in run_query(_three(), query, 1, 0.3, query)] == [2]


def test_an_instance_without_a_caption_is_average_on_the_caption_channel():
    # Image: 0.5, 0.3, 0.1 -> standardized 1.224745, 0, -1.224745. Caption, over the two
    # captioned Instances: 1.0, 0.0 -> standardized 1, -1; Instance 7 has none, so 0.
    # Fused: 7 -> 0.612372, 8 -> 0.5, 9 -> -1.112372.
    instances = [
        _instance(7, [0.5, 0.0]),
        _instance(8, [0.3, 0.0], [1.0, 0.0]),
        _instance(9, [0.1, 0.0], [0.0, 0.0]),
    ]
    query = np.array([1.0, 0.0])
    matches = run_query(instances, query, 3, None, query)
    assert [m.instance_id for m in matches] == [7, 8, 9]
    assert [m.score for m in matches] == pytest.approx([1.0, 0.816497, -1.816497], abs=1e-6)
    assert matches[0].caption_score is None


def test_tied_fused_values_are_ordered_by_image_similarity():
    # Image: 5 -> 0.8, 6 -> 0.6, standardized 1 and -1. Caption: 6 -> 1.0, 5 -> 0.0,
    # standardized 1 and -1. Both fuse to 0, so the higher image similarity comes first,
    # and with no best value to measure against both score 1.
    instances = [_instance(6, [0.6, 0.8], [1.0, 0.0]), _instance(5, [0.8, 0.6], [0.0, 1.0])]
    matches = run_query(instances, np.array([1.0, 0.0]), 2, 0.5, np.array([1.0, 0.0]))
    assert [m.instance_id for m in matches] == [5, 6]
    assert [m.score for m in matches] == [1.0, 1.0]


def test_tied_fused_values_with_equal_image_similarity_are_ordered_by_instance_id():
    instances = [_instance(9, [1.0, 0.0]), _instance(4, [1.0, 0.0])]
    matches = run_query(instances, np.array([1.0, 0.0]), 2, 0.5)
    assert [m.instance_id for m in matches] == [4, 9]


def test_a_query_about_one_aspect_ranks_captions_on_that_aspect():
    zero = np.zeros(3, dtype=np.float32)
    floor = Instance(
        1, zero, zero, zero, np.array([0.0, 1.0], np.float32), Caption("floor"),
        _vectors(identity=[0.0, 1.0], use=[1.0, 0.0]),
    )  # fmt: skip
    chair = Instance(
        2, zero, zero, zero, np.array([0.0, 1.0], np.float32), Caption("chair"),
        _vectors(identity=[0.8, 0.6]),
    )  # fmt: skip
    query = np.array([1.0, 0.0])
    # Every Aspect: the floor's Use (1.0) beats the chair's Identity (0.8).
    assert [m.instance_id for m in run_query([floor, chair], query, 1, None, query)] == [1]
    # Identity only: the floor's Identity is 0.0, so the chair wins.
    matches = run_query([floor, chair], query, 1, None, query, (Aspect.IDENTITY,))
    assert [m.instance_id for m in matches] == [2]
    assert matches[0].caption_aspect is Aspect.IDENTITY


def test_is_semantic_match_recognizes_name_and_synonym_words():
    zero = np.zeros(3, dtype=np.float32)
    chair = Instance(
        1,
        zero,
        zero,
        zero,
        np.array([1.0, 0.0], np.float32),
        Caption("office chair", synonyms=("desk chair", "work seat")),
    )
    assert is_semantic_match("chair", chair)
    assert is_semantic_match("office chair", chair)
    assert is_semantic_match("work seat", chair)
    assert is_semantic_match("desk", chair)


def test_is_semantic_match_rejects_unrelated_words():
    zero = np.zeros(3, dtype=np.float32)
    chair = Instance(
        1,
        zero,
        zero,
        zero,
        np.array([1.0, 0.0], np.float32),
        Caption("office chair", synonyms=("desk chair",)),
    )
    assert not is_semantic_match("cat", chair, caption_score=0.1)
    assert not is_semantic_match("refrigerator", chair, caption_score=0.0)


def test_negative_query_with_no_semantic_matches_returns_empty_list():
    zero = np.zeros(3, dtype=np.float32)
    chair = Instance(
        1,
        zero,
        zero,
        zero,
        np.array([1.0, 0.0], np.float32),
        Caption("chair"),
        _vectors(identity=[0.1, 0.0]),
    )
    desk = Instance(
        2,
        zero,
        zero,
        zero,
        np.array([0.0, 1.0], np.float32),
        Caption("desk"),
        _vectors(identity=[0.1, 0.0]),
    )
    query = np.array([1.0, 0.0])
    # "cat" matches neither "chair" nor "desk", so run_query returns []
    matches = run_query([chair, desk], query, None, 0.5, query, phrase="cat")
    assert matches == []


def test_semantic_label_filtering_excludes_unrelated_instances():
    zero = np.zeros(3, dtype=np.float32)
    chair1 = Instance(
        1,
        zero,
        zero,
        zero,
        np.array([1.0, 0.0], np.float32),
        Caption("office chair"),
        _vectors(identity=[1.0, 0.0]),
    )
    chair2 = Instance(
        2,
        zero,
        zero,
        zero,
        np.array([0.9, 0.1], np.float32),
        Caption("wooden chair"),
        _vectors(identity=[0.9, 0.1]),
    )
    floor = Instance(
        3,
        zero,
        zero,
        zero,
        np.array([0.5, 0.5], np.float32),
        Caption("floor"),
        _vectors(identity=[0.1, 0.0]),
    )
    query = np.array([1.0, 0.0])
    # Searching for "chair" returns only chair1 and chair2, excluding floor
    matches = run_query([chair1, chair2, floor], query, None, 0.5, query, phrase="chair")
    assert [m.instance_id for m in matches] == [1, 2]


def test_run_query_without_top_k_returns_all_qualifying_semantic_matches():
    zero = np.zeros(3, dtype=np.float32)
    chairs = [
        Instance(
            i,
            zero,
            zero,
            zero,
            np.array([1.0, 0.0], np.float32),
            Caption("chair"),
            _vectors(identity=[1.0, 0.0]),
        )
        for i in range(1, 9)
    ]
    query = np.array([1.0, 0.0])
    # All 8 chairs qualify; without top_k, all 8 are returned
    matches = run_query(chairs, query, None, 0.5, query, phrase="chair")
    assert len(matches) == 8
