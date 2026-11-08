import numpy as np
import pytest

from scenetiq.domain import Aspect, Caption, Instance
from scenetiq.query import best_description, rank_by_similarity, rrf_scores, run_query


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


def test_rrf_sums_reciprocal_ranks_over_the_rankings_an_instance_appears_in():
    # rrf_k = 0: Instance 3 is 1st and 2nd -> 1 + 1/2; Instance 1 is 2nd and 1st -> 1/2 + 1;
    # Instance 2 appears only in the first ranking, 3rd -> 1/3.
    scores = rrf_scores([[3, 1, 2], [1, 3]], rrf_k=0)
    assert scores == pytest.approx({3: 1.5, 1: 1.5, 2: 1 / 3})


def test_rrf_constant_flattens_the_ranks():
    assert rrf_scores([[5, 7]], rrf_k=60) == pytest.approx({5: 1 / 61, 7: 1 / 62})


def test_ranking_by_similarity_is_descending_with_ties_broken_by_instance_id():
    assert rank_by_similarity([4, 2, 9], [0.1, 0.5, 0.5]) == [2, 9, 4]


def test_tied_fused_values_are_ordered_by_image_similarity():
    # Image ranking: 5 (0.8), 6 (0.6). Caption ranking: 6 (1.0), 5 (0.0). With rrf_k = 0 both
    # fuse to 1 + 1/2 = 1.5, so the higher image similarity (Instance 5) comes first.
    instances = [_instance(6, [0.6, 0.8], [1.0, 0.0]), _instance(5, [0.8, 0.6], [0.0, 1.0])]
    matches = run_query(instances, np.array([1.0, 0.0]), 2, 0, np.array([1.0, 0.0]))
    assert [m.instance_id for m in matches] == [5, 6]
    # Normalised by the best value over two rankings, 2 / (0 + 1) = 2: 1.5 / 2 = 0.75.
    assert [m.score for m in matches] == pytest.approx([0.75, 0.75])


def test_tied_fused_values_with_equal_image_similarity_are_ordered_by_instance_id():
    instances = [_instance(9, [1.0, 0.0]), _instance(4, [1.0, 0.0])]
    matches = run_query(instances, np.array([1.0, 0.0]), 2, 60)
    assert [m.instance_id for m in matches] == [4, 9]
