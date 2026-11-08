import numpy as np
import pytest

from scenetiq.domain import Instance
from scenetiq.query import rank_by_similarity, rrf_scores, run_query


def _instance(iid: int, image: list[float], caption: list[float] | None = None) -> Instance:
    zero = np.zeros(3, dtype=np.float32)
    return Instance(
        instance_id=iid,
        centroid=zero,
        bbox_min=zero,
        bbox_max=zero,
        embedding=np.asarray(image, dtype=np.float32),
        caption_embedding=None if caption is None else np.asarray(caption, dtype=np.float32),
    )


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
