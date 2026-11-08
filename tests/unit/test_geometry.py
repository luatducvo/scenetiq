import numpy as np
import pytest

from scenetiq.domain import Caption, EmbeddedCaption, Instance
from scenetiq.geometry import build_instances, l2_normalize


def test_instance_centroid_and_bounding_box_from_its_points():
    points = np.array([[0, 0, 0], [0.5, 0, 0], [0, 1, 0], [0.5, 1, 1], [7, 7, 7]], dtype=np.float32)
    ids = np.array([3, 3, 3, 3, -1], dtype=np.int32)

    (inst,) = build_instances(points, ids)

    assert inst.instance_id == 3
    np.testing.assert_allclose(inst.centroid, [0.25, 0.5, 0.25])
    np.testing.assert_allclose(inst.bbox_min, [0, 0, 0])
    np.testing.assert_allclose(inst.bbox_max, [0.5, 1, 1])


def test_points_without_instance_produce_no_instance():
    points = np.zeros((2, 3), dtype=np.float32)
    assert build_instances(points, np.array([-1, -1], dtype=np.int32)) == []


def test_instances_are_sorted_by_id_and_carry_their_embedding():
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    emb = np.array([1, 0], dtype=np.float32)
    insts = build_instances(points, np.array([5, 2], dtype=np.int32), {5: emb})
    assert [i.instance_id for i in insts] == [2, 5]
    assert insts[0].embedding is None
    assert insts[1].embedding is emb


def test_l2_normalize_gives_unit_length_and_keeps_zero():
    np.testing.assert_allclose(l2_normalize([3, 4]), [0.6, 0.8])
    np.testing.assert_allclose(l2_normalize([0, 0]), [0, 0])


def test_instances_carry_the_caption_and_embedding_given_for_their_id():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    ids = np.array([3, 5], dtype=np.int32)
    sofa = Caption("sofa", ("couch",))
    vector = np.array([0.6, 0.8], dtype=np.float32)
    instances = build_instances(points, ids, captions={5: EmbeddedCaption(sofa, vector)})
    assert instances[0].caption is None
    assert instances[0].caption_embedding is None
    assert instances[1].caption == sofa
    np.testing.assert_array_equal(instances[1].caption_embedding, vector)


def test_an_instance_cannot_have_a_caption_embedding_without_a_caption():
    zero = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="Caption"):
        Instance(0, zero, zero, zero, caption_embedding=np.ones(2, dtype=np.float32))
