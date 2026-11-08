from pathlib import Path

import numpy as np

from scenetiq.artifacts import CAPTION_EMBEDDINGS_FILE, load_instances, save_reconstruction
from scenetiq.domain import Aspect, Caption, Instance, Reconstruction


def _instance(iid: int, caption: Caption | None = None, **vectors: list[float]) -> Instance:
    zero = np.zeros(3, dtype=np.float32)
    return Instance(
        iid,
        zero,
        zero,
        zero,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        caption=caption,
        caption_embeddings={Aspect(a): np.array(v, dtype=np.float32) for a, v in vectors.items()},
    )


def _save(directory: Path, instances: list[Instance]) -> None:
    rec = Reconstruction(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        instance_ids=np.array([-1], dtype=np.int32),
        instances=instances,
    )
    save_reconstruction(directory, rec)


def test_captions_and_their_description_embeddings_round_trip(tmp_path: Path):
    sofa = Caption("sofa", ("couch",), identity="A sofa.", material="It is fabric.")
    _save(
        tmp_path,
        [
            _instance(0, sofa, identity=[0.6, 0.8], material=[0.0, 1.0]),
            _instance(1),
        ],
    )

    first, second = load_instances(tmp_path)

    assert first.caption == sofa
    assert list(first.caption_embeddings) == [Aspect.IDENTITY, Aspect.MATERIAL]
    np.testing.assert_allclose(first.caption_embeddings[Aspect.IDENTITY], [0.6, 0.8])
    np.testing.assert_allclose(first.caption_embeddings[Aspect.MATERIAL], [0.0, 1.0])
    assert second.caption is None
    assert second.caption_embeddings == {}


def test_caption_embeddings_are_stored_by_instance_and_aspect_with_zero_rows(tmp_path: Path):
    sofa = Caption("sofa", identity="A sofa.", use="It is for sitting.")
    _save(tmp_path, [_instance(0), _instance(1, sofa, identity=[1.0, 0.0], use=[0.0, 1.0])])

    stored = np.load(tmp_path / CAPTION_EMBEDDINGS_FILE)

    # Instances x Aspects (identity, appearance, material, use) x dimensions.
    assert stored.shape == (2, 4, 2)
    np.testing.assert_array_equal(stored[0], np.zeros((4, 2)))
    np.testing.assert_array_equal(stored[1], [[1, 0], [0, 0], [0, 0], [0, 1]])


def test_no_caption_embeddings_file_without_captions(tmp_path: Path):
    _save(tmp_path, [_instance(0), _instance(1)])
    assert not (tmp_path / CAPTION_EMBEDDINGS_FILE).exists()
    assert [inst.caption for inst in load_instances(tmp_path)] == [None, None]
