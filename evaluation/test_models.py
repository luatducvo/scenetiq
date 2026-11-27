import numpy as np
import pytest

from scenetiq.domain.scene import Capture


def test_siglip2_encodes_query_text_to_a_unit_vector(encoder):
    # Runs without the Query environment variables, so a broken text tower is always caught.
    embedding = encoder.encode("a chair")
    assert embedding.ndim == 1
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-3)


def test_siglip2_encodes_a_query_longer_than_its_text_window(encoder):
    # About 500 characters, far beyond 64 tokens: raised ValueError before truncation.
    sentence = "the soft grey sofa with wide armrests next to the tall window, "
    text = (sentence * 8)[:500]
    assert len(text) == 500
    embedding = encoder.encode(text)
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-3)


def test_qwen3_embedding_ranks_the_description_a_query_describes_first(caption_encoder):
    documents = caption_encoder.encode_documents(
        [
            "It is used for sitting.",
            "It is made of polished wood.",
            "A lamp, also called a light.",
        ]
    )
    assert documents.shape[0] == 3
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-3)
    for text, expected in [("something to sit on", 0), ("a wooden thing", 1), ("cái đèn", 2)]:
        similarities = documents @ caption_encoder.encode_query(text)
        assert int(np.argmax(similarities)) == expected, (text, similarities)


def test_geocalib_up_points_up_in_upright_photos(smoke_capture: Capture):
    from PIL import Image

    from scenetiq.pipeline.stages.alignment import GeoCalibUpEstimator

    with Image.open(smoke_capture.views[0].path) as photo:
        image = np.asarray(photo.convert("RGB").resize((504, 378)), dtype=np.uint8)
    up = np.asarray(GeoCalibUpEstimator().estimate(image[None], None))[0]
    # In the OpenCV camera frame +Y points down the image, so up has a negative y.
    assert up[1] < 0


def _load_siglip2_from_two_threads() -> bool:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from scenetiq.pipeline.models.siglip2 import SigLIP2Encoder

    encoder = SigLIP2Encoder()
    barrier = Barrier(2)

    def load(_: int):
        barrier.wait()
        return encoder._load()[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(load, range(2))
    return first is second


def test_threads_sharing_siglip2_load_it_once():
    import gc

    import torch

    assert _load_siglip2_from_two_threads()
    # The second SigLIP2 must not stay beside the shared one, or the GPU memory check fails.
    gc.collect()
    torch.cuda.empty_cache()
