import os

import numpy as np
import pytest

from scenetiq.domain import Capture, Stage
from scenetiq.pipeline.real import RealPipeline

pytestmark = pytest.mark.gpu


class StageRecorder:
    """A job reporter that records the stages and notes of one run."""

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages
        self.notes: list[str] = []

    def stage(self, stage: Stage) -> None:
        self.stages.append(stage)

    def checkpoint(self) -> None:
        pass

    def note(self, message: str) -> None:
        self.notes.append(message)


@pytest.fixture(scope="module")
def encoder():
    """One SigLIP2 model shared by the Pipeline and the Query tests, as on the server."""
    from scenetiq.pipeline.siglip import SigLIP2Encoder

    return SigLIP2Encoder()


@pytest.fixture(scope="module")
def caption_encoder():
    """One Qwen3-Embedding model shared by the Pipeline and the Query tests, as on the server."""
    from scenetiq.pipeline.qwen_embedding import Qwen3TextEmbedder

    return Qwen3TextEmbedder()


@pytest.fixture(scope="module")
def captioner():
    """One Qwen3-VL model shared by the Pipeline and the Query parser, as on the server."""
    from scenetiq.pipeline.qwen_vl import QwenVLCaptioner

    return QwenVLCaptioner()


@pytest.fixture(scope="module")
def real_result(smoke_capture: Capture, encoder, caption_encoder, captioner):
    stages: list[Stage] = []
    pipeline = RealPipeline(
        image_encoder=encoder, caption_encoder=caption_encoder, captioner=captioner
    )
    reconstruction = pipeline.run(smoke_capture, StageRecorder(stages))
    return reconstruction, stages


def test_real_reconstruction_is_a_metric_point_cloud(real_result, smoke_capture: Capture):
    rec, stages = real_result
    assert stages[:2] == [Stage.RECONSTRUCTING, Stage.FUSING]
    assert len(rec.points) > 1000
    assert rec.points.dtype == np.float32
    assert rec.colors.shape == rec.points.shape
    assert len(rec.instance_ids) == len(rec.points)
    assert len(rec.cameras) == len(smoke_capture.views)
    # A room-scale Scene fits in tens of metres, not thousands or millimetres.
    extent = rec.points.max(axis=0) - rec.points.min(axis=0)
    assert 0.3 < float(extent.max()) < 50.0


def test_real_pipeline_reports_every_stage(real_result):
    _, stages = real_result
    assert stages == [
        Stage.RECONSTRUCTING, Stage.FUSING, Stage.ALIGNING, Stage.SUPERPOINTS, Stage.MASKING,
        Stage.MERGING, Stage.EMBEDDING, Stage.CAPTIONING,
    ]  # fmt: skip


def test_some_instances_get_a_caption_with_description_embeddings(real_result):
    rec, _ = real_result
    captioned = [inst for inst in rec.instances if inst.caption is not None]
    for inst in captioned:
        print(f"caption {inst.instance_id}: {inst.caption}")
    assert captioned, "no Instance got a Caption"
    for inst in captioned:
        assert inst.caption is not None
        assert inst.caption.name
        assert len(inst.caption.synonyms) <= 3
        # Every non-empty Description, and only those, has a unit-length embedding.
        assert inst.caption.identity, "Identity is filled in from the Name when empty"
        assert list(inst.caption_embeddings) == list(inst.caption.descriptions())
        for vector in inst.caption_embeddings.values():
            assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-3)


def test_cameras_of_upright_photos_end_up_with_up_near_minus_y(real_result):
    """An inverted sign from the up estimator would put this near 180 degrees.

    The smoke Capture's photos must be held roughly upright.
    """
    rec, _ = real_result
    camera_ups = np.stack([-cam.cam_to_world[:3, 1] for cam in rec.cameras]).astype(np.float64)
    mean_up = camera_ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    # cos 45 degrees = 0.7071, against (0, -1, 0).
    assert -mean_up[1] > 0.7071


def test_geocalib_up_points_up_in_upright_photos(smoke_capture: Capture):
    """Checks the sign directly, in case the Scene was left unrotated for lack of consensus."""
    from PIL import Image

    from scenetiq.pipeline.alignment import GeoCalibUpEstimator

    with Image.open(smoke_capture.views[0].path) as photo:
        image = np.asarray(photo.convert("RGB").resize((504, 378)), dtype=np.uint8)
    up = np.asarray(GeoCalibUpEstimator().estimate(image[None], None))[0]
    # In the OpenCV camera frame +Y points down the image, so up has a negative y.
    assert up[1] < 0


def test_real_reconstruction_has_consistent_instances(real_result):
    rec, _ = real_result
    assert len(rec.instances) >= 1
    ids = {i.instance_id for i in rec.instances}
    assert ids == set(np.unique(rec.instance_ids[rec.instance_ids >= 0]).tolist())
    for inst in rec.instances:
        member = rec.points[rec.instance_ids == inst.instance_id]
        np.testing.assert_allclose(inst.centroid, member.mean(axis=0), atol=1e-4)
        assert np.all(inst.bbox_min <= inst.centroid)
        assert np.all(inst.centroid <= inst.bbox_max)


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


def test_a_query_returns_at_most_the_configured_number_of_matches(
    real_result, encoder, caption_encoder
):
    # Runs without the Query environment variables, so the cut of Matches is always checked.
    from scenetiq.query import run_query

    rec, _ = real_result
    settings = _configured_settings()
    matches = run_query(
        rec.instances,
        encoder.encode("a chair"),
        settings.query_top_k,
        settings.query_score_ratio,
        caption_encoder.encode_query("a chair"),
    )
    assert 1 <= len(matches) <= min(settings.query_top_k, len(rec.instances))
    assert matches[0].score == pytest.approx(1.0)
    assert all(settings.query_score_ratio <= m.score <= 1.0 + 1e-6 for m in matches)
    for m in matches:
        assert (m.caption_aspect is None) == (m.caption_score is None)
    # Over every Instance, the Caption channel must name an Aspect for the captioned ones.
    everything = run_query(
        rec.instances,
        encoder.encode("a chair"),
        len(rec.instances),
        None,
        caption_encoder.encode_query("a chair"),
    )
    assert len(everything) == len(rec.instances)
    assert any(m.caption_aspect is not None for m in everything)


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


def test_only_the_query_models_stay_on_the_gpu_after_a_run(real_result):
    import torch

    # SigLIP2 (3.74 GB), Qwen3-Embedding-0.6B (1.19 GB) and Qwen3-VL-8B (about 17.5 GB) at
    # bf16 stay, about 22.5 GB; SAM2 (0.90 GB) or DA3 left behind would push this past 23 GB.
    assert torch.cuda.memory_allocated() < 23_000_000_000


def test_threads_sharing_siglip2_load_it_once():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from scenetiq.pipeline.siglip import SigLIP2Encoder

    encoder = SigLIP2Encoder()
    barrier = Barrier(2)

    def load(_: int):
        barrier.wait()
        return encoder._load()[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(load, range(2))
    assert first is second


def test_superpoints_cover_every_point_of_the_fused_mesh(smoke_capture: Capture):
    from scenetiq.pipeline.fusion import fuse
    from scenetiq.pipeline.reconstruct import DA3Reconstructor
    from scenetiq.pipeline.superpoints import mesh_superpoints

    view_data = DA3Reconstructor().reconstruct(smoke_capture.views)
    geometry = fuse(view_data)
    ids = mesh_superpoints(geometry.vertices, geometry.triangles)
    assert len(ids) == len(geometry.vertices)
    assert ids.min() == 0
    assert 10 < ids.max() + 1 < len(geometry.vertices)


SMOKE_QUERY_ENV = "SCENETIQ_SMOKE_QUERY"
SMOKE_QUERY_BOX_ENV = "SCENETIQ_SMOKE_QUERY_BOX"
MIN_POINTS_IN_BOX = 0.5
MIN_POINTS_IN_PHOTO = 0.2


def _configured_settings():
    from scenetiq.config import Settings, overrides_from_json

    return Settings.from_overrides("", overrides_from_json(os.environ.get("SCENETIQ_CONFIG")))


def _annotated_box(capture: Capture) -> tuple[int, tuple[float, float, float, float]]:
    """The View index and its box (left, top, right, bottom; original-photo pixels)."""
    raw = os.environ.get(SMOKE_QUERY_BOX_ENV)
    if not raw:
        pytest.fail(
            f"set {SMOKE_QUERY_BOX_ENV}='view,left,top,right,bottom': the queried object's "
            "box in that photo (view = index in the sorted smoke Capture)"
        )
    view, left, top, right, bottom = (float(x) for x in raw.split(","))
    assert 0 <= int(view) < len(capture.views), f"{SMOKE_QUERY_BOX_ENV} view index out of range"
    return int(view), (left, top, right, bottom)


def _box_overlap(points, camera, box, photo_size) -> tuple[float, float]:
    """``(share of points projecting into the photo, share of those inside box)``."""
    from scenetiq.pipeline.adapters import c2w_to_w2c

    w2c = c2w_to_w2c(camera.cam_to_world)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    front = cam[:, 2] > 1e-6
    cam = cam[front]
    k = camera.intrinsics
    # Pixels at the depth resolution, scaled to the original photo the box refers to.
    sx, sy = photo_size[0] / camera.width, photo_size[1] / camera.height
    u = (k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2]) * sx
    v = (k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]) * sy
    in_image = (u >= 0) & (u < photo_size[0]) & (v >= 0) & (v < photo_size[1])
    in_photo = float(np.count_nonzero(in_image)) / max(len(points), 1)
    if not np.any(in_image):
        return in_photo, 0.0
    left, top, right, bottom = box
    u, v = u[in_image], v[in_image]
    inside = (u >= left) & (u < right) & (v >= top) & (v < bottom)
    return in_photo, float(np.mean(inside))


def test_query_for_a_present_object_matches_that_object(
    real_result, smoke_capture: Capture, encoder, caption_encoder
):
    """A clearly visible object must be the best Match.

    Set SCENETIQ_SMOKE_QUERY to the object's name and SCENETIQ_SMOKE_QUERY_BOX to its
    box in one photo of the smoke Capture.
    """
    from PIL import Image

    from scenetiq.query import run_query

    text = os.environ.get(SMOKE_QUERY_ENV)
    if not text:
        pytest.skip(f"set {SMOKE_QUERY_ENV} to an object visible in the smoke Capture")
    view, box = _annotated_box(smoke_capture)
    rec, stages = real_result
    assert stages[-1] == Stage.CAPTIONING
    assert all(inst.embedding is not None for inst in rec.instances)

    settings = _configured_settings()
    matches = run_query(
        rec.instances,
        encoder.encode(text),
        settings.query_top_k,
        settings.query_score_ratio,
        caption_encoder.encode_query(text),
    )
    print(f"{text!r}: {[(m.instance_id, round(m.score, 3)) for m in matches]}")

    assert 1 <= len(matches) <= min(settings.query_top_k, len(rec.instances))
    best = matches[0].instance_id
    with Image.open(smoke_capture.views[view].path) as photo:
        photo_size = photo.size
    in_photo, share = _box_overlap(
        rec.points[rec.instance_ids == best], rec.cameras[view], box, photo_size
    )
    assert in_photo >= MIN_POINTS_IN_PHOTO, (
        f"best Match (Instance {best}) is mostly outside photo {view} ({in_photo:.0%} inside)"
    )
    assert share >= MIN_POINTS_IN_BOX, (
        f"best Match (Instance {best}): only {share:.0%} of its in-photo points are in the box"
    )


SMOKE_SPATIAL_QUERY_ENV = "SCENETIQ_SMOKE_SPATIAL_QUERY"
DEFAULT_SPATIAL_QUERY = "the object nearest to the chair"


@pytest.fixture(scope="module")
def query_parser(captioner):
    """The real Query parser on the Pipeline's Qwen3-VL, as on the server."""
    from scenetiq.query_parser import VLMQueryParser

    return VLMQueryParser(captioner, _configured_settings().query_parser)


def _answer(text, rec, query_parser, encoder, caption_encoder):
    """The parsed Query and Matches of ``text``, routed as the Query endpoint routes them."""
    import time

    from scenetiq.domain import QueryObject
    from scenetiq.query import run_query
    from scenetiq.spatial import run_spatial_query

    settings = _configured_settings()
    captioned = any(inst.caption_embeddings for inst in rec.instances)

    def rank(wanted: QueryObject, top_k: int, score_ratio: float | None = None):
        return run_query(
            rec.instances,
            encoder.encode(wanted.phrase),
            top_k,
            score_ratio,
            caption_encoder.encode_query(wanted.phrase) if captioned else None,
            wanted.aspects,
        )

    started = time.perf_counter()
    parsed = query_parser.parse(text)  # a QueryParseError fails the test: shape is checked
    print(f"parse of {text!r} took {time.perf_counter() - started:.1f} s: {parsed}")
    matches = None
    if parsed.constraints:
        matches = run_spatial_query(
            rec.instances, parsed, rank, settings.query_top_k, settings.spatial
        )
    if matches is None:
        wanted = next(o for o in parsed.objects if o.object_id == parsed.target)
        matches = rank(wanted, settings.query_top_k, settings.query_score_ratio)
    return parsed, matches


def test_a_query_with_a_relation_returns_matches_with_anchors(
    real_result, query_parser, encoder, caption_encoder
):
    """Shape only. Set SCENETIQ_SMOKE_SPATIAL_QUERY to a Relation between objects in the Scene."""
    rec, _ = real_result
    assert len(rec.instances) >= 2, "a Query with an Anchor needs two Instances"
    text = os.environ.get(SMOKE_SPATIAL_QUERY_ENV, DEFAULT_SPATIAL_QUERY)
    parsed, matches = _answer(text, rec, query_parser, encoder, caption_encoder)
    assert parsed.constraints, f"the parser found no Relation in {text!r}"
    print(f"{text!r}: {[(m.instance_id, round(m.score, 3), m.anchors) for m in matches]}")
    assert 1 <= len(matches) <= _configured_settings().query_top_k
    assert all(m.anchors for m in matches)


def test_a_query_without_a_relation_returns_no_anchors(
    real_result, query_parser, encoder, caption_encoder
):
    rec, _ = real_result
    parsed, matches = _answer("a chair", rec, query_parser, encoder, caption_encoder)
    assert parsed.constraints == ()
    assert matches
    assert all(m.anchors == () for m in matches)
