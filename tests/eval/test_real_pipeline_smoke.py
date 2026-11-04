import os

import numpy as np
import pytest

from scenetiq.domain import Capture, Stage
from scenetiq.pipeline.real import RealPipeline

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def encoder():
    """One SigLIP2 model shared by the Pipeline and the Query tests, as on the server."""
    from scenetiq.pipeline.siglip import SigLIP2Encoder

    return SigLIP2Encoder()


@pytest.fixture(scope="module")
def real_result(smoke_capture: Capture, encoder):
    stages: list[Stage] = []
    reconstruction = RealPipeline(image_encoder=encoder).run(smoke_capture, stages.append)
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


def test_real_pipeline_reports_stages_up_to_merging(real_result):
    _, stages = real_result
    assert stages[:5] == [
        Stage.RECONSTRUCTING, Stage.FUSING, Stage.SUPERPOINTS, Stage.MASKING, Stage.MERGING,
    ]  # fmt: skip


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


def test_only_siglip2_stays_on_the_gpu_after_a_run(real_result):
    import torch

    # SigLIP2 at bf16 is 3.74 GB; with SAM2 (0.90 GB) left behind it would be 4.64 GB.
    assert torch.cuda.memory_allocated() < 4_500_000_000


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
SMOKE_ABSENT_QUERY_ENV = "SCENETIQ_SMOKE_ABSENT_QUERY"
MIN_POINTS_IN_BOX = 0.5
MIN_POINTS_IN_PHOTO = 0.2


def _configured_threshold() -> float:
    from scenetiq.config import Settings, overrides_from_json

    overrides = overrides_from_json(os.environ.get("SCENETIQ_CONFIG"))
    return Settings.from_overrides("", overrides).match_threshold


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
    real_result, smoke_capture: Capture, encoder
):
    """A clearly visible object must be a confident Match, and the best Match must be it.

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
    assert stages[-1] == Stage.EMBEDDING
    assert all(inst.embedding is not None for inst in rec.instances)

    result = run_query(
        rec.instances, encoder.encode(text), _configured_threshold(), encoder.to_score
    )
    print(f"{text!r}: {[(m.instance_id, round(m.score, 3)) for m in result.matches]}")

    assert not result.low_confidence, f"{text!r} found only a Low-confidence Match"
    best = result.matches[0].instance_id
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


def test_query_for_an_absent_object_is_low_confidence(real_result, encoder):
    """Set SCENETIQ_SMOKE_ABSENT_QUERY to something not in the smoke Capture."""
    from scenetiq.query import run_query

    text = os.environ.get(SMOKE_ABSENT_QUERY_ENV)
    if not text:
        pytest.skip(f"set {SMOKE_ABSENT_QUERY_ENV} to an object absent from the smoke Capture")
    rec, _ = real_result
    result = run_query(
        rec.instances, encoder.encode(text), _configured_threshold(), encoder.to_score
    )
    assert result.low_confidence, f"{text!r} unexpectedly matched {result.matches}"
