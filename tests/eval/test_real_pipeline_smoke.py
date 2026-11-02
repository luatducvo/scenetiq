import numpy as np
import pytest

from scenetiq.domain import Capture, Stage
from scenetiq.pipeline.real import RealPipeline

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def real_result(smoke_capture: Capture):
    stages: list[Stage] = []
    reconstruction = RealPipeline().run(smoke_capture, stages.append)
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


def test_superpoints_cover_every_point_of_the_fused_mesh(smoke_capture: Capture):
    from scenetiq.pipeline.fusion import fuse
    from scenetiq.pipeline.reconstruct import DA3Reconstructor
    from scenetiq.pipeline.superpoints import mesh_superpoints

    frames = DA3Reconstructor().reconstruct(smoke_capture.views)
    geometry = fuse(frames)
    ids = mesh_superpoints(geometry.vertices, geometry.triangles)
    assert len(ids) == len(geometry.vertices)
    assert ids.min() == 0
    assert 10 < ids.max() + 1 < len(geometry.vertices)
