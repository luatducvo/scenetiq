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
