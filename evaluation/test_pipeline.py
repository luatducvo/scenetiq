import numpy as np
import pytest

from scenetiq.domain.scene import Capture, Stage


def test_reconstruction_is_a_metric_point_cloud(reconstruction_result, smoke_capture: Capture):
    rec, stages = reconstruction_result
    assert stages[:2] == [Stage.RECONSTRUCTING, Stage.FUSING]
    assert len(rec.points) > 1000
    assert rec.points.dtype == np.float32
    assert rec.colors.shape == rec.points.shape
    assert len(rec.instance_ids) == len(rec.points)
    assert len(rec.cameras) == len(smoke_capture.views)
    # A room-scale Scene fits in tens of metres, not thousands or millimetres.
    extent = rec.points.max(axis=0) - rec.points.min(axis=0)
    assert 0.3 < float(extent.max()) < 50.0


def test_model_pipeline_reports_every_stage(reconstruction_result):
    _, stages = reconstruction_result
    assert stages == [
        Stage.RECONSTRUCTING, Stage.FUSING, Stage.ALIGNING, Stage.SUPERPOINTS, Stage.MASKING,
        Stage.MERGING, Stage.EMBEDDING, Stage.CAPTIONING,
    ]  # fmt: skip


def test_reconstruction_has_consistent_instances(reconstruction_result):
    rec, _ = reconstruction_result
    assert len(rec.instances) >= 1
    ids = {i.instance_id for i in rec.instances}
    assert ids == set(np.unique(rec.instance_ids[rec.instance_ids >= 0]).tolist())
    for inst in rec.instances:
        member = rec.points[rec.instance_ids == inst.instance_id]
        np.testing.assert_allclose(inst.centroid, member.mean(axis=0), atol=1e-4)
        assert np.all(inst.bbox_min <= inst.centroid)
        assert np.all(inst.centroid <= inst.bbox_max)


def test_some_instances_get_a_caption_with_description_embeddings(reconstruction_result):
    rec, _ = reconstruction_result
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


def test_cameras_of_upright_photos_end_up_with_up_near_minus_y(reconstruction_result):
    rec, _ = reconstruction_result
    camera_ups = np.stack([-cam.cam_to_world[:3, 1] for cam in rec.cameras]).astype(np.float64)
    mean_up = camera_ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    # cos 45 degrees = 0.7071, against (0, -1, 0).
    assert -mean_up[1] > 0.7071


def test_superpoints_cover_every_point_of_the_fused_mesh(smoke_capture: Capture):
    from scenetiq.pipeline.stages.fusion import fuse
    from scenetiq.pipeline.stages.reconstructing import DA3Reconstructor
    from scenetiq.pipeline.stages.superpoints import mesh_superpoints

    view_data = DA3Reconstructor().reconstruct(smoke_capture.views)
    geometry = fuse(view_data)
    ids = mesh_superpoints(geometry.vertices, geometry.triangles)
    assert len(ids) == len(geometry.vertices)
    assert ids.min() == 0
    assert 10 < ids.max() + 1 < len(geometry.vertices)


def test_only_the_query_models_stay_on_the_gpu_after_a_run(reconstruction_result):
    import torch

    # SigLIP2 (3.74 GB), Qwen3-Embedding-0.6B (1.19 GB) and Qwen3-VL-8B (about 17.5 GB) at
    # bf16 stay, about 22.5 GB; SAM2 (0.90 GB) or DA3 left behind would push this past 23 GB.
    assert torch.cuda.memory_allocated() < 23_000_000_000
