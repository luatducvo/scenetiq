from pathlib import Path

from scenetiq.config import Settings
from scenetiq.fakes import FakeCaptionEncoder, FakePipeline, FakeTextEncoder
from scenetiq.pipeline.qwen_embedding import Qwen3TextEmbedder
from scenetiq.pipeline.real import RealPipeline
from scenetiq.pipeline.reconstruct import DA3Reconstructor
from scenetiq.pipeline.siglip import SigLIP2Encoder
from scenetiq.server import build_components


def test_real_pipeline_gets_every_stage_setting_without_loading_a_model(tmp_path: Path):
    settings = Settings.from_overrides(
        "tok",
        {
            "storage_root": str(tmp_path),
            "pipeline": "real",
            "da3": {"process_res": 392},
            "fusion": {"voxel_size": 0.05},
            "superpoints": {"min_verts": 40},
            "sam2": {"points_per_side": 16},
            "merging": {"min_instance_points": 80},
            "embedding": {"views_per_instance": 3},
            "siglip": {"batch_size": 8},
            "caption_embedding": {"batch_size": 4},
        },
    )
    pipeline, encoder, caption_encoder = build_components(settings)
    assert isinstance(pipeline, RealPipeline)
    assert isinstance(pipeline.reconstructor, DA3Reconstructor)
    assert isinstance(encoder, SigLIP2Encoder)
    assert pipeline.reconstructor.options.process_res == 392
    assert pipeline.options.fusion.voxel_size == 0.05
    assert pipeline.options.superpoints.min_verts == 40
    assert pipeline.masker.options.points_per_side == 16
    assert pipeline.options.merging.min_instance_points == 80
    assert pipeline.options.embedding.views_per_instance == 3
    assert encoder.options.batch_size == 8
    assert pipeline.image_encoder is encoder
    assert isinstance(caption_encoder, Qwen3TextEmbedder)
    assert caption_encoder.options.batch_size == 4
    assert pipeline.caption_encoder is caption_encoder


def test_fake_pipeline_comes_with_fake_encoders(tmp_path: Path):
    settings = Settings.from_overrides("tok", {"storage_root": str(tmp_path)})
    pipeline, encoder, caption_encoder = build_components(settings)
    assert isinstance(pipeline, FakePipeline)
    assert isinstance(encoder, FakeTextEncoder)
    assert isinstance(caption_encoder, FakeCaptionEncoder)
