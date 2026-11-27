import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scenetiq.config import Settings
from scenetiq.pipeline.model_pipeline import ModelPipeline
from scenetiq.pipeline.models.qwen3_embedding import Qwen3TextEmbedder
from scenetiq.pipeline.models.qwen3_vl import QwenVLCaptioner
from scenetiq.pipeline.models.siglip2 import SigLIP2Encoder
from scenetiq.pipeline.stages.reconstructing import DA3Reconstructor
from scenetiq.query.parser import VLMQueryParser
from scenetiq.server import build_components
from tests.stubs import server as stub_server


def test_model_pipeline_gets_every_stage_setting_without_loading_a_model(tmp_path: Path):
    settings = Settings.from_overrides(
        "tok",
        {
            "storage_root": str(tmp_path),
            "da3": {"process_res": 392},
            "fusion": {"voxel_size": 0.05},
            "superpoints": {"min_verts": 40},
            "sam2": {"points_per_side": 16},
            "merging": {"min_instance_points": 80},
            "embedding": {"views_per_instance": 3},
            "siglip": {"batch_size": 8},
            "caption_embedding": {"batch_size": 4},
            "captioning": {"max_new_tokens": 200},
        },
    )
    components = build_components(settings)
    pipeline, encoder, caption_encoder = (
        components.pipeline,
        components.text_encoder,
        components.caption_encoder,
    )
    assert isinstance(pipeline, ModelPipeline)
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
    assert isinstance(pipeline.captioner, QwenVLCaptioner)
    assert pipeline.captioner.options.max_new_tokens == 200


def test_the_backend_parses_queries_on_the_captioning_model_without_loading_it(
    tmp_path: Path,
):
    settings = Settings.from_overrides(
        "tok",
        {
            "storage_root": str(tmp_path),
            "captioning": {"max_new_tokens": 200},
            "query_parser": {"max_new_tokens": 512},
        },
    )
    components = build_components(settings)
    parser = components.query_parser
    assert isinstance(parser, VLMQueryParser)
    assert isinstance(components.pipeline, ModelPipeline)
    captioner = components.pipeline.captioner
    assert isinstance(captioner, QwenVLCaptioner)
    assert parser.model is captioner
    assert parser.options.max_new_tokens == 512
    assert captioner.options.max_new_tokens == 200


def test_the_stub_backend_starts_from_the_environment_without_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SCENETIQ_TOKEN", "tok")
    monkeypatch.setenv("SCENETIQ_CONFIG", json.dumps({"storage_root": str(tmp_path)}))
    with TestClient(stub_server.build_app()) as client:
        resp = client.get("/scenes", headers={"Authorization": "Bearer tok"})
    assert resp.status_code == 200
    assert resp.json() == []
