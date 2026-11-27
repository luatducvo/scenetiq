import os
from pathlib import Path

import pytest

from scenetiq.domain.scene import Capture, Stage, View
from scenetiq.pipeline.model_pipeline import ModelPipeline

from .support import StageRecorder, configured_settings

SMOKE_CAPTURE_ENV = "SCENETIQ_SMOKE_CAPTURE"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="the evaluation suite needs a CUDA GPU and the gpu extra")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def smoke_capture() -> Capture:
    root = os.environ.get(SMOKE_CAPTURE_ENV)
    if not root:
        pytest.skip(f"set {SMOKE_CAPTURE_ENV} to a directory of photos of one static Scene")
    paths = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return Capture(views=[View(path=p) for p in paths])


@pytest.fixture(scope="session")
def encoder():
    from scenetiq.pipeline.models.siglip2 import SigLIP2Encoder

    return SigLIP2Encoder()


@pytest.fixture(scope="session")
def caption_encoder():
    from scenetiq.pipeline.models.qwen3_embedding import Qwen3TextEmbedder

    return Qwen3TextEmbedder()


@pytest.fixture(scope="session")
def captioner():
    from scenetiq.pipeline.models.qwen3_vl import QwenVLCaptioner

    return QwenVLCaptioner()


@pytest.fixture(scope="session")
def reconstruction_result(smoke_capture: Capture, encoder, caption_encoder, captioner):
    stages: list[Stage] = []
    pipeline = ModelPipeline(
        image_encoder=encoder, caption_encoder=caption_encoder, captioner=captioner
    )
    reconstruction = pipeline.run(smoke_capture, StageRecorder(stages))
    return reconstruction, stages


@pytest.fixture(scope="session")
def query_parser(captioner):
    from scenetiq.query.parser import VLMQueryParser

    return VLMQueryParser(captioner, configured_settings().query_parser)
