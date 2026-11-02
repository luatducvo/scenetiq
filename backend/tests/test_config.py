from scene_core.config import ComputeConfig


def test_compute_config_expands_environment_and_tolerates_unset_values(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPUTE_TOKEN", "")
    monkeypatch.setenv("COLAB_URL", "https://colab.example")
    config = ComputeConfig.load("backend/config/compute.yaml")
    assert config.targets["colab"].url == "https://colab.example"
    assert config.targets["colab"].token == ""
    assert config.targets["colab"].chunk_size == 100000000
    assert config.targets["colab"].timeout == 300.0
    assert config.targets["local_gpu"].chunk_size == 25000000
    assert config.targets["local_gpu"].timeout == 29.0
    monkeypatch.delenv("COLAB_URL")
    assert ComputeConfig.load("backend/config/compute.yaml").targets["colab"].url is None
    path = tmp_path / "compute.yaml"
    path.write_text(
        "targets:\n  x:\n    backend: http\n    url: ${MISSING_URL}\n    headers: {Authorization: Bearer ${MISSING_KEY}}\n"
    )
    target = ComputeConfig.load(path).targets["x"]
    assert target.url is None and target.headers == {"Authorization": "Bearer"}


def test_pipeline_targets_come_from_environment(monkeypatch):
    from scene_core.config import PipelineConfig

    monkeypatch.setenv("GPU_TARGET", "colab")
    real = PipelineConfig.load("backend/config/pipeline.yaml")
    assert real.target == "colab"
    assert real.stages["ingest"].params["max_keyframes"] in (80, 200)
    assert real.stages["ingest"].params["quality_filter"] is False
    assert PipelineConfig.load("backend/config/pipeline.fake-gpu.yaml").target == "colab"
    assert PipelineConfig.load("backend/config/pipeline.fake.yaml").target == "cpu"
