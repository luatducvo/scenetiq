"""TargetWorkspace through its interface, on both adapters: in-process and HTTP to an in-memory scene-compute."""

import io
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from scene_compute.server import create_app as create_compute_app
from scene_core.config import ComputeConfig, PipelineConfig, TargetConfig
from scene_core.contracts import STAGES, SceneWorkspace
from scene_core.errors import SceneError
from scene_core.results import EMBEDDING_DIM, SceneResults
from scene_target import open_target_workspace

CONFIG = Path(__file__).parents[1] / "config"


def remote_target(**overrides):
    return TargetConfig(
        **{
            "backend": "http",
            "url": "http://testserver",
            "chunk_size": 512,
            "poll_interval": 0.001,
            **overrides,
        }
    )


@pytest.fixture(params=["inprocess", "http"])
def target(request, tmp_path):
    """A TargetWorkspace publishing into tmp_path/local, and the folder holding its Target Workspaces."""
    pipeline, local = PipelineConfig.load(CONFIG / "pipeline.fake.yaml"), tmp_path / "local"
    if request.param == "inprocess":
        yield open_target_workspace(local, pipeline, ComputeConfig()), local / "target"
        return
    # Small client chunks exercise chunked transfer; the server limit still fits one query embedding.
    with TestClient(create_compute_app(tmp_path / "remote", chunk_bytes=64_000)) as client:
        compute = ComputeConfig(targets={"remote": remote_target()})
        yield (
            open_target_workspace(
                local, pipeline.model_copy(update={"target": "remote"}), compute, client=client
            ),
            tmp_path / "remote",
        )


def processed(target, scene="room"):
    target.put_input(scene, "one.jpg", io.BytesIO(b"synthetic"))
    for stage in STAGES:
        target.run_stage(scene, stage)
    return scene


def published(target, version, scene="room"):
    return SceneWorkspace(target.data_dir, scene, version)


def test_inputs_live_only_on_the_target(target):
    workspace, root = target
    workspace.put_input("room", "a.jpg", io.BytesIO(b"x" * 1300))
    workspace.put_input("room", "b.jpg", io.BytesIO(b"y"))
    assert workspace.list_inputs("room") == ["a.jpg", "b.jpg"]
    assert (root / "scenes/room/raw/a.jpg").read_bytes() == b"x" * 1300
    assert not (workspace.data_dir / "scenes").exists()
    workspace.remove_inputs("room", ["b.jpg"])
    assert workspace.list_inputs("room") == ["a.jpg"]


def test_stages_run_on_the_target_and_only_results_are_published(target):
    workspace, root = target
    workspace.pipeline.stages["mesh"].params["progress_steps"] = 3
    workspace.put_input("room", "one.jpg", io.BytesIO(b"synthetic"))
    reports = []
    manifests = {
        stage: workspace.run_stage("room", stage, on_progress=reports.append) for stage in STAGES
    }
    assert manifests["ingest"]["num_keyframes"] == 1
    assert (
        reports[-1] == {"done": 1, "total": 1, "unit": "step"}
        and {"done": 3, "total": 3, "unit": "step"} in reports
    )
    assert (root / "scenes/room/keyframes").is_dir() and not (
        workspace.data_dir / "scenes"
    ).exists()
    staged = []
    workspace.replace_results(
        "room", "job-1", lambda results: staged.append(SceneResults(results).load())
    )
    assert [o.id for o in staged[0].objects] == ["chair-1", "table-1"]
    results = published(workspace, "job-1")
    assert [p.name for p in results.root.parent.iterdir()] == ["job-1"]
    assert sorted(p.name for p in results.root.iterdir()) == [
        "embeddings",
        "mesh",
        "objects",
    ]
    assert (
        SceneResults(results).assets()["mesh"]
        and not (workspace.data_dir / "staging/scenes/room").exists()
    )


def test_a_failed_publication_leaves_the_earlier_results_untouched(target):
    workspace, _ = target
    processed(workspace)
    workspace.replace_results("room", "job-1", lambda results: None)
    mesh = published(workspace, "job-1").path("mesh/mesh.glb").read_bytes()
    workspace.clear_outputs("room")
    processed(workspace)

    def publish(results):
        # The new version is complete before the database commits; a failure then must remove it.
        assert results.path("mesh/mesh.glb").is_file()
        raise SceneError("RESULTS_INVALID", "deliberate failure")

    with pytest.raises(SceneError, match="deliberate failure"):
        workspace.replace_results("room", "job-2", publish)
    assert published(workspace, "job-1").path("mesh/mesh.glb").read_bytes() == mesh
    assert [p.name for p in published(workspace, "job-1").root.parent.iterdir()] == ["job-1"]
    assert not (workspace.data_dir / "staging/scenes/room").exists()


def test_a_new_publication_replaces_older_versions(target):
    workspace, _ = target
    processed(workspace)
    workspace.replace_results("room", "job-1", lambda results: None)
    workspace.replace_results("room", "job-2", lambda results: None)
    assert [p.name for p in published(workspace, "job-2").root.parent.iterdir()] == ["job-2"]


def test_clearing_outputs_keeps_the_uploaded_inputs(target):
    workspace, root = target
    processed(workspace)
    workspace.clear_outputs("room")
    assert sorted(
        p.relative_to(root / "scenes/room").as_posix()
        for p in (root / "scenes/room").rglob("*")
        if p.is_file()
    ) == ["raw/one.jpg"]


def test_stage_failures_carry_the_algorithms_code_and_details_on_every_adapter(target):
    workspace, _ = target
    workspace.put_input("room", "one.jpg", io.BytesIO(b"synthetic"))
    workspace.pipeline.stages["ingest"].params.update(
        fail="Too few sharp Frames",
        fail_code="NOT_ENOUGH_KEYFRAMES",
        fail_details={"rejection_counts": {"blur": 3}},
    )
    with pytest.raises(SceneError) as error:
        workspace.run_stage("room", "ingest")
    assert (error.value.code, error.value.message, error.value.details) == (
        "NOT_ENOUGH_KEYFRAMES",
        "Too few sharp Frames",
        {"rejection_counts": {"blur": 3}},
    )


def test_queries_are_embedded_by_the_embed_algorithm(target):
    workspace, _ = target
    [vector] = workspace.embed_text(["chair"])
    assert len(vector) == EMBEDDING_DIM


def test_descriptions_are_reranked_by_the_embed_algorithm(target):
    workspace, _ = target
    scores = workspace.rerank("chair", ["A chair to sit on.", "A table and a chair.", "A lamp."])
    assert scores == pytest.approx([1.0, 0.5**0.5, 0.0])
    assert workspace.rerank("chair", []) == []


def test_a_rerank_failure_carries_the_algorithms_code(target):
    workspace, _ = target
    with pytest.raises(SceneError) as error:
        workspace.rerank(" ", ["A chair."])
    assert error.value.code == "INVALID_PARAMETERS"


def test_reranking_uses_the_embed_stages_configured_params(target):
    workspace, _ = target
    workspace.pipeline.stages["embed"].params.update(
        fail="Reranker unavailable", fail_code="RERANKER_DOWN", fail_details={"model": "x"}
    )
    with pytest.raises(SceneError) as error:
        workspace.rerank("chair", ["A chair."])
    assert (error.value.code, error.value.message, error.value.details) == (
        "RERANKER_DOWN",
        "Reranker unavailable",
        {"model": "x"},
    )


def test_delete_removes_the_target_workspace(target):
    workspace, root = target
    processed(workspace)
    workspace.delete("room")
    assert not (root / "scenes/room").exists()


class MeetingClient(TestClient):
    """The first two file transfers by `method` each wait until the other has started, so a sequential
    transfer fails. It records every request's method and path."""

    def __init__(self, app, method="GET"):
        super().__init__(app)
        self.method = method
        self.waiting = threading.Semaphore(2)
        self.meeting = threading.Barrier(2, timeout=5)
        self.corrupt = None
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, str(url)))
        transferring = method == self.method and "/files/" in str(url)
        if transferring and self.waiting.acquire(blocking=False):
            self.meeting.wait()
        response = super().request(method, url, **kwargs)
        if transferring and self.corrupt and str(url).endswith(self.corrupt):
            response._content = b"x" * len(response.content)
        return response


def remote_workspace(tmp_path, client, **overrides):
    pipeline = PipelineConfig.load(CONFIG / "pipeline.fake.yaml")
    compute = ComputeConfig(targets={"remote": remote_target(**overrides)})
    return open_target_workspace(
        tmp_path / "local", pipeline.model_copy(update={"target": "remote"}), compute, client=client
    )


def test_a_publication_downloads_files_concurrently(tmp_path):
    with MeetingClient(create_compute_app(tmp_path / "remote", chunk_bytes=64_000)) as client:
        workspace = remote_workspace(tmp_path, client, transfer_concurrency=2)
        processed(workspace)
        workspace.replace_results("room", "job-1", lambda results: None)
    assert SceneResults(published(workspace, "job-1")).assets()["mesh"]


def test_an_upload_waits_for_the_target_once_and_sends_files_concurrently(tmp_path):
    app = create_compute_app(tmp_path / "remote", chunk_bytes=64_000)
    with MeetingClient(app, method="PUT") as client:
        workspace = remote_workspace(tmp_path, client, transfer_concurrency=2)
        workspace.put_inputs(
            "room", {name: io.BytesIO(name.encode() * 400) for name in ("a.jpg", "b.jpg", "c.jpg")}
        )
        assert workspace.list_inputs("room") == ["a.jpg", "b.jpg", "c.jpg"]
    pings = [request for request in client.requests if request == ("GET", "/ping")]
    assert len(pings) == 2  # one for the upload, one for the listing
    assert (tmp_path / "remote/scenes/room/raw/b.jpg").read_bytes() == b"b.jpg" * 400


def test_a_corrupted_download_fails_the_publication(tmp_path):
    with MeetingClient(create_compute_app(tmp_path / "remote", chunk_bytes=64_000)) as client:
        workspace = remote_workspace(tmp_path, client)
        processed(workspace)
        client.corrupt = "mesh/mesh.glb"
        with pytest.raises(SceneError, match=r"mesh/mesh\.glb has a different checksum"):
            workspace.replace_results("room", "job-1", lambda results: None)
    assert not (workspace.data_dir / "scenes/room").exists()
    assert not (workspace.data_dir / "staging/scenes/room").exists()


def test_an_unreachable_target_is_compute_unavailable(tmp_path):
    pipeline = PipelineConfig.load(CONFIG / "pipeline.fake.yaml").model_copy(
        update={"target": "down"}
    )
    compute = ComputeConfig(
        targets={
            "down": remote_target(url="http://127.0.0.1:9", startup_timeout=0.2, poll_interval=0.05)
        }
    )
    workspace = open_target_workspace(tmp_path, pipeline, compute)
    for call in (
        lambda: workspace.list_inputs("room"),
        lambda: workspace.embed_text(["chair"]),
        lambda: workspace.delete("room"),
    ):
        with pytest.raises(SceneError) as error:
            call()
        assert error.value.code == "COMPUTE_UNAVAILABLE" and error.value.status_code == 503


def test_configuration_errors_name_what_to_set(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_TARGET", raising=False)
    pipeline = PipelineConfig.load(CONFIG / "pipeline.yaml")
    compute = ComputeConfig(targets={"cpu": TargetConfig(), "colab": TargetConfig(backend="http")})
    with pytest.raises(ValueError, match="GPU_TARGET"):
        open_target_workspace(tmp_path, pipeline, compute)
    for name, message in (
        ("missing", "unknown compute Target missing"),
        ("colab", "colab has no URL"),
    ):
        with pytest.raises(ValueError, match=message):
            open_target_workspace(tmp_path, pipeline.model_copy(update={"target": name}), compute)
