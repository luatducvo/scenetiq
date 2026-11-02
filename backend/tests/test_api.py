import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from scene_api.main import create_app
from scene_core.config import ComputeConfig, PipelineConfig, TargetConfig
from scene_core.membership import ObjectMembership
from scene_core.results import EMBEDDING_DIM
from scene_target import open_target_workspace
from scene_worker.runner import Worker

pytestmark = pytest.mark.db


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(database_url=os.environ["TEST_DATABASE_URL"], data_dir=tmp_path)
    ) as client:
        yield client


def worker(client):
    return Worker(
        os.environ["TEST_DATABASE_URL"],
        client.app.state.data_dir,
        PipelineConfig.load(Path(__file__).parents[1] / "config/pipeline.fake.yaml"),
        ComputeConfig(),
    )


def uploaded(client):
    scene = client.post("/api/v1/scenes", json={"name": "Test room"}).json()["id"]
    prefix = f"/api/v1/scenes/{scene}"
    assert client.get(prefix).json()["status"] == "created"
    assert any(s["id"] == scene for s in client.get("/api/v1/scenes").json())
    for name in ("one.jpg", "two.jpg"):
        assert (
            client.post(
                prefix + "/uploads", files={"files": (name, b"synthetic", "image/jpeg")}
            ).status_code
            == 200
        )
    assert client.get(prefix).json()["status"] == "uploaded"
    return prefix


def test_scene_upload_delete(client):
    prefix = uploaded(client)
    raw = sorted(
        p.name
        for p in (
            client.app.state.data_dir / "target" / "scenes" / prefix.rsplit("/", 1)[1] / "raw"
        ).iterdir()
    )
    assert [name.rsplit("-", 1)[0] for name in raw] == ["one", "two"]
    assert (
        client.post(
            prefix + "/uploads", files={"files": ("bad.txt", b"x", "text/plain")}
        ).status_code
        == 415
    )
    assert client.delete(prefix).status_code == 204
    assert client.get(prefix).json()["error"]["code"] == "SCENE_NOT_FOUND"


def test_pipeline_publish_and_rerun(client):
    prefix = uploaded(client)
    runner = worker(client)
    assert client.post(prefix + "/process", json={}).status_code == 202
    for method, path, kwargs in (
        (client.post, "/process", {"json": {}}),
        (client.delete, "", {}),
        (client.post, "/uploads", {"files": {"files": ("two.jpg", b"fake", "image/jpeg")}}),
    ):
        response = method(prefix + path, **kwargs)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "SCENE_BUSY"
    assert runner.run_once()
    detail = client.get(prefix).json()
    assert detail["status"] == "ready", detail
    assert [(s["stage"], s["status"]) for s in detail["stage_runs"]] == [
        (stage, "done")
        for stage in ("ingest", "recon", "align", "mesh", "segment", "lift", "objects", "embed")
    ]
    objects = client.get(prefix + "/objects").json()
    assert [o["id"] for o in objects] == ["chair-1", "table-1"]
    assert all("attributes" not in o for o in objects)
    last = next(s for s in detail["stage_runs"] if s["stage"] == "objects")
    assert last["manifest"] == {"num_objects": 2}
    assert objects[0]["best_views"][0]["frame_id"] == "0000"
    assert "image_url" not in objects[0]["best_views"][0]
    assert client.get(detail["assets"]["mesh_url"]).content.startswith(b"glTF")
    assert detail["assets"]["membership_url"].endswith("/objects/membership.bin")
    membership = ObjectMembership.decode(client.get(detail["assets"]["membership_url"]).content)
    assert membership.ids == ["chair-1", "table-1"]
    assert all(len(membership.points(i)) for i in range(2))
    # A ready Scene is searched and inspected; it cannot be asked questions.
    assert (
        client.post(prefix + "/queries", json={"question": "How many chairs?"}).status_code == 404
    )
    # Frame images stay on the Target.
    frame = detail["assets"]["mesh_url"].replace("mesh/mesh.glb", "recon/rgb/0000.png")
    assert client.get(frame).status_code == 404
    for private in ("raw/private.jpg", "keyframes/0000.jpg", "instances/instances.json"):
        assert client.get(prefix + "/files/" + private).status_code == 404
    for private in ("instances/points/chair-1.npy", "objects/objects.json"):
        assert client.get(frame.replace("recon/rgb/0000.png", private)).status_code == 404
    assert (
        client.post(
            prefix + "/uploads", files={"files": ("two.jpg", b"fake", "image/jpeg")}
        ).json()["error"]["code"]
        == "SCENE_LOCKED"
    )
    client.post(prefix + "/process", json={})
    assert client.get(prefix).json()["stage_runs"] == []  # the previous Job's Stage Runs are gone
    runner.run_once()
    rerun = client.get(prefix).json()
    assert [s["status"] for s in rerun["stage_runs"]] == ["done"] * 8
    # A new publication gets new URLs; the previous one is removed.
    assert rerun["assets"]["mesh_url"] != detail["assets"]["mesh_url"]
    assert client.get(detail["assets"]["mesh_url"]).status_code == 404
    assert client.get(rerun["assets"]["mesh_url"]).content.startswith(b"glTF")
    assert len(client.get(prefix + "/objects").json()) == 2


def test_progress_replay_and_search(client):
    prefix = uploaded(client)
    client.post(prefix + "/process", json={})
    worker(client).run_once()
    response = client.get(prefix + "/events")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: stage") == 8
    assert '"ready"' in response.text
    assert (
        client.get(prefix + "/search", params={"q": "cái ghế"}).json()[0]["object"]["id"]
        == "chair-1"
    )
    assert client.get(prefix + "/search", params={"q": "unknown"}).json() == []
    assert client.get(prefix + "/search", params={"q": ""}).json() == []


def test_search_reranks_the_nearest_objects_and_every_label_match(client):
    import math

    from psycopg.types.json import Jsonb

    prefix = uploaded(client)
    client.post(prefix + "/process", json={})
    worker(client).run_once()
    scene = prefix.rsplit("/", 1)[1]

    def embedding(similarity):
        # The fake embeds "chair" as the first axis, so this is the cosine similarity to the query.
        vector = [0.0] * EMBEDDING_DIM
        vector[0], vector[2] = similarity, math.sqrt(1 - similarity**2)
        return str(vector)

    # The fake reranker scores 1 for a Description about chairs only, 0.71 for one that also
    # mentions a table, and 0 for one about neither.
    with client.app.state.store.connect() as conn:
        for object_id, label, similarity, description in (
            ("near-a", "stool", 0.9, "A stool next to a table, used as a chair."),
            ("near-b", "stool", 0.8, "A stool to sit on."),
            ("near-c", "stool", 0.7, "A low wooden stool."),
            ("far", "stool", 0.5, "A stool shaped like a chair."),
            ("armchair-1", "armchair", 0.0, "A soft armchair beside a table."),
        ):
            conn.execute(
                "INSERT INTO objects(scene_id,id,label,box,embedding,description)"
                " VALUES(%s,%s,%s,%s,%s::vector,%s)",
                (
                    scene,
                    object_id,
                    label,
                    Jsonb({"center": [0, 0, 0], "size": [1, 1, 1], "yaw": 0}),
                    embedding(similarity),
                    description,
                ),
            )
    client.app.state.search_candidates = 3

    def search():
        return client.get(prefix + "/search", params={"q": "chair"}).json()

    # The 3 nearest are chair-1, near-a and near-b; armchair-1 joins them by label. near-b's
    # Description outranks near-a's, and far is never judged although its Description fits.
    hits = search()
    assert [hit["object"]["id"] for hit in hits] == ["chair-1", "armchair-1", "near-b"]
    assert hits[0]["score"] == pytest.approx(2.0)
    assert hits[1]["score"] == pytest.approx(1 + math.sqrt(0.5))
    assert hits[1]["raw_similarity"] == pytest.approx(math.sqrt(0.5))
    assert hits[2]["score"] == hits[2]["raw_similarity"] == pytest.approx(1.0)
    client.app.state.search_margin = 1.0
    assert [hit["object"]["id"] for hit in search()] == [
        "chair-1",
        "armchair-1",
        "near-b",
        "near-a",
    ]


def remote(tmp_path, compute_client, **overrides):
    """A TargetWorkspace on an HTTP Target served by an in-memory scene-compute app."""
    target = TargetConfig(
        **{
            "backend": "http",
            "url": "http://testserver",
            "chunk_size": 512,
            "poll_interval": 0.001,
            **overrides,
        }
    )
    pipeline = PipelineConfig.load(
        Path(__file__).parents[1] / "config/pipeline.fake.yaml"
    ).model_copy(update={"target": "remote"})
    return open_target_workspace(
        tmp_path / "local",
        pipeline,
        ComputeConfig(targets={"remote": target}),
        client=compute_client,
    )


def test_a_failed_upload_keeps_none_of_its_files_on_the_target(tmp_path):
    import httpx
    from scene_compute.server import create_app as create_compute_app

    class BrokenUpload(TestClient):
        def request(self, method, url, **kwargs):
            if method == "PUT" and "/raw/broken-" in str(url):
                raise httpx.ConnectError("tunnel closed")
            return super().request(method, url, **kwargs)

    with BrokenUpload(create_compute_app(tmp_path / "remote", chunk_bytes=512)) as compute:
        target = remote(tmp_path, compute, startup_timeout=0.2)
        with TestClient(
            create_app(
                database_url=os.environ["TEST_DATABASE_URL"],
                data_dir=tmp_path / "local",
                target=target,
            )
        ) as client:
            scene = client.post("/api/v1/scenes", json={"name": "Test room"}).json()["id"]
            prefix = f"/api/v1/scenes/{scene}"
            files = [
                ("files", (f"{name}.jpg", name.encode() * 300, "image/jpeg"))
                for name in ("first", "broken", "last")
            ]
            response = client.post(prefix + "/uploads", files=files)
            assert response.json()["error"]["code"] == "COMPUTE_UNAVAILABLE"
            assert target.list_inputs(scene) == []


def test_delete_removes_the_target_workspace_and_tolerates_an_unreachable_target(tmp_path):
    from scene_compute.server import create_app as create_compute_app

    with TestClient(create_compute_app(tmp_path / "remote", chunk_bytes=512)) as compute:
        target = remote(tmp_path, compute)
        with TestClient(
            create_app(
                database_url=os.environ["TEST_DATABASE_URL"],
                data_dir=tmp_path / "local",
                target=target,
            )
        ) as client:
            prefix = uploaded(client)
            scene = prefix.rsplit("/", 1)[1]
            assert (tmp_path / "remote" / "scenes" / scene / "raw").is_dir()
            assert client.delete(prefix).status_code == 204
            assert not (tmp_path / "remote" / "scenes" / scene).exists()
    down = PipelineConfig.load(Path(__file__).parents[1] / "config/pipeline.fake.yaml").model_copy(
        update={"target": "down"}
    )
    compute = ComputeConfig(
        targets={
            "down": TargetConfig(
                backend="http", url="http://127.0.0.1:9", startup_timeout=0.2, poll_interval=0.05
            )
        }
    )
    with TestClient(
        create_app(
            database_url=os.environ["TEST_DATABASE_URL"],
            data_dir=tmp_path,
            pipeline=down,
            compute=compute,
        )
    ) as client:
        scene = client.post("/api/v1/scenes", json={"name": "Unreachable"}).json()["id"]
        assert client.delete(f"/api/v1/scenes/{scene}").status_code == 204


def test_worker_refuses_to_start_without_configured_gpu_targets(tmp_path, monkeypatch):
    for name in ("GPU_TARGET", "COLAB_URL"):
        monkeypatch.delenv(name, raising=False)
    config = Path(__file__).parents[1] / "config"

    def start():
        return Worker(
            os.environ["TEST_DATABASE_URL"],
            tmp_path,
            PipelineConfig.load(config / "pipeline.fake-gpu.yaml"),
            ComputeConfig.load(config / "compute.yaml"),
        )

    with pytest.raises(ValueError, match="GPU_TARGET"):
        start()
    monkeypatch.setenv("GPU_TARGET", "colab")
    with pytest.raises(ValueError, match=r"colab.*URL"):
        start()


def test_whole_pipeline_runs_on_the_gpu_host_and_returns_only_results(tmp_path):
    from scene_compute.server import create_app as create_compute_app

    local, remote_root = tmp_path / "local", tmp_path / "remote"
    with TestClient(create_compute_app(remote_root, chunk_bytes=512)) as compute:
        target = remote(tmp_path, compute)
        runner = Worker(os.environ["TEST_DATABASE_URL"], local, target=target)
        with TestClient(
            create_app(database_url=os.environ["TEST_DATABASE_URL"], data_dir=local, target=target)
        ) as client:
            prefix = uploaded(client)
            scene = prefix.rsplit("/", 1)[1]
            assert not (local / "scenes" / scene).exists()
            assert len(list((remote_root / "scenes" / scene / "raw").iterdir())) == 2
            (remote_root / "scenes" / scene / "segment").mkdir(parents=True)
            (remote_root / "scenes" / scene / "segment" / "stale.npz").write_bytes(b"old")
            for _ in range(2):
                client.post(prefix + "/process", json={})
                runner.run_once()
                detail = client.get(prefix).json()
                assert detail["status"] == "ready", detail
                assert [s["status"] for s in detail["stage_runs"]] == ["done"] * 8
            assert not (remote_root / "scenes" / scene / "segment" / "stale.npz").exists()
            assert len(list((remote_root / "scenes" / scene / "raw").iterdir())) == 2
            [version] = list((local / "scenes" / scene).iterdir())
            assert f"/publications/{version.name}/" in detail["assets"]["mesh_url"]
            workspace = version
            assert sorted(p.name for p in workspace.iterdir()) == [
                "embeddings",
                "mesh",
                "objects",
            ]
            assert_failed_reprocess_keeps_results(client, runner, prefix, detail)


def test_a_failed_reprocess_keeps_results_on_the_in_process_target(client):
    prefix = uploaded(client)
    runner = worker(client)
    client.post(prefix + "/process", json={})
    runner.run_once()
    detail = client.get(prefix).json()
    assert detail["status"] == "ready"
    assert_failed_reprocess_keeps_results(client, runner, prefix, detail)
    # Search still reranks the earlier Objects by their Descriptions.
    hits = client.get(prefix + "/search", params={"q": "chair"}).json()
    assert hits[0]["object"]["id"] == "chair-1" and hits[0]["raw_similarity"] == pytest.approx(1.0)


def assert_failed_reprocess_keeps_results(client, runner, prefix, detail):
    """A Job failing at its last Stage leaves the earlier mesh and Objects in place."""
    mesh = client.get(detail["assets"]["mesh_url"]).content
    runner.target.pipeline.stages["embed"].params = {"fail": "deliberate failure"}
    client.post(prefix + "/process", json={})
    runner.run_once()
    failed = client.get(prefix).json()
    assert failed["status"] == "failed" and failed["failure"]["code"] == "FAKE_STAGE_FAILED"
    assert failed["assets"]["mesh_url"] and client.get(failed["assets"]["mesh_url"]).content == mesh
    objects = client.get(prefix + "/objects").json()
    assert len(objects) == 2
    runner.target.pipeline.stages["embed"].params = {}


def test_search_reports_unavailable_compute_target(client, monkeypatch):
    prefix = uploaded(client)
    client.post(prefix + "/process", json={})
    worker(client).run_once()
    monkeypatch.delenv("GPU_TARGET", raising=False)
    compute = ComputeConfig.model_validate(
        {
            "targets": {
                "cpu": {},
                "down": {
                    "backend": "http",
                    "url": "http://127.0.0.1:9",
                    "startup_timeout": 0.2,
                    "poll_interval": 0.1,
                },
            }
        }
    )
    for target in (None, "down"):
        if target:
            monkeypatch.setenv("GPU_TARGET", target)
        pipeline = PipelineConfig.load(Path(__file__).parents[1] / "config/pipeline.fake-gpu.yaml")
        with TestClient(
            create_app(
                database_url=os.environ["TEST_DATABASE_URL"],
                data_dir=client.app.state.data_dir,
                pipeline=pipeline,
                compute=compute,
            )
        ) as other:
            response = other.get(prefix + "/search", params={"q": "chair"})
        assert response.status_code == 503, target
        assert response.json()["error"]["code"] == "COMPUTE_UNAVAILABLE"


def test_worker_start_fails_jobs_interrupted_by_a_restart(client):
    prefix = uploaded(client)
    runner = worker(client)
    assert client.post(prefix + "/process", json={}).status_code == 202
    job = runner.store.claim()
    runner.store.stage_started(job, "embed", "embed.fake")  # a worker died mid-Stage
    worker(client).recover_interrupted()
    detail = client.get(prefix).json()
    assert detail["status"] == "failed"
    assert (
        detail["failure"]["code"] == "WORKER_INTERRUPTED" and detail["failure"]["stage"] == "embed"
    )
    assert next(s for s in detail["stage_runs"] if s["stage"] == "embed")["status"] == "failed"
    assert client.post(prefix + "/process", json={}).status_code == 202
    assert runner.run_once() and client.get(prefix).json()["status"] == "ready"


def test_stage_progress_is_throttled_stored_and_replayed(client):
    prefix = uploaded(client)
    scene = prefix.rsplit("/", 1)[1]
    runner = worker(client)
    runner.target.pipeline.stages["mesh"].params["progress_steps"] = 500
    client.post(prefix + "/process", json={})
    runner.run_once()
    with runner.store.connect() as conn:
        events = [
            r["data"]
            for r in conn.execute(
                "SELECT data FROM scene_events WHERE scene_id=%s AND event='progress' ORDER BY id",
                (scene,),
            ).fetchall()
        ]
    mesh = [e for e in events if e["stage"] == "mesh"]
    assert [(e["done"], e["total"], e["unit"]) for e in mesh] == [
        (0, 500, "step"),
        (500, 500, "step"),
    ]
    runs = {s["stage"]: s for s in client.get(prefix).json()["stage_runs"]}
    assert runs["mesh"]["progress"] == {"done": 500, "total": 500, "unit": "step"}
    assert all(s["started_at"] for s in runs.values())
    import json

    replay = [
        json.loads(line[6:])
        for line in client.get(prefix + "/events").text.splitlines()
        if line.startswith("data: ")
    ]
    mesh_replay = [e for e in replay if e.get("stage") == "mesh"]
    assert (
        [e["type"] for e in mesh_replay] == ["stage", "progress"]
        and mesh_replay[0]["started_at"]
        and "progress" not in mesh_replay[0]
    )
    assert (mesh_replay[1]["done"], mesh_replay[1]["total"], mesh_replay[1]["unit"]) == (
        500,
        500,
        "step",
    )


def test_unpublished_scenes_do_not_serve_flat_result_folders(client):
    scene = client.post("/api/v1/scenes", json={"name": "Old"}).json()["id"]
    flat_results = client.app.state.data_dir / "scenes" / scene / "mesh"
    flat_results.mkdir(parents=True)
    (flat_results / "mesh.glb").write_bytes(b"glTF-old")
    assert client.get(f"/api/v1/scenes/{scene}").json()["assets"] == {
        "mesh_url": None,
        "points_url": None,
        "membership_url": None,
    }
    assert (
        client.get(f"/api/v1/scenes/{scene}/publications/legacy/files/mesh/mesh.glb").status_code
        == 404
    )
    assert (
        client.get(
            f"/api/v1/scenes/{scene}/publications/..%2F..%2Fraw/files/mesh/mesh.glb"
        ).status_code
        == 404
    )
