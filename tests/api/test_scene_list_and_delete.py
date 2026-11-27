import threading
from collections.abc import Callable

from fastapi.testclient import TestClient

from scenetiq.fakes import FakePipeline

from .conftest import AUTH, upload, wait_for_job


def test_scenes_are_listed_newest_first_with_status(client: TestClient):
    first = upload(client, name="Kitchen")
    wait_for_job(client, first["job_id"])
    second = upload(client, name="Office")
    wait_for_job(client, second["job_id"])

    scenes = client.get("/scenes", headers=AUTH).json()

    assert [s["name"] for s in scenes] == ["Office", "Kitchen"]
    assert {s["status"] for s in scenes} == {"succeeded"}
    assert all(s["created_at"] for s in scenes)


def test_deleted_scene_is_gone_everywhere(client: TestClient):
    body = upload(client)
    wait_for_job(client, body["job_id"])

    resp = client.delete(f"/scenes/{body['scene_id']}", headers=AUTH)

    assert resp.status_code == 204
    assert client.get("/scenes", headers=AUTH).json() == []
    assert client.get(f"/scenes/{body['scene_id']}/points", headers=AUTH).status_code == 404
    assert client.get(f"/jobs/{body['job_id']}", headers=AUTH).status_code == 404


def test_deleting_unknown_scene_is_404(client: TestClient):
    assert client.delete("/scenes/nope", headers=AUTH).status_code == 404


def test_scene_whose_job_is_running_cannot_be_deleted(make_client: Callable[..., TestClient]):
    gate = threading.Event()
    client = make_client(pipeline=FakePipeline(gate=gate))
    body = upload(client)

    resp = client.delete(f"/scenes/{body['scene_id']}", headers=AUTH)

    assert resp.status_code == 409
    gate.set()
    wait_for_job(client, body["job_id"])
    assert client.delete(f"/scenes/{body['scene_id']}", headers=AUTH).status_code == 204


def test_failed_scene_can_be_deleted(make_client: Callable[..., TestClient]):
    client = make_client(pipeline=FakePipeline(error=RuntimeError("boom")))
    body = upload(client)
    wait_for_job(client, body["job_id"])
    assert client.delete(f"/scenes/{body['scene_id']}", headers=AUTH).status_code == 204


def test_a_finished_scene_lists_its_instance_count_and_when_it_ran(client: TestClient):
    job = wait_for_job(client, upload(client)["job_id"])

    [scene] = client.get("/scenes", headers=AUTH).json()

    # The fake Reconstruction has three Instances: a chair, a table and a lamp.
    assert scene["instance_count"] == 3
    assert (scene["started_at"], scene["finished_at"]) == (job["started_at"], job["finished_at"])


def test_a_failed_scene_lists_no_instance_count(make_client: Callable[..., TestClient]):
    client = make_client(pipeline=FakePipeline(error=RuntimeError("out of memory")))
    wait_for_job(client, upload(client)["job_id"])

    [scene] = client.get("/scenes", headers=AUTH).json()

    assert scene["instance_count"] is None
    assert scene["finished_at"] is not None
