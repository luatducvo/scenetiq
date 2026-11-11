"""Cancelling a running job through POST /jobs/{job_id}/cancel."""

import threading
from collections.abc import Callable

from fastapi.testclient import TestClient

from scenetiq.config import Settings
from scenetiq.fakes import FakePipeline

from .conftest import AUTH, start_capture, upload, wait_for_job


def _held_job(make_client: Callable[..., TestClient]) -> tuple[TestClient, threading.Event, dict]:
    """A client whose job stays in its first Stage until the gate is set."""
    gate = threading.Event()
    client = make_client(pipeline=FakePipeline(gate=gate))
    return client, gate, upload(client)


def test_a_cancelled_job_stops_and_deletes_its_scene_data(
    make_client: Callable[..., TestClient], settings: Settings
):
    client, gate, body = _held_job(make_client)

    resp = client.post(f"/jobs/{body['job_id']}/cancel", headers=AUTH)
    gate.set()
    job = wait_for_job(client, body["job_id"])

    assert resp.status_code == 202
    assert job["status"] == "cancelled"
    assert job["finished_at"] is not None
    assert not (settings.storage_root / "scenes" / body["scene_id"]).exists()


def test_a_cancelled_job_frees_the_slot_for_the_next_capture(
    make_client: Callable[..., TestClient],
):
    client, gate, body = _held_job(make_client)

    client.post(f"/jobs/{body['job_id']}/cancel", headers=AUTH)
    gate.set()
    wait_for_job(client, body["job_id"])

    assert client.get("/health", headers=AUTH).json()["job_running"] is False
    start_capture(client)


def test_a_cancelled_scene_is_listed_cancelled_and_can_be_deleted(
    make_client: Callable[..., TestClient],
):
    client, gate, body = _held_job(make_client)
    client.post(f"/jobs/{body['job_id']}/cancel", headers=AUTH)
    gate.set()
    wait_for_job(client, body["job_id"])

    [scene] = client.get("/scenes", headers=AUTH).json()
    points = client.get(f"/scenes/{body['scene_id']}/points", headers=AUTH)
    deleted = client.delete(f"/scenes/{body['scene_id']}", headers=AUTH)

    assert (scene["status"], scene["instance_count"]) == ("cancelled", None)
    assert points.status_code == 409
    assert deleted.status_code == 204


def test_asking_twice_while_the_job_stops_is_accepted(make_client: Callable[..., TestClient]):
    client, gate, body = _held_job(make_client)

    first = client.post(f"/jobs/{body['job_id']}/cancel", headers=AUTH)
    second = client.post(f"/jobs/{body['job_id']}/cancel", headers=AUTH)
    gate.set()

    assert (first.status_code, second.status_code) == (202, 202)
    assert wait_for_job(client, body["job_id"])["status"] == "cancelled"


def test_a_finished_job_cannot_be_cancelled(client: TestClient):
    body = upload(client)
    wait_for_job(client, body["job_id"])

    resp = client.post(f"/jobs/{body['job_id']}/cancel", headers=AUTH)

    assert resp.status_code == 409
    assert client.get(f"/jobs/{body['job_id']}", headers=AUTH).json()["status"] == "succeeded"


def test_cancelling_an_unknown_job_is_404(client: TestClient):
    assert client.post("/jobs/nope/cancel", headers=AUTH).status_code == 404
