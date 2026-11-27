import threading
from collections.abc import Callable

from fastapi.testclient import TestClient

from scenetiq.fakes import FakePipeline

from .conftest import AUTH, upload


def test_idle_backend_is_healthy_with_no_job_running(client: TestClient):
    resp = client.get("/health", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "job_running": False}


def test_health_needs_the_token(client: TestClient):
    assert client.get("/health").status_code == 401


def test_health_reports_a_running_job_until_it_ends(make_client: Callable[..., TestClient]):
    gate = threading.Event()
    client = make_client(pipeline=FakePipeline(gate=gate))
    upload(client)

    assert client.get("/health", headers=AUTH).json()["job_running"] is True

    gate.set()
    client.app.state.runner.join(timeout=5)  # type: ignore[attr-defined]
    assert client.get("/health", headers=AUTH).json()["job_running"] is False
