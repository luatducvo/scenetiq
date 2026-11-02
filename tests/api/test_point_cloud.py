import threading
from collections.abc import Callable

import numpy as np
from fastapi.testclient import TestClient

from scenetiq.artifacts import decode_ply
from scenetiq.fakes import FakePipeline

from .conftest import AUTH, upload, wait_for_job


def test_point_cloud_of_succeeded_scene_is_binary_ply_with_every_point(client: TestClient):
    body = upload(client)
    wait_for_job(client, body["job_id"])

    resp = client.get(f"/scenes/{body['scene_id']}/points", headers=AUTH)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.content.startswith(b"ply\nformat binary_little_endian 1.0\n")
    points, colors = decode_ply(resp.content)
    # Known values of the fake Reconstruction.
    assert points.shape == (12, 3)
    np.testing.assert_allclose(points[3], [0.5, 1.0, 1.0])
    np.testing.assert_allclose(points[11], [9.0, 9.0, 0.0])
    assert colors.tolist()[0] == [200, 30, 30]


def test_point_cloud_of_unknown_scene_is_404(client: TestClient):
    assert client.get("/scenes/nope/points", headers=AUTH).status_code == 404


def test_point_cloud_before_job_succeeds_is_409(make_client: Callable[..., TestClient]):
    gate = threading.Event()
    client = make_client(pipeline=FakePipeline(gate=gate))
    body = upload(client)

    resp = client.get(f"/scenes/{body['scene_id']}/points", headers=AUTH)

    assert resp.status_code == 409
    assert "not ready" in resp.json()["detail"]
    gate.set()
    wait_for_job(client, body["job_id"])


def test_point_cloud_of_failed_scene_is_409(make_client: Callable[..., TestClient]):
    client = make_client(pipeline=FakePipeline(error=RuntimeError("boom")))
    body = upload(client)
    wait_for_job(client, body["job_id"])
    assert client.get(f"/scenes/{body['scene_id']}/points", headers=AUTH).status_code == 409
