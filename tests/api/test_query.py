from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scenetiq.artifacts import decode_ply
from scenetiq.config import Settings
from scenetiq.domain import Reconstruction
from scenetiq.fakes import FakePipeline, FakeTextEncoder, fake_reconstruction

from .conftest import AUTH, TOKEN, upload, wait_for_job


def _ready_scene(client: TestClient) -> str:
    body = upload(client)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"
    return body["scene_id"]


def _query(client: TestClient, scene_id: str, text: str) -> dict:
    resp = client.post(f"/scenes/{scene_id}/query", json={"text": text}, headers=AUTH)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_query_returns_the_matching_instance(client: TestClient):
    scene = _ready_scene(client)
    result = _query(client, scene, "red chair")
    assert result["low_confidence"] is False
    assert [m["instance_id"] for m in result["matches"]] == [0]
    assert result["matches"][0]["score"] == pytest.approx(1.0)


def test_matches_are_ranked_by_descending_score(client: TestClient):
    scene = _ready_scene(client)
    result = _query(client, scene, "furniture light")
    assert [m["instance_id"] for m in result["matches"]] == [1, 2]
    assert [m["score"] for m in result["matches"]] == pytest.approx([0.7, 0.6], abs=1e-3)


def test_query_text_is_case_and_whitespace_insensitive_in_fake_encoder(client: TestClient):
    scene = _ready_scene(client)
    assert [m["instance_id"] for m in _query(client, scene, "  Red Chair ")["matches"]] == [0]


def test_no_match_above_threshold_returns_one_low_confidence_match(client: TestClient):
    scene = _ready_scene(client)
    result = _query(client, scene, "spaceship")
    assert result["low_confidence"] is True
    assert [m["instance_id"] for m in result["matches"]] == [0]
    assert result["matches"][0]["score"] == pytest.approx(0.3, abs=1e-3)


def test_score_equal_to_threshold_is_a_match(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=Settings(tmp_path / "s", token=TOKEN, match_threshold=1.0))
    scene = _ready_scene(client)
    result = _query(client, scene, "red chair")
    assert result["low_confidence"] is False


@pytest.mark.parametrize(("threshold", "low_confidence"), [(0.79, False), (0.81, True)])
def test_threshold_comes_from_settings(
    make_client: Callable[..., TestClient], tmp_path: Path, threshold: float, low_confidence: bool
):
    client = make_client(settings=Settings(tmp_path / "s", token=TOKEN, match_threshold=threshold))
    scene = _ready_scene(client)
    result = _query(client, scene, "chair")
    assert result["low_confidence"] is low_confidence
    assert [m["instance_id"] for m in result["matches"]] == [0]


def test_scene_without_instances_returns_no_matches(make_client: Callable[..., TestClient]):
    base = fake_reconstruction()
    empty = Reconstruction(
        points=base.points,
        colors=base.colors,
        instance_ids=np.full(len(base.points), -1, dtype=np.int32),
    )
    client = make_client(pipeline=FakePipeline(reconstruction=empty))
    scene = _ready_scene(client)
    assert _query(client, scene, "red chair") == {"matches": [], "low_confidence": False}


def test_scene_whose_instances_lack_embeddings_cannot_be_queried(
    make_client: Callable[..., TestClient],
):
    rec = fake_reconstruction()
    for inst in rec.instances:
        inst.embedding = None
    client = make_client(pipeline=FakePipeline(reconstruction=rec))
    scene = _ready_scene(client)
    resp = client.post(f"/scenes/{scene}/query", json={"text": "lamp"}, headers=AUTH)
    assert resp.status_code == 409


def test_empty_query_is_rejected(client: TestClient):
    scene = _ready_scene(client)
    resp = client.post(f"/scenes/{scene}/query", json={"text": "   "}, headers=AUTH)
    assert resp.status_code == 422


def test_query_on_unknown_scene_is_404(client: TestClient):
    resp = client.post("/scenes/nope/query", json={"text": "lamp"}, headers=AUTH)
    assert resp.status_code == 404


def test_instance_ids_align_with_point_cloud_order(client: TestClient):
    scene = _ready_scene(client)
    ply = client.get(f"/scenes/{scene}/points", headers=AUTH).content
    resp = client.get(f"/scenes/{scene}/instance-ids", headers=AUTH)

    assert resp.status_code == 200
    ids = np.frombuffer(resp.content, dtype="<i4")
    points, _ = decode_ply(ply)
    assert len(ids) == len(points)
    assert ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, -1, -1]


def test_instances_report_centroid_and_bounding_box_in_metres(client: TestClient):
    scene = _ready_scene(client)
    instances = client.get(f"/scenes/{scene}/instances", headers=AUTH).json()

    assert [i["instance_id"] for i in instances] == [0, 1, 2]
    chair = instances[0]
    assert chair["centroid"] == pytest.approx([0.25, 0.5, 0.25])
    assert chair["bbox_min"] == pytest.approx([0.0, 0.0, 0.0])
    assert chair["bbox_max"] == pytest.approx([0.5, 1.0, 1.0])


def test_instance_endpoints_require_a_ready_scene(client: TestClient):
    assert client.get("/scenes/nope/instance-ids", headers=AUTH).status_code == 404
    assert client.get("/scenes/nope/instances", headers=AUTH).status_code == 404


class _HalvingEncoder(FakeTextEncoder):
    """Maps cosine similarity to score = cosine / 2, like a calibrated model would rescale."""

    def to_score(self, cosine):
        return np.asarray(cosine, dtype=np.float32) / 2


def test_scores_come_from_the_encoders_calibration(make_client: Callable[..., TestClient]):
    client = make_client(text_encoder=_HalvingEncoder())
    scene = _ready_scene(client)
    result = _query(client, scene, "red chair")
    assert result["matches"][0]["score"] == pytest.approx(0.5)
    assert result["low_confidence"] is False  # 0.5 >= threshold 0.5
    assert _query(client, scene, "chair")["low_confidence"] is True  # 0.4 < 0.5
