import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scenetiq.artifacts import decode_ply
from scenetiq.config import Settings
from scenetiq.domain import Reconstruction
from scenetiq.fakes import FakeCaptionEncoder, FakePipeline, fake_reconstruction

from .conftest import AUTH, TOKEN, upload, wait_for_job


def _ready_scene(client: TestClient) -> str:
    body = upload(client)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"
    return body["scene_id"]


def _query(client: TestClient, scene_id: str, text: str) -> dict:
    resp = client.post(f"/scenes/{scene_id}/query", json={"text": text}, headers=AUTH)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _settings(tmp_path: Path, query_top_k: int = 1, rrf_k: int = 60) -> Settings:
    return Settings(tmp_path / "s", token=TOKEN, query_top_k=query_top_k, rrf_k=rrf_k)


def _uncaptioned_client(make_client: Callable[..., TestClient], settings: Settings) -> TestClient:
    rec = fake_reconstruction(with_captions=False)
    return make_client(settings=settings, pipeline=FakePipeline(reconstruction=rec))


def test_query_returns_the_best_instance_with_a_full_score(client: TestClient):
    scene = _ready_scene(client)
    result = _query(client, scene, "red chair")
    # First in both rankings: (1/61 + 1/61) / (2/61) = 1.
    assert result == {
        "matches": [{"instance_id": 0, "score": 1.0, "image_score": 1.0, "caption_score": 1.0}]
    }


def test_a_caption_lifts_an_instance_its_image_ranks_lower(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=_settings(tmp_path, query_top_k=3))
    scene = _ready_scene(client)
    result = _query(client, scene, "wooden thing")
    # Image ranking: lamp 0.5, table 0.3, chair 0.1. Caption ranking (lamp has no Caption):
    # table 1.0, chair 0.0. Fused over two channels, out of a maximum of 2/61:
    # table (1/62 + 1/61) -> 0.991935, chair (1/63 + 1/62) -> 0.976062, lamp 1/61 -> 0.5.
    assert [m["instance_id"] for m in result["matches"]] == [1, 0, 2]
    assert [m["score"] for m in result["matches"]] == pytest.approx(
        [0.991935, 0.976062, 0.5], abs=1e-6
    )
    assert [m["caption_score"] for m in result["matches"]] == [
        pytest.approx(1.0),
        pytest.approx(0.0),
        None,
    ]
    assert [m["image_score"] for m in result["matches"]] == pytest.approx([0.3, 0.1, 0.5], abs=1e-3)


def test_a_scene_stored_before_captions_is_ranked_by_image_alone(client: TestClient):
    scene = _ready_scene(client)
    directory = client.app.state.store.reconstruction_dir(scene)  # type: ignore[attr-defined]
    rows = json.loads((directory / "instances.json").read_text())
    for row in rows:
        del row["caption"]
    (directory / "instances.json").write_text(json.dumps(rows))
    (directory / "caption_embeddings.npy").unlink()

    result = _query(client, scene, "wooden thing")
    # Image ranking alone: lamp first, normalised over one channel.
    assert result["matches"] == [
        {"instance_id": 2, "score": 1.0, "image_score": pytest.approx(0.5, abs=1e-3),
         "caption_score": None}
    ]  # fmt: skip


def test_matches_are_the_top_k_ranked_by_fused_rank(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path, query_top_k=2))
    scene = _ready_scene(client)
    result = _query(client, scene, "furniture light")
    # Image similarities: table 0.7, lamp 0.6, chair 0. With one ranking and rrf_k = 60
    # the normalised score is 61 / (60 + rank): 1 for rank 1, 61/62 for rank 2.
    assert [m["instance_id"] for m in result["matches"]] == [1, 2]
    assert [m["score"] for m in result["matches"]] == pytest.approx([1.0, 0.983871], abs=1e-6)
    assert [m["image_score"] for m in result["matches"]] == pytest.approx([0.7, 0.6], abs=1e-3)


def test_a_scene_with_fewer_instances_than_k_returns_them_all(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path, query_top_k=5))
    scene = _ready_scene(client)
    result = _query(client, scene, "furniture light")
    assert [m["instance_id"] for m in result["matches"]] == [1, 2, 0]
    assert result["matches"][2]["score"] == pytest.approx(61 / 63, abs=1e-6)


def test_the_rrf_constant_comes_from_settings(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path, query_top_k=3, rrf_k=0))
    scene = _ready_scene(client)
    result = _query(client, scene, "furniture light")
    # rrf_k = 0: ranks 1, 2, 3 give 1/1, 1/2, 1/3 out of a maximum of 1.
    assert [m["score"] for m in result["matches"]] == pytest.approx([1.0, 0.5, 1 / 3], abs=1e-6)


def test_a_query_resembling_nothing_still_returns_the_closest_instance(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path))
    scene = _ready_scene(client)
    result = _query(client, scene, "spaceship")
    assert [m["instance_id"] for m in result["matches"]] == [0]
    assert result["matches"][0]["image_score"] == pytest.approx(0.3, abs=1e-3)
    assert "low_confidence" not in result


def test_query_text_is_case_and_whitespace_insensitive_in_fake_encoder(client: TestClient):
    scene = _ready_scene(client)
    assert [m["instance_id"] for m in _query(client, scene, "  Red Chair ")["matches"]] == [0]


def test_scene_without_instances_returns_no_matches(make_client: Callable[..., TestClient]):
    base = fake_reconstruction()
    empty = Reconstruction(
        points=base.points,
        colors=base.colors,
        instance_ids=np.full(len(base.points), -1, dtype=np.int32),
    )
    client = make_client(pipeline=FakePipeline(reconstruction=empty))
    scene = _ready_scene(client)
    assert _query(client, scene, "red chair") == {"matches": []}


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


def test_instances_report_only_the_name_of_their_caption(client: TestClient):
    scene = _ready_scene(client)
    instances = client.get(f"/scenes/{scene}/instances", headers=AUTH).json()
    assert [i["name"] for i in instances] == ["chair", "table", None]
    assert {"synonyms", "attributes", "use", "caption"}.isdisjoint(instances[0])


def test_instances_of_a_scene_without_captions_have_no_name(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path))
    scene = _ready_scene(client)
    instances = client.get(f"/scenes/{scene}/instances", headers=AUTH).json()
    assert [i["name"] for i in instances] == [None, None, None]


class _CountingCaptionEncoder(FakeCaptionEncoder):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, text: str):
        self.queries.append(text)
        return super().encode_query(text)


def test_a_scene_without_captions_never_uses_the_caption_encoder(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    counting = _CountingCaptionEncoder()
    rec = fake_reconstruction(with_captions=False)
    client = make_client(
        settings=_settings(tmp_path),
        pipeline=FakePipeline(reconstruction=rec),
        caption_encoder=counting,
    )
    scene = _ready_scene(client)
    assert [m["instance_id"] for m in _query(client, scene, "lamp")["matches"]] == [2]
    assert counting.queries == []


def test_a_scene_with_captions_encodes_the_query_for_them(make_client: Callable[..., TestClient]):
    counting = _CountingCaptionEncoder()
    client = make_client(caption_encoder=counting)
    scene = _ready_scene(client)
    _query(client, scene, "table")
    assert counting.queries == ["table"]
