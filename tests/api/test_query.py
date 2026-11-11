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


def _settings(tmp_path: Path, query_top_k: int = 1, query_score_ratio: float = 0.5) -> Settings:
    return Settings(
        tmp_path / "s", token=TOKEN, query_top_k=query_top_k, query_score_ratio=query_score_ratio
    )


def _uncaptioned_client(make_client: Callable[..., TestClient], settings: Settings) -> TestClient:
    rec = fake_reconstruction(with_captions=False)
    return make_client(settings=settings, pipeline=FakePipeline(reconstruction=rec))


def test_query_returns_the_best_instance_with_a_full_score(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=_settings(tmp_path, query_top_k=1))
    scene = _ready_scene(client)
    result = _query(client, scene, "red chair")
    # The fake parser reads no Relation, so the Query is plain and has no Anchors.
    # The chair is the best on both channels, and the best Match always scores 1.
    assert result == {
        "matches": [
            {
                "instance_id": 0,
                "score": 1.0,
                "image_score": 1.0,
                "caption_score": 1.0,
                "caption_aspect": "identity",
                "anchors": [],
            }
        ],
        "relations_applied": False,
        "relations_stated": False,
        "unsupported": [],
        "parse_failed": False,
    }


def test_both_channels_are_fused_and_an_instance_far_below_the_best_is_no_match(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=_settings(tmp_path, query_top_k=3))
    scene = _ready_scene(client)
    result = _query(client, scene, "wooden thing")
    # Image: lamp 0.5, table 0.3, chair 0.1 -> standardized 1.224745, 0, -1.224745.
    # Caption (the lamp has none, so it takes 0): table 1.0 (its Material Description),
    # chair 0.0 (every Description orthogonal, so the first Aspect, Identity, wins the tie)
    # -> standardized 1, -1. Fused, the mean of the two: lamp 0.612372, table 0.5,
    # chair -1.112372. Scores, out of the lamp's: 1, 0.816497 and a negative one, which is
    # below the ratio of 0.5, so the chair is no Match.
    matches = result["matches"]
    assert [m["instance_id"] for m in matches] == [2, 1]
    assert [m["score"] for m in matches] == pytest.approx([1.0, 0.816497], abs=1e-6)
    assert [m["caption_score"] for m in matches] == [None, pytest.approx(1.0)]
    assert [m["image_score"] for m in matches] == pytest.approx([0.5, 0.3], abs=1e-3)
    assert [m["caption_aspect"] for m in matches] == [None, "material"]


def test_a_query_matching_another_aspect_of_another_instance_lifts_that_instance(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=_settings(tmp_path, query_top_k=3, query_score_ratio=0.3))
    scene = _ready_scene(client)
    result = _query(client, scene, "red fabric")
    # Image: lamp 0.6, chair 0.2, table 0.1 -> mean 0.3, deviation sqrt(0.14 / 3) = 0.216025
    # -> standardized 1.388730, -0.462910, -0.925820. Caption query (0, 0.6, 0.8, 0, ...):
    # the chair's best Description is Appearance (0.8), the table's is Identity (0.6)
    # -> standardized 1, -1. Fused: lamp 0.694365, chair 0.268545, table -0.962910.
    # Scores: 1, 0.268545 / 0.694365 = 0.386749, and the table below the ratio of 0.3.
    matches = result["matches"]
    assert [m["instance_id"] for m in matches] == [2, 0]
    assert [m["score"] for m in matches] == pytest.approx([1.0, 0.386749], abs=1e-6)
    assert [m["caption_aspect"] for m in matches] == [None, "appearance"]
    assert [m["caption_score"] for m in matches] == [None, pytest.approx(0.8)]


def test_an_instance_is_scored_only_over_the_descriptions_it_has(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=_settings(tmp_path, query_top_k=3))
    scene = _ready_scene(client)
    result = _query(client, scene, "metal legs")
    # The chair has only Identity and Appearance Descriptions. The Caption query
    # (-0.6, 0, -0.8, 0, ...) gives them -0.6 and -0.8, so its best is Identity at -0.6; its
    # missing Material and Use do not count as 0. The table's Descriptions are all
    # orthogonal (0.0, Identity first). Caption: table 0.0, chair -0.6 -> standardized 1, -1.
    # Image (unknown text resembles the chair): chair 0.3, table 0, lamp 0 -> mean 0.1,
    # deviation sqrt(0.02) -> standardized 1.414214, -0.707107, -0.707107.
    # Fused: chair 0.207107, table 0.146447, lamp -0.353553. Scores: 1, 0.707107.
    matches = result["matches"]
    assert [m["instance_id"] for m in matches] == [0, 1]
    assert [m["score"] for m in matches] == pytest.approx([1.0, 0.707107], abs=1e-6)
    assert matches[0]["caption_score"] == pytest.approx(-0.6)
    assert matches[0]["caption_aspect"] == "identity"
    assert matches[1]["caption_score"] == pytest.approx(0.0)


def test_matches_are_scored_against_the_best_one(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path, query_top_k=2))
    scene = _ready_scene(client)
    result = _query(client, scene, "furniture light")
    assert {m["caption_aspect"] for m in result["matches"]} == {None}
    # Image similarities: table 0.7, lamp 0.6, chair 0 -> mean 0.433333, so the table is
    # 0.266667 above it and the lamp 0.166667. With one channel the score is the ratio of
    # those two distances: 0.166667 / 0.266667 = 0.625.
    assert [m["instance_id"] for m in result["matches"]] == [1, 2]
    assert [m["score"] for m in result["matches"]] == pytest.approx([1.0, 0.625], abs=1e-6)
    assert [m["image_score"] for m in result["matches"]] == pytest.approx([0.7, 0.6], abs=1e-3)


def test_k_is_only_the_most_matches_a_query_returns(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path, query_top_k=5))
    scene = _ready_scene(client)
    result = _query(client, scene, "furniture light")
    # The chair is below the Scene's average, so asking for five still returns two.
    assert [m["instance_id"] for m in result["matches"]] == [1, 2]


def test_the_score_ratio_comes_from_settings(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    settings = _settings(tmp_path, query_top_k=3, query_score_ratio=0.7)
    client = _uncaptioned_client(make_client, settings)
    scene = _ready_scene(client)
    result = _query(client, scene, "furniture light")
    # The lamp scores 0.625 (worked out above), below a ratio of 0.7.
    assert [m["instance_id"] for m in result["matches"]] == [1]


def test_a_query_resembling_nothing_still_returns_the_closest_instance(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = _uncaptioned_client(make_client, _settings(tmp_path))
    scene = _ready_scene(client)
    result = _query(client, scene, "spaceship")
    assert [m["instance_id"] for m in result["matches"]] == [0]
    assert result["matches"][0]["image_score"] == pytest.approx(0.3, abs=1e-3)
    assert "low_confidence" not in result


def test_query_text_is_case_and_whitespace_insensitive_in_fake_encoder(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=_settings(tmp_path, query_top_k=1))
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
    assert _query(client, scene, "red chair") == {
        "matches": [],
        "relations_applied": False,
        "relations_stated": False,
        "unsupported": [],
        "parse_failed": False,
    }


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
    hidden = {"synonyms", "identity", "appearance", "material", "use", "caption"}
    assert hidden.isdisjoint(instances[0])


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


def _query_k(client: TestClient, scene_id: str, text: str, top_k: int | None):
    body: dict[str, object] = {"text": text}
    if top_k is not None:
        body["top_k"] = top_k
    return client.post(f"/scenes/{scene_id}/query", json=body, headers=AUTH)


def test_a_query_returns_as_many_matches_as_it_asks_for(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client = make_client(settings=_settings(tmp_path, query_top_k=1))
    scene = _ready_scene(client)

    resp = _query_k(client, scene, "wooden thing", top_k=2)

    # The Matches worked out above: lamp, table.
    assert resp.status_code == 200
    assert [m["instance_id"] for m in resp.json()["matches"]] == [2, 1]
    assert [m["instance_id"] for m in _query(client, scene, "wooden thing")["matches"]] == [2]


def test_a_query_without_top_k_returns_up_to_five_matches_by_default(client: TestClient):
    scene = _ready_scene(client)

    matches = _query_k(client, scene, "wooden thing", top_k=None).json()["matches"]

    # Five allows both Instances that score close enough to the best (worked out above).
    assert [m["instance_id"] for m in matches] == [2, 1]


@pytest.mark.parametrize("top_k", [0, 21])
def test_a_query_asking_for_no_or_too_many_matches_is_refused(client: TestClient, top_k: int):
    scene = _ready_scene(client)

    assert _query_k(client, scene, "table", top_k=top_k).status_code == 422
