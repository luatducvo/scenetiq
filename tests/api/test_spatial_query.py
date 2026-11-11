"""Queries with a Relation, through the HTTP API with the fake parser (ADR 0009)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scenetiq.config import Settings, SpatialOptions
from scenetiq.domain import (
    Constraint,
    FloatArray,
    ParsedQuery,
    QueryObject,
    Reconstruction,
    Relation,
)
from scenetiq.fakes import FakePipeline, FakeQueryParser
from scenetiq.geometry import build_instances, l2_normalize

from .conftest import AUTH, TOKEN, upload, wait_for_job

# Image embeddings against the fake encoder's "chair" (0.8, 0, 0, 0.6, ...) and "lamp" (basis 2).
CHAIR_FIRST = l2_normalize([1, 0, 0, 0, 0, 0, 0, 0])  # chair 0.8, lamp 0
CHAIR_SECOND = l2_normalize([0.6, 0, 0, 0, 0.8, 0, 0, 0])  # chair 0.6 * 0.8 = 0.48, lamp 0
LAMP_FIRST = l2_normalize([0, 0, 1, 0, 0, 0, 0, 0])  # lamp 1.0, chair 0
LAMP_SECOND = l2_normalize([0, 0, 0.6, 0, 0, 0.8, 0, 0])  # lamp 0.6, chair 0

NEAR_QUERY = "the chair near the lamp"
NEAR_PARSE = ParsedQuery(
    objects=(QueryObject("a", "chair"), QueryObject("b", "lamp")),
    target="a",
    constraints=(Constraint(Relation.NEAR, "a", ("b",)),),
)
TWO_CHAIRS_QUERY = "the chair next to another chair"
TWO_CHAIRS_PARSE = ParsedQuery(
    objects=(QueryObject("a", "chair"), QueryObject("b", "chair")),
    target="a",
    constraints=(Constraint(Relation.NEAR, "a", ("b",)),),
)
PARSER = FakeQueryParser({NEAR_QUERY: NEAR_PARSE, TWO_CHAIRS_QUERY: TWO_CHAIRS_PARSE})


def _flat_scene(boxes: dict[int, tuple[list[float], FloatArray]]) -> Reconstruction:
    """Instances drawn as two corner points on z = 0, each box ``(x, y)`` to ``(x+1, y+1)``.

    Every scene here spans x 0-8 and y 0-6, so its extent is sqrt(8^2 + 6^2) = 10 and the
    default falloff scale is 0.1 * 10 = 1 metre: near = exp(-gap).
    """
    points, ids = [], []
    for iid, (corner, _) in boxes.items():
        x, y = corner
        points += [[x, y, 0.0], [x + 1, y + 1, 0.0]]
        ids += [iid, iid]
    pts = np.asarray(points, dtype=np.float32)
    instance_ids = np.asarray(ids, dtype=np.int32)
    return Reconstruction(
        points=pts,
        colors=np.zeros((len(pts), 3), dtype=np.uint8),
        instance_ids=instance_ids,
        instances=build_instances(pts, instance_ids, {i: e for i, (_, e) in boxes.items()}),
    )


# The lamp at (0-1, 0-1); the second-ranked chair at (2-3, 0-1), 1 m from it; the first-ranked
# chair in the far corner at (7-8, 5-6), 6 m along x and 4 m along y from it.
LAMP_SCENE = {
    0: ([7.0, 5.0], CHAIR_FIRST),
    1: ([2.0, 0.0], CHAIR_SECOND),
    2: ([0.0, 0.0], LAMP_FIRST),
}


def _client(
    make_client: Callable[..., TestClient],
    tmp_path: Path,
    boxes: dict[int, tuple[list[float], FloatArray]],
    *,
    candidates: int,
    top_k: int,
) -> tuple[TestClient, str]:
    settings = Settings(
        tmp_path / "s",
        token=TOKEN,
        query_top_k=top_k,
        spatial=SpatialOptions(candidates_per_object=candidates),
    )
    client = make_client(
        settings=settings,
        pipeline=FakePipeline(reconstruction=_flat_scene(boxes)),
        query_parser=PARSER,
    )
    body = upload(client)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"
    return client, body["scene_id"]


def _query(client: TestClient, scene_id: str, text: str) -> dict:
    resp = client.post(f"/scenes/{scene_id}/query", json={"text": text}, headers=AUTH)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_near_query_returns_the_candidate_close_to_the_anchor(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=2, top_k=1)
    # "chair" ranks Instance 0 (0.8) above Instance 1 (0.48); "lamp" ranks 2 (1.0) first.
    # Chair 1 is 1 m from the lamp: exp(-1) = 0.367879. Chair 0 is (6, 4) away:
    # exp(-sqrt(52)) = 0.000738. So chair 1 wins, with the lamp as its Anchor.
    assert _query(client, scene, NEAR_QUERY) == {
        "matches": [
            {
                "instance_id": 1,
                "score": pytest.approx(0.367879, abs=1e-6),
                "image_score": pytest.approx(0.48, abs=1e-6),
                "caption_score": None,
                "caption_aspect": None,
                "anchors": [{"instance_id": 2, "phrase": "lamp"}],
            }
        ],
        "relations_applied": True,
    }


def test_a_query_the_parser_reads_without_a_relation_is_a_plain_query(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=2, top_k=1)
    # "chair" alone: the plain ranking, chair 0 first with one ranking: 61/61 = 1.
    assert _query(client, scene, "chair") == {
        "matches": [
            {
                "instance_id": 0,
                "score": 1.0,
                "image_score": pytest.approx(0.8, abs=1e-6),
                "caption_score": None,
                "caption_aspect": None,
                "anchors": [],
            }
        ],
        "relations_applied": False,
    }


def test_a_wrong_first_candidate_for_the_anchor_does_not_decide_the_match(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    boxes = {
        0: ([4.0, 0.0], CHAIR_FIRST),  # 3 m from the real lamp
        1: ([7.0, 5.0], LAMP_FIRST),  # ranks first for "lamp", far from every chair
        2: ([0.0, 0.0], LAMP_SECOND),  # the real lamp, second for "lamp"
        3: ([1.0, 0.0], CHAIR_SECOND),  # touches the real lamp
    }
    client, scene = _client(make_client, tmp_path, boxes, candidates=2, top_k=2)
    # Candidates: chair [0, 3], lamp [1, 2].
    # Chair 3: with lamp 2 the gap is 0 -> 1; with lamp 1, (5, 4) -> exp(-6.403) = 0.001656.
    # Chair 0: with lamp 2 the gap is 3 -> exp(-3) = 0.049787; with lamp 1, (2, 4) ->
    # exp(-sqrt(20)) = 0.011423. Both keep the second lamp candidate.
    result = _query(client, scene, NEAR_QUERY)
    assert [m["instance_id"] for m in result["matches"]] == [3, 0]
    assert [m["score"] for m in result["matches"]] == pytest.approx([1.0, 0.049787], abs=1e-6)
    assert [m["anchors"] for m in result["matches"]] == [
        [{"instance_id": 2, "phrase": "lamp"}],
        [{"instance_id": 2, "phrase": "lamp"}],
    ]


def test_two_objects_with_one_phrase_never_resolve_to_one_instance(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=2, top_k=1)
    # Both objects have the candidates [0, 1]. Chair 0 near chair 1 and the reverse are the
    # only valid assignments, both gap (4, 4) -> exp(-sqrt(32)) = 0.003493. On the tie, chair
    # 0 comes first in the phrase ranking.
    [match] = _query(client, scene, TWO_CHAIRS_QUERY)["matches"]
    assert match["instance_id"] == 0
    assert match["score"] == pytest.approx(0.003493, abs=1e-6)
    assert match["anchors"] == [{"instance_id": 1, "phrase": "chair"}]


def test_a_query_without_a_valid_assignment_is_answered_as_a_plain_query(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=1, top_k=1)
    # One candidate per object: both chairs are Instance 0, so no assignment is valid. The
    # original text is unknown to the fake encoder and resembles chair 0 (0.3), chair 1
    # (0.3 * 0.6 = 0.18) and the lamp (0): chair 0, first of one ranking, scores 1.
    assert _query(client, scene, TWO_CHAIRS_QUERY) == {
        "matches": [
            {
                "instance_id": 0,
                "score": 1.0,
                "image_score": pytest.approx(0.3, abs=1e-3),
                "caption_score": None,
                "caption_aspect": None,
                "anchors": [],
            }
        ],
        "relations_applied": False,
    }


def test_the_wanted_object_shortlists_query_top_k_when_that_is_larger(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=1, top_k=2)
    # The chair shortlists 2 (query_top_k), the lamp 1: chair 1 exp(-1) = 0.367879,
    # chair 0 exp(-sqrt(52)) = 0.000738.
    result = _query(client, scene, NEAR_QUERY)
    assert [m["instance_id"] for m in result["matches"]] == [1, 0]
    assert [m["score"] for m in result["matches"]] == pytest.approx([0.367879, 0.000738], abs=1e-6)


def test_a_spatial_match_carries_the_similarities_of_the_wanted_objects_phrase(
    make_client: Callable[..., TestClient],
):
    parse = ParsedQuery(
        objects=(QueryObject("a", "chair"), QueryObject("b", "table")),
        target="a",
        constraints=(Constraint(Relation.NEAR, "a", ("b",)),),
    )
    client = make_client(query_parser=FakeQueryParser({"the chair near the table": parse}))
    body = upload(client)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"
    # The fake Reconstruction: chair (0) 0-0.5 x 0-1 x 0-1, table (1) 2-3 x 0-0.5 x 0-0.8, lamp
    # (2) at (5, 5, 1.5-1.9). Extent sqrt(5^2 + 5^2 + 1.9^2) = 7.321885, scale 0.732189.
    # Chair and table are 1.5 m apart: exp(-1.5 / 0.732189) = exp(-2.048653) = 0.128908. The
    # table as the chair, with the chair as Anchor, scores the same; the chair ranks first for
    # "chair". Its similarities for "chair": image 0.8, Caption 1.0 (Identity).
    [match] = _query(client, body["scene_id"], "the chair near the table")["matches"]
    assert match == {
        "instance_id": 0,
        "score": pytest.approx(0.128908, abs=1e-6),
        "image_score": pytest.approx(0.8, abs=1e-6),
        "caption_score": pytest.approx(1.0, abs=1e-6),
        "caption_aspect": "identity",
        "anchors": [{"instance_id": 1, "phrase": "table"}],
    }
