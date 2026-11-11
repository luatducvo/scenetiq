"""Queries with a Relation, through the HTTP API with the fake parser (ADR 0009)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scenetiq.config import Settings, SpatialOptions
from scenetiq.domain import (
    Aspect,
    Caption,
    Constraint,
    EmbeddedCaption,
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
LAMP_THIRD = l2_normalize([0, 0, 0.4, 0, 0, 0, 0.916515, 0])  # lamp 0.4, chair 0
TABLE_FIRST = l2_normalize([0, 1, 0, 0, 0, 0, 0, 0])  # table 1.0, chair 0, lamp 0

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
LAMP_TABLE_CHAIR = (QueryObject("a", "lamp"), QueryObject("b", "table"), QueryObject("c", "chair"))
ON_AND_NEAR_QUERY = "the lamp on the table near the chair"
ON_AND_NEAR_PARSE = ParsedQuery(
    objects=LAMP_TABLE_CHAIR,
    target="a",
    constraints=(
        Constraint(Relation.ON, "a", ("b",)),
        Constraint(Relation.NEAR, "a", ("c",)),
    ),
)
BETWEEN_QUERY = "the lamp between the table and the chair"
BETWEEN_PARSE = ParsedQuery(
    objects=LAMP_TABLE_CHAIR,
    target="a",
    constraints=(Constraint(Relation.BETWEEN, "a", ("b", "c")),),
)
LARGEST_QUERY = "the largest chair"
LARGEST_PARSE = ParsedQuery(
    objects=(QueryObject("a", "chair"),),
    target="a",
    constraints=(Constraint(Relation.LARGEST, "a"),),
)
NEAREST_QUERY = "the chair nearest the lamp"
NEAREST_PARSE = ParsedQuery(
    objects=(QueryObject("a", "chair"), QueryObject("b", "lamp")),
    target="a",
    constraints=(Constraint(Relation.NEAREST_TO, "a", ("b",)),),
)
PARSER = FakeQueryParser(
    {
        NEAR_QUERY: NEAR_PARSE,
        TWO_CHAIRS_QUERY: TWO_CHAIRS_PARSE,
        ON_AND_NEAR_QUERY: ON_AND_NEAR_PARSE,
        BETWEEN_QUERY: BETWEEN_PARSE,
        LARGEST_QUERY: LARGEST_PARSE,
        NEAREST_QUERY: NEAREST_PARSE,
    }
)


def _box_scene(
    boxes: dict[int, tuple[list[float], list[float], FloatArray]],
    captions: dict[int, EmbeddedCaption] | None = None,
) -> Reconstruction:
    """Instances drawn as their two corner points ``low`` and ``high``, so each Instance's
    bounding box is exactly ``low``-``high`` and its centroid the box's centre.

    Up is -Y (ADR 0007): a box's top is its smaller y, its footprint its X-Z rectangle.
    """
    points, ids = [], []
    for iid, (low, high, _) in boxes.items():
        points += [low, high]
        ids += [iid, iid]
    pts = np.asarray(points, dtype=np.float32)
    instance_ids = np.asarray(ids, dtype=np.int32)
    return Reconstruction(
        points=pts,
        colors=np.zeros((len(pts), 3), dtype=np.uint8),
        instance_ids=instance_ids,
        instances=build_instances(
            pts, instance_ids, {i: e for i, (*_, e) in boxes.items()}, captions
        ),
    )


def _flat_scene(boxes: dict[int, tuple[list[float], FloatArray]]) -> Reconstruction:
    """Instances drawn as two corner points on z = 0, each box ``(x, y)`` to ``(x+1, y+1)``.

    Every scene here spans x 0-8 and y 0-6, so its extent is sqrt(8^2 + 6^2) = 10 and the
    default falloff scale is 0.1 * 10 = 1 metre: near = exp(-gap).
    """
    return _box_scene(
        {i: ([x, y, 0.0], [x + 1, y + 1, 0.0], e) for i, ((x, y), e) in boxes.items()}
    )


WIDE = (2.0, 2.0)
"""Candidate margins wider than any difference of two cosines, so every shortlisted Instance
stays a candidate. For tests about something other than the margin, whose scenes rank a
poor second candidate for a phrase."""

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
    scene: dict[int, tuple[list[float], FloatArray]] | Reconstruction,
    *,
    candidates: int,
    top_k: int,
    margins: tuple[float, float] | None = None,
) -> tuple[TestClient, str]:
    """A client with ``scene`` (flat boxes for :func:`_flat_scene`, or a Reconstruction).

    ``margins`` are the Caption and image candidate margins; the defaults when omitted.
    """
    spatial = SpatialOptions(candidates_per_object=candidates)
    if margins is not None:
        spatial = SpatialOptions(
            candidates_per_object=candidates, caption_margin=margins[0], image_margin=margins[1]
        )
    settings = Settings(tmp_path / "s", token=TOKEN, query_top_k=top_k, spatial=spatial)
    reconstruction = scene if isinstance(scene, Reconstruction) else _flat_scene(scene)
    client = make_client(
        settings=settings,
        pipeline=FakePipeline(reconstruction=reconstruction),
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
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=2, top_k=1, margins=WIDE)
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
    client, scene = _client(make_client, tmp_path, boxes, candidates=2, top_k=2, margins=WIDE)
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
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=2, top_k=1, margins=WIDE)
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
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=1, top_k=2, margins=WIDE)
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
    # Chair and table are 1.5 m apart: exp(-1.5 / 0.732189) = exp(-2.048653) = 0.128908.
    # The candidate margin leaves one candidate per object: for "chair" the table and the
    # lamp are 0.8 below the chair's image similarity and the table 1.0 below its Caption
    # similarity, and likewise for "table". The chair's similarities for "chair": image 0.8,
    # Caption 1.0 (Identity).
    [match] = _query(client, body["scene_id"], "the chair near the table")["matches"]
    assert match == {
        "instance_id": 0,
        "score": pytest.approx(0.128908, abs=1e-6),
        "image_score": pytest.approx(0.8, abs=1e-6),
        "caption_score": pytest.approx(1.0, abs=1e-6),
        "caption_aspect": "identity",
        "anchors": [{"instance_id": 1, "phrase": "table"}],
    }


def test_a_query_with_two_relations_returns_the_candidate_that_satisfies_both(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    # Up is -Y: the table's top is y 0, and the lamps resting on it span y -1 to 0. The scene
    # spans x 0-6, y -1 to 1, z 0-3: extent sqrt(36 + 4 + 9) = 7, falloff scale 0.7.
    scene = _box_scene(
        {
            0: ([0, 0, 0], [3.5, 1, 3], TABLE_FIRST),
            # On the table, far from the chair. First for "lamp".
            1: ([0, -1, 0], [1, 0, 1], LAMP_FIRST),
            # On the table with half its footprint over the edge, 1 m from the chair.
            2: ([3, -1, 2], [4, 0, 3], LAMP_SECOND),
            3: ([5, -1, 2], [6, 1, 3], CHAIR_FIRST),
            # On the floor beside the table, touching the chair.
            4: ([5, 0, 1], [6, 1, 2], LAMP_THIRD),
        }
    )
    client, scene_id = _client(make_client, tmp_path, scene, candidates=1, top_k=3, margins=WIDE)
    # Candidates: lamp [1, 2, 4] (1.0, 0.6, 0.4; query_top_k 3), table [0], chair [3].
    # Lamp 2: on = above 1 (higher, footprints overlap) * share 0.5 (x 3-3.5 of its 1 x 1)
    #   * vertical 1 (bottom y 0 meets the table's top y 0) = 0.5; near: gap 1 along x,
    #   exp(-1 / 0.7) = 0.239651. Product 0.5 * 0.239651 = 0.119826.
    # Lamp 1: on = 1 (footprint inside, bottom meets top); near: gaps 4 along x, 1 along z,
    #   sqrt(17) = 4.123106, exp(-4.123106 / 0.7) = 0.002767. Product 0.002767.
    # Lamp 4: near = 1 (touches the chair), but its centroid y 0.5 equals the table's, so it
    #   is not above it: on = 0. Product 0.
    result = _query(client, scene_id, ON_AND_NEAR_QUERY)
    assert result["relations_applied"] is True
    assert [m["instance_id"] for m in result["matches"]] == [2, 1, 4]
    assert [m["score"] for m in result["matches"]] == pytest.approx(
        [0.119826, 0.002767, 0.0], abs=1e-6
    )
    assert [m["image_score"] for m in result["matches"]] == pytest.approx([0.6, 1.0, 0.4])
    assert result["matches"][0]["anchors"] == [
        {"instance_id": 0, "phrase": "table"},
        {"instance_id": 3, "phrase": "chair"},
    ]


def test_a_between_query_returns_a_match_with_both_anchors(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    boxes = {
        0: ([0.0, 0.0], TABLE_FIRST),  # centroid (0.5, 0.5, 0)
        1: ([6.0, 0.0], CHAIR_FIRST),  # centroid (6.5, 0.5, 0)
        2: ([7.0, 5.0], LAMP_FIRST),  # centroid (7.5, 5.5, 0), first for "lamp"
        3: ([3.0, 1.0], LAMP_SECOND),  # centroid (3.5, 1.5, 0), 1 m off the segment
    }
    client, scene = _client(make_client, tmp_path, boxes, candidates=1, top_k=2, margins=WIDE)
    # Candidates: lamp [2, 3] (query_top_k 2), table [0], chair [1]. The segment from
    # (0.5, 0.5) to (6.5, 0.5) is 6 long: scale 0.1 * 6 = 0.6. Lamp 3 is 1 from it:
    # exp(-1 / 0.6) = exp(-1.666667) = 0.188876. Lamp 2 lies past the chair's end, so it is
    # measured to (6.5, 0.5): sqrt(1 + 25) = 5.099020, exp(-5.099020 / 0.6) = 0.000204.
    result = _query(client, scene, BETWEEN_QUERY)
    assert result["relations_applied"] is True
    assert [m["instance_id"] for m in result["matches"]] == [3, 2]
    assert [m["score"] for m in result["matches"]] == pytest.approx([0.188876, 0.000204], abs=1e-6)
    assert result["matches"][0] == {
        "instance_id": 3,
        "score": pytest.approx(0.188876, abs=1e-6),
        "image_score": pytest.approx(0.6, abs=1e-6),
        "caption_score": None,
        "caption_aspect": None,
        "anchors": [
            {"instance_id": 0, "phrase": "table"},
            {"instance_id": 1, "phrase": "chair"},
        ],
    }


def test_a_largest_query_returns_the_biggest_box_over_the_first_ranked_candidate(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    scene = _box_scene(
        {
            0: ([0, 0, 0], [1, 1, 1], CHAIR_FIRST),  # volume 1, first for "chair"
            1: ([3, 0, 0], [5, 2, 2], CHAIR_SECOND),  # volume 2 * 2 * 2 = 8
            2: ([7, 0, 0], [8, 1, 1], LAMP_FIRST),
        }
    )
    client, scene_id = _client(make_client, tmp_path, scene, candidates=1, top_k=2, margins=WIDE)
    # Candidates: chair [0, 1] (0.8, 0.48; query_top_k 2). Of two, the bigger box scores 1
    # and the smaller 0. A Query with only a size Relation has no Anchor.
    assert _query(client, scene_id, LARGEST_QUERY) == {
        "matches": [
            {
                "instance_id": 1,
                "score": 1.0,
                "image_score": pytest.approx(0.48, abs=1e-6),
                "caption_score": None,
                "caption_aspect": None,
                "anchors": [],
            },
            {
                "instance_id": 0,
                "score": 0.0,
                "image_score": pytest.approx(0.8, abs=1e-6),
                "caption_score": None,
                "caption_aspect": None,
                "anchors": [],
            },
        ],
        "relations_applied": True,
    }


def test_a_nearest_query_returns_the_closest_candidate_with_its_anchor(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=1, top_k=2, margins=WIDE)
    # Candidates: chair [0, 1] (query_top_k 2), lamp [2]. Gaps to the lamp: chair 1 is 1 m
    # away, chair 0 (6, 4) away. Nearest first: chair 1 scores 1, chair 0 scores 0.
    result = _query(client, scene, NEAREST_QUERY)
    assert result["relations_applied"] is True
    assert [m["instance_id"] for m in result["matches"]] == [1, 0]
    assert [m["score"] for m in result["matches"]] == [1.0, 0.0]
    assert [m["anchors"] for m in result["matches"]] == [
        [{"instance_id": 2, "phrase": "lamp"}],
        [{"instance_id": 2, "phrase": "lamp"}],
    ]


def test_a_chair_standing_in_for_the_anchor_still_competes_for_nearest(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    boxes = {
        0: ([2.0, 0.0], CHAIR_SECOND),  # 1 m from the lamp
        1: ([7.0, 5.0], CHAIR_FIRST),  # first for "chair", (4, 4) from chair 0
        2: ([0.0, 0.0], LAMP_FIRST),
    }
    client, scene = _client(make_client, tmp_path, boxes, candidates=2, top_k=1, margins=WIDE)
    # Candidates: chair [1, 0]; lamp [2, 0] (both chairs resemble "lamp" by 0, lower id first).
    # Chair 1 with lamp 2: chair 0 is nearer (1 m against (6, 4)), so chair 1 scores 0.
    # Chair 1 with chair 0 as the lamp: chair 0 itself (gap 0) is nearest, chair 1 scores 0.
    # Chair 0 with lamp 2: nearest, 1. Chair 0 wins, with the real lamp as its Anchor.
    [match] = _query(client, scene, NEAREST_QUERY)["matches"]
    assert match["instance_id"] == 0
    assert match["score"] == 1.0
    assert match["anchors"] == [{"instance_id": 2, "phrase": "lamp"}]


# Image embeddings against the fake encoder's "chair" (0.8, 0, 0, 0.6, ...): 0.8 times the
# first component.
CHAIR_IMAGE_078 = l2_normalize([0.975, 0, 0, 0, 0.222205, 0, 0, 0])  # 0.78
CHAIR_IMAGE_06 = l2_normalize([0.75, 0, 0, 0, 0.661438, 0, 0, 0])  # 0.6


def _chair_caption(similarity: float) -> EmbeddedCaption:
    """A Caption whose Identity Description has ``similarity`` to the Caption query "chair"
    (the first basis vector of the fake Caption space)."""
    embedding = l2_normalize([similarity, (1 - similarity**2) ** 0.5, 0, 0, 0, 0, 0])
    return EmbeddedCaption(Caption("chair", identity="A chair."), {Aspect.IDENTITY: embedding})


def test_an_instance_far_below_the_best_on_both_channels_does_not_stand_in_for_the_object(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    scene = _box_scene(
        {
            0: ([0, 0, 0], [1, 1, 1], CHAIR_FIRST),  # volume 1
            1: ([3, 0, 0], [5, 2, 2], CHAIR_IMAGE_06),  # volume 8: the largest box
            2: ([7, 0, 0], [8, 1, 1], LAMP_FIRST),
        },
        {0: _chair_caption(0.9), 1: _chair_caption(0.6)},
    )
    client, scene_id = _client(make_client, tmp_path, scene, candidates=5, top_k=2)
    # Default margins: Caption 0.1, image 0.05. All three are shortlisted. Instance 1 is 0.3
    # below the best Caption (0.9) and 0.2 below the best image (0.8): dropped, although its
    # box is the largest. The lamp has no Caption and is 0.8 below on the image: dropped.
    # Chair 0, the only candidate, is the largest of one and scores 1: one Match although
    # query_top_k is 2.
    assert _query(client, scene_id, LARGEST_QUERY) == {
        "matches": [
            {
                "instance_id": 0,
                "score": 1.0,
                "image_score": pytest.approx(0.8, abs=1e-6),
                "caption_score": pytest.approx(0.9, abs=1e-6),
                "caption_aspect": "identity",
                "anchors": [],
            }
        ],
        "relations_applied": True,
    }


def test_instances_passing_on_one_channel_or_on_the_image_without_a_caption_stay_candidates(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    scene = _box_scene(
        {
            0: ([0, 0, 0], [1, 1, 1], CHAIR_FIRST),  # volume 1
            1: ([2, 0, 0], [4, 2, 2], CHAIR_IMAGE_06),  # volume 8
            2: ([5, 0, 0], [8, 3, 3], CHAIR_IMAGE_078),  # volume 27, no Caption
            3: ([0, 0, 4], [4, 4, 8], CHAIR_SECOND),  # volume 64
        },
        {0: _chair_caption(0.9), 1: _chair_caption(0.85), 3: _chair_caption(0.6)},
    )
    client, scene_id = _client(make_client, tmp_path, scene, candidates=5, top_k=4)
    # Best image 0.8 (chair 0), best Caption 0.9 (chair 0).
    # Chair 1: image 0.6 fails (0.2 below), Caption 0.85 passes (0.05 below): stays.
    # Chair 2: no Caption, image 0.78 passes (0.02 below): stays.
    # Chair 3: image 0.48 (0.32 below) and Caption 0.6 (0.3 below) fail: dropped, although
    # its box is the largest.
    # Largest among volumes 27, 8, 1: chair 2 scores 1, chair 1 1/2, chair 0 0. Had chair 3
    # stayed, it would have been the Match and the others 2/3, 1/3, 0.
    matches = _query(client, scene_id, LARGEST_QUERY)["matches"]
    assert [m["instance_id"] for m in matches] == [2, 1, 0]
    assert [m["score"] for m in matches] == [1.0, 0.5, 0.0]
    assert [m["image_score"] for m in matches] == pytest.approx([0.78, 0.6, 0.8], abs=1e-6)
    assert matches[0]["caption_score"] is None
    assert [m["caption_score"] for m in matches[1:]] == pytest.approx([0.85, 0.9], abs=1e-6)


def test_an_object_whose_shortlist_is_cut_to_one_instance_still_yields_a_match(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=2, top_k=1)
    # No Instance has a Caption, so only the image channel exists. "chair" shortlists 0
    # (0.8) and 1 (0.48, 0.32 below: dropped); "lamp" shortlists 2 (1.0) and 0 (0, 1.0 below:
    # dropped). Chair 1 would be near the lamp, but only chair 0 is left: (6, 4) from the
    # lamp, exp(-sqrt(52)) = 0.000738.
    assert _query(client, scene, NEAR_QUERY)["matches"] == [
        {
            "instance_id": 0,
            "score": pytest.approx(0.000738, abs=1e-6),
            "image_score": pytest.approx(0.8, abs=1e-6),
            "caption_score": None,
            "caption_aspect": None,
            "anchors": [{"instance_id": 2, "phrase": "lamp"}],
        }
    ]


def test_with_query_top_k_above_m_only_a_spatial_query_returns_fewer_matches(
    make_client: Callable[..., TestClient], tmp_path: Path
):
    client, scene = _client(make_client, tmp_path, LAMP_SCENE, candidates=1, top_k=3)
    # The chair shortlists query_top_k = 3 Instances: 0 (0.8), 1 (0.48) and the lamp (0).
    # Only chair 0 is within the image margin, so the spatial Query has one Match.
    spatial = _query(client, scene, NEAR_QUERY)["matches"]
    assert [m["instance_id"] for m in spatial] == [0]
    # Without a Relation no margin applies: query_top_k Matches, best first.
    plain = _query(client, scene, "chair")["matches"]
    assert [m["instance_id"] for m in plain] == [0, 1, 2]
