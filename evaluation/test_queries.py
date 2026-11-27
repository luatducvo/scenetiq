import os

import numpy as np
import pytest

from scenetiq.domain.scene import Capture, Stage

from .support import configured_settings

SMOKE_QUERY_ENV = "SCENETIQ_SMOKE_QUERY"
SMOKE_QUERY_BOX_ENV = "SCENETIQ_SMOKE_QUERY_BOX"
MIN_POINTS_IN_BOX = 0.5
MIN_POINTS_IN_PHOTO = 0.2


def _annotated_box(capture: Capture) -> tuple[int, tuple[float, float, float, float]]:
    raw = os.environ.get(SMOKE_QUERY_BOX_ENV)
    if not raw:
        pytest.fail(
            f"set {SMOKE_QUERY_BOX_ENV}='view,left,top,right,bottom': the queried object's "
            "box in that photo (view = index in the sorted smoke Capture)"
        )
    view, left, top, right, bottom = (float(x) for x in raw.split(","))
    assert 0 <= int(view) < len(capture.views), f"{SMOKE_QUERY_BOX_ENV} view index out of range"
    return int(view), (left, top, right, bottom)


def _box_overlap(points, camera, box, photo_size) -> tuple[float, float]:
    from scenetiq.pipeline.stages.da3_adapters import c2w_to_w2c

    w2c = c2w_to_w2c(camera.cam_to_world)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    front = cam[:, 2] > 1e-6
    cam = cam[front]
    k = camera.intrinsics
    # Pixels at the depth resolution, scaled to the original photo the box refers to.
    sx, sy = photo_size[0] / camera.width, photo_size[1] / camera.height
    u = (k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2]) * sx
    v = (k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]) * sy
    in_image = (u >= 0) & (u < photo_size[0]) & (v >= 0) & (v < photo_size[1])
    in_photo = float(np.count_nonzero(in_image)) / max(len(points), 1)
    if not np.any(in_image):
        return in_photo, 0.0
    left, top, right, bottom = box
    u, v = u[in_image], v[in_image]
    inside = (u >= left) & (u < right) & (v >= top) & (v < bottom)
    return in_photo, float(np.mean(inside))


def test_a_query_returns_at_most_the_configured_number_of_matches(
    reconstruction_result, encoder, caption_encoder
):
    # Runs without the Query environment variables, so the cut of Matches is always checked.
    from scenetiq.query.ranking import run_query

    rec, _ = reconstruction_result
    settings = configured_settings()
    matches = run_query(
        rec.instances,
        encoder.encode("a chair"),
        settings.query_top_k,
        settings.query_score_ratio,
        caption_encoder.encode_query("a chair"),
    )
    assert 1 <= len(matches) <= min(settings.query_top_k, len(rec.instances))
    assert matches[0].score == pytest.approx(1.0)
    assert all(settings.query_score_ratio <= m.score <= 1.0 + 1e-6 for m in matches)
    for m in matches:
        assert (m.caption_aspect is None) == (m.caption_score is None)
    # Over every Instance, the Caption channel must name an Aspect for the captioned ones.
    everything = run_query(
        rec.instances,
        encoder.encode("a chair"),
        len(rec.instances),
        None,
        caption_encoder.encode_query("a chair"),
    )
    assert len(everything) == len(rec.instances)
    assert any(m.caption_aspect is not None for m in everything)


def test_query_for_a_present_object_matches_that_object(
    reconstruction_result, smoke_capture: Capture, encoder, caption_encoder
):
    from PIL import Image

    from scenetiq.query.ranking import run_query

    text = os.environ.get(SMOKE_QUERY_ENV)
    if not text:
        pytest.skip(f"set {SMOKE_QUERY_ENV} to an object visible in the smoke Capture")
    view, box = _annotated_box(smoke_capture)
    rec, stages = reconstruction_result
    assert stages[-1] == Stage.CAPTIONING
    assert all(inst.embedding is not None for inst in rec.instances)

    settings = configured_settings()
    matches = run_query(
        rec.instances,
        encoder.encode(text),
        settings.query_top_k,
        settings.query_score_ratio,
        caption_encoder.encode_query(text),
    )
    print(f"{text!r}: {[(m.instance_id, round(m.score, 3)) for m in matches]}")

    assert 1 <= len(matches) <= min(settings.query_top_k, len(rec.instances))
    best = matches[0].instance_id
    with Image.open(smoke_capture.views[view].path) as photo:
        photo_size = photo.size
    in_photo, share = _box_overlap(
        rec.points[rec.instance_ids == best], rec.cameras[view], box, photo_size
    )
    assert in_photo >= MIN_POINTS_IN_PHOTO, (
        f"best Match (Instance {best}) is mostly outside photo {view} ({in_photo:.0%} inside)"
    )
    assert share >= MIN_POINTS_IN_BOX, (
        f"best Match (Instance {best}): only {share:.0%} of its in-photo points are in the box"
    )


SMOKE_SPATIAL_QUERY_ENV = "SCENETIQ_SMOKE_SPATIAL_QUERY"
DEFAULT_SPATIAL_QUERY = "the object nearest to the chair"


def _answer(text, rec, query_parser, encoder, caption_encoder):
    import time

    from scenetiq.domain.query import QueryObject
    from scenetiq.query.ranking import run_query
    from scenetiq.query.spatial import run_spatial_query

    settings = configured_settings()
    captioned = any(inst.caption_embeddings for inst in rec.instances)

    def rank(wanted: QueryObject, top_k: int, score_ratio: float | None = None):
        return run_query(
            rec.instances,
            encoder.encode(wanted.phrase),
            top_k,
            score_ratio,
            caption_encoder.encode_query(wanted.phrase) if captioned else None,
            wanted.aspects,
        )

    started = time.perf_counter()
    parsed = query_parser.parse(text)  # a QueryParseError fails the test: shape is checked
    print(f"parse of {text!r} took {time.perf_counter() - started:.1f} s: {parsed}")
    matches = None
    if parsed.constraints:
        matches = run_spatial_query(
            rec.instances, parsed, rank, settings.query_top_k, settings.spatial
        )
    if matches is None:
        wanted = next(o for o in parsed.objects if o.object_id == parsed.target)
        matches = rank(wanted, settings.query_top_k, settings.query_score_ratio)
    return parsed, matches


def test_a_query_with_a_relation_returns_matches_with_anchors(
    reconstruction_result, query_parser, encoder, caption_encoder
):
    rec, _ = reconstruction_result
    assert len(rec.instances) >= 2, "a Query with an Anchor needs two Instances"
    text = os.environ.get(SMOKE_SPATIAL_QUERY_ENV, DEFAULT_SPATIAL_QUERY)
    parsed, matches = _answer(text, rec, query_parser, encoder, caption_encoder)
    assert parsed.constraints, f"the parser found no Relation in {text!r}"
    print(f"{text!r}: {[(m.instance_id, round(m.score, 3), m.anchors) for m in matches]}")
    assert 1 <= len(matches) <= configured_settings().query_top_k
    assert all(m.anchors for m in matches)


def test_a_query_without_a_relation_returns_no_anchors(
    reconstruction_result, query_parser, encoder, caption_encoder
):
    rec, _ = reconstruction_result
    parsed, matches = _answer("a chair", rec, query_parser, encoder, caption_encoder)
    assert parsed.constraints == ()
    assert matches
    assert all(m.anchors == () for m in matches)
