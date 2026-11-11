import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import scenetiq
from scenetiq.config import CaptioningOptions
from scenetiq.domain import Aspect, Camera, Caption
from scenetiq.pipeline.captioning import (
    CAPTION_SCHEMA,
    CAPTIONING_LOGGER,
    MAX_SYNONYMS,
    UnusableReplyError,
    boxed_crop,
    caption_instances,
    embed_descriptions,
    parse_caption,
)
from scenetiq.pipeline.embedding import PlannedView

SOFA_REPLY = {
    "name": "sofa",
    "synonyms": ["couch", "settee"],
    "identity": "A two-seat sofa.",
    "appearance": "It is grey with wide armrests.",
    "material": "It is upholstered in fabric with wooden legs.",
    "use": "It is used for sitting.",
}
SOFA = json.dumps(SOFA_REPLY)
SOFA_CAPTION = Caption(
    "sofa",
    ("couch", "settee"),
    identity="A two-seat sofa.",
    appearance="It is grey with wide armrests.",
    material="It is upholstered in fabric with wooden legs.",
    use="It is used for sitting.",
)


def test_a_json_reply_becomes_a_caption():
    assert parse_caption(SOFA) == SOFA_CAPTION


def test_json_in_code_fences_or_among_other_text_is_found():
    assert parse_caption(f"```json\n{SOFA}\n```") == SOFA_CAPTION
    assert parse_caption(f"Here it is: {SOFA} Hope that helps.") == SOFA_CAPTION


def test_synonyms_beyond_three_are_dropped():
    reply = json.dumps({**SOFA_REPLY, "synonyms": ["couch", "settee", "divan", "lounge"]})
    assert parse_caption(reply).synonyms == ("couch", "settee", "divan")


def test_text_is_trimmed_and_unknown_or_non_text_values_are_ignored():
    reply = json.dumps(
        {
            "name": "  mug ",
            "synonyms": "cup",  # a single string where a list belongs is one item
            "identity": " A white mug. ",
            "appearance": 3,
            "material": None,
            "colour": "white",
        }
    )
    assert parse_caption(reply) == Caption("mug", ("cup",), identity="A white mug.")


def test_empty_descriptions_stay_empty_and_are_left_out_of_the_descriptions():
    reply = json.dumps({**SOFA_REPLY, "appearance": "", "use": "  "})
    caption = parse_caption(reply)
    assert caption.appearance == ""
    assert caption.use == ""
    assert list(caption.descriptions()) == [Aspect.IDENTITY, Aspect.MATERIAL]


@pytest.mark.parametrize(
    ("synonyms", "identity"),
    [
        (["couch", "settee"], "A sofa, also called a couch or settee."),
        (["couch"], "A sofa, also called a couch."),
        (["couch", "settee", "divan"], "A sofa, also called a couch, settee or divan."),
        ([], "A sofa."),
    ],
)
def test_an_empty_identity_is_written_from_the_name_and_synonyms(
    synonyms: list[str], identity: str
):
    reply = json.dumps({**SOFA_REPLY, "synonyms": synonyms, "identity": ""})
    assert parse_caption(reply).identity == identity


def test_a_name_starting_with_a_vowel_gets_an():
    reply = json.dumps({"name": "armchair", "synonyms": []})
    assert parse_caption(reply).identity == "An armchair."


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("I cannot tell what this is.", "no JSON object"),
        ('{"name": "sofa", "identity": "A two-se', "no JSON object"),
        ('{"name": ', "no JSON object"),
        ('{"name": "sofa",, }', "not valid JSON"),
        ('["sofa"]', "no JSON object"),
        ('{"name": "   "}', "no name"),
        ('{"synonyms": ["couch"]}', "no name"),
        ('{"name": 5}', "no name"),
        ("", "no JSON object"),
    ],
)
def test_a_reply_without_a_usable_name_is_refused_with_the_reason(reply: str, reason: str):
    with pytest.raises(UnusableReplyError, match=reason):
        parse_caption(reply)


def test_boxed_crop_adds_context_and_draws_the_tight_box():
    photo = Image.new("RGB", (100, 80))
    # Tight box columns 40..59, rows 30..49 (right/bottom exclusive: 60, 50). Growing each
    # side by 0.5 * 20 = 10 px gives the crop (30, 20, 70, 60), so the box sits at
    # (10, 10)..(29, 29) inside the 40 x 40 crop.
    u = np.array([40, 59])
    v = np.array([30, 49])
    crop = boxed_crop(photo, u, v, expand=0.5, color=(255, 0, 0))
    assert crop.size == (40, 40)
    assert crop.getpixel((10, 20)) == (255, 0, 0)  # left edge of the box
    assert crop.getpixel((29, 20)) == (255, 0, 0)  # right edge
    assert crop.getpixel((20, 10)) == (255, 0, 0)  # top edge
    assert crop.getpixel((20, 20)) == (0, 0, 0)  # inside the box: untouched
    assert crop.getpixel((2, 2)) == (0, 0, 0)  # context: untouched


def test_boxed_crop_darkens_everything_outside_the_instances_mask():
    photo = Image.new("RGB", (100, 80), (200, 100, 40))
    # The mask covers columns 40..59 and rows 30..39: the upper half of the tight box.
    mask = Image.new("L", (100, 80), 0)
    mask.paste(255, (40, 30, 60, 40))
    u = np.array([40, 59])
    v = np.array([30, 49])
    crop = boxed_crop(photo, u, v, expand=0.5, color=(255, 0, 0), mask=mask, dim=0.5)
    # The crop is (30, 20)..(70, 60), so the mask sits at (10, 10)..(29, 19) inside it.
    assert crop.getpixel((20, 15)) == (200, 100, 40)  # inside the mask: untouched
    assert crop.getpixel((20, 25)) == (100, 50, 20)  # in the box, outside the mask: halved
    assert crop.getpixel((2, 2)) == (100, 50, 20)  # context: halved
    assert crop.getpixel((10, 25)) == (255, 0, 0)  # the box is drawn over the darkened crop
    assert photo.getpixel((2, 2)) == (200, 100, 40)  # the photo is not modified


def test_boxed_crop_without_a_mask_or_at_full_brightness_darkens_nothing():
    photo = Image.new("RGB", (100, 80), (200, 100, 40))
    mask = Image.new("L", (100, 80), 0)
    u = np.array([40, 59])
    v = np.array([30, 49])
    assert boxed_crop(photo, u, v, 0.5, (255, 0, 0), dim=0.5).getpixel((2, 2)) == (200, 100, 40)
    full = boxed_crop(photo, u, v, 0.5, (255, 0, 0), mask=mask, dim=1.0)
    assert full.getpixel((2, 2)) == (200, 100, 40)


def test_an_instances_crops_are_darkened_outside_its_mask_when_its_plan_can_draw_one(
    tmp_path: Path,
):
    path = tmp_path / "0.png"
    Image.new("RGB", (100, 80), (200, 100, 40)).save(path)
    # Two points 10 px apart with a radius of 2: the mask is two 5 px squares.
    u, v = np.array([40, 50]), np.array([40, 40])
    plans = {1: [PlannedView(0, u, v, np.array([2, 2]))]}
    captioner = _RecordingCaptioner({1: SOFA})

    caption_instances(
        plans,
        [path],
        [_camera(100, 80)],
        captioner,
        CaptioningOptions(context_expand=1.0, background_dim=0.5),
    )

    # Columns 40..50 are 11 px, grown by 11 px per side: the crop starts at column 29. The
    # single row grows by (16 - 1) / 2 = 7.5 px to the 16 px minimum: the crop starts at row
    # 32. So the points are at (11, 8) and (21, 8) in the crop, their squares cover columns
    # 9..13 and 19..23 of rows 6..10, and the box is drawn along rows 8..9.
    [[crop]] = captioner.calls
    assert crop.getpixel((10, 7)) == (200, 100, 40)  # inside the first point's square
    assert crop.getpixel((16, 6)) == (100, 50, 20)  # between the two squares
    assert crop.getpixel((1, 1)) == (100, 50, 20)


class _RecordingCaptioner:
    def __init__(self, replies: dict[int, str]) -> None:
        self.replies = replies
        self.calls: list[list[Image.Image]] = []

    def describe(self, images: list[Image.Image]) -> str:
        self.calls.append(images)
        return self.replies.get(len(self.calls), "no idea")


def _camera(width: int, height: int) -> Camera:
    k = np.eye(3, dtype=np.float32)
    return Camera(np.eye(4, dtype=np.float32), k, width, height)


def test_each_instance_is_described_once_from_its_first_views(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=CAPTIONING_LOGGER)
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (200, 160)).save(path)  # twice the depth-map resolution
        paths.append(path)
    cameras = [_camera(100, 80)] * 3
    pixels = (np.array([10, 20]), np.array([10, 20]))
    plans = {
        4: [PlannedView(2, *pixels), PlannedView(0, *pixels), PlannedView(1, *pixels)],
        7: [PlannedView(1, *pixels)],
        9: [],
    }
    captioner = _RecordingCaptioner({1: SOFA})

    captions = caption_instances(
        plans, paths, cameras, captioner, CaptioningOptions(views_per_instance=2)
    )

    # Instance 4 gets its first two Views, Instance 7 its only one, Instance 9 none.
    assert [len(images) for images in captioner.calls] == [2, 1]
    # Pixels 10 and 20 at the depth resolution are 20 and 40 in the photo: a 21 px box,
    # grown by 0.6 * 21 = 12.6 px per side to columns 7..53, so 47 px wide.
    assert captioner.calls[0][0].size == (47, 47)
    assert captions == {4: SOFA_CAPTION}
    # Instance 7's reply ("no idea") is unusable: logged with the reason and the reply.
    assert any(
        "instance=7" in m and "no JSON object" in m and "no idea" in m for m in caplog.messages
    )
    assert "captioning captioned=1 uncaptioned=2" in caplog.messages


def test_colab_logging_lets_the_captioning_line_reach_stdout():
    path = Path(scenetiq.__file__).with_name("colab_logging.json")
    logger = json.loads(path.read_text(encoding="utf-8"))["loggers"][CAPTIONING_LOGGER]
    assert logger["level"] == "INFO"
    assert logger["handlers"] == ["progress"]
    assert logger["propagate"] is False


def test_images_are_shrunk_to_the_pixel_budget_keeping_their_shape():
    from scenetiq.pipeline.qwen_vl import fit_pixels

    assert fit_pixels(100, 80, 8000) == (100, 80)
    # 400 x 300 = 120,000 px; a 30,000 px budget halves each side.
    assert fit_pixels(400, 300, 30_000) == (200, 150)


def test_the_reply_schema_requires_name_synonyms_and_one_description_per_aspect():
    assert CAPTION_SCHEMA["required"] == [
        "name",
        "synonyms",
        "identity",
        "appearance",
        "material",
        "use",
    ]
    assert set(CAPTION_SCHEMA["properties"]) == set(CAPTION_SCHEMA["required"])
    assert CAPTION_SCHEMA["additionalProperties"] is False
    assert CAPTION_SCHEMA["properties"]["name"]["minLength"] == 1
    assert CAPTION_SCHEMA["properties"]["synonyms"]["maxItems"] == MAX_SYNONYMS == 3
    for aspect in ("identity", "appearance", "material", "use"):
        assert CAPTION_SCHEMA["properties"][aspect] == {"type": "string", "maxLength": 200}


def test_every_reply_the_schema_allows_becomes_a_caption():
    # The tightest reply the schema accepts: a one-letter Name, nothing else filled in.
    reply = {"name": "x", "synonyms": [], **dict.fromkeys(("identity", "appearance"), "")}
    reply |= dict.fromkeys(("material", "use"), "")
    assert parse_caption(json.dumps(reply)) == Caption("x", identity="A x.")


class _ListingEncoder:
    """Embeds the n-th document it is given as the basis vector e_n."""

    def __init__(self) -> None:
        self.documents: list[str] = []

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.documents.extend(texts)
        return np.eye(len(texts), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        raise AssertionError("not used")


def test_each_non_empty_description_is_embedded_once_and_kept_under_its_aspect():
    lamp = Caption("lamp", identity="A lamp.", use="It lights a room.")
    encoder = _ListingEncoder()

    embedded = embed_descriptions({4: SOFA_CAPTION, 9: lamp}, encoder)

    # Documents in Instance order, then Aspect order: four for the sofa, two for the lamp.
    assert encoder.documents == [
        "A two-seat sofa.",
        "It is grey with wide armrests.",
        "It is upholstered in fabric with wooden legs.",
        "It is used for sitting.",
        "A lamp.",
        "It lights a room.",
    ]
    assert embedded[4].caption == SOFA_CAPTION
    assert list(embedded[4].embeddings) == list(Aspect)
    assert list(embedded[9].embeddings) == [Aspect.IDENTITY, Aspect.USE]
    # The lamp's Use is the sixth document, so it is e_5.
    np.testing.assert_array_equal(embedded[9].embeddings[Aspect.USE], np.eye(6)[5])
