import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import scenetiq
from scenetiq.config import CaptioningOptions
from scenetiq.domain import Camera, Caption
from scenetiq.pipeline.captioning import (
    CAPTIONING_LOGGER,
    boxed_crop,
    caption_document,
    caption_instances,
    parse_caption,
)
from scenetiq.pipeline.embedding import PlannedView

SOFA = '{"name": "sofa", "synonyms": ["couch", "settee"], "attributes": ["grey", "fabric"], "use": "sitting"}'  # noqa: E501
SOFA_CAPTION = Caption("sofa", ("couch", "settee"), ("grey", "fabric"), "sitting")


def test_a_json_reply_becomes_a_caption():
    assert parse_caption(SOFA) == SOFA_CAPTION


def test_json_in_code_fences_or_among_other_text_is_found():
    assert parse_caption(f"```json\n{SOFA}\n```") == SOFA_CAPTION
    assert parse_caption(f"Here it is: {SOFA} Hope that helps.") == SOFA_CAPTION


def test_synonyms_beyond_three_are_dropped():
    reply = '{"name": "sofa", "synonyms": ["couch", "settee", "divan", "lounge"]}'
    assert parse_caption(reply) == Caption("sofa", ("couch", "settee", "divan"))


def test_missing_fields_default_and_text_is_trimmed():
    assert parse_caption('{"name": "  lamp "}') == Caption("lamp")
    # A single string where a list belongs is one item; non-text items are dropped.
    reply = '{"name": "mug", "synonyms": "cup", "attributes": ["white", 3], "use": null}'
    assert parse_caption(reply) == Caption("mug", ("cup",), ("white",), "")


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot tell what this is.",
        '{"name": ',
        '{"name": "   "}',
        '{"synonyms": ["couch"]}',
        '{"name": 5}',
        '["sofa"]',
        "",
    ],
)
def test_a_reply_without_a_usable_name_gives_no_caption(reply: str):
    assert parse_caption(reply) is None


def test_the_caption_document_joins_every_part():
    assert (
        caption_document(SOFA_CAPTION)
        == "sofa; also: couch, settee; grey, fabric; used for sitting"
    )
    assert caption_document(Caption("lamp")) == "lamp"


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


class _RecordingCaptioner:
    def __init__(self, replies: dict[int, str]) -> None:
        self.replies = replies
        self.calls: list[list[Image.Image]] = []

    def describe(self, images: list[Image.Image]) -> str:
        self.calls.append(images)
        return self.replies.get(len(self.calls), "no idea")

    def release(self) -> None:
        pass


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
