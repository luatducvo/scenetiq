"""Captioning stage: a Caption per Instance from boxed crops of its best Views (ADR 0008).

For every Instance, take the first Views of the Embedding stage's plan, crop the Instance
from the original photos with some context and its tight box drawn on, and ask a
vision-language model to describe the boxed Instance as JSON. Replies are parsed tolerantly;
an Instance whose reply has no usable Name gets no Caption.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from scenetiq.config import CaptioningOptions
from scenetiq.domain import Camera, Caption
from scenetiq.pipeline.embedding import ViewPlans, crop_box, photo_pixels

CAPTIONING_LOGGER = "scenetiq.captioning"
"""Logger of the per-job Captioning summary; ``colab_logging.json`` lets it through."""

_logger = logging.getLogger(CAPTIONING_LOGGER)

MAX_SYNONYMS = 3
"""Most Synonyms a Caption keeps; any beyond are dropped."""

BOX_COLOR = (255, 0, 0)
"""RGB colour of the box drawn around the Instance; :data:`CAPTION_PROMPT` calls it red."""

CAPTION_PROMPT = (
    "The images show one object from different viewpoints; it is the object inside the red "
    "box. Describe that object only. Reply with a single JSON object and nothing else, with "
    'the keys "name" (a short common name), "synonyms" (up to three other names people use '
    'for it), "attributes" (visible properties such as colour and material) and "use" (what '
    "it is used for, in a few words). Write in English."
)
"""The instruction sent with every Instance's images."""


class Captioner(Protocol):
    """A vision-language model that describes the boxed Instance in some images."""

    def describe(self, images: list[Any]) -> str:
        """Return the model's raw reply for PIL ``images`` of one Instance."""
        ...

    def release(self) -> None:
        """Free the model's GPU memory; called when the Captioning stage ends."""
        ...


def parse_caption(reply: str) -> Caption | None:
    """Read a Caption from a model reply.

    Accepts the JSON object bare, in code fences or surrounded by other text. Synonyms
    beyond three are dropped, a single string where a list belongs counts as one item, and
    non-text items are ignored.

    Returns:
        The Caption, or ``None`` when the reply holds no JSON object with a non-empty
        text ``name``.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        raw = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    use = raw.get("use")
    return Caption(
        name=name.strip(),
        synonyms=_texts(raw.get("synonyms"))[:MAX_SYNONYMS],
        attributes=_texts(raw.get("attributes")),
        use=use.strip() if isinstance(use, str) else "",
    )


def _texts(value: object) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def caption_document(caption: Caption) -> str:
    """The one text the Caption encoder embeds for ``caption``.

    Example: ``"sofa; also: couch, settee; grey, fabric; used for sitting"``.
    """
    parts = [caption.name]
    if caption.synonyms:
        parts.append("also: " + ", ".join(caption.synonyms))
    if caption.attributes:
        parts.append(", ".join(caption.attributes))
    if caption.use:
        parts.append("used for " + caption.use)
    return "; ".join(parts)


def boxed_crop(
    photo: Any,
    u: npt.NDArray[np.int64],
    v: npt.NDArray[np.int64],
    expand: float,
    color: Sequence[int],
) -> Any:
    """Crop the pixels' box from ``photo`` with context, and draw the tight box on it.

    Args:
        photo: PIL RGB image.
        u: Columns of the Instance's pixels in ``photo``.
        v: Rows of the Instance's pixels in ``photo``.
        expand: Context added on each side, as a fraction of the box size.
        color: RGB colour of the drawn box.

    Returns:
        A new PIL image; ``photo`` is not modified.
    """
    from PIL import ImageDraw

    left, top, right, bottom = crop_box(u, v, 0.0, photo.width, photo.height)
    c_left, c_top, c_right, c_bottom = crop_box(
        u, v, expand, photo.width, photo.height, min_size=16
    )
    crop = photo.crop((c_left, c_top, c_right, c_bottom))
    width = max(2, round(max(crop.size) / 150))
    ImageDraw.Draw(crop).rectangle(
        (left - c_left, top - c_top, right - 1 - c_left, bottom - 1 - c_top),
        outline=tuple(color),
        width=width,
    )
    return crop


def caption_instances(
    plans: ViewPlans,
    image_paths: Sequence[Path],
    cameras: Sequence[Camera],
    captioner: Captioner,
    options: CaptioningOptions | None = None,
) -> dict[int, Caption]:
    """Caption every Instance from the first Views of its plan.

    Args:
        plans: The Views chosen for every Instance, best first, with pixels at the
            depth-map resolution (:func:`scenetiq.pipeline.embedding.plan_views`).
        image_paths: Original photo of every View, in View order.
        cameras: Camera of every View; its size is the depth-map resolution.
        captioner: The vision-language model.
        options: Stage settings.

    Returns:
        Instance id -> Caption, for the Instances whose reply had a usable Name.
    """
    opts = options or CaptioningOptions()
    chosen = {iid: plan[: opts.views_per_instance] for iid, plan in plans.items() if plan}
    crops: dict[int, list[tuple[int, Any]]] = {iid: [] for iid in chosen}
    for seen in photo_pixels(chosen, image_paths, cameras):
        crop = boxed_crop(seen.photo, seen.u, seen.v, opts.context_expand, BOX_COLOR)
        crops[seen.instance_id].append((seen.rank, crop))

    captions: dict[int, Caption] = {}
    for iid, ordered in crops.items():
        images = [crop for _, crop in sorted(ordered, key=lambda pair: pair[0])]
        caption = parse_caption(captioner.describe(images))
        if caption is not None:
            captions[iid] = caption
    _logger.info(
        "captioning captioned=%d uncaptioned=%d", len(captions), len(plans) - len(captions)
    )
    return captions
