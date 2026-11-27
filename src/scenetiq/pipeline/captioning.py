import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from scenetiq.config import CaptioningOptions
from scenetiq.domain import Aspect, Camera, Caption, CaptionEncoder, EmbeddedCaption, FloatArray
from scenetiq.pipeline.embedding import ViewPlans, crop_box, photo_pixels

CAPTIONING_LOGGER = "scenetiq.captioning"

_logger = logging.getLogger(CAPTIONING_LOGGER)

MAX_SYNONYMS = 3

BOX_COLOR = (255, 0, 0)

_ASPECT_PROMPTS = {
    Aspect.IDENTITY: "what the object is and what else people call it",
    Aspect.APPEARANCE: "its colour, shape and how large it is",
    Aspect.MATERIAL: "what it is made of and its visible parts",
    Aspect.USE: "what it is used for",
}

CAPTION_PROMPT = (
    "The images show one object from different viewpoints; it is the object inside the red "
    "box. Where parts of an image are darkened, the object is only the part shown at full "
    "brightness, and the darkened parts are other things, even inside the red box. If the "
    "bright part is a surface such as a floor, a wall or a ceiling, the object is that "
    "surface, not what stands on it or hangs on it. Describe that object only, never what "
    "surrounds it. Reply with a single JSON "
    'object and nothing else, with the keys "name" (a short common name) and "synonyms" '
    "(up to three other names people use for it), and one key per aspect below, each holding "
    "one plain English sentence of at most 25 words about that aspect, or an empty string "
    "if the images do not show it: "
    + "; ".join(f'"{a.value}" ({text})' for a, text in _ASPECT_PROMPTS.items())
    + "."
)


MAX_NAME_LENGTH = 40

MAX_DESCRIPTION_LENGTH = 200

CAPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": MAX_NAME_LENGTH},
        "synonyms": {
            "type": "array",
            "items": {"type": "string", "maxLength": MAX_NAME_LENGTH},
            "maxItems": MAX_SYNONYMS,
        },
        **{a.value: {"type": "string", "maxLength": MAX_DESCRIPTION_LENGTH} for a in Aspect},
    },
    "required": ["name", "synonyms", *(a.value for a in Aspect)],
    "additionalProperties": False,
}


class UnusableReplyError(ValueError): ...


class Captioner(Protocol):
    def describe(self, images: list[Any]) -> str: ...


def parse_caption(reply: str) -> Caption:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise UnusableReplyError("no JSON object (the reply may be cut off)")
    try:
        raw = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as exc:
        raise UnusableReplyError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise UnusableReplyError("no JSON object")
    name = _text(raw.get("name"))
    if not name:
        raise UnusableReplyError("no name")
    synonyms = _texts(raw.get("synonyms"))[:MAX_SYNONYMS]
    descriptions = {a.value: _text(raw.get(a.value)) for a in Aspect}
    if not descriptions[Aspect.IDENTITY.value]:
        descriptions[Aspect.IDENTITY.value] = _identity(name, synonyms)
    return Caption(name=name, synonyms=synonyms, **descriptions)


def _identity(name: str, synonyms: tuple[str, ...]) -> str:
    text = f"{_article(name).capitalize()} {name}"
    if synonyms:
        others = ", ".join(synonyms[:-1]) + " or " if len(synonyms) > 1 else ""
        text += f", also called {_article(synonyms[0])} {others}{synonyms[-1]}"
    return text + "."


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _texts(value: object) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


def boxed_crop(
    photo: Any,
    u: npt.NDArray[np.int64],
    v: npt.NDArray[np.int64],
    expand: float,
    color: Sequence[int],
    mask: Any = None,
    dim: float = 1.0,
) -> Any:
    from PIL import Image, ImageDraw

    left, top, right, bottom = crop_box(u, v, 0.0, photo.width, photo.height)
    c_left, c_top, c_right, c_bottom = crop_box(
        u, v, expand, photo.width, photo.height, min_size=16
    )
    crop = photo.crop((c_left, c_top, c_right, c_bottom))
    if mask is not None and dim < 1:
        dark = crop.point(lambda value: round(value * dim))
        crop = Image.composite(crop, dark, mask.crop((c_left, c_top, c_right, c_bottom)))
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
    checkpoint: Callable[[], None] | None = None,
) -> dict[int, Caption]:
    opts = options or CaptioningOptions()
    chosen = {iid: plan[: opts.views_per_instance] for iid, plan in plans.items() if plan}
    crops: dict[int, list[tuple[int, Any]]] = {iid: [] for iid in chosen}
    for seen in photo_pixels(chosen, image_paths, cameras):
        crop = boxed_crop(
            seen.photo,
            seen.u,
            seen.v,
            opts.context_expand,
            BOX_COLOR,
            seen.mask,
            opts.background_dim,
        )
        crops[seen.instance_id].append((seen.rank, crop))

    captions: dict[int, Caption] = {}
    for iid, ordered in crops.items():
        images = [crop for _, crop in sorted(ordered, key=lambda pair: pair[0])]
        if checkpoint is not None:
            checkpoint()
        reply = captioner.describe(images)
        try:
            captions[iid] = parse_caption(reply)
        except UnusableReplyError as exc:
            _logger.info("captioning instance=%d no Caption: %s; reply=%.200r", iid, exc, reply)
    _logger.info(
        "captioning captioned=%d uncaptioned=%d", len(captions), len(plans) - len(captions)
    )
    return captions


def embed_descriptions(
    captions: dict[int, Caption], encoder: CaptionEncoder
) -> dict[int, EmbeddedCaption]:
    owners = [(iid, aspect) for iid, c in captions.items() for aspect in c.descriptions()]
    embeddings = encoder.encode_documents([captions[iid].description(a) for iid, a in owners])
    by_aspect: dict[int, dict[Aspect, FloatArray]] = {iid: {} for iid in captions}
    for row, (iid, aspect) in enumerate(owners):
        by_aspect[iid][aspect] = embeddings[row]
    return {iid: EmbeddedCaption(c, by_aspect[iid]) for iid, c in captions.items()}
