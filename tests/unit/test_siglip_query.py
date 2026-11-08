import json
from pathlib import Path

import pytest

import scenetiq
from scenetiq.pipeline.siglip import QUERY_LOGGER, siglip_query_text

PROMPT = "This is a photo of {}."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("lamp", "this is a photo of lamp."),
        ("Red Chair", "this is a photo of red chair."),
        ("  a big   red chair  ", "this is a photo of a big   red chair."),  # 4 words
        ("a big soft red chair", "a big soft red chair"),  # 5 words: as written
        (
            "The grey sofa next to the window, where I watch TV.",
            "the grey sofa next to the window, where i watch tv.",
        ),
    ],
)
def test_queries_of_up_to_four_words_are_wrapped_and_longer_ones_are_not(text: str, expected: str):
    assert siglip_query_text(text, PROMPT, prompt_max_words=4) == expected


def test_zero_words_means_the_prompt_is_never_used():
    assert siglip_query_text("Lamp", PROMPT, prompt_max_words=0) == "lamp"
    assert siglip_query_text("  ", PROMPT, prompt_max_words=0) == ""


def test_the_whole_text_is_lowercased_including_the_prompt():
    assert siglip_query_text("Lamp", "A PHOTO OF {}", prompt_max_words=1) == "a photo of lamp"


def test_colab_logging_lets_the_query_line_reach_stdout():
    path = Path(scenetiq.__file__).with_name("colab_logging.json")
    logger = json.loads(path.read_text(encoding="utf-8"))["loggers"][QUERY_LOGGER]
    assert logger == {"handlers": ["progress"], "level": "INFO", "propagate": False}
