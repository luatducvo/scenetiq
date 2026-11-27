import json
from pathlib import Path

import scenetiq
from scenetiq.pipeline.gpu_memory import GPU_LOGGER, format_merge_input, format_stage_memory


def test_stage_memory_line_shows_bytes_as_gigabytes():
    line = format_stage_memory("masking", 8_210_000_000, 9_500_000_000, 0)
    assert line == (
        "gpu-memory stage=masking peak_allocated=8.21GB peak_reserved=9.50GB allocated_after=0.00GB"
    )


def test_merge_input_line_names_both_counts():
    assert format_merge_input(1_200_000, 30_000) == "merge-input points=1200000 primitives=30000"


def test_colab_logging_lets_memory_lines_reach_stdout():
    path = Path(scenetiq.__file__).with_name("colab_logging.json")
    logger = json.loads(path.read_text(encoding="utf-8"))["loggers"][GPU_LOGGER]
    assert logger["level"] == "INFO"
    assert logger["handlers"] == ["progress"]
    assert logger["propagate"] is False
