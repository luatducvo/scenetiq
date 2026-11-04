"""GPU memory of the Pipeline: give back what a stage held, and report what it used."""

import gc
import logging
from collections.abc import Generator
from contextlib import contextmanager

from scenetiq.domain import Stage

GPU_LOGGER = "scenetiq.gpu"
"""Logger of the memory lines; ``colab_logging.json`` sends it to stdout unformatted."""

_logger = logging.getLogger(GPU_LOGGER)


def format_stage_memory(
    stage: str, peak_allocated: int, peak_reserved: int, allocated_after: int
) -> str:
    """Return the log line of one stage's GPU memory; arguments are in bytes, shown in GB."""
    return (
        f"gpu-memory stage={stage} peak_allocated={peak_allocated / 1e9:.2f}GB "
        f"peak_reserved={peak_reserved / 1e9:.2f}GB allocated_after={allocated_after / 1e9:.2f}GB"
    )


def format_merge_input(points: int, primitives: int) -> str:
    """Return the log line of the two counts that set the Merging stage's GPU memory."""
    return f"merge-input points={points} primitives={primitives}"


def report_merge_input(points: int, primitives: int) -> None:
    """Log the Merging stage's input size; it is only visible under ``colab_logging.json``."""
    _logger.info("%s", format_merge_input(points, primitives))


def free_gpu_memory() -> None:
    """Collect unreachable models and return PyTorch's cached GPU memory to the driver."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def track_stage_memory(stage: Stage) -> Generator[None]:
    """Log the peak GPU memory of the enclosed ``stage``, also when it raises.

    A Query running at the same time counts towards the peak, because PyTorch's
    counters are per device.
    """
    import torch

    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        line = format_stage_memory(
            stage.value,
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
            torch.cuda.memory_allocated(),
        )
        _logger.info("%s", line)
