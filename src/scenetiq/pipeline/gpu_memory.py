import gc
import logging
from collections.abc import Generator
from contextlib import contextmanager

from scenetiq.domain.scene import Stage

GPU_LOGGER = "scenetiq.gpu"

_logger = logging.getLogger(GPU_LOGGER)


def format_stage_memory(
    stage: str, peak_allocated: int, peak_reserved: int, allocated_after: int
) -> str:
    return (
        f"gpu-memory stage={stage} peak_allocated={peak_allocated / 1e9:.2f}GB "
        f"peak_reserved={peak_reserved / 1e9:.2f}GB allocated_after={allocated_after / 1e9:.2f}GB"
    )


def format_merge_input(points: int, primitives: int) -> str:
    return f"merge-input points={points} primitives={primitives}"


def report_merge_input(points: int, primitives: int) -> None:
    _logger.info("%s", format_merge_input(points, primitives))


def free_gpu_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def track_stage_memory(stage: Stage) -> Generator[None]:
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
