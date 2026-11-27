from collections.abc import Callable

from scenetiq.runtime_log import JobProgress, UploadProgress


def progress_bar(fraction: float, width: int) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def elapsed_clock(elapsed: float) -> str:
    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes:d}:{seconds:02d}"


def render_job_bar(progress: JobProgress | None, elapsed: float, width: int = 24) -> str:
    if progress is None:
        return f"[{progress_bar(0.0, width)}] no job yet"
    if progress.status == "succeeded":
        fraction = 1.0
        state = "succeeded"
    else:
        fraction = max(progress.step - 1, 0) / progress.total
        if progress.status in ("failed", "cancelled"):
            stopped = progress.status
            state = f"{stopped} at {progress.stage}" if progress.stage else stopped
        else:
            state = f"{progress.step}/{progress.total} {progress.stage}"
    bar = progress_bar(fraction, width)
    return f"[{bar}] {state} · scene {progress.scene_id[:8]} · {elapsed_clock(elapsed)}"


def render_upload_bar(progress: UploadProgress, elapsed: float, width: int = 24) -> str:
    bar = progress_bar(progress.received / progress.total, width)
    count = f"upload {progress.received}/{progress.total} Views"
    return f"[{bar}] {count} · capture {progress.capture_id[:8]} · {elapsed_clock(elapsed)}"


def live_display() -> Callable[[str], None]:
    try:
        from IPython.display import Pretty, display
    except ImportError:
        return print
    handle = display(Pretty(""), display_id=True)
    if handle is None:
        return print
    return lambda text: handle.update(Pretty(text))
