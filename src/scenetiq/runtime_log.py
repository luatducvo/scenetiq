"""What the backend tells the Colab notebook: progress lines and request log filtering.

The server runs in its own process, and the notebook reads its stdout. Standard library only,
because the notebook imports this through :mod:`scenetiq.colab` before anything is installed.
"""

import json
import logging
from dataclasses import asdict, dataclass

PROGRESS_LOGGER = "scenetiq.progress"
"""Logger of the progress lines; ``colab_logging.json`` sends it to stdout unformatted."""

_JOB_PREFIX = "job-progress "
_UPLOAD_PREFIX = "upload-progress "

_logger = logging.getLogger(PROGRESS_LOGGER)


@dataclass(frozen=True)
class JobProgress:
    """How far a job has come, as sent from the server to the notebook.

    Attributes:
        job_id: The job.
        scene_id: The Scene it reconstructs.
        status: ``"running"``, ``"succeeded"`` or ``"failed"``.
        stage: The current Stage's name; empty before the first Stage starts.
        step: 1-based position of ``stage``; 0 before the first Stage starts.
        total: Number of Stages in a job.
    """

    job_id: str
    scene_id: str
    status: str
    stage: str
    step: int
    total: int


@dataclass(frozen=True)
class UploadProgress:
    """How many Views of a Capture the server has stored so far.

    Attributes:
        capture_id: The staged Capture.
        received: Distinct Views stored; a retried View is not counted twice.
        total: Views the Capture was started with.
    """

    capture_id: str
    received: int
    total: int


def format_progress(progress: JobProgress | UploadProgress) -> str:
    """Return ``progress`` as one line that :func:`parse_progress` reads back."""
    prefix = _JOB_PREFIX if isinstance(progress, JobProgress) else _UPLOAD_PREFIX
    return prefix + json.dumps(asdict(progress))


def parse_progress(line: str) -> JobProgress | UploadProgress | None:
    """Return the progress held by a line of server output, or ``None`` for any other line."""
    if line.startswith(_JOB_PREFIX):
        return JobProgress(**json.loads(line[len(_JOB_PREFIX) :]))
    if line.startswith(_UPLOAD_PREFIX):
        return UploadProgress(**json.loads(line[len(_UPLOAD_PREFIX) :]))
    return None


def report_progress(progress: JobProgress | UploadProgress) -> None:
    """Log ``progress``; it is only visible under ``colab_logging.json``."""
    _logger.info("%s", format_progress(progress))


class HidePolling(logging.Filter):
    """Drops successful requests the frontend repeats on its own, keeping their failures.

    It sees uvicorn access records, whose args are
    ``(client, method, path, http_version, status_code)``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether the access line for ``record`` should be shown."""
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        _, method, path, _, status_code = args
        route = str(path).split("?")[0]
        polled = route == "/health" or route.startswith("/jobs/")
        return not (polled and method in ("GET", "OPTIONS") and int(str(status_code)) < 400)
