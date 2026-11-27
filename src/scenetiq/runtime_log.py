import json
import logging
from dataclasses import asdict, dataclass

PROGRESS_LOGGER = "scenetiq.progress"

_JOB_PREFIX = "job-progress "
_UPLOAD_PREFIX = "upload-progress "

_logger = logging.getLogger(PROGRESS_LOGGER)


@dataclass(frozen=True)
class JobProgress:
    job_id: str
    scene_id: str
    status: str
    stage: str
    step: int
    total: int


@dataclass(frozen=True)
class UploadProgress:
    capture_id: str
    received: int
    total: int


def format_progress(progress: JobProgress | UploadProgress) -> str:
    prefix = _JOB_PREFIX if isinstance(progress, JobProgress) else _UPLOAD_PREFIX
    return prefix + json.dumps(asdict(progress))


def parse_progress(line: str) -> JobProgress | UploadProgress | None:
    if line.startswith(_JOB_PREFIX):
        return JobProgress(**json.loads(line[len(_JOB_PREFIX) :]))
    if line.startswith(_UPLOAD_PREFIX):
        return UploadProgress(**json.loads(line[len(_UPLOAD_PREFIX) :]))
    return None


def report_progress(progress: JobProgress | UploadProgress) -> None:
    _logger.info("%s", format_progress(progress))


class HidePolling(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        _, method, path, _, status_code = args
        route = str(path).split("?")[0]
        polled = route == "/health" or route.startswith("/jobs/")
        return not (polled and method in ("GET", "OPTIONS") and int(str(status_code)) < 400)
