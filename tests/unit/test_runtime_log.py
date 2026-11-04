import logging

import pytest

from scenetiq.runtime_log import (
    HidePolling,
    JobProgress,
    UploadProgress,
    format_progress,
    parse_progress,
)


def test_progress_survives_the_trip_through_a_line():
    progress = JobProgress("j1", "s1", "running", "masking", 4, 6)
    line = format_progress(progress)
    assert "\n" not in line
    assert parse_progress(line) == progress


def test_upload_progress_survives_the_trip_through_a_line():
    progress = UploadProgress("c1", 120, 300)
    assert parse_progress(format_progress(progress)) == progress


@pytest.mark.parametrize(
    "line",
    ["", "job-progress", '2026-11-04 INFO: 1.2.3.4:0 - "GET /job-progress HTTP/1.1" 200'],
)
def test_other_lines_are_not_progress(line):
    assert parse_progress(line) is None


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:0", method, path, "1.1", status), None,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("method", "path", "status", "shown"),
    [
        ("GET", "/health", 200, False),
        ("OPTIONS", "/health", 200, False),
        ("GET", "/jobs/abc", 200, False),
        ("GET", "/jobs/abc?x=1", 200, False),
        ("GET", "/health", 401, True),
        ("GET", "/jobs/abc", 404, True),
        ("GET", "/jobs/abc", 500, True),
        ("GET", "/scenes", 200, True),
        ("POST", "/scenes", 201, True),
        ("POST", "/captures/abc/views", 204, True),
        ("GET", "/healthy", 200, True),
    ],
)
def test_hide_polling_drops_only_successful_polling(method, path, status, shown):
    assert HidePolling().filter(_access_record(method, path, status)) is shown


def test_hide_polling_keeps_a_record_of_any_other_shape():
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "plain", None, None)
    assert HidePolling().filter(record) is True
