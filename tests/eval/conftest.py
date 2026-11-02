import os
from pathlib import Path

import pytest

from scenetiq.domain import Capture, View

SMOKE_CAPTURE_ENV = "SCENETIQ_SMOKE_CAPTURE"


@pytest.fixture(scope="session")
def smoke_capture() -> Capture:
    """A small real Capture: a directory of 10+ JPG/PNG photos named by SCENETIQ_SMOKE_CAPTURE."""
    root = os.environ.get(SMOKE_CAPTURE_ENV)
    if not root:
        pytest.skip(f"set {SMOKE_CAPTURE_ENV} to a directory of photos of one static Scene")
    paths = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return Capture(views=[View(path=p) for p in paths])
