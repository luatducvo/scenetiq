"""GET /options: the settings the frontend configures itself from."""

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from scenetiq.config import Settings

from .conftest import AUTH, TOKEN


def test_options_give_the_view_limits_and_how_to_prepare_photos(client: TestClient):
    options = client.get("/options", headers=AUTH).json()

    assert options["min_views"] == 1
    assert options["max_views"] == 2000
    assert options["upload_max_side"] == 2048
    assert options["upload_jpeg_quality"] == 0.9


def test_options_follow_the_config(make_client: Callable[..., TestClient], tmp_path: Path):
    settings = Settings.from_overrides(
        TOKEN,
        {
            "storage_root": str(tmp_path / "s"),
            "min_views": 3,
            "max_views": 40,
            "upload_max_side": 1600,
            "upload_jpeg_quality": 0.8,
        },
    )
    options = make_client(settings=settings).get("/options", headers=AUTH).json()

    assert (options["min_views"], options["max_views"]) == (3, 40)
    assert (options["upload_max_side"], options["upload_jpeg_quality"]) == (1600, 0.8)


def test_options_need_the_token(client: TestClient):
    assert client.get("/options").status_code == 401
