from pathlib import Path

from scenetiq.storage import SceneStore


def test_views_keep_their_index_order_past_four_digits(tmp_path: Path):
    store = SceneStore(tmp_path / "store")
    try:
        capture_id = store.start_capture(10001)
        for index in (10000, 2, 9999):
            store.save_view(capture_id, index, ".png", b"photo")
        _, paths = store.create_scene_from_capture("Kitchen", capture_id)
    finally:
        store.close()
    assert [p.stem for p in paths] == ["0002", "9999", "10000"]
