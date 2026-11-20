import pytest

import scenetiq


@pytest.mark.smoke
def test_package_importable() -> None:
    assert scenetiq is not None
