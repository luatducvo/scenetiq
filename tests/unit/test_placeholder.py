import pytest

import scenetiq


@pytest.mark.unit
def test_package_importable() -> None:
    assert scenetiq is not None
