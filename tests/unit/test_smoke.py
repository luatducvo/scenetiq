import scenetiq


def test_package_exposes_entry_point():
    assert callable(scenetiq.main)
