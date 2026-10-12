from derby import config


def test_classes_are_the_six_in_fixed_order():
    assert config.CLASSES == ["bottle", "pen", "book", "phone", "backpack", "shoe"]


def test_race_constants():
    assert config.IMG == 96
    assert config.LAPS == 20
    assert config.BATCH == 64
    assert config.LAP_FLOOR_S == 4.0


def test_caps():
    assert config.CAPS == {"per_device": 3, "total": 120, "max_bytes": 5_000_000}
    assert config.MIN_LIVE_FOR_SPLIT == 12


def test_seven_horses_enabled_by_default():
    assert list(config.HORSES_ENABLED) == [
        "textbook", "stairs", "elevator", "featherweight", "spotlight", "trifocals", "influencer",
    ]
    assert all(config.HORSES_ENABLED.values())


def test_dirs_exist():
    assert config.DATA_DIR.name == "data"
    assert config.WEB_DIR.name == "web"
