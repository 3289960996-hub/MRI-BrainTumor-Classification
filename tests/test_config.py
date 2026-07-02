from config import get_nested


def test_get_nested_returns_existing_value():
    cfg = {"training": {"epochs": 10}}
    assert get_nested(cfg, "training.epochs") == 10


def test_get_nested_returns_default_for_missing_value():
    assert get_nested({}, "training.epochs", 10) == 10
