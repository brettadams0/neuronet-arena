import pytest

from neuronet.config import Config


def test_defaults_are_valid() -> None:
    Config()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("population", 0, "population must be positive"),
        ("vision", -1.0, "vision must be positive"),
        ("food_count", -1, "food_count must not be negative"),
        ("mutation_rate", 1.5, r"mutation_rate must be in \[0, 1\]"),
        ("selection", "roulette", "selection must be"),
    ],
)
def test_bad_values_are_rejected(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Config(**{field: value})  # type: ignore[arg-type]


def test_cross_field_checks() -> None:
    with pytest.raises(ValueError, match="elite cannot exceed population"):
        Config(population=2, elite=3)
    with pytest.raises(ValueError, match="start_energy cannot exceed max_energy"):
        Config(start_energy=200.0)
    with pytest.raises(ValueError, match="too small"):
        Config(width=40.0)


def test_dict_round_trip() -> None:
    cfg = Config(population=12, mutation_rate=0.2)
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown config keys: bogus"):
        Config.from_dict({"bogus": 1})


def test_integers_from_json_become_floats() -> None:
    # JavaScript serialises 960.0 as 960.
    cfg = Config.from_dict({"width": 960, "max_energy": 100})
    assert isinstance(cfg.width, float)
    assert isinstance(cfg.max_energy, float)
    assert isinstance(cfg.population, int)
