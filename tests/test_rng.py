import pytest

from neuronet.rng import MASK, Rng


def test_same_seed_same_stream() -> None:
    a, b = Rng(123), Rng(123)
    assert [a.next_u32() for _ in range(100)] == [b.next_u32() for _ in range(100)]


def test_different_seeds_differ() -> None:
    assert [Rng(1).next_u32() for _ in range(4)] != [Rng(2).next_u32() for _ in range(4)]


def test_seed_zero_is_usable() -> None:
    rng = Rng(0)
    assert len({rng.next_u32() for _ in range(100)}) == 100


def test_outputs_are_in_range() -> None:
    rng = Rng(9)
    for _ in range(10_000):
        assert 0 <= rng.next_u32() <= MASK
        assert 0.0 <= rng.random() < 1.0
        assert 0 <= rng.below(7) < 7
        assert -3.0 <= rng.uniform(-3.0, 5.0) < 5.0


def test_below_covers_every_value_roughly_evenly() -> None:
    rng = Rng(4)
    counts = [0] * 5
    for _ in range(50_000):
        counts[rng.below(5)] += 1
    assert all(9_400 < c < 10_600 for c in counts)


def test_normal_has_unit_moments() -> None:
    rng = Rng(5)
    xs = [rng.normal() for _ in range(40_000)]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert abs(mean) < 0.02
    assert abs(var - 1.0) < 0.03
    # Irwin-Hall with four terms is bounded at ±2√3.
    assert max(abs(x) for x in xs) <= 2 * 3**0.5


def test_state_round_trip_resumes_the_stream() -> None:
    rng = Rng(77)
    for _ in range(10):
        rng.next_u32()
    copy = Rng.from_state(rng.state())
    assert [rng.random() for _ in range(20)] == [copy.random() for _ in range(20)]


@pytest.mark.parametrize("state", [[0, 0, 0, 0], [1, 2, 3], [1, 2, 3, MASK + 1], [-1, 0, 0, 1]])
def test_from_state_rejects_invalid_states(state: list[int]) -> None:
    with pytest.raises(ValueError, match="invalid xoshiro128 state"):
        Rng.from_state(state)
