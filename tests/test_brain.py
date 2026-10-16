import math

import pytest

from neuronet.brain import INPUTS, OUTPUTS, Brain, genome_size, layer_sizes, softsign
from neuronet.rng import Rng


def test_genome_size_counts_weights_and_biases() -> None:
    assert genome_size([8, 10, 2]) == 8 * 10 + 10 + 10 * 2 + 2
    assert genome_size([3, 1]) == 4


def test_layer_sizes_match_the_sensor_and_motor_counts() -> None:
    assert layer_sizes(10) == (INPUTS, 10, OUTPUTS) == (8, 10, 2)


def test_softsign_is_odd_bounded_and_monotonic() -> None:
    xs = [-1e6, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 1e6]
    ys = [softsign(x) for x in xs]
    assert ys == sorted(ys)
    assert all(-1.0 < y < 1.0 for y in ys)
    assert softsign(1.0) == 0.5
    assert softsign(-2.0) == -softsign(2.0)


def test_forward_follows_the_documented_genome_layout() -> None:
    # 2 inputs -> 1 hidden -> 1 output.
    # Layer 1: weights [w00, w01], bias b0. Layer 2: weight [v], bias c.
    genome = [0.5, -1.0, 0.25, 2.0, -0.5]
    brain = Brain([2, 1, 1], genome)
    x = [1.0, 0.5]
    h = softsign(0.25 + 0.5 * 1.0 + -1.0 * 0.5)
    expected = softsign(-0.5 + 2.0 * h)
    assert brain.forward(x) == [expected]


def test_activations_end_with_forward_output() -> None:
    brain = Brain.random([8, 10, 2], Rng(3))
    x = [0.1 * i for i in range(8)]
    layers = brain.activations(x)
    assert [len(layer) for layer in layers] == [8, 10, 2]
    assert layers[0] == x
    assert layers[-1] == brain.forward(x)


def test_random_init_is_glorot_uniform_with_zero_bias() -> None:
    brain = Brain.random([8, 10, 2], Rng(11))
    first = brain.genome[: 8 * 10]
    limit = math.sqrt(6.0 / 18)
    assert all(-limit <= w <= limit for w in first)
    assert brain.genome[80:90] == [0.0] * 10
    assert brain.genome[-2:] == [0.0, 0.0]


def test_random_init_is_reproducible() -> None:
    assert Brain.random([8, 10, 2], Rng(5)).genome == Brain.random([8, 10, 2], Rng(5)).genome


def test_wrong_genome_length_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"genome has 2 values, layers \[2, 1\] need 3"):
        Brain([2, 1], [0.0, 0.0])


def test_brain_copies_its_genome() -> None:
    genome = [0.0] * genome_size([2, 1])
    brain = Brain([2, 1], genome)
    genome[0] = 99.0
    assert brain.genome[0] == 0.0
