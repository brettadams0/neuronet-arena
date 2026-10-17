"""A fully connected feedforward network over plain Python lists.

The weights live in one flat list — the *genome* — because that is what
evolution operates on. Layout, per layer in order: the weight matrix
row-major (one row of ``n_in`` weights per output neuron), then the
``n_out`` biases. ``web/src/brain.js`` reads the same layout.

The activation is softsign, ``x / (1 + |x|)``, not tanh. They have the same
shape, but tanh is not required to be correctly rounded, and CPython's libm and
V8 disagree on it in the last bit. That never changed a score in testing, but
it does move agents by fractions of a pixel, which would force the cross-engine
tests to allow a tolerance. Softsign is one add, one abs and one divide, all of
which IEEE 754 defines exactly, so both engines compute the same bits and the
tests can demand exact equality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from neuronet.rng import Rng

INPUT_LABELS = (
    "food: near",
    "food: ahead",
    "food: right",
    "hazard: near",
    "hazard: ahead",
    "hazard: right",
    "wall ahead",
    "energy",
)
OUTPUT_LABELS = ("steer", "throttle")

INPUTS = len(INPUT_LABELS)
OUTPUTS = len(OUTPUT_LABELS)


def layer_sizes(hidden: int) -> tuple[int, int, int]:
    return (INPUTS, hidden, OUTPUTS)


def genome_size(layers: Sequence[int]) -> int:
    return sum(n_out * n_in + n_out for n_in, n_out in pairwise(layers))


def softsign(x: float) -> float:
    return x / (1.0 + abs(x))


class Brain:
    __slots__ = ("_compiled", "genome", "layers")

    def __init__(self, layers: Sequence[int], genome: Sequence[float]) -> None:
        if len(layers) < 2:
            raise ValueError("a network needs at least an input and an output layer")
        expected = genome_size(layers)
        if len(genome) != expected:
            raise ValueError(
                f"genome has {len(genome)} values, layers {list(layers)} need {expected}"
            )
        self.layers = tuple(layers)
        self.genome = list(genome)

        # Slice the flat genome into per-neuron rows once, so the forward pass
        # is plain list iteration rather than index arithmetic.
        compiled: list[tuple[list[list[float]], list[float]]] = []
        offset = 0
        for n_in, n_out in pairwise(self.layers):
            rows = []
            for _ in range(n_out):
                rows.append(self.genome[offset : offset + n_in])
                offset += n_in
            biases = self.genome[offset : offset + n_out]
            offset += n_out
            compiled.append((rows, biases))
        self._compiled = compiled

    @classmethod
    def random(cls, layers: Sequence[int], rng: Rng) -> Brain:
        """Glorot-uniform weights, zero biases."""
        genome: list[float] = []
        for n_in, n_out in pairwise(layers):
            limit = math.sqrt(6.0 / (n_in + n_out))
            for _ in range(n_out * n_in):
                genome.append(rng.uniform(-limit, limit))
            genome.extend([0.0] * n_out)
        return cls(layers, genome)

    def forward(self, inputs: Sequence[float]) -> list[float]:
        x = list(inputs)
        for rows, biases in self._compiled:
            out = []
            for row, bias in zip(rows, biases, strict=True):
                # An explicit loop, never sum(): from Python 3.12, sum() of
                # floats is compensated and would no longer match JavaScript.
                acc = bias
                for w, v in zip(row, x, strict=True):
                    acc += w * v
                out.append(acc / (1.0 + abs(acc)))
            x = out
        return x

    def activations(self, inputs: Sequence[float]) -> list[list[float]]:
        """Every layer's values, input first — what the visualisers draw."""
        layers = [list(inputs)]
        x = layers[0]
        for rows, biases in self._compiled:
            out = []
            for row, bias in zip(rows, biases, strict=True):
                acc = bias
                for w, v in zip(row, x, strict=True):
                    acc += w * v
                out.append(acc / (1.0 + abs(acc)))
            layers.append(out)
            x = out
        return layers
