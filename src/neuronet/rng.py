"""Seeded random numbers, identical bit-for-bit to ``web/src/rng.js``.

Python's ``random`` module is a Mersenne Twister with no JavaScript twin, so
both engines carry this one instead: xoshiro128** for the stream, seeded
through splitmix32. It is all 32-bit integer arithmetic, which both languages
can do exactly — Python by masking, JavaScript with ``Math.imul`` and ``>>> 0``.

Every float handed out is built from integers by exact operations, and the
only "normal" sample is an Irwin-Hall sum rather than Box-Muller, because
Box-Muller needs ``log`` and ``cos`` and those are not required to round the
same way in every math library.
"""

from __future__ import annotations

import math

MASK = 0xFFFFFFFF

# 2**26 and 2**53: a 53-bit float in [0, 1) from a 27-bit and a 26-bit draw.
_TWO_26 = 67108864.0
_TWO_53 = 9007199254740992.0

# Four uniforms summed have variance 4/12; this scales it back to 1.
_SQRT3 = math.sqrt(3.0)


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & MASK


class Rng:
    """xoshiro128** with a serialisable state of four 32-bit words."""

    __slots__ = ("_s",)

    def __init__(self, seed: int) -> None:
        state = seed & MASK
        words = []
        for _ in range(4):
            state = (state + 0x9E3779B9) & MASK
            z = state
            z = ((z ^ (z >> 16)) * 0x85EBCA6B) & MASK
            z = ((z ^ (z >> 13)) * 0xC2B2AE35) & MASK
            words.append(z ^ (z >> 16))
        if not any(words):
            words[0] = 1  # the all-zero state is a fixed point
        self._s = words

    @classmethod
    def from_state(cls, state: list[int]) -> Rng:
        if len(state) != 4 or not all(0 <= w <= MASK for w in state) or not any(state):
            raise ValueError(f"invalid xoshiro128 state: {state!r}")
        rng = cls.__new__(cls)
        rng._s = list(state)
        return rng

    def state(self) -> list[int]:
        return list(self._s)

    def next_u32(self) -> int:
        s = self._s
        s0, s1, s2, s3 = s
        result = (_rotl((s1 * 5) & MASK, 7) * 9) & MASK
        t = (s1 << 9) & MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 11)
        s[0], s[1], s[2], s[3] = s0, s1, s2, s3
        return result

    def random(self) -> float:
        """A float in [0, 1) with 53 random bits."""
        a = self.next_u32() >> 5
        b = self.next_u32() >> 6
        return (a * _TWO_26 + b) / _TWO_53

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def below(self, n: int) -> int:
        """An int in [0, n)."""
        return min(int(self.random() * n), n - 1)

    def normal(self) -> float:
        """Approximately standard normal: Irwin-Hall with four terms."""
        u = self.random()
        u += self.random()
        u += self.random()
        u += self.random()
        return (u - 2.0) * _SQRT3
