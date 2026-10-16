"""The arena: a rectangle with food that respawns when eaten and fixed hazards.

Positions are parallel lists of floats rather than objects, because the
sensing loop touches every item for every agent on every step and attribute
lookups are the expensive part of that in CPython.
"""

from __future__ import annotations

from neuronet.config import Config
from neuronet.rng import Rng

# Rejection-sampling budget when looking for a point clear of every hazard.
# At the default density the exclusion zones cover about 7% of the arena, so
# the first try almost always lands; the cap only stops a pathological config
# from looping forever.
_PLACEMENT_TRIES = 32


class World:
    __slots__ = ("cfg", "food_x", "food_y", "hazard_x", "hazard_y")

    def __init__(self, cfg: Config, rng: Rng) -> None:
        self.cfg = cfg
        self.hazard_x: list[float] = []
        self.hazard_y: list[float] = []
        self.food_x: list[float] = []
        self.food_y: list[float] = []

        margin = 2 * cfg.hazard_radius
        for _ in range(cfg.hazard_count):
            self.hazard_x.append(rng.uniform(margin, cfg.width - margin))
            self.hazard_y.append(rng.uniform(margin, cfg.height - margin))
        for _ in range(cfg.food_count):
            x, y = self.free_point(rng, cfg.food_radius)
            self.food_x.append(x)
            self.food_y.append(y)

    def free_point(self, rng: Rng, radius: float) -> tuple[float, float]:
        """A uniformly random point whose ``radius``-disc clears every hazard.

        The clearance includes an agent's diameter, so food is never placed
        where reaching it means touching a hazard: avoiding hazards is always
        possible, which is what makes it worth evolving.
        """
        cfg = self.cfg
        margin = 2 * radius
        clear = cfg.hazard_radius + radius + 2 * cfg.agent_radius
        clear2 = clear * clear
        x = y = 0.0
        for _ in range(_PLACEMENT_TRIES):
            x = rng.uniform(margin, cfg.width - margin)
            y = rng.uniform(margin, cfg.height - margin)
            for hx, hy in zip(self.hazard_x, self.hazard_y, strict=True):
                dx = hx - x
                dy = hy - y
                if dx * dx + dy * dy < clear2:
                    break
            else:
                return x, y
        return x, y

    def respawn_food(self, index: int, rng: Rng) -> None:
        x, y = self.free_point(rng, self.cfg.food_radius)
        self.food_x[index] = x
        self.food_y[index] = y
