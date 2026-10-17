"""An agent: sense the world, run the brain, move, eat, take damage.

Everything an agent perceives is egocentric — "the nearest food is close,
slightly ahead and to the right" — rather than in world coordinates, so the
network does not have to learn trigonometry before it can learn to forage.

Heading is a unit vector, never an angle. Turning by an angle needs ``cos``
and ``sin``, which (like ``tanh``) are not guaranteed to round identically in
Python and JavaScript. Instead a turn is the Cayley rotation

    cos θ = (1 − t²) / (1 + t²),   sin θ = 2t / (1 + t²)

which is an exact rotation by θ = 2·atan(t) built from nothing but
arithmetic. ``t = steer × turn_rate`` with ``steer`` in (−1, 1).
"""

from __future__ import annotations

import math

from neuronet.brain import Brain
from neuronet.rng import Rng
from neuronet.world import World


class Agent:
    __slots__ = (
        "age",
        "alive",
        "brain",
        "energy",
        "food_eaten",
        "hazard_steps",
        "hx",
        "hy",
        "inputs",
        "speed",
        "steer",
        "throttle",
        "x",
        "y",
    )

    def __init__(
        self, brain: Brain, x: float, y: float, hx: float, hy: float, energy: float
    ) -> None:
        self.brain = brain
        self.x = x
        self.y = y
        self.hx = hx
        self.hy = hy
        self.energy = energy
        self.alive = True
        self.age = 0
        self.food_eaten = 0
        self.hazard_steps = 0
        self.inputs = [0.0] * 8
        self.steer = 0.0
        self.throttle = 0.0
        self.speed = 0.0

    @classmethod
    def spawn(cls, brain: Brain, world: World, rng: Rng) -> Agent:
        cfg = world.cfg
        x, y = world.free_point(rng, cfg.agent_radius)
        # A uniformly random direction without cos/sin: rejection-sample the
        # unit disc, then normalise.
        while True:
            vx = rng.uniform(-1.0, 1.0)
            vy = rng.uniform(-1.0, 1.0)
            d2 = vx * vx + vy * vy
            if 1e-9 < d2 <= 1.0:
                break
        d = math.sqrt(d2)
        return cls(brain, x, y, vx / d, vy / d, cfg.start_energy)

    @property
    def fitness(self) -> tuple[int, int]:
        """Food eaten, with survival time as the tie-break."""
        return (self.food_eaten, self.age)

    def sense(self, world: World) -> list[float]:
        cfg = world.cfg
        x, y, hx, hy = self.x, self.y, self.hx, self.hy
        vision = cfg.vision

        food = _nearest(world.food_x, world.food_y, x, y, hx, hy, vision)
        hazard = _nearest(world.hazard_x, world.hazard_y, x, y, hx, hy, vision)

        # Distance to the boundary straight ahead, as a fraction of vision.
        ahead = vision
        if hx > 0.0:
            t = (cfg.width - x) / hx
            if t < ahead:
                ahead = t
        elif hx < 0.0:
            t = x / -hx
            if t < ahead:
                ahead = t
        if hy > 0.0:
            t = (cfg.height - y) / hy
            if t < ahead:
                ahead = t
        elif hy < 0.0:
            t = y / -hy
            if t < ahead:
                ahead = t

        self.inputs = [
            food[0],
            food[1],
            food[2],
            hazard[0],
            hazard[1],
            hazard[2],
            1.0 - ahead / vision,
            self.energy / cfg.max_energy,
        ]
        return self.inputs

    def think(self) -> None:
        out = self.brain.forward(self.inputs)
        self.steer = out[0]
        self.throttle = (out[1] + 1.0) * 0.5

    def act(self, world: World, rng: Rng) -> None:
        cfg = world.cfg

        t = self.steer * cfg.turn_rate
        t2 = t * t
        c = (1.0 - t2) / (1.0 + t2)
        s = (2.0 * t) / (1.0 + t2)
        hx = self.hx * c - self.hy * s
        hy = self.hx * s + self.hy * c
        # The rotation is exact in real arithmetic; renormalising stops
        # rounding error from drifting the length over thousands of steps.
        n = math.sqrt(hx * hx + hy * hy)
        self.hx = hx / n
        self.hy = hy / n

        speed = self.throttle * cfg.max_speed
        self.speed = speed
        r = cfg.agent_radius
        x = self.x + self.hx * speed
        y = self.y + self.hy * speed
        if x < r:
            x = r
        elif x > cfg.width - r:
            x = cfg.width - r
        if y < r:
            y = r
        elif y > cfg.height - r:
            y = cfg.height - r
        self.x = x
        self.y = y

        self.energy -= cfg.idle_cost + cfg.move_cost * speed * speed

        # Eat at most one item per step: the lowest-indexed one in reach.
        reach = r + cfg.food_radius
        reach2 = reach * reach
        food_x, food_y = world.food_x, world.food_y
        for i in range(len(food_x)):
            dx = food_x[i] - x
            dy = food_y[i] - y
            if dx * dx + dy * dy < reach2:
                self.food_eaten += 1
                self.energy = min(cfg.max_energy, self.energy + cfg.food_energy)
                world.respawn_food(i, rng)
                break

        touch = r + cfg.hazard_radius
        touch2 = touch * touch
        for zx, zy in zip(world.hazard_x, world.hazard_y, strict=True):
            dx = zx - x
            dy = zy - y
            if dx * dx + dy * dy < touch2:
                self.energy -= cfg.hazard_damage
                self.hazard_steps += 1
                break

        self.age += 1
        if self.energy <= 0.0:
            self.energy = 0.0
            self.alive = False

    def update(self, world: World, rng: Rng) -> None:
        self.sense(world)
        self.think()
        self.act(world, rng)


def _nearest(
    xs: list[float], ys: list[float], x: float, y: float, hx: float, hy: float, vision: float
) -> tuple[float, float, float]:
    """(nearness, ahead, right) of the closest item within ``vision``, or zeros.

    ``nearness`` is 1 at zero distance and 0 at the edge of vision; ``ahead`` and
    ``right`` are the cosine and sine of the bearing relative to the heading.
    In screen coordinates (y down) positive ``right`` is clockwise, the same
    way a positive steer turns.
    """
    best = vision * vision
    bx = by = 0.0
    found = False
    for ix, iy in zip(xs, ys, strict=True):
        dx = ix - x
        dy = iy - y
        d2 = dx * dx + dy * dy
        if d2 < best:
            best = d2
            bx = dx
            by = dy
            found = True
    if not found:
        return (0.0, 0.0, 0.0)
    d = math.sqrt(best)
    if d == 0.0:
        return (1.0, 0.0, 0.0)
    return (1.0 - d / vision, (bx * hx + by * hy) / d, (by * hx - bx * hy) / d)
