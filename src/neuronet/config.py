"""Every tunable number in the simulation, in one frozen dataclass.

The same field names are used by ``web/src/config.js`` and by the JSON in a
snapshot, so a config written by one engine loads unchanged in the other.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Any, Literal

Selection = Literal["tournament", "random"]


@dataclass(frozen=True)
class Config:
    # World
    width: float = 960.0
    height: float = 600.0
    population: int = 40
    food_count: int = 60
    hazard_count: int = 12
    steps: int = 600
    """Simulation steps per generation."""

    # Bodies
    agent_radius: float = 6.0
    food_radius: float = 4.0
    hazard_radius: float = 16.0
    vision: float = 140.0
    max_speed: float = 3.0
    turn_rate: float = 0.1
    """Cayley rotation parameter; the sharpest turn per step is 2·atan(turn_rate), about 11.4°."""

    # Energy
    start_energy: float = 60.0
    max_energy: float = 100.0
    food_energy: float = 25.0
    idle_cost: float = 0.08
    move_cost: float = 0.02
    """Charged per step as move_cost × speed²."""
    hazard_damage: float = 4.0
    """Charged per step spent touching a hazard."""

    # Brain
    hidden: int = 10

    # Evolution
    selection: Selection = "tournament"
    """``"random"`` turns selection off: parents are drawn uniformly, nothing is kept as elite."""
    elite: int = 2
    tournament: int = 3
    crossover: float = 0.7
    mutation_rate: float = 0.08
    mutation_strength: float = 0.3

    def __post_init__(self) -> None:
        positive = (
            "width",
            "height",
            "population",
            "steps",
            "agent_radius",
            "food_radius",
            "hazard_radius",
            "vision",
            "max_speed",
            "max_energy",
            "start_energy",
            "hidden",
            "tournament",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        non_negative = (
            "food_count",
            "hazard_count",
            "turn_rate",
            "food_energy",
            "idle_cost",
            "move_cost",
            "hazard_damage",
            "elite",
            "mutation_strength",
        )
        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)!r}")
        for name in ("crossover", "mutation_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)!r}")
        if self.selection not in ("tournament", "random"):
            raise ValueError(f"selection must be 'tournament' or 'random', got {self.selection!r}")
        if self.elite > self.population:
            raise ValueError("elite cannot exceed population")
        if self.start_energy > self.max_energy:
            raise ValueError("start_energy cannot exceed max_energy")
        margin = 2 * max(self.agent_radius, self.food_radius, self.hazard_radius)
        if self.width <= 2 * margin or self.height <= 2 * margin:
            raise ValueError("arena is too small for the body radii")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        types = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(data) - set(types))
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        # JSON written by JavaScript has no 960.0, only 960; keep floats floats.
        values: dict[str, Any] = {
            k: float(v) if types[k] == "float" else v for k, v in data.items()
        }
        return cls(**values)

    def replace(self, **changes: Any) -> Config:
        return dataclasses.replace(self, **changes)
