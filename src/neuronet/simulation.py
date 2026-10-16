"""The generation loop, with no rendering in it.

One ``Rng`` drives everything in a run — world layout, spawn points, food
respawns, selection and mutation — and it is consumed in a fixed order. That
is what makes a run reproducible from its seed, and what lets a snapshot
(genomes plus the four words of RNG state) resume a run exactly, in either
engine.

The viewers drive this step by step; training and the experiment call
``run_generation``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from neuronet.agent import Agent
from neuronet.brain import Brain, layer_sizes
from neuronet.config import Config
from neuronet.evolution import next_generation, rank
from neuronet.rng import Rng
from neuronet.world import World


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best: int
    """Most food eaten by any one agent."""
    mean: float
    """Food eaten per agent."""
    survivors: int
    """Agents still alive at the final step."""
    hazard_steps: int
    """Agent-steps spent touching a hazard, summed over the population."""
    champion: int
    """Index of the best agent this generation."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationStats:
        return cls(**data)


class Simulation:
    def __init__(
        self,
        cfg: Config,
        seed: int = 0,
        *,
        genomes: Sequence[Sequence[float]] | None = None,
        generation: int = 0,
        rng: Rng | None = None,
    ) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = rng if rng is not None else Rng(seed)
        self.layers = layer_sizes(cfg.hidden)
        self.generation = generation
        if genomes is None:
            self.genomes = [
                Brain.random(self.layers, self.rng).genome for _ in range(cfg.population)
            ]
        else:
            if len(genomes) != cfg.population:
                raise ValueError(f"expected {cfg.population} genomes, got {len(genomes)}")
            self.genomes = [list(g) for g in genomes]
        self.history: list[GenerationStats] = []

        self.world: World | None = None
        self.agents: list[Agent] = []
        self.step_index = 0

    # -- stepping ---------------------------------------------------------

    def begin_generation(self) -> None:
        self.world = World(self.cfg, self.rng)
        world = self.world
        self.agents = [Agent.spawn(Brain(self.layers, g), world, self.rng) for g in self.genomes]
        self.step_index = 0

    @property
    def in_generation(self) -> bool:
        return self.world is not None

    @property
    def done(self) -> bool:
        """The current generation has used its steps, or nobody is left alive."""
        return self.step_index >= self.cfg.steps or not any(a.alive for a in self.agents)

    def step(self) -> None:
        world = self.world
        if world is None:
            raise RuntimeError("call begin_generation() first")
        rng = self.rng
        for agent in self.agents:
            if agent.alive:
                agent.update(world, rng)
        self.step_index += 1

    def end_generation(self) -> GenerationStats:
        if self.world is None:
            raise RuntimeError("no generation in progress")
        scores = [a.fitness for a in self.agents]
        eaten = [a.food_eaten for a in self.agents]
        stats = GenerationStats(
            generation=self.generation,
            best=max(eaten),
            mean=sum(eaten) / len(eaten),  # ints, so sum() is exact here
            survivors=sum(1 for a in self.agents if a.alive),
            hazard_steps=sum(a.hazard_steps for a in self.agents),
            champion=rank(scores)[0],
        )
        self.genomes = next_generation(self.genomes, scores, self.cfg, self.rng)
        self.history.append(stats)
        self.generation += 1
        self.world = None
        self.agents = []
        return stats

    def run_generation(self) -> GenerationStats:
        self.begin_generation()
        while not self.done:
            self.step()
        return self.end_generation()

    def run(self, generations: int) -> list[GenerationStats]:
        return [self.run_generation() for _ in range(generations)]
