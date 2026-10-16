"""Saving and resuming a run between generations.

A snapshot is plain JSON: the config, the generation number, the four words of
RNG state and every genome. Python writes floats with ``repr``, which
round-trips exactly, and JavaScript's ``JSON.parse`` reads the same digits to
the same doubles — so a population trained here resumes in the browser with
nothing lost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuronet.config import Config
from neuronet.rng import Rng
from neuronet.simulation import GenerationStats, Simulation

FORMAT = "neuronet-arena/snapshot"
VERSION = 1


def to_dict(sim: Simulation, *, expect: GenerationStats | None = None) -> dict[str, Any]:
    if sim.in_generation:
        raise RuntimeError("snapshots are taken between generations")
    data: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "seed": sim.seed,
        "generation": sim.generation,
        "rng": sim.rng.state(),
        "config": sim.cfg.to_dict(),
        "genomes": sim.genomes,
        "history": [s.to_dict() for s in sim.history],
    }
    if expect is not None:
        # What this engine scored on the generation that follows, so another
        # engine resuming from here can check that it scores the same.
        data["expect"] = expect.to_dict()
    return data


def from_dict(data: dict[str, Any]) -> Simulation:
    if data.get("format") != FORMAT:
        raise ValueError("not a neuronet-arena snapshot")
    if data.get("version") != VERSION:
        raise ValueError(f"unsupported snapshot version {data.get('version')!r}")
    sim = Simulation(
        Config.from_dict(data["config"]),
        seed=data["seed"],
        genomes=data["genomes"],
        generation=data["generation"],
        rng=Rng.from_state(data["rng"]),
    )
    sim.history = [GenerationStats.from_dict(s) for s in data.get("history", [])]
    return sim


def save(sim: Simulation, path: str | Path, *, expect: GenerationStats | None = None) -> None:
    Path(path).write_text(json.dumps(to_dict(sim, expect=expect), separators=(",", ":")) + "\n")


def load(path: str | Path) -> Simulation:
    return from_dict(json.loads(Path(path).read_text()))
