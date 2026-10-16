"""Does evolution do anything here? Run it with selection on and off and compare.

Each seed is run twice with an identical config apart from ``selection``:
``"tournament"`` (the real thing) and ``"random"``, where parents are drawn
uniformly and nothing is kept as elite — fitness is still measured but never
read. Both conditions start from the same seed, so generation 0 is the same
population in the same world; any gap that opens after that is selection.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuronet.config import Config
from neuronet.simulation import GenerationStats, Simulation

CONDITIONS = ("tournament", "random")


@dataclass(frozen=True)
class Job:
    cfg: Config
    seed: int
    generations: int


def run_job(job: Job) -> list[GenerationStats]:
    return Simulation(job.cfg, seed=job.seed).run(job.generations)


def run_experiment(
    cfg: Config,
    seeds: Sequence[int],
    generations: int,
    *,
    workers: int | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    jobs = [
        (condition, seed, Job(cfg.replace(selection=condition), seed, generations))
        for seed in seeds
        for condition in CONDITIONS
    ]
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {c: {} for c in CONDITIONS}
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(run_job, [job for _, _, job in jobs])
        for (condition, seed, _), history in zip(jobs, results, strict=True):
            runs[condition][str(seed)] = [s.to_dict() for s in history]
            if progress:
                progress(condition, seed)
    return {
        "config": cfg.replace(selection="tournament").to_dict(),
        "seeds": list(seeds),
        "generations": generations,
        "runs": runs,
    }


def series(result: dict[str, Any], condition: str, field: str = "mean") -> list[list[float]]:
    """One list per seed of ``field`` by generation."""
    return [[g[field] for g in result["runs"][condition][str(s)]] for s in result["seeds"]]


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def summarise(result: dict[str, Any], window: int = 10) -> dict[str, dict[str, float]]:
    """Per condition: generation-0 and final-window means of the per-seed values."""
    out: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        per_seed = series(result, condition)
        finals = [sum(s[-window:]) / len(s[-window:]) for s in per_seed]
        hazard = [
            [g["hazard_steps"] / max(1, g["mean"] * result["config"]["population"]) for g in run]
            for run in (result["runs"][condition][str(s)] for s in result["seeds"])
        ]
        out[condition] = {
            "start": sum(s[0] for s in per_seed) / len(per_seed),
            "final": sum(finals) / len(finals),
            "final_min": min(finals),
            "final_max": max(finals),
            "survivors": sum(
                sum(g["survivors"] for g in result["runs"][condition][str(s)][-window:]) / window
                for s in result["seeds"]
            )
            / len(result["seeds"]),
            "hazard_per_food": sum(sum(h[-window:]) / window for h in hazard) / len(hazard),
        }
    return out


def save(result: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, separators=(",", ":")) + "\n")


def load(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return data
