"""Turning one generation's genomes and scores into the next generation's.

Elitism, tournament selection, uniform crossover and per-gene mutation — the
standard toolkit, written out. With ``selection="random"`` the same code runs
with fitness never consulted, which is the control the experiment compares
against: if that does as well, evolution is not what is doing the work.
"""

from __future__ import annotations

from collections.abc import Sequence

from neuronet.config import Config
from neuronet.rng import Rng

Score = tuple[int, int]
Genome = list[float]


def better(a: Score, b: Score) -> bool:
    """More food wins; on equal food, surviving longer wins."""
    return a[0] > b[0] or (a[0] == b[0] and a[1] > b[1])


def rank(scores: Sequence[Score]) -> list[int]:
    """Indices from best to worst. Exact ties keep their original order."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i][0], -scores[i][1], i))


def select(scores: Sequence[Score], cfg: Config, rng: Rng) -> int:
    n = len(scores)
    if cfg.selection == "random":
        return rng.below(n)
    winner = rng.below(n)
    for _ in range(cfg.tournament - 1):
        challenger = rng.below(n)
        if better(scores[challenger], scores[winner]):
            winner = challenger
    return winner


def crossover(a: Sequence[float], b: Sequence[float], rng: Rng) -> Genome:
    child = []
    for ga, gb in zip(a, b, strict=True):
        child.append(ga if rng.random() < 0.5 else gb)
    return child


def mutate(genome: Genome, cfg: Config, rng: Rng) -> None:
    rate = cfg.mutation_rate
    strength = cfg.mutation_strength
    for i in range(len(genome)):
        if rng.random() < rate:
            genome[i] += strength * rng.normal()


def next_generation(
    genomes: Sequence[Sequence[float]], scores: Sequence[Score], cfg: Config, rng: Rng
) -> list[Genome]:
    if len(genomes) != len(scores):
        raise ValueError("one score per genome")
    size = len(genomes)

    offspring: list[Genome] = []
    if cfg.selection == "tournament":
        offspring = [list(genomes[i]) for i in rank(scores)[: cfg.elite]]

    while len(offspring) < size:
        a = select(scores, cfg, rng)
        if rng.random() < cfg.crossover:
            b = select(scores, cfg, rng)
            child = crossover(genomes[a], genomes[b], rng)
        else:
            child = list(genomes[a])
        mutate(child, cfg, rng)
        offspring.append(child)
    return offspring
