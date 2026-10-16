from neuronet.config import Config
from neuronet.evolution import better, crossover, mutate, next_generation, rank, select
from neuronet.rng import Rng


def _genomes(n: int, size: int = 6) -> list[list[float]]:
    return [[float(i)] * size for i in range(n)]


def test_better_prefers_food_then_survival() -> None:
    assert better((5, 10), (4, 600))
    assert better((5, 300), (5, 200))
    assert not better((5, 200), (5, 200))


def test_rank_orders_best_first_and_keeps_ties_stable() -> None:
    scores = [(1, 100), (5, 50), (5, 80), (0, 600), (5, 80)]
    assert rank(scores) == [2, 4, 1, 0, 3]


def test_elites_survive_unchanged() -> None:
    cfg = Config(population=10, elite=2, mutation_rate=1.0)
    scores = [(i, 0) for i in range(10)]
    children = next_generation(_genomes(10), scores, cfg, Rng(1))
    assert children[0] == [9.0] * 6
    assert children[1] == [8.0] * 6


def test_population_size_is_preserved() -> None:
    cfg = Config(population=13)
    children = next_generation(_genomes(13), [(0, 0)] * 13, cfg, Rng(2))
    assert len(children) == 13
    assert all(len(c) == 6 for c in children)


def test_tournament_selection_favours_fitter_genomes() -> None:
    cfg = Config(tournament=3)
    scores = [(i, 0) for i in range(20)]
    rng = Rng(3)
    picks = [select(scores, cfg, rng) for _ in range(5000)]
    assert sum(picks) / len(picks) > 13  # uniform would average 9.5


def test_random_selection_ignores_fitness() -> None:
    cfg = Config(selection="random")
    scores = [(i, 0) for i in range(20)]
    rng = Rng(3)
    picks = [select(scores, cfg, rng) for _ in range(20000)]
    assert 9.2 < sum(picks) / len(picks) < 9.8


def test_selection_off_keeps_no_elites() -> None:
    cfg = Config(population=10, selection="random", mutation_rate=0.0, crossover=0.0)
    scores = [(i, 0) for i in range(10)]
    a = next_generation(_genomes(10), scores, cfg, Rng(4))
    b = next_generation(_genomes(10), list(reversed(scores)), cfg, Rng(4))
    assert a == b  # fitness is never read


def test_crossover_takes_each_gene_from_a_parent() -> None:
    child = crossover([1.0] * 200, [2.0] * 200, Rng(5))
    assert set(child) == {1.0, 2.0}
    assert 70 < child.count(1.0) < 130


def test_mutation_rate_zero_changes_nothing() -> None:
    genome = [0.5] * 50
    mutate(genome, Config(mutation_rate=0.0), Rng(6))
    assert genome == [0.5] * 50


def test_mutation_rate_controls_how_many_genes_move() -> None:
    genome = [0.0] * 10_000
    mutate(genome, Config(mutation_rate=0.1, mutation_strength=0.3), Rng(7))
    moved = sum(1 for g in genome if g != 0.0)
    assert 900 < moved < 1100


def test_parents_are_not_modified() -> None:
    parents = _genomes(8)
    snapshot = [g[:] for g in parents]
    next_generation(parents, [(i, 0) for i in range(8)], Config(population=8), Rng(8))
    assert parents == snapshot
