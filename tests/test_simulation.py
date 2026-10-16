import json
from pathlib import Path

import pytest

from neuronet import snapshot
from neuronet.config import Config
from neuronet.simulation import Simulation

SMALL = Config(population=10, food_count=20, hazard_count=4, steps=120, hidden=6)


def test_same_seed_same_history() -> None:
    a = Simulation(SMALL, seed=3).run(3)
    b = Simulation(SMALL, seed=3).run(3)
    assert a == b


def test_different_seed_different_history() -> None:
    assert Simulation(SMALL, seed=3).run(2) != Simulation(SMALL, seed=4).run(2)


def test_evolved_genomes_carry_into_the_next_generation() -> None:
    # The regression that started this rewrite: the old loop rebuilt the
    # population from scratch every generation, so nothing was ever inherited.
    sim = Simulation(SMALL.replace(elite=2), seed=5)
    sim.run_generation()
    bred = [g[:] for g in sim.genomes]
    sim.begin_generation()
    assert [a.brain.genome for a in sim.agents] == bred


def test_elite_is_the_previous_champion() -> None:
    sim = Simulation(SMALL, seed=6)
    before = [g[:] for g in sim.genomes]
    stats = sim.run_generation()
    assert sim.genomes[0] == before[stats.champion]


def test_stats_are_consistent_with_the_agents() -> None:
    sim = Simulation(SMALL, seed=7)
    sim.begin_generation()
    while not sim.done:
        sim.step()
    eaten = [a.food_eaten for a in sim.agents]
    alive = sum(a.alive for a in sim.agents)
    stats = sim.end_generation()
    assert stats.best == max(eaten)
    assert stats.mean == sum(eaten) / len(eaten)
    assert stats.survivors == alive
    assert eaten[stats.champion] == stats.best


def test_generation_ends_early_when_everyone_is_dead() -> None:
    sim = Simulation(SMALL.replace(start_energy=1.0, idle_cost=0.5), seed=8)
    sim.begin_generation()
    while not sim.done:
        sim.step()
    assert sim.step_index < SMALL.steps
    assert not any(a.alive for a in sim.agents)


def test_stepping_without_a_generation_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        Simulation(SMALL).step()
    with pytest.raises(RuntimeError):
        Simulation(SMALL).end_generation()


def test_supplied_genomes_must_match_the_population() -> None:
    with pytest.raises(ValueError, match="expected 10 genomes"):
        Simulation(SMALL, genomes=[[0.0]])


def test_snapshot_resume_matches_an_uninterrupted_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    straight = Simulation(SMALL, seed=9)
    expected = straight.run(4)

    first = Simulation(SMALL, seed=9)
    first.run(2)
    path = tmp_path / "snap.json"
    snapshot.save(first, path)
    resumed = snapshot.load(path)
    assert resumed.generation == 2
    resumed.run(2)

    assert resumed.history == expected
    assert resumed.genomes == straight.genomes
    assert resumed.rng.state() == straight.rng.state()


def test_snapshot_records_what_comes_next() -> None:
    sim = Simulation(SMALL, seed=10)
    sim.run(1)
    probe = snapshot.from_dict(json.loads(json.dumps(snapshot.to_dict(sim))))
    expect = probe.run_generation()
    data = snapshot.to_dict(sim, expect=expect)
    assert snapshot.from_dict(data).run_generation() == expect


def test_snapshot_mid_generation_is_refused() -> None:
    sim = Simulation(SMALL)
    sim.begin_generation()
    with pytest.raises(RuntimeError):
        snapshot.to_dict(sim)


def test_snapshot_format_is_checked() -> None:
    with pytest.raises(ValueError, match="not a neuronet-arena snapshot"):
        snapshot.from_dict({"format": "something-else"})
    data = snapshot.to_dict(Simulation(SMALL))
    data["version"] = 99
    with pytest.raises(ValueError, match="unsupported snapshot version"):
        snapshot.from_dict(data)


def test_the_demo_population_resumes_to_its_recorded_next_generation() -> None:
    # web/data/trained.json is what the browser demo starts from, and the page
    # checks this same expectation live. If a change to the simulation makes
    # this fail, retrain it (see README, "Development").
    path = Path(__file__).resolve().parent.parent / "web" / "data" / "trained.json"
    data = json.loads(path.read_text())
    sim = snapshot.from_dict(data)
    assert sim.run_generation().to_dict() == data["expect"]
