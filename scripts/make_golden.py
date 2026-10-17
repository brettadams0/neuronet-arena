"""Regenerate fixtures/golden.json — the trace both engines are tested against.

    python scripts/make_golden.py

Only run this when a change to the simulation is *meant* to change its
output; then commit the new fixture in the same change. The Python test
(tests/test_golden.py) and the JavaScript test (web/test/golden.test.js) both
compare against this file with exact equality, so if they disagree with each
other, one of the two engines has drifted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuronet import Config, Rng, Simulation
from neuronet.brain import Brain, layer_sizes

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "golden.json"

# Small enough to run in well under a second in either engine, big enough to
# exercise eating, respawning, hazard contact, crossover and elitism. (No agent
# dies in it; deaths are covered by the trained-population replay instead.)
RUN_CONFIG = Config(population=12, food_count=30, hazard_count=6, steps=240, hidden=6)
RUN_SEED = 20261016
GENERATIONS = 4
TRACE_EVERY = 20


def golden() -> dict[str, Any]:
    rng = Rng(42)
    rng_section = {
        "seed": 42,
        "u32": [rng.next_u32() for _ in range(16)],
        "random": [rng.random() for _ in range(8)],
        "normal": [rng.normal() for _ in range(8)],
        "below7": [rng.below(7) for _ in range(16)],
        "state": rng.state(),
    }

    layers = layer_sizes(10)
    brain = Brain.random(layers, Rng(7))
    probe = Rng(8)
    inputs = [[probe.uniform(-1.5, 1.5) for _ in range(layers[0])] for _ in range(6)]
    brain_section = {
        "seed": 7,
        "layers": list(layers),
        "genome": brain.genome,
        "inputs": inputs,
        "outputs": [brain.forward(x) for x in inputs],
    }

    sim = Simulation(RUN_CONFIG, seed=RUN_SEED)
    sim.begin_generation()
    assert sim.world is not None
    trace = []
    while not sim.done:
        if sim.step_index % TRACE_EVERY == 0:
            trace.append(
                {
                    "step": sim.step_index,
                    "agents": [[a.x, a.y, a.hx, a.hy, a.energy, a.food_eaten] for a in sim.agents],
                    "food": [sim.world.food_x[:], sim.world.food_y[:]],
                }
            )
        sim.step()
    stats = [sim.end_generation()]
    stats += sim.run(GENERATIONS - 1)

    run_section = {
        "seed": RUN_SEED,
        "config": RUN_CONFIG.to_dict(),
        "generations": GENERATIONS,
        "trace_every": TRACE_EVERY,
        "trace": trace,
        "stats": [s.to_dict() for s in stats],
        "rng": sim.rng.state(),
        "genomes": sim.genomes,
    }
    return {"rng": rng_section, "brain": brain_section, "run": run_section}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(golden(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")
