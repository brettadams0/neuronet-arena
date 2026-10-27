# Changelog

## Unreleased

### Added

- **Selection on vs off, side by side, in the web demo.** The README's controlled experiment runs
  live: two arenas from the same seed and world, the right one with `selection: "random"`, a chart of
  food per agent by generation, and a running comparison. It starts on a click and pauses off screen.
- **`fixtures/golden.json` gains `run_random`**, a run with selection off, so the Python and
  JavaScript engines are tested against each other on that path too.

## 2.0.0 — 2026-10-16

A rewrite. The 1.x simulator ran, but it could not evolve anything, and nothing about it could be
measured or reproduced.

### Fixed

- **Evolved brains were discarded every generation.** `Simulation.run` called `setup()` at the top
  of each generation, which rebuilt every agent with a fresh random network, so the output of
  `evolve_population` was never used: 0 of 20 evolved brains reached the next generation.
- **Food was never consumed.** An agent scored +1 for every tick it spent within reach of a food
  node, and the node never ran out; circling a single one as tightly as possible could score up to about 80 points in 300 ticks. Food is
  now eaten, restores energy, and respawns elsewhere.
- **The second network output was computed and ignored.** It now drives the throttle.
- **The pygame window never pumped events**, so it could not be closed and was reported as not
  responding.

### Changed

- Headless, typed package (`src/neuronet`) with no runtime dependencies; rendering is a separate,
  optional viewer. Python 3.10+.
- Egocentric senses: nearest food and nearest hazard (nearness, ahead, right), wall ahead, energy.
  The old inputs averaged food, hazards and neutral nodes into one world-frame offset.
- Energy budget with movement cost, food, hazard damage and death.
- Elitism, tournament selection and uniform crossover; previously clone-and-mutate with every
  survivor mutated.
- Seeded and reproducible: one xoshiro128** stream per run, serialisable between generations.

### Added

- `neuronet` CLI: `train`, `view`, `experiment`, `plot`.
- A controlled experiment (selection on vs off across seeds) and its committed results.
- A JavaScript port of the engine and a live browser demo, bit-for-bit identical to the Python
  engine, checked against a shared golden trace in both test suites.
- Tests (pytest, node:test), ruff, mypy `--strict`, `tsc --checkJs`, and CI on Python 3.10–3.13 and
  Node 20/22.

## 1.0.0 — 2025-04-14

Initial upload: a pygame evolution simulator with a from-scratch network and genetic algorithm.
