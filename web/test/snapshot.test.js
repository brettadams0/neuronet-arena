// The demo's "trained in Python" population: resuming it here must score the
// next generation exactly as Python recorded — the same check the page runs
// live in the browser.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Simulation } from "../src/simulation.js";
import { makeConfig } from "../src/config.js";
import { fromSnapshot, toSnapshot } from "../src/snapshot.js";

const trained = JSON.parse(readFileSync(new URL("../data/trained.json", import.meta.url), "utf8"));

test("the trained snapshot resumes to exactly the generation Python recorded", () => {
  const sim = fromSnapshot(trained);
  assert.equal(sim.generation, trained.generation);
  assert.deepEqual(sim.runGeneration(), trained.expect);
});

test("toSnapshot / fromSnapshot round-trips mid-run", () => {
  const cfg = makeConfig({ population: 8, steps: 60, food_count: 12 });
  const straight = new Simulation(cfg, 11);
  const expected = [straight.runGeneration(), straight.runGeneration(), straight.runGeneration()];

  const first = new Simulation(cfg, 11);
  first.runGeneration();
  const resumed = fromSnapshot(JSON.parse(JSON.stringify(toSnapshot(first))));
  const rest = [resumed.runGeneration(), resumed.runGeneration()];
  assert.deepEqual([first.history[0], ...rest], expected);
  assert.deepEqual(resumed.rng.state(), straight.rng.state());
});

test("snapshots of the wrong format are refused", () => {
  assert.throws(() => fromSnapshot({ ...trained, format: "nope" }), /not a neuronet-arena snapshot/);
  assert.throws(() => fromSnapshot({ ...trained, version: 2 }), /unsupported snapshot version/);
});
