// The JavaScript engine against fixtures/golden.json, which the Python engine
// wrote. Every comparison is exact: the point of the fixture is that the two
// engines produce the same doubles, not merely close ones.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Brain } from "../src/brain.js";
import { makeConfig } from "../src/config.js";
import { Rng } from "../src/rng.js";
import { Simulation } from "../src/simulation.js";

const golden = JSON.parse(
  readFileSync(new URL("../../fixtures/golden.json", import.meta.url), "utf8"),
);

test("rng stream matches Python", () => {
  const g = golden.rng;
  const rng = new Rng(g.seed);
  assert.deepEqual(Array.from({ length: 16 }, () => rng.nextU32()), g.u32);
  assert.deepEqual(Array.from({ length: 8 }, () => rng.random()), g.random);
  assert.deepEqual(Array.from({ length: 8 }, () => rng.normal()), g.normal);
  assert.deepEqual(Array.from({ length: 16 }, () => rng.below(7)), g.below7);
  assert.deepEqual(rng.state(), g.state);
});

test("brain initialisation and forward pass match Python", () => {
  const g = golden.brain;
  const brain = Brain.random(g.layers, new Rng(g.seed));
  assert.deepEqual(brain.genome, g.genome);
  g.inputs.forEach((/** @type {number[]} */ x, /** @type {number} */ i) => {
    assert.deepEqual(brain.forward(x), g.outputs[i]);
  });
});

test("a full run matches Python step for step", () => {
  const g = golden.run;
  const sim = new Simulation(makeConfig(g.config), g.seed);
  sim.beginGeneration();
  const world = /** @type {import("../src/world.js").World} */ (sim.world);
  let checkpoint = 0;
  while (!sim.done) {
    if (sim.stepIndex % g.trace_every === 0) {
      const expected = g.trace[checkpoint++];
      assert.equal(sim.stepIndex, expected.step);
      const agents = sim.agents.map((a) => [a.x, a.y, a.hx, a.hy, a.energy, a.foodEaten]);
      assert.deepEqual(agents, expected.agents, `agents diverge at step ${sim.stepIndex}`);
      assert.deepEqual([world.foodX, world.foodY], expected.food, `food diverges at step ${sim.stepIndex}`);
    }
    sim.step();
  }
  assert.equal(checkpoint, g.trace.length);

  const stats = [sim.endGeneration()];
  for (let i = 1; i < g.generations; i++) stats.push(sim.runGeneration());
  assert.deepEqual(stats, g.stats);
  assert.deepEqual(sim.rng.state(), g.rng);
  assert.deepEqual(sim.genomes, g.genomes);
});
