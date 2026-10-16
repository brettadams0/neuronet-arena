// Unit checks on the JavaScript side that the golden trace does not reach.

import assert from "node:assert/strict";
import { test } from "node:test";

import { makeConfig, DEFAULTS } from "../src/config.js";
import { rank, nextGeneration } from "../src/evolution.js";
import { Rng } from "../src/rng.js";

test("rng values stay in range", () => {
  const rng = new Rng(9);
  for (let i = 0; i < 10000; i++) {
    const u = rng.nextU32();
    assert.ok(Number.isInteger(u) && u >= 0 && u <= 0xffffffff);
    const r = rng.random();
    assert.ok(r >= 0 && r < 1);
    const b = rng.below(7);
    assert.ok(Number.isInteger(b) && b >= 0 && b < 7);
  }
});

test("rng state rejects the all-zero fixed point", () => {
  assert.throws(() => Rng.fromState([0, 0, 0, 0]), /invalid xoshiro128 state/);
});

test("config defaults match the documented ones and validate overrides", () => {
  assert.equal(DEFAULTS.population, 40);
  assert.throws(() => makeConfig(/** @type {any} */ ({ bogus: 1 })), /unknown config keys: bogus/);
  assert.throws(() => makeConfig({ population: 0 }), /population must be positive/);
  assert.throws(() => makeConfig({ mutation_rate: 2 }), /mutation_rate must be in/);
  assert.equal(makeConfig({ population: 12 }).population, 12);
});

test("rank is best-first with stable ties", () => {
  assert.deepEqual(rank([[1, 100], [5, 50], [5, 80], [0, 600], [5, 80]]), [2, 4, 1, 0, 3]);
});

test("elites survive unchanged and the population size holds", () => {
  const cfg = makeConfig({ population: 10, elite: 2, mutation_rate: 1 });
  const genomes = Array.from({ length: 10 }, (_, i) => new Array(6).fill(i));
  /** @type {[number, number][]} */
  const scores = genomes.map((_, i) => [i, 0]);
  const next = nextGeneration(genomes, scores, cfg, new Rng(1));
  assert.equal(next.length, 10);
  assert.deepEqual(next[0], new Array(6).fill(9));
  assert.deepEqual(next[1], new Array(6).fill(8));
});
