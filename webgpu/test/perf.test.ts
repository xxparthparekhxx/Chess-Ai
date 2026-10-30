// Rough CPU reference numbers, measured in Node with onnxruntime-node.
// These are not browser numbers: WebGPU and WASM speed in a browser were not measured here.
import os from 'node:os';
import { describe, expect, it } from 'vitest';
import { Mcts } from '../src/engine/mcts';
import { ACTION_SIZE, type EvalOutput, type Evaluator } from '../src/engine/types';
import { PLANE_FLOATS } from '../src/engine/position';
import { OrtEvaluator } from '../src/engine/ortEvaluator';
import { loadFixtures } from './helpers/fixtures';
import { createNodeSession, hasModel, writeOut } from './helpers/ortNode';

const fx = loadFixtures();

function stats(samples: number[]) {
  const s = [...samples].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))]!;
  return { median: q(0.5), min: s[0]!, p90: q(0.9), runs: s.length };
}

function batchOf(n: number): Float32Array {
  const planes = new Float32Array(n * PLANE_FLOATS);
  for (let i = 0; i < n; i++) planes.set(fx.positions[i % fx.positions.length]!.planes, i * PLANE_FLOATS);
  return planes;
}

describe.skipIf(!hasModel)('timing (Node, CPU)', () => {
  it('forward pass at batch 1, 8 and 32', async () => {
    const result: Record<string, unknown> = {
      machine: { cpu: os.cpus()[0]?.model, logicalCores: os.cpus().length, node: process.version, onnxruntimeNode: '1.31.1' },
      forwardMs: {},
    };
    for (const threads of [0, 1]) {
      const { ort, session } = await createNodeSession(undefined, threads || undefined);
      const ev = new OrtEvaluator(ort, session);
      const label = threads ? `${threads} thread` : 'default threads';
      const table: Record<string, unknown> = {};
      for (const batch of [1, 8, 32]) {
        const planes = batchOf(batch);
        for (let i = 0; i < 3; i++) await ev.evaluate(planes, batch);
        const runs = batch === 32 && threads === 1 ? 5 : 20;
        const samples: number[] = [];
        for (let i = 0; i < runs; i++) {
          const t = performance.now();
          await ev.evaluate(planes, batch);
          samples.push(performance.now() - t);
        }
        const st = stats(samples);
        table[`batch${batch}`] = { ...st, positionsPerSecond: Math.round((batch * 1000) / st.median) };
      }
      (result.forwardMs as Record<string, unknown>)[label] = table;
    }
    console.log('[timing] forward pass, ms:', JSON.stringify(result, null, 1));
    writeOut('timing_forward.json', result);
    expect(result).toBeTruthy();
  }, 300_000);

  it('search overhead with a free network (everything except the network call)', async () => {
    // Fixed random logits, returned instantly: measures chess.js move generation,
    // encoding, selection and back-propagation only.
    const policy = new Float32Array(64 * ACTION_SIZE);
    let seed = 12345;
    for (let i = 0; i < policy.length; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      policy[i] = (seed / 2 ** 32) * 6 - 3;
    }
    const value = new Float32Array(64);
    const free: Evaluator = {
      async evaluate(_planes: Float32Array, n: number): Promise<EvalOutput> {
        return { policy: policy.subarray(0, n * ACTION_SIZE), value: value.subarray(0, n) };
      },
    };
    const mcts = new Mcts(free);
    const out: Record<string, unknown> = {};
    for (const batch of [8, 32]) {
      let sims = 0;
      let ms = 0;
      const t0 = performance.now();
      for (const p of fx.positions.filter((p) => p.legal_uci.length > 1).slice(0, 20)) {
        const r = await mcts.search(p.fen, { maxSimulations: 800, batchSize: batch, dirichlet: false });
        sims += r.simulations;
      }
      ms = performance.now() - t0;
      out[`batch${batch}`] = { simulations: sims, ms: Math.round(ms), simsPerSecond: Math.round((sims * 1000) / ms) };
    }
    console.log('[timing] search overhead only, free network:', JSON.stringify(out));
    writeOut('timing_search_overhead.json', out);
    expect(out).toBeTruthy();
  }, 300_000);

  it('end to end: simulations per second with the real network on CPU', async () => {
    const { ort, session } = await createNodeSession();
    const mcts = new Mcts(new OrtEvaluator(ort, session));
    const out: Record<string, unknown> = {};
    for (const batch of [8, 32]) {
      let sims = 0;
      let evalMs = 0;
      const t0 = performance.now();
      for (const p of fx.positions.filter((p) => p.legal_uci.length > 1).slice(0, 6)) {
        const r = await mcts.search(p.fen, { timeBudgetMs: 1000, batchSize: batch, dirichlet: false });
        sims += r.simulations;
        evalMs += r.evalMs;
      }
      const ms = performance.now() - t0;
      out[`batch${batch}`] = {
        simulations: sims,
        ms: Math.round(ms),
        simsPerSecond: Math.round((sims * 1000) / ms),
        shareOfTimeInNetwork: Number((evalMs / ms).toFixed(3)),
      };
    }
    console.log('[timing] end to end on CPU, 1 s budget per move:', JSON.stringify(out));
    writeOut('timing_search_e2e.json', out);
    expect(out).toBeTruthy();
  }, 300_000);
});
