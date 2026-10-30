import { describe, expect, it } from 'vitest';
import { ACTION_SIZE } from '../src/engine/moveMap';
import { MultiSessionEvaluator } from '../src/engine/ortEvaluator';
import { PLANE_FLOATS } from '../src/engine/position';
import { withFixedBatch } from '../src/worker/fixBatch';
import { cpuThreads } from '../src/worker/threads';
import { decodeLogits, loadFixtures } from './helpers/fixtures';
import { MODEL_PATH, createNodeSession, hasModel } from './helpers/ortNode';
import { readFileSync } from 'node:fs';

describe('cpuThreads', () => {
  it('is 1 without cross-origin isolation, whatever the core count', () => {
    expect(cpuThreads(32, false)).toBe(1);
  });
  it('leaves one core free and stops at 8 (more threads were not faster in the measurements)', () => {
    expect(cpuThreads(2, true)).toBe(1);
    expect(cpuThreads(4, true)).toBe(3);
    expect(cpuThreads(8, true)).toBe(7);
    expect(cpuThreads(9, true)).toBe(8);
    expect(cpuThreads(32, true)).toBe(8);
  });
  it('copes with a missing core count', () => {
    expect(cpuThreads(0, true)).toBe(1);
    expect(cpuThreads(Number.NaN, true)).toBe(1);
  });
});

/** Fake sessions: the policy row of a position is filled with the position's first plane value, so rows can be traced. */
function fakeSessions(sizes: number[], calls: number[]) {
  class Tensor {
    constructor(
      public type: string,
      public data: Float32Array,
      public dims: number[],
    ) {}
  }
  const sessions = new Map<number, { run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> }>();
  for (const size of sizes) {
    sessions.set(size, {
      async run(feeds) {
        const input = feeds.planes!;
        expect(input.dims[0]).toBe(size);
        calls.push(size);
        const policy = new Float32Array(size * ACTION_SIZE);
        const value = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          policy.fill(input.data[i * PLANE_FLOATS]!, i * ACTION_SIZE, (i + 1) * ACTION_SIZE);
          value[i] = input.data[i * PLANE_FLOATS]!;
        }
        return { policy: new Tensor('float32', policy, [size, ACTION_SIZE]), value: new Tensor('float32', value, [size, 1]) };
      },
    });
  }
  return { ort: { Tensor }, sessions };
}

function batch(n: number): Float32Array {
  const planes = new Float32Array(n * PLANE_FLOATS);
  for (let i = 0; i < n; i++) planes[i * PLANE_FLOATS] = i + 1;
  return planes;
}

describe('MultiSessionEvaluator', () => {
  it('runs a batch on the smallest session that holds it, padded', async () => {
    const calls: number[] = [];
    const { ort, sessions } = fakeSessions([1, 8], calls);
    const ev = new MultiSessionEvaluator(ort as never, sessions as never);
    const out = await ev.evaluate(batch(3), 3);
    expect(calls).toEqual([8]);
    expect(out.value.length).toBe(3);
    expect([...out.value]).toEqual([1, 2, 3]);
    expect(out.policy.length).toBe(3 * ACTION_SIZE);
    expect(out.policy[2 * ACTION_SIZE]).toBe(3);
  });

  it('splits a batch larger than every session (WebGL has only a batch-1 session)', async () => {
    const calls: number[] = [];
    const { ort, sessions } = fakeSessions([1], calls);
    const ev = new MultiSessionEvaluator(ort as never, sessions as never);
    const out = await ev.evaluate(batch(5), 5);
    expect(calls).toEqual([1, 1, 1, 1, 1]);
    expect([...out.value]).toEqual([1, 2, 3, 4, 5]);
    expect(out.policy[4 * ACTION_SIZE + 17]).toBe(5);
  });

  it('refuses to start without a session', () => {
    expect(() => new MultiSessionEvaluator({ Tensor: class {} } as never, new Map())).toThrow();
  });
});

describe.skipIf(!hasModel)('withFixedBatch on the real network', () => {
  const fx = loadFixtures();
  const bytes = () => new Uint8Array(readFileSync(MODEL_PATH));

  it('gives the same numbers as the original network at batch 1 and at batch 4', async () => {
    const { ort } = await createNodeSession();
    for (const n of [1, 4]) {
      const fixed = withFixedBatch(bytes(), n);
      expect(fixed.length).toBeGreaterThan(bytes().length - 64); // weights untouched, only the shape declarations changed
      const session = await ort.InferenceSession.create(fixed, { executionProviders: ['cpu'] });
      const planes = new Float32Array(n * PLANE_FLOATS);
      fx.positions.slice(0, n).forEach((p, i) => planes.set(p.planes, i * PLANE_FLOATS));
      const out = await session.run({ planes: new ort.Tensor('float32', planes, [n, 14, 8, 8]) });
      expect(out.policy!.dims).toEqual([n, ACTION_SIZE]);
      const policy = out.policy!.data as Float32Array;
      fx.positions.slice(0, n).forEach((p, i) => {
        const ref = decodeLogits(p.logits_f32_b64);
        for (const k of p.legal_idx) expect(Math.abs(policy[i * ACTION_SIZE + k]! - ref[k]!)).toBeLessThan(1e-3);
      });
    }
  });

  it('declares the batch axis as fixed', async () => {
    const { ort } = await createNodeSession();
    const session = await ort.InferenceSession.create(withFixedBatch(bytes(), 8), { executionProviders: ['cpu'] });
    // a fixed-batch model rejects a different batch
    await expect(session.run({ planes: new ort.Tensor('float32', new Float32Array(PLANE_FLOATS), [1, 14, 8, 8]) })).rejects.toThrow();
  });

  it('rejects things that are not a model or not a valid batch', () => {
    expect(() => withFixedBatch(new Uint8Array([1, 2, 3]), 1)).toThrow();
    expect(() => withFixedBatch(bytes(), 0)).toThrow();
  });
});

describe('backendChoice', () => {
  it('only times the CPU on automatic, for a GPU that is not clearly fast', async () => {
    const { needsCpuCheck, CLEARLY_FAST_POSITIONS_PER_SECOND } = await import('../src/worker/backendChoice');
    expect(needsCpuCheck('auto', 'webgl', 10)).toBe(true);
    expect(needsCpuCheck('auto', 'webgpu', CLEARLY_FAST_POSITIONS_PER_SECOND - 1)).toBe(true);
    expect(needsCpuCheck('auto', 'webgpu', 900)).toBe(false);
    expect(needsCpuCheck('webgl', 'webgl', 10)).toBe(false); // ?backend=webgl is what was asked for
    expect(needsCpuCheck('auto', 'wasm', 5)).toBe(false);
  });
  it('picks the faster; a tie goes to the CPU (no GPU memory held for nothing)', async () => {
    const { cpuIsFaster, slowGpuNote } = await import('../src/worker/backendChoice');
    expect(cpuIsFaster(10, 40)).toBe(true);
    expect(cpuIsFaster(120, 40)).toBe(false);
    expect(cpuIsFaster(40, 40)).toBe(true);
    expect(slowGpuNote('webgl', 'AMD Radeon', 10, 40)).toBe('WebGL on AMD Radeon ran at 10 positions/s and the CPU at 40, so the CPU is used.');
  });
});
