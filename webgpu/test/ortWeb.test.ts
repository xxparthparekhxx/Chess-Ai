// The browser's CPU fallback uses onnxruntime-web's WebAssembly build. That same build runs in Node,
// so it can be checked here: the exported network must load in it and agree with PyTorch, and its speed
// gives a rough idea of how slow the fallback is. This is Node's WebAssembly engine, not a browser.
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ACTION_SIZE } from '../src/engine/moveMap';
import { PLANE_FLOATS } from '../src/engine/position';
import { OrtEvaluator, type OrtModuleLike, type OrtSessionLike } from '../src/engine/ortEvaluator';
import { decodeLogits, loadFixtures, WEB_ROOT } from './helpers/fixtures';
import { hasModel, MODEL_PATH, writeOut } from './helpers/ortNode';

const fx = loadFixtures();

async function webSession(threads: number) {
  const ort = await import('onnxruntime-web/wasm');
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.wasmPaths = resolve(WEB_ROOT, 'node_modules', 'onnxruntime-web', 'dist') + '/';
  const bytes = new Uint8Array(readFileSync(MODEL_PATH));
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  return { ort: ort as unknown as OrtModuleLike, session: session as unknown as OrtSessionLike };
}

describe.skipIf(!hasModel)('onnxruntime-web WebAssembly build (run in Node)', () => {
  it('agrees with PyTorch on the fixture positions', async () => {
    const { ort, session } = await webSession(1);
    const ev = new OrtEvaluator(ort, session);
    let policyMax = 0;
    let legalMax = 0;
    let valueMax = 0;
    let top1 = 0;
    const positions = fx.positions;
    for (let start = 0; start < positions.length; start += 9) {
      const chunk = positions.slice(start, start + 9);
      const planes = new Float32Array(chunk.length * PLANE_FLOATS);
      chunk.forEach((p, i) => planes.set(p.planes, i * PLANE_FLOATS));
      const out = await ev.evaluate(planes, chunk.length);
      chunk.forEach((p, i) => {
        const ref = decodeLogits(p.logits_f32_b64);
        const got = out.policy.subarray(i * ACTION_SIZE, (i + 1) * ACTION_SIZE);
        for (let k = 0; k < ACTION_SIZE; k++) policyMax = Math.max(policyMax, Math.abs(ref[k]! - got[k]!));
        let best = p.legal_idx[0]!;
        let refBest = best;
        for (const k of p.legal_idx) {
          legalMax = Math.max(legalMax, Math.abs(ref[k]! - got[k]!));
          if (got[k]! > got[best]!) best = k;
          if (ref[k]! > ref[refBest]!) refBest = k;
        }
        if (best === refBest) top1++;
        valueMax = Math.max(valueMax, Math.abs(p.value - out.value[i]!));
      });
    }
    const result = {
      positions: positions.length,
      policyMaxAbs: policyMax,
      policyLegalMaxAbs: legalMax,
      valueMaxAbs: valueMax,
      legalTop1Agree: top1 / positions.length,
    };
    console.log('[parity] onnxruntime-web WASM vs PyTorch:', result);
    writeOut('ortweb_wasm_parity.json', result);
    expect(policyMax).toBeLessThan(1e-3);
    expect(valueMax).toBeLessThan(1e-4);
    expect(result.legalTop1Agree).toBe(1);
  }, 300_000);

  it('forward pass time at batch 1 and 8, single thread', async () => {
    const result: Record<string, unknown> = {};
    // Single thread only: thread scaling of this build inside Node/vitest was erratic, and says little about browsers.
    for (const threads of [1]) {
      const { ort, session } = await webSession(threads);
      const ev = new OrtEvaluator(ort, session);
      const table: Record<string, unknown> = {};
      for (const batch of [1, 8]) {
        const planes = new Float32Array(batch * PLANE_FLOATS);
        for (let i = 0; i < batch; i++) planes.set(fx.positions[i]!.planes, i * PLANE_FLOATS);
        await ev.evaluate(planes, batch);
        const times: number[] = [];
        for (let i = 0; i < 4; i++) {
          const t = performance.now();
          await ev.evaluate(planes, batch);
          times.push(performance.now() - t);
        }
        times.sort((a, b) => a - b);
        const median = times[Math.floor(times.length / 2)]!;
        table[`batch${batch}`] = { medianMs: Math.round(median), positionsPerSecond: Number(((batch * 1000) / median).toFixed(1)) };
      }
      result[`${threads} thread${threads > 1 ? 's' : ''}`] = table;
    }
    console.log('[timing] onnxruntime-web WASM in Node:', JSON.stringify(result));
    writeOut('timing_ortweb_wasm.json', result);
    expect(result).toBeTruthy();
  }, 300_000);
});
