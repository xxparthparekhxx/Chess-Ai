import { beforeAll, describe, expect, it } from 'vitest';
import { ACTION_SIZE } from '../src/engine/moveMap';
import { PLANE_FLOATS } from '../src/engine/position';
import { OrtEvaluator } from '../src/engine/ortEvaluator';
import { decodeLogits, loadFixtures } from './helpers/fixtures';
import { createNodeSession, FP16_MODEL_PATH, hasFp16Model, hasModel, writeOut } from './helpers/ortNode';

const fx = loadFixtures();
const n = fx.positions.length;

function allPlanes(): Float32Array {
  const planes = new Float32Array(n * PLANE_FLOATS);
  fx.positions.forEach((p, i) => planes.set(p.planes, i * PLANE_FLOATS));
  return planes;
}

interface Errors {
  policyMaxAbs: number;
  policyLegalMaxAbs: number;
  valueMaxAbs: number;
  logitRange: number;
  top1Agree: number;
  legalTop1Agree: number;
}

async function measure(path: string): Promise<Errors> {
  const { ort, session } = await createNodeSession(path);
  const ev = new OrtEvaluator(ort, session);
  const out = await ev.evaluate(allPlanes(), n);
  let policyMax = 0;
  let legalMax = 0;
  let valueMax = 0;
  let range = 0;
  let top1 = 0;
  let legalTop1 = 0;
  fx.positions.forEach((p, i) => {
    const ref = decodeLogits(p.logits_f32_b64);
    const got = out.policy.subarray(i * ACTION_SIZE, (i + 1) * ACTION_SIZE);
    let best = 0;
    let refBest = 0;
    for (let k = 0; k < ACTION_SIZE; k++) {
      policyMax = Math.max(policyMax, Math.abs(ref[k]! - got[k]!));
      range = Math.max(range, Math.abs(ref[k]!));
      if (got[k]! > got[best]!) best = k;
      if (ref[k]! > ref[refBest]!) refBest = k;
    }
    if (best === refBest) top1++;
    let lb = p.legal_idx[0]!;
    let lrb = p.legal_idx[0]!;
    for (const k of p.legal_idx) {
      legalMax = Math.max(legalMax, Math.abs(ref[k]! - got[k]!));
      if (got[k]! > got[lb]!) lb = k;
      if (ref[k]! > ref[lrb]!) lrb = k;
    }
    if (lb === lrb) legalTop1++;
    valueMax = Math.max(valueMax, Math.abs(p.value - out.value[i]!));
  });
  return {
    policyMaxAbs: policyMax,
    policyLegalMaxAbs: legalMax,
    valueMaxAbs: valueMax,
    logitRange: range,
    top1Agree: top1 / n,
    legalTop1Agree: legalTop1 / n,
  };
}

describe.skipIf(!hasModel)('ONNX (onnxruntime-node, CPU) vs PyTorch', () => {
  let errors: Errors;
  beforeAll(async () => {
    errors = await measure(process.env.SIGMA_MODEL ?? (await import('./helpers/ortNode')).MODEL_PATH);
    console.log(`[parity] fp32 ONNX vs PyTorch on ${n} positions:`, errors);
    writeOut('network_parity_fp32.json', { positions: n, ...errors });
  });

  it('policy logits agree to within 1e-3 (max abs error is printed and saved)', () => {
    expect(errors.policyMaxAbs).toBeLessThan(1e-3);
  });

  it('value agrees to within 1e-4', () => {
    expect(errors.valueMaxAbs).toBeLessThan(1e-4);
  });

  it('picks the same top move among legal moves on every position', () => {
    expect(errors.legalTop1Agree).toBe(1);
  });

  it('gives the same answer for a position alone, in a batch, and padded to a bucket', async () => {
    const { ort, session } = await createNodeSession();
    const plain = new OrtEvaluator(ort, session);
    const bucketed = new OrtEvaluator(ort, session, [1, 8, 16, 32]);
    const planes = allPlanes();
    const batch = await plain.evaluate(planes.subarray(0, 5 * PLANE_FLOATS), 5);
    const batchPolicy = Float32Array.from(batch.policy);
    const batchValue = Float32Array.from(batch.value);
    const padded = await bucketed.evaluate(planes.subarray(0, 5 * PLANE_FLOATS), 5);
    expect(padded.value.length).toBe(5);
    expect(padded.policy.length).toBe(5 * ACTION_SIZE);
    let worst = 0;
    for (let i = 0; i < batchPolicy.length; i++) worst = Math.max(worst, Math.abs(batchPolicy[i]! - padded.policy[i]!));
    for (let i = 0; i < 5; i++) worst = Math.max(worst, Math.abs(batchValue[i]! - padded.value[i]!));
    for (let i = 0; i < 5; i++) {
      const one = await plain.evaluate(planes.subarray(i * PLANE_FLOATS, (i + 1) * PLANE_FLOATS), 1);
      for (let k = 0; k < ACTION_SIZE; k++) worst = Math.max(worst, Math.abs(one.policy[k]! - batchPolicy[i * ACTION_SIZE + k]!));
    }
    console.log(`[parity] largest difference between batch 1, batch 5 and padded batch 8: ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-4);
  });
});

describe.skipIf(!hasModel || !hasFp16Model)('ONNX with fp16 weights (optional variant)', () => {
  it('reports its error against PyTorch', async () => {
    const e = await measure(FP16_MODEL_PATH);
    console.log(`[parity] fp16-weights ONNX vs PyTorch on ${n} positions:`, e);
    writeOut('network_parity_fp16w.json', { positions: n, ...e });
    expect(e.legalTop1Agree).toBeGreaterThan(0.9);
  });
});
