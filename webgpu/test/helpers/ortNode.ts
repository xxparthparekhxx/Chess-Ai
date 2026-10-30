import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { WEB_ROOT } from './fixtures';

export const MODEL_PATH = process.env.SIGMA_MODEL ?? resolve(WEB_ROOT, 'model', 'chess_net.onnx');
export const FP16_MODEL_PATH = resolve(dirname(MODEL_PATH), 'chess_net.fp16w.onnx');
export const hasModel = existsSync(MODEL_PATH);
export const hasFp16Model = existsSync(FP16_MODEL_PATH);

if (!hasModel) {
  console.warn(
    `\n[parity] ${MODEL_PATH} not found: network and search tests are skipped.\n` +
      `[parity] Create it with:  .venv/bin/python scripts/export_onnx.py   (from the repo root)\n`,
  );
}

export async function createNodeSession(path = MODEL_PATH, intraOpNumThreads?: number) {
  const ort = await import('onnxruntime-node');
  const session = await ort.InferenceSession.create(path, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    ...(intraOpNumThreads ? { intraOpNumThreads } : {}),
  });
  return { ort, session };
}

/** Write a small JSON result file under test/out/ (git-ignored). */
export function writeOut(name: string, data: unknown): void {
  const dir = resolve(WEB_ROOT, 'test', 'out');
  mkdirSync(dir, { recursive: true });
  writeFileSync(resolve(dir, name), JSON.stringify(data, null, 2));
}
