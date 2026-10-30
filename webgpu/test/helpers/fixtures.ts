import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
export const WEB_ROOT = resolve(here, '..', '..');
export const FIXTURE_PATH = resolve(WEB_ROOT, 'test', 'fixtures', 'fixtures.json');

export interface FixturePosition {
  name: string;
  category: string;
  fen: string;
  turn: 'w' | 'b';
  planes: number[];
  legal_uci: string[];
  legal_idx: number[];
  logits_f32_b64: string;
  value: number;
}

export interface MctsResult {
  index: number;
  fen: string;
  move: string;
  value: number;
  visits: Record<string, number>;
}

export interface Fixtures {
  meta: Record<string, string | number | number[]>;
  move_labels: string[];
  positions: FixturePosition[];
  mcts_runs: { simulations: number; batch_size: number; c_puct: number; results: MctsResult[] }[];
  order_cases: { fen: string; legal_uci: string[] }[];
  terminal_cases: { fen: string; label: string; terminal: boolean; value: number }[];
}

let cached: Fixtures | null = null;

export function loadFixtures(): Fixtures {
  cached ??= JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as Fixtures;
  return cached;
}

export function decodeLogits(b64: string): Float32Array {
  const buf = Buffer.from(b64, 'base64');
  // copy: a Buffer slice may not be 4-byte aligned
  const out = new Float32Array(buf.length / 4);
  new Uint8Array(out.buffer).set(buf);
  return out;
}
