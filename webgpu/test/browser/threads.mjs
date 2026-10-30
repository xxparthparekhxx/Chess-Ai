// Measures CPU-backend search speed at several thread counts, in headless Chrome, to pick the thread rule in
// src/worker/threads.ts. Usage: node test/browser/threads.mjs [1 2 4 8 ...]
// Speed depends on what else is running on the machine; run it on a quiet one and read the trend, not one number.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { launchChrome } from './chrome.mjs';
import { WEB_ROOT, blankPage, pageEngine, startPreview, workerPath } from './site.mjs';

const counts = process.argv.slice(2).map(Number).filter((n) => n > 0);
const threadCounts = counts.length ? counts : [1, 2, 3, 4, 6, 8];
const fx = JSON.parse(readFileSync(join(WEB_ROOT, 'test', 'fixtures', 'fixtures.json'), 'utf8'));
const fens = ['start position', 'Kiwipete', 'rook endgame'].map((n) => fx.positions.find((p) => p.name === n).fen);

async function inPage({ engineSrc, threads, fens, ms }) {
  const engine = eval(engineSrc);
  const info = await engine.init({ backend: 'wasm', threads });
  const rows = [];
  for (const fen of fens) {
    const r = await engine.search({ fen, timeBudgetMs: ms });
    rows.push(r.result.simsPerSecond);
  }
  // raw network speed at batch 1, no search
  const planes = new Float32Array(896);
  await engine.evaluate(planes, 1);
  const t = performance.now();
  for (let i = 0; i < 20; i++) await engine.evaluate(planes, 1);
  const fwdMs = (performance.now() - t) / 20;
  engine.stop();
  return { threads: info.threads, batch: info.autoBatchSize, rows, fwdMs };
}

const preview = await startPreview();
const out = [];
try {
  for (const threads of threadCounts) {
    const chrome = await launchChrome([]);
    try {
      await chrome.goto(preview.url + blankPage());
      const r = await chrome.run(inPage, { engineSrc: pageEngine(workerPath()), threads, fens, ms: 3000 });
      const mean = r.rows.reduce((a, b) => a + b, 0) / r.rows.length;
      out.push({ requested: threads, ...r, meanSimsPerSecond: Math.round(mean * 10) / 10 });
      console.log(JSON.stringify(out.at(-1)));
    } finally {
      await chrome.close();
    }
  }
} finally {
  preview.stop();
}
mkdirSync(join(WEB_ROOT, 'test', 'out'), { recursive: true });
writeFileSync(join(WEB_ROOT, 'test', 'out', 'browser_threads.json'), JSON.stringify(out, null, 2));
