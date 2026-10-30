// Runs the fixture positions through the built site's engine worker in headless Chrome, once per backend,
// and compares the network output with PyTorch (test/fixtures/fixtures.json). Also records search speed.
//
// Variants (see site.mjs): wasm, webgl-swiftshader, webgpu-swiftshader (software renderer: accuracy counts, speed
// does not), webgl-igpu, webgl-nvidia, webgpu-nvidia (real GPUs of the machine it was run on).
// Usage: node test/browser/parity.mjs [variant ...]
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { launchChrome } from './chrome.mjs';
import { VARIANTS, WEB_ROOT, blankPage, pageEngine, startPreview, workerPath } from './site.mjs';

const wanted = process.argv.slice(2).filter((a) => !a.startsWith('-'));
const variants = wanted.length ? wanted : Object.keys(VARIANTS);
const fx = JSON.parse(readFileSync(join(WEB_ROOT, 'test', 'fixtures', 'fixtures.json'), 'utf8'));
const positions = fx.positions.map((p) => ({ name: p.name, fen: p.fen, planes: p.planes, legal: p.legal_idx, logits: p.logits_f32_b64, value: p.value }));
const SEARCH_FENS = [
  fx.positions.find((p) => p.name === 'start position').fen,
  fx.positions.find((p) => p.name === 'Kiwipete').fen,
  fx.positions.find((p) => p.name === 'rook endgame').fen,
];

async function inPage({ engineSrc, backend, positions, searchFens, searchMs }) {
  const engine = eval(engineSrc);
  const info = await engine.init({ backend });
  const b64 = (s) => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); };
  let policyMax = 0, legalMax = 0, valueMax = 0, top1 = 0;
  const mismatches = [];
  const BATCH = 8;
  for (let at = 0; at < positions.length; at += BATCH) {
    const chunk = positions.slice(at, at + BATCH);
    const planes = new Float32Array(chunk.length * 896);
    chunk.forEach((p, i) => planes.set(p.planes, i * 896));
    const r = await engine.evaluate(planes, chunk.length);
    chunk.forEach((p, i) => {
      const ref = b64(p.logits);
      const got = r.policy.subarray(i * 4544, (i + 1) * 4544);
      for (let k = 0; k < 4544; k++) policyMax = Math.max(policyMax, Math.abs(ref[k] - got[k]));
      let best = p.legal[0], refBest = best;
      for (const k of p.legal) {
        legalMax = Math.max(legalMax, Math.abs(ref[k] - got[k]));
        if (got[k] > got[best]) best = k;
        if (ref[k] > ref[refBest]) refBest = k;
      }
      if (best === refBest) top1++;
      else mismatches.push({ name: p.name, margin: ref[refBest] - ref[best] });
      valueMax = Math.max(valueMax, Math.abs(p.value - r.value[i]));
    });
  }
  const searches = [];
  for (const fen of searchFens) {
    const r = await engine.search({ fen, timeBudgetMs: searchMs });
    searches.push({ simulations: r.result.simulations, simsPerSecond: r.result.simsPerSecond, move: r.result.move });
  }
  engine.stop();
  return { info, policyMax, legalMax, valueMax, top1, count: positions.length, mismatches, searches };
}

const preview = await startPreview();
const results = {};
let failed = false;
try {
  for (const variant of variants) {
    const { backend, renderer, args, env } = VARIANTS[variant];
    const chrome = await launchChrome(args, env);
    try {
      await chrome.goto(preview.url + blankPage());
      const isolated = await chrome.run(() => crossOriginIsolated);
      // Headless Chrome hands out its GPU adapter unreliably (about one run in three says "Failed to get GPU
      // adapter" although the probe just saw one), so a run that ended on another backend is retried.
      let r;
      for (let attempt = 1; attempt <= 4; attempt++) {
        r = await chrome.run(inPage, { engineSrc: pageEngine(workerPath()), backend, positions, searchFens: SEARCH_FENS, searchMs: 3000 });
        if (r.info.backend === backend) break;
        console.log(variant, `attempt ${attempt} ran on ${r.info.backend}: ${r.info.webgpuNote ?? r.info.webglNote}`);
      }
      const ok = r.info.backend === backend && r.legalMax < 1e-3 && r.valueMax < 1e-3 && r.top1 === r.count;
      if (!ok) failed = true;
      results[variant] = {
        browser: await chrome.version(),
        renderer,
        isolated,
        runtime: r.info.runtime,
        threads: r.info.threads,
        webgpuNote: r.info.webgpuNote,
        webglNote: r.info.webglNote,
        positions: r.count,
        policyMaxAbsAllLogits: r.policyMax,
        policyMaxAbsLegal: r.legalMax,
        valueMaxAbs: r.valueMax,
        sameBestLegalMove: `${r.top1}/${r.count}`,
        mismatches: r.mismatches,
        autoBatch: r.info.autoBatchSize,
        calibration: r.info.calibration,
        simsPerSecond: r.searches.map((x) => Math.round(x.simsPerSecond * 10) / 10),
        pass: ok,
      };
      console.log(variant, JSON.stringify(results[variant]));
    } catch (err) {
      failed = true;
      results[variant] = { error: String(err.message ?? err), console: chrome.consoleLines.slice(-8) };
      console.log(variant, 'FAILED', results[variant]);
    } finally {
      await chrome.close();
    }
  }
} finally {
  preview.stop();
}
mkdirSync(join(WEB_ROOT, 'test', 'out'), { recursive: true });
writeFileSync(join(WEB_ROOT, 'test', 'out', 'browser_parity.json'), JSON.stringify(results, null, 2));
process.exit(failed ? 1 : 0);
