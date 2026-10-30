// The engine: loads the network, runs the search. Nothing here touches the DOM.

import { Mcts } from '../engine/mcts';
import { OrtEvaluator } from '../engine/ortEvaluator';
import { PLANE_FLOATS, encodeFen } from '../engine/position';
import type { EvalOutput, Evaluator } from '../engine/types';
import type { CalibrationRow, FromWorker, InitRequest, ReadyInfo, SearchRequest, ToWorker } from '../messages';
import { clearModelCache, loadModel } from './modelLoader';
import { createSession } from './ortBackend';

const ctx = self as unknown as {
  postMessage(message: FromWorker): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};
const post = (m: FromWorker) => ctx.postMessage(m);

/** Let queued messages (a stop request) run between batches; CPU inference never awaits anything real. */
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    const { port1, port2 } = new MessageChannel();
    port1.onmessage = () => {
      port1.close();
      resolve();
    };
    port2.postMessage(0);
  });
}

class YieldingEvaluator implements Evaluator {
  constructor(private readonly inner: OrtEvaluator) {}
  async evaluate(planes: Float32Array, n: number): Promise<EvalOutput> {
    const out = await this.inner.evaluate(planes, n);
    await yieldToEventLoop();
    return out;
  }
}

let mcts: Mcts | null = null;
let autoBatch = 8;
let running: { id: number; signal: { aborted: boolean } } | null = null;
const abortedEarly = new Set<number>();
let queue: Promise<void> = Promise.resolve();

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}

/**
 * Time the network at a few batch sizes on this device and pick the batch size for search.
 * Search quality drops a little with bigger batches (virtual loss), so the batch only grows while the
 * gain is worth it: at least 15% more positions per second than the previous size, and under 300 ms
 * per network call. WebGPU starts from 8 (the Python engine's batch); on the CPU a batch gives no
 * speed-up, so it starts from 1.
 */
async function calibrate(ev: OrtEvaluator, backend: ReadyInfo['backend']): Promise<{ rows: CalibrationRow[]; batch: number }> {
  const start = encodeFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const gpu = backend === 'webgpu';
  const sizes = gpu ? [1, 8, 16, 32] : [1, 4, 8];
  const candidates = gpu ? [8, 16, 32] : [1, 4, 8];
  const timedRuns = gpu ? 3 : 1;
  const rows: CalibrationRow[] = [];
  for (const batch of sizes) {
    const planes = new Float32Array(batch * PLANE_FLOATS);
    for (let i = 0; i < batch; i++) planes.set(start, i * PLANE_FLOATS);
    // The first run at a new shape compiles shaders / allocates buffers on the GPU; not timed.
    if (gpu || batch === sizes[0]) await ev.evaluate(planes, batch);
    const times: number[] = [];
    for (let i = 0; i < timedRuns; i++) {
      const t = performance.now();
      await ev.evaluate(planes, batch);
      times.push(performance.now() - t);
    }
    const ms = median(times);
    rows.push({ batch, ms, positionsPerSecond: (batch * 1000) / ms });
    if (ms > 700 && batch >= 8) break; // already slow; do not try larger batches
  }
  let chosen = candidates[0]!;
  let chosenRate = rows.find((r) => r.batch === chosen)?.positionsPerSecond ?? 0;
  for (const c of candidates.slice(1)) {
    const r = rows.find((x) => x.batch === c);
    if (!r) break;
    if (r.ms <= 300 && r.positionsPerSecond >= chosenRate * 1.15) {
      chosen = c;
      chosenRate = r.positionsPerSecond;
    } else break;
  }
  return { rows, batch: chosen };
}

async function init(req: InitRequest): Promise<void> {
  const t0 = performance.now();
  try {
    post({ type: 'progress', phase: 'download', loaded: 0, message: 'Looking for the network' });
    let lastPost = 0;
    const model = await loadModel(req.modelUrl, req.modelVersion, (loaded, total, fromCache) => {
      const now = performance.now();
      if (now - lastPost < 80 && total !== loaded) return;
      lastPost = now;
      post({
        type: 'progress',
        phase: fromCache ? 'cache' : 'download',
        loaded,
        total,
        message: fromCache ? 'Loaded the network from this browser\'s cache' : 'Downloading the network',
      });
    });

    post({ type: 'progress', phase: 'runtime', message: 'Starting the inference runtime' });
    const created = await createSession(model.bytes, req.backend, req.ortExternalBase, (message) =>
      post({ type: 'progress', phase: 'compile', message }),
    );

    // On the GPU every new input shape can trigger a shader compile, so batches are padded up to a few
    // fixed sizes that get warmed up below. On the CPU cost is proportional to batch size, so no padding.
    const evaluator = new OrtEvaluator(created.ort, created.session, created.backend === 'webgpu' ? [1, 8, 16, 32] : null);
    post({ type: 'progress', phase: 'warmup', message: created.backend === 'webgpu' ? 'Compiling GPU shaders and timing batch sizes' : 'Warming up the CPU runtime' });
    const { rows, batch } = await calibrate(evaluator, created.backend);
    autoBatch = batch;
    mcts = new Mcts(new YieldingEvaluator(evaluator));

    post({
      type: 'ready',
      info: {
        backend: created.backend,
        runtime: created.runtime,
        threads: created.threads,
        crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
        webgpuNote: created.webgpuNote,
        modelBytes: model.bytes.byteLength,
        fromCache: model.fromCache,
        loadMs: performance.now() - t0,
        calibration: rows,
        autoBatchSize: batch,
      },
    });
  } catch (err) {
    post({ type: 'init-error', message: err instanceof Error ? err.message : String(err) });
  }
}

async function search(req: SearchRequest): Promise<void> {
  if (!mcts) {
    post({ type: 'search-error', id: req.id, message: 'The engine is not ready yet.' });
    return;
  }
  const signal = { aborted: abortedEarly.delete(req.id) };
  running = { id: req.id, signal };
  try {
    const result = await mcts.search(req.fen, {
      timeBudgetMs: req.timeBudgetMs,
      batchSize: req.batchSize > 0 ? req.batchSize : autoBatch,
      temperature: req.temperature,
      dirichlet: req.noise ? { alpha: 0.3, fraction: 0.25 } : false,
      signal,
      progressIntervalMs: 100,
      onProgress: (progress) => post({ type: 'search-progress', id: req.id, progress }),
    });
    post({ type: 'search-result', id: req.id, result });
  } catch (err) {
    post({ type: 'search-error', id: req.id, message: err instanceof Error ? err.message : String(err) });
  } finally {
    if (running?.id === req.id) running = null;
  }
}

ctx.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init':
      void init(msg);
      break;
    case 'search':
      // one search at a time; a new request waits for the previous one to finish
      queue = queue.then(() => search(msg));
      break;
    case 'abort':
      if (running && running.id === msg.id) running.signal.aborted = true;
      else abortedEarly.add(msg.id); // asked to stop before it started
      break;
    case 'clear-cache':
      void clearModelCache().then((ok) => post({ type: 'cache-cleared', ok }));
      break;
  }
};
