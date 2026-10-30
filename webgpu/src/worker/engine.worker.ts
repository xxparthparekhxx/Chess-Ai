// The engine: loads the network, runs the search. Nothing here touches the DOM.

import { Mcts } from '../engine/mcts';
import { MultiSessionEvaluator, OrtEvaluator } from '../engine/ortEvaluator';
import { PLANE_FLOATS, encodeFen } from '../engine/position';
import type { EvalOutput, Evaluator } from '../engine/types';
import type { BackendName, CalibrationRow, EvaluateRequest, FromWorker, InitRequest, SearchRequest, ToWorker } from '../messages';
import { clearModelCache, loadModel } from './modelLoader';
import { cpuIsFaster, needsCpuCheck, slowGpuNote } from './backendChoice';
import { createSession, type CreatedSession } from './ortBackend';

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
  constructor(private readonly inner: Evaluator) {}
  async evaluate(planes: Float32Array, n: number): Promise<EvalOutput> {
    const out = await this.inner.evaluate(planes, n);
    await yieldToEventLoop();
    return out;
  }
}

let mcts: Mcts | null = null;
let evaluator: Evaluator | null = null;
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
/** How long a GPU backend is kept busy at the first batch size before it is timed. */
const WARM_MS = 500;

async function calibrate(ev: Evaluator, backend: BackendName): Promise<{ rows: CalibrationRow[]; batch: number }> {
  const start = encodeFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const gpu = backend !== 'wasm';
  // WebGL only has a batch-1 session (its Conv is wrong for larger batches), so batching gives it nothing
  const sizes = backend === 'webgpu' ? [1, 8, 16, 32] : backend === 'webgl' ? [1] : [1, 4, 8];
  const candidates = backend === 'webgpu' ? [8, 16, 32] : backend === 'webgl' ? [1] : [1, 4, 8];
  const timedRuns = backend === 'webgpu' ? 3 : backend === 'webgl' ? 5 : 1;
  const rows: CalibrationRow[] = [];
  for (const batch of sizes) {
    const planes = new Float32Array(batch * PLANE_FLOATS);
    for (let i = 0; i < batch; i++) planes.set(start, i * PLANE_FLOATS);
    // The first run at a new shape compiles shaders / allocates buffers on the GPU; not timed. A GPU that has been
    // idle also takes a few hundred milliseconds of work to leave its low-power state (a laptop's dedicated card
    // measured 26 and 166 positions/s on a cold and a warm start), so it is kept busy for a moment before timing.
    if (gpu || batch === sizes[0]) await ev.evaluate(planes, batch);
    if (gpu) {
      const warmUntil = performance.now() + (batch === sizes[0] ? WARM_MS : WARM_MS / 4);
      while (performance.now() < warmUntil) await ev.evaluate(planes, batch);
    }
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

/** Builds the evaluator for a created session, warms it up and times it. */
interface Prepared {
  rows: CalibrationRow[];
  batch: number;
  buckets: number[] | null;
  ev: Evaluator;
}

async function release(created: CreatedSession): Promise<void> {
  for (const s of new Set([created.session, ...created.sessions.values()])) await s.release().catch(() => undefined);
}

async function prepare(created: CreatedSession): Promise<Prepared> {
  // On WebGPU every new input shape can trigger a shader compile, so batches are padded up to a few
  // fixed sizes that get warmed up below. WebGL has one session per fixed size. On the CPU cost is
  // proportional to batch size, so no padding.
  const buckets = created.backend === 'webgpu' ? [1, 8, 16, 32] : created.backend === 'webgl' ? [...created.sessions.keys()] : null;
  const ev: Evaluator =
    created.backend === 'webgl' ? new MultiSessionEvaluator(created.ort, created.sessions) : new OrtEvaluator(created.ort, created.session, buckets);
  evaluator = ev;
  const warm = {
    webgpu: 'Compiling WebGPU shaders and timing batch sizes',
    webgl: 'Compiling WebGL shaders and timing the network',
    wasm: `Warming up the CPU runtime (${created.threads} thread${created.threads === 1 ? '' : 's'})`,
  }[created.backend];
  post({ type: 'progress', phase: 'warmup', message: warm, backend: created.backend });
  const { rows, batch } = await calibrate(ev, created.backend);
  return { rows, batch, buckets, ev };
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
        message: fromCache ? 'Loaded the network from this browser\'s cache, no download needed' : 'Downloading the network',
      });
    });

    post({ type: 'progress', phase: 'runtime', message: 'Starting the inference runtime' });
    const onStep = (message: string) => post({ type: 'progress', phase: 'compile', message });
    let created = await createSession(model.bytes, req.backend, req.ortExternalBase, onStep, req.threads ?? 0);
    let prepared = await prepare(created);

    // A GPU that is not clearly fast is timed against the CPU, and the faster one is used. Only on automatic;
    // ?backend=webgl / webgpu stays what was asked for.
    const rate = (p: Prepared) => Math.max(...p.rows.map((r) => r.positionsPerSecond));
    if (needsCpuCheck(req.backend, created.backend, rate(prepared))) {
      post({ type: 'progress', phase: 'runtime', message: 'Checking that this GPU is faster than the CPU' });
      const gpu = { created, prepared };
      const cpuCreated = await createSession(model.bytes, 'wasm', req.ortExternalBase, onStep, req.threads ?? 0);
      const cpuPrepared = await prepare(cpuCreated);
      if (cpuIsFaster(rate(gpu.prepared), rate(cpuPrepared))) {
        const why = slowGpuNote(gpu.created.backend, gpu.created.gpu, rate(gpu.prepared), rate(cpuPrepared));
        await release(gpu.created);
        cpuCreated.webgpuNote = gpu.created.backend === 'webgpu' ? why : gpu.created.webgpuNote;
        cpuCreated.webgpuProblem = gpu.created.backend === 'webgpu' ? 'slow-gpu' : gpu.created.webgpuProblem;
        cpuCreated.webglNote = gpu.created.backend === 'webgl' ? why : gpu.created.webglNote;
        cpuCreated.gpu = gpu.created.gpu;
        created = cpuCreated;
        prepared = cpuPrepared;
      } else {
        await release(cpuCreated);
        evaluator = gpu.prepared.ev;
      }
    }
    const { rows, batch, buckets, ev } = prepared;
    autoBatch = batch;
    mcts = new Mcts(new YieldingEvaluator(ev));

    post({
      type: 'ready',
      info: {
        backend: created.backend,
        runtime: created.runtime,
        threads: created.threads,
        crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
        webgpuNote: created.webgpuNote,
        webgpuProblem: created.webgpuProblem,
        webglNote: created.webglNote,
        gpu: created.gpu,
        buckets,
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

async function evaluate(req: EvaluateRequest): Promise<void> {
  if (!evaluator) {
    post({ type: 'evaluate-error', id: req.id, message: 'The engine is not ready yet.' });
    return;
  }
  try {
    const t = performance.now();
    const out = await evaluator.evaluate(req.planes, req.n);
    const ms = performance.now() - t;
    // copy: the evaluator may hand out views of its own buffers
    post({ type: 'evaluate-result', id: req.id, policy: out.policy.slice(), value: out.value.slice(), ms });
  } catch (err) {
    post({ type: 'evaluate-error', id: req.id, message: err instanceof Error ? err.message : String(err) });
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
    case 'evaluate':
      queue = queue.then(() => evaluate(msg));
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
