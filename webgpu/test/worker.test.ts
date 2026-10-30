// Drives src/worker/engine.worker.ts through its message interface in Node: the real download / cache
// code and the real search, with onnxruntime-node standing in for onnxruntime-web (the one piece that only
// exists in a browser). Checks the protocol, progress reporting, start-up timing, queuing and stopping.
import { readFileSync } from 'node:fs';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import type { FromWorker, ToWorker } from '../src/messages';
import { hasModel, MODEL_PATH } from './helpers/ortNode';

const inbox: FromWorker[] = [];
const waiters: (() => void)[] = [];
const fakeSelf: { postMessage(m: FromWorker): void; onmessage: ((e: { data: ToWorker }) => void) | null } = {
  postMessage(m) {
    inbox.push(m);
    waiters.splice(0).forEach((w) => w());
  },
  onmessage: null,
};

vi.mock('../src/worker/ortBackend', async () => {
  const { createNodeSession: make } = await import('./helpers/ortNode');
  return {
    createSession: async () => {
      const { ort, session } = await make();
      return { ort, session, backend: 'wasm', runtime: 'wasm', threads: 1, webgpuNote: 'test double' };
    },
  };
});

function send(m: ToWorker) {
  fakeSelf.onmessage!({ data: m });
}

async function until<T>(pick: () => T | undefined, ms = 60_000): Promise<T> {
  const start = Date.now();
  for (;;) {
    const v = pick();
    if (v !== undefined) return v;
    if (Date.now() - start > ms) throw new Error('timed out waiting for the worker');
    await new Promise<void>((r) => {
      waiters.push(r);
      setTimeout(r, 25);
    });
  }
}

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

describe.skipIf(!hasModel)('engine worker', () => {
  beforeAll(async () => {
    const bytes = readFileSync(MODEL_PATH);
    const cacheStore = new Map<string, Response>();
    vi.stubGlobal('self', fakeSelf);
    vi.stubGlobal('caches', {
      open: async () => ({
        match: async (u: string) => cacheStore.get(u)?.clone(),
        put: async (u: string, r: Response) => void cacheStore.set(u, new Response(await r.arrayBuffer(), { headers: r.headers })),
        delete: async (u: string) => cacheStore.delete(u),
      }),
      keys: async () => [],
      delete: async () => true,
    });
    vi.stubGlobal('fetch', async () => new Response(bytes, { status: 200, headers: { 'content-length': String(bytes.length) } }));
    await import('../src/worker/engine.worker');
    expect(fakeSelf.onmessage).toBeTypeOf('function');
  });

  afterAll(() => vi.unstubAllGlobals());

  it('loads the network, reports progress, times batch sizes and says it is ready', async () => {
    send({ type: 'init', modelUrl: 'https://example.test/chess_net.onnx', modelVersion: 't', backend: 'auto', ortExternalBase: '' });
    const ready = await until(() => inbox.find((m) => m.type === 'ready' || m.type === 'init-error'));
    if (ready.type !== 'ready') throw new Error(ready.type === 'init-error' ? ready.message : 'unexpected');
    const downloads = inbox.filter((m) => m.type === 'progress' && m.phase === 'download') as Extract<FromWorker, { type: 'progress' }>[];
    expect(downloads.length).toBeGreaterThan(0);
    expect(downloads.at(-1)!.loaded).toBe(readFileSync(MODEL_PATH).length);
    const info = ready.info;
    expect(info.backend).toBe('wasm');
    expect(info.modelBytes).toBeGreaterThan(40_000_000);
    expect(info.calibration.map((r) => r.batch)).toEqual([1, 4, 8]);
    expect(info.calibration.every((r) => r.ms > 0 && r.positionsPerSecond > 0)).toBe(true);
    expect([1, 4, 8]).toContain(info.autoBatchSize);
  });

  it('searches for the requested time and streams progress', async () => {
    inbox.length = 0;
    send({ type: 'search', id: 1, fen: START, timeBudgetMs: 400, batchSize: 8, temperature: 0, noise: true });
    const result = await until(() => inbox.find((m) => m.type === 'search-result' && m.id === 1));
    if (result.type !== 'search-result') throw new Error('unexpected');
    expect(result.result.move).toMatch(/^[a-h][1-8][a-h][1-8]/);
    expect(result.result.simulations).toBeGreaterThanOrEqual(8);
    expect(result.result.elapsedMs).toBeGreaterThanOrEqual(400);
    expect(inbox.filter((m) => m.type === 'search-progress').length).toBeGreaterThan(0);
  });

  it('stops a long search when asked, and still returns a move', async () => {
    inbox.length = 0;
    send({ type: 'search', id: 2, fen: START, timeBudgetMs: 30_000, batchSize: 8, temperature: 0, noise: false });
    await until(() => inbox.find((m) => m.type === 'search-progress' && m.id === 2));
    const t = performance.now();
    send({ type: 'abort', id: 2 });
    const result = await until(() => inbox.find((m) => m.type === 'search-result' && m.id === 2));
    if (result.type !== 'search-result') throw new Error('unexpected');
    expect(performance.now() - t).toBeLessThan(2000);
    expect(result.result.aborted).toBe(true);
    expect(result.result.move).not.toBeNull();
  });

  it('runs searches one after another and honours a stop sent before the search starts', async () => {
    inbox.length = 0;
    send({ type: 'search', id: 3, fen: START, timeBudgetMs: 200, batchSize: 0, temperature: 0, noise: false });
    send({ type: 'abort', id: 4 });
    send({ type: 'search', id: 4, fen: START, timeBudgetMs: 30_000, batchSize: 8, temperature: 0, noise: false });
    const r3 = await until(() => inbox.find((m) => m.type === 'search-result' && m.id === 3));
    const r4 = await until(() => inbox.find((m) => m.type === 'search-result' && m.id === 4));
    if (r3.type !== 'search-result' || r4.type !== 'search-result') throw new Error('unexpected');
    expect(r3.result.aborted).toBe(false);
    expect(r4.result.aborted).toBe(true);
    expect(r4.result.simulations).toBe(0);
  });

  it('reports an error for a position it cannot search', async () => {
    inbox.length = 0;
    send({ type: 'search', id: 5, fen: 'not a fen', timeBudgetMs: 100, batchSize: 8, temperature: 0, noise: false });
    const m = await until(() => inbox.find((x) => x.type === 'search-error' && x.id === 5));
    expect(m.type).toBe('search-error');
  });
});
