// The download / Cache API logic, run against an in-memory stand-in for caches and fetch.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { clearModelCache, loadModel } from '../src/worker/modelLoader';

class FakeCache {
  entries = new Map<string, { body: Uint8Array; headers: Headers }>();
  failPut = false;
  async match(url: string) {
    const e = this.entries.get(url);
    return e ? new Response(e.body.slice(), { headers: e.headers }) : undefined;
  }
  async put(url: string, res: Response) {
    if (this.failPut) throw new DOMException('quota', 'QuotaExceededError');
    this.entries.set(url, { body: new Uint8Array(await res.arrayBuffer()), headers: res.headers });
  }
  async delete(url: string) {
    return this.entries.delete(url);
  }
}

class FakeCacheStorage {
  caches = new Map<string, FakeCache>();
  async open(name: string) {
    if (!this.caches.has(name)) this.caches.set(name, new FakeCache());
    return this.caches.get(name)!;
  }
  async keys() {
    return [...this.caches.keys()];
  }
  async delete(name: string) {
    return this.caches.delete(name);
  }
}

function streamOf(bytes: Uint8Array, chunk: number): ReadableStream<Uint8Array> {
  let off = 0;
  return new ReadableStream({
    pull(controller) {
      if (off >= bytes.length) return controller.close();
      controller.enqueue(bytes.slice(off, off + chunk));
      off += chunk;
    },
  });
}

const MODEL = new Uint8Array(3_500_000).map((_, i) => (i * 7) & 255);
const same = (a: Uint8Array, b: Uint8Array) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).equals(Buffer.from(b.buffer, b.byteOffset, b.byteLength));
const URL_A = 'https://example.test/chess_net.onnx';

let storage: FakeCacheStorage;
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  storage = new FakeCacheStorage();
  vi.stubGlobal('caches', storage);
  fetchMock = vi.fn(async () => new Response(streamOf(MODEL, 400_000), { status: 200, headers: { 'content-length': String(MODEL.length) } }));
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => vi.unstubAllGlobals());

describe('loadModel', () => {
  it('downloads with increasing progress, then serves the next load from the cache without fetching', async () => {
    const seen: number[] = [];
    const first = await loadModel(URL_A, '1', (loaded, total) => {
      expect(total).toBe(MODEL.length);
      seen.push(loaded);
    });
    expect(first.fromCache).toBe(false);
    expect(first.bytes.length).toBe(MODEL.length);
    expect(first.bytes[12345]).toBe(MODEL[12345]);
    expect(seen.length).toBeGreaterThan(3);
    expect(seen).toEqual([...seen].sort((a, b) => a - b));
    expect(seen.at(-1)).toBe(MODEL.length);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const second = await loadModel(URL_A, '1', () => {});
    expect(second.fromCache).toBe(true);
    expect(same(second.bytes, first.bytes)).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('downloads again, and removes the old cache, when the version changes', async () => {
    await loadModel(URL_A, '1', () => {});
    const again = await loadModel(URL_A, '2', () => {});
    expect(again.fromCache).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(await storage.keys()).toEqual(['sigma-go-model-2']);
  });

  it('works when the server does not send a content length', async () => {
    fetchMock.mockImplementation(async () => new Response(streamOf(MODEL, 333_333), { status: 200 }));
    const totals: (number | undefined)[] = [];
    const r = await loadModel(URL_A, '1', (_l, total) => totals.push(total));
    expect(r.bytes.length).toBe(MODEL.length);
    expect(same(r.bytes, MODEL)).toBe(true);
    expect(totals.every((t) => t === undefined)).toBe(true);
  });

  it('works when the content length is wrong (for example a compressed transfer)', async () => {
    fetchMock.mockImplementation(async () => new Response(streamOf(MODEL, 500_000), { status: 200, headers: { 'content-length': '1200000' } }));
    const r = await loadModel(URL_A, '1', () => {});
    expect(same(r.bytes, MODEL)).toBe(true);
  });

  it('still returns the model when the cache refuses to store it', async () => {
    const cache = await storage.open('sigma-go-model-1');
    cache.failPut = true;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const r = await loadModel(URL_A, '1', () => {});
    expect(r.bytes.length).toBe(MODEL.length);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('works when the Cache API is not available', async () => {
    vi.stubGlobal('caches', undefined);
    const r = await loadModel(URL_A, '1', () => {});
    expect(r.fromCache).toBe(false);
    expect(r.bytes.length).toBe(MODEL.length);
  });

  it('reports an HTTP error', async () => {
    fetchMock.mockImplementation(async () => new Response('nope', { status: 404 }));
    await expect(loadModel(URL_A, '1', () => {})).rejects.toThrow(/HTTP 404/);
  });

  it('rejects a response that is far too small to be the network', async () => {
    fetchMock.mockImplementation(async () => new Response(streamOf(new Uint8Array(5000), 5000), { status: 200, headers: { 'content-length': '5000' } }));
    await expect(loadModel(URL_A, '1', () => {})).rejects.toThrow(/only 5000 bytes/);
    expect((await storage.open('sigma-go-model-1')).entries.size).toBe(0);
  });

  it('clearModelCache removes the cached network', async () => {
    await loadModel(URL_A, '1', () => {});
    expect(await clearModelCache()).toBe(true);
    expect(await storage.keys()).toEqual([]);
    expect(await clearModelCache()).toBe(false);
  });
});
