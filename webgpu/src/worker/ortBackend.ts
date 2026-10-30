// Picks an onnxruntime-web build and creates the inference session.
//
// Order of preference:
//   1. WebGPU through onnxruntime-web's JSPI build (17 MiB .wasm, bundled with the site).
//      Needs WebAssembly JSPI (Chrome / Edge 137+).
//   2. WebGPU through the asyncify build. Its .wasm is 28 MiB, over Cloudflare's 25 MiB
//      per-file limit, so it is loaded from config.ortExternalBase (default: jsDelivr).
//   3. WebGL through onnxruntime-web's WebGL build (a 0.5 MiB script, no .wasm). It cannot take a symbolic
//      batch axis, so one copy of the network per batch size is created with the batch fixed (fixBatch.ts),
//      and it needs packed textures (env.webgl.pack) to give correct results. A software renderer
//      (SwiftShader, llvmpipe) is skipped: it is slower than the CPU path. ?backend=webgl forces it.
//   4. CPU through the plain WebAssembly build (14 MiB, bundled). Multi-threaded only when
//      the page is cross-origin isolated (COOP + COEP headers); see threads.ts.
// A failure in 1, 2 or 3 falls through to the next option and is reported in the UI.

import type { InferenceSession } from 'onnxruntime-web';
import { ORT_BASE } from '../config';
import type { OrtModuleLike, OrtSessionLike } from '../engine/ortEvaluator';
import type { BackendName, BackendPreference, ReadyInfo, WebGpuProblem } from '../messages';
import { withFixedBatch } from './fixBatch';
import { cpuThreads } from './threads';

type Runtime = ReadyInfo['runtime'];
type Session = OrtSessionLike & InferenceSession;

/**
 * Batch sizes the WebGL backend gets a session for (one copy of the network each). Only 1: onnxruntime-web 1.31.1's
 * WebGL Conv returns wrong numbers for any batch above 1 (checked on SwiftShader and on a hardware GPU; the first
 * Conv of the network already differs), so a batch is evaluated one position at a time.
 */
export const WEBGL_BATCHES = [1];

export interface CreatedSession {
  ort: OrtModuleLike;
  /** the session to run, for runtimes that take any batch size */
  session: Session;
  /** one session per fixed batch size (WebGL); empty otherwise */
  sessions: Map<number, Session>;
  backend: BackendName;
  runtime: Runtime;
  threads: number;
  webgpuNote: string | null;
  webgpuProblem: WebGpuProblem | null;
  webglNote: string | null;
  /** name of the GPU the running GPU backend (or, on the CPU backend, WebGL) reported, when it said */
  gpu: string | null;
}

interface GpuNavigator {
  gpu?: { requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string; description?: string } } | null> };
}

type WebGpuProbe = { problem: WebGpuProblem; note: string } | { problem: null; gpu: string | null };

async function probeWebGpu(): Promise<WebGpuProbe> {
  const gpu = (navigator as unknown as GpuNavigator).gpu;
  if (!gpu) return { problem: 'no-api', note: 'This browser does not expose WebGPU (navigator.gpu is missing).' };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { problem: 'no-adapter', note: 'WebGPU is present but no GPU adapter is available (blocklisted GPU or driver).' };
    const i = adapter.info;
    const name = [i?.vendor, i?.architecture, i?.description].filter(Boolean).join(' ');
    return { problem: null, gpu: name || null };
  } catch (err) {
    return { problem: 'adapter-error', note: `WebGPU adapter request failed: ${(err as Error).message}` };
  }
}

function hasJspi(): boolean {
  const wa = WebAssembly as unknown as Record<string, unknown>;
  return typeof wa.Suspending === 'function' && typeof wa.promising === 'function';
}

/** note = null when a hardware WebGL 2 context with float render targets exists, otherwise the reason it cannot be used. */
export function probeWebGl(force: boolean): { note: string | null; renderer: string | null } {
  let gl: WebGL2RenderingContext | null = null;
  try {
    const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
    gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
  } catch {
    gl = null;
  }
  if (!gl) return { note: 'WebGL 2 is not available in this browser (or is turned off).', renderer: null };
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = debug ? String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)) : null;
  if (!gl.getExtension('EXT_color_buffer_float')) {
    return { note: 'This WebGL context cannot render to 32-bit float textures, which the network needs.', renderer };
  }
  if (!force && renderer && /swiftshader|llvmpipe|softpipe|software|basic render/i.test(renderer)) {
    return { note: `WebGL is drawn by the CPU here (${renderer}), which is slower than the CPU backend.`, renderer };
  }
  return { note: null, renderer };
}

export async function createSession(
  bytes: Uint8Array,
  preference: BackendPreference,
  externalBase: string,
  onStep: (message: string) => void,
  threadsOverride = 0,
): Promise<CreatedSession> {
  const notes: string[] = [];
  const attempts: Runtime[] = [];
  let webgpuProblem: WebGpuProblem | null = null;
  let webglNote: string | null = null;
  let webgpuName: string | null = null;
  let webglName: string | null = null;

  if (preference === 'auto' || preference === 'webgpu') {
    const probe = await probeWebGpu();
    webgpuName = probe.problem ? null : probe.gpu;
    if (probe.problem) {
      notes.push(probe.note);
      webgpuProblem = probe.problem;
    } else if (hasJspi()) attempts.push('jspi');
    else if (externalBase) attempts.push('asyncify');
    else {
      notes.push('WebGPU is available, but this browser has no WebAssembly JSPI and the external WebGPU runtime is disabled.');
      webgpuProblem = 'no-jspi';
    }
  } else {
    notes.push(`${preference === 'wasm' ? 'CPU' : 'WebGL'} backend selected with ?backend=${preference}.`);
    webgpuProblem = 'forced-off';
  }

  if (preference !== 'wasm') {
    const gl = probeWebGl(preference === 'webgl');
    webglNote = gl.note;
    webglName = gl.renderer;
    if (!webglNote) attempts.push('webgl');
  }
  attempts.push('wasm');

  let lastError: unknown = null;
  for (const runtime of attempts) {
    try {
      onStep(
        runtime === 'wasm' ? 'Starting the CPU (WebAssembly) runtime' : runtime === 'webgl' ? 'Starting the WebGL runtime' : 'Starting the WebGPU runtime',
      );
      const created = await tryRuntime(runtime, bytes, externalBase, threadsOverride);
      return {
        ...created,
        webgpuNote: notes.length ? notes.join(' ') : null,
        webgpuProblem: created.backend === 'webgpu' ? null : webgpuProblem,
        webglNote,
        gpu: created.backend === 'webgpu' ? webgpuName : created.backend === 'webgl' ? webglName : webglName ?? webgpuName,
      };
    } catch (err) {
      lastError = err;
      console.warn(`onnxruntime-web (${runtime}) failed`, err);
      const text = `${runtime === 'wasm' ? 'CPU runtime' : runtime === 'webgl' ? 'WebGL runtime' : 'WebGPU runtime'} failed: ${(err as Error).message ?? err}`;
      if (runtime === 'webgl') webglNote = text;
      else {
        notes.push(text);
        if (runtime !== 'wasm') webgpuProblem = 'runtime-failed';
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function tryRuntime(runtime: Runtime, bytes: Uint8Array, externalBase: string, threadsOverride: number) {
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  // onnxruntime-web resolves file names against this with new URL(), so it must be absolute.
  const localBase = new URL(ORT_BASE, self.location.href).href;

  if (runtime === 'webgl') {
    const ort = (await import('onnxruntime-web/webgl')) as typeof import('onnxruntime-web');
    ort.env.webgl.pack = true; // unpacked textures give wrong results for this network
    const sessions = new Map<number, Session>();
    for (const batch of WEBGL_BATCHES) {
      const session = await ort.InferenceSession.create(withFixedBatch(bytes, batch), { executionProviders: ['webgl'] });
      sessions.set(batch, session as unknown as Session);
    }
    return {
      ort: ort as unknown as OrtModuleLike,
      session: sessions.get(WEBGL_BATCHES[0]!)!,
      sessions,
      backend: 'webgl' as const,
      runtime,
      threads: 1,
    };
  }

  let ort: typeof import('onnxruntime-web');
  let ep: string;
  let threads: number;

  if (runtime === 'jspi') {
    ort = (await import('onnxruntime-web/jspi')) as typeof ort;
    ort.env.wasm.wasmPaths = localBase;
    ep = 'webgpu';
    threads = 1;
  } else if (runtime === 'asyncify') {
    ort = (await import('onnxruntime-web/webgpu')) as typeof ort;
    ort.env.wasm.wasmPaths = externalBase;
    ep = 'webgpu';
    threads = 1;
  } else {
    ort = (await import('onnxruntime-web/wasm')) as typeof ort;
    ort.env.wasm.wasmPaths = localBase;
    ep = 'wasm';
    threads = threadsOverride > 0 && isolated ? Math.floor(threadsOverride) : cpuThreads(navigator.hardwareConcurrency, isolated);
  }
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = false; // already in a worker
  if (ep === 'webgpu') (ort.env as unknown as { webgpu: Record<string, unknown> }).webgpu.powerPreference = 'high-performance';

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
  });

  return {
    ort: ort as unknown as OrtModuleLike,
    session: session as unknown as Session,
    sessions: new Map<number, Session>(),
    backend: (ep === 'webgpu' ? 'webgpu' : 'wasm') as BackendName,
    runtime,
    threads,
  };
}
