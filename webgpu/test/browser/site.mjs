// Shared helpers for the browser tests: serve the built site with `vite preview`, open it in headless
// Chrome, and talk to the engine worker straight from the built bundle (no site code is modified).
//
//   npm run build:local     (builds dist/ with the network taken from this site's own /model/)
//   node test/browser/parity.mjs | threads.mjs | smoke.mjs

import { spawn } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const WEB_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const DIST = join(WEB_ROOT, 'dist');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Starts `vite preview` and returns { url, stop() }. Stops it by PID, not by pattern. */
export async function startPreview(port = 4170 + Math.floor(Math.random() * 20)) {
  const proc = spawn(process.execPath, [join(WEB_ROOT, 'node_modules', 'vite', 'bin', 'vite.js'), 'preview', '--port', String(port), '--strictPort', '--host', '127.0.0.1'], {
    cwd: WEB_ROOT,
    stdio: 'ignore',
  });
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try {
      if ((await fetch(url + '/')).ok) return { url, stop: () => proc.kill() };
    } catch {
      /* not up yet */
    }
    await sleep(200);
  }
  proc.kill();
  throw new Error('vite preview did not start (did you run `npm run build:local`?)');
}

/** Path of the built engine worker, e.g. /assets/engine.worker-C6wk_L5b.js */
export function workerPath() {
  const f = readdirSync(join(DIST, 'assets')).find((n) => /^engine\.worker-.*\.js$/.test(n));
  if (!f) throw new Error('no engine worker in dist/assets; run `npm run build:local` first');
  return '/assets/' + f;
}

/** A tiny same-origin document with the site's COOP/COEP headers, so the page does not start the game itself. */
export function blankPage() {
  const f = readdirSync(join(DIST, 'assets')).find((n) => /^index-.*\.css$/.test(n));
  return '/assets/' + f;
}

// What to launch Chrome with for each test variant. `backend` is the engine backend it must end up on.
// Headless Chrome here can use real GPUs, and the variants say which one:
//   *-swiftshader  Chrome's CPU renderer (what a machine with no usable GPU gets); accuracy only, speed means nothing
//   webgl-igpu     the AMD Radeon 610M integrated GPU, through ANGLE's OpenGL backend
//   webgl-nvidia   the NVIDIA GPU, through ANGLE's OpenGL backend with PRIME render offload
//   webgpu-nvidia  the NVIDIA GPU, through Vulkan (--enable-unsafe-webgpu)
const SOFTWARE_GL = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
const NVIDIA_GL_ENV = {
  DRI_PRIME: '1',
  __NV_PRIME_RENDER_OFFLOAD: '1',
  __GLX_VENDOR_LIBRARY_NAME: 'nvidia',
  __EGL_VENDOR_LIBRARY_FILENAMES: '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
};
export const VARIANTS = {
  wasm: { backend: 'wasm', renderer: 'CPU', args: [], env: {} },
  'webgl-swiftshader': { backend: 'webgl', renderer: 'SwiftShader (software)', args: SOFTWARE_GL, env: {} },
  'webgpu-swiftshader': { backend: 'webgpu', renderer: 'SwiftShader (software)', args: ['--enable-unsafe-webgpu', ...SOFTWARE_GL], env: {} },
  'webgl-igpu': { backend: 'webgl', renderer: 'AMD Radeon 610M (hardware)', args: ['--use-angle=gl', '--ignore-gpu-blocklist'], env: {} },
  'webgl-nvidia': { backend: 'webgl', renderer: 'NVIDIA RTX 5060 Laptop (hardware)', args: ['--use-angle=gl', '--ignore-gpu-blocklist'], env: NVIDIA_GL_ENV },
  'webgpu-nvidia': {
    backend: 'webgpu',
    renderer: 'NVIDIA RTX 5060 Laptop (hardware)',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan', '--ignore-gpu-blocklist'],
    env: {},
  },
};

/**
 * Runs in the page. Starts the built worker with `init`, then offers evaluate() and search().
 * Defined as a string-able function so it can be sent through Runtime.evaluate.
 */
export function pageEngine(workerUrl) {
  return `(() => {
    const worker = new Worker(${JSON.stringify(workerUrl)}, { type: 'module' });
    let id = 0;
    const waiting = new Map();
    let onReady, onFail;
    const progress = [];
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') onReady(m.info);
      else if (m.type === 'init-error') onFail(new Error(m.message));
      else if (m.type === 'progress') progress.push(m.message);
      else if (m.id !== undefined && waiting.has(m.id)) {
        const w = waiting.get(m.id);
        if (m.type.endsWith('-error')) { waiting.delete(m.id); w.reject(new Error(m.message)); }
        else if (m.type === 'evaluate-result' || m.type === 'search-result') { waiting.delete(m.id); w.resolve(m); }
      }
    };
    worker.onerror = (e) => onFail && onFail(new Error('worker error: ' + e.message));
    const call = (msg) => new Promise((resolve, reject) => { const i = ++id; waiting.set(i, { resolve, reject }); worker.postMessage({ ...msg, id: i }); });
    return {
      progress,
      init: (o) => new Promise((resolve, reject) => { onReady = resolve; onFail = reject; worker.postMessage({ type: 'init', modelUrl: '/model/chess_net.onnx', modelVersion: 'browser-test', ortExternalBase: '', ...o }); }),
      evaluate: (planes, n) => call({ type: 'evaluate', planes, n }),
      search: (o) => call({ type: 'search', batchSize: 0, temperature: 0, noise: false, ...o }),
      stop: () => worker.terminate(),
    };
  })()`;
}
