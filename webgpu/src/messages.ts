// Messages between the page and the engine worker.

import type { SearchProgress, SearchResult } from './engine/types';

export type BackendPreference = 'auto' | 'webgpu' | 'webgl' | 'wasm';

export interface InitRequest {
  type: 'init';
  modelUrl: string;
  modelVersion: string;
  backend: BackendPreference;
  ortExternalBase: string;
  /** CPU threads to use (needs cross-origin isolation); 0 or missing = the rule in cpuThreads() */
  threads?: number;
}

export interface SearchRequest {
  type: 'search';
  id: number;
  fen: string;
  timeBudgetMs: number;
  /** 0 = use the batch size picked at start-up */
  batchSize: number;
  temperature: number;
  noise: boolean;
}

/** Runs the network once on already-encoded positions. Used by the browser tests; the page never sends it. */
export interface EvaluateRequest {
  type: 'evaluate';
  id: number;
  planes: Float32Array;
  n: number;
}

export type ToWorker = InitRequest | SearchRequest | EvaluateRequest | { type: 'abort'; id: number } | { type: 'clear-cache' };

export type LoadPhase = 'download' | 'cache' | 'runtime' | 'compile' | 'warmup';

export interface CalibrationRow {
  batch: number;
  ms: number;
  positionsPerSecond: number;
}

export type BackendName = 'webgpu' | 'webgl' | 'wasm';

/** Why WebGPU is not in use, for the page to turn into advice. null = it is in use (or was not asked for). */
export type WebGpuProblem = 'no-api' | 'no-adapter' | 'adapter-error' | 'no-jspi' | 'forced-off' | 'runtime-failed' | 'slow-gpu';

export interface ReadyInfo {
  backend: BackendName;
  /** which onnxruntime-web build is running */
  runtime: 'jspi' | 'asyncify' | 'webgl' | 'wasm';
  threads: number;
  crossOriginIsolated: boolean;
  /** why the WebGPU backend is not in use, when it is not (a sentence) */
  webgpuNote: string | null;
  webgpuProblem: WebGpuProblem | null;
  /** why the WebGL backend is not in use, when it was tried and failed */
  webglNote: string | null;
  /** the GPU the browser reported for the backend in use (or, on the CPU backend, for WebGL / WebGPU), when it said */
  gpu: string | null;
  /** batch sizes the network was run at (WebGL / WebGPU pad to these), or null for any size */
  buckets: number[] | null;
  modelBytes: number;
  fromCache: boolean;
  loadMs: number;
  calibration: CalibrationRow[];
  /** batch size used when the user leaves it on Auto */
  autoBatchSize: number;
}

export type FromWorker =
  | { type: 'progress'; phase: LoadPhase; loaded?: number; total?: number; message: string; backend?: BackendName }
  | { type: 'ready'; info: ReadyInfo }
  | { type: 'init-error'; message: string }
  | { type: 'search-progress'; id: number; progress: SearchProgress }
  | { type: 'search-result'; id: number; result: SearchResult }
  | { type: 'search-error'; id: number; message: string }
  | { type: 'evaluate-result'; id: number; policy: Float32Array; value: Float32Array; ms: number }
  | { type: 'evaluate-error'; id: number; message: string }
  | { type: 'cache-cleared'; ok: boolean };
