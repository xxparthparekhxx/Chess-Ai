// Messages between the page and the engine worker.

import type { SearchProgress, SearchResult } from './engine/types';

export type BackendPreference = 'auto' | 'webgpu' | 'wasm';

export interface InitRequest {
  type: 'init';
  modelUrl: string;
  modelVersion: string;
  backend: BackendPreference;
  ortExternalBase: string;
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

export type ToWorker = InitRequest | SearchRequest | { type: 'abort'; id: number } | { type: 'clear-cache' };

export type LoadPhase = 'download' | 'cache' | 'runtime' | 'compile' | 'warmup';

export interface CalibrationRow {
  batch: number;
  ms: number;
  positionsPerSecond: number;
}

export interface ReadyInfo {
  backend: 'webgpu' | 'wasm';
  /** which onnxruntime-web build is running */
  runtime: 'jspi' | 'asyncify' | 'wasm';
  threads: number;
  crossOriginIsolated: boolean;
  /** why the WebGPU backend is not in use, when it is not */
  webgpuNote: string | null;
  modelBytes: number;
  fromCache: boolean;
  loadMs: number;
  calibration: CalibrationRow[];
  /** batch size used when the user leaves it on Auto */
  autoBatchSize: number;
}

export type FromWorker =
  | { type: 'progress'; phase: LoadPhase; loaded?: number; total?: number; message: string }
  | { type: 'ready'; info: ReadyInfo }
  | { type: 'init-error'; message: string }
  | { type: 'search-progress'; id: number; progress: SearchProgress }
  | { type: 'search-result'; id: number; result: SearchResult }
  | { type: 'search-error'; id: number; message: string }
  | { type: 'cache-cleared'; ok: boolean };
