// Page-side handle on the engine worker.

import type { SearchProgress, SearchResult } from './engine/types';
import type { BackendPreference, FromWorker, LoadPhase, ReadyInfo, ToWorker } from './messages';

export interface LoadProgress {
  phase: LoadPhase;
  loaded?: number;
  total?: number;
  message: string;
}

export interface SearchParams {
  fen: string;
  timeBudgetMs: number;
  batchSize: number;
  temperature: number;
  noise: boolean;
}

export interface SearchHandle {
  result: Promise<SearchResult>;
  /** stop early; the promise still resolves with the best move found so far */
  stop(): void;
}

export class EngineClient {
  private readonly worker: Worker;
  private nextId = 1;
  private readonly searches = new Map<
    number,
    { resolve(r: SearchResult): void; reject(e: Error): void; onProgress?: (p: SearchProgress) => void }
  >();
  private initResolve?: (info: ReadyInfo) => void;
  private initReject?: (e: Error) => void;
  private onLoad?: (p: LoadProgress) => void;
  private clearResolve?: (ok: boolean) => void;

  constructor() {
    this.worker = new Worker(new URL('./worker/engine.worker.ts', import.meta.url), { type: 'module', name: 'sigma-go-engine' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.handle(e.data);
    this.worker.onerror = (e) => {
      const err = new Error(e.message || 'The engine worker crashed');
      this.initReject?.(err);
      for (const s of this.searches.values()) s.reject(err);
      this.searches.clear();
    };
  }

  init(options: {
    modelUrl: string;
    modelVersion: string;
    backend: BackendPreference;
    ortExternalBase: string;
    /** CPU threads; 0 = automatic */
    threads?: number;
    onProgress: (p: LoadProgress) => void;
  }): Promise<ReadyInfo> {
    this.onLoad = options.onProgress;
    return new Promise((resolve, reject) => {
      this.initResolve = resolve;
      this.initReject = reject;
      this.send({
        type: 'init',
        modelUrl: options.modelUrl,
        modelVersion: options.modelVersion,
        backend: options.backend,
        ortExternalBase: options.ortExternalBase,
        threads: options.threads,
      });
    });
  }

  search(params: SearchParams, onProgress?: (p: SearchProgress) => void): SearchHandle {
    const id = this.nextId++;
    const result = new Promise<SearchResult>((resolve, reject) => {
      this.searches.set(id, { resolve, reject, onProgress });
    });
    this.send({ type: 'search', id, ...params });
    return { result, stop: () => this.send({ type: 'abort', id }) };
  }

  clearModelCache(): Promise<boolean> {
    return new Promise((resolve) => {
      this.clearResolve = resolve;
      this.send({ type: 'clear-cache' });
    });
  }

  private send(m: ToWorker): void {
    this.worker.postMessage(m);
  }

  private handle(m: FromWorker): void {
    switch (m.type) {
      case 'progress':
        this.onLoad?.(m);
        break;
      case 'ready':
        this.initResolve?.(m.info);
        break;
      case 'init-error':
        this.initReject?.(new Error(m.message));
        break;
      case 'search-progress':
        this.searches.get(m.id)?.onProgress?.(m.progress);
        break;
      case 'search-result': {
        const s = this.searches.get(m.id);
        this.searches.delete(m.id);
        s?.resolve(m.result);
        break;
      }
      case 'search-error': {
        const s = this.searches.get(m.id);
        this.searches.delete(m.id);
        s?.reject(new Error(m.message));
        break;
      }
      case 'cache-cleared':
        this.clearResolve?.(m.ok);
        break;
    }
  }
}
