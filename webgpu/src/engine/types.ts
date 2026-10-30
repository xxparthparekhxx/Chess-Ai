import { ACTION_SIZE } from './moveMap';

/** Output of one network call for n positions. Views may be reused by the evaluator, so consume them before the next call. */
export interface EvalOutput {
  /** n * 4544 policy logits */
  policy: Float32Array;
  /** n values in [-1, 1] from White's point of view (the net was trained on the game result, 1 = White wins) */
  value: Float32Array;
}

export interface Evaluator {
  /** planes: n * 14 * 8 * 8 floats (see Position.encodeInto). */
  evaluate(planes: Float32Array, n: number): Promise<EvalOutput>;
}

export interface RootMoveStat {
  uci: string;
  visits: number;
  /** prior used by the search after the root noise (if any) */
  prior: number;
  /** value of the move for the side to move at the root, in [-1, 1] */
  q: number;
}

export interface SearchProgress {
  simulations: number;
  elapsedMs: number;
  simsPerSecond: number;
  /** most visited root move so far */
  bestMove: string | null;
  /** value of that move for the side to move at the root */
  value: number;
  /** most visited line from the root, UCI */
  pv: string[];
}

export interface SearchResult extends SearchProgress {
  move: string | null;
  rootMoves: RootMoveStat[];
  aborted: boolean;
  /** time spent waiting for the network, ms */
  evalMs: number;
  /** number of network calls (the root call included) */
  evalCalls: number;
  /** positions sent to the network, padding excluded */
  evalPositions: number;
}

export interface SearchOptions {
  /** stop starting new batches once this much wall time has passed */
  timeBudgetMs?: number;
  /** stop after this many simulations (the root evaluation is not counted, as in Python) */
  maxSimulations?: number;
  /** leaves per network call; Python uses 8 */
  batchSize?: number;
  cPuct?: number;
  virtualLoss?: number;
  /** root Dirichlet noise; false switches it off. Python: alpha 0.3, fraction 0.25 */
  dirichlet?: { alpha: number; fraction: number } | false;
  /** 0 = most visited, otherwise sample from visits^(1/T) */
  temperature?: number;
  /** uniform [0, 1) source, defaults to Math.random */
  rng?: () => number;
  /** set .aborted = true to stop after the current batch and return the best move so far */
  signal?: { aborted: boolean };
  onProgress?: (p: SearchProgress) => void;
  /** minimum time between onProgress calls */
  progressIntervalMs?: number;
  now?: () => number;
}

export { ACTION_SIZE };
