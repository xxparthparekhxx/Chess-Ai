// TypeScript port of src/mcts.py.
//
// Same algorithm and the same quirks:
//  - PUCT with c_puct = 1.0; Q is taken from the parent's side (-child.value()).
//  - Leaves are collected in batches (8 in Python) with virtual loss. Virtual loss
//    is applied to path[1:] only, so the root's visit count (and with it the U term
//    of the root's children) does not change inside a batch.
//  - Nodes store values from the point of view of the side to move at that node.
//    The network speaks White's point of view, so a leaf value is negated when
//    Black is to move. Back-propagation flips the sign at every step.
//  - Terminal leaves use the game result and are not sent to the network.
//  - Priors are the softmax over the legal moves only (Python: softmax over all
//    4544 logits, then divided by the sum over legal moves; the same numbers).
//  - Root Dirichlet noise 0.75 * prior + 0.25 * noise with alpha 0.3.
//  - select_best_move: argmax of visits, or sampling from visits^(1/T).
//
// Differences:
//  - The loop can stop on a time budget instead of a simulation count.
//  - A leaf selected twice in one batch is sent to the network once; Python sends
//    it twice and gets the same answer.
//  - Everything is double precision. Python's numpy scalars are float32 in the PUCT
//    arithmetic, so a score tie in float32 may not be a tie here.

import { Position, PLANE_FLOATS, type EngineMove } from './position';
import { ACTION_SIZE, type Evaluator, type RootMoveStat, type SearchOptions, type SearchProgress, type SearchResult } from './types';

class Node {
  visitCount = 0;
  valueSum = 0;
  prior: number;
  /** children in legal-move order (python-chess order), null until expanded */
  kids: Node[] | null = null;
  /** the move leading to each child */
  moves: EngineMove[] | null = null;
  /** cached game result for the side to move here: number if terminal, null if not, undefined if not computed */
  terminal: number | null | undefined = undefined;

  constructor(prior: number) {
    this.prior = prior;
  }

  get expanded(): boolean {
    return this.kids !== null && this.kids.length > 0;
  }

  value(): number {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }
}

interface PendingLeaf {
  node: Node;
  /** game result if terminal */
  terminal: number | null;
  legal: EngineMove[] | null;
  slot: number;
  turn: 'w' | 'b';
}

function gaussian(rng: () => number): number {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Gamma(shape, 1) by Marsaglia-Tsang, with the usual boost for shape < 1. */
function gamma(shape: number, rng: () => number): number {
  if (shape < 1) {
    let u = 0;
    while (u === 0) u = rng();
    return gamma(shape + 1, rng) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = gaussian(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

export function sampleDirichlet(alpha: number, k: number, rng: () => number): number[] {
  const g = new Array<number>(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    g[i] = gamma(alpha, rng);
    sum += g[i]!;
  }
  if (!(sum > 0)) return g.map(() => 1 / k);
  return g.map((x) => x / sum);
}

export class Mcts {
  constructor(private readonly evaluator: Evaluator) {}

  async search(fen: string, options: SearchOptions = {}): Promise<SearchResult> {
    const now = options.now ?? (() => performance.now());
    const batchSize = Math.max(1, Math.floor(options.batchSize ?? 8));
    const cPuct = options.cPuct ?? 1.0;
    const virtualLoss = options.virtualLoss ?? 1.0;
    const temperature = options.temperature ?? 0;
    const rng = options.rng ?? Math.random;
    const noise = options.dirichlet === undefined ? { alpha: 0.3, fraction: 0.25 } : options.dirichlet;
    const maxSims = options.maxSimulations ?? Infinity;
    const budget = options.timeBudgetMs ?? Infinity;
    if (maxSims === Infinity && budget === Infinity) throw new Error('search needs timeBudgetMs or maxSimulations');
    const progressInterval = options.progressIntervalMs ?? 100;

    const position = new Position(fen);
    const planes = new Float32Array(batchSize * PLANE_FLOATS);
    const start = now();
    let evalMs = 0;
    let evalCalls = 0;
    let evalPositions = 0;

    const evaluate = async (n: number) => {
      const t = now();
      const out = await this.evaluator.evaluate(planes.subarray(0, n * PLANE_FLOATS), n);
      evalMs += now() - t;
      evalCalls++;
      evalPositions += n;
      return out;
    };

    // --- root ---
    const root = new Node(0);
    const rootMoves = position.legalMoves();
    if (rootMoves.length === 0) {
      return this.finish(root, { temperature: 0, rng, aborted: false, sims: 0, elapsedMs: now() - start, evalMs, evalCalls, evalPositions });
    }
    planes.fill(0, 0, PLANE_FLOATS);
    position.encodeInto(planes, 0);
    const rootOut = await evaluate(1);
    this.expand(root, rootMoves, rootOut.policy, 0);
    if (noise) this.addNoise(root, noise.alpha, noise.fraction, rng);

    // --- simulations ---
    let sims = 0;
    let aborted = false;
    let lastProgress = start;
    const pending = new Map<Node, PendingLeaf>();

    while (sims < maxSims) {
      if (options.signal?.aborted) {
        aborted = true;
        break;
      }
      if (sims > 0 && now() - start >= budget) break;
      const n = Math.min(batchSize, maxSims - sims);

      // 1. select n leaves, applying virtual loss as we go
      const leaves: { leaf: PendingLeaf; path: Node[] }[] = [];
      pending.clear();
      let slots = 0;
      planes.fill(0);
      for (let b = 0; b < n; b++) {
        let node = root;
        const path: Node[] = [root];
        let depth = 0;
        while (node.expanded) {
          const i = this.selectChild(node, cPuct);
          position.make(node.moves![i]!);
          node = node.kids![i]!;
          path.push(node);
          depth++;
        }
        for (let d = 1; d < path.length; d++) {
          path[d]!.visitCount += 1;
          path[d]!.valueSum += virtualLoss;
        }
        let leaf = pending.get(node);
        if (!leaf) {
          let terminal = node.terminal;
          let legal: EngineMove[] | null = null;
          if (terminal === undefined) {
            legal = position.legalMoves();
            terminal = position.terminalValue(legal.length);
            node.terminal = terminal;
          }
          if (terminal === null) {
            legal ??= position.legalMoves();
            position.encodeInto(planes, slots * PLANE_FLOATS);
            leaf = { node, terminal: null, legal, slot: slots++, turn: position.turn };
          } else {
            leaf = { node, terminal, legal: null, slot: -1, turn: position.turn };
          }
          pending.set(node, leaf);
        }
        leaves.push({ leaf, path });
        for (let d = 0; d < depth; d++) position.undo();
      }

      // 2. one network call for the non-terminal leaves
      let out: { policy: Float32Array; value: Float32Array } | null = null;
      if (slots > 0) out = await evaluate(slots);

      // 3. expand, remove virtual loss, back-propagate
      for (const { leaf, path } of leaves) {
        let value: number;
        if (leaf.terminal !== null) {
          value = leaf.terminal;
        } else {
          const v = out!.value[leaf.slot]!;
          value = leaf.turn === 'w' ? v : -v;
          if (!leaf.node.expanded) this.expand(leaf.node, leaf.legal!, out!.policy, leaf.slot * ACTION_SIZE);
        }
        for (let d = 1; d < path.length; d++) {
          path[d]!.visitCount -= 1;
          path[d]!.valueSum -= virtualLoss;
        }
        let v = value;
        for (let d = path.length - 1; d >= 0; d--) {
          path[d]!.visitCount += 1;
          path[d]!.valueSum += v;
          v = -v;
        }
      }
      sims += n;

      const t = now();
      if (options.onProgress && t - lastProgress >= progressInterval) {
        lastProgress = t;
        options.onProgress(this.progress(root, sims, t - start));
      }
    }

    return this.finish(root, { temperature, rng, aborted, sims, elapsedMs: now() - start, evalMs, evalCalls, evalPositions });
  }

  private selectChild(node: Node, cPuct: number): number {
    const kids = node.kids!;
    const sqrtN = Math.sqrt(node.visitCount);
    let best = -Infinity;
    let bestIndex = 0;
    for (let i = 0; i < kids.length; i++) {
      const c = kids[i]!;
      const u = cPuct * c.prior * (sqrtN / (1 + c.visitCount));
      const q = c.visitCount === 0 ? 0 : -(c.valueSum / c.visitCount);
      const score = q + u;
      if (score > best) {
        best = score;
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  private expand(node: Node, legal: EngineMove[], logits: Float32Array, offset: number): void {
    const kids: Node[] = [];
    const moves: EngineMove[] = [];
    let max = -Infinity;
    for (const m of legal) {
      if (m.action < 0) continue;
      const l = logits[offset + m.action]!;
      if (l > max) max = l;
    }
    let sum = 0;
    const e: number[] = [];
    for (const m of legal) {
      if (m.action < 0) continue;
      const x = Math.exp(logits[offset + m.action]! - max);
      e.push(x);
      sum += x;
    }
    let j = 0;
    for (const m of legal) {
      if (m.action < 0) continue;
      kids.push(new Node(sum > 0 ? e[j++]! / sum : 0));
      moves.push(m);
    }
    node.kids = kids;
    node.moves = moves;
  }

  private addNoise(root: Node, alpha: number, fraction: number, rng: () => number): void {
    const kids = root.kids!;
    const noise = sampleDirichlet(alpha, kids.length, rng);
    for (let i = 0; i < kids.length; i++) {
      kids[i]!.prior = (1 - fraction) * kids[i]!.prior + fraction * noise[i]!;
    }
  }

  private principalVariation(root: Node, maxLen = 8): string[] {
    const pv: string[] = [];
    let node = root;
    while (node.expanded && pv.length < maxLen) {
      let bi = 0;
      let bv = -1;
      node.kids!.forEach((c, i) => {
        if (c.visitCount > bv) {
          bv = c.visitCount;
          bi = i;
        }
      });
      if (bv <= 0) break;
      pv.push(node.moves![bi]!.uci);
      node = node.kids![bi]!;
    }
    return pv;
  }

  private progress(root: Node, sims: number, elapsedMs: number): SearchProgress {
    let bestIndex = -1;
    let bestVisits = -1;
    root.kids?.forEach((c, i) => {
      if (c.visitCount > bestVisits) {
        bestVisits = c.visitCount;
        bestIndex = i;
      }
    });
    const best = bestIndex >= 0 ? root.kids![bestIndex]! : null;
    return {
      simulations: sims,
      elapsedMs,
      simsPerSecond: elapsedMs > 0 ? (sims * 1000) / elapsedMs : 0,
      bestMove: best ? root.moves![bestIndex]!.uci : null,
      value: best ? -best.value() : 0,
      pv: this.principalVariation(root),
    };
  }

  private finish(
    root: Node,
    r: {
      temperature: number;
      rng: () => number;
      aborted: boolean;
      sims: number;
      elapsedMs: number;
      evalMs: number;
      evalCalls: number;
      evalPositions: number;
    },
  ): SearchResult {
    const base = this.progress(root, r.sims, r.elapsedMs);
    const stats = { aborted: r.aborted, evalMs: r.evalMs, evalCalls: r.evalCalls, evalPositions: r.evalPositions };
    if (!root.kids || root.kids.length === 0) {
      return { ...base, move: null, rootMoves: [], ...stats };
    }
    const kids = root.kids;
    const visits = kids.map((c) => c.visitCount);
    let chosen = 0;
    if (r.temperature <= 0) {
      // select_best_move: greedy on visit counts, first of equals wins
      for (let i = 1; i < visits.length; i++) if (visits[i]! > visits[chosen]!) chosen = i;
      if (visits[chosen] === 0) {
        // stopped before any simulation finished: fall back to the highest prior
        for (let i = 1; i < kids.length; i++) if (kids[i]!.prior > kids[chosen]!.prior) chosen = i;
      }
    } else {
      const max = Math.max(...visits);
      if (max <= 0) {
        chosen = Math.min(visits.length - 1, Math.floor(r.rng() * visits.length));
      } else {
        // sample from (visits / max)^(1 / T)
        const w = visits.map((v) => Math.pow(v / max, 1 / r.temperature));
        const total = w.reduce((a, b) => a + b, 0);
        let x = r.rng() * total;
        chosen = w.length - 1;
        for (let i = 0; i < w.length; i++) {
          x -= w[i]!;
          if (x < 0) {
            chosen = i;
            break;
          }
        }
      }
    }
    const rootMoves: RootMoveStat[] = kids.map((c, i) => ({
      uci: root.moves![i]!.uci,
      visits: c.visitCount,
      prior: c.prior,
      q: -c.value(),
    }));
    return { ...base, move: root.moves![chosen]!.uci, value: -kids[chosen]!.value(), rootMoves, ...stats };
  }
}
