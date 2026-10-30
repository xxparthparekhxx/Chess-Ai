// Behaviour of the search itself, with stand-in networks (no model file needed).
import { describe, expect, it } from 'vitest';
import { Mcts, sampleDirichlet } from '../src/engine/mcts';
import { ACTION_SIZE, type EvalOutput, type Evaluator } from '../src/engine/types';

/** uniform policy, constant value (White's point of view) */
function constant(value: number, delayMs = 0): Evaluator & { calls: number[] } {
  const calls: number[] = [];
  return {
    calls,
    async evaluate(_planes: Float32Array, n: number): Promise<EvalOutput> {
      calls.push(n);
      if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
      return { policy: new Float32Array(n * ACTION_SIZE), value: new Float32Array(n).fill(value) };
    },
  };
}

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const BACK_RANK_MATE = '6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1';

describe('search', () => {
  it('finds a mate in one with a blind network (terminal values and signs)', async () => {
    const r = await new Mcts(constant(0)).search(BACK_RANK_MATE, { maxSimulations: 100, dirichlet: false });
    expect(r.move).toBe('a1a8');
    expect(r.value).toBeCloseTo(1, 1);
  });

  it('finds the mate for Black as well', async () => {
    const r = await new Mcts(constant(0)).search('r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1', { maxSimulations: 100, dirichlet: false });
    expect(r.move).toBe('a8a1');
    expect(r.value).toBeCloseTo(1, 1);
  });

  it('reports value from the side to move: a White-favouring net is good for White, bad for Black', async () => {
    const white = await new Mcts(constant(0.5)).search(START, { maxSimulations: 1, batchSize: 1, dirichlet: false });
    expect(white.value).toBeCloseTo(0.5, 6);
    const black = await new Mcts(constant(0.5)).search(
      'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
      { maxSimulations: 1, batchSize: 1, dirichlet: false },
    );
    expect(black.value).toBeCloseTo(-0.5, 6);
  });

  it('with virtual loss and equal priors, the first batch visits the first 8 moves in python-chess order', async () => {
    const r = await new Mcts(constant(0)).search(START, { maxSimulations: 8, batchSize: 8, dirichlet: false });
    // python-chess lists the highest from-square first: g1h3, g1f3, b1c3, b1a3, h2h3, g2g3, ...
    expect(r.rootMoves.map((m) => m.uci).slice(0, 4)).toEqual(['g1h3', 'g1f3', 'b1c3', 'b1a3']);
    expect(r.rootMoves.slice(0, 8).every((m) => m.visits === 1)).toBe(true);
    expect(r.rootMoves.slice(8).every((m) => m.visits === 0)).toBe(true);
  });

  it('counts every simulation once, even when the only line is forced', async () => {
    // Black has a single legal move (king in check, one escape), so a batch keeps selecting the same leaf.
    const fen = 'k7/8/1K6/8/8/8/8/R7 b - - 0 1';
    const r = await new Mcts(constant(0)).search(fen, { maxSimulations: 40, batchSize: 8, dirichlet: false });
    expect(r.rootMoves.length).toBe(1);
    expect(r.rootMoves[0]!.visits).toBe(40);
  });

  it('sends each distinct leaf to the network once per batch and skips terminal leaves', async () => {
    const ev = constant(0);
    await new Mcts(ev).search(BACK_RANK_MATE, { maxSimulations: 64, batchSize: 8, dirichlet: false });
    expect(ev.calls[0]).toBe(1); // root alone
    expect(Math.max(...ev.calls)).toBeLessThanOrEqual(8);
  });

  it('returns no move when the game is over', async () => {
    const mated = 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3';
    const r = await new Mcts(constant(0)).search(mated, { maxSimulations: 8, dirichlet: false });
    expect(r.move).toBeNull();
  });

  it('stops on the time budget and reports progress', async () => {
    const ev = constant(0, 5);
    const progress: number[] = [];
    const t0 = performance.now();
    const r = await new Mcts(ev).search(START, {
      timeBudgetMs: 120,
      dirichlet: false,
      progressIntervalMs: 0,
      onProgress: (p) => progress.push(p.simulations),
    });
    const took = performance.now() - t0;
    expect(r.simulations).toBeGreaterThan(8);
    expect(r.simulations % 8).toBe(0);
    expect(took).toBeGreaterThanOrEqual(120);
    expect(took).toBeLessThan(400);
    expect(progress.length).toBeGreaterThan(1);
    expect(progress).toEqual([...progress].sort((a, b) => a - b));
    expect(r.simsPerSecond).toBeGreaterThan(0);
  });

  it('can be stopped early and still returns a move', async () => {
    const signal = { aborted: false };
    const r = await new Mcts(constant(0, 5)).search(START, {
      timeBudgetMs: 10_000,
      dirichlet: false,
      signal,
      progressIntervalMs: 0,
      onProgress: (p) => {
        if (p.simulations >= 16) signal.aborted = true;
      },
    });
    expect(r.aborted).toBe(true);
    expect(r.simulations).toBeLessThan(100);
    expect(r.move).not.toBeNull();
  });

  it('is deterministic without noise and at temperature 0', async () => {
    const a = await new Mcts(constant(0.1)).search(START, { maxSimulations: 80, dirichlet: false });
    const b = await new Mcts(constant(0.1)).search(START, { maxSimulations: 80, dirichlet: false });
    expect(a.rootMoves).toEqual(b.rootMoves);
  });

  it('samples by temperature with a seeded generator', async () => {
    let s = 7;
    const rng = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    const picks = new Set<string>();
    for (let i = 0; i < 30; i++) {
      const r = await new Mcts(constant(0)).search(START, { maxSimulations: 40, dirichlet: false, temperature: 1, rng });
      picks.add(r.move!);
    }
    expect(picks.size).toBeGreaterThan(3);
  });

  it('adds Dirichlet noise to the root priors when asked', async () => {
    const quiet = await new Mcts(constant(0)).search(START, { maxSimulations: 8, dirichlet: false });
    const noisy = await new Mcts(constant(0)).search(START, { maxSimulations: 8 });
    const q = quiet.rootMoves.map((m) => m.prior);
    const n = noisy.rootMoves.map((m) => m.prior);
    expect(q.every((p) => Math.abs(p - 1 / 20) < 1e-9)).toBe(true);
    expect(n.some((p, i) => Math.abs(p - q[i]!) > 1e-3)).toBe(true);
    expect(n.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
});

describe('Dirichlet sampler', () => {
  it('returns a probability vector with the right mean', () => {
    let s = 99;
    const rng = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    const k = 5;
    const mean = new Array(k).fill(0);
    const trials = 4000;
    for (let t = 0; t < trials; t++) {
      const d = sampleDirichlet(0.3, k, rng);
      expect(d.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
      d.forEach((x, i) => (mean[i] += x / trials));
    }
    for (const m of mean) expect(Math.abs(m - 1 / k)).toBeLessThan(0.02);
  });
});
