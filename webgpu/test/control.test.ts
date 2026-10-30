// Negative control: the parity above is only meaningful if the test can fail. This
// runs the same 200-simulation comparison with the legal moves left in chess.js's
// own order instead of python-chess's order. The result is logged and saved, and the
// test only checks that the sort matters (some visit counts change).
import { describe, expect, it } from 'vitest';
import { Mcts } from '../src/engine/mcts';
import { OrtEvaluator } from '../src/engine/ortEvaluator';
import { Position } from '../src/engine/position';
import { loadFixtures } from './helpers/fixtures';
import { createNodeSession, hasModel, writeOut } from './helpers/ortNode';

const fx = loadFixtures();

describe.skipIf(!hasModel)('control: search with chess.js native move order', () => {
  it('diverges from Python on some positions, so the python-chess order is needed', async () => {
    const { ort, session } = await createNodeSession();
    const mcts = new Mcts(new OrtEvaluator(ort, session));
    const original = Position.prototype.legalMoves;
    Position.prototype.legalMoves = function (this: Position) {
      const moves = original.call(this);
      // undo the sort: back to the order chess.js generated them in
      type R = { from: number; to: number; promotion?: string };
      const key = (r: R) => `${r.from}-${r.to}-${r.promotion ?? ''}`;
      const raw = (this.chess as unknown as { _moves(): R[] })._moves();
      const rank = new Map(raw.map((r, i) => [key(r), i]));
      return [...moves].sort((a, b) => rank.get(key(a.raw))! - rank.get(key(b.raw))!);
    };
    try {
      const run = fx.mcts_runs.find((r) => r.simulations === 200)!;
      let sameMove = 0;
      let identical = 0;
      for (const r of run.results) {
        const res = await mcts.search(r.fen, { maxSimulations: 200, batchSize: 8, dirichlet: false, temperature: 0 });
        if (res.move === r.move) sameMove++;
        const ts = new Map(res.rootMoves.map((m) => [m.uci, m.visits]));
        if (Object.entries(r.visits).every(([u, v]) => ts.get(u) === v)) identical++;
      }
      const out = { simulations: 200, positions: run.results.length, sameMove, identicalVisitCounts: identical };
      console.log('[control] chess.js native move order:', out);
      writeOut('control_native_order.json', out);
      expect(identical).toBeLessThan(run.results.length);
    } finally {
      Position.prototype.legalMoves = original;
    }
  }, 300_000);
});
