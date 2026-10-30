import { beforeAll, describe, expect, it } from 'vitest';
import { Mcts } from '../src/engine/mcts';
import { OrtEvaluator } from '../src/engine/ortEvaluator';
import { loadFixtures } from './helpers/fixtures';
import { createNodeSession, hasModel, writeOut } from './helpers/ortNode';

const fx = loadFixtures();

interface Row {
  name: string;
  fen: string;
  simulations: number;
  python: string;
  ts: string;
  agree: boolean;
  visitsIdentical: boolean;
  /** sum |ts - python| over root moves, as a share of the simulation count */
  visitDiffShare: number;
  /** Python's two most visited moves and their counts */
  pythonTop: [string, number][];
  /** the same two moves' counts in the TS search */
  tsCountsForPythonTop: number[];
  /** top-1 minus top-2 visits in Python, as a share of the simulations */
  pythonGapShare: number;
}

describe.skipIf(!hasModel)('TS MCTS vs Python MCTS (noise off, temperature 0, batch 8)', () => {
  let mcts: Mcts;
  beforeAll(async () => {
    const { ort, session } = await createNodeSession();
    mcts = new Mcts(new OrtEvaluator(ort, session));
  });

  for (const run of fx.mcts_runs) {
    it(`picks the same move as Python at ${run.simulations} simulations`, async () => {
      const rows: Row[] = [];
      for (const r of run.results) {
        const pos = fx.positions[r.index]!;
        const res = await mcts.search(r.fen, {
          maxSimulations: run.simulations,
          batchSize: run.batch_size,
          cPuct: run.c_puct,
          dirichlet: false,
          temperature: 0,
        });
        const tsVisits = new Map(res.rootMoves.map((m) => [m.uci, m.visits]));
        let diff = 0;
        let identical = true;
        for (const [uci, v] of Object.entries(r.visits)) {
          const d = Math.abs((tsVisits.get(uci) ?? 0) - v);
          diff += d;
          if (d) identical = false;
        }
        const top = Object.entries(r.visits).sort((a, b) => b[1] - a[1]);
        rows.push({
          name: pos.name,
          fen: r.fen,
          simulations: run.simulations,
          python: r.move,
          ts: res.move!,
          agree: res.move === r.move,
          visitsIdentical: identical,
          visitDiffShare: diff / run.simulations,
          pythonTop: top.slice(0, 2) as [string, number][],
          tsCountsForPythonTop: top.slice(0, 2).map(([u]) => tsVisits.get(u) ?? 0),
          pythonGapShare: ((top[0]?.[1] ?? 0) - (top[1]?.[1] ?? 0)) / run.simulations,
        });
      }
      const agree = rows.filter((x) => x.agree).length;
      const identical = rows.filter((x) => x.visitsIdentical).length;
      const summary = {
        simulations: run.simulations,
        positions: rows.length,
        sameMove: agree,
        agreement: agree / rows.length,
        identicalVisitCounts: identical,
        disagreements: rows.filter((x) => !x.agree),
        rows,
      };
      writeOut(`mcts_parity_${run.simulations}.json`, summary);
      console.log(
        `[parity] ${run.simulations} sims: ${agree}/${rows.length} same move (${((100 * agree) / rows.length).toFixed(1)}%), ` +
          `${identical}/${rows.length} identical root visit counts`,
      );
      for (const d of summary.disagreements) {
        console.log(
          `[parity]   differs: ${d.name} | Python ${d.python} ${JSON.stringify(d.pythonTop)} vs TS ${d.ts} (TS counts for Python's top two: ${d.tsCountsForPythonTop})`,
        );
      }
      expect(summary.agreement).toBeGreaterThanOrEqual(0.8);
    }, 600_000);
  }
});
