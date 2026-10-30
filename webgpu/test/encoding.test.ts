import { describe, expect, it } from 'vitest';
import { encodeFen, Position, PLANE_FLOATS } from '../src/engine/position';
import { ACTION_SIZE, MOVE_TO_INT, UCI_LABELS } from '../src/engine/moveMap';
import { loadFixtures } from './helpers/fixtures';
import { writeOut } from './helpers/ortNode';

const fx = loadFixtures();

describe('move map', () => {
  it('has 4544 entries', () => {
    expect(UCI_LABELS.length).toBe(ACTION_SIZE);
    expect(fx.move_labels.length).toBe(ACTION_SIZE);
  });

  it('matches the Python table entry for entry', () => {
    expect([...UCI_LABELS]).toEqual(fx.move_labels);
    for (let i = 0; i < fx.move_labels.length; i++) expect(MOVE_TO_INT.get(fx.move_labels[i]!)).toBe(i);
  });
});

describe('input planes', () => {
  it(`match board_to_tensor exactly on ${fx.positions.length} positions`, () => {
    let compared = 0;
    for (const p of fx.positions) {
      const planes = encodeFen(p.fen);
      expect(planes.length).toBe(PLANE_FLOATS);
      for (let i = 0; i < PLANE_FLOATS; i++) {
        if (planes[i] !== p.planes[i]) throw new Error(`${p.name}: plane value ${i} is ${planes[i]}, Python has ${p.planes[i]}`);
        compared++;
      }
    }
    writeOut('encoding.json', { positions: fx.positions.length, valuesCompared: compared, mismatches: 0 });
  });
});

describe('legal moves', () => {
  it('give the same move indices, in the same order, as python-chess on the fixture positions', () => {
    for (const p of fx.positions) {
      const moves = new Position(p.fen).legalMoves();
      expect(moves.map((m) => m.uci), p.name).toEqual(p.legal_uci);
      expect(moves.map((m) => m.action), p.name).toEqual(p.legal_idx);
    }
  });

  it(`give the same moves in the same order on ${fx.order_cases.length} random-playout positions`, () => {
    let moves = 0;
    for (const c of fx.order_cases) {
      const got = new Position(c.fen).legalMoves().map((m) => m.uci);
      expect(got, c.fen).toEqual(c.legal_uci);
      moves += got.length;
    }
    writeOut('legal_moves.json', {
      fixturePositions: fx.positions.length,
      randomPositions: fx.order_cases.length,
      randomMovesCompared: moves,
      mismatches: 0,
    });
  });

  it("is not chess.js's own order (so the sort is doing real work)", () => {
    let differing = 0;
    for (const c of fx.order_cases) {
      const pos = new Position(c.fen);
      const raw = (pos.chess as unknown as { _moves(): { from: number; to: number; promotion?: string }[] })._moves();
      const rawUci = raw.map((m) => {
        const sq = (s: number) => String.fromCharCode(97 + (s & 7)) + String.fromCharCode(49 + (7 - (s >> 4)));
        return sq(m.from) + sq(m.to) + (m.promotion ?? '');
      });
      if (rawUci.join() !== c.legal_uci.join()) differing++;
      expect([...rawUci].sort()).toEqual([...c.legal_uci].sort()); // same set of moves
    }
    expect(differing).toBeGreaterThan(fx.order_cases.length / 2);
  });

  it('make and undo restore the position exactly', () => {
    for (const c of fx.order_cases.slice(0, 60)) {
      const pos = new Position(c.fen);
      const before = pos.chess.fen();
      for (const m of pos.legalMoves()) {
        pos.make(m);
        pos.undo();
        expect(pos.chess.fen()).toBe(before);
      }
    }
  });
});

describe('game result (MCTS.get_game_result)', () => {
  it(`agrees on ${fx.terminal_cases.length} positions, including draws by material and the 75-move rule`, () => {
    const kinds: Record<string, number> = {};
    for (const c of fx.terminal_cases) {
      const pos = new Position(c.fen);
      const n = pos.legalMoves().length;
      const got = pos.terminalValue(n);
      const label = `${c.label}: ${c.fen}`;
      expect(got !== null, label).toBe(c.terminal);
      if (c.terminal) expect(got, label).toBe(c.value);
      kinds[c.label] = (kinds[c.label] ?? 0) + 1;
    }
    writeOut('terminal.json', {
      cases: fx.terminal_cases.length,
      terminal: fx.terminal_cases.filter((c) => c.terminal).length,
      checkmates: fx.terminal_cases.filter((c) => c.terminal && c.value === -1).length,
      mismatches: 0,
    });
  });
});
