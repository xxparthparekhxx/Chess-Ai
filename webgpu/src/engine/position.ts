// Rules layer. chess.js does the chess; this file adapts it to what the search needs.
//
// Two things here are not public chess.js API:
//  - chess.js 1.x builds a full SAN string, a "before" FEN and an "after" FEN for
//    every move it returns from moves({ verbose: true }) / move(). That is about
//    1 ms per call to moves(), far too slow for a search that expands thousands of
//    nodes. The search therefore uses the internal _moves(), _makeMove() and
//    _undoMove() (about 25 us / 0.5 us / 0.5 us in Node), the same functions the
//    public methods are built on.
//  - The package version is pinned exactly in package.json, and the parity tests
//    compare the legal move set and its order against python-chess on a few hundred
//    positions, so a chess.js update that changes behaviour fails the tests.
//
// The legal moves are returned in python-chess's generation order, because the
// Python search breaks ties by dict insertion order (= board.legal_moves order).
// With virtual loss this decides which moves the first batch visits, so the
// order is part of the behaviour being mirrored.

import { Chess } from 'chess.js';
import { moveToIndex, squareName } from './moveMap';

const BITS = {
  CAPTURE: 2,
  BIG_PAWN: 4,
  EP_CAPTURE: 8,
  PROMOTION: 16,
  KSIDE_CASTLE: 32,
  QSIDE_CASTLE: 64,
} as const;

/** chess.js internal move record (0x88 squares). */
export interface RawMove {
  color: 'w' | 'b';
  from: number;
  to: number;
  piece: 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
  captured?: string;
  promotion?: 'q' | 'r' | 'b' | 'n';
  flags: number;
}

interface PieceRec {
  type: 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
  color: 'w' | 'b';
}

interface ChessInternals {
  _board: (PieceRec | undefined)[];
  _halfMoves: number;
  _turn: 'w' | 'b';
  _moves(opts?: { legal?: boolean }): RawMove[];
  _makeMove(m: RawMove): void;
  _undoMove(): unknown;
}

export interface EngineMove {
  raw: RawMove;
  /** Index into the 4544-entry policy, -1 if the move is not in the table (never happens for legal chess moves). */
  action: number;
  uci: string;
  /** python-chess square numbers, a1 = 0 */
  from: number;
  to: number;
}

/** 0x88 square -> python-chess square number (a1 = 0 ... h8 = 63). */
export function toPySquare(sq0x88: number): number {
  return (7 - (sq0x88 >> 4)) * 8 + (sq0x88 & 7);
}

const PROMO_ORDER: Record<string, number> = { q: 0, r: 1, b: 2, n: 3 };

/**
 * Sort key reproducing python-chess's Board.generate_legal_moves() order.
 *
 * Not in check:
 *   1. non-pawn pieces (king included), from-square high to low, then to-square high to low
 *   2. castling, king side first
 *   3. pawn captures, from-square high to low, to-square high to low, promotions Q R B N
 *   4. single pawn pushes, to-square high to low, promotions Q R B N
 *   5. double pawn pushes, to-square high to low
 *   6. en passant, from-square high to low
 * In check: king moves come first (to-square high to low), then the same groups
 * 1, 3, 4, 5, 6 for the other pieces (castling is impossible in check).
 */
function orderKey(m: RawMove, from: number, to: number, inCheck: boolean): number {
  let phase: number;
  let k1 = 0;
  let k2 = 0;
  if (m.piece === 'p') {
    if (m.flags & BITS.EP_CAPTURE) {
      phase = 6;
      k1 = 63 - from;
    } else if (m.flags & BITS.CAPTURE) {
      phase = 3;
      k1 = 63 - from;
      k2 = 63 - to;
    } else if (m.flags & BITS.BIG_PAWN) {
      phase = 5;
      k1 = 63 - to;
    } else {
      phase = 4;
      k1 = 63 - to;
    }
  } else if (m.flags & (BITS.KSIDE_CASTLE | BITS.QSIDE_CASTLE)) {
    phase = 2;
    k1 = m.flags & BITS.KSIDE_CASTLE ? 0 : 1;
  } else if (m.piece === 'k' && inCheck) {
    phase = 0;
    k1 = 63 - to;
  } else {
    phase = 1;
    k1 = 63 - from;
    k2 = 63 - to;
  }
  const promo = m.promotion ? PROMO_ORDER[m.promotion]! : 0;
  return ((phase * 64 + k1) * 64 + k2) * 4 + promo;
}

export class Position {
  readonly chess: Chess;
  private readonly g: ChessInternals;

  constructor(fen: string) {
    this.chess = new Chess(fen);
    this.g = this.chess as unknown as ChessInternals;
    if (typeof this.g._moves !== 'function' || typeof this.g._makeMove !== 'function' || typeof this.g._undoMove !== 'function') {
      throw new Error('chess.js internals not found; the package version changed (pin chess.js in package.json)');
    }
  }

  get turn(): 'w' | 'b' {
    return this.g._turn;
  }

  get halfmoveClock(): number {
    return this.g._halfMoves;
  }

  isCheck(): boolean {
    return this.chess.isCheck();
  }

  /** Legal moves in python-chess order, with their policy indices. */
  legalMoves(): EngineMove[] {
    const raw = this.g._moves({ legal: true });
    const inCheck = raw.length > 0 && this.chess.isCheck();
    const out: { m: EngineMove; key: number }[] = new Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const r = raw[i]!;
      const from = toPySquare(r.from);
      const to = toPySquare(r.to);
      const uci = squareName(from) + squareName(to) + (r.promotion ?? '');
      out[i] = {
        m: { raw: r, action: moveToIndex(from, to, r.promotion), uci, from, to },
        key: orderKey(r, from, to, inCheck),
      };
    }
    out.sort((a, b) => a.key - b.key);
    return out.map((e) => e.m);
  }

  make(m: EngineMove): void {
    this.g._makeMove(m.raw);
  }

  undo(): void {
    this.g._undoMove();
  }

  /**
   * Result for the side to move if the game is over, mirroring MCTS.get_game_result()
   * on top of python-chess's Board.is_game_over() (claim_draw=False):
   *   -1 the side to move is checkmated, 0 any draw, null if the game goes on.
   * Checked in python-chess's order: checkmate, insufficient material, stalemate,
   * seventy-five-move rule (needs a legal move, so checkmate wins at 150).
   * Fivefold repetition is not tracked (see README, known limits).
   */
  terminalValue(legalCount: number): number | null {
    if (legalCount === 0) return this.chess.isCheck() ? -1 : 0; // checkmate, or stalemate
    if (this.isInsufficientMaterial()) return 0;
    if (this.halfmoveClock >= 150) return 0;
    return null;
  }

  /** python-chess Board.is_insufficient_material(): both sides lack winning material. */
  isInsufficientMaterial(): boolean {
    const board = this.g._board;
    const count = { w: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0, all: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0, all: 0 } };
    let bishopsOnParity0 = 0;
    let bishopsOnParity1 = 0;
    for (let rank = 0; rank < 8; rank++) {
      for (let file = 0; file < 8; file++) {
        const p = board[rank * 16 + file];
        if (!p) continue;
        count[p.color][p.type]++;
        count[p.color].all++;
        if (p.type === 'b') {
          if ((rank + file) & 1) bishopsOnParity1++;
          else bishopsOnParity0++;
        }
      }
    }
    const totalPawns = count.w.p + count.b.p;
    const totalKnights = count.w.n + count.b.n;
    const insufficient = (color: 'w' | 'b'): boolean => {
      const mine = count[color];
      const theirs = count[color === 'w' ? 'b' : 'w'];
      if (mine.p || mine.r || mine.q) return false;
      if (mine.n) {
        // one knight and the king, and the opponent has no pawns, knights, bishops or rooks
        return mine.all <= 2 && !(theirs.p || theirs.n || theirs.b || theirs.r);
      }
      if (mine.b) {
        const sameColour = bishopsOnParity0 === 0 || bishopsOnParity1 === 0;
        return sameColour && totalPawns === 0 && totalKnights === 0;
      }
      return true;
    };
    return insufficient('w') && insufficient('b');
  }

  /**
   * Fill 14 * 8 * 8 floats at out[offset...] with the same planes as board_to_tensor():
   * channels 0-5 White P N B R Q K, 6-11 Black, 12 all ones when White is to move,
   * 13 all ones when Black is to move. Plane layout is [channel][rank 1..8][file a..h].
   * out must already be zero in that range.
   */
  encodeInto(out: Float32Array, offset: number): void {
    const board = this.g._board;
    for (let rank = 0; rank < 8; rank++) {
      const rowBase = (7 - rank) * 16;
      for (let file = 0; file < 8; file++) {
        const p = board[rowBase + file];
        if (!p) continue;
        const channel = PIECE_CHANNEL[p.type] + (p.color === 'b' ? 6 : 0);
        out[offset + channel * 64 + rank * 8 + file] = 1;
      }
    }
    const turnPlane = this.g._turn === 'w' ? 12 : 13;
    out.fill(1, offset + turnPlane * 64, offset + turnPlane * 64 + 64);
  }
}

const PIECE_CHANNEL: Record<string, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

export const PLANE_FLOATS = 14 * 8 * 8;

/** Convenience for tests and one-off use. */
export function encodeFen(fen: string): Float32Array {
  const out = new Float32Array(PLANE_FLOATS);
  new Position(fen).encodeInto(out, 0);
  return out;
}
