// The move list as numbered rows: "1. e4 e5", "2. Nf3 Nc6". A game that starts with Black to move (or from a
// position with a later move number) starts its first row at the right number, with a gap for the missing White move.

import type { Move } from 'chess.js';

export interface MoveRow {
  number: number;
  /** SAN, or null when the row starts with Black's move */
  white: string | null;
  black: string | null;
  /** index into the history of each ply shown in the row, -1 when absent */
  whiteIndex: number;
  blackIndex: number;
}

export function moveRows(history: Pick<Move, 'san' | 'color' | 'before'>[]): MoveRow[] {
  const rows: MoveRow[] = [];
  history.forEach((m, i) => {
    const number = Number(m.before.split(' ')[5]) || Math.floor(i / 2) + 1;
    let row = rows[rows.length - 1];
    if (m.color === 'w' || !row || row.number !== number) {
      row = { number, white: null, black: null, whiteIndex: -1, blackIndex: -1 };
      rows.push(row);
    }
    if (m.color === 'w') {
      row.white = m.san;
      row.whiteIndex = i;
    } else {
      row.black = m.san;
      row.blackIndex = i;
    }
  });
  return rows;
}
