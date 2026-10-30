// Port of create_uci_labels() / MOVE_TO_INT from src/dataset.py.
//
// The label list is every "from to" pair of different squares (64 * 63 = 4032)
// plus the promotion moves 7th->8th rank for White and 2nd->1st rank for Black on
// any pair of files, with q/r/b/n (2 * 8 * 8 * 4 = 512), sorted as strings.
// Index = position in that sorted list. 4544 entries.

export const ACTION_SIZE = 4544;

const LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8'];
const PROMOTED_TO = ['q', 'r', 'b', 'n'];

export function createUciLabels(): string[] {
  const labels = new Set<string>();
  for (const l1 of LETTERS) {
    for (const n1 of NUMBERS) {
      for (const l2 of LETTERS) {
        for (const n2 of NUMBERS) {
          if (l1 === l2 && n1 === n2) continue;
          labels.add(`${l1}${n1}${l2}${n2}`);
        }
      }
    }
  }
  for (const l1 of LETTERS) {
    for (const l2 of LETTERS) {
      for (const p of PROMOTED_TO) labels.add(`${l1}7${l2}8${p}`);
      for (const p of PROMOTED_TO) labels.add(`${l1}2${l2}1${p}`);
    }
  }
  // Array.prototype.sort() with no comparator orders by UTF-16 code units, which
  // for these ASCII strings is the same order as Python's sorted().
  return [...labels].sort();
}

export const UCI_LABELS: readonly string[] = createUciLabels();

export const MOVE_TO_INT: ReadonlyMap<string, number> = new Map(UCI_LABELS.map((m, i) => [m, i]));

/** 0 = no promotion, then q, r, b, n. */
const PROMO_CODE: Record<string, number> = { q: 1, r: 2, b: 3, n: 4 };

// (from * 64 + to) * 5 + promoCode -> action index, -1 when not in the table.
// Squares use python-chess numbering: a1 = 0, b1 = 1, ... h8 = 63.
const LOOKUP = new Int16Array(64 * 64 * 5).fill(-1);

function squareIndex(name: string): number {
  return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
}

for (let i = 0; i < UCI_LABELS.length; i++) {
  const uci = UCI_LABELS[i]!;
  const from = squareIndex(uci.slice(0, 2));
  const to = squareIndex(uci.slice(2, 4));
  const promo = uci.length > 4 ? PROMO_CODE[uci[4]!]! : 0;
  LOOKUP[(from * 64 + to) * 5 + promo] = i;
}

export function promoCode(promotion: string | undefined): number {
  return promotion ? PROMO_CODE[promotion] ?? 0 : 0;
}

/** Action index for a move given python-chess square numbers, or -1 if it is not in the table. */
export function moveToIndex(from: number, to: number, promotion?: string): number {
  return LOOKUP[(from * 64 + to) * 5 + promoCode(promotion)]!;
}

export function squareName(sq: number): string {
  return String.fromCharCode(97 + (sq & 7)) + String.fromCharCode(49 + (sq >> 3));
}
