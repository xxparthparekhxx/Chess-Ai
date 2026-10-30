// A small chess board: drag and drop, click to move, move dots, highlights, flip.
// Written for this project (no board library). It knows nothing about the rules;
// the owner tells it which pieces can be picked up and where they can go.

export type PieceType = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
export type PieceColor = 'w' | 'b';
export interface PieceInfo {
  type: PieceType;
  color: PieceColor;
}

const FILES = 'abcdefgh';
// The same solid glyph is used for both colours and coloured with fill/stroke, as in the original UI.
// U+FE0E asks for text (not emoji) presentation, which matters on phones.
const GLYPH: Record<PieceType, string> = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' };

export function pieceSvg(p: PieceInfo): string {
  const g = GLYPH[p.type] + '︎';
  const fill = p.color === 'w' ? '#f0f0f0' : '#111';
  const stroke = p.color === 'w' ? '#111' : '#fff';
  const font = `font-family="'Segoe UI Symbol','Noto Sans Symbols 2','DejaVu Sans','Apple Symbols',sans-serif"`;
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" aria-hidden="true">` +
    `<text x="50%" y="54%" dominant-baseline="central" text-anchor="middle" font-size="60" ${font} fill="rgba(0,0,0,0.2)">${g}</text>` +
    `<text x="50%" y="50%" dominant-baseline="central" text-anchor="middle" font-size="60" ${font} fill="${fill}" stroke="${stroke}" stroke-width="1.5" style="paint-order: stroke fill">${g}</text>` +
    `</svg>`
  );
}

export interface BoardHandlers {
  /** may the piece on this square be picked up right now? */
  canPick(square: string): boolean;
  /** squares the piece on `square` can move to (empty if none) */
  destinations(square: string): string[];
  /** the user dropped / clicked a piece onto a legal destination */
  onMove(from: string, to: string): void;
}

export class BoardView {
  private readonly root: HTMLElement;
  private readonly squaresEl: HTMLElement;
  private readonly arrow: SVGSVGElement;
  private readonly squares = new Map<string, HTMLElement>();
  private pieces = new Map<string, PieceInfo>();
  private white = true; // white at the bottom
  private locked = false;
  private selected: string | null = null;
  private dests: string[] = [];
  private lastMove: [string, string] | null = null;
  private checkSquare: string | null = null;
  private hint: [string, string] | null = null;
  private drag: { from: string; ghost: HTMLElement | null; startX: number; startY: number; pointerId: number } | null = null;

  constructor(container: HTMLElement, private readonly h: BoardHandlers) {
    this.root = document.createElement('div');
    this.root.className = 'board';
    this.squaresEl = document.createElement('div');
    this.squaresEl.className = 'squares';
    this.arrow = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.arrow.setAttribute('viewBox', '0 0 8 8');
    this.arrow.classList.add('hint-arrow');
    this.root.append(this.squaresEl, this.arrow);
    container.append(this.root);

    this.squaresEl.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    this.squaresEl.addEventListener('pointerover', (e) => this.onHover(e, true));
    this.squaresEl.addEventListener('pointerout', (e) => this.onHover(e, false));
    this.buildSquares();
  }

  get orientation(): 'white' | 'black' {
    return this.white ? 'white' : 'black';
  }

  setOrientation(o: 'white' | 'black'): void {
    if ((o === 'white') === this.white) return;
    this.white = o === 'white';
    this.buildSquares();
    this.paint();
  }

  flip(): void {
    this.setOrientation(this.white ? 'black' : 'white');
  }

  /** the board as returned by chess.js board(): 8 rows from rank 8 down, each 8 entries from file a */
  setPosition(rows: ({ type: PieceType; color: PieceColor } | null)[][]): void {
    this.pieces.clear();
    rows.forEach((row, r) =>
      row.forEach((p, f) => {
        if (p) this.pieces.set(FILES[f]! + (8 - r), { type: p.type, color: p.color });
      }),
    );
    this.clearSelection();
    this.paint();
  }

  /** while the engine is thinking or the game is over */
  setLocked(locked: boolean): void {
    this.locked = locked;
    this.root.classList.toggle('locked', locked);
    if (locked) this.clearSelection();
  }

  setThinking(on: boolean): void {
    this.root.classList.toggle('thinking', on);
  }

  setLastMove(m: [string, string] | null): void {
    this.lastMove = m;
    this.paintHighlights();
  }

  setCheck(square: string | null): void {
    this.checkSquare = square;
    this.paintHighlights();
  }

  setHint(m: [string, string] | null): void {
    this.hint = m;
    this.paintHighlights();
    this.paintArrow();
  }

  private buildSquares(): void {
    this.squaresEl.replaceChildren();
    this.squares.clear();
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const file = this.white ? col : 7 - col;
        const rank = this.white ? 8 - row : row + 1;
        const name = FILES[file]! + rank;
        const el = document.createElement('div');
        el.className = 'sq ' + ((file + rank) % 2 === 1 ? 'light' : 'dark');
        el.dataset.sq = name;
        if (row === 7) el.insertAdjacentHTML('beforeend', `<span class="coord file">${FILES[file]}</span>`);
        if (col === 0) el.insertAdjacentHTML('beforeend', `<span class="coord rank">${rank}</span>`);
        this.squaresEl.append(el);
        this.squares.set(name, el);
      }
    }
  }

  private paint(): void {
    for (const [name, el] of this.squares) {
      el.querySelector('.piece')?.remove();
      const p = this.pieces.get(name);
      if (p) {
        const div = document.createElement('div');
        div.className = 'piece';
        div.innerHTML = pieceSvg(p);
        el.append(div);
      }
    }
    this.paintHighlights();
    this.paintArrow();
  }

  private paintHighlights(): void {
    for (const [name, el] of this.squares) {
      el.classList.toggle('last', !!this.lastMove && (this.lastMove[0] === name || this.lastMove[1] === name));
      el.classList.toggle('selected', this.selected === name);
      el.classList.toggle('check', this.checkSquare === name);
      el.classList.toggle('hint', !!this.hint && (this.hint[0] === name || this.hint[1] === name));
      const isDest = this.dests.includes(name);
      el.classList.toggle('move-hint', isDest);
      el.classList.toggle('capture', isDest && this.pieces.has(name));
    }
  }

  private center(square: string): [number, number] {
    const file = FILES.indexOf(square[0]!);
    const rank = Number(square[1]);
    const x = this.white ? file : 7 - file;
    const y = this.white ? 8 - rank : rank - 1;
    return [x + 0.5, y + 0.5];
  }

  private paintArrow(): void {
    this.arrow.replaceChildren();
    if (!this.hint) return;
    const [x1, y1] = this.center(this.hint[0]);
    const [x2, y2] = this.center(this.hint[1]);
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len === 0) return;
    const ux = dx / len;
    const uy = dy / len;
    const head = 0.42;
    const ex = x2 - ux * head * 0.8;
    const ey = y2 - uy * head * 0.8;
    const px = -uy;
    const py = ux;
    const ns = 'http://www.w3.org/2000/svg';
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', String(x1 + ux * 0.25));
    line.setAttribute('y1', String(y1 + uy * 0.25));
    line.setAttribute('x2', String(ex));
    line.setAttribute('y2', String(ey));
    line.setAttribute('stroke', 'rgba(74, 222, 128, 0.9)');
    line.setAttribute('stroke-width', '0.16');
    line.setAttribute('stroke-linecap', 'round');
    const tri = document.createElementNS(ns, 'polygon');
    tri.setAttribute(
      'points',
      `${x2},${y2} ${x2 - ux * head + px * head * 0.55},${y2 - uy * head + py * head * 0.55} ${x2 - ux * head - px * head * 0.55},${y2 - uy * head - py * head * 0.55}`,
    );
    tri.setAttribute('fill', 'rgba(74, 222, 128, 0.9)');
    this.arrow.append(line, tri);
  }

  private select(square: string): void {
    this.selected = square;
    this.dests = this.h.destinations(square);
    this.paintHighlights();
  }

  private clearSelection(): void {
    this.selected = null;
    this.dests = [];
    this.paintHighlights();
  }

  private squareFromEvent(e: Event): string | null {
    const el = (e.target as HTMLElement | null)?.closest?.('.sq') as HTMLElement | null;
    return el?.dataset.sq ?? null;
  }

  private squareAt(x: number, y: number): string | null {
    const el = document.elementFromPoint(x, y)?.closest('.sq') as HTMLElement | null;
    return el?.dataset.sq ?? null;
  }

  private onHover(e: PointerEvent, over: boolean): void {
    if (e.pointerType !== 'mouse' || this.locked || this.drag || this.selected) return;
    const sq = this.squareFromEvent(e);
    if (!sq) return;
    if (over && this.pieces.has(sq) && this.h.canPick(sq)) {
      this.dests = this.h.destinations(sq);
    } else {
      this.dests = [];
    }
    this.paintHighlights();
  }

  private onPointerDown(e: PointerEvent): void {
    if (this.locked || (e.pointerType === 'mouse' && e.button !== 0)) return;
    const sq = this.squareFromEvent(e);
    if (!sq) return;

    // second click of a click-click move
    if (this.selected && this.selected !== sq && this.dests.includes(sq)) {
      const from = this.selected;
      this.clearSelection();
      this.h.onMove(from, sq);
      return;
    }

    if (this.pieces.has(sq) && this.h.canPick(sq)) {
      this.select(sq);
      this.drag = { from: sq, ghost: null, startX: e.clientX, startY: e.clientY, pointerId: e.pointerId };
      window.addEventListener('pointermove', this.onPointerMove);
      window.addEventListener('pointerup', this.onPointerUp);
      window.addEventListener('pointercancel', this.onPointerCancel);
    } else {
      this.clearSelection();
    }
  }

  private onPointerMove = (e: PointerEvent): void => {
    const d = this.drag;
    if (!d || e.pointerId !== d.pointerId) return;
    if (!d.ghost) {
      if (Math.hypot(e.clientX - d.startX, e.clientY - d.startY) < 5) return;
      const src = this.squares.get(d.from)!;
      const size = src.getBoundingClientRect().width;
      const piece = src.querySelector('.piece');
      if (!piece) return;
      const ghost = document.createElement('div');
      ghost.className = 'ghost';
      ghost.style.width = ghost.style.height = `${size}px`;
      ghost.innerHTML = piece.innerHTML;
      document.body.append(ghost);
      piece.classList.add('dragging');
      d.ghost = ghost;
    }
    d.ghost.style.left = `${e.clientX}px`;
    d.ghost.style.top = `${e.clientY}px`;
  };

  private endDrag(): void {
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('pointercancel', this.onPointerCancel);
    this.drag?.ghost?.remove();
    this.squares.get(this.drag?.from ?? '')?.querySelector('.piece')?.classList.remove('dragging');
    this.drag = null;
  }

  private onPointerUp = (e: PointerEvent): void => {
    const d = this.drag;
    if (!d || e.pointerId !== d.pointerId) return;
    const dragged = !!d.ghost;
    const from = d.from;
    this.endDrag();
    if (!dragged) return; // a plain click: the piece stays selected
    const to = this.squareAt(e.clientX, e.clientY);
    if (to && to !== from && this.dests.includes(to)) {
      this.clearSelection();
      this.h.onMove(from, to);
    } else {
      this.clearSelection(); // snap back
    }
  };

  private onPointerCancel = (): void => {
    this.endDrag();
    this.clearSelection();
  };
}
