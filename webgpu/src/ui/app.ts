// Page logic: the game (chess.js), the board, the side panel and the engine worker.

import { Chess, type Move } from 'chess.js';
import { config } from '../config';
import type { SearchProgress, SearchResult } from '../engine/types';
import { EngineClient, type LoadProgress, type SearchHandle } from '../engineClient';
import type { BackendPreference, ReadyInfo } from '../messages';
import { webgpuAdvice } from './advice';
import { BoardView, pieceSvg, type PieceType } from './board';
import { gameOverInfo } from './gameOver';
import { moveRows } from './moveList';

type Sq = Parameters<Chess['get']>[0];

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing from index.html`);
  return el as T;
};

interface Settings {
  thinkSeconds: number;
  batch: number; // 0 = auto
}

const SETTINGS_KEY = 'sigma-go-settings';

/** The nearest of the time-per-move choices (an older version stored any value from 0.5 to 10). */
function snapThink(seconds: number): number {
  let best: number = config.defaultThinkSeconds;
  let gap = Infinity;
  for (const option of config.thinkOptions) {
    if (Math.abs(option - seconds) < gap) {
      gap = Math.abs(option - seconds);
      best = option;
    }
  }
  return best;
}

function loadSettings(): Settings {
  const fallback: Settings = { thinkSeconds: config.defaultThinkSeconds, batch: 0 };
  try {
    const raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? 'null') as Partial<Settings> | null;
    if (!raw) return fallback;
    const t = Number(raw.thinkSeconds);
    const b = Number(raw.batch);
    return { thinkSeconds: t > 0 ? snapThink(t) : fallback.thinkSeconds, batch: [0, 8, 16, 32].includes(b) ? b : 0 };
  } catch {
    return fallback;
  }
}

function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    /* storage blocked: settings just do not persist */
  }
}

const fmtMiB = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MiB`;
const BACKEND_LABEL = { webgpu: 'WebGPU', webgl: 'WebGL', wasm: 'CPU' } as const;

const toUci = (m: Move) => m.from + m.to + (m.promotion ?? '');

export function startApp(): void {
  const params = new URLSearchParams(location.search);
  const backendParam = params.get('backend');
  const backend: BackendPreference =
    backendParam === 'wasm' || backendParam === 'webgpu' || backendParam === 'webgl' ? backendParam : 'auto';

  // ?fen=... starts from a position, with the player to move first (for analysis and for the browser tests)
  const game = new Chess();
  const fenParam = params.get('fen');
  if (fenParam) {
    try {
      game.load(fenParam);
    } catch {
      game.reset();
    }
  }
  let human: 'w' | 'b' = game.turn();
  let engineInfo: ReadyInfo | null = null;
  let engineFailed = false;
  let pendingAi = false;
  let token = 0; // bumped whenever a search result must be ignored
  let busy: { kind: 'ai' | 'hint'; handle: SearchHandle } | null = null;
  let errorMessage: string | null = null;
  let notice: string | null = null; // a short message in the status line (a click during the engine's turn)
  let noticeTimer = 0;
  let aiTimer = 0;
  let gameOverDismissed = false;
  const settings = loadSettings();

  // ---- elements ----
  const statusEl = $('status');
  const statusBox = statusEl.closest('.status-box') as HTMLElement;
  const evalEl = $('evaluation');
  const evalBar = $('eval-bar');
  const movesEl = $<HTMLOListElement>('moves');
  const movesEmpty = $('moves-empty');
  const chip = $('engine-chip');
  const chipText = $('engine-chip-text');
  const whyBtn = $<HTMLButtonElement>('why-btn');
  const loader = $('loader');
  const loaderBar = $('loader-bar');
  const loaderTrack = $('loader-track');
  const loaderText = $('loader-text');
  const loaderTitle = $('loader-title');
  const loaderRetry = $<HTMLButtonElement>('loader-retry');
  const promo = $('promotion');
  const gameOverEl = $('gameover');
  const undoBtn = $<HTMLButtonElement>('undoBtn');
  const hintBtn = $<HTMLButtonElement>('hintBtn');
  const moveNowBtn = $<HTMLButtonElement>('moveNowBtn');
  const batchSelect = $<HTMLSelectElement>('batch-select');
  const thinkBar = $('think-bar');
  const thinkTrack = $('think-track');
  const detailsEl = $<HTMLDetailsElement>('engine-details');
  const moveForm = $<HTMLFormElement>('move-form');
  const moveInput = $<HTMLInputElement>('move-input');
  const moveError = $('move-error');

  // ---- board ----
  const legalFrom = (sq: string) => game.moves({ square: sq as Sq, verbose: true });
  const board = new BoardView($('board'), {
    canPick: (sq) => !busy && !game.isGameOver() && game.turn() === human && game.get(sq as Sq)?.color === human,
    destinations: (sq) => legalFrom(sq).map((m) => m.to),
    captures: (sq) => legalFrom(sq).filter((m) => m.isCapture()).map((m) => m.to),
    onMove: (from, to) => void tryUserMove(from, to),
    onBlocked: onBoardBlocked,
  });

  // ---- engine ----
  const engine = new EngineClient();

  function startEngine(): void {
    engineFailed = false;
    loader.className = 'loader';
    loaderTitle.textContent = 'Loading the engine';
    loaderRetry.hidden = true;
    chip.className = 'chip loading';
    chipText.textContent = 'Loading engine';
    engine
      .init({
        modelUrl: config.modelUrl,
        modelVersion: config.modelVersion,
        backend,
        ortExternalBase: config.ortExternalBase,
        onProgress: showLoadProgress,
      })
      .then(onEngineReady, onEngineFailed);
  }

  function setLoaderPercent(pct: number): void {
    loaderBar.style.width = `${pct}%`;
    loaderTrack.setAttribute('aria-valuenow', String(Math.round(pct)));
  }

  function showLoadProgress(p: LoadProgress): void {
    let pct: number | null = null;
    if (p.phase === 'download' && p.total) pct = Math.min(100, (100 * (p.loaded ?? 0)) / p.total);
    if (p.phase === 'cache') pct = 100;
    const size = p.total ? ` (${fmtMiB(p.loaded ?? 0)} of ${fmtMiB(p.total)})` : p.loaded ? ` (${fmtMiB(p.loaded)})` : '';
    loaderText.textContent = p.message + (p.phase === 'download' || p.phase === 'cache' ? size : '...');
    if (pct !== null) setLoaderPercent(pct);
    else if (p.phase !== 'download') setLoaderPercent(100);
    if (p.phase === 'cache') {
      loaderTitle.textContent = 'Loading the engine from cache';
      chipText.textContent = 'Network from cache';
    } else if (p.phase === 'download') {
      chipText.textContent = pct !== null ? `Downloading network ${pct.toFixed(0)}%` : 'Downloading network';
    } else if (p.phase === 'warmup') {
      chipText.textContent = p.backend ? `Warming up ${BACKEND_LABEL[p.backend]}` : 'Warming up';
    } else {
      chipText.textContent = 'Starting engine';
    }
  }

  function describeBackend(info: ReadyInfo): string {
    if (info.backend === 'wasm') return `CPU (WebAssembly), ${info.threads} thread${info.threads === 1 ? '' : 's'}`;
    return BACKEND_LABEL[info.backend];
  }

  function onEngineReady(info: ReadyInfo): void {
    engineInfo = info;
    loader.classList.add('done');
    window.setTimeout(() => loader.classList.add('hidden'), 450);
    chip.className = `chip ${info.backend}`;
    chipText.textContent =
      info.backend === 'wasm' ? `CPU (WebAssembly, ${info.threads} thread${info.threads === 1 ? '' : 's'})` : BACKEND_LABEL[info.backend];
    chip.title = [describeBackend(info), info.gpu, info.fromCache ? 'network loaded from this browser\'s cache' : 'network downloaded'].filter(Boolean).join(' - ');

    $('stat-backend').textContent = describeBackend(info);
    $('stat-model').textContent = info.fromCache
      ? `${fmtMiB(info.modelBytes)}, from this browser's cache (no download)`
      : `${fmtMiB(info.modelBytes)}, downloaded in ${(info.loadMs / 1000).toFixed(1)} s`;
    const runtimes = { jspi: 'JSPI build', asyncify: 'asyncify build', webgl: 'WebGL build', wasm: 'plain WASM build' } as const;
    $('stat-runtime').textContent = `onnxruntime-web (${runtimes[info.runtime]}), ${info.crossOriginIsolated ? 'cross-origin isolated' : 'not isolated'}`;
    $('stat-gpu').textContent = info.gpu ?? (info.backend === 'wasm' ? 'not used' : 'not reported');
    $('stat-threads').textContent =
      info.backend === 'wasm' ? String(info.threads) : `1 (the ${BACKEND_LABEL[info.backend]} backend does the work)`;
    $('stat-calibration').textContent = info.calibration
      .map((r) => `batch ${r.batch}: ${r.ms.toFixed(0)} ms (${r.positionsPerSecond.toFixed(0)}/s)`)
      .join('; ');

    const advice = webgpuAdvice(info, navigator.userAgent);
    const helpEl = $('webgpu-help');
    helpEl.hidden = !advice;
    whyBtn.hidden = !advice;
    if (advice) {
      $('webgpu-reason').textContent = advice.reason;
      const list = $('webgpu-steps');
      list.replaceChildren(...advice.steps.map(adviceItem));
      list.hidden = advice.steps.length === 0;
    }

    const notes: string[] = [];
    if (info.backend === 'wasm' && !info.crossOriginIsolated) notes.push('Single-threaded: the page is not cross-origin isolated.');
    if (info.backend === 'wasm') {
      const row = info.calibration.find((r) => r.batch === info.autoBatchSize) ?? info.calibration[info.calibration.length - 1];
      if (row) {
        notes.push(
          `Measured at start-up: about ${row.positionsPerSecond.toFixed(0)} positions per second, so ${settings.thinkSeconds} s is roughly ` +
            `${Math.max(1, Math.round(row.positionsPerSecond * settings.thinkSeconds))} simulations. Raise the time per move for stronger play.`,
        );
      }
    }
    $('stat-note').textContent = notes.join(' ');
    updateBatchStat();
    refresh();
    if (pendingAi) {
      pendingAi = false;
      void requestAiMove();
    }
  }

  /** a step of advice; chrome://flags/... and about:config style addresses and names are set in <code> */
  function adviceItem(text: string): HTMLLIElement {
    const li = document.createElement('li');
    const pattern = /((?:chrome|edge):\/\/[^\s,]+|about:config|dom\.webgpu\.enabled|DRI_PRIME=1|__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia|vulkaninfo --summary)/g;
    let last = 0;
    for (const m of text.matchAll(pattern)) {
      const at = m.index ?? 0;
      const token = m[0].replace(/[.:]+$/, '');
      li.append(text.slice(last, at));
      const code = document.createElement('code');
      code.textContent = token;
      li.append(code);
      last = at + token.length;
    }
    li.append(text.slice(last));
    return li;
  }

  function onEngineFailed(err: Error): void {
    engineFailed = true;
    chip.className = 'chip error';
    chipText.textContent = 'Engine failed to start';
    loader.className = 'loader failed';
    loaderTitle.textContent = 'The engine could not start';
    loaderText.textContent = err.message;
    setLoaderPercent(0);
    loaderRetry.hidden = false;
    refresh();
  }

  function effectiveBatch(): number {
    return settings.batch > 0 ? settings.batch : engineInfo?.autoBatchSize ?? config.defaultBatchSize;
  }

  function updateBatchStat(): void {
    $('stat-batch').textContent = `${effectiveBatch()}${settings.batch === 0 ? ' (auto)' : ''}`;
  }

  // ---- game ----
  function turnName(): string {
    return game.turn() === 'b' ? 'Black' : 'White';
  }

  function kingSquare(color: 'w' | 'b'): string | null {
    for (const row of game.board()) {
      for (const p of row) if (p && p.type === 'k' && p.color === color) return p.square;
    }
    return null;
  }

  function refresh(): void {
    board.setPosition(game.board());
    const history = game.history({ verbose: true });
    const last = history[history.length - 1];
    board.setLastMove(last ? [last.from, last.to] : null);
    board.setCheck(game.inCheck() ? kingSquare(game.turn()) : null);
    board.setLocked(!!busy || game.isGameOver() || game.turn() !== human);
    board.setThinking(!!busy);
    renderStatus();
    renderMoves(history);
    renderGameOver();
    undoBtn.disabled = !canUndo();
    hintBtn.disabled = !engineInfo || !!busy || game.isGameOver() || game.turn() !== human;
    moveNowBtn.disabled = busy?.kind !== 'ai';
    for (const b of document.querySelectorAll<HTMLButtonElement>('#side-seg .seg-btn')) b.setAttribute('aria-pressed', String(b.dataset.side === human));
    for (const b of document.querySelectorAll<HTMLButtonElement>('#time-seg .seg-btn')) {
      b.setAttribute('aria-pressed', String(Number(b.dataset.seconds) === settings.thinkSeconds));
    }
  }

  function renderStatus(): void {
    let status: string;
    statusEl.classList.remove('thinking');
    const over = gameOverInfo(game, human);
    if (errorMessage) status = errorMessage;
    else if (over) status = `${over.title}. ${over.reason}`;
    else if (notice) status = notice;
    else if (busy?.kind === 'ai') {
      status = 'AI is Thinking (MCTS)';
      statusEl.classList.add('thinking');
    } else if (busy?.kind === 'hint') {
      status = 'Looking for a hint';
      statusEl.classList.add('thinking');
    } else if (engineFailed && game.turn() !== human) status = 'The engine is not running, so the AI cannot move.';
    else if (!engineInfo && game.turn() !== human) status = 'Waiting for the engine to load';
    else {
      status = `${turnName()} to move`;
      if (game.inCheck()) status += `, ${turnName()} is in check`;
    }
    statusEl.textContent = status;
  }

  /** Numbered rows ("1. e4 e5"), the latest move marked, scrolled to the end. Rebuilt from the game, so undo cannot drift. */
  let movesKey = '';
  function renderMoves(history: Move[]): void {
    const key = history.map((m) => m.san).join(' ');
    if (key === movesKey && movesEl.childElementCount === (key ? Math.ceil(history.length / 2) : 0)) return;
    movesKey = key;
    const rows = moveRows(history);
    const lastIndex = history.length - 1;
    movesEl.replaceChildren(
      ...rows.map((row) => {
        const li = document.createElement('li');
        const num = document.createElement('span');
        num.className = 'num';
        num.textContent = `${row.number}.`;
        li.append(num, plyCell(row.white, row.whiteIndex === lastIndex, row.white === null), plyCell(row.black, row.blackIndex === lastIndex, false));
        return li;
      }),
    );
    movesEl.hidden = rows.length === 0;
    movesEmpty.hidden = rows.length > 0;
    movesEl.scrollTop = movesEl.scrollHeight;
  }

  function plyCell(san: string | null, current: boolean, gap: boolean): HTMLSpanElement {
    const span = document.createElement('span');
    span.className = 'ply' + (current ? ' current' : '');
    span.textContent = gap ? '...' : san ?? '';
    if (current) span.setAttribute('aria-current', 'true');
    return span;
  }

  function renderGameOver(): void {
    const over = gameOverInfo(game, human);
    if (!over || gameOverDismissed) {
      gameOverEl.hidden = true;
      return;
    }
    const wasHidden = gameOverEl.hidden;
    $('gameover-title').textContent = over.title;
    $('gameover-reason').textContent = over.reason;
    gameOverEl.hidden = false;
    if (wasHidden) $('gameover-new').focus({ preventScroll: true });
  }

  function showEval(value: number): void {
    evalEl.textContent = value.toFixed(2);
    evalBar.style.width = `${Math.max(0, Math.min(100, 50 + value * 50))}%`;
  }

  function resetEval(): void {
    evalEl.textContent = '0.0';
    evalBar.style.width = '50%';
  }

  function resetThinking(): void {
    thinkBar.style.width = '0%';
    thinkTrack.setAttribute('aria-valuenow', '0');
    $('think-time').innerHTML = '&nbsp;';
    $('think-sims').innerHTML = '&nbsp;';
  }

  function cancelSearch(): void {
    token++;
    if (busy) {
      busy.handle.stop();
      busy = null;
    }
    resetThinking();
    $('stat-line').textContent = '';
  }

  /** A press on the board while it is locked: say why nothing happened instead of ignoring it. */
  function onBoardBlocked(): void {
    if (game.isGameOver()) {
      gameOverDismissed = false;
      renderGameOver();
      return;
    }
    if (busy?.kind === 'ai') flash("It is the engine's turn. It moves when the bar is full, or press Move now.");
    else if (busy?.kind === 'hint') flash('Looking for a hint. Your move is next.');
    else if (game.turn() !== human && !engineInfo && !engineFailed) flash('Waiting for the engine to load.');
  }

  function flash(message: string): void {
    notice = message;
    window.clearTimeout(noticeTimer);
    noticeTimer = window.setTimeout(() => {
      notice = null;
      renderStatus();
    }, 2600);
    statusBox.classList.remove('nudge');
    void statusBox.offsetWidth; // restart the animation
    statusBox.classList.add('nudge');
    renderStatus();
  }

  async function chooseMove(from: string, to: string): Promise<Move | null> {
    const candidates = game.moves({ square: from as Sq, verbose: true }).filter((m) => m.to === to);
    if (candidates.length === 0) return null;
    if (candidates.length === 1) return candidates[0]!;
    // promotion
    const piece = await askPromotion(candidates[0]!.color);
    return piece ? candidates.find((m) => m.promotion === piece) ?? null : null;
  }

  function askPromotion(color: 'w' | 'b'): Promise<PieceType | null> {
    return new Promise((resolve) => {
      promo.replaceChildren();
      const box = document.createElement('div');
      box.className = 'choices';
      box.setAttribute('role', 'group');
      box.setAttribute('aria-label', 'Promote the pawn to');
      const close = (value: PieceType | null) => {
        promo.hidden = true;
        promo.onclick = null;
        promo.onkeydown = null;
        resolve(value);
      };
      for (const t of ['q', 'r', 'b', 'n'] as const) {
        const b = document.createElement('button');
        b.type = 'button';
        const name = { q: 'Queen', r: 'Rook', b: 'Bishop', n: 'Knight' }[t];
        b.title = name;
        b.setAttribute('aria-label', name);
        b.innerHTML = pieceSvg({ type: t, color });
        b.addEventListener('click', (e) => {
          e.stopPropagation();
          close(t);
        });
        box.append(b);
      }
      promo.append(box);
      promo.hidden = false;
      promo.onclick = () => close(null);
      promo.onkeydown = (e) => {
        if (e.key === 'Escape') close(null);
      };
      box.querySelector('button')?.focus();
    });
  }

  async function tryUserMove(from: string, to: string): Promise<void> {
    if (busy || game.isGameOver() || game.turn() !== human) return;
    const move = await chooseMove(from, to);
    if (!move) {
      refresh();
      return;
    }
    playUserMove(move);
  }

  function playUserMove(move: Move): void {
    errorMessage = null;
    game.move({ from: move.from, to: move.to, promotion: move.promotion });
    board.setHint(null);
    refresh();
    if (!game.isGameOver()) void requestAiMove();
  }

  async function requestAiMove(): Promise<void> {
    if (busy || game.isGameOver() || game.turn() === human) return;
    if (engineFailed) {
      refresh();
      return;
    }
    if (!engineInfo) {
      pendingAi = true;
      refresh();
      return;
    }
    const my = ++token;
    const ply = game.history().length;
    const budget = settings.thinkSeconds * 1000;
    const handle = engine.search(
      {
        fen: game.fen(),
        timeBudgetMs: budget,
        batchSize: settings.batch,
        temperature: ply < config.openingPlies ? config.openingTemperature : 0,
        noise: true,
      },
      (p) => {
        if (my === token) showProgress(p, budget, 'ai');
      },
    );
    busy = { kind: 'ai', handle };
    refresh();
    try {
      const result = await handle.result;
      if (my !== token) return; // undone, reset or replaced while thinking
      busy = null;
      finishSearch(result);
      if (result.move) {
        game.move({ from: result.move.slice(0, 2), to: result.move.slice(2, 4), promotion: result.move.length > 4 ? result.move[4] : undefined });
        showEval(result.value);
        board.setHint(null);
      }
      refresh();
    } catch (err) {
      if (my !== token) return;
      busy = null;
      errorMessage = `Engine error: ${(err as Error).message}`;
      refresh();
    }
  }

  async function requestHint(): Promise<void> {
    if (!engineInfo || busy || game.isGameOver() || game.turn() !== human) return;
    const my = ++token;
    const budget = settings.thinkSeconds * 1000;
    const handle = engine.search({ fen: game.fen(), timeBudgetMs: budget, batchSize: settings.batch, temperature: 0, noise: false }, (p) => {
      if (my === token) showProgress(p, budget, 'hint');
    });
    busy = { kind: 'hint', handle };
    refresh();
    try {
      const result = await handle.result;
      if (my !== token) return;
      busy = null;
      finishSearch(result);
      if (result.move) board.setHint([result.move.slice(0, 2), result.move.slice(2, 4)]);
      refresh();
      if (result.move) {
        const san = sanFor(result.move);
        $('stat-line').textContent = `Hint: ${san} (value ${result.value.toFixed(2)} for you)`;
      }
    } catch (err) {
      if (my !== token) return;
      busy = null;
      errorMessage = `Engine error: ${(err as Error).message}`;
      refresh();
    }
  }

  function sanFor(uci: string): string {
    try {
      const probe = new Chess(game.fen());
      return probe.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.length > 4 ? uci[4] : undefined }).san;
    } catch {
      return uci;
    }
  }

  function lineFor(pv: string[]): string {
    const probe = new Chess(game.fen());
    const out: string[] = [];
    for (const uci of pv) {
      try {
        out.push(probe.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.length > 4 ? uci[4] : undefined }).san);
      } catch {
        break;
      }
    }
    return out.join(' ');
  }

  /** Called every ~100 ms while the worker searches: time used out of the budget, simulations, best line, evaluation. */
  function showProgress(p: SearchProgress, budgetMs: number, kind: 'ai' | 'hint'): void {
    const pct = Math.min(100, (100 * p.elapsedMs) / budgetMs);
    thinkBar.style.width = `${pct}%`;
    thinkTrack.setAttribute('aria-valuenow', String(Math.round(pct)));
    $('think-time').textContent = `${(Math.min(p.elapsedMs, budgetMs) / 1000).toFixed(1)} / ${(budgetMs / 1000).toFixed(1)} s`;
    $('think-sims').textContent = `${p.simulations} simulations, ${p.simsPerSecond.toFixed(0)}/s`;
    $('stat-sims').textContent = String(p.simulations);
    $('stat-speed').textContent = `${p.simsPerSecond.toFixed(0)} sims/s`;
    if (p.pv.length) {
      const sign = kind === 'ai' ? 1 : -1; // the value is for the side to move; the evaluation box is from the AI's side
      const eval_ = (sign * p.value).toFixed(2);
      $('stat-line').textContent = `Best line: ${lineFor(p.pv)} (eval ${Number(eval_) > 0 ? '+' : ''}${eval_} for the AI)`;
    }
    if (kind === 'ai' && p.simulations > 0) showEval(p.value);
  }

  function finishSearch(r: SearchResult): void {
    notice = null;
    $('stat-sims').textContent = String(r.simulations);
    $('stat-speed').textContent = `${r.simsPerSecond.toFixed(0)} sims/s`;
    resetThinking();
    $('think-sims').textContent = `${r.simulations} simulations, ${r.simsPerSecond.toFixed(0)}/s`;
    $('stat-line').textContent = r.pv.length ? `Line: ${lineFor(r.pv)}` : '';
  }

  // ---- controls ----
  /** Ask before throwing away a game in progress. */
  function confirmDiscard(what: string): boolean {
    if (game.isGameOver() || game.history().length === 0) return true;
    return window.confirm(`${what} The current game will be lost.`);
  }

  function newGame(color: 'w' | 'b'): void {
    cancelSearch();
    errorMessage = null;
    notice = null;
    pendingAi = false;
    gameOverDismissed = false;
    window.clearTimeout(aiTimer);
    human = color;
    game.reset();
    board.setOrientation(color === 'w' ? 'white' : 'black');
    board.setHint(null);
    resetEval();
    refresh();
    if (human === 'b') aiTimer = window.setTimeout(() => void requestAiMove(), 500);
  }

  function canUndo(): boolean {
    const hist = game.history().length;
    if (game.turn() === human) return hist >= (human === 'w' ? 2 : 3);
    return hist >= 1;
  }

  function undo(): void {
    if (busy?.kind === 'hint') cancelSearch();
    if (!canUndo()) return;
    cancelSearch();
    pendingAi = false;
    gameOverDismissed = false;
    // On the player's turn take back the AI reply and the player's move; while the AI is thinking, only the player's move.
    const plies = game.turn() === human ? 2 : 1;
    for (let i = 0; i < plies; i++) game.undo();
    board.setHint(null);
    errorMessage = null;
    refresh();
  }

  /** "e4", "Nf3", "O-O", "e7e8q": the move typed in the form, as one of the legal moves, or null. */
  function parseTypedMove(text: string): Move | null {
    const typed = text.trim().replace(/[+#!?]+$/, '');
    if (!typed) return null;
    const legal = game.moves({ verbose: true });
    if (/^[a-h][1-8][a-h][1-8][qrbn]?$/i.test(typed)) {
      const uci = typed.toLowerCase();
      const found = legal.find((m) => toUci(m) === uci) ?? legal.find((m) => m.from + m.to === uci.slice(0, 4) && m.promotion === 'q' && uci.length === 4);
      if (found) return found;
    }
    const norm = (s: string) => s.replace(/[+#x]/g, '').replace(/0/g, 'O').replace('=', '');
    return legal.find((m) => norm(m.san) === norm(typed)) ?? legal.find((m) => norm(m.san).toLowerCase() === norm(typed).toLowerCase()) ?? null;
  }

  $('playWhiteBtn').addEventListener('click', () => confirmDiscard('Start a new game as White?') && newGame('w'));
  $('playBlackBtn').addEventListener('click', () => confirmDiscard('Start a new game as Black?') && newGame('b'));
  $('newGameBtn').addEventListener('click', () => confirmDiscard('Start a new game?') && newGame(human));
  $('gameover-new').addEventListener('click', () => newGame(human));
  $('gameover-close').addEventListener('click', () => {
    gameOverDismissed = true;
    renderGameOver();
    $('newGameBtn').focus({ preventScroll: true });
  });
  undoBtn.addEventListener('click', undo);
  hintBtn.addEventListener('click', () => void requestHint());
  $('flipBtn').addEventListener('click', () => board.flip());
  moveNowBtn.addEventListener('click', () => busy?.handle.stop());
  loaderRetry.addEventListener('click', startEngine);
  whyBtn.addEventListener('click', () => {
    detailsEl.open = true;
    $('webgpu-help').scrollIntoView({ block: 'center', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  });
  $('clearCacheBtn').addEventListener('click', async () => {
    const ok = await engine.clearModelCache();
    $('stat-note').textContent = ok ? 'Cached network removed. Reload the page to download it again.' : 'There was no cached network to remove.';
  });
  moveForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (game.isGameOver()) moveError.textContent = 'The game is over.';
    else if (busy || game.turn() !== human) moveError.textContent = "It is the engine's turn.";
    else {
      const move = parseTypedMove(moveInput.value);
      if (!move) moveError.textContent = `"${moveInput.value.trim()}" is not a legal move here.`;
      else {
        moveError.textContent = '';
        moveInput.value = '';
        playUserMove(move);
      }
    }
  });
  moveInput.addEventListener('input', () => (moveError.textContent = ''));

  // ---- settings ----
  for (const b of document.querySelectorAll<HTMLButtonElement>('#time-seg .seg-btn')) {
    b.addEventListener('click', () => {
      settings.thinkSeconds = Number(b.dataset.seconds);
      saveSettings(settings);
      refresh();
    });
  }
  batchSelect.value = String(settings.batch);
  batchSelect.addEventListener('change', () => {
    settings.batch = Number(batchSelect.value);
    saveSettings(settings);
    updateBatchStat();
  });

  board.setOrientation(human === 'w' ? 'white' : 'black');
  refresh();
  resetEval();
  startEngine();
}
