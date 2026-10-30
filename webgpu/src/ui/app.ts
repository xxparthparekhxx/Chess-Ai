// Page logic: the game (chess.js), the board, the side panel and the engine worker.

import { Chess, type Move } from 'chess.js';
import { config } from '../config';
import type { SearchProgress, SearchResult } from '../engine/types';
import { EngineClient, type LoadProgress, type SearchHandle } from '../engineClient';
import type { BackendPreference, ReadyInfo } from '../messages';
import { BoardView, pieceSvg, type PieceType } from './board';

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

function loadSettings(): Settings {
  const fallback: Settings = { thinkSeconds: config.defaultThinkSeconds, batch: 0 };
  try {
    const raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? 'null') as Partial<Settings> | null;
    if (!raw) return fallback;
    const t = Number(raw.thinkSeconds);
    const b = Number(raw.batch);
    return {
      thinkSeconds: t >= config.minThinkSeconds && t <= config.maxThinkSeconds ? t : fallback.thinkSeconds,
      batch: [0, 8, 16, 32].includes(b) ? b : 0,
    };
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

export function startApp(): void {
  const params = new URLSearchParams(location.search);
  const backendParam = params.get('backend');
  const backend: BackendPreference = backendParam === 'wasm' || backendParam === 'webgpu' ? backendParam : 'auto';

  const game = new Chess();
  let human: 'w' | 'b' = 'w';
  let engineInfo: ReadyInfo | null = null;
  let engineFailed = false;
  let pendingAi = false;
  let token = 0; // bumped whenever a search result must be ignored
  let busy: { kind: 'ai' | 'hint'; handle: SearchHandle } | null = null;
  let errorMessage: string | null = null;
  const settings = loadSettings();

  // ---- elements ----
  const statusEl = $('status');
  const evalEl = $('evaluation');
  const evalBar = $('eval-bar');
  const pgnEl = $('pgn');
  const chip = $('engine-chip');
  const chipText = $('engine-chip-text');
  const loader = $('loader');
  const loaderBar = $('loader-bar');
  const loaderText = $('loader-text');
  const loaderTitle = $('loader-title');
  const loaderRetry = $<HTMLButtonElement>('loader-retry');
  const promo = $('promotion');
  const undoBtn = $<HTMLButtonElement>('undoBtn');
  const hintBtn = $<HTMLButtonElement>('hintBtn');
  const moveNowBtn = $<HTMLButtonElement>('moveNowBtn');
  const thinkRange = $<HTMLInputElement>('think-range');
  const thinkValue = $('think-value');
  const batchSelect = $<HTMLSelectElement>('batch-select');

  // ---- board ----
  const board = new BoardView($('board'), {
    canPick: (sq) => !busy && !game.isGameOver() && game.turn() === human && game.get(sq as Sq)?.color === human,
    destinations: (sq) => game.moves({ square: sq as Sq, verbose: true }).map((m) => m.to),
    onMove: (from, to) => void tryUserMove(from, to),
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

  function showLoadProgress(p: LoadProgress): void {
    let pct: number | null = null;
    if (p.phase === 'download' && p.total) pct = Math.min(100, (100 * (p.loaded ?? 0)) / p.total);
    if (p.phase === 'cache') pct = 100;
    const detail = p.total ? ` (${fmtMiB(p.loaded ?? 0)} of ${fmtMiB(p.total)})` : p.loaded ? ` (${fmtMiB(p.loaded)})` : '';
    loaderText.textContent = p.message + (p.phase === 'download' || p.phase === 'cache' ? detail : '...');
    if (pct !== null) loaderBar.style.width = `${pct}%`;
    else if (p.phase !== 'download') loaderBar.style.width = '100%';
    chipText.textContent = pct !== null && p.phase === 'download' ? `Downloading network ${pct.toFixed(0)}%` : 'Starting engine';
  }

  function onEngineReady(info: ReadyInfo): void {
    engineInfo = info;
    loader.classList.add('done');
    window.setTimeout(() => loader.classList.add('hidden'), 450);
    chip.className = `chip ${info.backend}`;
    chipText.textContent =
      info.backend === 'webgpu'
        ? 'WebGPU'
        : `CPU (WebAssembly, ${info.threads} thread${info.threads === 1 ? '' : 's'})`;
    $('stat-backend').textContent = info.backend === 'webgpu' ? 'WebGPU' : `CPU, ${info.threads} thread${info.threads === 1 ? '' : 's'}`;
    $('stat-runtime').textContent =
      `onnxruntime-web ${info.runtime === 'jspi' ? '(JSPI build)' : info.runtime === 'asyncify' ? '(asyncify build)' : '(plain WASM build)'}` +
      (info.crossOriginIsolated ? ', isolated' : ', not isolated');
    $('stat-model').textContent = `${fmtMiB(info.modelBytes)}, ${info.fromCache ? 'from cache' : 'downloaded'} (${(info.loadMs / 1000).toFixed(1)} s)`;
    $('stat-calibration').textContent = info.calibration
      .map((r) => `batch ${r.batch}: ${r.ms.toFixed(0)} ms (${r.positionsPerSecond.toFixed(0)}/s)`)
      .join('; ');
    const notes: string[] = [];
    if (info.webgpuNote && info.backend === 'wasm') notes.push(`Running on the CPU: ${info.webgpuNote}`);
    if (info.backend === 'wasm' && !info.crossOriginIsolated) notes.push('Single-threaded: the page is not cross-origin isolated.');
    if (info.backend === 'wasm') {
      const row = info.calibration.find((r) => r.batch === info.autoBatchSize) ?? info.calibration[info.calibration.length - 1];
      if (row) {
        notes.push(
          `Measured at start-up: about ${row.positionsPerSecond.toFixed(0)} positions per second, so ${settings.thinkSeconds.toFixed(1)} s is roughly ` +
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

  function onEngineFailed(err: Error): void {
    engineFailed = true;
    chip.className = 'chip error';
    chipText.textContent = 'Engine failed to start';
    loader.className = 'loader failed';
    loaderTitle.textContent = 'The engine could not start';
    loaderText.textContent = err.message;
    loaderBar.style.width = '0%';
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

  function moveSquares(m: Move | undefined): [string, string] | null {
    return m ? [m.from, m.to] : null;
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
    board.setLastMove(moveSquares(history[history.length - 1]));
    board.setCheck(game.inCheck() ? kingSquare(game.turn()) : null);
    board.setLocked(!!busy || game.isGameOver() || game.turn() !== human);
    board.setThinking(!!busy);
    renderStatus();
    renderPgn();
    undoBtn.disabled = !canUndo();
    hintBtn.disabled = !engineInfo || !!busy || game.isGameOver() || game.turn() !== human;
    moveNowBtn.disabled = busy?.kind !== 'ai';
  }

  function renderStatus(): void {
    let status: string;
    statusEl.classList.remove('thinking');
    if (errorMessage) status = errorMessage;
    else if (game.isCheckmate()) status = `Game over, ${turnName()} is in checkmate.`;
    else if (game.isDraw()) status = 'Game over, drawn position';
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

  function renderPgn(): void {
    const sans = game.history();
    const parts: string[] = [];
    sans.forEach((san, i) => parts.push(i % 2 === 0 ? `${i / 2 + 1}. ${san}` : san));
    pgnEl.textContent = parts.join(' ');
    pgnEl.scrollTop = pgnEl.scrollHeight;
  }

  function showEval(value: number): void {
    evalEl.textContent = value.toFixed(2);
    evalBar.style.width = `${Math.max(0, Math.min(100, 50 + value * 50))}%`;
  }

  function resetEval(): void {
    evalEl.textContent = '0.0';
    evalBar.style.width = '50%';
  }

  function cancelSearch(): void {
    token++;
    if (busy) {
      busy.handle.stop();
      busy = null;
    }
    $('think-bar').style.width = '0%';
    $('stat-line').textContent = '';
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
      for (const t of ['q', 'r', 'b', 'n'] as const) {
        const b = document.createElement('button');
        b.type = 'button';
        b.title = { q: 'Queen', r: 'Rook', b: 'Bishop', n: 'Knight' }[t];
        b.innerHTML = pieceSvg({ type: t, color });
        b.addEventListener('click', (e) => {
          e.stopPropagation();
          promo.hidden = true;
          resolve(t);
        });
        box.append(b);
      }
      promo.append(box);
      promo.hidden = false;
      promo.onclick = () => {
        promo.hidden = true;
        resolve(null);
      };
    });
  }

  async function tryUserMove(from: string, to: string): Promise<void> {
    if (busy || game.isGameOver() || game.turn() !== human) return;
    errorMessage = null;
    const move = await chooseMove(from, to);
    if (!move) {
      refresh();
      return;
    }
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
    const handle = engine.search(
      {
        fen: game.fen(),
        timeBudgetMs: settings.thinkSeconds * 1000,
        batchSize: settings.batch,
        temperature: ply < config.openingPlies ? config.openingTemperature : 0,
        noise: true,
      },
      (p) => {
        if (my === token) showProgress(p, settings.thinkSeconds * 1000);
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
    const handle = engine.search(
      { fen: game.fen(), timeBudgetMs: settings.thinkSeconds * 1000, batchSize: settings.batch, temperature: 0, noise: false },
      (p) => {
        if (my === token) showProgress(p, settings.thinkSeconds * 1000);
      },
    );
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

  function showProgress(p: SearchProgress, budgetMs: number): void {
    $('stat-sims').textContent = String(p.simulations);
    $('stat-speed').textContent = `${p.simsPerSecond.toFixed(0)} sims/s`;
    $('think-bar').style.width = `${Math.min(100, (100 * p.elapsedMs) / budgetMs)}%`;
    if (p.pv.length) $('stat-line').textContent = `Line: ${lineFor(p.pv)}`;
  }

  function finishSearch(r: SearchResult): void {
    $('stat-sims').textContent = String(r.simulations);
    $('stat-speed').textContent = `${r.simsPerSecond.toFixed(0)} sims/s`;
    $('think-bar').style.width = '0%';
    $('stat-line').textContent = r.pv.length ? `Line: ${lineFor(r.pv)}` : '';
  }

  // ---- controls ----
  function newGame(color: 'w' | 'b'): void {
    cancelSearch();
    errorMessage = null;
    pendingAi = false;
    human = color;
    game.reset();
    board.setOrientation(color === 'w' ? 'white' : 'black');
    board.setHint(null);
    resetEval();
    refresh();
    if (human === 'b') window.setTimeout(() => void requestAiMove(), 500);
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
    // On the player's turn take back the AI reply and the player's move; while the AI is thinking, only the player's move.
    const plies = game.turn() === human ? 2 : 1;
    for (let i = 0; i < plies; i++) game.undo();
    board.setHint(null);
    errorMessage = null;
    refresh();
  }

  $('playWhiteBtn').addEventListener('click', () => newGame('w'));
  $('playBlackBtn').addEventListener('click', () => newGame('b'));
  undoBtn.addEventListener('click', undo);
  hintBtn.addEventListener('click', () => void requestHint());
  $('flipBtn').addEventListener('click', () => board.flip());
  moveNowBtn.addEventListener('click', () => busy?.handle.stop());
  loaderRetry.addEventListener('click', startEngine);
  $('clearCacheBtn').addEventListener('click', async () => {
    const ok = await engine.clearModelCache();
    $('stat-note').textContent = ok ? 'Cached network removed. Reload the page to download it again.' : 'There was no cached network to remove.';
  });

  // ---- settings ----
  thinkRange.min = String(config.minThinkSeconds);
  thinkRange.max = String(config.maxThinkSeconds);
  thinkRange.value = String(settings.thinkSeconds);
  thinkValue.textContent = `${settings.thinkSeconds.toFixed(1)} s`;
  thinkRange.addEventListener('input', () => {
    settings.thinkSeconds = Number(thinkRange.value);
    thinkValue.textContent = `${settings.thinkSeconds.toFixed(1)} s`;
    saveSettings(settings);
  });
  batchSelect.value = String(settings.batch);
  batchSelect.addEventListener('change', () => {
    settings.batch = Number(batchSelect.value);
    saveSettings(settings);
    updateBatchStat();
  });

  refresh();
  resetEval();
  startEngine();
}
