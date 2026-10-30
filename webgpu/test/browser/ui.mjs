// Checks the built site's page behaviour in headless Chrome at a 360 px wide phone viewport: loading, backend
// chip, navigation links, tap-to-move, the engine's reply, the move list (and undo), a press during the
// engine's turn, typing a move, game over, accessibility basics, reduced motion, and no horizontal scroll.
// Assertions on the DOM, not screenshots. Usage: npm run build:local && node test/browser/ui.mjs
import { launchChrome } from './chrome.mjs';
import { blankPage, startPreview } from './site.mjs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const failures = [];
let checks = 0;
function check(name, ok, detail = '') {
  checks++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${ok ? '' : ` ${detail}`}`);
  if (!ok) failures.push(name);
}

const preview = await startPreview();
const chrome = await launchChrome([]);
const { call } = chrome;
void blankPage;

async function waitFor(fn, arg, timeoutMs = 60000) {
  const end = Date.now() + timeoutMs;
  for (;;) {
    const v = await chrome.run(fn, arg);
    if (v) return v;
    if (Date.now() > end) return v;
    await sleep(150);
  }
}

// a finger rests on the glass for a moment: a zero-length touch is not a tap to the browser
async function tap(x, y) {
  await call('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y }] });
  await sleep(50);
  await call('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await sleep(250);
}
// Scroll the board into view, wait for the scroll to settle, and only tap once the point under the finger is
// really that square (input is routed with the compositor's scroll offset, which can lag a script-driven scroll).
async function tapSquare(sq) {
  await chrome.run(() => document.querySelector('.board').scrollIntoView({ block: 'center' }));
  for (let attempt = 0; attempt < 10; attempt++) {
    await sleep(120);
    const hit = await chrome.run((s) => {
      const r = document.querySelector(`.sq[data-sq="${s}"]`).getBoundingClientRect();
      const x = r.left + r.width / 2;
      const y = r.top + r.height / 2;
      return { x, y, under: document.elementFromPoint(x, y)?.closest('.sq')?.dataset.sq };
    }, sq);
    if (hit.under === sq) return tap(hit.x, hit.y);
  }
  throw new Error(`square ${sq} is not under its own centre`);
}
const noHorizontalScroll = () => chrome.run(() => ({ scroll: document.documentElement.scrollWidth, inner: window.innerWidth, body: document.body.scrollWidth }));

try {
  await call('Emulation.setDeviceMetricsOverride', { width: 360, height: 740, deviceScaleFactor: 2, mobile: true });
  await call('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 1 });
  await call('Page.enable');
  // a 1 s budget: the CPU runtime here searches about 25 positions a second
  await call('Page.addScriptToEvaluateOnNewDocument', { source: `window.__ev = []; for (const t of ['pointerdown','pointerup','pointercancel']) document.addEventListener(t, (e) => window.__ev.push(t + ':' + e.pointerType + ':' + (e.target.closest?.('.sq')?.dataset.sq ?? e.target.tagName)), true);` });
  await call('Page.addScriptToEvaluateOnNewDocument', { source: `localStorage.setItem('sigma-go-settings', JSON.stringify({ thinkSeconds: 1, batch: 0 }))` });
  await chrome.goto(preview.url + '/');

  // ---- loading ----
  const ready = await waitFor(() => document.getElementById('loader').classList.contains('done'));
  check('engine loads', !!ready, chrome.consoleLines.slice(-3).join(' | '));
  const chipText = await chrome.run(() => document.getElementById('engine-chip-text').textContent);
  check('chip names the backend', /^(WebGPU|WebGL|CPU)/.test(chipText), chipText);
  const stats = await chrome.run(() => ({
    backend: document.getElementById('stat-backend').textContent,
    model: document.getElementById('stat-model').textContent,
    gpu: document.getElementById('stat-gpu').textContent,
    threads: document.getElementById('stat-threads').textContent,
    help: !document.getElementById('webgpu-help').hidden,
    whyShown: !document.getElementById('why-btn').hidden,
    steps: document.getElementById('webgpu-steps').textContent,
  }));
  console.log('     engine:', JSON.stringify(stats));
  check('Engine box names the backend', /WebGPU|WebGL|CPU/.test(stats.backend), stats.backend);
  check('network line says where it came from', /downloaded|cache/.test(stats.model), stats.model);
  if (stats.backend.startsWith('CPU')) {
    check('without WebGPU: Details explains why and how (Chrome flags)', stats.help && stats.whyShown && /enable-unsafe-webgpu/.test(stats.steps) && /enable-vulkan/.test(stats.steps), stats.steps.slice(0, 200));
  }
  let sw = await noHorizontalScroll();
  check('no horizontal scroll at 360 px (loaded)', sw.scroll <= sw.inner && sw.body <= sw.inner, JSON.stringify(sw));

  // ---- navigation ----
  const links = await chrome.run(() => [...document.querySelectorAll('.site-nav a')].map((a) => ({ text: a.textContent.trim(), href: a.href, target: a.target, rel: a.rel })));
  const byText = (t) => links.find((l) => l.text.startsWith(t));
  check('link to parthparekhdev.com', byText('← Parth Parekh')?.href === 'https://www.parthparekhdev.com/');
  check('How it works link', byText('How it works')?.href === 'https://www.parthparekhdev.com/work/sigma-go-chess');
  check('GitHub link opens in a new tab with noopener', byText('GitHub')?.href === 'https://github.com/xxparthparekhxx/Chess-Ai' && byText('GitHub')?.target === '_blank' && /noopener/.test(byText('GitHub')?.rel));
  check('Hugging Face link opens in a new tab with noopener', byText('Hugging Face')?.href === 'https://huggingface.co/xxparthparekhxx/sigma-go' && byText('Hugging Face')?.target === '_blank' && /noopener/.test(byText('Hugging Face')?.rel));
  check('the page embeds nothing from those sites', await chrome.run(() => !document.querySelector('iframe, embed, object, img[src^="http"], script[src^="http"]')));

  // ---- layout ----
  const layout = await chrome.run(() => {
    const b = document.querySelector('.board').getBoundingClientRect();
    const panel = document.querySelector('.panel').getBoundingClientRect();
    return { boardWidth: b.width, boardBottom: b.bottom + scrollY, panelTop: panel.top + scrollY, innerWidth };
  });
  check('board fills the width', layout.boardWidth >= layout.innerWidth * 0.9, `${layout.boardWidth} of ${layout.innerWidth}`);
  check('panels stack below the board', layout.panelTop >= layout.boardBottom - 1, JSON.stringify(layout));

  // ---- play: tap-to-move ----
  await call('Runtime.evaluate', { expression: 'window.scrollTo(0, 0)' });
  await tapSquare('e2');
  const dots = await chrome.run(() => ({ dests: [...document.querySelectorAll('.move-hint')].map((e) => e.dataset.sq).sort(), selected: document.querySelector('.sq.selected')?.dataset.sq }));
  check('tap selects a piece and shows its legal moves', dots.selected === 'e2' && dots.dests.join() === 'e3,e4', JSON.stringify(dots));
  await tapSquare('e4');
  const afterMove = await chrome.run(() => ({
    moves: [...document.querySelectorAll('#moves li')].map((li) => li.textContent),
    last: [...document.querySelectorAll('.sq.last')].map((e) => e.dataset.sq).sort(),
  }));
  check('tap-to-move plays e4', /^1\.\s*e4/.test(afterMove.moves[0] ?? ''), JSON.stringify(afterMove) + ' ' + JSON.stringify(await chrome.run(() => window.__ev)));
  check('last move is highlighted', afterMove.last.join() === 'e2,e4', JSON.stringify(afterMove.last));

  // a press during the engine's turn is explained, not swallowed
  await tapSquare('d2');
  const notice = await chrome.run(() => document.getElementById('status').textContent);
  check("press during the engine's turn: status explains", /engine/i.test(notice), notice);
  const thinking = await chrome.run(() => ({ status: document.getElementById('status').textContent, now: !document.getElementById('moveNowBtn').disabled }));
  check('Move now is enabled while thinking', thinking.now, JSON.stringify(thinking));
  const midProgress = await waitFor(() => /\/ 1\.0 s/.test(document.getElementById('think-time').textContent) ? document.getElementById('think-time').textContent : '', null, 5000);
  check('progress over the time budget is shown', !!midProgress, String(midProgress));
  const reply = await waitFor(() => document.querySelectorAll('#moves li')[0]?.querySelectorAll('.ply')[1]?.textContent ? document.querySelectorAll('#moves li')[0].textContent : '');
  check('engine replies; the row shows both moves', /^1\.e4\S+/.test(reply.replace(/\s/g, '')), reply);
  const statusAfter = await chrome.run(() => document.getElementById('status').textContent);
  check('status goes back to the player', /White to move/.test(statusAfter), statusAfter);
  sw = await noHorizontalScroll();
  check('no horizontal scroll after a move', sw.scroll <= sw.inner, JSON.stringify(sw));

  // move list stays in sync with undo
  await chrome.run(() => document.getElementById('undoBtn').click());
  const afterUndo = await chrome.run(() => ({ rows: document.querySelectorAll('#moves li').length, empty: !document.getElementById('moves-empty').hidden, piece: !!document.querySelector('.sq[data-sq="e2"] .piece'), undoDisabled: document.getElementById('undoBtn').disabled }));
  check('undo empties the move list and restores the board', afterUndo.rows === 0 && afterUndo.empty && afterUndo.piece && afterUndo.undoDisabled, JSON.stringify(afterUndo));

  // type a move
  await chrome.run(() => { document.querySelector('.type-move').open = true; });
  await chrome.run(() => { const i = document.getElementById('move-input'); i.value = 'Nf3'; document.getElementById('move-form').requestSubmit(); });
  const typed = await chrome.run(() => document.querySelector('#moves li')?.textContent ?? '');
  check('typed move Nf3 is played', /Nf3/.test(typed), typed);
  await waitFor(() => document.querySelectorAll('#moves li .ply.current').length && !document.getElementById('moveNowBtn').disabled === false && document.getElementById('status').textContent.includes('to move'));
  await chrome.run(() => { const i = document.getElementById('move-input'); i.value = 'Ke5'; document.getElementById('move-form').requestSubmit(); });
  check('an illegal typed move is explained', /not a legal move/.test(await chrome.run(() => document.getElementById('move-error').textContent)));

  // ---- settings ----
  await chrome.run(() => document.querySelector('#time-seg [data-seconds="5"]').click());
  const pressed = await chrome.run(() => [...document.querySelectorAll('#time-seg .seg-btn')].map((b) => `${b.dataset.seconds}:${b.getAttribute('aria-pressed')}`).join());
  check('time per move: 1/2/5/10 s, current one pressed', pressed === '1:false,2:false,5:true,10:false', pressed);
  check('time per move is remembered', (await chrome.run(() => JSON.parse(localStorage.getItem('sigma-go-settings')).thinkSeconds)) === 5);
  await chrome.run(() => document.getElementById('flipBtn').click());
  check('flip puts Black at the bottom', (await chrome.run(() => document.querySelector('.squares .sq').dataset.sq)) === 'h1');

  // ---- game over (set up with ?fen): White mates with Ra8# ----
  const fen = '6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1';
  await chrome.goto(`${preview.url}/?fen=${encodeURIComponent(fen)}`);
  await waitFor(() => document.getElementById('loader').classList.contains('done'));
  const second = await chrome.run(() => ({ model: document.getElementById('stat-model').textContent, title: document.getElementById('loader-title').textContent, chip: document.getElementById('engine-chip').title }));
  check('return visit: the network is said to come from the cache', /from this browser's cache \(no download\)/.test(second.model) && /from cache/.test(second.title), JSON.stringify(second));
  await chrome.run(() => { document.querySelector('.type-move').open = true; const i = document.getElementById('move-input'); i.value = 'Ra8'; document.getElementById('move-form').requestSubmit(); });
  const over = await waitFor(() => (document.getElementById('gameover').hidden ? null : { title: document.getElementById('gameover-title').textContent, reason: document.getElementById('gameover-reason').textContent, status: document.getElementById('status').textContent, check: document.querySelector('.sq.check')?.dataset.sq, newBtn: document.activeElement?.id }), null, 10000);
  console.log('     game over:', JSON.stringify(over));
  check('game-over card with the reason', over && over.title === 'Checkmate, you win' && /Black is in check and has no legal move/.test(over.reason));
  check('check highlighted on the mated king', over?.check === 'g8', over?.check);
  check('game-over status line', /Checkmate/.test(over?.status ?? ''));
  check('focus moves to New game', over?.newBtn === 'gameover-new', over?.newBtn);
  sw = await noHorizontalScroll();
  check('no horizontal scroll with the game-over card', sw.scroll <= sw.inner, JSON.stringify(sw));
  await chrome.run(() => document.getElementById('gameover-close').click());
  check('View board dismisses the card', await chrome.run(() => document.getElementById('gameover').hidden));
  await tapSquare('a8');
  check('pressing the finished board shows the card again', await chrome.run(() => !document.getElementById('gameover').hidden));
  await chrome.run(() => document.getElementById('gameover-new').click());
  const fresh = await chrome.run(() => ({ over: document.getElementById('gameover').hidden, rows: document.querySelectorAll('#moves li').length, status: document.getElementById('status').textContent, piece: !!document.querySelector('.sq[data-sq="e2"] .piece') }));
  check('New game starts over', fresh.over && fresh.rows === 0 && /White to move/.test(fresh.status) && fresh.piece, JSON.stringify(fresh));

  // ---- accessibility ----
  const a11y = await chrome.run(() => {
    const name = (el) => (el.getAttribute('aria-label') || el.textContent || el.title || '').trim();
    const unnamedButtons = [...document.querySelectorAll('button')].filter((b) => !name(b)).map((b) => b.id || b.className);
    const unlabelled = [...document.querySelectorAll('input, select')].filter((i) => !i.labels?.length && !i.getAttribute('aria-label')).map((i) => i.id);
    const clickableDivs = [...document.querySelectorAll('div[onclick], span[onclick]')].length;
    return { unnamedButtons, unlabelled, clickableDivs, lang: document.documentElement.lang, main: !!document.querySelector('main') };
  });
  check('every button has a name', a11y.unnamedButtons.length === 0, a11y.unnamedButtons.join());
  check('every input has a label', a11y.unlabelled.length === 0, a11y.unlabelled.join());
  check('page has lang and a main landmark', a11y.lang === 'en' && a11y.main);
  await call('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
  await call('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
  const focus = await chrome.run(() => { const el = document.activeElement; const cs = getComputedStyle(el); return { tag: el.tagName, outline: cs.outlineStyle, width: cs.outlineWidth, focusVisible: el.matches(':focus-visible') }; });
  check('keyboard focus is visible', focus.focusVisible && focus.outline !== 'none' && parseFloat(focus.width) >= 2, JSON.stringify(focus));
  const contrast = await chrome.run(() => {
    const lum = (rgb) => { const [r, g, b] = rgb.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
    const ratio = (a, b) => { const la = lum(a), lb = lum(b); return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05); };
    const light = getComputedStyle(document.querySelector('.sq.light')); const dark = getComputedStyle(document.querySelector('.sq.dark'));
    const parse = (c) => c.match(/[\d.]+/g).slice(0, 3).map(Number);
    return { lightSquare: ratio(parse(light.color), parse(light.backgroundColor)), darkSquare: ratio(parse(dark.color), parse(dark.backgroundColor)), secondary: ratio([148, 163, 184], [30, 41, 59]) };
  });
  check('board coordinates and secondary text have 4.5:1 contrast', contrast.lightSquare >= 4.5 && contrast.darkSquare >= 4.5 && contrast.secondary >= 4.5, JSON.stringify(contrast));

  // ---- reduced motion ----
  await call('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
  const motion = await chrome.run(() => { const el = document.querySelector('.board'); el.classList.add('thinking'); const d = getComputedStyle(el).animationDuration; el.classList.remove('thinking'); return d; });
  check('prefers-reduced-motion stops the animations', parseFloat(motion) < 0.01, motion);
  await call('Emulation.setEmulatedMedia', { features: [] });

  // ---- 360 px with everything open ----
  await chrome.run(() => { document.getElementById('engine-details').open = true; });
  sw = await noHorizontalScroll();
  check('no horizontal scroll with Details open', sw.scroll <= sw.inner, JSON.stringify(sw));
  const errors = chrome.consoleLines.filter((l) => /^(error|exception)/.test(l) && !/Failed to load resource|favicon/.test(l));
  check('no console errors', errors.length === 0, errors.join(' | '));
} catch (err) {
  failures.push(String(err.message ?? err));
  console.log('FAIL (script)', err);
} finally {
  await chrome.close();
  preview.stop();
}
console.log(`\n${checks - failures.length}/${checks} checks passed`);
process.exit(failures.length ? 1 : 0);
