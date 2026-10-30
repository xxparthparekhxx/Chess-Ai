import { Chess } from 'chess.js';
import { describe, expect, it } from 'vitest';
import { webgpuAdvice, detectBrowser, detectSystem } from '../src/ui/advice';
import { gameOverInfo } from '../src/ui/gameOver';
import { moveRows } from '../src/ui/moveList';

const play = (game: Chess, sans: string) => sans.split(' ').forEach((m) => game.move(m));

describe('gameOverInfo', () => {
  it('is null while the game is on', () => {
    expect(gameOverInfo(new Chess(), 'w')).toBeNull();
  });
  it('fool\'s mate: checkmate, and who won from the player\'s side', () => {
    const g = new Chess();
    play(g, 'f3 e5 g4 Qh4#');
    expect(gameOverInfo(g, 'w')).toMatchObject({ title: 'Checkmate, the engine wins', result: -1, reason: 'White is in check and has no legal move.' });
    expect(gameOverInfo(g, 'b')).toMatchObject({ title: 'Checkmate, you win', result: 1 });
  });
  it('stalemate, insufficient material, repetition, fifty moves', () => {
    expect(gameOverInfo(new Chess('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'), 'w')).toMatchObject({ title: 'Draw', result: 0, reason: expect.stringContaining('Stalemate: Black') });
    expect(gameOverInfo(new Chess('8/8/4k3/8/8/3K4/8/8 w - - 0 1'), 'w')?.reason).toMatch(/enough pieces/);
    const rep = new Chess();
    play(rep, 'Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8');
    expect(gameOverInfo(rep, 'w')?.reason).toMatch(/three times/);
    expect(gameOverInfo(new Chess('8/8/4k3/8/8/3K4/R7/8 w - - 100 80'), 'w')?.reason).toMatch(/Fifty moves/);
  });
});

describe('moveRows', () => {
  it('pairs White and Black moves and numbers the rows', () => {
    const g = new Chess();
    play(g, 'e4 e5 Nf3');
    expect(moveRows(g.history({ verbose: true }))).toEqual([
      { number: 1, white: 'e4', black: 'e5', whiteIndex: 0, blackIndex: 1 },
      { number: 2, white: 'Nf3', black: null, whiteIndex: 2, blackIndex: -1 },
    ]);
  });
  it('follows undo', () => {
    const g = new Chess();
    play(g, 'e4 e5 Nf3 Nc6');
    g.undo();
    g.undo();
    expect(moveRows(g.history({ verbose: true }))).toHaveLength(1);
  });
  it('starts a row at Black\'s move with a gap, and keeps the move number of a set-up position', () => {
    const g = new Chess('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 17');
    play(g, 'Kd7 Kd2');
    expect(moveRows(g.history({ verbose: true }))).toEqual([
      { number: 17, white: null, black: 'Kd7', whiteIndex: -1, blackIndex: 0 },
      { number: 18, white: 'Kd2', black: null, whiteIndex: 1, blackIndex: -1 },
    ]);
  });
});

describe('webgpuAdvice', () => {
  const CHROME_LINUX = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36';
  const FIREFOX_LINUX = 'Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0';
  const base = { webgpuNote: null, webglNote: null, gpu: null };

  it('detects browser and system', () => {
    expect(detectBrowser(CHROME_LINUX)).toBe('chrome');
    expect(detectBrowser(FIREFOX_LINUX)).toBe('firefox');
    expect(detectBrowser('Mozilla/5.0 (Windows NT 10.0) Chrome/150 Safari/537.36 Edg/150')).toBe('edge');
    expect(detectSystem(CHROME_LINUX)).toBe('linux');
    expect(detectSystem('Mozilla/5.0 (Linux; Android 14) Chrome/150')).toBe('android');
  });
  it('is null when WebGPU is running', () => {
    expect(webgpuAdvice({ ...base, backend: 'webgpu', webgpuProblem: null }, CHROME_LINUX)).toBeNull();
  });
  it('Chrome on Linux: the two flags, by name', () => {
    const a = webgpuAdvice({ ...base, backend: 'webgl', gpu: 'ANGLE (NVIDIA)', webgpuProblem: 'no-api' }, CHROME_LINUX)!;
    expect(a.reason).toMatch(/^Running on WebGL \(ANGLE \(NVIDIA\)\)\. This browser does not offer WebGPU/);
    const text = a.steps.join('\n');
    expect(text).toContain('chrome://flags/#enable-unsafe-webgpu');
    expect(text).toContain('chrome://flags/#enable-vulkan');
  });
  it('Firefox: dom.webgpu.enabled, and the Linux caveat', () => {
    const a = webgpuAdvice({ ...base, backend: 'wasm', webglNote: 'WebGL 2 is not available.', webgpuProblem: 'no-api' }, FIREFOX_LINUX)!;
    const text = a.steps.join('\n');
    expect(text).toContain('dom.webgpu.enabled');
    expect(text).toMatch(/Nightly/);
    expect(text).toContain('WebGL was not used either');
    const slowGl = webgpuAdvice({ ...base, backend: 'wasm', webgpuProblem: 'no-api', webglNote: 'WebGL on AMD ran at 10 positions/s and the CPU at 40, so the CPU is used.' }, CHROME_LINUX)!;
    expect(slowGl.steps.join('\n')).toMatch(/DRI_PRIME=1/);
  });
  it('a hand-picked backend gets no steps; a slow GPU gets the dual-GPU tip', () => {
    expect(webgpuAdvice({ ...base, backend: 'wasm', webgpuProblem: 'forced-off' }, CHROME_LINUX)!.steps).toEqual([]);
    const slow = webgpuAdvice({ ...base, backend: 'wasm', webgpuProblem: 'slow-gpu', webgpuNote: 'WebGPU on google swiftshader ran at 2 positions/s and the CPU at 40, so the CPU is used.' }, CHROME_LINUX)!;
    expect(slow.steps[0]).toMatch(/DRI_PRIME=1/);
  });
});
