// When to keep a GPU backend and when the CPU is the better choice. A GPU can be slower than the CPU (a software
// renderer, a small integrated GPU, a hybrid laptop where the browser picked the weak card), and the CPU path
// runs 12 to 75 positions per second here depending on how busy the machine is, so a fixed limit would be wrong
// both ways. Instead, a GPU that is not clearly fast is timed against the CPU and the faster one is used.

import type { BackendName } from '../messages';

/** A GPU backend at or above this many positions per second is kept without timing the CPU: no CPU in a browser gets near it. */
export const CLEARLY_FAST_POSITIONS_PER_SECOND = 150;

/** Only on automatic: a backend picked with ?backend= is what was asked for. */
export function needsCpuCheck(preference: string, backend: BackendName, gpuPositionsPerSecond: number): boolean {
  return preference === 'auto' && backend !== 'wasm' && gpuPositionsPerSecond < CLEARLY_FAST_POSITIONS_PER_SECOND;
}

export function cpuIsFaster(gpuPositionsPerSecond: number, cpuPositionsPerSecond: number): boolean {
  return cpuPositionsPerSecond >= gpuPositionsPerSecond;
}

export function slowGpuNote(backend: BackendName, gpu: string | null, gpuRate: number, cpuRate: number): string {
  const name = backend === 'webgpu' ? 'WebGPU' : 'WebGL';
  return `${name}${gpu ? ` on ${gpu}` : ''} ran at ${gpuRate.toFixed(0)} positions/s and the CPU at ${cpuRate.toFixed(0)}, so the CPU is used.`;
}
