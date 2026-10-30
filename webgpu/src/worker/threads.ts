// How many WebAssembly threads the CPU backend uses.
//
// Measured in headless Chrome 156 on a 16-core / 32-thread Ryzen (test/browser/threads.mjs, search speed in
// simulations per second, mean of three positions, machine busy with other jobs): 1 thread 12.5, 2 → 19.3,
// 3 → 20.8, 4 → 23-25, 6 → 24.1, 8 → 23-25, 12 → 23.7, 16 → 23.6. The network is small (8x8 boards), so the
// per-layer work is too little to keep more than about four threads busy, and extra threads only burn CPU.
// Hence: all cores but one (the page and the search need one), at most 4.

export const MAX_CPU_THREADS = 4;

/** Threads for the CPU runtime. Without cross-origin isolation there are no threads at all. */
export function cpuThreads(hardwareConcurrency: number, isolated: boolean): number {
  if (!isolated) return 1;
  const cores = Number.isFinite(hardwareConcurrency) && hardwareConcurrency > 0 ? Math.floor(hardwareConcurrency) : 2;
  return Math.max(1, Math.min(cores - 1, MAX_CPU_THREADS));
}
