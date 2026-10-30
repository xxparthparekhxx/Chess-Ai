// How many WebAssembly threads the CPU backend uses.
//
// Measured in headless Chrome 156 on a 16-core / 32-thread Ryzen 9 8940HX (test/browser/threads.mjs; search speed in
// simulations per second, mean of three positions, with other jobs running on the machine, so read the trend):
//
//   threads   1     2     3     4         6         8         12        16
//   sims/s    12.5  22.7  54    47-68     54-77     57-80     68-83     70       (ranges are repeated runs)
//
// With the machine saturated (load average 33) every count from 4 to 16 gave the same ~24. The network is small
// (8x8 boards), so the work per layer is short and extra threads help less and less: 8 threads give most of what 12
// do, and 16 are slower than 12. So: all cores but one (the page and the search need one), at most 8.

export const MAX_CPU_THREADS = 8;

/** Threads for the CPU runtime. Without cross-origin isolation there are no threads at all. */
export function cpuThreads(hardwareConcurrency: number, isolated: boolean): number {
  if (!isolated) return 1;
  const cores = Number.isFinite(hardwareConcurrency) && hardwareConcurrency > 0 ? Math.floor(hardwareConcurrency) : 2;
  return Math.max(1, Math.min(cores - 1, MAX_CPU_THREADS));
}
