// Network evaluator on top of an onnxruntime session. Works with onnxruntime-web
// (browser) and onnxruntime-node (tests); both expose the same Tensor / run API.

import { ACTION_SIZE, type EvalOutput, type Evaluator } from './types';
import { PLANE_FLOATS } from './position';

export interface OrtModuleLike {
  Tensor: new (type: 'float32', data: Float32Array, dims: readonly number[]) => unknown;
}

export interface OrtSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: unknown }>>;
}

export class OrtEvaluator implements Evaluator {
  /**
   * @param buckets sizes the session is run at. A batch of n positions is zero-padded
   *   up to the next bucket, so the GPU only ever sees a few distinct shapes (each new
   *   shape can cost a shader compile). null = run at exactly n.
   */
  constructor(
    private readonly ort: OrtModuleLike,
    private readonly session: OrtSessionLike,
    private buckets: number[] | null = null,
  ) {}

  setBuckets(buckets: number[] | null): void {
    this.buckets = buckets ? [...buckets].sort((a, b) => a - b) : null;
  }

  private bucketFor(n: number): number {
    if (!this.buckets) return n;
    for (const b of this.buckets) if (b >= n) return b;
    return n;
  }

  async evaluate(planes: Float32Array, n: number): Promise<EvalOutput> {
    const size = this.bucketFor(n);
    let input = planes;
    if (size !== n) {
      input = new Float32Array(size * PLANE_FLOATS);
      input.set(planes.subarray(0, n * PLANE_FLOATS));
    } else if (planes.length !== n * PLANE_FLOATS) {
      input = planes.subarray(0, n * PLANE_FLOATS);
    }
    const tensor = new this.ort.Tensor('float32', input, [size, 14, 8, 8]);
    const out = await this.session.run({ planes: tensor });
    const policy = out.policy!.data as Float32Array;
    const value = out.value!.data as Float32Array;
    return { policy: policy.subarray(0, n * ACTION_SIZE), value: value.subarray(0, n) };
  }
}

/**
 * Evaluator for runtimes that need one session per input shape (onnxruntime-web's WebGL backend cannot take a
 * symbolic batch axis, so each session is a copy of the network with the batch fixed). A batch of n positions
 * runs on the smallest session that holds it, zero-padded; a batch larger than every session is split.
 */
export class MultiSessionEvaluator implements Evaluator {
  private readonly sizes: number[];

  constructor(
    private readonly ort: OrtModuleLike,
    private readonly sessions: ReadonlyMap<number, OrtSessionLike>,
  ) {
    this.sizes = [...sessions.keys()].sort((a, b) => a - b);
    if (this.sizes.length === 0) throw new Error('MultiSessionEvaluator needs at least one session');
  }

  async evaluate(planes: Float32Array, n: number): Promise<EvalOutput> {
    const largest = this.sizes[this.sizes.length - 1]!;
    if (n <= largest) return this.run(planes, 0, n);
    const policy = new Float32Array(n * ACTION_SIZE);
    const value = new Float32Array(n);
    for (let at = 0; at < n; at += largest) {
      const count = Math.min(largest, n - at);
      const out = await this.run(planes, at, count);
      policy.set(out.policy, at * ACTION_SIZE);
      value.set(out.value, at);
    }
    return { policy, value };
  }

  private async run(planes: Float32Array, from: number, count: number): Promise<EvalOutput> {
    const size = this.sizes.find((s) => s >= count)!;
    const input = new Float32Array(size * PLANE_FLOATS);
    input.set(planes.subarray(from * PLANE_FLOATS, (from + count) * PLANE_FLOATS));
    const out = await this.sessions.get(size)!.run({ planes: new this.ort.Tensor('float32', input, [size, 14, 8, 8]) });
    return { policy: (out.policy!.data as Float32Array).subarray(0, count * ACTION_SIZE), value: (out.value!.data as Float32Array).subarray(0, count) };
  }
}
