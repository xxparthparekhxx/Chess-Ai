// Build-time configuration (Vite env variables, see webgpu/.env.example).

/** onnxruntime-web version the app is built against (read from node_modules at build time). */
export const ORT_VERSION = __ORT_VERSION__;

/** Where the bundled onnxruntime-web runtime files (JSPI and plain WASM) are served from. */
export const ORT_BASE = __ORT_BASE__;

const env = import.meta.env;

export const config = {
  /** Where the network is fetched from. A relative path is served by the dev server from webgpu/model/. */
  modelUrl: env.VITE_MODEL_URL || '/model/chess_net.onnx',

  /** Part of the Cache API cache name; change it when you publish a new network. */
  modelVersion: env.VITE_MODEL_VERSION || '1',

  /**
   * onnxruntime-web's WebGPU build for browsers without JSPI ships a 28 MiB .wasm, which is over
   * Cloudflare's 25 MiB per-file limit for static assets, so it is not part of the site. Browsers
   * that have WebGPU but not JSPI load it from here instead. Set VITE_ORT_EXTERNAL_BASE to an empty
   * string to disable that path (those browsers then use the CPU backend).
   */
  ortExternalBase:
    env.VITE_ORT_EXTERNAL_BASE !== undefined
      ? env.VITE_ORT_EXTERNAL_BASE
      : `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,

  /** Thinking time per move, seconds. */
  defaultThinkSeconds: 2,
  minThinkSeconds: 0.5,
  maxThinkSeconds: 10,

  /** Python's MCTS uses 8 leaves per network call. */
  defaultBatchSize: 8,

  /** Temperature 0.5 for the first 10 plies, then greedy (the schedule in main.py / web/app.py). */
  openingPlies: 10,
  openingTemperature: 0.5,
};
