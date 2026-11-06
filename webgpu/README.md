# Sigma-Go in the browser

The Sigma-Go engine (ResNet policy/value network + Monte Carlo tree search) as a static web page.
The network runs on the visitor's own GPU through WebGPU, or through WebGL where WebGPU isn't available, with a
WebAssembly (CPU) fallback.
There is no server-side inference and no server code.

This directory is separate from the Python engine and does not change it. The Python side only
provides two helper scripts in `../scripts/` (export the network, dump test fixtures).

## What runs where

| Where | What |
| --- | --- |
| Visitor's browser, main thread | Board, controls, game state (chess.js). |
| Visitor's browser, Web Worker | Network download and cache, onnxruntime-web, the MCTS search, move generation for the search. |
| Visitor's GPU (or CPU) | The network forward pass: WebGPU if available, then WebGL, otherwise WebAssembly. |
| Cloudflare static assets | `index.html`, JS, CSS, font, and the onnxruntime-web runtime files (see below). 32 MB in total, largest file 17.1 MiB. |
| Hugging Face (or any static host with CORS) | `chess_net.onnx`, 47.5 MiB. Downloaded once, then kept in the browser's Cache API. |

The network file is too big for Cloudflare static assets (25 MiB per file), so it is not part of the site.
`vite.config.ts` fails the build if any file in `dist/` is over that limit or if a `model/` folder ends up in it.

## Quick start

All commands from `webgpu/` unless noted. Node 22 was used.

```bash
npm install                      # includes onnxruntime-node (CPU), used by the tests

# 1. Export the network (needs the repo's .venv; run from the repo root or use the npm script)
cd .. && .venv/bin/python scripts/export_onnx.py && cd webgpu     # writes webgpu/model/chess_net.onnx
#    or: npm run export:model

# 2. Run
npm run dev                      # http://localhost:5173, serves webgpu/model/ at /model/
```

The dev server and `vite preview` send the same COOP/COEP headers as production, so WebAssembly threads
behave the same in all three.

### Commands

| Command | What it does |
| --- | --- |
| `npm run dev` | Dev server. The network is fetched from `/model/chess_net.onnx`, served from `webgpu/model/`. |
| `npm run build` | Type-check and production build into `dist/`. The network URL comes from `.env.production` (Hugging Face). |
| `npm run build:local` | Same, but the network URL is `/model/chess_net.onnx` (for trying the production bundle locally). |
| `npm run preview` | Serves `dist/` (and `webgpu/model/`) locally. |
| `npm test` | Whole test suite, one file at a time. It took about 10 minutes here with the machine busy (the search-parity file is most of that). |
| `npm run test:fast` | Quick tests (a few seconds): encoding, search, model loader, static checks, UI helpers, the WebGL batch rewrite. |
| `npm run test:browser` | Needs `npm run build:local` first and Chrome (`CHROME=...` to point at another binary). Runs the parity test and the page test in headless Chrome, see "Browser tests". Also: `node test/browser/threads.mjs`, `node test/browser/auto.mjs`. |
| `npm run typecheck` | `tsc --noEmit`. |
| `npm run export:model` | `../scripts/export_onnx.py`. Add `-- --fp16-weights` for the smaller variant. |
| `npm run fixtures` | `../scripts/make_fixtures.py` (about 2 minutes on CPU). |

Heavy commands were run under `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 timeout 1200 ...`.

### Configuration

Build-time variables (see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VITE_MODEL_URL` | dev: `/model/chess_net.onnx`; production (`.env.production`): the Hugging Face file pinned to commit `37316d5` | Where the network is fetched from. |
| `VITE_MODEL_VERSION` | `1` | Part of the Cache API cache name. Change it when you publish a new network. |
| `VITE_ORT_EXTERNAL_BASE` | jsDelivr, pinned to the onnxruntime-web version | Where WebGPU browsers without JSPI load the 28 MiB WebGPU runtime from. Set empty to turn that path off. |

URL parameters: `?backend=wasm`, `?backend=webgl` or `?backend=webgpu` force a backend (to compare, or to test a fallback; a
forced backend is never swapped for another because it is slow). `?fen=<FEN>` starts from a position with the side to move
playing it (for analysis and for the browser tests); New game goes back to the normal start.

## Using it

Same features as the Flask page: drag and drop (or click, click; touch is tap, tap), move dots, play as White or Black,
undo, flip, evaluation bar, move list. Changed and added:

- **Board.** Last move highlighted, the king in check highlighted, legal-move dots with a ring for captures (en passant
  included), and a game-over card on the board that says why the game ended (checkmate, stalemate, repetition,
  insufficient material, fifty moves) with a New game button and a View board button. A press on the board while the
  engine is thinking explains why nothing happened instead of being ignored.
- **Moves.** Numbered rows ("1. e4 e5"), the latest move marked, scrolls to the end, rebuilt from the game on every change so it
  cannot drift from undo. A "Type a move" box (SAN such as `Nf3`, or `e2e4`) for keyboard and screen-reader use.
- **Thinking.** A bar for the time used out of the move's budget, the time and simulations so far, the line the search
  prefers and its evaluation, updating about ten times a second while the worker searches. Move now stops the search early.
  Hint draws an arrow.
- **Settings in one place.** Time per move (1, 2, 5 or 10 s), Play as White / Black (starts a new game, after a confirmation if a
  game is in progress), Flip board. Leaves per network call and Clear cached network are under Engine > Details.
- **Loading.** The overlay shows the download progress (MiB of MiB), then the warm-up named after the backend ("Compiling WebGL
  shaders ..."). When the network comes from the browser's cache the overlay title says so, and Engine shows
  "from this browser's cache (no download)".
- **Backend.** The chip at the top and Engine > Backend say WebGPU, WebGL or CPU (with the thread count). When WebGPU is not
  in use, Engine > Details says why and how to turn it on for the visitor's browser, and a "Why not WebGPU?" link next to the chip
  opens it.
- **Links.** A small header with "Parth Parekh" (parthparekhdev.com), "How it works" (the case study), GitHub and Hugging Face. The page
  is cross-origin isolated, so these are plain links; nothing is embedded or fetched from those sites.
- **Phones.** The board is the full width and the panels stack under it. No horizontal scroll at 360 px.
- **Accessibility.** Focus outlines, real labelled buttons, `prefers-reduced-motion` switches animations off, board coordinates
  and secondary text have at least 4.5:1 contrast.

The evaluation shown is from the engine's side (+1 means the engine is winning), like the original page. It is
the search's value for the move it played, not a centipawn score.

## How the search differs from the Python one

The TypeScript search in `src/engine/mcts.ts` follows `src/mcts.py` line by line: PUCT with `c_puct = 1.0`,
batched leaf evaluation with virtual loss applied to `path[1:]`, Dirichlet noise (alpha 0.3, 25 %) at the root,
sampling from visit counts at a temperature, terminal results, value sign handling in `backpropagate`.
Differences:

- **Stops on time, not on a simulation count.** It starts new batches until the time budget is used up
  (default 2 s) and then picks the move. Because the stopping point depends on speed, two runs on the same
  position can differ. The parity tests use a fixed simulation count instead.
- **Batch size is chosen on the device.** Python uses 8. On WebGPU the page times batches of 1, 8, 16 and 32 at
  start-up and uses the largest size that still gives at least 15 % more positions per second than the
  previous one and takes under 300 ms per call (never less than 8). On the CPU backend it times 1, 4 and 8
  with the same rule, starting from 1. WebGL always uses 1 (see below). You can pin 8, 16 or 32 in the settings. The timings are shown under
  Engine > Details. These are measured in the visitor's browser at start-up; they were not measured while
  writing this (see below).
- **Same leaf twice in a batch is evaluated once.** Python evaluates it twice and gets the same answer.
- **Legal moves are in python-chess order.** The Python search breaks ties by dict order, which is
  `board.legal_moves` order. With virtual loss that decides which moves the first batch visits, so the port
  sorts chess.js's moves into python-chess's order. A control test shows that leaving chess.js's own order
  changes the root visit counts in most positions (see Parity results).
- **Temperature schedule.** `main.py` uses 0.5 for the first 10 plies and 0 after. The Flask page meant the same
  but rebuilt the board from a FEN every move, so `len(board.move_stack)` was always 0 and it always used 0.5.
  The browser version follows the intended schedule, using the game's ply count.
- **Floating point.** The search uses doubles. Python's PUCT scores are float32 (NumPy 2 scalars), so an exact
  float32 tie there is not necessarily a tie here.
- **Game result.** Checkmate, stalemate, insufficient material (python-chess's definition, not chess.js's) and
  the 75-move rule are checked inside the search. Fivefold repetition is not (see Known limits).

## Parity with the Python engine

Everything below is run by `npm test` against `test/fixtures/fixtures.json`, which `scripts/make_fixtures.py`
generates from the Python code and the trained weights (sha256 of `chess_net.pth` is recorded in the file).
The network used by the tests is the exported `model/chess_net.onnx`, run with onnxruntime-node on CPU.

| Check | Result |
| --- | --- |
| Move table (`MOVE_TO_INT`) | 4,544 entries, identical to Python entry for entry. |
| Input planes | 54 positions, 48,384 values, 0 mismatches. |
| Legal moves, as indices and in python-chess order | 54 fixture positions + 400 random-playout positions (12,060 moves), 0 mismatches. |
| Game result (`get_game_result`) | 528 positions (122 terminal, 53 checkmates, insufficient material, 75-move rule), 0 mismatches. |
| ONNX vs PyTorch (54 positions) | Max abs error 1.5e-5 over all 4,544 logits (logits reach 37.4), 5.7e-6 over legal-move logits, 2.4e-7 on the value. Same best legal move in 54/54. |
| onnxruntime-web WASM build (run in Node) vs PyTorch | Max abs error 1.3e-5 (all logits), 8.6e-6 (legal), 1.9e-7 (value). 54/54. |
| MCTS, 200 simulations, noise off, temperature 0, batch 8 | Same move as Python in 54/54. Root visit counts identical in 54/54. |
| MCTS, 800 simulations (Python's default) | Same move in 54/54. Root visit counts identical in 54/54. |
| Control: same 200-simulation run with chess.js's own move order | Same move in 49/54, identical visit counts in 11/54. |

There are no disagreements to explain. The control row shows why: the comparison is sensitive, since a
change as small as the order moves are listed in breaks it. For context, Python itself picks a different move at 200
vs 800 simulations in 10 of these 54 positions, so many of them are close calls. If a future network or
library version produces disagreements, the likely causes are rounding differences between onnxruntime and
PyTorch (about 1e-5 on logits) flipping a near-tie, and float32-versus-double tie-breaking in the PUCT score.
`test/out/mcts_parity_*.json` (written by the test) lists, for each position, both visit distributions.

The fixtures are 54 positions, not a large benchmark: openings, middlegames, endgames, promotions (with
captures and underpromotion), castling rights and blocked castling, en passant (including an illegal pinned one and
one that captures a checking pawn), black to move, checks, double check, pins, near-mate.

## Node timings (CPU, rough reference only)

Measured by `test/perf.test.ts` and `test/ortWeb.test.ts` on an AMD Ryzen 9 8940HX (32 logical cores), Node 22.14,
onnxruntime-node 1.31.1. The machine was busy with other work (load average about 25 to 30), so these vary a lot between
runs; ranges are over three runs, and a single earlier run on a quieter machine is given in brackets.
They are **not browser numbers**.

| Forward pass (onnxruntime-node, CPU) | batch 1 | batch 8 | batch 32 |
| --- | --- | --- | --- |
| default threads | 14 to 20 ms (6.7) | 32 to 43 ms (19.7) | 104 to 116 ms (58) |
| 1 thread | 22 to 27 ms (13.7) | 127 to 144 ms (85) | 505 to 514 ms (340) |
| onnxruntime-web WebAssembly build, 1 thread, in Node | 83 to 109 ms | 621 to 654 ms (up to 1.1 s in other runs) | not run in the suite (4.6 s once) |

Search with the real network on CPU (onnxruntime-node, default threads, 1 s per move): 180 to 206 simulations/s at
batch 8 and 239 to 278 at batch 32 (392 and 487 on the quieter run), with 92 to 94 % of the time inside the network.

Search without the network (stand-in network that answers instantly, so this is chess.js move generation, encoding,
selection and back-propagation): 6.9k to 7.9k simulations/s at batch 8 (15.9k on the quieter run). Network calls are
the cost, not the JavaScript.

What this suggests for the CPU fallback: in these runs the WebAssembly build was roughly 5 to 13 times slower than native
single-threaded onnxruntime on this model, about 7 to 13 positions per second on one thread. That is roughly 15 to 25
simulations in a 2 s move: enough to follow the network's own preferences, not enough for deep search. Thread
scaling of the WebAssembly build in Node was inconsistent in my runs (between 0.8x and 1.6x with 4 threads), so no
claim is made about browser threads.

## Backends, in order

1. **WebGPU, JSPI build** (Chrome / Edge 137+ and anything else with WebAssembly JSPI).
2. **WebGPU, asyncify build**, loaded from jsDelivr, for browsers that have WebGPU but not JSPI.
3. **WebGL**, onnxruntime-web's `onnxruntime-web/webgl` build (a 0.5 MiB script, no `.wasm`).
4. **CPU**, onnxruntime-web's WebAssembly build, multi-threaded when the page is cross-origin isolated.

A GPU backend that is not clearly fast (under 150 positions/s in the start-up timing) is timed against the CPU backend and the
faster one is used, so a software renderer, a small integrated GPU, or a hybrid laptop where the browser picked the weak card
does not end up slower than the CPU. The note says what was measured. `?backend=` skips that check. A WebGL context that is
drawn by the CPU (SwiftShader, llvmpipe) is not used in automatic mode at all.

### What the WebGL tier does and does not do

- It runs this graph (Conv, Add, Relu, Flatten, Gemm, Tanh) and matches PyTorch: see the table below.
- **It needs two workarounds.** `env.webgl.pack = true` (unpacked textures gave wrong numbers for this network, even at batch 1), and a
  fixed batch axis: the backend cannot take a symbolic dimension and ignores `freeDimensionOverrides`, so `src/worker/fixBatch.ts`
  rewrites the batch dimension in the `.onnx` bytes before the session is created (the 47 MiB of weights are copied untouched).
- **It only works at batch 1.** onnxruntime-web 1.31.1's WebGL `Conv` returns wrong numbers for any batch above 1 (policy error up to 13 on the
  logits; already the first Conv of the network differs, on SwiftShader and on a hardware GPU). So there is one session, batch 1, and the
  search runs with one leaf per network call. A bigger batch would be split into one call per position (`MultiSessionEvaluator`).
  That caps WebGL at about the speed of a good multi-threaded CPU on this machine; its use is machines where the GPU is much faster than
  the CPU cores, and as a second choice where WebGPU is not available.
- The WebGL context needs float render targets (`EXT_color_buffer_float`); without them the tier is skipped and the reason is shown.

### Browser tests (headless Chrome 156, Linux)

`test/browser/` drives the built site in headless Chrome over the DevTools protocol (Node's built-in WebSocket, no dependencies; Chrome
is `/usr/bin/google-chrome-stable`, override with `CHROME=`). They load `vite preview` of a `build:local` build and talk to the real
worker bundle.

- `parity.mjs` runs the 54 fixture positions through the network in each backend variant and compares with PyTorch
  (`test/fixtures/fixtures.json`): max abs error on all 4544 logits, on the legal moves' logits, on the value, whether the best legal move
  is the same, and search speed (3 s per position, 3 positions).
- `ui.mjs` loads the page at 360 px wide with touch emulation and checks (DOM assertions, no screenshots): loading, the backend chip, the links,
  tap-to-move, last-move highlight, a press during the engine's turn, the engine's reply, progress over the time budget, the move list and
  undo, typed moves, settings, game over (mate in one set up with `?fen=`), the cache message on the second visit, labels, focus outline,
  contrast, `prefers-reduced-motion`, and no horizontal scroll.
- `threads.mjs` measures search speed against CPU thread count. `auto.mjs` prints what the page chooses on each GPU setup.

Variants: `wasm`; `webgl-swiftshader` and `webgpu-swiftshader` (Chrome's software renderer, what a machine without a usable GPU gets);
`webgl-igpu` (AMD Radeon 610M through ANGLE-OpenGL); `webgl-nvidia` (RTX 5060 Laptop through ANGLE-OpenGL with the PRIME offload
variables); `webgpu-nvidia` (RTX 5060 Laptop through Vulkan with `--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=native
--use-angle=vulkan`). Headless Chrome hands out its WebGPU adapter unreliably (a run in two or three), so `webgpu-nvidia` is repeated in a
fresh Chrome until it gets one.

Results, 2026-10-30, Chrome 156.0.8078.79, Ryzen 9 8940HX, onnxruntime-web 1.31.1, fp32 network. Errors are max abs differences from PyTorch:

| Variant | Logits, all 4544 | Logits, legal moves | Value | Same best legal move | Search speed (simulations/s) | Batch |
| --- | --- | --- | --- | --- | --- | --- |
| CPU, 8 threads | 1.3e-5 | 8.6e-6 | 1.9e-7 | 54/54 | 82 to 87 | 1 |
| WebGL, SwiftShader (software) | 1.7e-5 | 9.3e-6 | 4.2e-7 | 54/54 | 2.4 to 3.2 | 1 |
| WebGPU, SwiftShader (software) | 1.9e-5 | 9.1e-6 | 4.5e-7 | 54/54 | 1.1 to 1.6 | 8 |
| WebGL, AMD Radeon 610M | 1.9e-5 | 8.1e-6 | 3.6e-7 | 54/54 | 10 to 16 | 1 |
| WebGL, RTX 5060 Laptop | 1.9e-5 | 8.1e-6 | 3.6e-7 | 54/54 | 68 to 117 | 1 |
| WebGPU (JSPI), RTX 5060 Laptop | 2.1e-5 | 9.5e-6 | 4.8e-7 | 54/54 | 1060 to 1520 | 16 to 32 |

Read the speed column with care. These are headless runs on a machine that was shared with other jobs: another process was keeping the
RTX 5060 at 94 to 100 % for part of the runs (the WebGL RTX figure is 68 when it was busy and 115 on an earlier run; the table gives the range),
and the CPU figures swing between about 25 and 87 simulations/s with the machine's load (an earlier run with a load average of 33 gave 11 to 25).
SwiftShader numbers only show that those paths work; they say nothing about a real GPU. None of this is a measurement on a visitor's machine.

### CPU threads

`test/browser/threads.mjs`, search speed in simulations per second (mean of three positions), machine busy with other jobs:

| Threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Simulations/s | 12.5 | 22.7 | 54 | 47 to 68 | 54 to 77 | 57 to 80 | 68 to 83 | 70 |

While the machine was saturated (load average 33) every count from 4 to 16 gave the same 24. The network is small, so extra threads help less
and less; 8 give most of what 12 do and 16 are slower. The rule (`src/worker/threads.ts`) is all cores but one, at most 8, and 1 when the page is
not cross-origin isolated. Before, onnxruntime picked min(4, cores / 2).

## Browser support

- **WebGPU backend.** Needs a WebGPU adapter (`navigator.gpu.requestAdapter()` returns one). Then:
  - if the browser supports WebAssembly JSPI (feature-detected: `WebAssembly.Suspending` and `WebAssembly.promising`
    exist), the bundled onnxruntime-web JSPI runtime is used (17.1 MiB `.wasm`, served by the site);
  - if it has WebGPU but not JSPI, onnxruntime-web's asyncify build is loaded from `VITE_ORT_EXTERNAL_BASE`
    (default jsDelivr), because its `.wasm` is 28 MiB, above Cloudflare's 25 MiB limit;
  - if that is disabled or fails, the CPU backend is used.
- **WebGL backend.** Needs a WebGL 2 context with `EXT_color_buffer_float` that is not a software renderer. Batch 1 only (see above).
- **CPU backend (fallback).** onnxruntime-web's plain WebAssembly build (14.2 MiB `.wasm`, bundled). It needs
  WebAssembly SIMD. Multi-threaded only when the page is cross-origin isolated (the `_headers` file sets COOP and COEP);
  otherwise one thread. Thread count: see above.
- **Turning WebGPU on.** On Linux, Chrome (checked against the flag list of Chrome 156) needs `chrome://flags/#enable-unsafe-webgpu`
  ("Unsafe WebGPU Support") and `chrome://flags/#enable-vulkan` ("Vulkan") set to Enabled, then a relaunch, and a working Vulkan driver. Firefox:
  `about:config`, `dom.webgpu.enabled` = true; WebGPU in Firefox is on by default on Windows from version 141, and on Linux was
  still Nightly-only when last looked up (the gpuweb implementation-status wiki, undated), so on Linux Firefox the setting may do nothing.
  Engine > Details shows the steps for the visitor's browser (`src/ui/advice.ts`); the Firefox text was not tried in Firefox.
- Which browsers have WebGPU and JSPI, and since which version, was not checked here. The page looks at the features, not user agents (the
  user agent only picks which advice to show). The chip at the top of the page and Engine > Details say what was actually chosen and why WebGPU was
  not used.

## Deploying to chess.parthparekhdev.com

The site is live at https://chess.parthparekhdev.com (Workers static assets, network on Hugging Face). These are the steps.

1. **Export** the network (above) and, if you want it, the fp16-weights variant (below).
2. **Upload to Hugging Face.** Create a public model repo `xxparthparekhxx/sigma-go`, then for example:
   ```bash
   hf auth login
   hf repos create xxparthparekhxx/sigma-go --public --exist-ok
   hf upload xxparthparekhxx/sigma-go webgpu/model/chess_net.onnx chess_net.onnx
   ```
   `.env.production` points at the file pinned to a commit, `https://huggingface.co/xxparthparekhxx/sigma-go/resolve/37316d501c7ad1d4af9054f81cac116c57281fae/chess_net.onnx`, so a later upload can't change what the
   live site loads. After uploading a new network, put the new commit in that URL. Check that cross-origin fetching
   works:
   ```bash
   curl -sIL -H 'Origin: https://chess.parthparekhdev.com' \
     https://huggingface.co/xxparthparekhxx/sigma-go/resolve/main/chess_net.onnx | grep -i -E '^HTTP|access-control-allow-origin|content-length'
   ```
   You want `access-control-allow-origin` (either `*` or your origin) on the redirect and on the final response.
   For the published file both hops have it: the redirect echoes the origin and the CDN sends `*`.
   If you publish a new network later, change `VITE_MODEL_VERSION` so returning visitors fetch it.
3. **Build.** `cd webgpu && npm ci && npm run build`. This also copies `public/_headers` into `dist/`.
4. **Deploy as Workers static assets** (no Worker script). `wrangler.jsonc` is ready:
   ```bash
   npx wrangler@latest login
   npx wrangler@latest deploy         # from webgpu/
   ```
   `wrangler.jsonc` declares the custom domain `chess.parthparekhdev.com`. Because the `parthparekhdev.com` zone is already on
   your Cloudflare account, `wrangler deploy` should create the DNS record and certificate for it. If you prefer the dashboard:
   Workers & Pages > sigma-go-chess > Settings > Domains & Routes > Add > Custom domain.
5. **Check** (see the list below), in particular `crossOriginIsolated` in the console and the Network tab.

### Headers (`public/_headers`)

- `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on everything. Together they make the
  page cross-origin isolated, which WebAssembly threads need. Under COEP, every cross-origin request must be CORS-enabled (or send
  `Cross-Origin-Resource-Policy`). The page makes exactly two: the network file (Hugging Face) and, only on WebGPU browsers without JSPI,
  the runtime from jsDelivr. Both answer with `Access-Control-Allow-Origin`. If you add anything else cross-origin (an analytics
  script, fonts from a CDN), check it still loads, or remove the two isolation lines: the page then runs the CPU backend on one thread and
  WebGPU is unaffected.
- `/assets/*` and `/ort/*`: one-year immutable cache (the file names carry a content hash or the runtime version).
- `/` and `/index.html`: revalidate on every load.

The Cloudflare limit that matters here is 25 MiB per static asset. The largest file in `dist/` is the JSPI runtime at 17.1 MiB.

### fp16-weights variant (optional)

`python scripts/export_onnx.py --fp16-weights` also writes `chess_net.fp16w.onnx`: weights stored as fp16 and cast back to fp32 when the
session starts; computation stays fp32. It is 23.75 MiB (24,905,884 bytes) instead of 47.46 MiB. Measured parity (54 fixture positions,
onnxruntime-node CPU): max abs error 7.0e-3 on the logits (4.3e-3 on legal-move logits), 1.6e-4 on the value, same best legal move in 54/54,
and the MCTS picks the same move as Python in 54/54 at 200 simulations. Not tested in a browser; in particular I do not know whether
onnxruntime-web's WebGPU backend folds the weight `Cast` nodes at load or runs them on every call. To use it, upload it and point
`VITE_MODEL_URL` at it. At 24.9 MB it would also fit under Cloudflare's per-file limit, but it is not configured to ship with the site.

## Checking it by hand

The headless tests above cover the logic and the layout at 360 px. Not covered, so check on real browsers:

1. Chrome on Linux with the two flags, then without: the chip goes WebGPU / WebGL / CPU, and Engine > Details gives the right steps.
2. Firefox (any OS): the chip, the Details text, and whether `dom.webgpu.enabled` does what the page says. Safari and Edge as well.
3. A real phone: tap-to-move, the board filling the width, the keyboard for "Type a move", scrolling the page from the board.
4. A screen reader on the board, the move list and the game-over card.
5. The first load against the production Hugging Face URL (no CORS errors in the console), then a reload (network from cache).
6. `crossOriginIsolated` is `true` in the console; if not, the CPU backend runs on one thread.

## Known limits

- Everything in a browser was run headless in one Chrome (156, Linux). Not run: Firefox, Safari, Edge, Windows, macOS, a phone, a headed Chrome
  with a person's own flags, the asyncify WebGPU build (jsDelivr), screen readers. The 360 px layout was checked with device emulation, not a phone.
- The speed numbers are from a machine shared with other jobs (see the table); they are not what a visitor will get.
- WebGL is batch 1 only (see above), so it is about as fast as a good multi-threaded CPU on this machine, not much faster. WebGPU is where the speed is.
- The CPU fallback is slow on a weak CPU. The default 2 s per move gives few simulations there; the page says roughly how many.
- Fivefold repetition is not detected inside the search (the Python web page did not have game history either; `main.py` did). The page itself
  ends the game on chess.js's draw rules (threefold repetition, 50 moves, insufficient material, stalemate).
- The search uses chess.js's internal move functions (`_moves`, `_makeMove`, `_undoMove`) because its public API is about 40 times slower per
  call. chess.js is pinned to 1.4.0 and the tests compare against python-chess, but a chess.js upgrade needs the tests re-run.
- Only one search at a time; no pondering during the human's move, no opening book, no tablebases.
- If the GPU is lost mid-game (driver reset), a search reports an error; reload the page (or use `?backend=wasm`).
- The Cache API can be unavailable (some private modes) or full; then the network is downloaded each visit.
- The network export uses torch's TorchScript-based ONNX exporter, which recent torch versions deprecate.
- Piece glyphs are Unicode chess symbols, as in the original page, so how they look depends on the visitor's fonts.
- Strength: this is the same network as the Python engine. Nothing here claims how strong it is.

## Dependencies and licenses

Shipped in the page:

| Package | Version | License |
| --- | --- | --- |
| chess.js | 1.4.0 | BSD-2-Clause |
| onnxruntime-web | 1.31.1 | MIT |
| ... its dependencies: onnxruntime-common | 1.31.1 | MIT |
| ... flatbuffers | 25.9.23 | Apache-2.0 |
| ... long | 5.3.2 | Apache-2.0 |
| ... protobufjs (and @protobufjs/*) | 7.6.6 | BSD-3-Clause |
| ... platform | 1.3.6 | MIT |
| ... guid-typescript | 1.0.9 | ISC |
| @fontsource-variable/outfit (the Outfit font) | 5.3.0 | OFL-1.1 (font license) |

There is no board library: the board is written in `src/ui/board.ts`. chessground (GPL) is not used.
The onnxruntime-web `.wasm` files are built from the onnxruntime C++ sources (MIT) with Emscripten.

Development only: vite 8.3.5 (MIT; it depends on lightningcss, MPL-2.0, not shipped), vitest 5.0.5 (MIT), typescript 7.0.2 (Apache-2.0),
onnxruntime-node 1.31.1 (MIT), @types/node 26.6.6 (MIT).

Python side (export and fixtures, not shipped): onnx and onnxruntime (see `scripts/requirements-export.txt`), in addition to torch,
numpy and python-chess already used by the repo.

## Layout

```
webgpu/
  index.html, src/style.css          page and styles (the dark look of the Flask page)
  src/main.ts                        entry
  src/ui/app.ts                      game, controls, panels
  src/ui/board.ts                    the board
  src/ui/advice.ts                   why WebGPU is not in use and how to turn it on, per browser
  src/ui/gameOver.ts                 the reason a game ended
  src/ui/moveList.ts                 numbered rows for the move list
  src/engineClient.ts                page-side handle on the worker
  src/messages.ts                    worker protocol
  src/worker/engine.worker.ts        loads the network, runs searches, times batch sizes
  src/worker/modelLoader.ts          download with progress + Cache API
  src/worker/ortBackend.ts           WebGPU / WebGL / WASM selection
  src/worker/fixBatch.ts             sets the batch axis of the .onnx bytes (WebGL)
  src/worker/backendChoice.ts        when a GPU is timed against the CPU
  src/worker/threads.ts              CPU thread rule and the measurements behind it
  src/engine/moveMap.ts              port of create_uci_labels / MOVE_TO_INT
  src/engine/position.ts             chess.js adapter: legal moves in python-chess order, encoding, game result
  src/engine/mcts.ts                 port of src/mcts.py
  src/engine/ortEvaluator.ts         onnxruntime session -> policy / value (browser and Node); one-session-per-size variant for WebGL
  public/_headers                    Cloudflare headers
  wrangler.jsonc                     Workers static assets config (not deployed)
  vite.config.ts                     dev model route, runtime files, dist size guard
  test/                              vitest suites; test/fixtures/fixtures.json from the Python engine
  test/browser/                      headless Chrome tests: parity, page, threads, auto-selection
  model/                             exported .onnx files (git-ignored)
../scripts/export_onnx.py            PyTorch -> ONNX
../scripts/make_fixtures.py          fixtures from the Python code
```
