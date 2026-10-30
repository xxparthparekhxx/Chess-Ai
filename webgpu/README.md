# Sigma-Go in the browser

The Sigma-Go engine (ResNet policy/value network + Monte Carlo tree search) as a static web page.
The network runs on the visitor's own GPU through WebGPU, with a WebAssembly (CPU) fallback.
There is no server-side inference and no server code.

This directory is separate from the Python engine and does not change it. The Python side only
provides two helper scripts in `../scripts/` (export the network, dump test fixtures).

## What runs where

| Where | What |
| --- | --- |
| Visitor's browser, main thread | Board, controls, game state (chess.js). |
| Visitor's browser, Web Worker | Network download and cache, onnxruntime-web, the MCTS search, move generation for the search. |
| Visitor's GPU (or CPU) | The network forward pass: WebGPU if available, otherwise WebAssembly. |
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
| `npm run test:fast` | Tests that do not need the network file (a few seconds). |
| `npm run typecheck` | `tsc --noEmit`. |
| `npm run export:model` | `../scripts/export_onnx.py`. Add `-- --fp16-weights` for the smaller variant. |
| `npm run fixtures` | `../scripts/make_fixtures.py` (about 2 minutes on CPU). |

Heavy commands were run under `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 timeout 1200 ...`.

### Configuration

Build-time variables (see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VITE_MODEL_URL` | dev: `/model/chess_net.onnx`; production (`.env.production`): `https://huggingface.co/xxparthparekhxx/sigma-go/resolve/main/chess_net.onnx` | Where the network is fetched from. |
| `VITE_MODEL_VERSION` | `1` | Part of the Cache API cache name. Change it when you publish a new network. |
| `VITE_ORT_EXTERNAL_BASE` | jsDelivr, pinned to the onnxruntime-web version | Where WebGPU browsers without JSPI load the 28 MiB WebGPU runtime from. Set empty to turn that path off. |

URL parameter: `?backend=wasm` forces the CPU backend (to compare, or to test the fallback).

## Using it

Same features as the Flask page: drag and drop (or click, click), move dots, play as White or Black, undo, flip,
evaluation bar, move list. Added: a Hint button (the engine searches the position for you and draws an arrow),
a "Move now" button that stops the search and plays the best move so far, a time-per-move slider
(0.5 to 10 s, default 2 s), a batch-size setting, and an Engine panel showing the backend, simulations done,
simulations per second, the line the search currently prefers, and start-up timing.

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
  with the same rule, starting from 1. You can pin 8, 16 or 32 in the settings. The timings are shown under
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

## Browser speed: not measured

I did not run the page in a browser, so there is no measured WebGPU or in-browser WebAssembly speed anywhere in this
repo. The page measures its own forward-pass time at start-up (Engine > Details) and shows simulations per second
while thinking; read those on the target hardware before quoting a number.

## Browser support

- **WebGPU backend.** Needs a WebGPU adapter (`navigator.gpu.requestAdapter()` returns one). Then:
  - if the browser supports WebAssembly JSPI (feature-detected: `WebAssembly.Suspending` and `WebAssembly.promising`
    exist), the bundled onnxruntime-web JSPI runtime is used (17.1 MiB `.wasm`, served by the site);
  - if it has WebGPU but not JSPI, onnxruntime-web's asyncify build is loaded from `VITE_ORT_EXTERNAL_BASE`
    (default jsDelivr), because its `.wasm` is 28 MiB, above Cloudflare's 25 MiB limit;
  - if that is disabled or fails, the CPU backend is used.
- **CPU backend (fallback).** onnxruntime-web's plain WebAssembly build (14.2 MiB `.wasm`, bundled). It needs
  WebAssembly SIMD. Multi-threaded only when the page is cross-origin isolated (the `_headers` file sets COOP and COEP);
  otherwise one thread.
- Which browsers have WebGPU and JSPI, and since which version, was not checked here. The page looks at the features, not user agents. The
  chip at the top of the page and Engine > Details say what was actually chosen and why WebGPU was not used.

## Deploying to chess.parthparekhdev.com (not done yet)

Nothing has been deployed or uploaded. These are the steps.

1. **Export** the network (above) and, if you want it, the fp16-weights variant (below).
2. **Upload to Hugging Face.** Create a public model repo `xxparthparekhxx/sigma-go`, then for example:
   ```bash
   hf auth login
   hf repos create xxparthparekhxx/sigma-go --public --exist-ok
   hf upload xxparthparekhxx/sigma-go webgpu/model/chess_net.onnx chess_net.onnx
   ```
   The resulting URL is `https://huggingface.co/xxparthparekhxx/sigma-go/resolve/main/chess_net.onnx`, which is what
   `.env.production` already points at. Check that cross-origin fetching works:
   ```bash
   curl -sIL -H 'Origin: https://chess.parthparekhdev.com' \
     https://huggingface.co/xxparthparekhxx/sigma-go/resolve/main/chess_net.onnx | grep -i -E '^HTTP|access-control-allow-origin|content-length'
   ```
   You want `access-control-allow-origin` (either `*` or your origin) on the redirect and on the final response.
   (For other public files on Hugging Face I saw exactly that, on both hops; your repo does not exist yet, so it was not checked.)
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

## Checking it in a browser

Not done by me. Things to check, in a Chromium browser with WebGPU first, then without:

1. `npm run build:local && npm run preview` (or `npm run dev`). Open the page. The chip should go from "Loading engine" to "WebGPU" (green)
   or "CPU (WebAssembly, N threads)" (amber). Engine > Details says which runtime and why.
2. First load shows download progress; reload: the network should come from cache ("from cache" in Details, no 47 MB transfer in the Network tab).
3. DevTools console: `crossOriginIsolated` should be `true`. If not, the CPU backend runs on one thread.
4. Play a few moves: drag and drop, click-click, promotion picker (move a pawn to the last rank), castling, en passant, undo, flip, play as Black
   (the engine moves first), Hint (arrow appears), "Move now" while thinking.
5. While thinking, the status line shows the thinking message, the Engine panel shows simulations, sims/s and a line. Read sims/s and the start-up
   batch timings from Details; that is the browser speed number nobody has yet.
6. Change "Time per move" to 0.5 s and 10 s and confirm the move arrives about when expected.
7. `?backend=wasm` should force the CPU backend. Try a browser or flag combination without WebGPU to see the fallback note.
8. After deploying: the production page should fetch the network from Hugging Face with no CORS errors in the console, and `/ort/<version>/...wasm`
   should be served as `application/wasm`.

## Known limits

- No browser run, so no browser speed, and the UI, worker wiring to real onnxruntime-web, and the WebGPU path are untested beyond type checks,
  a production build, Node tests of the worker with onnxruntime-node standing in, and onnxruntime-web's WebAssembly build in Node.
- The CPU fallback is slow (see the Node timings). The default 2 s per move gives few simulations there; the page says roughly how many.
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
  src/engineClient.ts                page-side handle on the worker
  src/messages.ts                    worker protocol
  src/worker/engine.worker.ts        loads the network, runs searches, times batch sizes
  src/worker/modelLoader.ts          download with progress + Cache API
  src/worker/ortBackend.ts           WebGPU / WASM selection
  src/engine/moveMap.ts              port of create_uci_labels / MOVE_TO_INT
  src/engine/position.ts             chess.js adapter: legal moves in python-chess order, encoding, game result
  src/engine/mcts.ts                 port of src/mcts.py
  src/engine/ortEvaluator.ts         onnxruntime session -> policy / value (browser and Node)
  public/_headers                    Cloudflare headers
  wrangler.jsonc                     Workers static assets config (not deployed)
  vite.config.ts                     dev model route, runtime files, dist size guard
  test/                              vitest suites; test/fixtures/fixtures.json from the Python engine
  model/                             exported .onnx files (git-ignored)
../scripts/export_onnx.py            PyTorch -> ONNX
../scripts/make_fixtures.py          fixtures from the Python code
```
