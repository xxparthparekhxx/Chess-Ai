"""Label every self-play position with full-strength Stockfish: its best move and its evaluation.

Self-play then only decides which positions the network trains on (the ones it actually reaches);
the targets come from Stockfish, as in the Lichess fine-tune: policy = Stockfish's move, value =
tanh(cp / 400) from White's side (mate = +-1). See diagnose_gen1.py for why the network's own
search results made worse targets.

    python selfplay/relabel.py runs/sp2/games/gen_001 --nodes 50000 --workers 11

Reads <dir>/positions.npz (needs the "fen" array the actor writes) and writes <dir>/sf.npz:
  sf_move   int16 [N]    index of Stockfish's move in the row's legal list (-1 if it isn't there)
  sf_value  float32 [N]  Stockfish's evaluation from White's side, squashed to [-1, 1]
"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import chess
import chess.engine
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import move_index  # noqa: E402

STOCKFISH = os.environ.get("STOCKFISH", os.path.expanduser("~/.local/share/stockfish/stockfish"))
_sf = None
_nodes = None


def _init(nodes):
    global _sf, _nodes
    _sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    _sf.configure({"Threads": 1, "Hash": 32})
    _nodes = nodes


def _label(fen):
    b = chess.Board(fen)
    info = _sf.analyse(b, chess.engine.Limit(nodes=_nodes))
    pv = info.get("pv") or []
    move = move_index(pv[0]) if pv else -1
    s = info["score"].white()
    if s.is_mate():
        v = 1.0 if s.mate() > 0 else -1.0
    else:
        v = math.tanh(s.score() / 400.0)
    return move, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--nodes", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    d = np.load(os.path.join(a.dir, "positions.npz"))
    fens, legal = d["fen"], d["legal"]
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.nodes,)) as pool:
        out = pool.map(_label, list(fens), chunksize=64)
    moves = np.array([m for m, _ in out], dtype=np.int64)
    pos = np.full(len(out), -1, dtype=np.int16)
    hit = legal == moves[:, None]
    found = hit.any(1)
    pos[found] = hit[found].argmax(1)
    vals = np.array([v for _, v in out], dtype=np.float32)
    np.savez(os.path.join(a.dir, "sf.npz"), sf_move=pos, sf_value=vals)
    print(f"{len(out)} positions in {time.time() - t0:.0f} s, {int((~found).sum())} without a usable move; "
          f"mean |value| {np.abs(vals).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
