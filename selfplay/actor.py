"""Self-play: one network plays both sides of many games at once and records training positions.

Per move, as in KataGo's playout cap randomization: with probability --p-full the move gets a full
search (--full-sims, Dirichlet noise at the root) and its visit distribution is recorded as a policy
target; otherwise a fast search (--fast-sims, no noise) just plays the move. Every position gets the
game result as its value target. The tree under the played move is kept for the next move.

The first --temp-plies plies are sampled in proportion to visits, later moves take the most visited.
A side resigns after two of its own moves in a row with root value below --resign; --no-resign of the
games are played out to measure how often resigning would have been wrong.

Output in --out: positions.npz, games.pgn, stats.json (written last; its presence marks the run done).
  board  uint8 [N,64]   0 empty, 1..12 = plane + 1 (data/lichess_evals layout)
  turn   uint8 [N]      1 = White to move
  legal  int16 [N,128]  legal move indices, -1 padded
  pi     float16 [N,128] visit share of each legal move (all 0 when the move had a fast search)
  z      float32 [N]    game result from White's side
  q      float32 [N]    root value from White's side after the search
  fen    str [N]        the position, so relabel.py can hand it to Stockfish exactly (castling, en passant)
"""
import argparse
import json
import os
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (MAX_LEGAL, Evaluator, Search, bits_to_codes, board_bits, collect_round,  # noqa: E402
                    finish_round, load_net, move_index, start_round)


class Game:
    def __init__(self, gid, ev, rng, a):
        self.gid, self.rng, self.a = gid, rng, a
        self.board = chess.Board()
        self.search = Search(self.board, ev, a.c_puct)
        self.no_resign = rng.random() < a.no_resign
        self.rec = []                  # (bits, turn, legal idx, pi or None, q white, fen)
        self.low = {chess.WHITE: 0, chess.BLACK: 0}
        self.would_resign = None       # side that would have resigned, in no-resign games
        self.result = None
        self.termination = None
        self.plan()

    def plan(self):
        self.full = self.rng.random() < self.a.p_full
        self.target = self.a.full_sims if self.full else self.a.fast_sims
        self.noised = False
        if not self.full:
            self.search.clear_noise()

    def ready(self):
        """True when the root has its visits; adds root noise once the root is expanded."""
        r = self.search.root
        if r.pending:
            return False
        if self.full and not self.noised:
            self.search.add_noise(self.rng)
            self.noised = True
        return r.n >= self.target or len(r.moves) == 1

    def move(self):
        a, s, b = self.a, self.search, self.board
        r = s.root
        visits = r.N
        q = s.root_q()
        if len(r.moves) <= MAX_LEGAL:
            pi = (visits / visits.sum()).astype(np.float16) if self.full and len(r.moves) > 1 else None
            self.rec.append((board_bits(b), b.turn, [move_index(m) for m in r.moves], pi,
                             q if b.turn == chess.WHITE else -q, b.fen()))
        if len(b.move_stack) < a.temp_plies:
            p = visits / visits.sum()
            i = int(self.rng.choice(len(p), p=p))
        else:
            i = int(np.argmax(visits))

        mover = b.turn
        self.low[mover] = self.low[mover] + 1 if q < a.resign else 0
        if self.low[mover] >= 2:
            if not self.no_resign:
                self.finish("0-1" if mover == chess.WHITE else "1-0", "resign")
                return
            if self.would_resign is None:
                self.would_resign = mover

        s.advance(i)
        o = b.outcome(claim_draw=False)
        if o is not None:
            self.finish(o.result(), o.termination.name.lower())
        elif b.is_repetition(3):
            self.finish("1/2-1/2", "threefold")
        elif b.halfmove_clock >= 100:
            self.finish("1/2-1/2", "fifty moves")
        elif len(b.move_stack) >= a.max_plies:
            self.finish("1/2-1/2", "max plies")
        else:
            self.plan()

    def finish(self, result, termination):
        self.result, self.termination = result, termination
        self.search = None             # free the tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--concurrent", type=int, default=128, help="games in flight (two alternating halves)")
    ap.add_argument("--leaves", type=int, default=4, help="leaves per game per network call")
    ap.add_argument("--full-sims", type=int, default=400)
    ap.add_argument("--fast-sims", type=int, default=80)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--c-puct", type=float, default=1.0)
    ap.add_argument("--temp-plies", type=int, default=20)
    ap.add_argument("--resign", type=float, default=-0.95)
    ap.add_argument("--no-resign", type=float, default=0.15)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cudnn.benchmark = False   # batch sizes vary every call
    rng = np.random.default_rng(a.seed)
    ev = Evaluator(load_net(a.net))

    started, next_id = 0, 0
    halves = [[], []]
    for k in range(min(a.concurrent, a.games)):
        halves[k % 2].append(Game(next_id, ev, rng, a))
        next_id += 1
    done, evals, t0 = [], 0, time.time()
    inflight = [None, None]
    h = 0
    last_print = t0
    while any(halves) or any(inflight):
        # queue this half's leaves, then finish the other half's call (the GPU works on both in turn)
        reqs = collect_round([g.search for g in halves[h]], a.leaves)
        inflight[h] = start_round(reqs) if reqs else None
        evals += len(reqs)
        o = 1 - h
        if inflight[o] is not None:
            finish_round(inflight[o])
            inflight[o] = None
        # moves for the half whose results are all in
        keep = []
        for g in halves[o]:
            while g.result is None and g.ready():
                g.move()
            if g.result is None:
                keep.append(g)
            else:
                done.append(g)
                if next_id < a.games:
                    keep.append(Game(next_id, ev, rng, a))
                    next_id += 1
        halves[o] = keep
        h = o
        if time.time() - last_print > 60:
            last_print = time.time()
            el = last_print - t0
            print(f"{len(done)}/{a.games} games, {evals / el:.0f} evals/s, {el / 60:.1f} min", flush=True)

    write(a, done, evals, time.time() - t0)


def write(a, games, evals, seconds):
    rows = [(g, r) for g in games for r in g.rec]
    n = len(rows)
    bits = np.array([r[0] for _, r in rows], dtype=np.uint64).reshape(n, 12)
    legal = np.full((n, MAX_LEGAL), -1, dtype=np.int16)
    pi = np.zeros((n, MAX_LEGAL), dtype=np.float16)
    z = np.empty(n, dtype=np.float32)
    zval = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
    for i, (g, r) in enumerate(rows):
        legal[i, :len(r[2])] = r[2]
        if r[3] is not None:
            pi[i, :len(r[3])] = r[3]
        z[i] = zval[g.result]
    np.savez(os.path.join(a.out, "positions.npz"), board=bits_to_codes(bits),
             turn=np.array([r[1] for _, r in rows], dtype=np.uint8), legal=legal, pi=pi, z=z,
             q=np.array([r[4] for _, r in rows], dtype=np.float32), fen=np.array([r[5] for _, r in rows]))

    with open(os.path.join(a.out, "games.pgn"), "w") as f:
        for g in games:
            pg = chess.pgn.Game.from_board(g.board)
            pg.headers.update({"Event": "Sigma-Go self-play", "Round": str(g.gid), "White": "self", "Black": "self",
                               "Result": g.result, "Termination": g.termination})
            print(pg, file=f, end="\n\n")

    res = [g.result for g in games]
    wr = [g for g in games if g.would_resign is not None]
    wrong = [g for g in wr if g.result != ("0-1" if g.would_resign == chess.WHITE else "1-0")]
    terms = {}
    for g in games:
        terms[g.termination] = terms.get(g.termination, 0) + 1
    stats = {
        "games": len(games), "white_wins": res.count("1-0"), "draws": res.count("1/2-1/2"),
        "black_wins": res.count("0-1"), "terminations": terms,
        "avg_plies": float(np.mean([len(g.board.move_stack) for g in games])),
        "positions": n, "policy_positions": int((pi.sum(1) > 0).sum()),
        "no_resign_games": sum(g.no_resign for g in games), "would_resign": len(wr),
        "would_resign_wrong": len(wrong), "evals": evals, "evals_per_s": round(evals / seconds),
        "minutes": round(seconds / 60, 1),
        "settings": {k: v for k, v in vars(a).items() if k not in ("out",)},
    }
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
