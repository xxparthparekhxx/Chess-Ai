"""Net against net: each opening is played twice, colours swapped, all games searched at once.

Every move gets a fresh search of --sims simulations, no noise, most-visited move. A side resigns after
two of its own moves in a row below --resign when the other side's last root value agrees (above
-resign). Draws by rule are claimed; a game reaching --max-plies is drawn.

  python selfplay/match.py --a runs/x/cand.pth --b runs/x/best.pth --out runs/x/gate_0003

Writes <out>.json (score of A, W/D/L, Elo difference with a 95% interval) and <out>.pgn.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Evaluator, Search, collect_round, finish_round, load_net, start_round  # noqa: E402

OPENINGS = os.path.join(HERE, "gate_openings.json")
STOCKFISH = os.environ.get("STOCKFISH", os.path.expanduser("~/.local/share/stockfish/stockfish"))


def make_openings(n, seed=777):
    """n positions of 4-6 random plies that full-strength Stockfish (20k nodes) scores within 80 cp,
    none of them in eval/openings.json (the Stockfish ladder's set)."""
    with open(os.path.join(os.path.dirname(HERE), "eval", "openings.json")) as f:
        ladder = {o["fen"] for o in json.load(f)}
    rng = random.Random(seed)
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    sf.configure({"Threads": 1, "Hash": 16})
    out, seen = [], set(ladder)
    try:
        while len(out) < n:
            b = chess.Board()
            for _ in range(rng.choice([4, 5, 6])):
                b.push(rng.choice(list(b.legal_moves)))
            if b.fen() in seen or b.is_game_over():
                continue
            seen.add(b.fen())
            cp = sf.analyse(b, chess.engine.Limit(nodes=20_000))["score"].white().score(mate_score=10000)
            if abs(cp) <= 80:
                out.append({"fen": b.fen(), "moves": [m.uci() for m in b.move_stack], "cp": cp})
    finally:
        sf.quit()
    return out


def load_openings(n):
    data = []
    if os.path.exists(OPENINGS):
        with open(OPENINGS) as f:
            data = json.load(f)
    if len(data) < n:
        data = make_openings(n)
        with open(OPENINGS, "w") as f:
            json.dump(data, f, indent=1)
    return data[:n]


class MatchGame:
    def __init__(self, gid, opening, a_white, evs):
        self.gid, self.opening, self.a_white = gid, opening, a_white
        self.board = chess.Board(opening["fen"])
        self.evs = evs                      # {chess.WHITE: Evaluator, chess.BLACK: Evaluator}
        self.search = Search(self.board, evs[self.board.turn])
        self.low = {chess.WHITE: 0, chess.BLACK: 0}
        self.last_q = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        self.moves = []
        self.result = self.termination = None

    def move(self, a):
        s, b = self.search, self.board
        r = s.root
        q = s.root_q()
        mover = b.turn
        self.last_q[mover] = q
        self.low[mover] = self.low[mover] + 1 if q < a.resign else 0
        if self.low[mover] >= 2 and self.last_q[not mover] > -a.resign:
            self.result = "0-1" if mover == chess.WHITE else "1-0"
            self.termination = "resign"
            return
        m = r.moves[int(np.argmax(r.N))]
        b.push(m)
        self.moves.append(m.uci())
        o = b.outcome(claim_draw=False)
        if o is not None:
            self.result, self.termination = o.result(), o.termination.name.lower()
        elif b.is_repetition(3):
            self.result, self.termination = "1/2-1/2", "threefold"
        elif b.halfmove_clock >= 100:
            self.result, self.termination = "1/2-1/2", "fifty moves"
        elif len(self.moves) >= a.max_plies:
            self.result, self.termination = "1/2-1/2", "max plies"
        else:
            s.reset_tree()
            s.ev = self.evs[b.turn]

    def score_a(self):
        w = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[self.result]
        return w if self.a_white else 1 - w


def elo(score):
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="candidate network")
    ap.add_argument("--b", required=True, help="reference network")
    ap.add_argument("--out", required=True, help="path prefix for .json and .pgn")
    ap.add_argument("--openings", type=int, default=32)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--leaves", type=int, default=4)
    ap.add_argument("--resign", type=float, default=-0.95)
    ap.add_argument("--max-plies", type=int, default=300)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False   # batch sizes vary every call
    ev_a, ev_b = Evaluator(load_net(a.a)), Evaluator(load_net(a.b))
    ops = load_openings(a.openings)

    games = []
    for i, op in enumerate(ops):
        for a_white in (True, False):
            evs = {chess.WHITE: ev_a if a_white else ev_b, chess.BLACK: ev_b if a_white else ev_a}
            games.append(MatchGame(len(games), op, a_white, evs))
    halves = [games[0::2], games[1::2]]
    inflight = [None, None]
    h, t0 = 0, time.time()
    while any(halves) or any(inflight):
        reqs = collect_round([g.search for g in halves[h]], a.leaves)
        inflight[h] = start_round(reqs) if reqs else None
        o = 1 - h
        if inflight[o] is not None:
            finish_round(inflight[o])
            inflight[o] = None
        keep = []
        for g in halves[o]:
            r = g.search.root
            if not r.pending and (r.n >= a.sims or len(r.moves) == 1):
                g.move(a)
            if g.result is None:
                keep.append(g)
        halves[o] = keep
        h = o

    w = sum(g.score_a() == 1 for g in games)
    d = sum(g.score_a() == 0.5 for g in games)
    l = len(games) - w - d
    n = len(games)
    score = (w + 0.5 * d) / n
    # standard error of the mean per-game score, then a normal 95% interval
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * score ** 2) / n
    se = math.sqrt(var / n)
    terms = {}
    for g in games:
        terms[g.termination] = terms.get(g.termination, 0) + 1
    res = {"a": a.a, "b": a.b, "games": n, "wins": w, "draws": d, "losses": l, "score": score,
           "score_ci95": [score - 1.96 * se, score + 1.96 * se], "elo": elo(score),
           "elo_ci95": [elo(score - 1.96 * se), elo(score + 1.96 * se)], "terminations": terms,
           "avg_plies": float(np.mean([len(g.moves) for g in games])), "sims": a.sims,
           "minutes": round((time.time() - t0) / 60, 1)}
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    with open(a.out + ".pgn", "w") as f:
        for g in games:
            pg = chess.pgn.Game()
            pg.setup(chess.Board(g.opening["fen"]))
            node = pg
            for u in g.moves:
                node = node.add_variation(chess.Move.from_uci(u))
            na, nb = os.path.basename(a.a), os.path.basename(a.b)
            pg.headers.update({"Event": "Sigma-Go net match", "Round": str(g.gid),
                               "White": na if g.a_white else nb, "Black": nb if g.a_white else na,
                               "Result": g.result, "Termination": g.termination})
            print(pg, file=f, end="\n\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
