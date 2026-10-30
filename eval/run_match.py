"""Play the Python engine (src/mcts.py MCTS on the GPU) against Stockfish.

Stockfish is only a test opponent here, run with UCI_LimitStrength at a few UCI_Elo levels. It is never
shipped. Settings are fixed so a second run with another network is a like-for-like re-measurement:
800 simulations per move, temperature 0, no Dirichlet noise, 8 leaves per network call (the Python
engine's default), Stockfish Threads=1 and a fixed node budget per move, colours alternated, and every
opening played once with each colour.

Usage:
  .venv/bin/python eval/run_match.py --ckpt models/chess_net.pth --tag base --levels 1320 1600 2000 \
      --pairs 20 --workers 3

Results go to eval/results/<tag>/: games.pgn, games.jsonl (one line per finished game, resumable) and
summary.json. Re-running the same command skips games already in games.jsonl.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

STOCKFISH = os.environ.get("STOCKFISH", os.path.expanduser("~/.local/share/stockfish/stockfish"))
OPENINGS_FILE = os.path.join(ROOT, "eval", "openings.json")

SIMS = 800
BATCH = 8
SF_NODES = 100_000          # per move for the limited-strength opponent
JUDGE_NODES = 20_000        # full-strength evaluation used for adjudication
ADJ_CP = 1000               # |eval| beyond this, held for ADJ_PLIES consecutive plies, ends the game
ADJ_PLIES = 6
MAX_PLIES = 300             # then drawn


def make_openings(n, seed=20261030):
    """n opening positions: 2-4 random legal plies from the start, kept only if full-strength
    Stockfish thinks they are roughly level (|eval| <= 100 cp), so no side starts lost."""
    rng = random.Random(seed)
    judge = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    judge.configure({"Threads": 1, "Hash": 16})
    out, seen = [], set()
    try:
        while len(out) < n:
            b = chess.Board()
            for _ in range(rng.choice([2, 3, 4])):
                b.push(rng.choice(list(b.legal_moves)))
            fen = b.fen()
            if fen in seen or b.is_game_over():
                continue
            seen.add(fen)
            cp = judge.analyse(b, chess.engine.Limit(nodes=JUDGE_NODES))["score"].white().score(mate_score=10000)
            if abs(cp) <= 100:
                out.append({"fen": fen, "moves": [m.uci() for m in b.move_stack]})
    finally:
        judge.quit()
    return out


def load_openings(n):
    if os.path.exists(OPENINGS_FILE):
        with open(OPENINGS_FILE) as f:
            data = json.load(f)
        if len(data) >= n:
            return data[:n]
    data = make_openings(n)
    os.makedirs(os.path.dirname(OPENINGS_FILE), exist_ok=True)
    with open(OPENINGS_FILE, "w") as f:
        json.dump(data, f, indent=1)
    return data


# ---------------------------------------------------------------- worker side
_W = {}


def _init_worker(ckpt):
    import torch
    from config import DEVICE, NUM_RES_BLOCKS
    from mcts import MCTS
    from model import ChessNet

    class NoNoiseMCTS(MCTS):
        def add_dirichlet_noise(self, node):
            pass

    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    _W["mcts"] = NoNoiseMCTS(model, num_simulations=SIMS, batch_size=BATCH)
    _W["judge"] = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    _W["judge"].configure({"Threads": 1, "Hash": 16})
    _W["opp"] = chess.engine.SimpleEngine.popen_uci(STOCKFISH)


def _play(task):
    level, oi, ours_white, opening = task["level"], task["opening"], task["ours_white"], task["fen"]
    mcts, judge, opp = _W["mcts"], _W["judge"], _W["opp"]
    opp.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": level})
    opp.protocol.send_line("ucinewgame")
    judge.protocol.send_line("ucinewgame")

    board = chess.Board(opening)
    start_fen = board.fen()
    ours = chess.WHITE if ours_white else chess.BLACK
    moves, streak, streak_sign = [], 0, 0
    result, term = None, None
    t0 = time.time()
    while True:
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            term = "rules"
            break
        if len(board.move_stack) + len(moves) >= MAX_PLIES:
            result, term = "1/2-1/2", "max plies"
            break
        if board.turn == ours:
            uci, _ = mcts.search(board, temperature=0.0)
            move = chess.Move.from_uci(uci)
        else:
            move = opp.play(board, chess.engine.Limit(nodes=SF_NODES)).move
        moves.append(move.uci())
        board.push(move)
        # adjudicate on full-strength eval
        if not board.is_game_over(claim_draw=True):
            cp = judge.analyse(board, chess.engine.Limit(nodes=JUDGE_NODES))["score"].white().score(mate_score=10000)
            sign = 1 if cp >= ADJ_CP else (-1 if cp <= -ADJ_CP else 0)
            streak = streak + 1 if sign != 0 and sign == streak_sign else (1 if sign != 0 else 0)
            streak_sign = sign
            if streak >= ADJ_PLIES:
                result = "1-0" if sign > 0 else "0-1"
                term = f"adjudicated (eval beyond {ADJ_CP} cp for {ADJ_PLIES} plies)"
                break
    score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    if not ours_white:
        score = 1.0 - score
    return {
        "level": level, "opening": oi, "ours_white": ours_white, "start_fen": start_fen,
        "moves": moves, "result": result, "termination": term, "score": score,
        "plies": len(moves), "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------- stats
def wilson(score_sum, n, z=1.96):
    """Wilson interval for a score treated as n Bernoulli-ish trials (draws count 1/2). Slightly
    conservative with many draws."""
    if n == 0:
        return 0.0, 1.0
    p = score_sum / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def elo_diff(p):
    """Elo of us minus Elo of the opponent from expected score p; None at 0 or 1."""
    if p <= 0.0 or p >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / p - 1.0)


def summarise(games, levels):
    rows = []
    for lv in levels:
        g = [x for x in games if x["level"] == lv]
        n = len(g)
        if n == 0:
            continue
        w = sum(1 for x in g if x["score"] == 1.0)
        d = sum(1 for x in g if x["score"] == 0.5)
        l = n - w - d
        s = sum(x["score"] for x in g)
        lo, hi = wilson(s, n)
        rows.append({
            "stockfish_uci_elo": lv, "games": n, "wins": w, "draws": d, "losses": l,
            "score": s / n, "score_ci95": [lo, hi],
            "elo_diff": elo_diff(s / n), "elo_diff_ci95": [elo_diff(lo), elo_diff(hi)],
            "elo_estimate": None if elo_diff(s / n) is None else lv + elo_diff(s / n),
            "adjudicated": sum(1 for x in g if str(x["termination"]).startswith("adjudicated")),
            "max_plies": sum(1 for x in g if x["termination"] == "max plies"),
            "avg_plies": sum(x["plies"] for x in g) / n,
        })
    return rows


def write_pgn(games, path, tag):
    with open(path, "w") as f:
        for i, x in enumerate(sorted(games, key=lambda g: (g["level"], g["opening"], not g["ours_white"]))):
            game = chess.pgn.Game()
            ours, sf = f"Sigma-Go ({tag})", f"Stockfish UCI_Elo {x['level']}"
            game.headers.update({
                "Event": "Sigma-Go vs Stockfish", "Round": str(i + 1),
                "White": ours if x["ours_white"] else sf, "Black": sf if x["ours_white"] else ours,
                "Result": x["result"], "Termination": x["termination"],
                "FEN": x["start_fen"], "SetUp": "1",
            })
            board = chess.Board(x["start_fen"])
            node = game
            for u in x["moves"]:
                m = chess.Move.from_uci(u)
                node = node.add_variation(m)
                board.push(m)
            print(game, file=f, end="\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1320, 1600, 2000])
    ap.add_argument("--pairs", type=int, default=20, help="openings per level; each is played with both colours")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    out = os.path.join(ROOT, "eval", "results", a.tag)
    os.makedirs(out, exist_ok=True)
    jl = os.path.join(out, "games.jsonl")
    done = []
    if os.path.exists(jl):
        with open(jl) as f:
            done = [json.loads(line) for line in f if line.strip()]
    have = {(g["level"], g["opening"], g["ours_white"]) for g in done}

    openings = load_openings(a.pairs)
    tasks = [
        {"level": lv, "opening": oi, "ours_white": cw, "fen": o["fen"]}
        for oi, o in enumerate(openings) for lv in a.levels for cw in (True, False)
        if (lv, oi, cw) not in have
    ]
    # interleave levels so a partial run is balanced
    tasks.sort(key=lambda t: (t["opening"], t["level"], not t["ours_white"]))
    print(f"{len(done)} games done, {len(tasks)} to play, {a.workers} workers", flush=True)

    games = list(done)
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, initializer=_init_worker, initargs=(os.path.abspath(a.ckpt),)) as pool, \
                open(jl, "a") as f:
            for g in pool.imap_unordered(_play, tasks):
                games.append(g)
                f.write(json.dumps(g) + "\n")
                f.flush()
                print(f"[{len(games)}] sf{g['level']} op{g['opening']} {'W' if g['ours_white'] else 'B'} "
                      f"{g['result']} score={g['score']} plies={g['plies']} {g['seconds']}s {g['termination']}",
                      flush=True)

    summary = {
        "tag": a.tag, "checkpoint": os.path.relpath(a.ckpt, ROOT) if os.path.abspath(a.ckpt).startswith(ROOT) else a.ckpt,
        "settings": {
            "simulations": SIMS, "leaves_per_call": BATCH, "temperature": 0, "dirichlet_noise": False, "c_puct": 1.0,
            "stockfish": "Stockfish 19 (official linux x86-64 universal), UCI_LimitStrength, Threads=1, Hash=16",
            "stockfish_nodes_per_move": SF_NODES,
            "adjudication": f"full-strength Stockfish ({JUDGE_NODES} nodes) beyond +/-{ADJ_CP} cp for {ADJ_PLIES} "
                            f"consecutive plies; draw at {MAX_PLIES} plies; rules draws claimed",
            "openings": f"{a.pairs} positions of 2-4 random plies, |eval| <= 100 cp, each played with both colours "
                        f"(eval/openings.json)",
            "ci": "95% Wilson interval on the score, draws counted as half",
        },
        "levels": summarise(games, a.levels),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    write_pgn(games, os.path.join(out, "games.pgn"), a.tag)
    print(json.dumps(summary["levels"], indent=1))


if __name__ == "__main__":
    main()
