#!/usr/bin/env python
"""Dump test fixtures from the Python engine for the TypeScript port (webgpu/).

For about 50 varied positions it records, straight from src/:
  - the encoded planes (dataset.board_to_tensor)
  - the legal moves in python-chess order and their move-map indices
  - the PyTorch policy logits and value (models/chess_net.pth, eval mode, CPU)
  - for a subset: the move the Python MCTS picks with Dirichlet noise off,
    temperature 0 and a fixed simulation count, plus the root visit counts
It also records the whole move table, the legal-move order for a few hundred
random-playout positions, and terminal-state results (MCTS.get_game_result).

Noise is switched off by a subclass of MCTS defined here; src/mcts.py is not
modified. Runs on CPU.

    .venv/bin/python scripts/make_fixtures.py
    .venv/bin/python scripts/make_fixtures.py --sims 200 800 --out webgpu/test/fixtures/fixtures.json
"""
import argparse
import base64
import hashlib
import json
import os
import random
import sys
import time

# The GPU is usually busy with other work; fixtures are generated on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import MODEL_PATH, NUM_RES_BLOCKS  # noqa: E402
from dataset import ACTION_SIZE, INT_TO_MOVE, MOVE_TO_INT, board_to_tensor  # noqa: E402
from mcts import MCTS  # noqa: E402
from model import ChessNet  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "webgpu", "test", "fixtures", "fixtures.json")

# (name, category, kind, spec). kind "fen" -> spec is a FEN; kind "san" -> spec is a
# move list played from the start position.
POSITIONS = [
    # --- openings ---
    ("start position", "opening", "fen", chess.STARTING_FEN),
    ("1.e4", "opening", "san", "e4"),
    ("Ruy Lopez, white has castled", "opening", "san", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O"),
    ("Sicilian Najdorf", "opening", "san", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    ("Queen's Gambit Declined", "opening", "san", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3"),
    ("Italian, white in check", "opening", "san", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 cxd4 Bb4+"),
    ("French Advance", "opening", "san", "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3"),
    ("King's Indian", "opening", "san", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5"),
    ("English", "opening", "san", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3"),
    ("Scandinavian", "opening", "san", "e4 d5 exd5 Qxd5 Nc3 Qa5"),
    # --- middlegames and endgames ---
    ("Kiwipete", "middlegame", "fen", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("perft pos 4 (promotions both sides)", "middlegame", "fen", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("perft pos 4 mirrored, black to move", "middlegame", "fen", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"),
    ("perft pos 5", "middlegame", "fen", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
    ("perft pos 6", "middlegame", "fen", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("closed centre", "middlegame", "fen", "r2q1rk1/ppp2ppp/2n1bn2/3pp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8"),
    ("perft pos 3 (pins, ep)", "endgame", "fen", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("K+R vs K", "endgame", "fen", "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"),
    ("K+Q vs K, black to move", "endgame", "fen", "8/8/8/8/3k4/8/4Q3/4K3 b - - 0 1"),
    ("K+P vs K", "endgame", "fen", "8/8/8/3k4/8/3K4/3P4/8 w - - 0 1"),
    ("rook endgame", "endgame", "fen", "8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 1"),
    ("opposite-coloured bishops", "endgame", "fen", "8/5k2/4b3/8/3B4/4K3/5P2/8 w - - 0 1"),
    # --- promotions ---
    ("white pawn on a7", "promotion", "fen", "8/P6k/8/8/8/8/8/K7 w - - 0 1"),
    ("black pawn on a2, black to move", "promotion", "fen", "8/8/8/8/8/6K1/p7/7k b - - 0 1"),
    ("white promotion with two captures", "promotion", "fen", "r1n5/1P6/8/8/8/8/k7/7K w - - 0 1"),
    ("black promotion with two captures", "promotion", "fen", "4k3/8/8/8/8/8/1p6/R1N1K3 b - - 0 1"),
    # --- castling rights ---
    ("castling: both sides, all rights", "castling", "fen", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("castling: black to move, all rights", "castling", "fen", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
    ("castling: white K-side, black Q-side only", "castling", "fen", "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"),
    ("castling: no rights", "castling", "fen", "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"),
    ("castling: f1 attacked", "castling", "fen", "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1"),
    ("castling: king in check", "castling", "fen", "r3k2r/8/8/8/4q3/8/8/R3K2R w KQkq - 0 1"),
    # --- en passant ---
    ("ep f6, white to move", "en passant", "fen", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("ep d6 after e4 a6 e5 d5", "en passant", "san", "e4 a6 e5 d5"),
    ("ep e3 for black", "en passant", "san", "a3 d5 a4 d4 e4"),
    ("ep illegal because of a rank pin", "en passant", "fen", "8/8/8/KPp4r/8/8/8/7k w - c6 0 1"),
    ("ep that captures the checking pawn", "en passant", "fen", "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1"),
    # --- near mate / checks ---
    ("back-rank mate in 1", "near-mate", "fen", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("Qxf7# available", "near-mate", "san", "e4 e5 Qh5 Nc6 Bc4 Nf6"),
    ("black mate in 1 (Ra1#)", "near-mate", "fen", "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"),
    ("mate or stalemate", "near-mate", "fen", "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"),
    ("smothered mate idea", "near-mate", "fen", "6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1"),
    ("mating attack", "near-mate", "fen", "r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1"),
    ("single check, king can capture", "near-mate", "fen", "4k3/8/8/8/8/2n5/3q4/4K3 w - - 0 1"),
    ("double check", "near-mate", "fen", "4k3/8/8/8/8/3n4/8/r3K3 w - - 0 1"),
    ("pinned bishop", "near-mate", "fen", "4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1"),
]

# Positions where the game is over, plus close calls. (fen, label)
TERMINAL_FENS = [
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "fool's mate, white is mated"),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "stalemate"),
    ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", "K vs K"),
    ("8/8/4k3/8/8/4KB2/8/8 w - - 0 1", "K+B vs K"),
    ("8/8/4k3/8/8/4KN2/8/8 b - - 0 1", "K+N vs K"),
    ("8/8/4k3/2b5/8/4K3/8/6B1 w - - 0 1", "bishops on the same colour"),
    ("8/8/4k3/2b5/8/4K3/8/5B2 w - - 0 1", "bishops on opposite colours"),
    ("8/8/4k3/2n5/8/4K3/8/5N2 w - - 0 1", "K+N vs K+N"),
    ("8/8/4k3/8/8/4K3/8/5NN1 w - - 0 1", "K+N+N vs K"),
    ("8/8/4k3/8/8/4K3/8/R7 w - - 150 100", "75-move rule reached"),
    ("8/8/4k3/8/8/4K3/8/R7 w - - 149 100", "75-move rule one ply short"),
    ("8/8/4k3/8/8/4K3/8/R7 w - - 99 100", "50-move claim only"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 150 80", "75-move rule, mate in 1 available"),
    ("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 150 80", "checkmate with clock at 150"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "not terminal"),
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FixtureMCTS(MCTS):
    """The real MCTS with root noise switched off and the root kept for inspection."""

    def add_dirichlet_noise(self, node):  # deterministic mode
        return

    def select_best_move(self, root, board, temperature=0.0):
        self.last_root = root
        return super().select_best_move(root, board, temperature=temperature)


def random_playout_positions(rng, games, max_plies):
    """Yield (fen) for positions along random games, including their last position."""
    for _ in range(games):
        board = chess.Board()
        plies = rng.randint(1, max_plies)
        for _ in range(plies):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            yield board.fen()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ckpt", default=MODEL_PATH, help="PyTorch checkpoint (default: models/chess_net.pth)")
    ap.add_argument("--sims", type=int, nargs="+", default=[200, 800], help="MCTS simulation counts to record")
    ap.add_argument("--random-games", type=int, default=120, help="random playouts used for order/terminal cases")
    ap.add_argument("--no-mcts", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS)
    state = torch.load(args.ckpt, map_location="cpu")
    if "model" in state:  # a trainer state (model + optimizer)
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    # --- move table ---
    labels = [INT_TO_MOVE[i] for i in range(ACTION_SIZE)]
    assert all(MOVE_TO_INT[m] == i for i, m in enumerate(labels))

    # --- main positions ---
    positions = []
    for name, category, kind, spec in POSITIONS:
        board = chess.Board()
        if kind == "san":
            for san in spec.split():
                board.push_san(san)
        else:
            board = chess.Board(spec)
        assert board.is_valid(), f"invalid position: {name}: {board.fen()} ({board.status()})"
        positions.append((name, category, board))

    rng = random.Random(20260530)
    for i in range(8):
        board = chess.Board()
        for _ in range(rng.randint(8, 60)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            board.pop()
        positions.append((f"random playout {i}", "random", board))

    batch = np.stack([board_to_tensor(b) for _, _, b in positions])
    policy_logits, values = model(torch.from_numpy(batch))
    policy_logits = policy_logits.numpy().astype("<f4")
    values = values.numpy().reshape(-1)

    entries = []
    for i, (name, category, board) in enumerate(positions):
        legal = list(board.legal_moves)
        entries.append(
            {
                "name": name,
                "category": category,
                "fen": board.fen(),
                "turn": "w" if board.turn == chess.WHITE else "b",
                "planes": [int(v) for v in batch[i].reshape(-1)],
                "legal_uci": [m.uci() for m in legal],
                "legal_idx": [MOVE_TO_INT[m.uci()] for m in legal],
                "logits_f32_b64": base64.b64encode(policy_logits[i].tobytes()).decode("ascii"),
                "value": float(values[i]),
            }
        )

    # --- MCTS decisions ---
    runs = []
    if not args.no_mcts:
        for sims in args.sims:
            print(f"MCTS fixtures at {sims} simulations (batch 8, noise off, temperature 0)")
            run = {"simulations": sims, "batch_size": 8, "c_puct": 1.0, "results": []}
            t0 = time.time()
            for i, (name, category, board) in enumerate(positions):
                if len(entries[i]["legal_uci"]) < 2 or board.is_game_over():
                    continue
                mcts = FixtureMCTS(model, num_simulations=sims, c_puct=1.0, batch_size=8)
                uci, value = mcts.search(board.copy(), temperature=0.0)
                visits = {INT_TO_MOVE[a]: int(n.visit_count) for a, n in mcts.last_root.children.items()}
                run["results"].append({"index": i, "fen": board.fen(), "move": uci, "value": float(value), "visits": visits})
            print(f"  {len(run['results'])} positions in {time.time()-t0:.1f}s")
            runs.append(run)

    # --- legal-move order and terminal results on random playouts ---
    prng = random.Random(777)
    order_fens = []
    seen = set()
    for fen in random_playout_positions(prng, args.random_games, 200):
        if fen not in seen:
            seen.add(fen)
            order_fens.append(fen)
    prng.shuffle(order_fens)
    order_fens = order_fens[:400]
    order_cases = []
    for fen in order_fens:
        board = chess.Board(fen)
        order_cases.append({"fen": fen, "legal_uci": [m.uci() for m in board.legal_moves]})

    checker = MCTS(model)  # only used for get_game_result
    terminal_cases = []
    for fen, label in TERMINAL_FENS:
        board = chess.Board(fen)
        value, terminal = checker.get_game_result(board)
        terminal_cases.append({"fen": fen, "label": label, "terminal": bool(terminal), "value": int(value)})
    for fen in order_fens:
        board = chess.Board(fen)
        value, terminal = checker.get_game_result(board)
        terminal_cases.append({"fen": fen, "label": "random playout", "terminal": bool(terminal), "value": int(value)})
    # Final positions of random games, so checkmates and draws by material show up.
    final_prng = random.Random(4242)
    finals = 0
    for _ in range(400):
        board = chess.Board()
        for _ in range(300):
            if board.is_game_over():
                break
            board.push(final_prng.choice(list(board.legal_moves)))
        if board.is_game_over():
            value, terminal = checker.get_game_result(board)
            terminal_cases.append({"fen": board.fen(), "label": "random game end", "terminal": True, "value": int(value)})
            finals += 1
    print(f"order cases: {len(order_cases)}, terminal cases: {len(terminal_cases)} ({finals} game endings)")

    fixtures = {
        "meta": {
            "model_sha256": sha256_of(args.ckpt),
            "torch": torch.__version__,
            "python_chess": chess.__version__,
            "numpy": np.__version__,
            "action_size": ACTION_SIZE,
            "planes_shape": [14, 8, 8],
            "logits_encoding": "base64 of little-endian float32[4544]",
            "mcts_note": "FixtureMCTS from scripts/make_fixtures.py: src/mcts.py with Dirichlet noise disabled, temperature 0, CPU",
        },
        "move_labels": labels,
        "positions": entries,
        "mcts_runs": runs,
        "order_cases": order_cases,
        "terminal_cases": terminal_cases,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fixtures, f, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB), {len(entries)} positions")


if __name__ == "__main__":
    main()
