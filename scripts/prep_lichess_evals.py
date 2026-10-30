"""Build a fine-tuning set from Lichess's public engine evaluations (CC0).

Source: https://huggingface.co/datasets/Lichess/chess-position-evaluations (license cc0-1.0; the same
data as https://database.lichess.org/#evals). Rows are Stockfish analyses of positions, several rows
(MultiPV lines) per position. Only a few row groups are read, by HTTP range requests; the 44 GB are
never downloaded.

Conventions (checked against Stockfish on sampled positions): `cp` and `mate` are from WHITE's side.

Per position:
  * keep the row with the greatest depth; among those, the best line for the side to move;
  * value  = tanh(cp / K) from White's side (the network's convention), mate -> +/-1;
  * policy = first move of that line, mapped through MOVE_TO_INT, plus the list of legal move indices.

Output (in --out, outside git): {train,val}_{board,turn,policy,value,legal}.npy
  board  uint8  [N, 64]   0 = empty, 1..12 = plane index + 1 (see dataset.board_to_tensor)
  turn   uint8  [N]       1 = White to move
  policy int16  [N]
  value  float32 [N]      from White's side
  legal  int16  [N, 128]  legal move indices, -1 padded
"""
import argparse
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

REPO = "datasets/Lichess/chess-position-evaluations/data/data_{:04d}.parquet"
K = 400.0          # value = tanh(cp / K): 100 cp -> 0.24, 400 cp -> 0.76, 1000 cp -> 0.99
MIN_DEPTH = 16
MAX_LEGAL = 128
PIECE_PLANE = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}   # python-chess piece type -> plane (white); +6 for black


def score_of(cp, mate):
    """Signed centipawn-like score from White's side, mates dominating."""
    if mate is not None:
        return (100000 - abs(mate)) * (1 if mate > 0 else -1)
    return cp


def encode(fen):
    import chess
    board = chess.Board(fen + " 0 1")
    arr = np.zeros(64, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        arr[sq] = PIECE_PLANE[piece.piece_type] + (6 if piece.color == chess.BLACK else 0) + 1
    return board, arr


def process_row_group(args):
    split, shard, rg, out = args
    import chess
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from dataset import MOVE_TO_INT

    with HfFileSystem().open(REPO.format(shard), "rb") as f:
        table = pq.ParquetFile(f).read_row_group(rg)
    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    del table
    rows = [dict(zip(cols, v)) for v in zip(*cols.values())]
    del cols

    best = {}   # fen -> (depth, stm_score, row)
    for r in rows:
        if r["depth"] < MIN_DEPTH or not r["line"]:
            continue
        s = score_of(r["cp"], r["mate"])
        stm = 1 if " w " in r["fen"] else -1
        key = (r["depth"], stm * s)
        cur = best.get(r["fen"])
        if cur is None or key > cur[0]:
            best[r["fen"]] = (key, r)

    out_board, out_turn, out_pol, out_val, out_legal, skipped = [], [], [], [], [], 0
    for fen, (_, r) in best.items():
        board, arr = encode(fen)
        first = r["line"].split()[0]
        pol = MOVE_TO_INT.get(first)
        legal = [MOVE_TO_INT[m.uci()] for m in board.legal_moves if m.uci() in MOVE_TO_INT]
        if pol is None or pol not in legal or len(legal) > MAX_LEGAL or board.is_game_over():
            skipped += 1
            continue
        if r["mate"] is not None:
            v = 1.0 if r["mate"] > 0 else -1.0
        else:
            v = math.tanh(r["cp"] / K)
        pad = np.full(MAX_LEGAL, -1, dtype=np.int16)
        pad[:len(legal)] = legal
        out_board.append(arr)
        out_turn.append(1 if board.turn == chess.WHITE else 0)
        out_pol.append(pol)
        out_val.append(v)
        out_legal.append(pad)
    np.savez(os.path.join(out, f"part_{split}_{shard}_{rg}.npz"), board=np.stack(out_board),
             turn=np.array(out_turn, np.uint8), policy=np.array(out_pol, np.int16),
             value=np.array(out_val, np.float32), legal=np.stack(out_legal))
    return split, len(rows), len(best), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-groups", type=int, default=12, help="row groups (~1M rows each) for training")
    ap.add_argument("--val-groups", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rng = random.Random(a.seed)
    # one row group from each of several different shards, validation from shards not used for training
    shards = list(range(20))
    rng.shuffle(shards)
    val_shards = shards[:a.val_groups]
    train_shards = shards[a.val_groups:a.val_groups + a.train_groups]
    jobs = {"train": [(s, rng.randrange(50)) for s in train_shards], "val": [(s, rng.randrange(50)) for s in val_shards]}
    print(jobs, flush=True)

    import multiprocessing as mp
    parts_dir = os.path.join(a.out, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    flat = [(sp, s, rg, parts_dir) for sp, js in jobs.items() for s, rg in js]
    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for sp, nrows, uniq, skipped in pool.imap_unordered(process_row_group, flat):
            for i, v in enumerate((nrows, uniq, skipped)):
                stats[sp][i] += v
            print("done", sp, nrows, uniq, skipped, flush=True)

    for split in jobs:
        files = sorted(f for f in os.listdir(parts_dir) if f.startswith(f"part_{split}_"))
        parts = [np.load(os.path.join(parts_dir, f)) for f in files]
        names = ["board", "turn", "policy", "value", "legal"]
        cat = [np.concatenate([p[n] for p in parts]) for n in names]
        # drop duplicate positions that appeared in more than one row group
        _, keep = np.unique(np.concatenate([cat[0], cat[1][:, None]], axis=1), axis=0, return_index=True)
        keep.sort()
        cat = [c[keep] for c in cat]
        for n, c in zip(names, cat):
            np.save(os.path.join(a.out, f"{split}_{n}.npy"), c)
        rows, uniq, skipped = stats[split]
        print(f"{split}: {rows} rows read, {uniq} positions, {skipped} skipped, {len(cat[0])} kept; "
              f"value mean {cat[3].mean():+.3f} std {cat[3].std():.3f}", flush=True)


if __name__ == "__main__":
    main()
