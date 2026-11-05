"""Checks of the batched engine against src/ (run: .venv/bin/python selfplay/tests/test_engine.py)."""
import os, random, sys
import chess, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import (Evaluator, Search, bits_to_codes, board_bits, collect_round, finish_round, load_net,
                    move_index, start_round)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import MOVE_TO_INT, board_to_tensor
from mcts import MCTS

net = load_net(os.path.join(ROOT, "models/chess_net.pth"))
ev = Evaluator(net)
rng = random.Random(1)

# random positions, including promotions and black to move
boards = []
while len(boards) < 64:
    b = chess.Board()
    for _ in range(rng.randrange(1, 120)):
        ms = list(b.legal_moves)
        if not ms: break
        b.push(rng.choice(ms))
    if not b.is_game_over():
        boards.append(b)

# 1. move index table == MOVE_TO_INT on uci strings
for b in boards:
    for m in b.legal_moves:
        assert move_index(m) == MOVE_TO_INT[m.uci()], m

# 2. codes == prep_lichess_evals layout, planes == board_to_tensor
bits = np.array([board_bits(b) for b in boards], dtype=np.uint64)
codes = bits_to_codes(bits)
for b, c in zip(boards, codes):
    t = board_to_tensor(b)
    ref = np.zeros(64, np.uint8)
    for sq in range(64):
        ch = np.nonzero(t[:12, sq // 8, sq % 8])[0]
        if len(ch): ref[sq] = ch[0] + 1
    assert (ref == c).all()

# 3. priors and values == src/mcts.py evaluate_batch (renormalized over legal moves)
old = MCTS(net)
pol, val = old.evaluate_batch(boards)
legal = [[move_index(m) for m in b.legal_moves] for b in boards]
L = np.full((len(boards), max(map(len, legal))), -1, np.int64)
for i, l in enumerate(legal): L[i, :len(l)] = l
pri, v = Evaluator.fetch(ev.submit(bits, np.array([b.turn for b in boards]), L))
for i, b in enumerate(boards):
    ref = pol[i][legal[i]]; ref = ref / ref.sum()
    assert np.abs(pri[i][:len(legal[i])] - ref).max() < 1e-4, i
    stm = v[i] if b.turn == chess.WHITE else -v[i]
    assert abs(stm - val[i]) < 1e-4, (stm, val[i])
print("encoding, priors, values match src/")

# 4. searches: mate in one found, visit bookkeeping consistent, no virtual loss left
def run(board, sims, leaves=4):
    s = Search(board, ev)
    while s.root.pending or s.root.n < sims:
        reqs = collect_round([s], leaves)
        if reqs: finish_round(start_round(reqs))
    return s

for fen, best in [("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8"),
                  ("3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "d8d1"),
                  ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", None)]:
    b = chess.Board(fen)
    stack = len(b.move_stack)
    s = run(b, 400)
    assert len(b.move_stack) == stack and b.fen() == chess.Board(fen).fen()
    r = s.root
    assert abs(r.N.sum() - (r.n - 1)) < 1e-9, (r.N.sum(), r.n)
    def check(node):
        if node is None or node.N is None: return
        for i, c in enumerate(node.children):
            if c is not None and c.N is not None:
                assert abs(c.N.sum() - (node.N[i] - 1)) < 1e-9 and c.n == node.N[i], (c.N.sum(), node.N[i], c.n)
            assert -1 - 1e-9 <= node.W[i] / max(node.N[i], 1) <= 1 + 1e-9
            check(c)
    check(r)
    top = r.moves[int(np.argmax(r.N))]
    q = s.root_q()
    print(fen, "->", top.uci(), f"q={q:+.3f}", f"visits={int(r.N.max())}/{int(r.N.sum())}")
    if best: assert top.uci() == best and q > 0.99
    else: assert b.is_checkmate() is False and q > 0.5
print("search checks passed")
