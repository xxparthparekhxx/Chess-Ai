"""Batched MCTS for self-play and net-vs-net matches.

Same network, inputs and PUCT rule as src/mcts.py (c_puct 1.0, unvisited moves scored Q = 0, priors =
softmax over the legal moves), but many games are searched at once: every round each game selects a few
leaves (virtual loss keeps them apart), all leaves of all games go to the GPU in one call, then each
leaf is expanded and backed up. src/mcts.py is unchanged; the web app, the WebGPU port and the Stockfish
harness keep using it.

Values: the network outputs White's view (+1 White wins). Inside the tree every value is from the view of
the player who chooses the edge, so W[a] / N[a] is the mover's expected score after edge a.
"""
import math
import os
import sys

import chess
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataset import MOVE_TO_INT  # noqa: E402
from model import ChessNet  # noqa: E402

MAX_LEGAL = 128          # same padding as data/lichess_evals; positions with more legal moves are not recorded
VIRTUAL_LOSS = 1.0

# move index lookup without building UCI strings: key = from * 512 + to * 8 + promotion piece (0 if none)
_IDX = np.full(64 * 64 * 8, -1, dtype=np.int32)
for _uci, _i in MOVE_TO_INT.items():
    _m = chess.Move.from_uci(_uci)
    _IDX[_m.from_square * 512 + _m.to_square * 8 + (_m.promotion or 0)] = _i


def move_index(m):
    return int(_IDX[m.from_square * 512 + m.to_square * 8 + (m.promotion or 0)])


def board_bits(b):
    """12 bitboards in dataset.board_to_tensor's plane order (white P N B R Q K, then black)."""
    w, k = b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK]
    return (b.pawns & w, b.knights & w, b.bishops & w, b.rooks & w, b.queens & w, b.kings & w,
            b.pawns & k, b.knights & k, b.bishops & k, b.rooks & k, b.queens & k, b.kings & k)


def bits_to_codes(bits):
    """[N,12] uint64 bitboards -> [N,64] uint8 codes (0 empty, plane + 1), the data/lichess_evals layout."""
    planes = np.unpackbits(bits.astype("<u8").view(np.uint8).reshape(-1, 12, 8), axis=2, bitorder="little")
    planes = planes.reshape(-1, 12, 64)
    return (planes * np.arange(1, 13, dtype=np.uint8)[None, :, None]).sum(1).astype(np.uint8)


def load_net(path, device="cuda"):
    net = ChessNet(num_res_blocks=10)
    sd = torch.load(path, map_location="cpu")
    if "model" in sd:           # a trainer state (model + optimizer)
        sd = sd["model"]
    net.load_state_dict(sd)
    return net.to(device).eval()


class Evaluator:
    """Runs the network on a batch of leaves. Planes are unpacked on the GPU; priors are a softmax over
    each leaf's legal moves only (what src/mcts.py gets by renormalizing the full softmax)."""

    def __init__(self, net, device="cuda"):
        self.net, self.device = net, device
        self.shifts = torch.arange(64, device=device, dtype=torch.int64)

    @torch.no_grad()
    def submit(self, bits, turns, legal):
        """bits [B,12] uint64, turns [B] bool (True = White), legal [B,L] int64 (-1 pad). Returns a handle.
        Inputs and outputs go through pinned memory and the work is queued without waiting, so the caller
        can prepare the next batch on the CPU while this one runs; fetch() waits for it."""
        d = self.device
        bb = torch.from_numpy(bits.view(np.int64)).pin_memory().to(d, non_blocking=True)
        tt = torch.from_numpy(turns.astype(np.float32)).pin_memory().to(d, non_blocking=True)
        li = torch.from_numpy(legal).pin_memory().to(d, non_blocking=True)
        planes = ((bb.unsqueeze(-1) >> self.shifts) & 1).float().view(-1, 12, 8, 8)
        t = tt.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
        logits, v = self.net(torch.cat([planes, t, 1 - t], 1))
        pri = torch.softmax(logits.gather(1, li.clamp(min=0)).masked_fill(li < 0, float("-inf")), 1)
        pri_h = torch.empty(pri.shape, dtype=pri.dtype, pin_memory=True)
        v_h = torch.empty(v.shape[0], dtype=v.dtype, pin_memory=True)
        pri_h.copy_(pri, non_blocking=True)
        v_h.copy_(v.view(-1), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        return pri_h, v_h, ev

    @staticmethod
    def fetch(handle):
        pri, v, ev = handle
        ev.synchronize()
        return pri.numpy(), v.numpy()


class Node:
    __slots__ = ("moves", "P", "P_raw", "N", "W", "children", "n", "terminal", "pending")

    def __init__(self):
        self.moves = None
        self.P = self.P_raw = self.N = self.W = None
        self.children = None
        self.n = 0
        self.terminal = None    # value for the side to move if the position ends the game (in search terms)
        self.pending = False    # waiting for the network


def leaf_terminal(board):
    """Value for the side to move if the search should stop here, else None. Within the tree a position
    that repeats one already on the board stack counts as a draw (as Leela does)."""
    if board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(2):
        return 0.0
    return None


class Search:
    """The tree for one game. `board` is the game's board; selection pushes moves on it and pops them back."""

    def __init__(self, board, ev=None, c_puct=1.0):
        self.board = board
        self.ev = ev                # Evaluator used for this tree's leaves
        self.c = c_puct
        self.root = Node()
        self.root.pending = True
        self.root_request = True    # the root itself still needs a network call

    # -- one simulation ------------------------------------------------------------------------------
    def collect(self):
        """Walk to a leaf. Returns a request (path, node, bits, turn, legal moves, legal idx) for the network,
        'done' if the simulation finished without the network (terminal), or None on a collision."""
        b, node, path = self.board, self.root, []
        if self.root_request:
            self.root_request = False
            moves = list(b.legal_moves)
            return (path, node, board_bits(b), b.turn, moves, [move_index(m) for m in moves])
        if node.pending:
            return None
        while True:
            N = node.N
            sq = math.sqrt(node.n if node.n > 0 else 1)
            a = int(np.argmax(node.W / np.maximum(N, 1) + self.c * sq * node.P / (1 + N)))
            path.append((node, a))
            N[a] += 1
            node.W[a] -= VIRTUAL_LOSS
            node.n += 1
            b.push(node.moves[a])
            child = node.children[a]
            if child is None:
                child = Node()
                node.children[a] = child
                t = leaf_terminal(b)
                moves = None
                if t is None:
                    moves = list(b.legal_moves)
                    if not moves:
                        t = -1.0 if b.is_check() else 0.0
                if t is not None:
                    child.terminal = t
                    for _ in path:
                        b.pop()
                    self.backup(path, t)
                    return "done"
                child.pending = True
                req = (path, child, board_bits(b), b.turn, moves, [move_index(m) for m in moves])
                for _ in path:
                    b.pop()
                return req
            if child.terminal is not None:
                for _ in path:
                    b.pop()
                self.backup(path, child.terminal)
                return "done"
            if child.pending:
                for _ in path:
                    b.pop()
                for nd, e in path:
                    nd.N[e] -= 1
                    nd.W[e] += VIRTUAL_LOSS
                    nd.n -= 1
                return None
            node = child

    def expand(self, req, priors, value_white):
        path, node, _, turn, moves, _ = req
        k = len(moves)
        node.moves = moves
        node.P = priors[:k].astype(np.float64)
        node.P_raw = node.P.copy()
        node.N = np.zeros(k, dtype=np.float64)
        node.W = np.zeros(k, dtype=np.float64)
        node.children = [None] * k
        node.pending = False
        node.n += 1               # the visit that expanded it
        if path:
            self.backup(path, float(value_white) if turn == chess.WHITE else -float(value_white))

    @staticmethod
    def backup(path, v):
        """v is the leaf value for the side to move at the leaf; the virtual loss taken on the way down
        is replaced by the real value (the visit it added stays as the real visit)."""
        for node, a in reversed(path):
            v = -v
            node.W[a] += VIRTUAL_LOSS + v

    # -- between moves -------------------------------------------------------------------------------
    def add_noise(self, rng, alpha=0.3, eps=0.25):
        r = self.root
        if r.P is not None and len(r.P) > 1:
            r.P = (1 - eps) * r.P_raw + eps * rng.dirichlet([alpha] * len(r.P))

    def clear_noise(self):
        r = self.root
        if r.P is not None:
            r.P = r.P_raw.copy()

    def advance(self, move_i):
        """Play root move move_i on the board and keep its subtree as the new root."""
        r = self.root
        self.board.push(r.moves[move_i])
        child = r.children[move_i]
        if child is None or child.pending or child.terminal is not None or child.moves is None:
            child = Node()
            child.pending = True
            self.root_request = True
        self.root = child

    def reset_tree(self):
        self.root = Node()
        self.root.pending = True
        self.root_request = True

    def root_q(self):
        """Mover's expected score of the most visited root move."""
        r = self.root
        a = int(np.argmax(r.N))
        return float(r.W[a] / max(r.N[a], 1))


def collect_round(searches, leaves_per_search):
    """Up to `leaves_per_search` network requests from each search (fewer after a collision)."""
    reqs = []
    for s in searches:
        for _ in range(leaves_per_search):
            r = s.collect()
            if r is None:
                break
            if r != "done":
                reqs.append((s, r))
    return reqs


def start_round(reqs):
    """Queue the network calls for reqs, one call per evaluator (each Search carries its own `ev`)."""
    by_ev = {}
    for item in reqs:
        by_ev.setdefault(id(item[0].ev), (item[0].ev, []))[1].append(item)
    out = []
    for ev, items in by_ev.values():
        n = len(items)
        bits = np.empty((n, 12), dtype=np.uint64)
        turns = np.empty(n, dtype=bool)
        legal = np.full((n, max(len(r[5]) for _, r in items)), -1, dtype=np.int64)
        for i, (_, r) in enumerate(items):
            bits[i] = r[2]
            turns[i] = r[3]
            legal[i, :len(r[5])] = r[5]
        out.append((items, ev.submit(bits, turns, legal)))
    return out


def finish_round(started):
    for items, handle in started:
        pri, val = Evaluator.fetch(handle)
        for i, (s, r) in enumerate(items):
            s.expand(r, pri[i], val[i])
