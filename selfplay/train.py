"""Train the network on recent self-play games, anchored with Stockfish-evaluated Lichess positions.

Each batch is (1 - --anchor-frac) self-play positions from the replay window (--games dirs) and
--anchor-frac positions from data/lichess_evals. Policy loss is cross-entropy over the legal moves:
against the search's visit shares for self-play positions that had a full search, against Stockfish's
best move for anchor positions. Value loss is MSE against the game result (self-play) or
tanh(cp / 400) (anchor), both from White's side, weighted by --value-weight.

Model and optimizer state carry over between generations in --state. The trained weights are also
written as a plain state dict to --out (what match.py, eval/run_match.py and the exporters load).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from model import ChessNet  # noqa: E402

dev = "cuda"


def planes(board, turn):
    """uint8 [B,64] codes + [B] turn -> float [B,14,8,8], as dataset.board_to_tensor."""
    b = F.one_hot(board.long(), 13)[..., 1:].permute(0, 2, 1).reshape(-1, 12, 8, 8).float()
    t = turn.float().view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
    return torch.cat([b, t, 1 - t], 1)


def row_losses(model, board, turn, legal, target, value):
    """Per row: policy cross-entropy, whether the row has a policy target, squared value error, and
    whether the network's top legal move is the target's top move."""
    logits, v = model(planes(board, turn))
    legal = legal.long()
    mask = legal >= 0
    g = logits.gather(1, legal.clamp(min=0)).masked_fill(~mask, float("-inf"))
    logp = F.log_softmax(g, 1).masked_fill(~mask, 0.0)
    ce = -(target * logp).sum(1)
    has = (target.sum(1) > 0).float()
    se = (v.view(-1) - value) ** 2
    hit = (g.argmax(1) == target.argmax(1)).float()
    return ce, has, se, hit


def part(ce, has, se, sl):
    h = has[sl]
    return (ce[sl] * h).sum() / h.sum().clamp(min=1), se[sl].mean()


def onehot_target(legal, policy):
    return (legal == policy.view(-1, 1)).float()


def load_games(dirs):
    parts = [np.load(os.path.join(d, "positions.npz")) for d in dirs]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("board", "turn", "legal", "pi", "z")}


class Anchor:
    def __init__(self, path, split):
        self.a = {k: np.load(os.path.join(path, f"{split}_{k}.npy"), mmap_mode="r")
                  for k in ("board", "turn", "policy", "value", "legal")}
        self.n = self.a["turn"].shape[0]

    def batch(self, idx):
        idx = np.sort(idx)
        t = {k: torch.from_numpy(np.ascontiguousarray(v[idx])).to(dev) for k, v in self.a.items()}
        return t["board"], t["turn"], t["legal"], onehot_target(t["legal"].long(), t["policy"].long()), t["value"]


@torch.no_grad()
def evaluate(model, anchor_val, n=20000, bs=1000):
    model.eval()
    ce = mse = top1 = 0.0
    for i in range(0, n, bs):
        c, _, se, hit = row_losses(model, *anchor_val.batch(np.arange(i, min(i + bs, n))))
        ce += c.sum().item(); mse += se.sum().item(); top1 += hit.sum().item()
    model.train()
    return {"anchor_val_ce": ce / n, "anchor_val_mse": mse / n, "anchor_val_top1": top1 / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="trainer state (model + optimizer), created from --init if missing")
    ap.add_argument("--init", help="plain state dict to start from when --state does not exist")
    ap.add_argument("--out", required=True, help="plain state dict of the trained network")
    ap.add_argument("--games", nargs="+", required=True, help="self-play dirs in the replay window")
    ap.add_argument("--anchor", default="data/lichess_evals")
    ap.add_argument("--anchor-frac", type=float, default=0.25)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    model = ChessNet(num_res_blocks=10).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    if os.path.exists(a.state):
        st = torch.load(a.state, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        for g in opt.param_groups:
            g["lr"] = a.lr
        trained_steps = st.get("steps", 0)
    else:
        model.load_state_dict(torch.load(a.init, map_location=dev))
        trained_steps = 0
    model.train()

    sp = load_games(a.games)
    sp_t = {k: torch.from_numpy(v).to(dev) for k, v in sp.items()}
    n_sp = sp["z"].shape[0]
    anc, anc_val = Anchor(a.anchor, "train"), Anchor(a.anchor, "val")
    n_anc = int(round(a.batch * a.anchor_frac))
    n_self = a.batch - n_anc
    before = evaluate(model, anc_val)
    print(f"{n_sp} self-play positions ({int((sp['pi'].sum(1) > 0).sum())} with policy targets) from "
          f"{len(a.games)} generations; {a.steps} steps of {n_self} self-play + {n_anc} anchor", flush=True)

    t0 = time.time()
    sums = np.zeros(6)
    for step in range(a.steps):
        i = torch.from_numpy(rng.integers(0, n_sp, n_self)).to(dev)
        own = (sp_t["board"][i], sp_t["turn"][i], sp_t["legal"][i], sp_t["pi"][i].float(), sp_t["z"][i])
        ref = anc.batch(rng.integers(0, anc.n, n_anc))
        ce, has, se, _ = row_losses(model, *[torch.cat([x, y.to(x.dtype)]) for x, y in zip(own, ref)])
        s_ce, s_mse = part(ce, has, se, slice(0, n_self))
        a_ce, a_mse = part(ce, has, se, slice(n_self, None))
        # weight the two parts by their share of the batch
        f = a.anchor_frac
        loss = (1 - f) * (s_ce + a.value_weight * s_mse) + f * (a_ce + a.value_weight * a_mse)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sums += [s_ce.item(), s_mse.item(), a_ce.item(), a_mse.item(), loss.item(), 1]
        if (step + 1) % 100 == 0:
            m = sums / sums[5]
            print(f"step {step + 1}/{a.steps} self ce {m[0]:.3f} mse {m[1]:.3f} | anchor ce {m[2]:.3f} "
                  f"mse {m[3]:.3f} | {(step + 1) * a.batch / (time.time() - t0):.0f} pos/s", flush=True)

    m = sums / max(sums[5], 1)
    after = evaluate(model, anc_val)
    torch.save(model.state_dict(), a.out)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "steps": trained_steps + a.steps}, a.state)
    res = {"steps": a.steps, "total_steps": trained_steps + a.steps, "self_positions": n_sp,
           "train_self_ce": m[0], "train_self_mse": m[1], "train_anchor_ce": m[2], "train_anchor_mse": m[3],
           "before": before, "after": after, "minutes": round((time.time() - t0) / 60, 1),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(json.dumps(res), flush=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
