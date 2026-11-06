"""Why did self-play make the network worse? Retrain generation 1 from the same start and data in
several ways and watch the Stockfish-labelled validation set as it trains.

Variants (same steps, batch and data as the loop's first generation):
  control   what the loop did: AdamW at a constant 1e-4 from step 0, value target = game result z
  low_lr    2e-5 with a 100-step warmup, value target z
  low_lr_q  2e-5 with warmup, value target = the search's root value q (soft, like tanh(cp/400))
  anchor    2e-5 with warmup, anchor data only (no self-play): what the optimizer change alone does

    python selfplay/diagnose_gen1.py --init runs/ft2/best.pth --games runs/sp1/games/gen_001 \\
        --anchor data/lichess_evals_big --steps 495 --out runs/diag_gen1.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)  # first, so "train" is selfplay/train.py, not src/train.py
from model import ChessNet  # noqa: E402
from train import Anchor, evaluate, load_games, part, row_losses  # noqa: E402

dev = "cuda"
VARIANTS = {
    "control": dict(lr=1e-4, warmup=0, value="z", self_frac=0.75),
    "low_lr": dict(lr=2e-5, warmup=100, value="z", self_frac=0.75),
    "low_lr_q": dict(lr=2e-5, warmup=100, value="q", self_frac=0.75),
    "anchor": dict(lr=2e-5, warmup=100, value="z", self_frac=0.0),
}


def run(name, v, a, sp_t, n_sp, anc, anc_val):
    rng = np.random.default_rng(1)
    model = ChessNet(num_res_blocks=10).to(dev)
    model.load_state_dict(torch.load(a.init, map_location=dev))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=v["lr"], weight_decay=1e-4)
    n_self = int(round(a.batch * v["self_frac"]))
    n_anc = a.batch - n_self
    curve = [{"step": 0, **evaluate(model, anc_val, n=a.val_n)}]
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = v["lr"] * min(1.0, (step + 1) / v["warmup"]) if v["warmup"] else v["lr"]
        parts = []
        if n_self:
            i = torch.from_numpy(rng.integers(0, n_sp, n_self)).to(dev)
            parts.append((sp_t["board"][i], sp_t["turn"][i], sp_t["legal"][i], sp_t["pi"][i].float(), sp_t[v["value"]][i]))
        parts.append(anc.batch(rng.integers(0, anc.n, n_anc)))
        batch = [torch.cat([p[k].to(parts[0][k].dtype) for p in parts]) for k in range(5)]
        ce, has, se, _ = row_losses(model, *batch)
        a_ce, a_mse = part(ce, has, se, slice(n_self, None))
        if n_self:
            s_ce, s_mse = part(ce, has, se, slice(0, n_self))
            f = n_anc / a.batch
            loss = (1 - f) * (s_ce + 1.5 * s_mse) + f * (a_ce + 1.5 * a_mse)
        else:
            loss = a_ce + 1.5 * a_mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step + 1 == a.steps:
            curve.append({"step": step + 1, **evaluate(model, anc_val, n=a.val_n)})
            c = curve[-1]
            print(f"{name:9s} step {step + 1:4d}  ce {c['anchor_val_ce']:.3f}  mse {c['anchor_val_mse']:.4f}  "
                  f"top1 {c['anchor_val_top1']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--anchor", default="data/lichess_evals_big")
    ap.add_argument("--steps", type=int, default=495)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--val-n", type=int, default=20000)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    sp = load_games([a.games])
    sp["q"] = np.concatenate([np.load(os.path.join(a.games, "positions.npz"))["q"]])
    sp_t = {k: torch.from_numpy(v).to(dev) for k, v in sp.items()}
    n_sp = sp["z"].shape[0]
    print(f"{n_sp} self-play positions, {int((sp['pi'].sum(1) > 0).sum())} with policy targets; "
          f"z/q correlation {np.corrcoef(sp['z'], sp['q'])[0, 1]:.2f}", flush=True)
    anc, anc_val = Anchor(a.anchor, "train"), Anchor(a.anchor, "val")
    res = {}
    for name in a.variants.split(","):
        res[name] = run(name, VARIANTS[name], a, sp_t, n_sp, anc, anc_val)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
