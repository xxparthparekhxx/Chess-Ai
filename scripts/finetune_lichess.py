"""Fine-tune ChessNet on engine-evaluated Lichess positions (see prep_lichess_evals.py).

Loss = policy cross-entropy over the legal moves (target: the first move of the best principal
variation) + VALUE_WEIGHT * value MSE (target: tanh(cp / 400) from White's side, as the network
outputs). bf16 autocast, AdamW, linear warm-up then cosine decay. Validation (held-out shard) logs
policy top-1 and value MSE; the starting weights are evaluated first for a baseline.

  .venv/bin/python scripts/finetune_lichess.py --data data/lichess_evals --out models/finetune \
      --init models/chess_net.pth --steps 60000
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import NUM_RES_BLOCKS  # noqa: E402
from dataset import ACTION_SIZE  # noqa: E402
from model import ChessNet  # noqa: E402

VALUE_WEIGHT = 2.0
dev = "cuda"
BF16 = True            # --fp32 turns autocast off (GPUs without bf16, such as Turing, only emulate it)
CHANNELS_LAST = True   # --no-channels-last (slower on some GPUs, e.g. 40% on a GTX 1650 Ti)


NAMES = ("board", "turn", "policy", "value", "legal")


def load_split(path, split, on_gpu=("board", "turn", "policy", "value")):
    """Arrays named in on_gpu are loaded onto the GPU; the rest stay memory-mapped on the host and are
    gathered per batch. legal (256 bytes a row, most of the data) is always on the host, so tens of
    millions of positions fit; on a small GPU pass on_gpu=() to keep everything on the host."""
    out = []
    for n in NAMES:
        a = np.load(os.path.join(path, f"{split}_{n}.npy"), mmap_mode="r")
        out.append(torch.from_numpy(np.ascontiguousarray(a)).to(dev) if n in on_gpu and n != "legal" else a)
    return out


def take(data, idx):
    """Rows idx (a sorted int64 numpy array) of a split from load_split, as GPU tensors."""
    i = torch.from_numpy(idx).to(dev)
    return [d[i] if torch.is_tensor(d) else torch.from_numpy(np.ascontiguousarray(d[idx])).to(dev) for d in data]


def planes(board, turn):
    """uint8 [B,64] codes + [B] turn -> float [B,14,8,8], same layout as dataset.board_to_tensor."""
    b = F.one_hot(board.long(), 13)[..., 1:].permute(0, 2, 1).reshape(-1, 12, 8, 8)
    t = turn.bool()
    tw = t.view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
    return torch.cat([b.float(), tw.float(), (~tw).float()], dim=1)


def legal_mask(legal):
    idx = legal.long()
    idx[idx < 0] = ACTION_SIZE
    m = torch.zeros(idx.shape[0], ACTION_SIZE + 1, dtype=torch.bool, device=legal.device)
    m.scatter_(1, idx, True)
    return m[:, :ACTION_SIZE]


def forward_loss(model, batch):
    board, turn, pol, val, legal = batch
    x = planes(board, turn)
    if CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
        logits, v = model(x)
    logits = logits.float().masked_fill(~legal_mask(legal), float("-inf"))
    logp = F.log_softmax(logits, dim=1)
    ce = -logp.gather(1, pol.long().view(-1, 1)).mean()
    v = v.float().view(-1)
    mse = F.mse_loss(v, val)
    top1 = (logits.argmax(1) == pol.long()).float().mean()
    return ce, mse, top1


@torch.no_grad()
def evaluate(model, data, n, bs=2048):
    model.eval()
    ce = mse = top1 = 0.0
    seen = 0
    for i in range(0, n, bs):
        batch = take(data, np.arange(i, min(i + bs, n)))
        c, m, t = forward_loss(model, batch)
        k = batch[0].shape[0]
        ce += c.item() * k; mse += m.item() * k; top1 += t.item() * k; seen += k
    model.train()
    return {"val_ce": ce / seen, "val_mse": mse / seen, "val_top1": top1 / seen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=os.path.join(ROOT, "models", "chess_net.pth"))
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--val-n", type=int, default=200_000)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop early (checkpoint kept) after this long")
    ap.add_argument("--bench", type=int, default=0, help="only time this many steps and exit")
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast")
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--host-data", action="store_true", help="keep all training arrays on the host (small GPUs)")
    a = ap.parse_args()
    global BF16, CHANNELS_LAST
    BF16, CHANNELS_LAST = not a.fp32, not a.no_channels_last
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    train = load_split(a.data, "train", on_gpu=() if a.host_data else ("board", "turn", "policy", "value"))
    val = load_split(a.data, "val")
    n_train = train[0].shape[0]
    n_val = min(a.val_n, val[0].shape[0])
    print(f"train {n_train} positions, val {n_val} (of {val[0].shape[0]}), {a.steps} steps of {a.batch} "
          f"= {a.steps * a.batch / n_train:.2f} epochs", flush=True)

    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(dev)
    model.load_state_dict(torch.load(a.init, map_location=dev))
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 1e-5 + (a.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * p))

    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def record(d):
        print(json.dumps(d), flush=True)
        log.write(json.dumps(d) + "\n"); log.flush()

    base = evaluate(model, val, n_val)
    train_mean = float(train[3].float().mean()) if torch.is_tensor(train[3]) else float(np.mean(train[3]))
    const = float(((val[3][:n_val] - train_mean) ** 2).mean())
    record({"step": 0, **base, "val_mse_constant_baseline": const})

    best = base["val_ce"] + VALUE_WEIGHT * base["val_mse"]
    perm = np.random.default_rng(0).permutation(n_train)
    pos = 0
    t0 = time.time()
    run_ce = run_mse = 0.0
    for step in range(1, a.steps + 1):
        if pos + a.batch > n_train:
            perm = np.random.default_rng(step).permutation(n_train); pos = 0
        idx = np.sort(perm[pos:pos + a.batch]); pos += a.batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ce, mse, _ = forward_loss(model, take(train, idx))
        loss = ce + VALUE_WEIGHT * mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_ce += ce.item(); run_mse += mse.item()

        if a.bench and step == a.bench:
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"{step} steps in {dt:.1f}s = {step * a.batch / dt:.0f} positions/s; "
                  f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            return

        stop = a.max_minutes and (time.time() - t0) / 60 > a.max_minutes
        if step % a.eval_every == 0 or step == a.steps or stop:
            k = a.eval_every if step % a.eval_every == 0 else step % a.eval_every
            ev = evaluate(model, val, n_val)
            record({"step": step, "epochs": round(step * a.batch / n_train, 3), "train_ce": run_ce / k,
                    "train_mse": run_mse / k, **ev, "lr": lr_at(step), "minutes": round((time.time() - t0) / 60, 1)})
            run_ce = run_mse = 0.0
            torch.save(model.state_dict(), os.path.join(a.out, "last.pth"))
            score = ev["val_ce"] + VALUE_WEIGHT * ev["val_mse"]
            if score < best:
                best = score
                torch.save(model.state_dict(), os.path.join(a.out, "best.pth"))
        if stop:
            print("time budget reached", flush=True)
            break


if __name__ == "__main__":
    main()
