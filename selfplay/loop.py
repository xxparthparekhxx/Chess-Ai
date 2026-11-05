"""The self-play loop. Resumable: state lives in <run>/state.json and each step's output marks it done.

Per generation g:
  1. self-play: the current best network plays --games games       -> games/gen_<g>/
  2. train: the trainer's weights (carried over from the previous generation, as in AlphaGo Zero)
     learn from the last --window generations plus the Lichess anchor -> nets/cand_<g>.pth
  3. gate: candidate vs best, --gate-openings openings x 2 colours    -> gates/gen_<g>.json
     the candidate becomes the best network if it scores at least --promote.
Every --ladder-every promotions the best network also plays limited Stockfish at UCI_Elo 2000 with the
README's harness (eval/run_match.py, 800 simulations) so the numbers compare with the baseline.

  python selfplay/loop.py --run runs/sp1 --start models/finetune/best.pth
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def sh(args, log):
    print("$", " ".join(args), flush=True)
    with open(log, "a") as f:
        r = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(args)}; see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--start", help="network for generation 0 (first run only)")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--reuse", type=float, default=4.0, help="times each self-play position is sampled, on average")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--anchor", default="data/lichess_evals", help="Stockfish-labelled positions (prep_lichess_evals.py)")
    ap.add_argument("--anchor-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate-openings", type=int, default=32)
    ap.add_argument("--gate-sims", type=int, default=200)
    ap.add_argument("--promote", type=float, default=0.53)
    ap.add_argument("--ladder-every", type=int, default=5, help="promotions between Stockfish checks (0 = never)")
    ap.add_argument("--gens", type=int, default=10_000)
    ap.add_argument("--actor-args", default="", help="extra actor.py arguments, e.g. '--full-sims 600'")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    for d in ("games", "nets", "gates", "logs"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    state_path = os.path.join(run, "state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    else:
        g0 = os.path.join(run, "nets", "gen_000.pth")
        shutil.copy(a.start, g0)
        state = {"gen": 0, "best": g0, "best_gen": 0, "promotions": 0, "start": os.path.abspath(a.start)}
    trainer = os.path.join(run, "trainer.pt")

    def save():
        with open(state_path + ".tmp", "w") as f:
            json.dump(state, f, indent=1)
        os.replace(state_path + ".tmp", state_path)

    save()
    while state["gen"] < a.gens:
        g = state["gen"] + 1
        tag = f"gen_{g:03d}"
        log = os.path.join(run, "logs", tag + ".log")
        t0 = time.time()

        sp = os.path.join(run, "games", tag)
        if not os.path.exists(os.path.join(sp, "stats.json")):
            sh([PY, "-u", "selfplay/actor.py", "--net", state["best"], "--out", sp, "--games", str(a.games),
                "--seed", str(g), *a.actor_args.split()], log)
        with open(os.path.join(sp, "stats.json")) as f:
            sp_stats = json.load(f)

        cand = os.path.join(run, "nets", f"cand_{g:03d}.pth")
        if not os.path.exists(cand + ".json"):
            dirs = sorted(d for d in os.listdir(os.path.join(run, "games"))
                          if os.path.exists(os.path.join(run, "games", d, "stats.json")) and d <= tag)[-a.window:]
            n_self = a.batch - round(a.batch * a.anchor_frac)
            steps = max(100, math.ceil(a.reuse * sp_stats["positions"] / n_self))
            sh([PY, "-u", "selfplay/train.py", "--state", trainer, "--init", state["best"], "--out", cand,
                "--games", *[os.path.join(run, "games", d) for d in dirs], "--steps", str(steps),
                "--batch", str(a.batch), "--anchor", a.anchor, "--anchor-frac", str(a.anchor_frac), "--lr", str(a.lr), "--seed", str(g)], log)
        with open(cand + ".json") as f:
            tr = json.load(f)

        gate = os.path.join(run, "gates", tag)
        if not os.path.exists(gate + ".json"):
            sh([PY, "-u", "selfplay/match.py", "--a", cand, "--b", state["best"], "--out", gate,
                "--openings", str(a.gate_openings), "--sims", str(a.gate_sims)], log)
        with open(gate + ".json") as f:
            gt = json.load(f)

        promoted = gt["score"] >= a.promote
        prev_best = state["best_gen"]
        if promoted:
            best = os.path.join(run, "nets", f"gen_{g:03d}.pth")
            shutil.copy(cand, best)
            state.update(best=best, best_gen=g, promotions=state["promotions"] + 1)

        ladder = None
        if promoted and a.ladder_every and state["promotions"] % a.ladder_every == 0:
            ltag = f"{os.path.basename(run)}_{tag}"
            sh([PY, "-u", "eval/run_match.py", "--ckpt", state["best"], "--tag", ltag, "--levels", "2000",
                "--pairs", "24", "--workers", "2"], log)
            with open(os.path.join(ROOT, "eval", "results", ltag, "summary.json")) as f:
                ladder = json.load(f)["levels"][0]

        entry = {
            "gen": g, "time": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t0) / 60, 1),
            "selfplay": {k: sp_stats[k] for k in ("games", "white_wins", "draws", "black_wins", "avg_plies",
                                                  "positions", "policy_positions", "would_resign",
                                                  "would_resign_wrong", "evals_per_s", "minutes")},
            "train": {k: tr[k] for k in ("steps", "total_steps", "self_positions", "train_self_ce", "train_self_mse",
                                         "train_anchor_ce", "train_anchor_mse", "minutes")} | {"anchor_val": tr["after"]},
            "gate": {k: gt[k] for k in ("wins", "draws", "losses", "score", "elo", "elo_ci95", "minutes")},
            "vs_gen": prev_best, "promoted": promoted, "best_gen": state["best_gen"],
        }
        if ladder:
            entry["stockfish_2000"] = {k: ladder[k] for k in ("wins", "draws", "losses", "score", "elo_estimate")}
        with open(os.path.join(run, "log.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        state["gen"] = g
        save()


if __name__ == "__main__":
    main()
