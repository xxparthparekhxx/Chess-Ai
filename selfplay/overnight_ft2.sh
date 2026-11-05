#!/usr/bin/env bash
# One unattended chain on the training laptop (GTX 1650 Ti, fp32):
#   1. fine-tune the current best net on the large Stockfish-labelled Lichess set (one pass over 23.7M positions, ~6.2 h)
#   2. play it against the net it started from (64 games), keep the winner
#   3. Stockfish UCI_Elo 2000 check of the new net with the README's harness
#   4. self-play from the winner, anchored on the same large set
# Every step skips itself when its output already exists, so the script can be re-run after a stop.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OLD=models/finetune/best.pth
DATA=data/lichess_evals_big
mkdir -p runs/ft2

[ -f runs/ft2/done ] || {
  $PY -u scripts/finetune_lichess.py --data $DATA --out runs/ft2 --init $OLD --steps 46000 --batch 512 --lr 2e-4 \
    --warmup 500 --eval-every 2500 --val-n 50000 --fp32 --no-channels-last --host-data
  touch runs/ft2/done
}
[ -f runs/ft2/gate_vs_ft1.json ] || $PY -u selfplay/match.py --a runs/ft2/best.pth --b $OLD --out runs/ft2/gate_vs_ft1
WIN=$($PY -c "import json; print('runs/ft2/best.pth' if json.load(open('runs/ft2/gate_vs_ft1.json'))['score'] >= 0.5 else '$OLD')")
echo "self-play starts from $WIN"
[ "$WIN" = "$OLD" ] || $PY -u eval/run_match.py --ckpt runs/ft2/best.pth --tag ft2 --levels 2000 --pairs 24 --workers 2
exec $PY -u selfplay/loop.py --run runs/sp1 --start "$WIN" --anchor $DATA
