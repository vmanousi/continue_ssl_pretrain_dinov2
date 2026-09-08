#!/bin/bash
#SBATCH --job-name=hkv_ds_sweep
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/downstream/sweep/slurm_%j.out
#SBATCH --error=outputs/downstream/sweep/slurm_%j.err

# Phase E, step 1 -- linear-probe the 10 continued-SSL teacher checkpoints (plus
# the generic wrapped checkpoint as reference) on GastroHUN, pick the one with
# the best validation macro-F1. Linear probe = backbone frozen, features cached,
# only a linear head trained -> fast, a few minutes each.
#   sbatch cluster/downstream_sweep.sh

set -uo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dinov2
cd "$PROJECT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT/dinov2:$PROJECT/downstream:${PYTHONPATH:-}"

OUT="$PROJECT/outputs/downstream/sweep"
mkdir -p "$OUT"
GENERIC="$PROJECT/checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth"

echo "===== generic (wrapped) reference ====="
python downstream/train_gastrohun.py --mode linear_probe \
  --backbone-weights "$GENERIC" --backbone-kind wrapped \
  --output-dir "$OUT/generic" 2>&1 | tail -3

for IT in 9999 19999 29999 39999 49999 59999 69999 79999 89999 99999; do
  CKPT="$PROJECT/outputs/full_run/eval/training_$IT/teacher_checkpoint.pth"
  [ -f "$CKPT" ] || { echo "MISSING $CKPT"; continue; }
  echo "===== teacher iter $IT ====="
  python downstream/train_gastrohun.py --mode linear_probe \
    --backbone-weights "$CKPT" --backbone-kind teacher \
    --output-dir "$OUT/teacher_$IT" 2>&1 | tail -3
done

echo ""
echo "=========================  RANKING (val macro-F1)  ========================="
python - <<'PY'
import glob, json, os
rows = []
for p in glob.glob(os.path.expanduser("~/continue_ssl_pretrain_dinov2/outputs/downstream/sweep/*/summary.json")):
    d = json.load(open(p))
    name = os.path.basename(os.path.dirname(p))
    rows.append((d["linear_probe"]["best_f1_macro"], name))
for f1, name in sorted(rows, reverse=True):
    print(f"  {f1:6.2f}   {name}")
best = max((r for r in rows if r[1].startswith("teacher_")), default=None)
if best:
    print(f"\nBEST continued checkpoint: {best[1]}  (val macro-F1 {best[0]:.2f})")
    print(f"-> export BEST_ITER={best[1].split('_')[1]} for cluster/downstream_experiment.sh")
PY
echo "sweep finished: $(date)"
