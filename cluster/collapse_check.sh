#!/bin/bash
#SBATCH --job-name=hkv_collapse
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/collapse_check/slurm_%j.out
#SBATCH --error=outputs/collapse_check/slurm_%j.err

# Phase C3 -- a real short run (2000 iters at the production batch size) to check
# the loss is healthy: drifts DOWN gradually, stays well above ~1, no NaN.
# Collapse = plunge toward ~0.05 or NaN. If it collapses, retry with
#   train.centering=centering  dino.koleo_loss_weight=0   (Darcet mitigations).
#   sbatch cluster/collapse_check.sh

set -uo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dinov2
cd "$PROJECT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT/dinov2:${PYTHONPATH:-}"

CFG="$PROJECT/dinov2/dinov2/configs/train/vits14_reg4_hyperkvasir_continued.yaml"
OUT="$PROJECT/outputs/collapse_check"
rm -rf "$OUT"; mkdir -p "$OUT"
nvidia-smi

torchrun --standalone --nproc_per_node=1 -m dinov2.train.train \
  --config-file "$CFG" --output-dir "$OUT" \
  train.OFFICIAL_EPOCH_LENGTH=500 \
  optim.epochs=4 \
  optim.warmup_epochs=1

echo ""
echo "=====================  LOSS TRAJECTORY  ====================="
python - <<'PY'
import json, math
rows = [json.loads(l) for l in open("outputs/collapse_check/training_metrics.json")]
print(f"{'iter':>6}  {'total':>8}  {'dino_g':>8}  {'dino_l':>8}  {'koleo':>8}  {'ibot':>8}")
for r in rows:
    print(f"{r['iteration']:>6}  {r.get('total_loss',float('nan')):>8.3f}  "
          f"{r.get('dino_global_crops_loss',float('nan')):>8.3f}  "
          f"{r.get('dino_local_crops_loss',float('nan')):>8.3f}  "
          f"{r.get('koleo_loss',float('nan')):>8.3f}  "
          f"{r.get('ibot_loss',float('nan')):>8.3f}")
last = rows[-1]['total_loss']
bad = math.isnan(last) or last < 1.0
print()
print("VERDICT:", "!!! LOOKS LIKE COLLAPSE -- see mitigations in the script header"
      if bad else "healthy (final total_loss >= 1, no NaN)")
PY
echo "collapse_check finished: $(date)"
rm -f "$OUT"/model_final.rank_0.pth   # 590 MB, not needed
