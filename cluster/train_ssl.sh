#!/bin/bash
#SBATCH --job-name=hkv_dinov2_ssl
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/full_run/slurm_%j.out
#SBATCH --error=outputs/full_run/slurm_%j.err

set -euo pipefail

# ---- EDIT if your cluster needs modules (check `module avail`) ----
# module load gcc cuda python
# ------------------------------------------------------------------

PROJECT=$HOME/continue_ssl_pretrain_dinov2
cd "$PROJECT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT/dinov2:${PYTHONPATH:-}"

NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs visible: $NGPU"
nvidia-smi

OUT="$PROJECT/outputs/full_run"
mkdir -p "$OUT"

# BATCH_PER_GPU: set this from the memory probe (Phase C2). Placeholder below.
BATCH_PER_GPU=${BATCH_PER_GPU:-48}

torchrun --standalone --nproc_per_node="$NGPU" -m dinov2.train.train \
  --config-file "$PROJECT/dinov2/dinov2/configs/train/vits14_reg4_hyperkvasir_continued.yaml" \
  --output-dir "$OUT" \
  train.batch_size_per_gpu="$BATCH_PER_GPU"

echo "done: $(date)"
