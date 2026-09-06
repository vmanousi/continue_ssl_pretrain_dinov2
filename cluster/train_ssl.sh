#!/bin/bash
#SBATCH --job-name=hkv_dinov2_ssl
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=6-00:00:00

# 1 GPU on purpose. With world_size==1 FSDP collapses SHARD_GRAD_OP -> NO_SHARD,
# so none of dinov2's sharded-FSDP-internals code runs (free_if_fsdp / _handles /
# _reshard), which is what breaks under torch 2.11's FSDP1. ViT-S (~22M params)
# fits a single A100-40GB comfortably at the probed batch size. Multi-GPU would
# need the torch-2.11 FSDP1 internals patched or a torch downgrade -- not worth it
# for a model this small.
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

# BATCH_PER_GPU: from the Phase C2 probe (largest safe on one A100-40GB).
BATCH_PER_GPU=${BATCH_PER_GPU:-224}

torchrun --standalone --nproc_per_node="$NGPU" -m dinov2.train.train \
  --config-file "$PROJECT/dinov2/dinov2/configs/train/vits14_reg4_hyperkvasir_continued.yaml" \
  --output-dir "$OUT" \
  train.batch_size_per_gpu="$BATCH_PER_GPU"

echo "done: $(date)"
