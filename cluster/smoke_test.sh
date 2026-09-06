#!/bin/bash
#SBATCH --job-name=hkv_smoke
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/smoke/slurm_%j.out
#SBATCH --error=outputs/smoke/slurm_%j.err

# Short (10-iteration) smoke test: does the whole training pipeline run on a
# GPU? Kept small + 30 min so the scheduler can backfill it into gaps quickly.
#   sbatch cluster/smoke_test.sh

set -euo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dinov2
cd "$PROJECT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT/dinov2:${PYTHONPATH:-}"

mkdir -p outputs/smoke
nvidia-smi

torchrun --standalone --nproc_per_node=1 -m dinov2.train.train \
  --config-file "$PROJECT/dinov2/dinov2/configs/train/vits14_reg4_hyperkvasir_continued.yaml" \
  --output-dir "$PROJECT/outputs/smoke" \
  train.batch_size_per_gpu=8 \
  train.OFFICIAL_EPOCH_LENGTH=10 \
  optim.epochs=1 \
  train.num_workers=4

echo "smoke test finished: $(date)"
