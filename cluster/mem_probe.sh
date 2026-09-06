#!/bin/bash
#SBATCH --job-name=hkv_memprobe
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/memprobe/slurm_%j.out
#SBATCH --error=outputs/memprobe/slurm_%j.err

# Phase C2 -- find the largest batch_size_per_gpu that fits on one A100-40GB.
# Runs ~15 iterations per candidate, records peak GPU mem, catches CUDA OOM,
# deletes the per-candidate output dir afterwards (each model_final is ~590 MB).
#   sbatch cluster/mem_probe.sh
# then read the summary at the end of the .out file.

set -uo pipefail
PROJECT=$HOME/continue_ssl_pretrain_dinov2
cd "$PROJECT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT/dinov2:${PYTHONPATH:-}"

CFG="$PROJECT/dinov2/dinov2/configs/train/vits14_reg4_hyperkvasir_continued.yaml"
mkdir -p outputs/memprobe
nvidia-smi

CANDIDATES="64 96 128 160 192 224 256"
RESULTS=()

for BS in $CANDIDATES; do
  echo ""
  echo "=================  batch_size_per_gpu = $BS  ================="
  OUT="outputs/memprobe/bs_$BS"
  rm -rf "$OUT"; mkdir -p "$OUT"
  LOG="$OUT/run.log"

  timeout 600 torchrun --standalone --nproc_per_node=1 -m dinov2.train.train \
    --config-file "$CFG" --output-dir "$OUT" \
    train.batch_size_per_gpu=$BS \
    train.OFFICIAL_EPOCH_LENGTH=15 \
    optim.epochs=1 \
    optim.warmup_epochs=0 \
    train.num_workers=4 \
    > "$LOG" 2>&1
  RC=$?

  if grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG"; then
    VERDICT="OOM"
  elif [ $RC -eq 124 ]; then
    VERDICT="TIMEOUT(>600s for 15 iters -- treat as too big / too slow)"
  elif [ $RC -ne 0 ]; then
    VERDICT="FAILED rc=$RC (see $LOG)"
  else
    MEM=$(grep -oE "max mem: [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")
    SPI=$(grep -oE "s / it\)" "$LOG" >/dev/null && grep -oE "\([0-9.]+ s / it\)" "$LOG" | tail -1)
    VERDICT="OK  peak_mem=${MEM}MB  ${SPI}"
  fi
  echo ">>> BS=$BS : $VERDICT"
  RESULTS+=("BS=$BS : $VERDICT")
  rm -rf "$OUT"   # reclaim the ~590 MB model_final
done

echo ""
echo "=========================  SUMMARY  ========================="
for r in "${RESULTS[@]}"; do echo "$r"; done
echo "============================================================"
echo "Pick the largest OK batch, then use ~10-15% below it for the full run."
echo "memprobe finished: $(date)"
