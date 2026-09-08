#!/bin/bash
#SBATCH --job-name=hkv_ds_exp
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/downstream/exp/slurm_%j.out
#SBATCH --error=outputs/downstream/exp/slurm_%j.err

# Phase E, step 2 -- the 2x2 comparison on GastroHUN.
#   BEST_ITER=<from the sweep> sbatch cluster/downstream_experiment.sh
#
# 2 backbones (generic wrapped / continued teacher @ BEST_ITER)
#   x 2 modes (linear_probe / finetune)
#   -> 4 trained classifiers, each evaluated on the Test split with bootstrap CI.

set -uo pipefail
: "${BEST_ITER:?set BEST_ITER, e.g. BEST_ITER=59999 sbatch cluster/downstream_experiment.sh}"

PROJECT=$HOME/continue_ssl_pretrain_dinov2
cd "$PROJECT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT/dinov2:$PROJECT/downstream:${PYTHONPATH:-}"

OUT="$PROJECT/outputs/downstream/exp"
mkdir -p "$OUT"
GENERIC="$PROJECT/checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth"
CONTINUED="$PROJECT/outputs/full_run/eval/training_${BEST_ITER}/teacher_checkpoint.pth"
[ -f "$CONTINUED" ] || { echo "MISSING $CONTINUED"; exit 1; }

run() {  # $1 name  $2 weights  $3 kind  $4 mode
  local d="$OUT/$1"
  echo ""; echo "############  $1  ############"
  python downstream/train_gastrohun.py --mode "$4" \
    --backbone-weights "$2" --backbone-kind "$3" --output-dir "$d"
  python downstream/run_test.py \
    --classifier-ckpt "$d/best-model-val_f1_macro.pt" \
    --backbone-weights "$2" --backbone-kind "$3" --output-dir "$d"
}

run generic_frozen    "$GENERIC"   wrapped linear_probe
run generic_finetuned "$GENERIC"   wrapped finetune
run continued_frozen  "$CONTINUED" teacher linear_probe
run continued_finetuned "$CONTINUED" teacher finetune

echo ""
echo "=========================  RESULTS (test macro-F1)  ========================="
python - <<'PY'
import csv, json, os
base = os.path.expanduser("~/continue_ssl_pretrain_dinov2/outputs/downstream/exp")
print(f"{'condition':22} {'point':>7}   bootstrap mean [95% CI]")
for name in ("generic_frozen", "generic_finetuned", "continued_frozen", "continued_finetuned"):
    d = os.path.join(base, name)
    pt = dict(csv.reader(open(os.path.join(d, "metrics.csv"))))["macro_f1"]
    b = json.load(open(os.path.join(d, "bootstrap.json")))
    print(f"{name:22} {float(pt):7.2f}   {b['mean']:.2f} [{b['ci95_lo']:.2f}, {b['ci95_hi']:.2f}]")
PY
echo "experiment finished: $(date)"
