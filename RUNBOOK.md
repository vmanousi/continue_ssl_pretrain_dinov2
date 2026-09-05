# Runbook — continued DINOv2 SSL on HyperKvasir → GastroHUN

`[L]` = run on the local machine, `[C]` = run on the Aristotle cluster.
The cluster hard-requires a GPU for training (no CPU path), so every training
step is `[C]`.

---

## Phase A — push code + move data

### A1 `[L]` create the GitHub remote (browser, one-time)
github.com/new → name `continue_ssl_pretrain_dinov2` → **Private** → do **not**
add README / .gitignore / licence → Create repository.

### A2 `[L]` push
```bash
cd ~/continue_ssl_pretrain_dinov2
git remote add origin https://github.com/vmanousi/continue_ssl_pretrain_dinov2.git
git branch -M main
git push -u origin main
```

### A3 `[C]` clone on the cluster
```bash
ssh aristotle.it.auth.gr
cd ~ && git clone https://github.com/vmanousi/continue_ssl_pretrain_dinov2.git
```

### A4 `[L]` package + transfer the corpus (~2.6 GB) and checkpoint (~83 MB)
```bash
cd ~/continue_ssl_pretrain_dinov2
tar czf /tmp/hyperkvasir_ssl.tar.gz -C data hyperkvasir_ssl
scp /tmp/hyperkvasir_ssl.tar.gz aristotle.it.auth.gr:~/continue_ssl_pretrain_dinov2/data/
scp checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth \
    aristotle.it.auth.gr:~/continue_ssl_pretrain_dinov2/checkpoints/
```

### A5 `[C]` unpack + sanity-check
```bash
cd ~/continue_ssl_pretrain_dinov2/data
tar xzf hyperkvasir_ssl.tar.gz && rm hyperkvasir_ssl.tar.gz
ls hyperkvasir_ssl/images | wc -l          # expect 108321
wc -l hyperkvasir_ssl/train_deduped.txt    # expect 108321
```

---

## Phase B — cluster environment

### B1 `[C]` discover modules
```bash
module avail 2>&1 | grep -iE "cuda|python|gcc"
```

### B2 `[C]` venv + dependencies
```bash
cd ~/continue_ssl_pretrain_dinov2
# module load ...   # whatever B1 revealed (a CUDA toolkit + python >= 3.9)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# torch matching the cluster CUDA (check `nvidia-smi` top-right for the driver's max CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install omegaconf fvcore iopath submitit pillow numpy
# xformers is OPTIONAL — dinov2 falls back to native attention without it
# (only affects speed/memory, which we probe empirically). Try, skip if it fails:
pip install xformers || echo "no xformers — proceeding without it"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### B3 `[C]` regenerate the wrapped checkpoint on the cluster (or keep the scp'd one)
```bash
python scripts/prepare_pretrained_checkpoint.py \
  --output checkpoints/dinov2_vits14_reg4_pretrain_wrapped_224.pth --target-crop-size 224
```

### B4 `[C]` config paths — auto-resolve via `${oc.env:HOME}`
No editing needed as long as the repo lives at `$HOME/continue_ssl_pretrain_dinov2`.
Just sanity-check they resolve to real files:
```bash
python - <<'PY'
import os, sys; sys.path.insert(0, "dinov2")
from omegaconf import OmegaConf
from dinov2.configs import load_and_merge_config
c = OmegaConf.to_container(load_and_merge_config("train/vits14_reg4_hyperkvasir_continued"), resolve=True)
root = c["train"]["dataset_path"].split("root=")[1].split(":")[0]
w = c["student"]["pretrained_weights"]
print("corpus dir exists:", os.path.isdir(root))
print("checkpoint exists:", os.path.isfile(w))
PY
```

---

## Phase C — smoke test → memory probe → collapse check

Grab a short interactive GPU session for C1–C2:
```bash
srun --partition=ampere --qos=ampere-extd --gres=gpu:1 --cpus-per-task=8 \
     --mem=32G --time=01:00:00 --pty bash
cd ~/continue_ssl_pretrain_dinov2 && source .venv/bin/activate
export PYTHONPATH="$PWD/dinov2:${PYTHONPATH:-}"
CFG=dinov2/dinov2/configs/train/vits14_reg4_hyperkvasir_continued.yaml
```

### C1 `[C]` smoke test — does the whole pipeline run (10 iterations)
```bash
torchrun --standalone --nproc_per_node=1 dinov2/train/train.py --config-file $CFG \
  --output-dir /tmp/smoke \
  train.batch_size_per_gpu=8 train.OFFICIAL_EPOCH_LENGTH=10 optim.epochs=1 train.num_workers=2
```
Watch for: `# of dataset samples: 108,321`, `pretrained weights: loading from ...`,
a finite loss (~9 initial for K=8192), no crash.

### C2 `[C]` memory probe — largest batch that fits on one GPU
```bash
for BS in 32 48 64 96 128; do
  echo "===== batch_size_per_gpu=$BS ====="
  timeout 400 torchrun --standalone --nproc_per_node=1 dinov2/train/train.py --config-file $CFG \
    --output-dir /tmp/probe_$BS \
    train.batch_size_per_gpu=$BS train.OFFICIAL_EPOCH_LENGTH=15 optim.epochs=1 train.num_workers=4 \
    2>&1 | tail -3
done
```
Pick the largest `BS` that finishes without `CUDA out of memory`; use ~10-20% below it.

### C3 `[C]` collapse check — a real short run (~2000 iterations), inspect the loss
```bash
BS=<from C2>
torchrun --standalone --nproc_per_node=1 dinov2/train/train.py --config-file $CFG \
  --output-dir ~/continue_ssl_pretrain_dinov2/outputs/collapse_check \
  train.batch_size_per_gpu=$BS train.OFFICIAL_EPOCH_LENGTH=500 optim.epochs=4
python - <<'PY'
import json
rows=[json.loads(l) for l in open("outputs/collapse_check/training_metrics.json")]
for r in rows[::5]:
    print(r["iteration"], round(r.get("total_loss", float("nan")),3))
PY
```
Healthy: loss drifts **down gradually** and stays well above ~1. Collapse =
plunges to ~0.05 fast / goes NaN. If it collapses → set `train.centering=centering`
and `dino.koleo_loss_weight=0` (Darcet's mitigations) and repeat C3.

---

## Phase D — the full run

### D1 `[C]` set the probed batch size, submit
```bash
cd ~/continue_ssl_pretrain_dinov2
mkdir -p outputs/full_run
# edit cluster/train_ssl.sh: set BATCH_PER_GPU= and --gres=gpu:N (1 or 2)
BATCH_PER_GPU=<from C2> sbatch cluster/train_ssl.sh
```

### D2 `[C]` monitor
```bash
squeue -u $USER
tail -f outputs/full_run/slurm_*.out
```
Checkpoints land in `outputs/full_run/` every `saveckp_freq` epochs
(`eval/training_*` dirs and `model_*.rank_*.pth`).

---

## Phase E — downstream on GastroHUN  *(code not built yet)*

Needs a standalone port of the `Gastrohun_official` `dinov2_vits14` recipe
(warmup + ~40% unfreeze + discriminative LR + class-weighted loss + macro-F1
checkpoint + bootstrap-CI test evaluation). Build this once Phase D produces
backbone checkpoints, then:
- extract each candidate SSL checkpoint's backbone,
- frozen-probe on GastroHUN val → pick the best epoch,
- fine-tune (student + teacher) with the ported recipe,
- evaluate generic-DINOv2 vs continued-DINOv2 on the GastroHUN **test** split,
  same protocol, + a frozen-backbone comparison.
