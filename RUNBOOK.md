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

**Python 3.11 required** (dinov2's newer code uses `X | None` union syntax → 3.10+).
xformers is **required** (ssl_meta_arch asserts it) and pins torch → we end up on
torch 2.11. All cu128, all matched:

```bash
cd ~/continue_ssl_pretrain_dinov2
~/venv_gastrovision/bin/python -m venv .venv      # bootstrap Python 3.11
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128
pip install omegaconf fvcore iopath submitit torchmetrics pillow numpy
# verify the FSDP private API dinov2 needs is present, and everything imports:
python -c "from torch.distributed.fsdp._runtime_utils import _reshard; print('_reshard OK')"
export PYTHONPATH="$PWD/dinov2"
python -c "import dinov2.train.train; print('train imports OK')"
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
     --mem=32G --time=02:00:00 --pty bash
cd ~/continue_ssl_pretrain_dinov2 && source .venv/bin/activate
export PYTHONPATH="$PWD/dinov2:${PYTHONPATH:-}"
CFG=$PWD/dinov2/dinov2/configs/train/vits14_reg4_hyperkvasir_continued.yaml
```

### C1 `[C]` smoke test — does the whole pipeline run (10 iterations)
```bash
# NOTE: any short override run needs optim.warmup_epochs small enough that
# warmup_iters <= total_iters, else CosineScheduler asserts. The real config
# (epochs=100, warmup_epochs=20) is fine as-is.
torchrun --standalone --nproc_per_node=1 -m dinov2.train.train --config-file $CFG \
  --output-dir /tmp/smoke \
  train.batch_size_per_gpu=8 train.OFFICIAL_EPOCH_LENGTH=10 optim.epochs=1 optim.warmup_epochs=0 train.num_workers=2
```
Watch for: `# of dataset samples: 108,321`, `OPTIONS -- pretrained weights: loading from ...`,
`sqrt scaling learning rate`, a finite loss, no crash.

### C2 `[C]` memory probe — largest batch that fits on one GPU
```bash
sbatch cluster/mem_probe.sh
tail -n 30 outputs/memprobe/slurm_*.out     # read the SUMMARY block
```
DONE 2026-09-06: 64..256 all fit on A100-40GB. Picked **224** (30.6 GB, 77%,
3.45 s/it, ~65 img/s) — now the config default. 256 fit too (87%) but 224 keeps
fragmentation headroom for the ~4-day run.

### C3 `[C]` collapse check — a real short run (2000 iterations), inspect the loss
```bash
sbatch cluster/collapse_check.sh
tail -n 40 outputs/collapse_check/slurm_*.out   # LOSS TRAJECTORY + VERDICT
```
Healthy: loss drifts **down gradually** and stays well above ~1. Collapse =
plunges to ~0.05 fast / goes NaN. If it collapses → rerun with
`train.centering=centering dino.koleo_loss_weight=0` (Darcet's mitigations).

---

## Phase D — the full run

### D1 `[C]` set the probed batch size, submit
```bash
cd ~/continue_ssl_pretrain_dinov2
mkdir -p outputs/full_run
BATCH_PER_GPU=<from C2> sbatch cluster/train_ssl.sh
```
**1 GPU on purpose.** With `world_size==1`, FSDP downgrades `SHARD_GRAD_OP` to
`NO_SHARD`, so dinov2's sharded-FSDP-internals paths (`free_if_fsdp` / `_handles`
/ `_reshard`) never execute -- those are what breaks under torch 2.11's FSDP1.
ViT-S fits one A100-40GB easily. Going multi-GPU would require patching the
torch-2.11 FSDP1 internals (or downgrading torch), not worth it here.

### D2 `[C]` monitor
```bash
squeue -u $USER
tail -f outputs/full_run/slurm_*.out
```
DONE 2026-09-08 (job 2616177, 25h51m). 10 teacher checkpoints in
`outputs/full_run/eval/training_{9999..99999}/teacher_checkpoint.pth`; loss
healthy (11.52 -> 7.06, no collapse). `model_*.rank_0.pth` are resume-only.

---

## Phase E — downstream on GastroHUN

Standalone port of the `gastrohun-dino` image-classification recipe, in plain
PyTorch, under `downstream/`. Experiment = **2 backbones x 2 modes**:
generic (`checkpoints/...wrapped_224.pth`, kind `wrapped` = our SSL init point)
vs continued (`teacher_checkpoint.pth`, kind `teacher`), each as a frozen linear
probe and as the 2-phase fine-tune (warm-up head, then unfreeze last 40% blocks).

### E1 `[C]` extra venv deps (one-time)
```bash
cd ~/continue_ssl_pretrain_dinov2 && source .venv/bin/activate
pip install scikit-learn pandas scipy matplotlib
```
GastroHUN data is already on the cluster at
`~/Datasets/GastroHun/{Labeled_Images_GastroHun,official_splits_GastroHun}` --
the `downstream/` scripts default to those paths.

### E2 `[C]` checkpoint-selection sweep
```bash
sbatch cluster/downstream_sweep.sh
tail -n 20 outputs/downstream/sweep/slurm_*.out     # RANKING block -> BEST_ITER
```
Linear-probes all 10 teacher checkpoints (+ generic as reference) on GastroHUN
val, picks the best by val macro-F1.

### E3 `[C]` the 2x2 comparison
```bash
BEST_ITER=<from E2> sbatch cluster/downstream_experiment.sh
tail -n 20 outputs/downstream/exp/slurm_*.out        # RESULTS block
```
Trains the 4 conditions, evaluates each on the **Test** split with bootstrap CI.
Per condition, `outputs/downstream/exp/<name>/`: `best-model-val_f1_macro.pt`,
`summary.json`, `predict.json`, `metrics.csv`, `bootstrap.json`,
`confusion_matrix.png`.

### E4 sanity cross-check
Compare `generic_finetuned` test macro-F1 against `gastrohun-dino`'s published
`dinov2_vits14` = 72.96 (that used the **no-register** variant + a different
pipeline, so expect a few points of drift; close = port validated).
