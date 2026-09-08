"""Evaluate a trained GastroHUN classifier on the Test split.

Writes to --output-dir:
  predict.json         per-image: true label, predicted class, probabilities
  metrics.csv          accuracy / macro & weighted P/R/F1 / MCC (point estimate)
  bootstrap.json       B-resample macro-F1: mean, std, 95% percentile CI, t-margin
  confusion_matrix.png row-normalised confusion matrix

Bootstrap: nonparametric resampling of the test predictions with replacement,
B iterations, macro-F1 each. Internally consistent across all experiment arms
(that is what the generic-vs-continued comparison relies on).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

from data import (MAP_CATEGORIES, GastroHunDataset, NUM_CLASSES, build_transform,
                  load_split_df)
from engine import make_loader
from model import GastroHunClassifier, build_backbone

DEFAULT_DATA = os.path.expanduser("~/Datasets/GastroHun/Labeled_Images_GastroHun")
DEFAULT_SPLIT = os.path.expanduser("~/Datasets/GastroHun/official_splits_GastroHun/image_classification.csv")
CLASS_NAMES = [k if k != "OTHERCLASS" else "NA" for k in MAP_CATEGORIES]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier-ckpt", required=True)
    p.add_argument("--backbone-weights", required=True)
    p.add_argument("--backbone-kind", required=True, choices=["wrapped", "teacher"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-path", default=DEFAULT_DATA)
    p.add_argument("--official-split", default=DEFAULT_SPLIT)
    p.add_argument("--label", default="Complete agreement")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    logits, ys, idxs = [], [], []
    for x, y, row_idx in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
        idxs.append(row_idx)
    logits = torch.cat(logits)
    return logits.softmax(-1).numpy(), torch.cat(ys).numpy(), torch.cat(idxs).numpy()


def bootstrap_macro_f1(y_true, y_pred, b, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(b)
    for i in range(b):
        s = rng.integers(0, n, n)
        vals[i] = f1_score(y_true[s], y_pred[s], average="macro", zero_division=0) * 100
    from scipy import stats
    sem = stats.sem(vals)
    t = stats.t.ppf(0.975, b - 1)
    return {
        "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
        "ci95_lo": float(np.percentile(vals, 2.5)),
        "ci95_hi": float(np.percentile(vals, 97.5)),
        "t_margin": float(t * sem), "b": b,
    }


def plot_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm.astype(float) / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cmn, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=6, color="w" if cmn[i, j] < 0.6 else "k")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_split_df(args.official_split, args.label)
    test_df = df[df["set_type"] == "Test"]
    ds = GastroHunDataset(test_df, args.data_path, build_transform(args.input_size),
                          args.label, return_index=True)
    loader = make_loader(ds, args.batch_size, False, args.num_workers)

    model = GastroHunClassifier(
        build_backbone(args.backbone_weights, args.backbone_kind), NUM_CLASSES).to(device)
    state = torch.load(args.classifier_ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(state)

    probs, y_true, row_idx = infer(model, loader, device)
    y_pred = probs.argmax(1)

    pd.DataFrame({
        "row_index": row_idx, "true": y_true, "pred": y_pred,
        "max_prob": probs.max(1), "probs": [p.round(4).tolist() for p in probs],
    }).to_json(os.path.join(args.output_dir, "predict.json"), orient="records")

    metrics = {
        "accuracy": (y_pred == y_true).mean() * 100,
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "mcc": matthews_corrcoef(y_true, y_pred) * 100,
        "n_test": len(y_true),
    }
    pd.Series(metrics).to_csv(os.path.join(args.output_dir, "metrics.csv"))

    boot = bootstrap_macro_f1(y_true, y_pred, args.bootstrap, args.seed)
    with open(os.path.join(args.output_dir, "bootstrap.json"), "w") as fh:
        json.dump(boot, fh, indent=2)

    plot_confusion(confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)),
                   os.path.join(args.output_dir, "confusion_matrix.png"))

    print(f"macro-F1 {metrics['macro_f1']:.2f}  |  bootstrap {boot['mean']:.2f} "
          f"[{boot['ci95_lo']:.2f}, {boot['ci95_hi']:.2f}]  (±{boot['t_margin']:.2f})")


if __name__ == "__main__":
    main()
