"""Train a GastroHUN classifier on one DINOv2 backbone.

modes:
  linear_probe : backbone fully frozen, train only the linear head. Used both
                 for the 10-checkpoint selection sweep and as experiment
                 conditions 1 & 3 (frozen probe).
  finetune     : the 2-phase `gastrohun-dino` recipe -- warm up the head
                 (backbone frozen), then unfreeze the last 40% of blocks and
                 fine-tune. Experiment conditions 2 & 4.

Recipe defaults match image_classification/IMAGECLASSIFICATION.md.
"""
import argparse
import json
import os

import numpy as np
import torch

from data import (GastroHunDataset, NUM_CLASSES, build_transform,
                  compute_class_weights, load_split_df)
from engine import (evaluate, extract_features, make_loader, run_phase,
                    train_linear_head)
from model import GastroHunClassifier, build_backbone, set_finetune_mode, set_frozen_mode

DEFAULT_DATA = os.path.expanduser("~/Datasets/GastroHun/Labeled_Images_GastroHun")
DEFAULT_SPLIT = os.path.expanduser("~/Datasets/GastroHun/official_splits_GastroHun/image_classification.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-weights", required=True)
    p.add_argument("--backbone-kind", required=True, choices=["wrapped", "teacher"])
    p.add_argument("--mode", required=True, choices=["linear_probe", "finetune"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-path", default=DEFAULT_DATA)
    p.add_argument("--official-split", default=DEFAULT_SPLIT)
    p.add_argument("--label", default="Complete agreement")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # linear-probe / warm-up
    p.add_argument("--lr-warmup", type=float, default=1e-3)
    p.add_argument("--epochs-warmup", type=int, default=10)
    p.add_argument("--epochs-probe", type=int, default=50)
    # finetune phase
    p.add_argument("--lr-finetuning", type=float, default=7e-4)
    p.add_argument("--epochs-finetuning", type=int, default=100)
    p.add_argument("--gamma-finetuning", type=float, default=0.3)
    p.add_argument("--step-size-finetuning", type=int, default=5)
    p.add_argument("--unfrozen-pct", type=float, default=40)
    p.add_argument("--early-stopping", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_split_df(args.official_split, args.label)
    tf = build_transform(args.input_size)
    ds = {s: GastroHunDataset(df[df["set_type"] == name], args.data_path, tf, args.label)
          for s, name in [("train", "Train"), ("val", "Validation")]}
    print(f"train {len(ds['train'])}  val {len(ds['val'])}")
    train_loader = make_loader(ds["train"], args.batch_size, True, args.num_workers)
    val_loader = make_loader(ds["val"], args.batch_size, False, args.num_workers)
    class_weights = compute_class_weights(df, args.label, device)

    backbone = build_backbone(args.backbone_weights, args.backbone_kind)
    model = GastroHunClassifier(backbone, NUM_CLASSES).to(device)

    ckpt_path = os.path.join(args.output_dir, "best-model-val_f1_macro.pt")
    results = {"args": vars(args)}

    if args.mode == "linear_probe":
        f_tr, y_tr = extract_features(model.backbone, train_loader, device)
        f_va, y_va = extract_features(model.backbone, val_loader, device)
        head, r = train_linear_head(f_tr, y_tr, f_va, y_va, num_classes=NUM_CLASSES,
                                    class_weights=class_weights, device=device)
        model.head.load_state_dict(head.state_dict())
        torch.save({"state_dict": model.state_dict(),
                    "val_f1_macro": r["best_f1_macro"], "epoch": r["best_epoch"]},
                   ckpt_path)
        results["linear_probe"] = r
    else:
        set_frozen_mode(model)
        w = run_phase(model, train_loader, val_loader, device=device,
                      lr=args.lr_warmup, epochs=args.epochs_warmup,
                      step_size=args.epochs_warmup + 1, gamma=1.0,
                      class_weights=class_weights, patience=None,
                      best_ckpt_path=None, log_prefix="[warmup] ")
        set_finetune_mode(model, args.unfrozen_pct)
        f = run_phase(model, train_loader, val_loader, device=device,
                      lr=args.lr_finetuning, epochs=args.epochs_finetuning,
                      step_size=args.step_size_finetuning, gamma=args.gamma_finetuning,
                      class_weights=class_weights, patience=args.early_stopping,
                      best_ckpt_path=ckpt_path, log_prefix="[finetune] ")
        results["warmup"], results["finetune"] = w, f

    val = evaluate(model, val_loader, device)
    results["val_final"] = {k: val[k] for k in ("f1_macro", "f1_weighted", "accuracy")}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"DONE  best val macro-F1 = "
          f"{results.get('linear_probe', results.get('finetune'))['best_f1_macro']:.2f}")


if __name__ == "__main__":
    main()
