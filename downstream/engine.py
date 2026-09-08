"""Shared train / eval loop — plain-PyTorch port of utils/train_module_image.py
(ModelTrainer) from `gastrohun-dino`.

Original recipe, preserved:
  - optimiser: Adam over the trainable params, one flat LR per phase
  - scheduler: StepLR(step_size, gamma)
  - loss: CrossEntropyLoss(weight=class_weights)  (sklearn 'balanced')
  - model selection: max val macro-F1, weights only
  - early stopping: patience epochs with no val macro-F1 improvement
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from model import trainable_parameters


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        all_logits.append(model(x.to(device)).float().cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def evaluate(model, loader, device):
    logits, y = _predict(model, loader, device)
    probs = logits.softmax(dim=-1).numpy()
    pred = probs.argmax(axis=1)
    y = y.numpy()
    return {
        "y_true": y,
        "y_pred": pred,
        "probs": probs,
        "f1_macro": f1_score(y, pred, average="macro") * 100,
        "f1_weighted": f1_score(y, pred, average="weighted") * 100,
        "accuracy": (pred == y).mean() * 100,
    }


def run_phase(model, train_loader, val_loader, *, device, lr, epochs,
              step_size, gamma, class_weights, patience=None,
              best_ckpt_path=None, log_prefix=""):
    """Train `model` for one phase. Returns (best_state_dict, history).

    If `patience`/`best_ckpt_path` are given, keeps the weights with the best
    val macro-F1; otherwise returns the final weights.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_f1, best_state, best_epoch, since_improved = -1.0, None, -1, 0
    history = []

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": running / len(train_loader.dataset),
               "val_f1_macro": val["f1_macro"], "val_f1_weighted": val["f1_weighted"],
               "val_acc": val["accuracy"], "seconds": time.time() - t0}
        history.append(row)
        print(f"{log_prefix}epoch {epoch:3d}  loss {row['train_loss']:.4f}  "
              f"val_f1_macro {val['f1_macro']:.2f}  lr {row['lr']:.2e}")

        if val["f1_macro"] > best_f1:
            best_f1, best_epoch, since_improved = val["f1_macro"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if best_ckpt_path:
                torch.save({"state_dict": best_state, "val_f1_macro": best_f1,
                            "epoch": epoch}, best_ckpt_path)
        else:
            since_improved += 1
            if patience is not None and since_improved >= patience:
                print(f"{log_prefix}early stop at epoch {epoch} "
                      f"(best {best_f1:.2f} @ {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_f1_macro": best_f1, "best_epoch": best_epoch, "history": history}


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0, drop_last=False)


# --- linear probe on cached features -------------------------------------------
# The recipe has NO train-time augmentation (only resize+normalise), so backbone
# features are deterministic and caching them is exact, not an approximation.

@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        feats.append(backbone(x).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_linear_head(f_tr, y_tr, f_va, y_va, *, num_classes, class_weights,
                      lr=1e-3, epochs=200, patience=20, device="cuda",
                      log_prefix="[probe] "):
    f_tr, y_tr = f_tr.to(device), y_tr.to(device)
    f_va, y_va = f_va.to(device), y_va.to(device)
    head = nn.Linear(f_tr.size(1), num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, best_epoch, since = -1.0, None, -1, 0
    history = []
    for epoch in range(epochs):
        head.train()
        opt.zero_grad(set_to_none=True)
        crit(head(f_tr), y_tr).backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            pred = head(f_va).argmax(1).cpu().numpy()
        f1 = f1_score(y_va.cpu().numpy(), pred, average="macro") * 100
        history.append({"epoch": epoch, "val_f1_macro": f1})
        if f1 > best_f1:
            best_f1, best_epoch, since = f1, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best_state)
    print(f"{log_prefix}best val macro-F1 {best_f1:.2f} @ epoch {best_epoch}")
    return head, {"best_f1_macro": best_f1, "best_epoch": best_epoch, "history": history}
