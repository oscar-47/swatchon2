"""
Multi-Model Training Script for Fabric Classification
=====================================================
Trains 5 models x 3 stages = 15 runs:
  Models:  EfficientNet-V2-S, MaxViT-T, RegNet-Y-8GF, DenseNet-161, ResNeXt-101
  Stages:  Stage1 (Knit vs Woven binary), Stage2 Woven (5-class), Stage2 Knit (5-class)

Expects pre-split dataset:
  <data-dir>/train/<ClassName>/images...
  <data-dir>/val/<ClassName>/images...
  <data-dir>/test/<ClassName>/images...

Usage:
  python train_multi_model.py --data-dir . --amp
  python train_multi_model.py --data-dir . --models efficientnet_v2_s --stages stage1 --epochs 1
"""

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# ── Constants ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Model Registry ──────────────────────────────────────────────────────────

def _build_efficientnet_v2_s(n_classes: int) -> nn.Module:
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def _build_maxvit_t(n_classes: int) -> nn.Module:
    model = models.maxvit_t(weights="IMAGENET1K_V1")
    model.classifier[5] = nn.Linear(model.classifier[5].in_features, n_classes)
    return model


def _build_regnet_y_8gf(n_classes: int) -> nn.Module:
    model = models.regnet_y_8gf(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def _build_densenet161(n_classes: int) -> nn.Module:
    model = models.densenet161(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model


def _build_resnext101_32x8d(n_classes: int) -> nn.Module:
    model = models.resnext101_32x8d(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


MODEL_REGISTRY: Dict[str, Any] = {
    "efficientnet_v2_s": _build_efficientnet_v2_s,
    "maxvit_t":          _build_maxvit_t,
    "regnet_y_8gf":      _build_regnet_y_8gf,
    "densenet161":       _build_densenet161,
    "resnext101_32x8d":  _build_resnext101_32x8d,
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    out_dir: str
    epochs: int = 5
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 5e-4
    wd: float = 0.05
    optimizer: str = "adamw"
    sched: str = "cosine"
    amp: bool = True
    seed: int = 42
    class_weight: str = "none"
    accum_steps: int = 1
    patience: int = 12
    label_smoothing: float = 0.05


# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ──────────────────────────────────────────────────────────────────

class SimpleImageDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Data Loading (pre-split) ────────────────────────────────────────────────

def scan_imagefolder(root: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Scan <root>/<class>/<image> and return (items, sorted class_names)."""
    classes = sorted([n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items: List[Tuple[str, int]] = []
    for c in classes:
        d = os.path.join(root, c)
        for fn in os.listdir(d):
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                items.append((os.path.join(d, fn), class_to_idx[c]))
    return items, classes


def load_presplit(data_dir: str):
    """Load pre-split train/val/test from data_dir/{train,val,test}/."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            raise SystemExit(f"Directory not found: {d}")
    train_items, classes = scan_imagefolder(train_dir)
    val_items, _ = scan_imagefolder(val_dir)
    test_items, _ = scan_imagefolder(test_dir)
    return train_items, val_items, test_items, classes


def remap_to_binary(items: List[Tuple[str, int]], classes: List[str]):
    """Remap Knit_* -> 0, Woven_* -> 1."""
    binary_classes = ["Knit", "Woven"]
    new_items = []
    for path, label in items:
        class_name = classes[label]
        binary_label = 0 if class_name.startswith("Knit") else 1
        new_items.append((path, binary_label))
    return new_items, binary_classes


def filter_by_prefix(items: List[Tuple[str, int]], classes: List[str], prefix: str):
    """Keep only classes starting with prefix, remap labels to 0..N-1."""
    filtered_classes = sorted([c for c in classes if c.startswith(prefix)])
    class_to_new_idx = {c: i for i, c in enumerate(filtered_classes)}
    new_items = []
    for path, label in items:
        class_name = classes[label]
        if class_name.startswith(prefix):
            new_items.append((path, class_to_new_idx[class_name]))
    return new_items, filtered_classes


# ── Transforms ───────────────────────────────────────────────────────────────

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.08),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ── Metrics ──────────────────────────────────────────────────────────────────

def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long)
    for p, t in zip(preds, targets):
        cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm: torch.Tensor) -> float:
    f1s = []
    for c in range(cm.size(0)):
        tp = cm[c, c].item()
        fp = int(cm[:, c].sum().item() - tp)
        fn = int(cm[c, :].sum().item() - tp)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


# ── Utilities ────────────────────────────────────────────────────────────────

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[List], header: List[str], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')


# ── Optimizer / Scheduler ────────────────────────────────────────────────────

def get_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    else:
        raise ValueError("Unsupported optimizer: " + cfg.optimizer)


def get_scheduler(optimizer, cfg: TrainConfig):
    if cfg.sched.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    elif cfg.sched.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    return None


# ── Training / Evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, accum_steps=1):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * x.size(0)
        running_acc += (logits.argmax(1) == y).float().sum().item()
        n += x.size(0)
    return running_loss / max(1, n), running_acc / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes: int, want_probs: bool = False):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    all_preds, all_targets = [], []
    all_probs = [] if want_probs else None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        running_acc += (logits.argmax(1) == y).float().sum().item()
        n += x.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(y.cpu())
        if want_probs:
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu())
    if n == 0:
        return 0.0, 0.0, torch.zeros((n_classes, n_classes), dtype=torch.long), None
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    cm = confusion_matrix(preds, targets, n_classes)
    probs_all = torch.cat(all_probs) if (want_probs and all_probs) else None
    return running_loss / n, running_acc / n, cm, probs_all


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: List[Dict], out_path: str):
    epochs = [r["epoch"] for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, [r["train_loss"] for r in history], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, [r["val_loss"] for r in history], "o-", label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [r["train_acc"] for r in history], "o-", label="Train Acc", color="#2ca02c")
    ax2.plot(epochs, [r["val_acc"] for r in history], "o-", label="Val Acc", color="#d62728")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Train & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(classes)), max(5, 0.5 * len(classes))))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_bar(values: np.ndarray, classes: List[str], out_path: str,
                       title: str, ylabel: str):
    order = np.argsort(values)[::-1]
    values = values[order]
    classes = [classes[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(classes)), 5))
    ax.bar(np.arange(len(classes)), values, color="#4C72B0")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ── Report Generation ────────────────────────────────────────────────────────

def generate_full_report(test_report: Dict, history: List[Dict], out_dir: str,
                         model_name: str, stage_name: str):
    cm = np.array(test_report["confusion_matrix"], dtype=np.int64)
    classes = test_report["classes"]
    test_acc = float(test_report.get("test_acc", 0.0))
    test_f1 = float(test_report.get("test_macro_f1", 0.0))
    best_epoch = test_report.get("best_epoch")
    best_val_acc = test_report.get("best_val_acc")
    roc_auc = test_report.get("roc_auc")

    plot_training_curves(history, os.path.join(out_dir, "training_curves.png"))

    plot_confusion_matrix(
        cm, classes,
        os.path.join(out_dir, "confusion_matrix.png"),
        title=f"{model_name} / {stage_name}\n(acc={test_acc:.3f}, F1={test_f1:.3f})")

    tp = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    col_sum = cm.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)

    plot_per_class_bar(recall, classes,
                       os.path.join(out_dir, "per_class_accuracy.png"),
                       title="Per-class Accuracy (Recall)", ylabel="Accuracy")

    rows = []
    for i, c in enumerate(classes):
        fp_val = col_sum[i] - tp[i]
        fn_val = row_sum[i] - tp[i]
        rows.append([c, int(row_sum[i]), int(tp[i]), int(fp_val), int(fn_val),
                      f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}",
                      f"{recall[i]:.4f}"])
    save_csv(rows,
             header=["class", "support", "tp", "fp", "fn", "precision", "recall", "f1", "per_class_acc"],
             out_path=os.path.join(out_dir, "per_class_metrics.csv"))

    roc_line = f"<b>ROC-AUC</b>: {roc_auc:.3f} &nbsp; " if roc_auc is not None else ""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<title>{model_name} - {stage_name} Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 16px; }}
section {{ margin-bottom: 24px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 8px; border: 1px solid #ddd; text-align: left; }}
</style>
</head>
<body>
<h2>{model_name} / {stage_name}</h2>
<p>
  <b>Test Accuracy</b>: {test_acc:.3f} &nbsp;
  <b>Macro F1</b>: {test_f1:.3f} &nbsp;
  {roc_line}
  <b>Best Val Acc</b>: {best_val_acc:.3f} (epoch {best_epoch})
</p>
<section>
  <h3>Training Curves</h3>
  <img src="training_curves.png" alt="training curves"/>
</section>
<section>
  <h3>Confusion Matrix</h3>
  <img src="confusion_matrix.png" alt="confusion matrix"/>
</section>
<section>
  <h3>Per-class Accuracy (Recall)</h3>
  <img src="per_class_accuracy.png" alt="per class accuracy"/>
  <p>Details: <code>per_class_metrics.csv</code></p>
</section>
</body>
</html>"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ── GPU Cleanup ──────────────────────────────────────────────────────────────

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Core Training Function ───────────────────────────────────────────────────

def run_single_training(
    model_name: str,
    model_builder,
    train_items: List[Tuple[str, int]],
    val_items: List[Tuple[str, int]],
    test_items: List[Tuple[str, int]],
    classes: List[str],
    n_classes: int,
    cfg: TrainConfig,
    is_binary: bool = False,
) -> Dict[str, Any]:
    """Run one complete training cycle with pre-split data."""

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if model_name == "maxvit_t" and cfg.img_size % 32 != 0:
        raise ValueError(f"MaxViT requires img_size divisible by 32, got {cfg.img_size}")

    with open(os.path.join(cfg.out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    print(f"  Data: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")

    # Transforms
    train_tf, eval_tf = build_transforms(cfg.img_size)

    # Datasets & loaders
    dl_kwargs = {}
    if sys.platform == "win32" and cfg.num_workers > 0:
        dl_kwargs["persistent_workers"] = True

    ds_train = SimpleImageDataset(train_items, transform=train_tf)
    ds_val = SimpleImageDataset(val_items, transform=eval_tf)
    ds_test = SimpleImageDataset(test_items, transform=eval_tf)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, **dl_kwargs)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True, **dl_kwargs)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = model_builder(n_classes).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    if cfg.class_weight == "auto":
        counts = [0] * n_classes
        for _, y in train_items:
            counts[y] += 1
        total = sum(counts)
        weights = [total / (n_classes * max(1, c)) for c in counts]
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=cfg.label_smoothing)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_acc = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pth")
    history: List[Dict] = []
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, dl_train, criterion, optimizer, device, scaler, cfg.accum_steps)
        val_loss, val_acc, val_cm, _ = evaluate(model, dl_val, criterion, device, n_classes)
        if scheduler is not None:
            scheduler.step()
        val_f1 = macro_f1_from_cm(val_cm)
        elapsed = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "time_sec": round(elapsed, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(rec)
        print(f"  Epoch {epoch:03d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f} time={elapsed:.1f}s")
        save_json(history, os.path.join(cfg.out_dir, "metrics.json"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "model_arch": model_name,
                "classes": classes,
                "n_classes": n_classes,
                "cfg": {
                    "epochs": cfg.epochs, "img_size": cfg.img_size, "lr": cfg.lr,
                    "wd": cfg.wd, "optimizer": cfg.optimizer, "batch_size": cfg.batch_size,
                    "label_smoothing": cfg.label_smoothing, "sched": cfg.sched,
                },
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Test evaluation on best model ──
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, test_cm, test_probs = evaluate(
        model, dl_test, criterion, device, n_classes, want_probs=is_binary)
    test_f1 = macro_f1_from_cm(test_cm)

    # ROC-AUC for binary
    roc_auc = None
    if is_binary and roc_auc_score is not None and test_probs is not None:
        try:
            y_true = torch.tensor([y for _, y in test_items], dtype=torch.int64)
            y_score = test_probs[:, 1].numpy()
            roc_auc = float(roc_auc_score(y_true.numpy(), y_score))
        except Exception:
            roc_auc = None

    print(f"  Test: acc={test_acc:.4f} macro_f1={test_f1:.4f}"
          + (f" roc_auc={roc_auc:.4f}" if roc_auc is not None else ""))

    # Save test report
    report = {
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "test_macro_f1": round(test_f1, 6),
        "roc_auc": roc_auc,
        "confusion_matrix": test_cm.tolist(),
        "classes": classes,
        "best_epoch": ckpt.get("epoch"),
        "best_val_acc": round(ckpt.get("val_acc", 0.0), 6),
        "model_arch": model_name,
    }
    save_json(report, os.path.join(cfg.out_dir, "test_report.json"))

    # Generate all visual reports
    generate_full_report(report, history, cfg.out_dir, model_name, cfg.out_dir.split(os.sep)[-1])

    # Cleanup
    del model, optimizer, scheduler, scaler, criterion
    del dl_train, dl_val, dl_test, ds_train, ds_val, ds_test

    return {
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "best_epoch": ckpt.get("epoch"),
        "roc_auc": roc_auc,
        "status": "OK",
    }


# ── Summary Table ────────────────────────────────────────────────────────────

def print_summary_table(summary: List[Dict]):
    print(f"\n{'=' * 85}")
    print("MULTI-MODEL TRAINING SUMMARY")
    print(f"{'=' * 85}")
    header = f"{'Run':<40} {'Accuracy':>10} {'Macro F1':>10} {'Epoch':>7} {'Status':>12}"
    print(header)
    print("-" * 85)
    for row in summary:
        acc = f"{row['test_acc']:.4f}" if row['test_acc'] is not None else "N/A"
        f1 = f"{row['test_macro_f1']:.4f}" if row['test_macro_f1'] is not None else "N/A"
        ep = str(row['best_epoch']) if row['best_epoch'] is not None else "N/A"
        status = row["status"][:12]
        print(f"{row['run']:<40} {acc:>10} {f1:>10} {ep:>7} {status:>12}")
    print(f"{'=' * 85}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 5 models x 3 stages for fabric classification (pre-split data)")
    parser.add_argument("--data-dir", default=".",
                        help="Root dir containing train/val/test subdirs (default: current dir)")
    parser.add_argument("--runs-dir", default="runs",
                        help="Parent dir for all run outputs (default: runs/)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--sched", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (recommended for GPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["none", "auto"], default="none")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models (default: all). "
                             "Choices: efficientnet_v2_s maxvit_t regnet_y_8gf "
                             "densenet161 resnext101_32x8d")
    parser.add_argument("--stages", nargs="*", default=None,
                        help="Subset of stages (default: all). "
                             "Choices: stage1 woven knit")
    return parser


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate model names
    model_names = args.models if args.models else list(MODEL_REGISTRY.keys())
    for m in model_names:
        if m not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model: {m}. Choose from: {list(MODEL_REGISTRY.keys())}")

    # Stage filter
    all_stage_names = ["stage1", "woven", "knit"]
    stage_filter = set(args.stages) if args.stages else set(all_stage_names)

    # Load pre-split data
    print(f"{'=' * 70}")
    print(f"Multi-Model Fabric Classification Training")
    print(f"{'=' * 70}")
    print(f"\nLoading data from: {os.path.abspath(args.data_dir)}")

    raw_train, raw_val, raw_test, all_classes = load_presplit(args.data_dir)
    print(f"  All classes ({len(all_classes)}): {all_classes}")
    print(f"  Raw split: {len(raw_train)} train / {len(raw_val)} val / {len(raw_test)} test")

    # Build stage data by transforming raw data
    stage_defs = []

    if "stage1" in stage_filter:
        s1_train, s1_classes = remap_to_binary(raw_train, all_classes)
        s1_val, _ = remap_to_binary(raw_val, all_classes)
        s1_test, _ = remap_to_binary(raw_test, all_classes)
        stage_defs.append({
            "name": "stage1",
            "display": "Stage1 Binary (Knit vs Woven)",
            "train": s1_train, "val": s1_val, "test": s1_test,
            "classes": s1_classes, "is_binary": True,
        })
        print(f"  Stage1 Binary: {len(s1_classes)} classes {s1_classes} "
              f"({len(s1_train)}/{len(s1_val)}/{len(s1_test)})")

    if "woven" in stage_filter:
        w_train, w_classes = filter_by_prefix(raw_train, all_classes, "Woven")
        w_val, _ = filter_by_prefix(raw_val, all_classes, "Woven")
        w_test, _ = filter_by_prefix(raw_test, all_classes, "Woven")
        stage_defs.append({
            "name": "woven",
            "display": f"Stage2 Woven ({len(w_classes)}-class)",
            "train": w_train, "val": w_val, "test": w_test,
            "classes": w_classes, "is_binary": False,
        })
        print(f"  Stage2 Woven: {len(w_classes)} classes {w_classes} "
              f"({len(w_train)}/{len(w_val)}/{len(w_test)})")

    if "knit" in stage_filter:
        k_train, k_classes = filter_by_prefix(raw_train, all_classes, "Knit")
        k_val, _ = filter_by_prefix(raw_val, all_classes, "Knit")
        k_test, _ = filter_by_prefix(raw_test, all_classes, "Knit")
        stage_defs.append({
            "name": "knit",
            "display": f"Stage2 Knit ({len(k_classes)}-class)",
            "train": k_train, "val": k_val, "test": k_test,
            "classes": k_classes, "is_binary": False,
        })
        print(f"  Stage2 Knit: {len(k_classes)} classes {k_classes} "
              f"({len(k_train)}/{len(k_val)}/{len(k_test)})")

    total_runs = len(model_names) * len(stage_defs)
    print(f"\nModels: {model_names}")
    print(f"Stages: {[s['name'] for s in stage_defs]}")
    print(f"Total runs: {total_runs}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | AMP: {args.amp}")
    print(f"{'=' * 70}")

    if total_runs == 0:
        raise SystemExit("No valid runs to execute.")

    # Main training loop
    run_idx = 0
    summary: List[Dict] = []
    total_t0 = time.time()

    for model_idx, model_name in enumerate(model_names, 1):
        model_t0 = time.time()
        model_results: List[Dict] = []

        print(f"\n{'#' * 70}")
        print(f"# MODEL {model_idx}/{len(model_names)}: {model_name}")
        print(f"{'#' * 70}")

        for stage in stage_defs:
            run_idx += 1
            run_dir_name = f"{model_name}_{stage['name']}"
            out_dir = os.path.join(args.runs_dir, run_dir_name)

            print(f"\n{'=' * 70}")
            print(f"[{run_idx}/{total_runs}] {model_name} >> {stage['display']}")
            print(f"  Output: {out_dir}")
            print(f"{'=' * 70}")

            cfg = TrainConfig(
                out_dir=out_dir,
                epochs=args.epochs,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                lr=args.lr,
                wd=args.wd,
                optimizer=args.optimizer,
                sched=args.sched,
                amp=args.amp,
                seed=args.seed,
                class_weight=args.class_weight,
                accum_steps=max(1, args.accum_steps),
                patience=args.patience,
                label_smoothing=args.label_smoothing,
            )

            try:
                result = run_single_training(
                    model_name=model_name,
                    model_builder=MODEL_REGISTRY[model_name],
                    train_items=stage["train"],
                    val_items=stage["val"],
                    test_items=stage["test"],
                    classes=stage["classes"],
                    n_classes=len(stage["classes"]),
                    cfg=cfg,
                    is_binary=stage["is_binary"],
                )
                row = {
                    "run": run_dir_name,
                    "model": model_name,
                    "stage": stage["name"],
                    "test_acc": result["test_acc"],
                    "test_macro_f1": result["test_macro_f1"],
                    "best_epoch": result["best_epoch"],
                    "roc_auc": result.get("roc_auc"),
                    "status": "OK",
                }
            except Exception as e:
                tb = traceback.format_exc()
                print(f"\n  [ERROR] {run_dir_name} FAILED:\n{tb}")
                row = {
                    "run": run_dir_name,
                    "model": model_name,
                    "stage": stage["name"],
                    "test_acc": None,
                    "test_macro_f1": None,
                    "best_epoch": None,
                    "roc_auc": None,
                    "status": f"FAILED: {str(e)[:50]}",
                }

            summary.append(row)
            model_results.append(row)
            cleanup_gpu()

            # Save incremental summary after every run
            save_json(summary, os.path.join(args.runs_dir, "multi_model_summary.json"))

        # ── Per-model summary ──
        model_elapsed = time.time() - model_t0
        print(f"\n{'*' * 70}")
        print(f"* {model_name} DONE  ({model_elapsed / 60:.1f} min)")
        print(f"{'*' * 70}")
        for r in model_results:
            acc = f"{r['test_acc']:.4f}" if r['test_acc'] is not None else "FAIL"
            f1 = f"{r['test_macro_f1']:.4f}" if r['test_macro_f1'] is not None else "FAIL"
            print(f"  {r['stage']:<12} acc={acc}  f1={f1}  status={r['status']}")
        remaining = len(model_names) - model_idx
        if remaining > 0:
            print(f"  >> {remaining} model(s) remaining")
        print(f"{'*' * 70}")

    total_elapsed = time.time() - total_t0
    print(f"\nAll runs completed in {total_elapsed / 60:.1f} minutes.")

    # Summary
    print_summary_table(summary)
    save_json(summary, os.path.join(args.runs_dir, "multi_model_summary.json"))
    print(f"\nSummary saved to: {os.path.join(args.runs_dir, 'multi_model_summary.json')}")


if __name__ == "__main__":
    main()
