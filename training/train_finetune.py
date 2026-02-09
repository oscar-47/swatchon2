"""
Fine-tuning Experiment Script for Fabric Classification
========================================================
Systematic fine-tuning with 5 layer-freezing strategies across multiple models.

Strategies:
  A: Full fine-tune — all layers trainable, lr=1e-4
  B: Head-only — freeze all backbone, lr=1e-3
  C: Last stage — freeze stem + stages 1-3, lr=5e-4
  D: Last 2 stages — freeze stem + stages 1-2, lr=5e-4
  E: Discriminative LR — all trainable, head=1e-3, mid=3e-4, low=1e-4

Models: regnet_y_8gf, convnext_base, maxvit_t, densenet161
Tasks: stage1 (binary Knit/Woven), woven (5-class), knit (5-class)

Expects pre-split dataset:
  <data-dir>/train/<ClassName>/images...
  <data-dir>/val/<ClassName>/images...
  <data-dir>/test/<ClassName>/images...

Usage:
  python train_finetune.py --data-dir . --amp
  python train_finetune.py --data-dir . --amp --models regnet_y_8gf --strategies A --tasks stage1 --epochs 2
"""

import argparse
import gc
import json
import math
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


# ── Model Builders ───────────────────────────────────────────────────────────

def _build_regnet_y_8gf(n_classes: int) -> nn.Module:
    model = models.regnet_y_8gf(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def _build_convnext_base(n_classes: int) -> nn.Module:
    model = models.convnext_base(weights="IMAGENET1K_V1")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_classes)
    return model


def _build_maxvit_t(n_classes: int) -> nn.Module:
    model = models.maxvit_t(weights="IMAGENET1K_V1")
    model.classifier[5] = nn.Linear(model.classifier[5].in_features, n_classes)
    return model


def _build_densenet161(n_classes: int) -> nn.Module:
    model = models.densenet161(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model


MODEL_REGISTRY: Dict[str, Any] = {
    "regnet_y_8gf":  _build_regnet_y_8gf,
    "convnext_base": _build_convnext_base,
    "maxvit_t":      _build_maxvit_t,
    "densenet161":   _build_densenet161,
}


# ── Stage Groups (module-based freezing) ─────────────────────────────────────

def get_stage_groups(model: nn.Module, model_name: str) -> List[Tuple[str, List[nn.Module]]]:
    """Return ordered (group_name, [modules]) using direct module references.

    Groups ordered shallowest (stem) to deepest (head).
    Uses direct references — NOT string name matching — to avoid silent bugs.
    """
    if model_name == "regnet_y_8gf":
        return [
            ("stem",   [model.stem]),
            ("stage1", [model.trunk_output[0]]),
            ("stage2", [model.trunk_output[1]]),
            ("stage3", [model.trunk_output[2]]),
            ("stage4", [model.trunk_output[3]]),
            ("head",   [model.fc]),
        ]
    elif model_name == "convnext_base":
        # features: [0]=stem, [1]=stage1, [2]=ds1→2, [3]=stage2, [4]=ds2→3,
        #           [5]=stage3, [6]=ds3→4, [7]=stage4
        return [
            ("stem",   [model.features[0]]),
            ("stage1", [model.features[1], model.features[2]]),
            ("stage2", [model.features[3], model.features[4]]),
            ("stage3", [model.features[5], model.features[6]]),
            ("stage4", [model.features[7]]),
            ("head",   [model.classifier]),
        ]
    elif model_name == "maxvit_t":
        return [
            ("stem",   [model.stem]),
            ("stage1", [model.blocks[0]]),
            ("stage2", [model.blocks[1]]),
            ("stage3", [model.blocks[2]]),
            ("stage4", [model.blocks[3]]),
            ("head",   [model.classifier]),
        ]
    elif model_name == "densenet161":
        f = model.features
        return [
            ("stem",   [f.conv0, f.norm0, f.relu0, f.pool0]),
            ("stage1", [f.denseblock1, f.transition1]),
            ("stage2", [f.denseblock2, f.transition2]),
            ("stage3", [f.denseblock3, f.transition3]),
            ("stage4", [f.denseblock4, f.norm5]),
            ("head",   [model.classifier]),
        ]
    else:
        raise ValueError(f"No stage groups defined for: {model_name}")


# ── Freezing Strategies ──────────────────────────────────────────────────────

# freeze_through: last group to freeze (inclusive). None = freeze nothing.
STRATEGIES = {
    "A": {"desc": "Full fine-tune",    "freeze_through": None,     "base_lr": 1e-4},
    "B": {"desc": "Head-only",         "freeze_through": "stage4", "base_lr": 1e-3},
    "C": {"desc": "Last stage",        "freeze_through": "stage3", "base_lr": 5e-4},
    "D": {"desc": "Last 2 stages",     "freeze_through": "stage2", "base_lr": 5e-4},
    "E": {"desc": "Discriminative LR", "freeze_through": None,     "base_lr": None,
           "group_lrs": {"stem": 1e-4, "stage1": 1e-4, "stage2": 1e-4,
                         "stage3": 3e-4, "stage4": 3e-4, "head": 1e-3}},
}


def apply_freeze(
    stage_groups: List[Tuple[str, List[nn.Module]]],
    freeze_through: Optional[str],
) -> List[Tuple[str, List[nn.Module]]]:
    """Freeze params AND set modules to eval() to prevent BN/LN running_stats drift.

    Returns list of (group_name, modules) that were frozen.
    """
    frozen = []
    if freeze_through is None:
        return frozen

    for group_name, modules in stage_groups:
        for m in modules:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        frozen.append((group_name, modules))
        if group_name == freeze_through:
            break

    return frozen


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ── Optimizer ────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    stage_groups: List[Tuple[str, List[nn.Module]]],
    strategy: Dict,
    wd: float,
) -> torch.optim.Optimizer:
    """Build AdamW with correct param groups per strategy."""
    if strategy.get("group_lrs"):
        # Strategy E: discriminative LR per group
        group_lrs = strategy["group_lrs"]
        param_groups = []
        for group_name, modules in stage_groups:
            lr = group_lrs.get(group_name, 1e-4)
            params = []
            for m in modules:
                params.extend([p for p in m.parameters() if p.requires_grad])
            if params:
                param_groups.append({"params": params, "lr": lr})
        if not param_groups:
            raise ValueError("No trainable parameters!")
        return torch.optim.AdamW(param_groups, weight_decay=wd)
    else:
        # Uniform LR
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters after freezing!")
        return torch.optim.AdamW(trainable, lr=strategy["base_lr"], weight_decay=wd)


# ── Warmup + Cosine Scheduler ───────────────────────────────────────────────

def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """LambdaLR with linear warmup then cosine decay. Steps per optimizer step."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    out_dir: str
    epochs: int = 30
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    wd: float = 0.01
    amp: bool = True
    seed: int = 42
    class_weight: str = "auto"
    accum_steps: int = 1
    patience: int = 7
    label_smoothing: float = 0.05
    warmup_pct: float = 0.05


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


# ── Data Loading ─────────────────────────────────────────────────────────────

def scan_imagefolder(root: str) -> Tuple[List[Tuple[str, int]], List[str]]:
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


def remap_to_binary(items, classes):
    binary_classes = ["Knit", "Woven"]
    new_items = [(p, 0 if classes[l].startswith("Knit") else 1) for p, l in items]
    return new_items, binary_classes


def filter_by_prefix(items, classes, prefix):
    filtered = sorted([c for c in classes if c.startswith(prefix)])
    c2i = {c: i for i, c in enumerate(filtered)}
    new_items = [(p, c2i[classes[l]]) for p, l in items if classes[l].startswith(prefix)]
    return new_items, filtered


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

def confusion_matrix_fn(preds: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
    cm = torch.zeros((n, n), dtype=torch.long)
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
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def weighted_f1_from_cm(cm: torch.Tensor) -> float:
    total = cm.sum().item()
    if total == 0:
        return 0.0
    weighted = 0.0
    for c in range(cm.size(0)):
        tp = cm[c, c].item()
        fp = int(cm[:, c].sum().item() - tp)
        fn = int(cm[c, :].sum().item() - tp)
        support = cm[c, :].sum().item()
        denom = 2 * tp + fp + fn
        weighted += ((2 * tp / denom) if denom > 0 else 0.0) * support
    return weighted / total


# ── Utilities ────────────────────────────────────────────────────────────────

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows, header, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')


# ── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler, accum_steps, scheduler, frozen_groups):
    """Training with frozen BN/LN handling and per-step scheduling."""
    model.train()
    # Critical: re-set frozen modules to eval() after model.train()
    for _, modules in frozen_groups:
        for m in modules:
            m.eval()

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
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * x.size(0)
        running_acc += (logits.argmax(1) == y).float().sum().item()
        n += x.size(0)

    return running_loss / max(1, n), running_acc / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes, want_probs=False):
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
    cm = confusion_matrix_fn(preds, targets, n_classes)
    probs_all = torch.cat(all_probs) if (want_probs and all_probs) else None
    return running_loss / n, running_acc / n, cm, probs_all


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history, out_path):
    epochs = [r["epoch"] for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, [r["train_loss"] for r in history], "o-", label="Train", color="#1f77b4")
    ax1.plot(epochs, [r["val_loss"] for r in history], "o-", label="Val", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [r["train_acc"] for r in history], "o-", label="Train", color="#2ca02c")
    ax2.plot(epochs, [r["val_acc"] for r in history], "o-", label="Val", color="#d62728")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, classes, out_path, title):
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


def plot_per_class_bar(values, classes, out_path, title, ylabel):
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


def plot_strategy_comparison(summary, out_path):
    """Bar chart comparing strategies across models and tasks."""
    ok_runs = [r for r in summary if r["status"] == "OK"]
    if not ok_runs:
        return
    tasks = sorted(set(r["task"] for r in ok_runs))
    mdl_names = sorted(set(r["model"] for r in ok_runs))
    strats = sorted(set(r["strategy"] for r in ok_runs))

    n_tasks = len(tasks)
    if n_tasks == 0:
        return

    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 5.5), squeeze=False)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(mdl_names), 1)))
    bar_w = 0.8 / max(1, len(mdl_names))

    for ti, task in enumerate(tasks):
        ax = axes[0, ti]
        for mi, mdl in enumerate(mdl_names):
            vals = []
            for s in strats:
                matches = [r for r in ok_runs
                           if r["task"] == task and r["model"] == mdl and r["strategy"] == s]
                if matches:
                    f1_vals = [r["test_macro_f1"] for r in matches if r["test_macro_f1"] is not None]
                    vals.append(np.mean(f1_vals) if f1_vals else 0)
                else:
                    vals.append(0)
            x = np.arange(len(strats)) + mi * bar_w
            ax.bar(x, vals, bar_w, label=mdl, color=colors[mi])

        ax.set_xticks(np.arange(len(strats)) + bar_w * (len(mdl_names) - 1) / 2)
        ax.set_xticklabels([f"Strat {s}" for s in strats])
        ax.set_ylabel("Macro F1")
        ax.set_title(f"Task: {task}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Strategy Comparison — Test Macro F1", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ── Report Generation ────────────────────────────────────────────────────────

def generate_full_report(test_report, history, out_dir, run_label):
    cm = np.array(test_report["confusion_matrix"], dtype=np.int64)
    classes = test_report["classes"]
    test_acc = float(test_report.get("test_acc", 0.0))
    test_f1 = float(test_report.get("test_macro_f1", 0.0))
    best_epoch = test_report.get("best_epoch", "?")
    best_val_f1 = float(test_report.get("best_val_f1", 0.0))
    roc_auc = test_report.get("roc_auc")
    trainable_pct = test_report.get("trainable_pct", "?")
    peak_gpu_mb = test_report.get("peak_gpu_mb")

    plot_training_curves(history, os.path.join(out_dir, "training_curves.png"))
    plot_confusion_matrix(
        cm, classes, os.path.join(out_dir, "confusion_matrix.png"),
        title=f"{run_label}\n(acc={test_acc:.3f}, F1={test_f1:.3f})")

    tp = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    col_sum = cm.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)

    plot_per_class_bar(recall, classes, os.path.join(out_dir, "per_class_accuracy.png"),
                       title="Per-class Accuracy (Recall)", ylabel="Accuracy")

    rows = []
    for i, c in enumerate(classes):
        rows.append([c, int(row_sum[i]), int(tp[i]), int(col_sum[i] - tp[i]),
                      int(row_sum[i] - tp[i]), f"{precision[i]:.4f}", f"{recall[i]:.4f}",
                      f"{f1[i]:.4f}", f"{recall[i]:.4f}"])
    save_csv(rows,
             header=["class", "support", "tp", "fp", "fn", "precision", "recall", "f1", "per_class_acc"],
             out_path=os.path.join(out_dir, "per_class_metrics.csv"))

    roc_line = f"<b>ROC-AUC</b>: {roc_auc:.3f} &nbsp; " if roc_auc is not None else ""
    gpu_line = f"<b>Peak GPU</b>: {peak_gpu_mb:.0f} MB &nbsp; " if peak_gpu_mb else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'/><title>{run_label}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 16px; }}
section {{ margin-bottom: 24px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
</style></head>
<body>
<h2>{run_label}</h2>
<p>
  <b>Test Acc</b>: {test_acc:.3f} &nbsp;
  <b>Macro F1</b>: {test_f1:.3f} &nbsp;
  {roc_line}{gpu_line}
  <b>Trainable</b>: {trainable_pct} &nbsp;
  <b>Best Val F1</b>: {best_val_f1:.3f} (epoch {best_epoch})
</p>
<section><h3>Training Curves</h3><img src="training_curves.png"/></section>
<section><h3>Confusion Matrix</h3><img src="confusion_matrix.png"/></section>
<section><h3>Per-class Accuracy</h3><img src="per_class_accuracy.png"/>
<p>Details: <code>per_class_metrics.csv</code></p></section>
</body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ── GPU Cleanup ──────────────────────────────────────────────────────────────

def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Core Fine-tuning Run ────────────────────────────────────────────────────

def run_single_finetune(
    model_name: str,
    model_builder,
    strategy_name: str,
    strategy: Dict,
    train_items: List[Tuple[str, int]],
    val_items: List[Tuple[str, int]],
    test_items: List[Tuple[str, int]],
    classes: List[str],
    n_classes: int,
    cfg: FinetuneConfig,
    is_binary: bool = False,
) -> Dict[str, Any]:
    """Run one complete fine-tuning experiment."""

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    run_label = f"{model_name} / Strat-{strategy_name}"

    if model_name == "maxvit_t" and cfg.img_size % 32 != 0:
        raise ValueError(f"MaxViT requires img_size % 32 == 0, got {cfg.img_size}")

    with open(os.path.join(cfg.out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    print(f"  Strategy {strategy_name}: {strategy['desc']}")
    print(f"  Data: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")

    # Transforms & loaders
    train_tf, eval_tf = build_transforms(cfg.img_size)
    dl_kwargs = {}
    if sys.platform == "win32" and cfg.num_workers > 0:
        dl_kwargs["persistent_workers"] = True

    dl_train = DataLoader(SimpleImageDataset(train_items, train_tf),
                          batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, **dl_kwargs)
    dl_val = DataLoader(SimpleImageDataset(val_items, eval_tf),
                        batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, **dl_kwargs)
    dl_test = DataLoader(SimpleImageDataset(test_items, eval_tf),
                         batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True, **dl_kwargs)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_builder(n_classes).to(device)

    # Stage groups & freezing
    stage_groups = get_stage_groups(model, model_name)
    frozen_groups = apply_freeze(stage_groups, strategy.get("freeze_through"))

    trainable, total = count_parameters(model)
    trainable_pct = f"{trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / max(1, total):.1f}%)"
    print(f"  Trainable: {trainable_pct}")
    if frozen_groups:
        print(f"  Frozen: {[g[0] for g in frozen_groups]}")

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    if cfg.class_weight == "auto":
        counts = [0] * n_classes
        for _, y in train_items:
            counts[y] += 1
        total_samples = sum(counts)
        weights = [total_samples / (n_classes * max(1, c)) for c in counts]
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=cfg.label_smoothing)

    # Optimizer
    optimizer = build_optimizer(model, stage_groups, strategy, cfg.wd)

    # Scheduler: warmup + cosine (per optimizer-step)
    steps_per_epoch = max(1, len(dl_train) // max(1, cfg.accum_steps))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_pct * total_steps)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Scheduler: warmup {warmup_steps} / {total_steps} total steps")

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # Reset GPU memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pth")
    history: List[Dict] = []
    epochs_no_improve = 0
    total_images = 0
    train_t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, dl_train, criterion, optimizer, device,
            scaler, cfg.accum_steps, scheduler, frozen_groups)

        val_loss, val_acc, val_cm, _ = evaluate(model, dl_val, criterion, device, n_classes)
        val_f1 = macro_f1_from_cm(val_cm)
        elapsed = time.time() - t0
        total_images += len(train_items)

        lr_now = optimizer.param_groups[0]["lr"]
        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "time_sec": round(elapsed, 2),
            "lr": lr_now,
        }
        history.append(rec)
        print(f"  Epoch {epoch:03d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f} lr={lr_now:.2e} t={elapsed:.1f}s")
        save_json(history, os.path.join(cfg.out_dir, "metrics.json"))

        # Best checkpoint by val_macro_f1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model": model.state_dict(),
                "model_arch": model_name,
                "strategy": strategy_name,
                "classes": classes,
                "n_classes": n_classes,
                "cfg": {
                    "epochs": cfg.epochs, "img_size": cfg.img_size,
                    "wd": cfg.wd, "batch_size": cfg.batch_size,
                    "label_smoothing": cfg.label_smoothing,
                    "strategy": strategy_name, "seed": cfg.seed,
                },
                "val_f1": val_f1,
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_elapsed = time.time() - train_t0
    throughput = total_images / max(1, train_elapsed)

    # GPU memory
    peak_gpu_mb = None
    if device.type == "cuda":
        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Test on best model
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, test_cm, test_probs = evaluate(
        model, dl_test, criterion, device, n_classes, want_probs=is_binary)
    test_f1 = macro_f1_from_cm(test_cm)
    test_wf1 = weighted_f1_from_cm(test_cm)

    # ROC-AUC for binary
    roc_auc = None
    if is_binary and roc_auc_score is not None and test_probs is not None:
        try:
            y_true = torch.tensor([y for _, y in test_items], dtype=torch.int64)
            y_score = test_probs[:, 1].numpy()
            roc_auc = float(roc_auc_score(y_true.numpy(), y_score))
        except Exception:
            roc_auc = None

    print(f"  Test: acc={test_acc:.4f} macro_f1={test_f1:.4f} weighted_f1={test_wf1:.4f}"
          + (f" roc_auc={roc_auc:.4f}" if roc_auc is not None else "")
          + (f" peak_gpu={peak_gpu_mb:.0f}MB" if peak_gpu_mb else "")
          + f" throughput={throughput:.0f} img/s")

    # Save report
    report = {
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "test_macro_f1": round(test_f1, 6),
        "test_weighted_f1": round(test_wf1, 6),
        "roc_auc": roc_auc,
        "confusion_matrix": test_cm.tolist(),
        "classes": classes,
        "best_epoch": ckpt.get("epoch"),
        "best_val_f1": round(ckpt.get("val_f1", 0.0), 6),
        "best_val_acc": round(ckpt.get("val_acc", 0.0), 6),
        "model_arch": model_name,
        "strategy": strategy_name,
        "trainable_pct": trainable_pct,
        "trainable_params": trainable,
        "total_params": total,
        "peak_gpu_mb": peak_gpu_mb,
        "throughput_img_per_sec": round(throughput, 1),
        "seed": cfg.seed,
    }
    save_json(report, os.path.join(cfg.out_dir, "test_report.json"))
    generate_full_report(report, history, cfg.out_dir, run_label)

    # Cleanup
    del model, optimizer, scheduler, scaler, criterion
    del dl_train, dl_val, dl_test

    return {
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_weighted_f1": test_wf1,
        "best_epoch": ckpt.get("epoch"),
        "roc_auc": roc_auc,
        "trainable_pct": trainable_pct,
        "peak_gpu_mb": peak_gpu_mb,
        "throughput": throughput,
        "status": "OK",
    }


# ── Summary Table ────────────────────────────────────────────────────────────

def print_summary_table(summary):
    print(f"\n{'=' * 105}")
    print("FINE-TUNING EXPERIMENT SUMMARY")
    print(f"{'=' * 105}")
    hdr = (f"{'Run':<45} {'Strat':>5} {'Acc':>8} {'MacroF1':>8} "
           f"{'WtdF1':>8} {'Epoch':>6} {'GPU MB':>8} {'Status':>10}")
    print(hdr)
    print("-" * 105)
    for r in summary:
        acc = f"{r['test_acc']:.4f}" if r.get('test_acc') is not None else "N/A"
        mf1 = f"{r['test_macro_f1']:.4f}" if r.get('test_macro_f1') is not None else "N/A"
        wf1 = f"{r.get('test_weighted_f1', 0):.4f}" if r.get('test_weighted_f1') is not None else "N/A"
        ep = str(r.get('best_epoch', 'N/A'))
        gpu = f"{r.get('peak_gpu_mb', 0):.0f}" if r.get('peak_gpu_mb') else "N/A"
        st = r.get("status", "?")[:10]
        print(f"{r['run']:<45} {r.get('strategy','?'):>5} {acc:>8} {mf1:>8} "
              f"{wf1:>8} {ep:>6} {gpu:>8} {st:>10}")
    print(f"{'=' * 105}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Fine-tuning experiments with layer-freezing strategies")
    p.add_argument("--data-dir", default=".",
                   help="Root containing train/val/test (default: .)")
    p.add_argument("--runs-dir", default="runs",
                   help="Output parent directory (default: runs/)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--amp", action="store_true", help="Enable AMP (recommended)")
    p.add_argument("--seeds", nargs="*", type=int, default=[42],
                   help="Random seeds; multiple for Phase 2 validation (default: 42)")
    p.add_argument("--class-weight", choices=["none", "auto"], default="auto")
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--warmup-pct", type=float, default=0.05,
                   help="Warmup fraction of total optimizer steps (default: 0.05)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Model subset. Choices: " + " ".join(MODEL_REGISTRY.keys()))
    p.add_argument("--strategies", nargs="*", default=None,
                   help="Strategy subset. Choices: A B C D E")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Task subset. Choices: stage1 woven knit")
    return p


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = build_arg_parser().parse_args()

    # Validate
    model_names = args.models if args.models else list(MODEL_REGISTRY.keys())
    for m in model_names:
        if m not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model: {m}. Choose from: {list(MODEL_REGISTRY.keys())}")

    strat_names = args.strategies if args.strategies else list(STRATEGIES.keys())
    for s in strat_names:
        if s not in STRATEGIES:
            raise SystemExit(f"Unknown strategy: {s}. Choose from: {list(STRATEGIES.keys())}")

    task_filter = set(args.tasks) if args.tasks else {"stage1", "woven", "knit"}
    seeds = args.seeds if args.seeds else [42]
    multi_seed = len(seeds) > 1

    # Load data
    print(f"{'=' * 70}")
    print("Fine-tuning Experiment — Fabric Classification")
    print(f"{'=' * 70}")
    print(f"\nData: {os.path.abspath(args.data_dir)}")

    raw_train, raw_val, raw_test, all_classes = load_presplit(args.data_dir)
    print(f"  Classes ({len(all_classes)}): {all_classes}")
    print(f"  Split: {len(raw_train)} train / {len(raw_val)} val / {len(raw_test)} test")

    # Build task data
    task_defs = []
    if "stage1" in task_filter:
        s1_tr, s1_cls = remap_to_binary(raw_train, all_classes)
        s1_va, _ = remap_to_binary(raw_val, all_classes)
        s1_te, _ = remap_to_binary(raw_test, all_classes)
        task_defs.append({"name": "stage1", "display": "Binary (Knit vs Woven)",
                          "train": s1_tr, "val": s1_va, "test": s1_te,
                          "classes": s1_cls, "is_binary": True})
    if "woven" in task_filter:
        w_tr, w_cls = filter_by_prefix(raw_train, all_classes, "Woven")
        w_va, _ = filter_by_prefix(raw_val, all_classes, "Woven")
        w_te, _ = filter_by_prefix(raw_test, all_classes, "Woven")
        task_defs.append({"name": "woven", "display": f"Woven ({len(w_cls)}-class)",
                          "train": w_tr, "val": w_va, "test": w_te,
                          "classes": w_cls, "is_binary": False})
    if "knit" in task_filter:
        k_tr, k_cls = filter_by_prefix(raw_train, all_classes, "Knit")
        k_va, _ = filter_by_prefix(raw_val, all_classes, "Knit")
        k_te, _ = filter_by_prefix(raw_test, all_classes, "Knit")
        task_defs.append({"name": "knit", "display": f"Knit ({len(k_cls)}-class)",
                          "train": k_tr, "val": k_va, "test": k_te,
                          "classes": k_cls, "is_binary": False})

    total_runs = len(model_names) * len(strat_names) * len(task_defs) * len(seeds)
    print(f"\nModels: {model_names}")
    print(f"Strategies: {strat_names}")
    print(f"Tasks: {[t['name'] for t in task_defs]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    print(f"Epochs={args.epochs} Batch={args.batch_size} WD={args.wd} Patience={args.patience}")
    print(f"{'=' * 70}")

    if total_runs == 0:
        raise SystemExit("No valid runs.")

    os.makedirs(args.runs_dir, exist_ok=True)
    run_idx = 0
    summary: List[Dict] = []
    total_t0 = time.time()

    for model_name in model_names:
        for strat_name in strat_names:
            strat = STRATEGIES[strat_name]
            for task in task_defs:
                for seed in seeds:
                    run_idx += 1

                    dir_name = f"{model_name}_strat{strat_name}_{task['name']}"
                    if multi_seed:
                        dir_name += f"_seed{seed}"
                    out_dir = os.path.join(args.runs_dir, dir_name)

                    print(f"\n{'=' * 70}")
                    print(f"[{run_idx}/{total_runs}] {model_name} | Strat-{strat_name} "
                          f"| {task['display']}"
                          + (f" | seed={seed}" if multi_seed else ""))
                    print(f"  Output: {out_dir}")
                    print(f"{'=' * 70}")

                    cfg = FinetuneConfig(
                        out_dir=out_dir,
                        epochs=args.epochs,
                        img_size=args.img_size,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        wd=args.wd,
                        amp=args.amp,
                        seed=seed,
                        class_weight=args.class_weight,
                        accum_steps=max(1, args.accum_steps),
                        patience=args.patience,
                        label_smoothing=args.label_smoothing,
                        warmup_pct=args.warmup_pct,
                    )

                    try:
                        result = run_single_finetune(
                            model_name=model_name,
                            model_builder=MODEL_REGISTRY[model_name],
                            strategy_name=strat_name,
                            strategy=strat,
                            train_items=task["train"],
                            val_items=task["val"],
                            test_items=task["test"],
                            classes=task["classes"],
                            n_classes=len(task["classes"]),
                            cfg=cfg,
                            is_binary=task["is_binary"],
                        )
                        row = {
                            "run": dir_name,
                            "model": model_name,
                            "strategy": strat_name,
                            "task": task["name"],
                            "seed": seed,
                            "test_acc": result["test_acc"],
                            "test_macro_f1": result["test_macro_f1"],
                            "test_weighted_f1": result.get("test_weighted_f1"),
                            "best_epoch": result["best_epoch"],
                            "roc_auc": result.get("roc_auc"),
                            "peak_gpu_mb": result.get("peak_gpu_mb"),
                            "status": "OK",
                        }
                    except Exception as e:
                        print(f"\n  [ERROR] {dir_name} FAILED:\n{traceback.format_exc()}")
                        row = {
                            "run": dir_name,
                            "model": model_name,
                            "strategy": strat_name,
                            "task": task["name"],
                            "seed": seed,
                            "test_acc": None,
                            "test_macro_f1": None,
                            "test_weighted_f1": None,
                            "best_epoch": None,
                            "roc_auc": None,
                            "peak_gpu_mb": None,
                            "status": f"FAILED: {str(e)[:50]}",
                        }

                    summary.append(row)
                    cleanup_gpu()
                    # Incremental save after every run
                    save_json(summary, os.path.join(args.runs_dir, "finetune_summary.json"))

    total_elapsed = time.time() - total_t0
    print(f"\nAll runs completed in {total_elapsed / 60:.1f} minutes.")

    print_summary_table(summary)

    # Strategy comparison chart
    try:
        plot_strategy_comparison(
            summary, os.path.join(args.runs_dir, "finetune_comparison.png"))
        print(f"Comparison chart: {os.path.join(args.runs_dir, 'finetune_comparison.png')}")
    except Exception as e:
        print(f"Warning: could not generate comparison chart: {e}")

    save_json(summary, os.path.join(args.runs_dir, "finetune_summary.json"))
    print(f"Summary: {os.path.join(args.runs_dir, 'finetune_summary.json')}")


if __name__ == "__main__":
    main()
