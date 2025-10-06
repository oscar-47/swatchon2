import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def scan_two_roots(woven_root: str, knit_root: str) -> List[Tuple[str, int]]:
    """Return items list of (image_path, label) where label: 0=Knit, 1=Woven"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items: List[Tuple[str, int]] = []
    # Knit -> label 0
    if os.path.isdir(knit_root):
        for cat in os.listdir(knit_root):
            d = os.path.join(knit_root, cat)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if os.path.splitext(fn)[1].lower() in exts:
                    items.append((os.path.join(d, fn), 0))
    # Woven -> label 1
    if os.path.isdir(woven_root):
        for cat in os.listdir(woven_root):
            d = os.path.join(woven_root, cat)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if os.path.splitext(fn)[1].lower() in exts:
                    items.append((os.path.join(d, fn), 1))
    return items


def stratified_split(items: List[Tuple[str, int]], train_ratio=0.7, val_ratio=0.15, seed=42):
    by_class = {0: [], 1: []}
    for i, (_, y) in enumerate(items):
        by_class[y].append(i)
    rng = random.Random(seed)
    for k in by_class:
        rng.shuffle(by_class[k])
    train_idx, val_idx, test_idx = [], [], []
    for k in [0, 1]:
        lst = by_class[k]
        n = len(lst)
        n_train = max(1, int(n * train_ratio)) if n >= 3 else max(1, n - 1)
        n_val = max(1, int(n * val_ratio)) if n - n_train >= 2 else 1 if n >= 2 else 0
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = max(0, n - n_train - n_val)
        train_idx += lst[:n_train]
        val_idx += lst[n_train:n_train + n_val]
        test_idx += lst[n_train + n_val:]
    return train_idx, val_idx, test_idx


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


@dataclass
class TrainConfig:
    woven_root: str
    knit_root: str
    out_dir: str
    epochs: int
    img_size: int
    batch_size: int
    num_workers: int
    lr: float
    wd: float
    optimizer: str
    sched: str
    amp: bool
    seed: int
    class_weight: str
    accum_steps: int
    patience: int
    label_smoothing: float


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


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
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def build_model(n_classes: int) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, n_classes)
    return model


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
    else:
        return None


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, accum_steps=1):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
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
    all_preds = []
    all_targets = []
    all_probs = [] if want_probs else None
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        running_acc += (logits.argmax(1) == y).float().sum().item()
        n += x.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(y.cpu())
        if want_probs:
            probs = torch.softmax(logits, dim=1).detach().cpu()
            all_probs.append(probs)
    if n == 0:
        return 0.0, 0.0, torch.zeros((n_classes, n_classes), dtype=torch.long), None
    preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long)
    if preds.numel() > 0:
        cm = confusion_matrix(preds, targets, n_classes)
    probs_all = torch.cat(all_probs) if (want_probs and all_probs) else None
    return running_loss / n, running_acc / n, cm, probs_all


def main():
    parser = argparse.ArgumentParser(description="Binary classify Woven vs Knit using ResNet50")
    parser.add_argument("--woven-root", required=True)
    parser.add_argument("--knit-root", required=True)
    parser.add_argument("--out", default=os.path.join("runs", "woven_vs_knit"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--sched", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["none", "auto"], default="none")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    args = parser.parse_args()

    cfg = TrainConfig(
        woven_root=args.woven_root,
        knit_root=args.knit_root,
        out_dir=args.out,
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

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    # Data
    items = scan_two_roots(cfg.woven_root, cfg.knit_root)
    if not items:
        raise SystemExit("No images found under given roots")

    classes = ["Knit", "Woven"]

    train_idx, val_idx, test_idx = stratified_split(items, seed=cfg.seed)
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]

    with open(os.path.join(cfg.out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    train_tf, eval_tf = build_transforms(cfg.img_size)

    ds_train = SimpleImageDataset(train_items, transform=train_tf)
    ds_val = SimpleImageDataset(val_items, transform=eval_tf)
    ds_test = SimpleImageDataset(test_items, transform=eval_tf)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(2).to(device)

    # Loss (optionally class-weighted) + label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=float(getattr(cfg, 'label_smoothing', 0.0) or 0.0))
    if cfg.class_weight == "auto":
        counts = [0, 0]
        for _, y in train_items:
            counts[y] += 1
        total = sum(counts)
        weights = [total / (2 * max(1, c)) for c in counts]
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=float(getattr(cfg, 'label_smoothing', 0.0) or 0.0))

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_acc = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pth")
    history = []
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, dl_train, criterion, optimizer, device, scaler, cfg.accum_steps)
        val_loss, val_acc, val_cm, _ = evaluate(model, dl_val, criterion, device, 2, want_probs=False)
        if scheduler is not None:
            scheduler.step()
        val_f1 = macro_f1_from_cm(val_cm)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "time_sec": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(rec)
        print(f"Epoch {epoch:03d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={elapsed:.1f}s")
        save_json(history, os.path.join(cfg.out_dir, "metrics.json"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "cfg": vars(cfg),
                "val_acc": val_acc,
                "epoch": epoch,
            }, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test evaluation on best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, test_cm, test_probs = evaluate(model, dl_test, criterion, device, 2, want_probs=True)
    test_f1 = macro_f1_from_cm(test_cm)

    # Optional ROC-AUC for binary case
    roc_auc = None
    if roc_auc_score is not None and test_probs is not None and len(test_probs.shape) == 2 and test_probs.shape[1] == 2:
        try:
            # Positive class = Woven(label=1)
            y_true = torch.cat([y for _, y in ds_test], dim=0) if False else None  # placeholder to avoid IO; we'll reconstruct below
        except Exception:
            y_true = None
        # Reconstruct y_true from loader iteration (stored earlier not to save memory). Simpler: recompute via ds_test indices
        # Here we re-iterate test_items to get labels
        y_true_list = [y for _, y in test_items]
        y_true = torch.tensor(y_true_list, dtype=torch.int64)
        y_score = test_probs[:, 1].numpy()
        try:
            roc_auc = float(roc_auc_score(y_true.numpy(), y_score))
        except Exception:
            roc_auc = None

    print(f"Test: acc={test_acc:.4f} macro_f1={test_f1:.4f}" + (f" roc_auc={roc_auc:.4f}" if roc_auc is not None else ""))

    report = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": test_cm.tolist(),
        "classes": classes,
        "best_epoch": ckpt.get("epoch"),
        "best_val_acc": ckpt.get("val_acc"),
    }
    save_json(report, os.path.join(cfg.out_dir, "test_report.json"))


if __name__ == "__main__":
    main()

