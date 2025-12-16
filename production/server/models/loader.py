import os
import json
from typing import Tuple, List

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(n_classes: int) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, n_classes)
    return model


def build_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load model checkpoint. Classes are read from checkpoint or JSON file."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Try to get classes from checkpoint
    classes = ckpt.get("classes")

    # If not found, try to load from JSON file in same directory
    if not classes:
        ckpt_dir = os.path.dirname(ckpt_path)
        json_files = [f for f in os.listdir(ckpt_dir) if f.endswith('_results.json')]

        if json_files:
            json_path = os.path.join(ckpt_dir, json_files[0])
            with open(json_path, 'r') as f:
                results = json.load(f)
                classes = results.get('classes')

        # Fallback for stage1 (Knit vs Woven)
        if not classes:
            if 'stage1' in ckpt_path:
                classes = ['Knit', 'Woven']
            else:
                raise RuntimeError(f"Cannot determine classes for {ckpt_path}")

    # Build model with correct number of classes
    model = build_model(len(classes)).to(device)

    # Load state dict - handle different checkpoint formats
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, classes

