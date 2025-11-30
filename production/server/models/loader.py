import os
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


def load_checkpoint(ckpt_path: str, device: torch.device, classes: List[str] = None) -> Tuple[nn.Module, List[str]]:
    """Load model checkpoint. If classes not provided, try to read from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Try to get classes from checkpoint or use provided classes
    if classes is None:
        classes = ckpt.get("classes")
        if not classes:
            raise RuntimeError("Checkpoint missing 'classes' and no classes provided")
    
    # Build model
    model = build_model(len(classes)).to(device)
    
    # Load state dict - handle both old format (direct 'model') and new format ('model_state')
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        raise RuntimeError("Checkpoint missing both 'model' and 'model_state' keys")
    
    model.eval()
    return model, classes

