import os
import json
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# New taxonomy class labels (alphabetical order matching training folder structure)
CLASS_LABELS = {
    "stage1": ["KNIT", "WOVEN", "OTHERS"],
    "stage2_woven": [
        "Corduroy", "Leno_Gauze", "Plain_Weave",
        "Ribbed_Poplin", "Satin", "Twill", "Woven+Jacquard",
    ],
    "stage2_knit": [
        "French_Terry", "Interlock", "Jersey",
        "Knit+Jacquard", "Rib_Knit", "Tricot",
    ],
}


def build_convnext(variant: str, n_classes: int) -> nn.Module:
    """Build a ConvNeXt model with a custom classifier head."""
    if variant == "small":
        model = models.convnext_small(weights=None)
    elif variant == "tiny":
        model = models.convnext_tiny(weights=None)
    else:
        raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, n_classes)
    return model


def build_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_checkpoint(
    ckpt_path: str,
    device: torch.device,
    stage_key: Optional[str] = None,
) -> Tuple[nn.Module, List[str]]:
    """Load a ConvNeXt checkpoint.

    Args:
        ckpt_path: Path to .pth file.
        device: Target device.
        stage_key: One of 'stage1', 'stage2_woven', 'stage2_knit'.
            Used to determine class labels and ConvNeXt variant.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine classes
    if stage_key and stage_key in CLASS_LABELS:
        classes = CLASS_LABELS[stage_key]
    else:
        classes = ckpt.get("classes")
        if not classes:
            raise RuntimeError(
                f"Cannot determine classes for {ckpt_path}. "
                "Provide stage_key or ensure checkpoint contains 'classes'."
            )

    # Determine ConvNeXt variant from checkpoint config
    config = ckpt.get("config", {})
    model_name = config.get("model", {}).get("name", "convnext_tiny")
    if "small" in model_name:
        variant = "small"
    else:
        variant = "tiny"

    model = build_convnext(variant, len(classes)).to(device)

    # Load weights
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, classes
