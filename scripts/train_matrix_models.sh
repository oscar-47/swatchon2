#!/usr/bin/env bash
set -euo pipefail

# Train matrix (send to teammate)
#
# Model cards:
# - mobilenet_v3_large (TorchVision): https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html
# - google/efficientnet-b3 (Hugging Face): https://huggingface.co/google/efficientnet-b3
# - facebook/convnext-tiny-224 (Hugging Face): https://huggingface.co/facebook/convnext-tiny-224
# - resnet50 (TorchVision baseline): https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
#
# TorchVision model cards (extra candidates):
# - efficientnet_b4: https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b4.html
# - efficientnet_v2_s: https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html
# - convnext_small: https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_small.html
# - convnext_base: https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html
# - swin_s: https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_s.html
# - swin_b: https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_b.html
# - vit_b_16: https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html
# - vit_l_16: https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_l_16.html
# - maxvit_t: https://pytorch.org/vision/stable/models/generated/torchvision.models.maxvit_t.html
# - regnet_y_8gf: https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_8gf.html
# - densenet161: https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet161.html
# - resnext101_32x8d: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnext101_32x8d.html
#
# TorchVision notes (weights='IMAGENET1K_V1' in training code):
# - efficientnet_b4: bigger than B3, higher accuracy (but slower).
# - efficientnet_v2_s: newer EfficientNetV2, usually faster to train.
# - convnext_small/base: stronger than convnext_tiny.
# - swin_s/b, vit_b_16/vit_l_16, maxvit_t: transformer family candidates (heavier).
# - regnet_y_8gf, densenet161, resnext101_32x8d: classic strong CNN baselines.
#
# Paper / GitHub-only (NOT runnable via tools/train.py until you integrate adapters):
# - Deep TEN (PyTorch-Encoding): https://github.com/zhanghang1989/PyTorch-Encoding
# - EfficientNet-Lion (paper-only, no official HF model card)
# - SE-SSDNet (paper-only, no official HF model card)
#
# Assumed trainer CLI (per teammate message):
#   python tools/train.py --config configs/stage1_binary.yaml --model mobilenet_v3_large
#
# Copy/paste (2-GPU DDP, all models listed below):
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model mobilenet_v3_large
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model resnet50
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model google/efficientnet-b3
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model facebook/convnext-tiny-224
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model efficientnet_b4
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model efficientnet_v2_s
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model convnext_small
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model convnext_base
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model swin_s
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model swin_b
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model vit_b_16
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model vit_l_16
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model maxvit_t
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model regnet_y_8gf
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model densenet161
#   torchrun --standalone --nproc_per_node 2 --master_port 29500 tools/train.py --config configs/stage1_binary.yaml --model resnext101_32x8d
#
# Usage:
#   bash scripts/train_matrix_models.sh             # prints commands (default)
#   DRY_RUN=0 bash scripts/train_matrix_models.sh   # actually runs
#   CONFIG=configs/stage1_binary.yaml bash scripts/train_matrix_models.sh
#   DDP=1 bash scripts/train_matrix_models.sh       # print torchrun (2 GPU by default)
#   DDP=1 NPROC_PER_NODE=2 DRY_RUN=0 bash scripts/train_matrix_models.sh

DRY_RUN="${DRY_RUN:-1}"
CONFIG="${CONFIG:-configs/stage1_binary.yaml}"
DDP="${DDP:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
INCLUDE_UNSUPPORTED="${INCLUDE_UNSUPPORTED:-0}"

MODELS=(
  mobilenet_v3_large
  resnet50
  google/efficientnet-b3
  facebook/convnext-tiny-224
  efficientnet_b4
  efficientnet_v2_s
  convnext_small
  convnext_base
  swin_s
  swin_b
  vit_b_16
  vit_l_16
  maxvit_t
  regnet_y_8gf
  densenet161
  resnext101_32x8d
)

for model in "${MODELS[@]}"; do
  if [[ "$DDP" == "1" ]]; then
    cmd=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" tools/train.py --config "$CONFIG" --model "$model")
  else
    cmd=(python tools/train.py --config "$CONFIG" --model "$model")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    for a in "${cmd[@]}"; do
      printf ' %q' "$a"
    done
    printf '\n'
  else
    "${cmd[@]}"
  fi
done

if [[ "$INCLUDE_UNSUPPORTED" == "1" ]]; then
  cat <<'TXT'

# ---- Unsupported / paper-only models (placeholders) ----
# These are NOT available as Hugging Face model cards and are not wired into `tools/train.py`.
# To run them, you must:
#   1) decide the exact repo/implementation to use,
#   2) vendor/clone it,
#   3) write an adapter so `tools/train.py --model <name>` can build the model,
#   4) ensure training heads/datasets match your stage configs (stage1_binary, stage2_multi, etc).
#
# Deep TEN (PyTorch-Encoding): https://github.com/zhanghang1989/PyTorch-Encoding
# EfficientNet-Lion: paper-specific implementation (no official HF)
# SE-SSDNet: paper-specific implementation (no official HF)
#
# Example placeholder commands (will fail until integrated):
#   python tools/train.py --config configs/stage1_binary.yaml --model deep_ten
#   python tools/train.py --config configs/stage1_binary.yaml --model efficientnet_lion
#   python tools/train.py --config configs/stage1_binary.yaml --model se_ssdnet

TXT
fi
