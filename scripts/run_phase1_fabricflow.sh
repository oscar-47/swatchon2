#!/usr/bin/env bash
set -euo pipefail

# Phase-1 execution for FabricFlow classes:
# - KNIT/Tricot -> 300
# - WOVEN/Ribbed_Poplin (from Poplin) -> 300
# - WOVEN/Leno_Gauze (from Gauze) -> 200

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/swatchon_scrape_knit_link.py --only Tricot
python3 scripts/swatchon_scrape_woven_link.py --only Poplin,Gauze

python3 scripts/scrape_knit_category_details.py \
  --only Tricot \
  --per-category-limit Tricot=300 \
  --all-products

python3 scripts/scrape_woven_category_details.py \
  --only Poplin,Gauze \
  --per-category-limit Poplin=300,Gauze=200 \
  --all-products

python3 scripts/build_phase1_dataset.py \
  --config scripts/config/targets_phase1_fabricflow.json \
  --dataset-root FabricFlow_Dataset
