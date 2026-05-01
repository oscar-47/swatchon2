#!/usr/bin/env python3
"""
SwatchOn dataset collector — API-based, no browser needed.
Output directly to FabricFlow_Dataset/ with correct naming convention.

Naming: {class_lower}_base_swatchon_{seq:04d}.jpg
Output: FabricFlow_Dataset/{L1}/{Class}/swatchon/

Strategy:
  - Tricot (79 products, target 300): grab ALL color variants per product
  - Poplin (254 products, target 300): round-by-round until target
  - Gauze (203 products, target 200): round-by-round until target

Usage:
  python scripts/collect_swatchon_dataset.py --clean          # all 3, clean start
  python scripts/collect_swatchon_dataset.py --only Tricot    # one category
  python scripts/collect_swatchon_dataset.py --dry-run        # preview only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlsplit

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.6422.112 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 Chrome/124.0.6367.201 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123.0.6312.122 Safari/537.36",
]

DATASET_ROOT = "FabricFlow_Dataset"

# ── Category configs ──────────────────────────────────────────────
#  key         = SwatchOn category name (matches link folder)
#  l1          = top-level folder (KNIT / WOVEN)
#  class_name  = dataset class folder name
#  file_prefix = lowercase prefix for filenames
CATEGORIES = {
    "Tricot": {
        "links_root": os.path.join("outputs", "knit_categories"),
        "l1": "KNIT",
        "class_name": "Tricot",
        "file_prefix": "tricot",
        "target": 300,
        "strategy": "all_variants",
    },
    "Poplin": {
        "links_root": os.path.join("outputs", "categories"),
        "l1": "WOVEN",
        "class_name": "Ribbed_Poplin",
        "file_prefix": "ribbed_poplin",
        "target": 300,
        "strategy": "round_by_round",
    },
    "Gauze": {
        "links_root": os.path.join("outputs", "categories"),
        "l1": "WOVEN",
        "class_name": "Leno_Gauze",
        "file_prefix": "leno_gauze",
        "target": 200,
        "strategy": "round_by_round",
    },
}


# ── Helpers ───────────────────────────────────────────────────────

def extract_quality_id(url: str) -> Optional[str]:
    path = urlsplit(url).path.rstrip("/")
    seg = path.split("/")[-1] if path else ""
    m = re.search(r"(\d+)$", seg)
    return m.group(1) if m else None


def load_links(links_root: str, category_name: str) -> List[str]:
    d = os.path.join(links_root, category_name)
    if not os.path.isdir(d):
        print(f"  [warn] Link directory not found: {d}")
        return []
    files = sorted(
        [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")],
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not files:
        return []
    with open(files[0], "r", encoding="utf-8") as f:
        obj = json.load(f)
    raw = []
    if isinstance(obj, dict):
        raw = obj.get("all_links") or obj.get("links") or []
    elif isinstance(obj, list):
        raw = obj
    seen: Set[str] = set()
    uniq: List[str] = []
    for u in raw:
        u = str(u)
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    print(f"  Loaded {len(uniq)} links from {os.path.basename(files[0])}")
    return uniq


def fetch_quality(quality_id: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    url = f"https://api.swatchon.com/api/mall/v1/qualities/{quality_id}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) * random.uniform(0.5, 1.5))
        except Exception:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) * random.uniform(0.5, 1.5))
    return None


def download_image(img_url: str, out_path: str, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(img_url, headers={
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "image/*,*/*;q=0.8",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) * random.uniform(0.5, 1.0))
        except Exception:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) * random.uniform(0.5, 1.0))
    return False


# ── Build JSON (old format) ───────────────────────────────────────

def _join_contents(contents: list) -> str:
    parts = []
    for c in (contents or []):
        if not isinstance(c, dict):
            continue
        name = c.get("name") or ""
        pct = c.get("percentage")
        if name and pct is not None:
            parts.append(f"{name} {pct}%")
        elif name:
            parts.append(name)
    return " / ".join(parts)


def _join_categories(categories: list) -> str:
    names = [c.get("name", "") for c in (categories or []) if isinstance(c, dict)]
    return " > ".join(n for n in names if n)


def _join_names(items: list) -> str:
    return ", ".join(
        i.get("name", "") for i in (items or []) if isinstance(i, dict) and i.get("name")
    )


def build_json(quality: Dict[str, Any], image_url: str, detail_url: str) -> Dict[str, Any]:
    metric = quality.get("metric") or {}
    specs: Dict[str, str] = {}

    fabric_type = _join_categories(quality.get("categories"))
    if fabric_type:
        specs["Fabric Type"] = fabric_type

    fiber = _join_contents(quality.get("contents"))
    if fiber:
        specs["Fiber Content"] = fiber

    patterns = _join_names(quality.get("patterns"))
    if patterns:
        specs["Pattern"] = patterns

    for api_key, label in [("weight", "Weight"), ("width", "Width"), ("thickness", "Thickness")]:
        val = metric.get(api_key)
        if val:
            unit = metric.get(f"{api_key}Unit") or ""
            specs[label] = f"{val} {unit}".strip()

    finishes = _join_names(quality.get("finishes"))
    if finishes:
        specs["Dye Method"] = finishes

    performances = _join_names(quality.get("performances"))
    if performances:
        specs["Characteristics"] = performances

    care_advices = _join_names(quality.get("careAdvices"))
    if care_advices:
        specs["Care Advice"] = care_advices

    care_instructions = _join_names(quality.get("careInstructions"))
    if care_instructions:
        specs["Care Instructions"] = care_instructions

    country = quality.get("shippingDepartureCountryCode") or ""
    if country:
        specs["Country"] = country

    return {
        "detail_url": detail_url,
        "image_src": image_url,
        "specifications": specs,
    }


# ── Products with images ─────────────────────────────────────────

def get_products_with_images(quality: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for p in (quality.get("products") or []):
        if not isinstance(p, dict):
            continue
        img = p.get("image")
        if not isinstance(img, dict) or not img:
            continue
        src = img.get("original") or img.get("large") or img.get("medium") or img.get("small")
        if src:
            result.append({"image_url": src, "product": p})
    return result


# ── Filename helper ───────────────────────────────────────────────

def make_filename(prefix: str, seq: int) -> str:
    """e.g. tricot_base_swatchon_0001"""
    return f"{prefix}_base_swatchon_{seq:04d}"


# ── Core collection logic ─────────────────────────────────────────

def _save_item(out_dir: str, prefix: str, seq: int,
               image_url: str, quality: Dict[str, Any], detail_url: str,
               dry_run: bool, sleep_sec: float) -> bool:
    """Download image + write JSON. Returns True on success."""
    basename = make_filename(prefix, seq)
    out_jpg = os.path.join(out_dir, basename + ".jpg")
    out_json = os.path.join(out_dir, basename + ".json")

    if dry_run:
        return True

    ok = download_image(image_url, out_jpg)
    if ok:
        meta = build_json(quality, image_url, detail_url)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        time.sleep(sleep_sec)
        return True
    return False


def collect_all_variants(
    links: List[str], out_dir: str, prefix: str, target: int,
    sleep_sec: float, dry_run: bool,
) -> int:
    """Tricot strategy: for each product, grab ALL color variants until target."""
    seq = _count_existing(out_dir)
    for i, link in enumerate(links, 1):
        if seq >= target:
            break
        qid = extract_quality_id(link)
        if not qid:
            continue

        quality = fetch_quality(qid)
        if not quality:
            continue

        detail_url = "https://swatchon.com" + (quality.get("landingUrl") or f"/quality/{qid}")
        variants = get_products_with_images(quality)

        for v in variants:
            if seq >= target:
                break
            seq += 1
            ok = _save_item(out_dir, prefix, seq, v["image_url"], quality, detail_url, dry_run, sleep_sec)
            if not ok:
                seq -= 1

        if seq % 20 == 0 and seq > 0:
            print(f"  [{i}/{len(links)}] {seq} images so far...")

    return seq


def collect_round_by_round(
    links: List[str], out_dir: str, prefix: str, target: int,
    sleep_sec: float, dry_run: bool, max_rounds: int = 20,
) -> int:
    """Poplin/Gauze strategy: R1 first color → R2 second color → ... until target."""
    seq = _count_existing(out_dir)
    if seq >= target:
        print(f"  Already have {seq} >= {target}, skipping.")
        return seq

    quality_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    for round_num in range(1, max_rounds + 1):
        if seq >= target:
            break
        round_new = 0
        round_no_variant = 0

        for link in links:
            if seq >= target:
                break
            qid = extract_quality_id(link)
            if not qid:
                continue

            if qid not in quality_cache:
                quality_cache[qid] = fetch_quality(qid)
                time.sleep(sleep_sec)
            quality = quality_cache[qid]
            if not quality:
                continue

            variants = get_products_with_images(quality)
            if round_num > len(variants):
                round_no_variant += 1
                continue

            v = variants[round_num - 1]
            detail_url = "https://swatchon.com" + (quality.get("landingUrl") or f"/quality/{qid}")

            seq += 1
            ok = _save_item(out_dir, prefix, seq, v["image_url"], quality, detail_url, dry_run, sleep_sec)
            if ok:
                round_new += 1
            else:
                seq -= 1

        print(f"  Round {round_num}: +{round_new} new, {round_no_variant} no variant, total={seq}")
        if round_new == 0:
            print(f"  No new images in round {round_num}, stopping.")
            break

    return seq


def _count_existing(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    return sum(1 for f in os.listdir(out_dir) if f.endswith(".jpg"))


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect SwatchOn images + JSON for FabricFlow dataset")
    parser.add_argument("--only", type=str, default="", help="Run only selected categories, comma-separated")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep between API calls")
    parser.add_argument("--clean", action="store_true", help="Delete existing swatchon/ folder before starting")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no downloads")
    args = parser.parse_args()

    selected = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()

    for cat_name, cfg in CATEGORIES.items():
        if selected and cat_name not in selected:
            continue

        out_dir = os.path.join(DATASET_ROOT, cfg["l1"], cfg["class_name"], "swatchon")
        target = cfg["target"]
        prefix = cfg["file_prefix"]

        print(f"\n{'='*60}")
        print(f"  {cfg['class_name']}  |  target: {target}  |  strategy: {cfg['strategy']}")
        print(f"  -> {out_dir}/")
        print(f"  -> {prefix}_base_swatchon_NNNN.jpg + .json")
        print(f"{'='*60}")

        if args.clean and os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            print(f"  Cleaned {out_dir}")

        os.makedirs(out_dir, exist_ok=True)

        existing = _count_existing(out_dir)
        if existing >= target:
            print(f"  Already have {existing} images >= target {target}. Use --clean to restart.")
            continue

        links = load_links(cfg["links_root"], cat_name)
        if not links:
            print(f"  No links found, skipping.")
            continue

        if cfg["strategy"] == "all_variants":
            count = collect_all_variants(links, out_dir, prefix, target, args.sleep, args.dry_run)
        else:
            count = collect_round_by_round(links, out_dir, prefix, target, args.sleep, args.dry_run)

        print(f"  DONE: {count} images {'(dry-run)' if args.dry_run else ''}")

    print(f"\n{'='*60}")
    print("  Final counts:")
    for cat_name, cfg in CATEGORIES.items():
        if selected and cat_name not in selected:
            continue
        out_dir = os.path.join(DATASET_ROOT, cfg["l1"], cfg["class_name"], "swatchon")
        c = _count_existing(out_dir)
        print(f"    {cfg['class_name']}: {c} images  ({out_dir})")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main() or 0)
