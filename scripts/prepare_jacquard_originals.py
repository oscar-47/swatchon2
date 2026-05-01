#!/usr/bin/env python3
"""
Prepare original (uncropped) Woven+Jacquard images for Google Drive upload.

Reads jacquard_relabel_mapping.csv → for each valid entry:
  1. Download full-res original from SwatchOn API (variant 1)
  2. Fall back to local thumbnail if API fails
  3. Save with new filename: woven+jacquard_jacquard_swatchon_NNNN.jpg

Local files are 360px thumbnails ("small"), API gives 1932px originals.
Always prefer API for best quality.

Usage:
  python scripts/prepare_jacquard_originals.py                # full run
  python scripts/prepare_jacquard_originals.py --dry-run      # preview only
  python scripts/prepare_jacquard_originals.py --include-bad  # include quality_ok=N entries too
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.6422.112 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 Chrome/124.0.6367.201 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123.0.6312.122 Safari/537.36",
]

# Paths
CSV_PATH = "jacquard_relabel_mapping.csv"
LOCAL_ORIGINALS = os.path.join("outputs", "woven_category_details", "Jacquard Weave")
OUTPUT_DIR = os.path.join("FabricFlow_Dataset", "WOVEN", "Woven+Jacquard", "swatchon")


def parse_ql_id(swatchon_id: str) -> tuple[Optional[int], Optional[int]]:
    """Parse QL-004896_1 → (4896, 1)"""
    m = re.match(r"QL-0*(\d+)_(\d+)", swatchon_id)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def fetch_quality(quality_id: int, max_retries: int = 3) -> Optional[Dict[str, Any]]:
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


def get_variant_image_url(quality_data: Dict[str, Any], variant_index: int) -> Optional[str]:
    """Get image URL for variant N (1-based), skipping products without images."""
    products = quality_data.get("products", [])
    if not isinstance(products, list):
        return None
    with_images = []
    for p in products:
        if not isinstance(p, dict):
            continue
        img = p.get("image")
        if not isinstance(img, dict) or not img:
            continue
        src = img.get("original") or img.get("large") or img.get("medium") or img.get("small")
        if src:
            with_images.append(src)
    if len(with_images) >= variant_index:
        return with_images[variant_index - 1]
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


def main():
    parser = argparse.ArgumentParser(description="Prepare Woven+Jacquard originals")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--include-bad", action="store_true", help="Include quality_ok=N entries")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep between API calls")
    args = parser.parse_args()

    # Read CSV
    entries: List[Dict[str, str]] = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["original_folder"] == "Woven_Jacquard":
                entries.append(row)

    print(f"Total Woven_Jacquard entries in CSV: {len(entries)}")

    # Filter
    if not args.include_bad:
        before = len(entries)
        entries = [e for e in entries if e.get("quality_ok", "").strip() != "N"]
        print(f"After excluding quality_ok=N: {len(entries)} (removed {before - len(entries)})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats = {"api_download": 0, "local_fallback": 0, "already_exists": 0, "failed": 0}

    for i, entry in enumerate(entries, 1):
        swatchon_id = entry["swatchon_id"]
        new_filename = entry["new_filename"]
        qid, variant = parse_ql_id(swatchon_id)

        if qid is None:
            print(f"  [{i}/{len(entries)}] SKIP - can't parse {swatchon_id}")
            stats["failed"] += 1
            continue

        out_path = os.path.join(OUTPUT_DIR, new_filename)

        if os.path.exists(out_path):
            stats["already_exists"] += 1
            continue

        # Always try API first for full-res originals (1932px vs 360px local)
        quality = fetch_quality(qid)
        if quality:
            img_url = get_variant_image_url(quality, variant or 1)
            if img_url:
                if not args.dry_run:
                    ok = download_image(img_url, out_path)
                    if ok:
                        stats["api_download"] += 1
                    else:
                        print(f"  [{i}/{len(entries)}] FAIL - download error for qid={qid}")
                        stats["failed"] += 1
                else:
                    stats["api_download"] += 1

                total_done = stats["api_download"] + stats["local_fallback"] + stats["already_exists"]
                if total_done % 20 == 0 and total_done > 0:
                    print(f"  [{i}/{len(entries)}] Progress: {total_done} done...")

                time.sleep(args.sleep)
                continue

        # Fallback: local thumbnail (360px)
        local_path = os.path.join(LOCAL_ORIGINALS, f"{qid}.jpg")
        if os.path.exists(local_path):
            if not args.dry_run:
                shutil.copy2(local_path, out_path)
            stats["local_fallback"] += 1
            print(f"  [{i}/{len(entries)}] WARN - used local 360px fallback for qid={qid}")
            continue

        print(f"  [{i}/{len(entries)}] FAIL - no source for qid={qid}")
        stats["failed"] += 1
        time.sleep(args.sleep)

    print(f"\n{'='*60}")
    print(f"  Results {'(DRY RUN)' if args.dry_run else ''}:")
    print(f"    Downloaded (full-res): {stats['api_download']}")
    print(f"    Local fallback (360px): {stats['local_fallback']}")
    print(f"    Already existed:       {stats['already_exists']}")
    print(f"    Failed:                {stats['failed']}")
    print(f"    Total output:          {stats['api_download'] + stats['local_fallback'] + stats['already_exists']}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main() or 0)
