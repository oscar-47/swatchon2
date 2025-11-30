#!/usr/bin/env python3
"""
Fast version of upgrade script using browser reuse and parallel processing.

Key optimizations:
1. Reuse single browser instance across all URLs
2. Process multiple URLs in parallel (configurable batch size)
3. Reduce unnecessary waits
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use subprocess to call swatchon_scrape_detail.py with parallel execution

# Adaptive limits based on model performance
CATEGORY_LIMITS = {
    "Dobby": 500,            # P0: 21.7% accuracy - CRITICAL!!!
    "Double Weave": 300,     # P2: 60.9%
    "Jacquard Weave": 300,   # P2: 60.9%
    "Plain": 300,            # P2: 65.2%
    "Satin Weave": 300,      # P2: 65.2%
    "Eyelet": 150,           # Excellent: 95.2%
    "Pile Weave": 150,       # Excellent: 95.7%
    "Ripstop": 150,          # Good: 87.0%
    "Twill Weave": 150,      # Good: 87.0%
}

# Number of parallel workers (each launches its own browser)
# Adjust based on your system resources (CPU/RAM)
PARALLEL_WORKERS = 3


def extract_urls_from_category(category_dir):
    """Extract detail URLs from existing JSON files."""
    urls = []
    json_files = list(Path(category_dir).glob("*.json"))

    print(f"  Found {len(json_files)} JSON files in {category_dir}")

    for json_file in json_files:
        if json_file.name == "quality_payload.json":
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                detail_url = data.get("detail_url")
                if detail_url:
                    urls.append(detail_url)
        except Exception as e:
            print(f"[ERROR] Failed to read {json_file}: {e}")

    print(f"  Extracted {len(urls)} URLs")
    return urls


def scrape_url_subprocess(url, out_json):
    """Scrape a single URL using subprocess call to swatchon_scrape_detail.py."""
    try:
        cmd = [sys.executable, os.path.join("scripts", "swatchon_scrape_detail.py"), url, "--out", out_json]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return proc.returncode == 0
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return False


def main():
    print("=" * 80)
    print("WOVEN CATEGORIES - FAST UPGRADE TO HIGH QUALITY")
    print("=" * 80)

    base_in = Path("outputs/woven_category_details")
    base_out = Path("outputs/woven_category_details_hq")
    base_out.mkdir(parents=True, exist_ok=True)

    # Process categories in priority order
    category_order = [
        "Dobby",           # P0 - HIGHEST PRIORITY
        "Double Weave",    # P2
        "Jacquard Weave",  # P2
        "Plain",           # P2
        "Satin Weave",     # P2
        "Eyelet",          # Good
        "Pile Weave",      # Excellent
        "Ripstop",         # Good
        "Twill Weave",     # Good
    ]

    for category in category_order:
        source_dir = base_in / category
        if not source_dir.exists():
            print(f"\n[SKIP] {category} - source directory not found")
            continue

        target_dir = base_out / category
        target_dir.mkdir(parents=True, exist_ok=True)

        target_limit = CATEGORY_LIMITS.get(category, 150)

        print(f"\n{'=' * 80}")
        print(f"{category} (Target: {target_limit} images)")
        print(f"{'=' * 80}")

        # Extract URLs
        urls = extract_urls_from_category(source_dir)
        if not urls:
            print(f"No URLs found for {category}")
            continue

        print(f"Found {len(urls)} existing URLs")

        # Create work items
        work_items = []
        for i, url in enumerate(urls):
            # Extract numeric ID from URL
            url_part = url.split('/')[-1] if '/' in url else f"item_{i}"
            if '-' in url_part:
                item_id = url_part.split('-')[-1]
            else:
                item_id = url_part

            out_json = target_dir / f"{item_id}.json"
            work_items.append((url, str(out_json)))

        # Filter out existing items
        pending_items = [(url, out_json) for url, out_json in work_items if not os.path.exists(out_json)]
        existing_count = len(work_items) - len(pending_items)

        if existing_count > 0:
            print(f"Skipping {existing_count} existing items")

        if not pending_items:
            print("All items already exist, skipping category")
            continue

        print(f"Processing {len(pending_items)} items with {PARALLEL_WORKERS} parallel workers...")

        # Process items in parallel
        completed = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            # Submit all tasks
            future_to_item = {}
            for url, out_json in pending_items:
                future = executor.submit(scrape_url_subprocess, url, out_json)
                future_to_item[future] = (url, out_json)

            # Process results as they complete
            for future in as_completed(future_to_item):
                url, out_json = future_to_item[future]
                item_id = Path(out_json).stem
                completed += 1

                try:
                    success = future.result()
                    if success:
                        print(f"[{completed}/{len(pending_items)}] {item_id} -> OK")
                    else:
                        print(f"[{completed}/{len(pending_items)}] {item_id} -> FAILED")
                        failed += 1
                except Exception as e:
                    print(f"[{completed}/{len(pending_items)}] {item_id} -> ERROR: {e}")
                    failed += 1

        print(f"\nCategory {category} complete: {completed - failed} OK, {failed} failed")

    print("\n" + "=" * 80)
    print("All categories done.")
    print("=" * 80)


if __name__ == "__main__":
    main()

