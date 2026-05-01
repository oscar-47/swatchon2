from __future__ import annotations

import os
import sys
import json
import time
import argparse
import urllib.request
from typing import List, Set, Dict

# Knit categories mapping (name -> categoryIds/url)
CATEGORIES: Dict[str, Dict[str, str]] = {
    "Single": {
        "categoryIds": "199,248",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=199,248&sort=&from=/wholesale-fabric",
    },
    "Jacquard Knit": {
        "categoryIds": "208",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=208&sort=&from=/wholesale-fabric",
    },
    "Double": {
        "categoryIds": "209,200,251,204,207,210,214,250",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=209,200,251,204,207,210,214,250&sort=&from=/wholesale-fabric",
    },
    "Pile Knit": {
        "categoryIds": "201,203,202,211,220,212,213",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=201,203,202,211,220,212,213&sort=&from=/wholesale-fabric",
    },
    "Tricot": {
        "categoryIds": "219",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=219&sort=&from=/wholesale-fabric",
    },
    "Crepe Knit": {
        "categoryIds": "206",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=206&sort=&from=/wholesale-fabric",
    },
    "Pique": {
        "categoryIds": "205",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=205&sort=&from=/wholesale-fabric",
    },
    "Mesh": {
        "categoryIds": "249",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=249&sort=&from=/wholesale-fabric",
    },
    "Low Gauge Knit": {
        "categoryIds": "252",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=252&sort=&from=/wholesale-fabric",
    },
    "Lace Knit": {
        "categoryIds": "216,217,218",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=216,217,218&sort=&from=/wholesale-fabric",
    },
}

CONFIG_PATH = os.path.join("scripts", "config", "targets_phase1_fabricflow.json")

# SwatchOn API base for direct queries (bypasses Nuxt frontend pagination issues)
SWATCHON_API_BASE = "https://api.swatchon.com/api/mall/v1/search/qualities"
SWATCHON_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
PER_PAGE = 48


def parse_only_categories(only_arg: str | None) -> Set[str]:
    """Parse --only argument to a normalized category-name set."""
    if not only_arg:
        return set()
    return {name.strip().lower() for name in only_arg.split(",") if name.strip()}


def load_phase_targets() -> Dict[str, int]:
    """Load optional per-class targets from phase config."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        out: Dict[str, int] = {}
        for item in obj.get("class_plan", []):
            key = item.get("key")
            target = item.get("target")
            if isinstance(key, str) and isinstance(target, int):
                out[key] = target
        return out
    except Exception:
        return {}


def fetch_api_page(category_ids: str, page: int, retries: int = 3) -> dict:
    """Fetch one page of results from SwatchOn search API with retries."""
    url = (
        f"{SWATCHON_API_BASE}?sort=&page={page}&perPage={PER_PAGE}"
        f"&categoryIds={category_ids}&preferredCurrency=usd&shippingCountry=US"
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=SWATCHON_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
            else:
                raise


def scrape_category(category_name: str, category_config: dict, target_count: int = 150, max_pages: int = 50) -> dict:
    """Scrape product links via SwatchOn API (no browser needed)."""
    all_links: Set[str] = set()
    page_results = []
    category_ids = category_config["categoryIds"]
    current_page = 1

    while current_page <= max_pages:
        try:
            data = fetch_api_page(category_ids, current_page)
            total = data.get("total", 0)
            items = data.get("items", [])

            if not items:
                print(f"  [INFO] Page {current_page} returned 0 items, stopping")
                break

            before = len(all_links)
            for item in items:
                landing = item.get("landingUrl", "")
                if landing:
                    full_url = "https://swatchon.com" + landing
                    all_links.add(full_url)

            page_results.append({
                "page": current_page,
                "links_found": len(items),
                "new_unique_links": len(all_links) - before,
                "total_unique_links": len(all_links),
            })
            print(f"  [INFO] Page {current_page}: {len(items)} items, {len(all_links)}/{total} unique links")

            if len(all_links) >= total or current_page * PER_PAGE >= total:
                break
            if len(all_links) >= target_count:
                break

            current_page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  [ERROR] Page {current_page} exception: {e}")
            current_page += 1
            time.sleep(1)
            continue

    return {
        "category": category_name,
        "timestamp": time.time(),
        "target_count": target_count,
        "actual_count": len(all_links),
        "pages_scraped": len(page_results),
        "page_details": page_results,
        "all_links": sorted(list(all_links)),
    }


def main():
    parser = argparse.ArgumentParser(description="Scrape SwatchOn knit category links")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Run only selected categories, comma-separated (example: Tricot,Mesh)",
    )
    args = parser.parse_args()

    selected_only = parse_only_categories(args.only)

    print("Knit categories link scraper starting...")
    base_output_dir = os.path.join(os.getcwd(), "outputs", "knit_categories")
    os.makedirs(base_output_dir, exist_ok=True)

    # Adaptive target counts based on model performance
    # P0 (极差): 400-500, P1 (数据不足): 250-300, P2 (中等): 300, Good: 150 (保持不变)
    CATEGORY_TARGET_COUNTS = {
        "Jacquard Knit": 500,    # P0: 39.1% accuracy - CRITICAL
        "Crepe Knit": 300,       # P1: 66.7% but only 51 images
        "Tricot": 300,           # P1: 58.3% with only 80 images
        "Pique": 300,            # P1: 61.5% with only 83 images
        "Double": 350,           # P2: 65.2% - moderate improvement needed
        "Low Gauge Knit": 350,   # P2: 60.9% - moderate improvement needed
        "Pile Knit": 350,        # P2: 56.5% - moderate improvement needed
        "Lace Knit": 200,        # Good: 73.9% - keep current (slight buffer)
        "Mesh": 200,             # Good: 82.6% - keep current (slight buffer)
        "Single": 200,           # Excellent: 78.3% - keep current (slight buffer)
    }
    phase_targets = load_phase_targets()
    if "Tricot" in phase_targets:
        CATEGORY_TARGET_COUNTS["Tricot"] = phase_targets["Tricot"]

    run_items = [
        (name, cfg)
        for name, cfg in CATEGORIES.items()
        if not selected_only or name.lower() in selected_only
    ]
    if selected_only and not run_items:
        print(f"[ERROR] No matched categories for --only={args.only}")
        print(f"[INFO] Available: {', '.join(CATEGORIES.keys())}")
        sys.exit(2)

    overall = {"total_categories": len(run_items), "done": 0, "total_links": 0, "category_results": {}}
    for i, (category_name, category_config) in enumerate(run_items, 1):
        try:
            target = CATEGORY_TARGET_COUNTS.get(category_name, 200)
            print(f"\nProcessing {i}/{len(run_items)}: {category_name} (target: {target} links)")
            result = scrape_category(category_name, category_config, target_count=target)
            cat_dir = os.path.join(base_output_dir, category_name)
            os.makedirs(cat_dir, exist_ok=True)
            out_path = os.path.join(cat_dir, f"{category_name}_links_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            overall["done"] += 1
            overall["total_links"] += result["actual_count"]
            overall["category_results"][category_name] = {"links_count": result["actual_count"], "pages_scraped": result["pages_scraped"], "file_path": out_path}
            print(f"Saved -> {out_path}")
            if i < len(run_items):
                time.sleep(2)
        except Exception as e:
            print(f"Category {category_name} failed: {e}")
            continue
    report_file = os.path.join(base_output_dir, f"overall_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Report: {report_file}\nOutput dir: {base_output_dir}")


if __name__ == "__main__":
    main()
