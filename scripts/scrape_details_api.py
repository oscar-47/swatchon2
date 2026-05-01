"""
SwatchOn detail scraper using API directly (no Playwright).

Usage:
  # Round 1: first color variant per product (default)
  python scripts/scrape_details_api.py --category Tricot --type knit

  # Round 2: second color variant
  python scripts/scrape_details_api.py --category Tricot --type knit --round 2

  # Round 3: third color variant
  python scripts/scrape_details_api.py --category Tricot --type knit --round 3

  # Multiple categories
  python scripts/scrape_details_api.py --category Tricot,Poplin,Gauze --type knit,woven,woven

  # Limit number of products to process
  python scripts/scrape_details_api.py --category Tricot --type knit --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlsplit

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.112 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.201 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.122 Safari/537.36",
]

# Map category type to links root directory
LINKS_ROOTS = {
    "knit": os.path.join("outputs", "knit_categories"),
    "woven": os.path.join("outputs", "categories"),
}


def extract_quality_id(url: str) -> Optional[str]:
    path = urlsplit(url).path.rstrip("/")
    seg = path.split("/")[-1] if path else ""
    m = re.search(r"(\d+)$", seg)
    return m.group(1) if m else None


def load_links(links_root: str, category_name: str) -> List[str]:
    d = os.path.join(links_root, category_name)
    if not os.path.isdir(d):
        print(f"[warn] Directory not found: {d}")
        return []
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".json")]
    if not files:
        return []
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
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
    return uniq


def fetch_quality(quality_id: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    url = f"https://api.swatchon.com/api/mall/v1/qualities/{quality_id}"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
    }
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
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


def get_variant_with_image(quality_data: Dict[str, Any], variant_index: int) -> Optional[Dict[str, Any]]:
    """Get the Nth product variant that has an image (1-based, skips variants without images).
    Returns dict with 'image_url' and 'product' keys, or None."""
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
            with_images.append({"image_url": src, "product": p})

    if len(with_images) >= variant_index:
        return with_images[variant_index - 1]
    return None


def build_metadata(quality_data: Dict[str, Any], product: Dict[str, Any],
                   image_url: str, variant_index: int, detail_url: str) -> Dict[str, Any]:
    """Build a metadata JSON dict from quality + product API data."""
    return {
        "detail_url": detail_url,
        "quality_id": quality_data.get("id"),
        "quality_code": quality_data.get("code"),
        "title": quality_data.get("title"),
        "image_url": image_url,
        "variant_index": variant_index,
        "product_code": product.get("code"),
        "label_color_number": product.get("labelColorNumber"),
        "categories": [
            {"id": c.get("id"), "name": c.get("name")}
            for c in (quality_data.get("categories") or [])
            if isinstance(c, dict)
        ],
        "contents": [
            {"name": c.get("name"), "percentage": c.get("percentage")}
            for c in (quality_data.get("contents") or [])
            if isinstance(c, dict)
        ],
        "specifications": {
            "stretchability": quality_data.get("stretchability"),
            "metric": quality_data.get("metric"),
        },
        "finishes": [f.get("name") for f in (quality_data.get("finishes") or []) if isinstance(f, dict)],
        "patterns": [p.get("name") for p in (quality_data.get("patterns") or []) if isinstance(p, dict)],
        "performances": [p.get("name") for p in (quality_data.get("performances") or []) if isinstance(p, dict)],
    }


def download_image(img_url: str, out_path: str, max_retries: int = 3) -> bool:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(img_url, headers=headers)
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


def process_category(
    category_name: str,
    cat_type: str,
    round_num: int,
    base_out: str,
    limit: Optional[int] = None,
    sleep_between: float = 0.3,
) -> Dict[str, int]:
    links_root = LINKS_ROOTS.get(cat_type)
    if not links_root:
        print(f"[error] Unknown type '{cat_type}', use 'knit' or 'woven'")
        return {"ok": 0, "skip": 0, "fail": 0, "no_variant": 0}

    links = load_links(links_root, category_name)
    if limit:
        links = links[:limit]
    total = len(links)
    print(f"\n===== {category_name} (Round {round_num}, {total} products) =====")

    out_dir = os.path.join(base_out, category_name)
    os.makedirs(out_dir, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "fail": 0, "no_variant": 0}

    for i, link in enumerate(links, 1):
        qid = extract_quality_id(link)
        if not qid:
            print(f"  [{i}/{total}] SKIP - no quality ID in {link}")
            stats["fail"] += 1
            continue

        # Output file naming: round 1 = {qid}.jpg, round 2 = {qid}_r2.jpg, etc.
        suffix = "" if round_num == 1 else f"_r{round_num}"
        out_jpg = os.path.join(out_dir, f"{qid}{suffix}.jpg")

        if os.path.exists(out_jpg):
            stats["skip"] += 1
            continue

        quality_data = fetch_quality(qid)
        if not quality_data:
            print(f"  [{i}/{total}] FAIL - API error for {qid}")
            stats["fail"] += 1
            time.sleep(sleep_between)
            continue

        variant = get_variant_with_image(quality_data, round_num)
        if not variant:
            stats["no_variant"] += 1
            continue

        img_url = variant["image_url"]
        ok = download_image(img_url, out_jpg)
        if ok:
            # Save metadata JSON alongside the image
            out_json = out_jpg.replace(".jpg", ".json")
            detail_url = "https://swatchon.com" + (quality_data.get("landingUrl") or f"/quality/{qid}")
            meta = build_metadata(quality_data, variant["product"], img_url, round_num, detail_url)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            stats["ok"] += 1
            if stats["ok"] % 20 == 0:
                print(f"  [{i}/{total}] Downloaded {stats['ok']} images so far...")
        else:
            print(f"  [{i}/{total}] FAIL download {qid}")
            stats["fail"] += 1

        time.sleep(sleep_between)

    print(f"  Done: {stats['ok']} new, {stats['skip']} skipped, {stats['no_variant']} no variant #{round_num}, {stats['fail']} failed")
    return stats


def main():
    parser = argparse.ArgumentParser(description="SwatchOn detail scraper via API (fast, no browser)")
    parser.add_argument("--category", required=True, help="Category name(s), comma-separated (e.g. Tricot,Poplin,Gauze)")
    parser.add_argument("--type", required=True, help="Category type(s), comma-separated: knit or woven (must match --category order)")
    parser.add_argument("--round", type=int, default=1, help="Which color variant to grab: 1=first, 2=second, etc. (default: 1)")
    parser.add_argument("--base-out", default=os.path.join("outputs", "category_images"), help="Base output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max products per category")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep between API calls (seconds)")
    args = parser.parse_args()

    categories = [c.strip() for c in args.category.split(",")]
    types = [t.strip() for t in args.type.split(",")]

    if len(types) == 1:
        types = types * len(categories)
    if len(types) != len(categories):
        print("[error] --type count must be 1 or match --category count")
        return 1

    total_stats = {"ok": 0, "skip": 0, "fail": 0, "no_variant": 0}
    for cat, ctype in zip(categories, types):
        stats = process_category(cat, ctype, args.round, args.base_out, args.limit, args.sleep)
        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"\n===== TOTAL =====")
    print(f"  New: {total_stats['ok']}, Skipped: {total_stats['skip']}, No variant: {total_stats['no_variant']}, Failed: {total_stats['fail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
