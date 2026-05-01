#!/usr/bin/env python3
"""
Scrape Raschel knit/lace fabric images from Shopify-based fabric stores.

Sources:
  - BridalFabrics (186 products, ~1191 images)
  - Trimplace (288 products, ~856 images)
  - BuyFabrics (13 products)
  - FashionFabricsClub (13 products)

Usage:
  python scripts/scrape_raschel.py                # full run
  python scripts/scrape_raschel.py --target 200   # stop at 200
  python scripts/scrape_raschel.py --dry-run      # preview only
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

OUTPUT_DIR = os.path.join("FabricFlow_Dataset", "KNIT", "Raschel")

SOURCES = {
    "bridal_fabrics": {
        "url": "https://www.bridalfabrics.com/collections/raschel-lace/products.json",
        "name": "bridal_fabrics",
    },
    "trimplace": {
        "url": "https://trimplace.com/collections/raschel/products.json",
        "name": "trimplace",
    },
    "buyfabrics": {
        "url": "https://buyfabrics.com/collections/raschel-lace/products.json",
        "name": "buyfabrics",
    },
    "fashionfabricsclub": {
        "url": "https://fashionfabricsclub.com/collections/raschel-lace/products.json",
        "name": "fashionfabricsclub",
    },
}


def fetch_all_products(base_url: str) -> List[Dict]:
    """Fetch all products from Shopify JSON API with pagination."""
    all_products = []
    page = 1
    while True:
        url = f"{base_url}?page={page}&limit=250"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
        except Exception as e:
            print(f"    Error fetching page {page}: {e}")
            break
        products = data.get("products", [])
        if not products:
            break
        all_products.extend(products)
        page += 1
        time.sleep(0.5)
    return all_products


def dhash(image_data: bytes, hash_size: int = 8) -> str:
    from PIL import Image
    img = Image.open(io.BytesIO(image_data)).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            idx = row * (hash_size + 1) + col
            bits.append(1 if pixels[idx] < pixels[idx + 1] else 0)
    return "".join(str(b) for b in bits)


def hamming_distance(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def is_duplicate(new_hash: str, existing_hashes: set, threshold: int = 5) -> bool:
    for h in existing_hashes:
        if hamming_distance(new_hash, h) <= threshold:
            return True
    return False


def download_image(url: str) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Scrape Raschel fabric images")
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hashes: set = set()
    global_seq = 0
    total_skipped = 0
    total_failed = 0

    for src_key, src_cfg in SOURCES.items():
        if global_seq >= args.target:
            break

        src_dir = os.path.join(OUTPUT_DIR, src_cfg["name"])
        os.makedirs(src_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"  {src_key}: fetching products...")

        products = fetch_all_products(src_cfg["url"])
        # Collect all image URLs with metadata
        candidates = []
        for p in products:
            product_url = f"https://{src_cfg['url'].split('/')[2]}/products/{p.get('handle','')}"
            for img in p.get("images", []):
                src_url = img.get("src", "")
                if not src_url:
                    continue
                candidates.append({
                    "image_url": src_url,
                    "product_title": p.get("title", ""),
                    "product_url": product_url,
                    "product_type": p.get("product_type", ""),
                    "vendor": p.get("vendor", ""),
                })

        print(f"  {len(products)} products, {len(candidates)} images")

        if args.dry_run:
            continue

        src_seq = 0
        for item in candidates:
            if global_seq >= args.target:
                break

            img_data = download_image(item["image_url"])
            if not img_data:
                total_failed += 1
                continue

            # Size check
            from PIL import Image
            try:
                img = Image.open(io.BytesIO(img_data))
                if img.width < 224 or img.height < 224:
                    total_skipped += 1
                    continue
            except Exception:
                total_failed += 1
                continue

            # Dedup
            h = dhash(img_data)
            if is_duplicate(h, hashes):
                total_skipped += 1
                continue
            hashes.add(h)

            global_seq += 1
            src_seq += 1
            filename = f"raschel_base_{src_cfg['name']}_{global_seq:04d}.jpg"
            out_path = os.path.join(src_dir, filename)

            # Save image
            with open(out_path, "wb") as f:
                f.write(img_data)

            # Save metadata
            meta = {
                "source": src_cfg["name"],
                "source_url": item["product_url"],
                "image_url": item["image_url"],
                "product_title": item["product_title"],
                "product_type": item["product_type"],
                "vendor": item["vendor"],
            }
            with open(out_path.replace(".jpg", ".json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            if global_seq % 25 == 0:
                print(f"  Progress: {global_seq}/{args.target}")

            time.sleep(0.2)

        print(f"  {src_key}: downloaded {src_seq} images")

    print(f"\n{'='*50}")
    print(f"  Total: {global_seq}/{args.target}")
    print(f"  Skipped (dup/small): {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main() or 0)
