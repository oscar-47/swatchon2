#!/usr/bin/env python3
"""
Scrape Fabric Wholesale Direct - Interlock collection
Downloads ALL non-group images, auto-detects and skips swirl/vortex shots,
then crops fabric-only patches from the remaining usable images.

Products (7 total):
  1. polyester-interlock-knit-lining-fabric
  2a. neoprene-scuba-fabric (1.5mm)
  2b. neoprene-scuba-3-mm-fabric (3mm)
  3. metallic-stretch-foil-lame-on-interlock-knit-fabric
  4. shiny-metallic-moonlight-on-interlock-fabrics
  5. closed-cell-neoprene-bonded-sponge-waterproof-fabric
  6. compression-wicking-performance-interlock-fabric

Swirl detection: uses radial gradient balance to detect images where
fabric is twisted into a vortex/rose shape (not useful for texture training).
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/fwd_interlock")
RAW_DIR = OUTPUT_DIR / "raw_images"
CROP_DIR = OUTPUT_DIR / "cropped_patches"
SWIRL_DIR = OUTPUT_DIR / "rejected_swirl"  # save rejected for review

CROP_SIZE = 512
MIN_FABRIC_RATIO = 0.85
WHITE_THRESHOLD = 240
MAX_CROPS_PER_IMAGE = 6

PRODUCTS = [
    "polyester-interlock-knit-lining-fabric",
    "neoprene-scuba-fabric",
    "neoprene-scuba-3-mm-fabric",
    "metallic-stretch-foil-lame-on-interlock-knit-fabric",
    "shiny-metallic-moonlight-on-interlock-fabrics",
    "closed-cell-neoprene-bonded-sponge-waterproof-fabric",
    "compression-wicking-performance-interlock-fabric",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


# ── Swirl detection ────────────────────────────────────────────────────────
def is_swirl(img_pil: Image.Image) -> tuple[bool, dict]:
    """
    Detect swirl/vortex fabric images using radial gradient balance.

    Swirl images have fabric folds radiating from a center spiral,
    creating evenly distributed gradients across all angular sectors.
    Non-swirl images (flat hang, close-up texture) have directional bias.

    Returns (is_swirl, metrics_dict)
    """
    gray = np.array(img_pil.convert("L")).astype(np.float64)
    h, w = gray.shape

    # --- Radial gradient balance ---
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    mag = np.sqrt(gx**2 + gy**2)

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    angles = np.arctan2(yy - cy, xx - cx)

    # Divide into 8 radial sectors, compute gradient energy per sector
    sector_energies = []
    for i in range(8):
        lo = -np.pi + i * np.pi / 4
        hi = lo + np.pi / 4
        mask = (angles >= lo) & (angles < hi)
        sector_energies.append(np.mean(mag[mask]))

    se = np.array(sector_energies)
    radial_balance = float(np.min(se) / (np.max(se) + 1e-6))

    # --- Corner whiteness ---
    cs = min(h, w) // 8
    corners = [gray[:cs, :cs], gray[:cs, -cs:], gray[-cs:, :cs], gray[-cs:, -cs:]]
    white_corners = sum(1 for c in corners if np.mean(c) > 235)

    # --- Decision rule ---
    # Swirl: balanced radial gradients (folds radiate evenly from center)
    # Rule: radial_balance > 0.4 OR (> 0.3 with few white corners = fabric fills frame)
    detected = radial_balance > 0.4 or (radial_balance > 0.3 and white_corners <= 1)

    metrics = {
        "radial_balance": round(radial_balance, 3),
        "white_corners": white_corners,
    }
    return detected, metrics


# ── Shopify API ────────────────────────────────────────────────────────────
def fetch_product_images(handle: str) -> tuple[list[dict], str]:
    """Fetch all images for a product from Shopify JSON API."""
    url = f"https://fabricwholesaledirect.com/products/{handle}.json"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    product = data["product"]
    return product["images"], product["title"]


def filter_non_group(images: list[dict]) -> list[dict]:
    """Remove group/composite images (keep everything else for swirl detection)."""
    result = []
    for img in images:
        filename = img["src"].split("/")[-1].split("?")[0].lower()
        if "group" in filename:
            continue
        if filename.endswith(".png") and "swatch" in filename:
            continue  # skip swatch thumbnails
        result.append(img)
    return result


# ── Download ───────────────────────────────────────────────────────────────
def download_image(url: str, save_path: Path) -> bool:
    if save_path.exists():
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            save_path.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"    FAILED download: {e}")
        return False


# ── Smart crop ─────────────────────────────────────────────────────────────
def find_fabric_bbox(img_array: np.ndarray) -> tuple[int, int, int, int]:
    """Find bounding box of fabric region (non-white area)."""
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    fabric_mask = gray < WHITE_THRESHOLD

    rows = np.any(fabric_mask, axis=1)
    cols = np.any(fabric_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, img_array.shape[1], img_array.shape[0])

    ri = np.where(rows)[0]
    ci = np.where(cols)[0]

    pad = 10
    top = min(int(ri[0]) + pad, int(ri[-1]))
    bottom = max(int(ri[-1]) - pad, top)
    left = min(int(ci[0]) + pad, int(ci[-1]))
    right = max(int(ci[-1]) - pad, left)

    return (left, top, right, bottom)


def is_good_crop(crop_array: np.ndarray) -> bool:
    gray = np.mean(crop_array, axis=2) if len(crop_array.shape) == 3 else crop_array
    return float(np.mean(gray < WHITE_THRESHOLD)) >= MIN_FABRIC_RATIO


def extract_crops(img_path: Path, max_crops: int = MAX_CROPS_PER_IMAGE) -> list[np.ndarray]:
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)

    left, top, right, bottom = find_fabric_bbox(img_array)
    fabric_w = right - left
    fabric_h = bottom - top

    actual_crop = CROP_SIZE
    if fabric_w < CROP_SIZE or fabric_h < CROP_SIZE:
        actual_crop = min(CROP_SIZE, fabric_w, fabric_h)
        if actual_crop < 224:
            return []

    crops = []
    attempts = 0
    max_attempts = max_crops * 10

    while len(crops) < max_crops and attempts < max_attempts:
        attempts += 1
        x = np.random.randint(left, max(left + 1, right - actual_crop + 1))
        y = np.random.randint(top, max(top + 1, bottom - actual_crop + 1))
        crop = img_array[y : y + actual_crop, x : x + actual_crop]

        if crop.shape[0] != actual_crop or crop.shape[1] != actual_crop:
            continue
        if is_good_crop(crop):
            crops.append(crop)

    return crops


# ── Main pipeline ──────────────────────────────────────────────────────────
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    SWIRL_DIR.mkdir(parents=True, exist_ok=True)

    global_seq = 0
    stats = {
        "products": 0,
        "raw_downloaded": 0,
        "swirl_rejected": 0,
        "usable_images": 0,
        "crops_generated": 0,
    }

    for handle in PRODUCTS:
        print(f"\n{'='*60}")
        print(f"Product: {handle}")

        images, title = fetch_product_images(handle)
        non_group = filter_non_group(images)
        print(f"  Title: {title}")
        print(f"  Total images: {len(images)} → non-group: {len(non_group)}")

        # Download all non-group images
        product_dir = RAW_DIR / handle.replace("-", "_")[:40]
        product_dir.mkdir(exist_ok=True)

        downloaded_files = []
        for img_data in non_group:
            url = img_data["src"]
            filename = url.split("/")[-1].split("?")[0]
            save_path = product_dir / filename
            if download_image(url, save_path):
                downloaded_files.append(save_path)
            time.sleep(0.2)

        print(f"  Downloaded: {len(downloaded_files)}")
        stats["raw_downloaded"] += len(downloaded_files)

        # Swirl detection + crop
        product_swirl = 0
        product_usable = 0
        product_crops = 0

        for img_file in sorted(downloaded_files):
            img_pil = Image.open(img_file).convert("RGB")
            detected, metrics = is_swirl(img_pil)

            if detected:
                product_swirl += 1
                # Move to rejected folder for review
                swirl_dest = SWIRL_DIR / img_file.name
                if not swirl_dest.exists():
                    img_pil.save(swirl_dest)
                continue

            product_usable += 1

            # Crop patches
            crops = extract_crops(img_file)
            for crop_array in crops:
                global_seq += 1
                crop_img = Image.fromarray(crop_array)
                crop_name = f"interlock_base_fabric_wholesale_direct_{global_seq:04d}.jpg"
                crop_img.save(CROP_DIR / crop_name, quality=95)
                product_crops += 1

        print(f"  Swirl rejected: {product_swirl}")
        print(f"  Usable images: {product_usable}")
        print(f"  Crops: {product_crops}")

        stats["swirl_rejected"] += product_swirl
        stats["usable_images"] += product_usable
        stats["crops_generated"] += product_crops
        stats["products"] += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  Products: {stats['products']}")
    print(f"  Raw downloaded: {stats['raw_downloaded']}")
    print(f"  Swirl rejected: {stats['swirl_rejected']}")
    print(f"  Usable images: {stats['usable_images']}")
    print(f"  Cropped patches: {stats['crops_generated']}")
    print(f"  Output: {CROP_DIR}/")
    print(f"  Rejected (review): {SWIRL_DIR}/")


if __name__ == "__main__":
    np.random.seed(42)
    main()
