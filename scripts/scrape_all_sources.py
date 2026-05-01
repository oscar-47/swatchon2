#!/usr/bin/env python3
"""
Phase 2 Multi-Source Fabric Image Scraper
Scrapes professional fabric images from multiple B2B sources.
Target: 6 classes that failed QC from stock photo sources.

Sources:
  1. apparel-x.com — JSON API (no browser needed)
  2. runtangtex.com — WordPress (simple HTTP)
  3. fabricwholesaledirect.com — Shopify (simple HTTP)
  4. bosforustextile.com — Static (simple HTTP)
  5. efabrichouse.com — WooCommerce (simple HTTP)
  6. tgtekstil.com — Raschel specialist
  7. alibaba.com — B2B marketplace
"""

import os, sys, json, time, csv, hashlib, re, logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "FabricFlow_Dataset"
LOG_DIR  = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

MIN_SIZE = 224  # min dimension in pixels
DELAY = 1.5     # seconds between requests

# Class definitions: class_name → (L1, level, pattern_token)
CLASSES = {
    "Double_Jersey":   ("KNIT",  "L2", "base"),
    "Basket_Hopsack":  ("WOVEN", "L2", "base"),
    "Cable_Knit":      ("KNIT",  "L2", "base"),
    "Purl_Knit":       ("KNIT",  "L2", "base"),
    "Intarsia":        ("KNIT",  "L3", "intarsia"),  # L3 uses technique name
    "Raschel":         ("KNIT",  "L2", "base"),
}

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"scrape_all_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Helpers ─────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)

def fetch(url, timeout=30):
    """Fetch URL with delay and error handling."""
    time.sleep(DELAY)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        log.warning(f"Fetch failed: {url} → {e}")
        return None

def download_image(url, dest_path, min_dim=MIN_SIZE):
    """Download image, validate size, save. Returns True on success."""
    try:
        r = session.get(url, timeout=30, stream=True)
        r.raise_for_status()
        data = r.content
        img = Image.open(BytesIO(data))
        w, h = img.size
        if min(w, h) < min_dim:
            log.debug(f"Too small {w}x{h}: {url}")
            return False
        # Convert to RGB JPEG
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(str(dest_path), "JPEG", quality=92)
        return True
    except Exception as e:
        log.warning(f"Download failed: {url} → {e}")
        return False

def get_next_seq(directory):
    """Get the next sequence number for files in a directory."""
    existing = list(directory.glob("*.jpg"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        m = re.search(r"_(\d{4})\.jpg$", f.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1

def save_image_and_meta(url, class_name, source_name, directory, seq, meta_extra=None):
    """Download image and save with correct naming + metadata JSON."""
    l1, level, pattern = CLASSES[class_name]
    fname = f"{class_name.lower()}_{pattern}_{source_name}_{seq:04d}"
    jpg_path = directory / f"{fname}.jpg"
    json_path = directory / f"{fname}.json"

    if jpg_path.exists():
        return False

    if not download_image(url, jpg_path):
        return False

    meta = {
        "source": source_name,
        "source_url": url,
        "class": class_name,
        "level": level,
        "pattern": pattern,
        "downloaded_at": datetime.now().isoformat(),
    }
    if meta_extra:
        meta.update(meta_extra)

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"✓ {fname}.jpg ({Image.open(jpg_path).size[0]}x{Image.open(jpg_path).size[1]})")
    return True

def ensure_dir(class_name, source_name):
    """Create and return the output directory."""
    l1 = CLASSES[class_name][0]
    d = BASE_DIR / l1 / class_name / source_name
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── CSV Log ─────────────────────────────────────────────────────────────
csv_log_path = LOG_DIR / f"scrape_phase2_{datetime.now():%Y%m%d}.csv"
csv_log_file = open(csv_log_path, "a", newline="")
csv_writer = csv.writer(csv_log_file)
if csv_log_path.stat().st_size == 0:
    csv_writer.writerow(["class", "source", "seq", "image_url", "filename", "status", "timestamp"])

def log_csv(class_name, source, seq, url, filename, status):
    csv_writer.writerow([class_name, source, seq, url, filename, status, datetime.now().isoformat()])
    csv_log_file.flush()

# ═══════════════════════════════════════════════════════════════════════
# SOURCE 1: apparel-x.com (JSON API)
# ═══════════════════════════════════════════════════════════════════════
APPAREL_X_API = "https://www.apparel-x.com/php/GetApparelXItemList.php"
APPAREL_X_DETAIL = "https://www.apparel-x.com/item.php?itemid={item_id}"

APPAREL_X_KEYWORDS = {
    "Double_Jersey":  ["double jersey", "double knit jersey", "double knit", "ponte", "double face"],
    "Basket_Hopsack": ["hopsack", "basket weave", "basket fabric", "panama", "butcher"],
    "Cable_Knit":     ["cable knit", "cable knitted", "cable fabric"],
    "Purl_Knit":      ["purl knit", "purl stitch", "reverse knit", "links-links"],
    "Intarsia":       ["intarsia", "intarsia knit", "color block knit"],
    "Raschel":        ["raschel", "raschel lace", "raschel knit", "raschel mesh",
                       "double raschel", "warp knit", "power net"],
}

def scrape_apparel_x():
    """Scrape apparel-x.com via their JSON API."""
    log.info("═══ SOURCE: apparel-x.com ═══")
    source = "apparel_x"
    total = 0

    for class_name, keywords in APPAREL_X_KEYWORDS.items():
        out_dir = ensure_dir(class_name, source)
        seq = get_next_seq(out_dir)
        seen_ids = set()
        class_count = 0

        for kw in keywords:
            params = {
                "loginid": "",
                "category": "FABRIC",
                "displaytype": "Item Level",
                "itemcolortype": "0",
                "itemcode": kw,
                "sortorder": "bestaccess",
                "onepagecount": "200",
                "onloaded": "true",
                "aspectlanguagecode": "en",
                "aspectcurrencycode": "USD",
                "datatype": "json2",
            }

            log.info(f"  [{class_name}] keyword='{kw}'")
            time.sleep(DELAY)

            try:
                r = session.get(APPAREL_X_API, params=params, timeout=30)
                data = r.json()
            except Exception as e:
                log.warning(f"  API error: {e}")
                continue

            for item in data:
                if not isinstance(item, dict) or not item.get("ItemId"):
                    continue

                item_id = item.get("ItemId", "")
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                company = item.get("Company", item.get("CompanyCode", ""))
                code = item.get("ItemCode", "")
                name = item.get("ECommerceItemName", item.get("ItemName", ""))

                # Image URL pattern: img/item/{Company}/{ItemId}.jpg (full size)
                detail_url = f"https://www.apparel-x.com/item.php?itemid={item_id}"
                img_url = f"https://www.apparel-x.com/img/item/{company}/{item_id}.jpg"

                meta = {
                    "item_id": item_id,
                    "item_code": code,
                    "item_name": name,
                    "company": company,
                    "keyword": kw,
                    "detail_url": detail_url,
                }

                if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                    log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                    seq += 1
                    class_count += 1
                else:
                    log_csv(class_name, source, seq, img_url, "", "failed")

        log.info(f"  [{class_name}] → {class_count} images from apparel-x")
        total += class_count

    log.info(f"═══ apparel-x total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 2: runtangtex.com (WordPress)
# ═══════════════════════════════════════════════════════════════════════
RUNTANG_PAGES = {
    "Double_Jersey": [
        "https://runtangtex.com/double-knit-fabric/",
    ],
    "Cable_Knit": [
        "https://runtangtex.com/cable-knit-fabric/",
    ],
    "Purl_Knit": [
        "https://runtangtex.com/purl-knit-fabric/",
    ],
    "Raschel": [
        "https://runtangtex.com/fabrics/",  # check for raschel in general listing
    ],
}

def scrape_runtangtex():
    """Scrape runtangtex.com product images."""
    log.info("═══ SOURCE: runtangtex.com ═══")
    source = "runtangtex"
    total = 0

    for class_name, urls in RUNTANG_PAGES.items():
        out_dir = ensure_dir(class_name, source)
        seq = get_next_seq(out_dir)
        class_count = 0

        for page_url in urls:
            r = fetch(page_url)
            if not r:
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            seen_urls = set()

            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "") or img_tag.get("data-src", "")
                if not src or "/wp-content/uploads/" not in src:
                    continue
                # Skip tiny icons, logos, SVGs
                if any(x in src.lower() for x in [".svg", "logo", "icon", "banner", "arrow"]):
                    continue
                # Get full-size (remove dimension suffix like -400x400)
                full_src = re.sub(r"-\d+x\d+\.", ".", src)
                if full_src in seen_urls:
                    continue
                seen_urls.add(full_src)

                img_url = urljoin(page_url, full_src)
                meta = {"page_url": page_url, "alt": img_tag.get("alt", "")}

                if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                    log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                    seq += 1
                    class_count += 1
                else:
                    log_csv(class_name, source, seq, img_url, "", "failed")

        log.info(f"  [{class_name}] → {class_count} images from runtangtex")
        total += class_count

    log.info(f"═══ runtangtex total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 3: fabricwholesaledirect.com (Shopify)
# ═══════════════════════════════════════════════════════════════════════
FWD_COLLECTIONS = {
    "Double_Jersey": [
        "https://fabricwholesaledirect.com/collections/double-knit-fabric",
    ],
    "Cable_Knit": [
        "https://fabricwholesaledirect.com/collections/sweater-knit-fabric",
    ],
}

def scrape_fwd():
    """Scrape Fabric Wholesale Direct (Shopify) product images."""
    log.info("═══ SOURCE: fabricwholesaledirect.com ═══")
    source = "fwd"
    total = 0

    for class_name, urls in FWD_COLLECTIONS.items():
        out_dir = ensure_dir(class_name, source)
        seq = get_next_seq(out_dir)
        class_count = 0

        for collection_url in urls:
            # Shopify collections have JSON endpoint
            page = 1
            while True:
                json_url = f"{collection_url}.json?page={page}&limit=250"
                r = fetch(json_url)
                if not r:
                    break

                try:
                    data = r.json()
                    products = data.get("products", [])
                except:
                    # Fallback: parse HTML
                    r = fetch(f"{collection_url}?page={page}")
                    if not r:
                        break
                    products = []
                    soup = BeautifulSoup(r.text, "html.parser")
                    for img in soup.select("img[src*='cdn.shopify.com']"):
                        src = img.get("src", "")
                        if src:
                            products.append({"images": [{"src": "https:" + src if src.startswith("//") else src}]})

                if not products:
                    break

                for product in products:
                    images = product.get("images", [])
                    for img_data in images[:2]:  # max 2 images per product
                        img_url = img_data.get("src", "")
                        if not img_url:
                            continue
                        # Get large size
                        img_url = re.sub(r"_\d+x\d+\.", ".", img_url)
                        img_url = re.sub(r"\?v=\d+", "", img_url)

                        meta = {
                            "product_title": product.get("title", ""),
                            "product_type": product.get("product_type", ""),
                            "vendor": product.get("vendor", ""),
                            "collection_url": collection_url,
                        }

                        if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                            log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                            seq += 1
                            class_count += 1

                page += 1
                if page > 20:  # safety limit
                    break

        log.info(f"  [{class_name}] → {class_count} images from FWD")
        total += class_count

    log.info(f"═══ FWD total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 4: bosforustextile.com (Intarsia specialist)
# ═══════════════════════════════════════════════════════════════════════
def scrape_bosforus():
    """Scrape Bosforus Textile — Intarsia specialist."""
    log.info("═══ SOURCE: bosforustextile.com ═══")
    source = "bosforus"
    total = 0

    # Intarsia page
    class_name = "Intarsia"
    out_dir = ensure_dir(class_name, source)
    seq = get_next_seq(out_dir)

    urls_to_try = [
        "https://bosforustextile.com/portfolio/intarsia-knit-fabric/",
        "https://bosforustextile.com/portfolio/",
    ]

    seen = set()
    for page_url in urls_to_try:
        r = fetch(page_url)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "html.parser")

        for img_tag in soup.find_all("img"):
            src = img_tag.get("src", "") or img_tag.get("data-src", "")
            if not src:
                continue
            if any(x in src.lower() for x in [".svg", "logo", "icon", "placeholder"]):
                continue
            full_src = re.sub(r"-\d+x\d+\.", ".", src)
            img_url = urljoin(page_url, full_src)
            if img_url in seen:
                continue
            seen.add(img_url)

            meta = {"page_url": page_url, "alt": img_tag.get("alt", "")}
            if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                seq += 1
                total += 1

    log.info(f"  [Intarsia] → {total} images from bosforus")
    log.info(f"═══ bosforus total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 5: efabrichouse.com (WooCommerce — Intarsia)
# ═══════════════════════════════════════════════════════════════════════
def scrape_efabrichouse():
    """Scrape efabrichouse.com — Intarsia knit fabric category."""
    log.info("═══ SOURCE: efabrichouse.com ═══")
    source = "efabrichouse"
    total = 0

    class_name = "Intarsia"
    out_dir = ensure_dir(class_name, source)
    seq = get_next_seq(out_dir)

    page = 1
    seen = set()
    while page <= 10:
        url = f"https://www.efabrichouse.com/product-category/fabric/knitted-fabric/intarsia-knit-fabric/page/{page}/" if page > 1 else "https://www.efabrichouse.com/product-category/fabric/knitted-fabric/intarsia-knit-fabric/"
        r = fetch(url)
        if not r or r.status_code == 404:
            break

        soup = BeautifulSoup(r.text, "html.parser")
        products = soup.select("li.product a img, .products img, .product-image img")
        if not products:
            break

        for img_tag in products:
            src = img_tag.get("src", "") or img_tag.get("data-src", "")
            if not src:
                continue
            full_src = re.sub(r"-\d+x\d+\.", ".", src)
            img_url = urljoin(url, full_src)
            if img_url in seen:
                continue
            seen.add(img_url)

            meta = {"page_url": url, "alt": img_tag.get("alt", "")}
            if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                seq += 1
                total += 1

        page += 1

    log.info(f"  [Intarsia] → {total} images from efabrichouse")
    log.info(f"═══ efabrichouse total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 6: tgtekstil.com (Raschel specialist)
# ═══════════════════════════════════════════════════════════════════════
def scrape_tgtekstil():
    """Scrape TG Tekstil — Raschel fabric specialist."""
    log.info("═══ SOURCE: tgtekstil.com ═══")
    source = "tgtekstil"
    total = 0

    class_name = "Raschel"
    out_dir = ensure_dir(class_name, source)
    seq = get_next_seq(out_dir)

    urls = [
        "https://tgtekstil.com/en/product_category/raschel-fabric/",
        "https://tgtekstil.com/en/products/",
    ]

    seen = set()
    for page_url in urls:
        r = fetch(page_url)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "html.parser")

        # Find product links first, then visit each
        product_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/product/" in href or "/en/product/" in href:
                product_links.add(urljoin(page_url, href))

        # Also get images directly from listing
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src", "") or img_tag.get("data-src", "")
            if not src or any(x in src.lower() for x in [".svg", "logo", "icon", "placeholder"]):
                continue
            full_src = re.sub(r"-\d+x\d+\.", ".", src)
            img_url = urljoin(page_url, full_src)
            if img_url in seen:
                continue
            seen.add(img_url)

            meta = {"page_url": page_url, "alt": img_tag.get("alt", "")}
            if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                seq += 1
                total += 1

        # Visit individual product pages
        for prod_url in list(product_links)[:50]:
            pr = fetch(prod_url)
            if not pr:
                continue
            psoup = BeautifulSoup(pr.text, "html.parser")
            for img_tag in psoup.find_all("img"):
                src = img_tag.get("src", "") or img_tag.get("data-src", "")
                if not src or any(x in src.lower() for x in [".svg", "logo", "icon"]):
                    continue
                if "upload" in src or "product" in src or "fabric" in src.lower():
                    full_src = re.sub(r"-\d+x\d+\.", ".", src)
                    img_url = urljoin(prod_url, full_src)
                    if img_url in seen:
                        continue
                    seen.add(img_url)

                    meta = {"page_url": prod_url, "alt": img_tag.get("alt", "")}
                    if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                        log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                        seq += 1
                        total += 1

    log.info(f"  [Raschel] → {total} images from tgtekstil")
    log.info(f"═══ tgtekstil total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 7: alibaba.com (B2B marketplace)
# ═══════════════════════════════════════════════════════════════════════
ALIBABA_SEARCHES = {
    "Double_Jersey":  ["double jersey knit fabric", "double jersey textile"],
    "Basket_Hopsack": ["hopsack fabric wholesale", "basket weave textile"],
    "Cable_Knit":     ["cable knit fabric wholesale"],
    "Purl_Knit":      ["purl knit fabric", "reverse jersey knit fabric"],
    "Intarsia":       ["intarsia knit fabric wholesale"],
    "Raschel":        ["raschel fabric wholesale", "raschel lace fabric"],
}

def scrape_alibaba():
    """Scrape Alibaba product listing images."""
    log.info("═══ SOURCE: alibaba.com ═══")
    source = "alibaba"
    total = 0

    for class_name, keywords in ALIBABA_SEARCHES.items():
        out_dir = ensure_dir(class_name, source)
        seq = get_next_seq(out_dir)
        class_count = 0
        seen = set()

        for kw in keywords:
            search_url = f"https://www.alibaba.com/trade/search?SearchText={kw.replace(' ', '+')}"
            r = fetch(search_url)
            if not r:
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "") or img_tag.get("data-src", "")
                if not src:
                    continue
                # Alibaba product images are on their CDN
                if "s.alicdn.com" not in src and "cbu01.alicdn.com" not in src:
                    continue
                # Get larger version
                img_url = re.sub(r"_\d+x\d+\.", ".", src)
                img_url = re.sub(r"\.jpg_\d+x\d+\.jpg", ".jpg", img_url)
                if img_url in seen:
                    continue
                seen.add(img_url)

                meta = {"keyword": kw, "search_url": search_url}
                if save_image_and_meta(img_url, class_name, source, out_dir, seq, meta):
                    log_csv(class_name, source, seq, img_url, f"{seq:04d}", "ok")
                    seq += 1
                    class_count += 1

        log.info(f"  [{class_name}] → {class_count} images from alibaba")
        total += class_count

    log.info(f"═══ alibaba total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("Phase 2 Multi-Source Fabric Scraper")
    log.info("=" * 60)

    results = {}

    # Run all scrapers
    scrapers = [
        ("apparel_x", scrape_apparel_x),
        ("runtangtex", scrape_runtangtex),
        ("fwd", scrape_fwd),
        ("bosforus", scrape_bosforus),
        ("efabrichouse", scrape_efabrichouse),
        ("tgtekstil", scrape_tgtekstil),
        ("alibaba", scrape_alibaba),
    ]

    for name, func in scrapers:
        try:
            count = func()
            results[name] = count
        except Exception as e:
            log.error(f"Scraper {name} failed: {e}")
            results[name] = f"ERROR: {e}"

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    grand_total = 0
    for name, count in results.items():
        if isinstance(count, int):
            log.info(f"  {name}: {count} images")
            grand_total += count
        else:
            log.info(f"  {name}: {count}")

    log.info(f"\n  GRAND TOTAL: {grand_total} new images")

    # Per-class summary
    log.info("\nPer-class breakdown:")
    for class_name in CLASSES:
        l1 = CLASSES[class_name][0]
        class_dir = BASE_DIR / l1 / class_name
        total_class = 0
        if class_dir.exists():
            for source_dir in class_dir.iterdir():
                if source_dir.is_dir():
                    n = len(list(source_dir.glob("*.jpg")))
                    if n > 0:
                        log.info(f"  {class_name}/{source_dir.name}: {n}")
                    total_class += n
        log.info(f"  → {class_name} TOTAL: {total_class}")

    csv_log_file.close()
    log.info(f"\nLog saved to: {csv_log_path}")


if __name__ == "__main__":
    # Allow running individual sources: python scrape_all_sources.py apparel_x
    if len(sys.argv) > 1:
        source_name = sys.argv[1]
        scraper_map = {
            "apparel_x": scrape_apparel_x,
            "runtangtex": scrape_runtangtex,
            "fwd": scrape_fwd,
            "bosforus": scrape_bosforus,
            "efabrichouse": scrape_efabrichouse,
            "tgtekstil": scrape_tgtekstil,
            "alibaba": scrape_alibaba,
        }
        if source_name in scraper_map:
            scraper_map[source_name]()
        else:
            print(f"Unknown source: {source_name}")
            print(f"Available: {', '.join(scraper_map.keys())}")
    else:
        main()
