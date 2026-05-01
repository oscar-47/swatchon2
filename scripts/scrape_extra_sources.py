#!/usr/bin/env python3
"""
Extra sources scraper — supplements scrape_all_sources.py
Targets sites found via web search that specialize in our hard-to-find classes.

New sources:
  - made-in-china.com — Raschel lace, Intarsia (huge B2B catalog)
  - nochintz.com — Basket/Hopsack (Australian fabric retailer)
  - fabricuk.com — Hopsack weave
  - iwantfabric.com — Basket weave hopsack
  - knitfabric.com — Cable knit
  - aroragroupofcompanies.com — Intarsia specialist (India)
  - intarsiaknits.com — Intarsia products
  - moodfabrics.com — Various knits
  - bridalfabrics.com — Raschel lace
  - fabricwholesaledirect.com — Double knit, Raschel lace
  - expressknitinc.com — Knit fabrics
  - springair-textile.com — Intarsia knitwear
"""

import os, sys, json, time, re, logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# Import shared helpers from main scraper
sys.path.insert(0, str(Path(__file__).parent))
from scrape_all_sources import (
    CLASSES, BASE_DIR, LOG_DIR, HEADERS, MIN_SIZE, DELAY,
    download_image, get_next_seq, save_image_and_meta, ensure_dir,
    log_csv, session, log,
)

# ═══════════════════════════════════════════════════════════════════════
# GENERIC: Scrape all product images from a list of URLs
# ═══════════════════════════════════════════════════════════════════════
def scrape_generic_pages(class_name, source_name, urls, img_filter=None):
    """Generic scraper: fetch pages, extract images, download."""
    out_dir = ensure_dir(class_name, source_name)
    seq = get_next_seq(out_dir)
    count = 0
    seen = set()

    for page_url in urls:
        r = None
        try:
            time.sleep(DELAY)
            r = session.get(page_url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log.warning(f"  Failed to fetch {page_url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # Also follow product links on the page
        product_urls = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_href = urljoin(page_url, href)
            if any(x in href for x in ["/product/", "/products/", "/portfolio/", "/collections/"]):
                if full_href not in urls:  # don't revisit
                    product_urls.add(full_href)

        all_pages = [(page_url, soup)] + [(pu, None) for pu in list(product_urls)[:30]]

        for url, s in all_pages:
            if s is None:
                try:
                    time.sleep(DELAY)
                    pr = session.get(url, timeout=30)
                    pr.raise_for_status()
                    s = BeautifulSoup(pr.text, "html.parser")
                except:
                    continue

            for img_tag in s.find_all("img"):
                src = img_tag.get("src", "") or img_tag.get("data-src", "") or img_tag.get("data-lazy-src", "")
                if not src:
                    continue
                # Skip non-content images
                if any(x in src.lower() for x in [".svg", "logo", "icon", "placeholder", "spinner",
                                                    "banner", "arrow", "payment", "flag", "social",
                                                    "avatar", "gravatar", "emoji"]):
                    continue
                # Apply custom filter if provided
                if img_filter and not img_filter(src, img_tag):
                    continue

                # Get full-size
                full_src = re.sub(r"-\d+x\d+\.", ".", src)
                full_src = re.sub(r"_\d+x\d+\.", ".", full_src)
                img_url = urljoin(url, full_src)

                if img_url in seen:
                    continue
                seen.add(img_url)

                meta = {"page_url": url, "alt": img_tag.get("alt", "")}
                if save_image_and_meta(img_url, class_name, source_name, out_dir, seq, meta):
                    log_csv(class_name, source_name, seq, img_url, f"{seq:04d}", "ok")
                    seq += 1
                    count += 1

    log.info(f"  [{class_name}] → {count} images from {source_name}")
    return count


# ═══════════════════════════════════════════════════════════════════════
# Made-in-China.com — Raschel, Intarsia
# ═══════════════════════════════════════════════════════════════════════
def scrape_made_in_china():
    log.info("═══ SOURCE: made-in-china.com ═══")
    total = 0

    searches = {
        "Raschel": [
            "https://www.made-in-china.com/products-search/hot-china-products/Raschel_Lace_Fabric.html",
            "https://www.made-in-china.com/products-search/hot-china-products/Raschel_Fabrics.html",
        ],
        "Intarsia": [
            "https://www.made-in-china.com/products-search/hot-china-products/Intarsia_Knitting.html",
        ],
        "Basket_Hopsack": [
            "https://www.made-in-china.com/products-search/hot-china-products/Basket_Weave_Fabric.html",
        ],
        "Double_Jersey": [
            "https://www.made-in-china.com/products-search/hot-china-products/Double_Jersey_Fabric.html",
        ],
    }

    def mic_filter(src, tag):
        return "cdnimg" in src or "pic" in src

    for class_name, urls in searches.items():
        n = scrape_generic_pages(class_name, "made_in_china", urls, mic_filter)
        total += n

    log.info(f"═══ made-in-china total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# nochintz.com — Basket/Hopsack specialist (Australia)
# ═══════════════════════════════════════════════════════════════════════
def scrape_nochintz():
    log.info("═══ SOURCE: nochintz.com ═══")
    urls = [
        "https://www.nochintz.com/hopsack-aqua",
        "https://www.nochintz.com/hopsack-chambray",
        "https://www.nochintz.com/hopsack-hibiscus",
        "https://www.nochintz.com/hopsack-heavy-basket-weave-cotton-fabric-forest",
        "https://www.nochintz.com/hopsack-heavy-basket-weave-cotton-fabric-ginger",
        "https://www.nochintz.com/hopsack-charcoal-and-white",
    ]
    n = scrape_generic_pages("Basket_Hopsack", "nochintz", urls)
    log.info(f"═══ nochintz total: {n} ═══")
    return n


# ═══════════════════════════════════════════════════════════════════════
# fabricuk.com — Hopsack
# ═══════════════════════════════════════════════════════════════════════
def scrape_fabricuk():
    log.info("═══ SOURCE: fabricuk.com ═══")
    urls = [
        "https://www.fabricuk.com/fabrics/1234-hopsack-weave-d.html",
    ]
    n = scrape_generic_pages("Basket_Hopsack", "fabricuk", urls)
    log.info(f"═══ fabricuk total: {n} ═══")
    return n


# ═══════════════════════════════════════════════════════════════════════
# knitfabric.com — Cable knit
# ═══════════════════════════════════════════════════════════════════════
def scrape_knitfabric():
    log.info("═══ SOURCE: knitfabric.com ═══")
    urls = [
        "https://knitfabric.com/cable-knit-fabric/",
        "https://knitfabric.com/knit/",
    ]
    def kf_filter(src, tag):
        return "cdn" in src or "upload" in src or "product" in src.lower()
    n = scrape_generic_pages("Cable_Knit", "knitfabric", urls, kf_filter)
    log.info(f"═══ knitfabric total: {n} ═══")
    return n


# ═══════════════════════════════════════════════════════════════════════
# Intarsia specialists
# ═══════════════════════════════════════════════════════════════════════
def scrape_intarsia_sources():
    log.info("═══ SOURCE: intarsia specialists ═══")
    total = 0

    # aroragroupofcompanies.com
    n = scrape_generic_pages("Intarsia", "arora", [
        "https://www.aroragroupofcompanies.com/intarsia-knit-fabric-2163650.html",
    ])
    total += n

    # intarsiaknits.com
    n = scrape_generic_pages("Intarsia", "intarsiaknits", [
        "https://intarsiaknits.com/collections/all",
    ])
    total += n

    # springair-textile.com
    n = scrape_generic_pages("Intarsia", "springair", [
        "https://www.springair-textile.com/sweater-knitwear-manufacturing/intarsia-knitwear-manufacturers-f3753774.html",
    ])
    total += n

    log.info(f"═══ intarsia specialists total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# fabricwholesaledirect.com — Double Knit + Raschel Lace
# ═══════════════════════════════════════════════════════════════════════
def scrape_fwd_extra():
    log.info("═══ SOURCE: fabricwholesaledirect.com (extra) ═══")
    total = 0

    def shopify_filter(src, tag):
        return "cdn.shopify.com" in src

    # Double Knit
    n = scrape_generic_pages("Double_Jersey", "fwd", [
        "https://fabricwholesaledirect.com/collections/double-knit-fabric",
    ], shopify_filter)
    total += n

    # Raschel Lace
    n = scrape_generic_pages("Raschel", "fwd", [
        "https://fabricwholesaledirect.com/products/raschel-lace-fabric",
    ], shopify_filter)
    total += n

    log.info(f"═══ FWD extra total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# moodfabrics.com — Various knits
# ═══════════════════════════════════════════════════════════════════════
def scrape_mood():
    log.info("═══ SOURCE: moodfabrics.com ═══")
    total = 0

    searches = {
        "Cable_Knit": "https://www.moodfabrics.com/catalogsearch/result/?q=cable+knit",
        "Purl_Knit": "https://www.moodfabrics.com/catalogsearch/result/?q=purl+knit",
        "Double_Jersey": "https://www.moodfabrics.com/catalogsearch/result/?q=double+jersey",
        "Basket_Hopsack": "https://www.moodfabrics.com/catalogsearch/result/?q=basket+weave",
    }

    for class_name, url in searches.items():
        n = scrape_generic_pages(class_name, "mood", [url])
        total += n

    log.info(f"═══ mood total: {total} ═══")
    return total


# ═══════════════════════════════════════════════════════════════════════
# bridalfabrics.com — Raschel lace
# ═══════════════════════════════════════════════════════════════════════
def scrape_bridal():
    log.info("═══ SOURCE: bridalfabrics.com ═══")
    n = scrape_generic_pages("Raschel", "bridalfabrics", [
        "https://www.bridalfabrics.com/collections/raschel-lace",
        "https://www.bridalfabrics.com/collections/lace",
    ])
    log.info(f"═══ bridalfabrics total: {n} ═══")
    return n


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("Extra Sources Fabric Scraper")
    log.info("=" * 60)

    results = {}
    scrapers = [
        ("made_in_china", scrape_made_in_china),
        ("nochintz", scrape_nochintz),
        ("fabricuk", scrape_fabricuk),
        ("knitfabric", scrape_knitfabric),
        ("intarsia_specialists", scrape_intarsia_sources),
        ("fwd_extra", scrape_fwd_extra),
        ("mood", scrape_mood),
        ("bridal", scrape_bridal),
    ]

    for name, func in scrapers:
        try:
            count = func()
            results[name] = count
        except Exception as e:
            log.error(f"Scraper {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"ERROR: {e}"

    # Summary
    log.info("\n" + "=" * 60)
    log.info("EXTRA SOURCES SUMMARY")
    log.info("=" * 60)
    grand_total = 0
    for name, count in results.items():
        if isinstance(count, int):
            log.info(f"  {name}: {count}")
            grand_total += count
        else:
            log.info(f"  {name}: {count}")
    log.info(f"\n  GRAND TOTAL: {grand_total} new images")

    # Per-class
    log.info("\nPer-class totals:")
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


if __name__ == "__main__":
    main()
