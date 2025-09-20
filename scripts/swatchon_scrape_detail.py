import argparse
import json
import os
import re
import sys
import time
import random
import urllib.request
from typing import Dict, Optional

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

FILTERED_URL = (
    "https://swatchon.com/wholesale-fabric?categoryIds=167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170&sort=&from=/wholesale-fabric"
)
# --- Anti-bot/stealth helpers ---
USER_AGENTS = [
    # A small rotation of realistic desktop Chrome agents
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.112 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.201 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.122 Safari/537.36",
]
LOCALES = ["en-US", "en-GB"]
TIMEZONES = ["UTC", "America/New_York", "Europe/Berlin", "Asia/Seoul", "Asia/Shanghai"]

_STEALTH_JS = """
// Basic stealth tweaks
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
window.chrome = window.chrome || { runtime: {} };
try {
  const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
  if (originalQuery) {
    window.navigator.permissions.query = (parameters) => (
      parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters)
    );
  }
} catch (e) {}
"""

def _random_context_options():
    ua = random.choice(USER_AGENTS)
    viewport = {"width": random.randint(1280, 1920), "height": random.randint(800, 1200)}
    locale = random.choice(LOCALES)
    tz = random.choice(TIMEZONES)
    return {
        "user_agent": ua,
        "locale": locale,
        "viewport": viewport,
        "timezone_id": tz,
        "device_scale_factor": random.choice([1.0, 1.25, 1.5, 2.0]),
        "is_mobile": False,
        "has_touch": False,
        "color_scheme": random.choice(["light", "dark"]),
    }

def _apply_stealth(context):
    try:
        context.add_init_script(_STEALTH_JS)
    except Exception:
        pass

def _get_status(resp):
    try:
        s = getattr(resp, "status", None)
        if isinstance(s, int):
            return s
        if hasattr(resp, "status"):
            return resp.status()
    except Exception:
        return None
    return None

def safe_goto(page, url: str, max_retries: int = 3):
    last_resp = None
    for attempt in range(max_retries):
        try:
            # Small human-like pause before navigation
            time.sleep(random.uniform(0.3, 1.2))
            resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                pass
            status = _get_status(resp)
            if status is None or status >= 400:
                backoff = (2 ** attempt) * random.uniform(0.8, 1.6)
                time.sleep(backoff)
                last_resp = resp
                continue
            return resp
        except PlaywrightTimeoutError:
            time.sleep((2 ** attempt) * random.uniform(0.8, 1.6))
            continue
        except Exception:
            time.sleep((2 ** attempt) * random.uniform(0.8, 1.6))
            continue
    return last_resp



def _extract_bg_image(style: Optional[str]) -> Optional[str]:
    if not style:
        return None
    # e.g., background-image: url("https://...jpg")
    m = re.search(r"background-image\s*:\s*url\((['\"]?)(.+?)\1\)", style, re.I)
    if m:
        return m.group(2)
    return None


def scrape_first_item_detail(preset_url: Optional[str] = None, out_json: Optional[str] = None) -> Dict:
    with sync_playwright() as p:
        launch_kwargs = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        }
        proxy = os.getenv("SWATCHON_PROXY")
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}
        browser = p.chromium.launch(**launch_kwargs)

        ctx_opts = _random_context_options()
        context = browser.new_context(**ctx_opts)
        _apply_stealth(context)
        page = context.new_page()
        page.set_default_timeout(random.randint(50000, 70000))

        # Listen to interesting responses for debugging and capture detail payload
        capture = {"quality": None}
        def _log_response(resp):
            try:
                url = resp.url or ""
                if any(k in url.lower() for k in ["qualities", "quality", "filters", "detail"]):
                    status = resp.status if isinstance(getattr(resp, "status", None), int) else (resp.status() if hasattr(resp, "status") else 0)
                    print(f"[resp] {status} {url}")
                    u = url.lower()
                    if "/api/mall/v1/qualities/" in u and all(x not in u for x in ["/similar", "/deadstock"]):
                        try:
                            capture["quality"] = resp.json()
                        except Exception:
                            pass
            except Exception:
                pass
        page.on("response", _log_response)

        # Prepare output directory
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)

        if preset_url:
            # Directly navigate to the provided detail URL
            target_url = preset_url
            print(f"[info] Navigating to detail: {target_url}")
            safe_goto(page, target_url)
        else:
            # 1) Open filtered result page
            safe_goto(page, FILTERED_URL)

            # Cookie banner
            try:
                cookie_btn = page.locator("button:has-text('Accept')").first
                if cookie_btn.is_visible():
                    cookie_btn.click()
            except Exception:
                pass

            # Ensure search results present
            search_items = page.locator("div.search-items").first
            search_items.wait_for(state="visible", timeout=60000)
            # extra time for DOM to populate
            page.wait_for_timeout(3000)

            # Save debug screenshot + HTML
            try:
                page.screenshot(path=os.path.join(out_dir, "search_items.png"), full_page=True)
                with open(os.path.join(out_dir, "search_page.html"), "w", encoding="utf-8") as f:
                    f.write(page.content())
            except Exception:
                pass

            # 2) Get first product link under search-items (robust fallbacks)
            href = None
            try:
                cand = search_items.locator("a[href^='/quality']")
                if cand.count() > 0:
                    href = cand.first.get_attribute("href")
            except Exception:
                pass
            if not href:
                try:
                    cand = search_items.locator("a[href*='quality']")
                    if cand.count() > 0:
                        href = cand.first.get_attribute("href")
                except Exception:
                    pass
            if not href:
                try:
                    cand = search_items.locator("a")
                    if cand.count() > 0:
                        href = cand.first.get_attribute("href")
                except Exception:
                    pass
            # If still no href, try clicking first card-like element to navigate
            target_url = None
            if not href:
                try:
                    card = search_items.locator(".card, .card-item, article, .search-item, .product-item, [data-testid*='item']").first
                    if card and card.count() > 0:
                        card.scroll_into_view_if_needed()
                        card.click()
                        page.wait_for_load_state("domcontentloaded")
                        page.wait_for_timeout(2000)
                        target_url = page.url
                except Exception:
                    pass
            if not target_url:
                if not href:
                    raise RuntimeError("Failed to find first product link under search-items")
                if href.startswith("/"):
                    target_url = "https://swatchon.com" + href
                elif href.startswith("http"):
                    target_url = href
                else:
                    target_url = "https://swatchon.com/" + href
                print(f"[info] Navigating to detail: {target_url}")
                safe_goto(page, target_url)

        # Save detail page HTML for debugging
        try:
            with open(os.path.join(out_dir, "detail_page.html"), "w", encoding="utf-8") as f:
                f.write(page.content())
            page.screenshot(path=os.path.join(out_dir, "detail_page.png"), full_page=True)
        except Exception:
            pass

        # 3) Extract first product image from qda-products > grid > qda-product-card > product-image-container
        image_src = None
        try:
            qda_products = page.locator(".qda-products.quality-detail-accordion").first
            # Narrow down to the product grid
            grid = qda_products.locator(".grid.m-t-24").first
            card = grid.locator(".qda-product-card").first
            pic_cont = card.locator(".product-image-container").first
            # Priority: <img src>
            img = pic_cont.locator("img[src]").first
            if img.count() > 0 and img.is_visible():
                image_src = img.get_attribute("src")
            if not image_src:
                # picture > img
                pimg = pic_cont.locator("picture img[src]").first
                if pimg.count() > 0 and pimg.is_visible():
                    image_src = pimg.get_attribute("src")
            if not image_src:
                # sometimes srcset only
                srcset = pic_cont.locator("img[srcset]").first
                if srcset.count() > 0:
                    s = srcset.get_attribute("srcset") or ""
                    image_src = (s.split(',')[0].strip().split(' ')[0]) if s else None
            if not image_src:
                # attributes on container
                image_src = pic_cont.get_attribute("src") or pic_cont.get_attribute("data-src") or image_src
            if not image_src:
                # fallback: style background-image on container
                style = pic_cont.get_attribute("style") if pic_cont and pic_cont.count() > 0 else None
                image_src = _extract_bg_image(style)
        except Exception:
            pass

        # Fallback to API quality payload
        if not image_src and capture.get("quality"):
            q = capture["quality"] or {}
            # 1) medias array (marketing images / videos)
            medias = q.get("medias") or []
            for m in medias:
                if isinstance(m, dict) and m.get("classType") == "image":
                    image_src = m.get("original") or m.get("large") or m.get("medium") or m.get("small")
                    if image_src:
                        break
            # 2) products[0].image
            if not image_src and isinstance(q.get("products"), list) and q["products"]:
                imgd = (q["products"][0] or {}).get("image") or {}
                if isinstance(imgd, dict):
                    image_src = imgd.get("original") or imgd.get("large") or imgd.get("medium") or imgd.get("small") or image_src
            # 3) legacy shapes
            if not image_src and isinstance(q.get("images"), list) and q["images"]:
                cand = q["images"][0]
                image_src = cand.get("original") or cand.get("large") or cand.get("url") or image_src
            if not image_src and isinstance(q.get("image"), dict):
                image_src = q["image"].get("original") or q["image"].get("large") or q["image"].get("url") or image_src

        # 4) Extract specifications under qda-fabric-specification quality-detail-accordion
        specs: Dict[str, str] = {}
        try:
            spec_root = page.locator(".qda-fabric-specification.quality-detail-accordion").first
            spec_root.wait_for(state="visible", timeout=30000)
            # A generic parse: headings and values may appear as two columns; gather label-value pairs
            # Collect visible rows; use common tags
            rows = spec_root.locator("css=*:text('Specifications')")
            # Instead of relying on headers, parse all dt/dd or labels/values blocks
            # dt/dd
            dts = spec_root.locator("dt")
            dds = spec_root.locator("dd")
            if dts.count() and dds.count():
                for i in range(min(dts.count(), dds.count())):
                    key = (dts.nth(i).inner_text() or "").strip()
                    val = (dds.nth(i).inner_text() or "").strip()
                    if key:
                        specs[key] = re.sub(r"\s+", " ", val)
            else:
                # fallback: two-column label/value layout
                labels = spec_root.locator(".label, .title, h6, h5, h4")
                values = spec_root.locator(".value, .text-body, p, span")
                # best-effort: iterate visible elements and try to stitch pairs by proximity
                label_texts = [l for l in [e.strip() for e in labels.all_inner_texts()] if l]
                value_texts = [v for v in [e.strip() for e in values.all_inner_texts()] if v]
                # If label list matches known keys, try to map sequentially
                KNOWN_ORDER = [
                    "Specifications",
                    "Fabric Type",
                    "Fiber Content",
                    "Pattern",
                    "Dimensions",
                    "Weight",
                    "Width",
                    "Thickness",
                    "Finish",
                    "Characteristics",
                    "Dye Method",
                    "Care Advice",
                    "Care Instructions",
                    "Country",
                ]
                # naive approach: scan full text in the spec root and split by known labels
                full_text = spec_root.inner_text()
                # Build segments by each known key
                pos = {}
                for k in KNOWN_ORDER:
                    i = full_text.find(k)
                    if i >= 0:
                        pos[k] = i
                sorted_keys = sorted(pos.keys(), key=lambda k: pos[k])
                for idx, k in enumerate(sorted_keys):
                    start = pos[k] + len(k)
                    end = pos[sorted_keys[idx + 1]] if idx + 1 < len(sorted_keys) else len(full_text)
                    val = full_text[start:end].strip()
                    val = re.sub(r"\s+", " ", val)
                    if val and val != "-":
                        specs[k] = val
        except Exception:
            pass

        # Normalize specification values
        try:
            for k, v in list(specs.items()):
                v2 = re.sub(r"^\s*>\s*", "", v)
                v2 = re.sub(r"\s*-\s*$", "", v2)
                specs[k] = v2.strip()
        except Exception:
            pass
        # Extract tags (separate field) and clean any merged 'Tags' text from specs
        tags = []
        try:
            def _uniq_keep_order(seq):
                seen = set()
                out = []
                for x in seq:
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
                return out

            def _hashtags(text: str):
                return re.findall(r"(#[A-Za-z0-9_+-]+)", text or "")

            # 1) Explicit 'Tags' field inside specs
            if "Tags" in specs:
                raw = specs.pop("Tags")
                tags.extend(_hashtags(raw))

            # 2) Values that merged 'Tags #...' into another field (e.g., Country)
            for k, v in list(specs.items()):
                if not isinstance(v, str):
                    continue
                hs = _hashtags(v)
                if not hs:
                    continue
                lowered = v.lower()
                if "tags" in lowered:
                    # Drop everything from 'Tags' onwards
                    cleaned = re.split(r"(?i)\bTags?\b", v)[0].strip().strip(",;")
                    specs[k] = cleaned
                    tags.extend(hs)
                elif k.lower() in ("country",):
                    # Often appears as 'Korea Tags #Kids' -> keep the left part before first hashtag
                    cleaned = re.split(r"\s#[^\s,;]+", v, maxsplit=1)[0].strip()
                    specs[k] = cleaned
                    tags.extend(hs)
            if tags:
                tags = _uniq_keep_order(tags)
        except Exception:
            pass


        result = {
            "detail_url": target_url,
            "image_src": image_src,
            "specifications": specs,
        }
        if tags:
            result["tags"] = tags

        # Save captured quality payload for debugging
        try:
            if capture.get("quality"):
                with open(os.path.join(out_dir, "quality_payload.json"), "w", encoding="utf-8") as f:
                    json.dump(capture["quality"], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Save JSON
        out_path = out_json if out_json else os.path.join(out_dir, "first_plain_detail.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Also save image as JPG next to the JSON (same base name), if available
        try:
            img_url = result.get("image_src")
            if img_url:
                base, ext = (out_path[:-5], ".json") if out_path.lower().endswith(".json") else (out_path, "")
                img_out = base + ".jpg"
                ua = random.choice(USER_AGENTS)
                headers = {
                    "User-Agent": ua,
                    "Referer": target_url or "https://swatchon.com/",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                }
                for attempt in range(3):
                    try:
                        req = urllib.request.Request(img_url, headers=headers)
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = resp.read()
                        with open(img_out, "wb") as imgf:
                            imgf.write(data)
                        break
                    except Exception:
                        time.sleep((2 ** attempt) * random.uniform(0.6, 1.4))
        except Exception:
            pass

        print(json.dumps(result, ensure_ascii=False, indent=2))

        context.close()
        browser.close()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape swatchon detail page")
    parser.add_argument("url", nargs="?", help="Detail page URL to scrape")
    parser.add_argument("--out", dest="out", help="Output JSON file path")
    args = parser.parse_args()

    ok = scrape_first_item_detail(preset_url=args.url, out_json=args.out)
    sys.exit(0 if ok else 1)

