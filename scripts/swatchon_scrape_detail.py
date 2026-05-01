import argparse
import json
import os
import re
import sys
import time
import random
import urllib.request
import urllib.error
from typing import Dict, Optional, List, Any

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

FILTERED_URL = (
    "https://swatchon.com/wholesale-fabric?categoryIds=167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170&sort=&from=/wholesale-fabric"
)

# --- Anti-bot/stealth helpers ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.112 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.201 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.122 Safari/537.36",
]
LOCALES = ["en-US", "en-GB"]
TIMEZONES = ["UTC", "America/New_York", "Europe/Berlin", "Asia/Seoul", "Asia/Shanghai"]
BROWSER_ENGINE = os.getenv("SWATCHON_BROWSER", "chromium").strip().lower()

_STEALTH_JS = """
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


def _random_context_options() -> Dict[str, Any]:
    return {
        "user_agent": random.choice(USER_AGENTS),
        "locale": random.choice(LOCALES),
        "viewport": {"width": random.randint(1280, 1920), "height": random.randint(800, 1200)},
        "timezone_id": random.choice(TIMEZONES),
        "device_scale_factor": random.choice([1.0, 1.25, 1.5, 2.0]),
        "is_mobile": False,
        "has_touch": False,
        "color_scheme": random.choice(["light", "dark"]),
    }


def _apply_stealth(context) -> None:
    try:
        context.add_init_script(_STEALTH_JS)
    except Exception:
        pass


def _browser_launcher(playwright):
    if BROWSER_ENGINE == "firefox":
        return playwright.firefox
    if BROWSER_ENGINE == "webkit":
        return playwright.webkit
    return playwright.chromium


def _get_status(resp) -> Optional[int]:
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
            time.sleep(random.uniform(0.3, 1.2))
            resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                pass
            status = _get_status(resp)
            if status is None or status >= 400:
                time.sleep((2 ** attempt) * random.uniform(0.8, 1.6))
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
    m = re.search(r"background-image\s*:\s*url\((['\"]?)(.+?)\1\)", style, re.I)
    if m:
        return m.group(2)
    return None


def _upgrade_image_url(url: str) -> str:
    if not url:
        return url
    if "/images/small/" in url:
        return url.replace("/images/small/", "/images/original/")
    if "/images/medium/" in url:
        return url.replace("/images/medium/", "/images/original/")
    if "/images/large/" in url:
        return url.replace("/images/large/", "/images/original/")
    return url


def _extract_image_from_container(pic_cont) -> Optional[str]:
    image_src = None
    try:
        img = pic_cont.locator("img[src]").first
        if img.count() > 0 and img.is_visible():
            image_src = img.get_attribute("src")
    except Exception:
        pass

    if not image_src:
        try:
            pimg = pic_cont.locator("picture img[src]").first
            if pimg.count() > 0 and pimg.is_visible():
                image_src = pimg.get_attribute("src")
        except Exception:
            pass

    if not image_src:
        try:
            srcset = pic_cont.locator("img[srcset]").first
            if srcset.count() > 0:
                s = srcset.get_attribute("srcset") or ""
                image_src = (s.split(",")[0].strip().split(" ")[0]) if s else None
        except Exception:
            pass

    if not image_src:
        try:
            image_src = pic_cont.get_attribute("src") or pic_cont.get_attribute("data-src")
        except Exception:
            pass

    if not image_src:
        try:
            style = pic_cont.get_attribute("style")
            image_src = _extract_bg_image(style)
        except Exception:
            pass

    return _upgrade_image_url(image_src) if image_src else None


def _extract_variant_hint(card) -> str:
    hint_selectors = [
        ".color-name",
        ".option-name",
        ".item-code",
        ".product-code",
        ".product-name",
        ".title",
    ]
    for sel in hint_selectors:
        try:
            node = card.locator(sel).first
            if node.count() > 0:
                txt = (node.inner_text() or "").strip()
                if txt:
                    return re.sub(r"\s+", " ", txt)
        except Exception:
            continue
    return ""


def _collect_images_from_dom(page, all_products: bool) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    try:
        qda_products = page.locator(".qda-products.quality-detail-accordion").first
        grid = qda_products.locator(".grid.m-t-24").first
        cards = grid.locator(".qda-product-card")
        total = cards.count()
        if total == 0:
            return images

        max_items = total if all_products else 1
        for i in range(min(total, max_items)):
            try:
                card = cards.nth(i)
                pic_cont = card.locator(".product-image-container").first
                image_src = _extract_image_from_container(pic_cont)
                if not image_src:
                    continue
                images.append(
                    {
                        "image_src": image_src,
                        "variant_index": i + 1,
                        "variant_hint": _extract_variant_hint(card),
                    }
                )
            except Exception:
                continue
    except Exception:
        pass
    return images


def _collect_images_from_api(capture_quality: Optional[Dict[str, Any]], all_products: bool) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    if not capture_quality:
        return images

    q = capture_quality

    products = q.get("products") if isinstance(q, dict) else None
    if isinstance(products, list) and products:
        max_items = len(products) if all_products else 1
        for i, p in enumerate(products[:max_items], start=1):
            if not isinstance(p, dict):
                continue
            imgd = p.get("image") or {}
            if isinstance(imgd, dict):
                src = imgd.get("original") or imgd.get("large") or imgd.get("medium") or imgd.get("small")
            else:
                src = None
            if not src:
                continue
            hint = ""
            for k in ["code", "name", "colorName", "color", "displayName"]:
                val = p.get(k)
                if isinstance(val, str) and val.strip():
                    hint = val.strip()
                    break
            images.append(
                {
                    "image_src": _upgrade_image_url(src),
                    "variant_index": i,
                    "variant_hint": hint,
                }
            )

    if not images:
        medias = q.get("medias") if isinstance(q, dict) else None
        if isinstance(medias, list):
            max_items = len(medias) if all_products else 1
            idx = 1
            for m in medias:
                if idx > max_items:
                    break
                if not isinstance(m, dict):
                    continue
                if m.get("classType") != "image":
                    continue
                src = m.get("original") or m.get("large") or m.get("medium") or m.get("small")
                if not src:
                    continue
                images.append(
                    {
                        "image_src": _upgrade_image_url(src),
                        "variant_index": idx,
                        "variant_hint": (m.get("name") or "").strip() if isinstance(m.get("name"), str) else "",
                    }
                )
                idx += 1

    return images


def _dedupe_images(images: List[Dict[str, Any]], all_products: bool) -> List[Dict[str, Any]]:
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(images, start=1):
        src = (item.get("image_src") or "").strip()
        if not src or src in seen:
            continue
        seen.add(src)
        uniq.append(
            {
                "image_src": src,
                "variant_index": len(uniq) + 1,
                "variant_hint": (item.get("variant_hint") or "").strip(),
            }
        )
        if not all_products and uniq:
            break
    return uniq


def _extract_specifications(page) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    try:
        spec_root = page.locator(".qda-fabric-specification.quality-detail-accordion").first
        spec_root.wait_for(state="visible", timeout=30000)

        dts = spec_root.locator("dt")
        dds = spec_root.locator("dd")
        if dts.count() and dds.count():
            for i in range(min(dts.count(), dds.count())):
                key = (dts.nth(i).inner_text() or "").strip()
                val = (dds.nth(i).inner_text() or "").strip()
                if key:
                    specs[key] = re.sub(r"\s+", " ", val)
        else:
            known_order = [
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
            full_text = spec_root.inner_text()
            pos = {}
            for k in known_order:
                i = full_text.find(k)
                if i >= 0:
                    pos[k] = i
            sorted_keys = sorted(pos.keys(), key=lambda k: pos[k])
            for idx, key in enumerate(sorted_keys):
                start = pos[key] + len(key)
                end = pos[sorted_keys[idx + 1]] if idx + 1 < len(sorted_keys) else len(full_text)
                val = full_text[start:end].strip()
                val = re.sub(r"\s+", " ", val)
                if val and val != "-":
                    specs[key] = val

        for key, val in list(specs.items()):
            v2 = re.sub(r"^\s*>\s*", "", val)
            v2 = re.sub(r"\s*-\s*$", "", v2)
            specs[key] = v2.strip()
    except Exception:
        pass
    return specs


def _extract_tags_and_clean_specs(specs: Dict[str, str]) -> List[str]:
    tags: List[str] = []

    def _hashtags(text: str) -> List[str]:
        return re.findall(r"(#[A-Za-z0-9_+-]+)", text or "")

    try:
        if "Tags" in specs:
            raw = specs.pop("Tags")
            tags.extend(_hashtags(raw))

        for k, v in list(specs.items()):
            if not isinstance(v, str):
                continue
            hs = _hashtags(v)
            if not hs:
                continue
            lowered = v.lower()
            if "tags" in lowered:
                cleaned = re.split(r"(?i)\bTags?\b", v)[0].strip().strip(",;")
                specs[k] = cleaned
                tags.extend(hs)
            elif k.lower() in ("country",):
                cleaned = re.split(r"\s#[^\s,;]+", v, maxsplit=1)[0].strip()
                specs[k] = cleaned
                tags.extend(hs)

        uniq = []
        seen = set()
        for t in tags:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq
    except Exception:
        return []


def _download_image_with_fallback(img_url: str, img_out: str, referer_url: str) -> bool:
    if not img_url:
        return False

    ua = random.choice(USER_AGENTS)
    headers = {
        "User-Agent": ua,
        "Referer": referer_url or "https://swatchon.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }

    urls_to_try = [img_url]
    if "/images/original/" in img_url:
        urls_to_try.append(img_url.replace("/images/original/", "/images/large/"))
    elif "/images/large/" in img_url:
        urls_to_try.append(img_url.replace("/images/large/", "/images/original/"))

    for candidate in urls_to_try:
        for attempt in range(3):
            try:
                req = urllib.request.Request(candidate, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                with open(img_out, "wb") as imgf:
                    imgf.write(data)
                return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                if attempt == 2:
                    print(f"[ERROR] Failed image download ({candidate}): {e}")
                else:
                    time.sleep((2 ** attempt) * random.uniform(0.6, 1.4))
            except Exception as e:
                if attempt == 2:
                    print(f"[ERROR] Failed image download ({candidate}): {e}")
                else:
                    time.sleep((2 ** attempt) * random.uniform(0.6, 1.4))
    return False


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def scrape_first_item_detail(
    preset_url: Optional[str] = None,
    out_json: Optional[str] = None,
    all_products: bool = False,
) -> Dict[str, Any]:
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if BROWSER_ENGINE == "chromium":
            launch_kwargs["args"] = [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ]
        proxy = os.getenv("SWATCHON_PROXY")
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}
        browser_type = _browser_launcher(p)
        browser = browser_type.launch(**launch_kwargs)

        ctx_opts = _random_context_options()
        context = browser.new_context(**ctx_opts)
        _apply_stealth(context)
        page = context.new_page()
        page.set_default_timeout(random.randint(50000, 70000))

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

        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)

        if preset_url:
            target_url = preset_url
            print(f"[info] Navigating to detail: {target_url}")
            safe_goto(page, target_url)
        else:
            safe_goto(page, FILTERED_URL)
            try:
                cookie_btn = page.locator("button:has-text('Accept')").first
                if cookie_btn.is_visible():
                    cookie_btn.click()
            except Exception:
                pass

            search_items = page.locator("div.search-items").first
            search_items.wait_for(state="visible", timeout=60000)
            page.wait_for_timeout(3000)

            try:
                page.screenshot(path=os.path.join(out_dir, "search_items.png"), full_page=True)
                with open(os.path.join(out_dir, "search_page.html"), "w", encoding="utf-8") as f:
                    f.write(page.content())
            except Exception:
                pass

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

            target_url = None
            if not href:
                try:
                    card = search_items.locator(
                        ".card, .card-item, article, .search-item, .product-item, [data-testid*='item']"
                    ).first
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

        try:
            with open(os.path.join(out_dir, "detail_page.html"), "w", encoding="utf-8") as f:
                f.write(page.content())
            page.screenshot(path=os.path.join(out_dir, "detail_page.png"), full_page=True)
        except Exception:
            pass

        dom_images = _collect_images_from_dom(page, all_products=all_products)
        api_images = _collect_images_from_api(capture.get("quality"), all_products=all_products)
        images = _dedupe_images(dom_images + api_images, all_products=all_products)

        specs = _extract_specifications(page)
        tags = _extract_tags_and_clean_specs(specs)

        result: Dict[str, Any] = {
            "detail_url": target_url,
            "image_src": images[0]["image_src"] if images else None,
            "images": images,
            "specifications": specs,
        }
        if tags:
            result["tags"] = tags

        try:
            if capture.get("quality"):
                with open(os.path.join(out_dir, "quality_payload.json"), "w", encoding="utf-8") as f:
                    json.dump(capture["quality"], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        out_path = out_json if out_json else os.path.join(out_dir, "first_plain_detail.json")

        # Backward-compatible single output (default): one JSON + one JPG
        if not all_products:
            _write_json(out_path, result)
            if result.get("image_src"):
                base = out_path[:-5] if out_path.lower().endswith(".json") else out_path
                _download_image_with_fallback(result["image_src"], base + ".jpg", target_url)
        else:
            # all-products mode: write aggregate JSON + per-variant JSON/JPG files
            _write_json(out_path, result)
            base = out_path[:-5] if out_path.lower().endswith(".json") else out_path
            for item in result.get("images", []):
                idx = item.get("variant_index")
                if not isinstance(idx, int) or idx <= 0:
                    continue
                suffix = f"__v{idx}"
                variant_json = base + suffix + ".json"
                variant_jpg = base + suffix + ".jpg"

                variant_payload = dict(result)
                variant_payload["image_src"] = item.get("image_src")
                variant_payload["images"] = [item]
                _write_json(variant_json, variant_payload)

                img_src = item.get("image_src")
                if isinstance(img_src, str) and img_src:
                    _download_image_with_fallback(img_src, variant_jpg, target_url)

        print(json.dumps(result, ensure_ascii=False, indent=2))

        context.close()
        browser.close()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape swatchon detail page")
    parser.add_argument("url", nargs="?", help="Detail page URL to scrape")
    parser.add_argument("--out", dest="out", help="Output JSON file path")
    parser.add_argument(
        "--all-products",
        action="store_true",
        help="Capture all product-card variants and write __vN JSON/JPG outputs",
    )
    args = parser.parse_args()

    ok = scrape_first_item_detail(preset_url=args.url, out_json=args.out, all_products=args.all_products)
    sys.exit(0 if ok else 1)
