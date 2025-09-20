import json
import os
import sys
import time
from typing import List

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

URL = "https://swatchon.com/wholesale-fabric?sort=&from=/wholesale-fabric"


def extract_woven_subcategories() -> List[str]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()
        page.set_default_timeout(60000)
        # Capture interesting XHR responses (to discover filter APIs)
        api_capture = {}
        def _log_response(resp):
            try:
                url = resp.url or ""
                if any(k in url.lower() for k in ["filter", "filters", "category", "facet"]):
                    # status can be a method in sync API
                    status = resp.status if isinstance(getattr(resp, "status", None), int) else (resp.status() if hasattr(resp, "status") else 0)
                    try:
                        body = resp.text()[:1000]
                    except Exception:
                        body = "<non-text or failed to read>"
                    print(f"[resp] {status} {url}\n{body}\n---")
                    if "qualities/filter" in url.lower():
                        try:
                            api_capture["data"] = resp.json()
                        except Exception:
                            pass
            except Exception:
                pass
        page.on("response", _log_response)


        # Navigate
        page.goto(URL, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=60000)
        except PlaywrightTimeoutError:
            pass  # proceed anyway

        # Give extra time for client-side hydration/data
        try:
            page.wait_for_timeout(5000)
        except Exception:
            pass


        # Try to accept cookies if present
        try:
            cookie_btn = page.locator("button:has-text('Accept')").first
            if cookie_btn.is_visible():
                cookie_btn.click()
        except Exception:
            pass

        # Heuristic: Ensure filters are visible. The page may require a small delay for hydration.
        time.sleep(2)

        # Try expanding likely filter panels first
        for label in ["Fabric Type", "Fabric Types", "Category", "Fabric type", "Fabric types"]:
            try:
                el = page.locator(f"text={label}").first
                if el and el.count() > 0 and el.is_visible():
                    el.click()
                    time.sleep(0.3)
            except Exception:
                pass

        # Primary path: use API response to extract Woven subcategories
        texts: List[str] = []
        try:
            # Give a moment for XHR to complete and capture
            for _ in range(20):
                if "data" in api_capture:
                    break
                try:
                    page.wait_for_timeout(500)
                except Exception:
                    time.sleep(0.5)
            data = api_capture.get("data")
            if data:
                cat = (data or {}).get("category", {})
                fltrs = cat.get("filters", []) or []
                woven_id = next((f.get("id") for f in fltrs if (f.get("name", "").strip().lower() == "woven" and f.get("parentId") is None)), None)
                if woven_id:
                    texts = [ (f.get("name") or "").strip() for f in fltrs if f.get("parentId") == woven_id ]
                    texts = [t for t in texts if t]
                print(f"[debug] API filters: {len(fltrs)}, woven_id={woven_id}, subs={len(texts)}")
            else:
                print("[debug] No API data captured yet")
        except Exception as e:
            print(f"[debug] API parse failed: {e}")
            pass

        # Fallback path: interact with DOM to reveal and scrape
        if not texts:
            try:
                # Locate the Woven filter section (could be a button/label/div)
                woven_header = page.locator("text=Woven").first
                woven_header.wait_for(state="visible", timeout=60000)

                # Attempt to click Woven to reveal children (if collapsible)
                try:
                    if woven_header.is_enabled():
                        woven_header.click()
                        time.sleep(0.3)
                except Exception:
                    pass

                # Expand if it's a collapsible section by checking nearest interactive ancestor
                try:
                    parent = woven_header.locator(
                        "xpath=ancestor::*[self::button or @role='button' or contains(@class,'filter')][1]"
                    )
                    aria_expanded = parent.get_attribute("aria-expanded")
                    if aria_expanded is not None and aria_expanded.lower() == "false":
                        parent.click()
                        time.sleep(0.5)
                except Exception:
                    pass

                # Find the items container under Woven
                container = woven_header.locator(
                    "xpath=ancestor::*[contains(@class,'filter') or contains(@class,'Filter')][1]"
                )
                items_container = container.locator(".filter-items.depth-1.p-t-8").first
                if not items_container or items_container.count() == 0:
                    # Strategy 2: search globally
                    items_container = page.locator(".filter-items.depth-1.p-t-8").first

                if items_container and items_container.count() > 0:
                    # Extract labels from common element types
                    item_labels = items_container.locator("li, a, label, span, div")
                    raw_texts = [t.strip() for t in item_labels.all_inner_texts()]
                    texts = [t for t in raw_texts if t]
            except Exception:
                pass

        # Fallback: try any visible depth-1 list on the page
        if not texts:
            try:
                items_container = page.locator(".filter-items.depth-1.p-t-8").first
                if items_container and items_container.count() > 0:
                    item_labels = items_container.locator("li, a, label, span, div")
                    raw_texts = [t.strip() for t in item_labels.all_inner_texts()]
                    texts = [t for t in raw_texts if t]
            except Exception:
                pass

        # Deduplicate while preserving order and remove very long lines (likely noise)
        seen = set()
        cleaned: List[str] = []
        for t in texts:
            tt = " ".join(t.split())
            if 0 < len(tt) < 120 and tt not in seen:
                seen.add(tt)
                cleaned.append(tt)

        # Save debug artifacts: screenshot + HTML
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        try:
            page.screenshot(path=os.path.join(out_dir, "swatchon_filters.png"), full_page=True)
        except Exception:
            pass
        try:
            html = page.content()
            with open(os.path.join(out_dir, "page.html"), "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            pass

        context.close()
        browser.close()
        return cleaned


def main():
    items = extract_woven_subcategories()
    print(json.dumps({"url": URL, "woven_subcategories": items}, ensure_ascii=False, indent=2))

    # Also write to a file for convenience
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "woven_subcategories.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"url": URL, "woven_subcategories": items}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON to: {out_path}")


if __name__ == "__main__":
    sys.exit(main())

