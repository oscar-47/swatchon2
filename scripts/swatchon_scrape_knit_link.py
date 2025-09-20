import os
import sys
import json
import time
from typing import List, Set, Dict

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

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


def build_category_page_url(category_ids: str, page: int) -> str:
    return f"https://swatchon.com/wholesale-fabric?categoryIds={category_ids}&sort=&page={page}&from=/wholesale-fabric"


def accept_cookies(page) -> None:
    try:
        cookie_selectors = [
            "button:has-text('Accept')",
            "button:has-text('I agree')",
            "button:has-text('同意')",
            "text=Accept",
        ]
        for sel in cookie_selectors:
            try:
                btn = page.locator(sel).first
                if btn and btn.count() > 0 and btn.is_visible():
                    btn.click()
                    page.wait_for_timeout(800)
                    return
            except Exception:
                continue
    except Exception:
        pass


def collect_page_links(page, page_num: int, category_name: str) -> List[str]:
    try:
        container = page.locator("div.search-items").first
        if container.count() == 0:
            return []
        container.wait_for(state="visible", timeout=10000)
        cards = container.locator(".c-quality")
        total = cards.count()
        if total == 0:
            return []
        page_links: List[str] = []
        for i in range(total):
            try:
                c = cards.nth(i)
                href = c.get_attribute("href", timeout=2000)
                if href:
                    if href.startswith("/"):
                        href = "https://swatchon.com" + href
                    elif not href.startswith("http"):
                        href = "https://swatchon.com/" + href
                    page_links.append(href)
            except Exception:
                continue
        return page_links
    except Exception:
        return []


def scrape_category(category_name: str, category_config: dict, target_count: int = 150, max_pages: int = 20) -> dict:
    all_links: Set[str] = set()
    page_results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.set_default_timeout(30000)
        current_page = 1
        cookies_accepted = False
        while len(all_links) < target_count and current_page <= max_pages:
            try:
                url = build_category_page_url(category_config["categoryIds"], current_page)
                response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if not response or response.status != 200:
                    current_page += 1
                    continue
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except PlaywrightTimeoutError:
                    pass
                try:
                    page.wait_for_selector(".c-quality", timeout=10000)
                    time.sleep(1.5)
                except PlaywrightTimeoutError:
                    pass
                if not cookies_accepted:
                    accept_cookies(page)
                    cookies_accepted = True
                page_links = collect_page_links(page, current_page, category_name)
                if not page_links:
                    break
                before = len(all_links)
                all_links.update(page_links)
                page_results.append({
                    "page": current_page,
                    "url": url,
                    "links_found": len(page_links),
                    "new_unique_links": len(all_links) - before,
                    "total_unique_links": len(all_links),
                })
                if len(all_links) >= target_count:
                    break
                current_page += 1
                time.sleep(0.8)
            except Exception:
                current_page += 1
                continue
        context.close()
        browser.close()
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
    print("Knit categories link scraper starting...")
    base_output_dir = os.path.join(os.getcwd(), "outputs", "knit_categories")
    os.makedirs(base_output_dir, exist_ok=True)
    overall = {"total_categories": len(CATEGORIES), "done": 0, "total_links": 0, "category_results": {}}
    for i, (category_name, category_config) in enumerate(CATEGORIES.items(), 1):
        try:
            print(f"\nProcessing {i}/{len(CATEGORIES)}: {category_name}")
            result = scrape_category(category_name, category_config, target_count=150)
            cat_dir = os.path.join(base_output_dir, category_name)
            os.makedirs(cat_dir, exist_ok=True)
            out_path = os.path.join(cat_dir, f"{category_name}_links_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            overall["done"] += 1
            overall["total_links"] += result["actual_count"]
            overall["category_results"][category_name] = {"links_count": result["actual_count"], "pages_scraped": result["pages_scraped"], "file_path": out_path}
            print(f"Saved -> {out_path}")
            if i < len(CATEGORIES):
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

