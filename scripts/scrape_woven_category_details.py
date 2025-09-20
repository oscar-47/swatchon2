import os
import sys
import json
import time
import random
import argparse
import subprocess
import re
from typing import List, Dict, Set
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
# --- Anti-bot/stealth helpers (lightweight) ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.112 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.201 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.122 Safari/537.36",
]
LOCALES = ["en-US", "en-GB"]
TIMEZONES = ["UTC", "America/New_York", "Europe/Berlin", "Asia/Seoul", "Asia/Shanghai"]

_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
window.chrome = window.chrome || { runtime: {} };
"""

def _random_context_options():
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
        except Exception:
            time.sleep((2 ** attempt) * random.uniform(0.8, 1.6))
    return last_resp


# Categories aligned with scripts/test_all.py
CATEGORIES: Dict[str, Dict[str, str]] = {
    "Plain": {
        "categoryIds": "167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=167,181,178,179,182,169,176,168,193,192,172,258,187,234,173,170&sort=&from=/wholesale-fabric",
    },
    "Twill_Weave": {
        "categoryIds": "186,189,175,253,196,194",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=186,189,175,253,196,194&sort=&from=/wholesale-fabric",
    },
    "Satin_Weave": {
        "categoryIds": "254,255,256",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=254,255,256&sort=&from=/wholesale-fabric",
    },
    "Jacquard_Weave": {
        "categoryIds": "184,183",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=184,183&sort=&from=/wholesale-fabric",
    },
    "Pile_Weave": {
        "categoryIds": "188,247",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=188,247&sort=&from=/wholesale-fabric",
    },
    "Dobby": {
        "categoryIds": "171",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=171&sort=&from=/wholesale-fabric",
    },
    "Double_Weave": {
        "categoryIds": "185",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=185&sort=&from=/wholesale-fabric",
    },
    "Eyelet": {
        "categoryIds": "177",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=177&sort=&from=/wholesale-fabric",
    },
    "Ripstop": {
        "categoryIds": "191",
        "url": "https://swatchon.com/wholesale-fabric?categoryIds=191&sort=&from=/wholesale-fabric",
    },
}

DISPLAY_NAME = {
    "Plain": "Plain",
    "Twill_Weave": "Twill Weave",
    "Satin_Weave": "Satin Weave",
    "Jacquard_Weave": "Jacquard Weave",
    "Pile_Weave": "Pile Weave",
    "Dobby": "Dobby",
    "Double_Weave": "Double Weave",
    "Eyelet": "Eyelet",
    "Ripstop": "Ripstop",
}


def build_category_page_url(category_ids: str, page: int) -> str:
    return (
        f"https://swatchon.com/wholesale-fabric?categoryIds={category_ids}&sort=&page={page}&from=/wholesale-fabric"
    )


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
        res: List[str] = []
        for i in range(total):
            try:
                c = cards.nth(i)
                href = c.get_attribute("href", timeout=2000)
                if href:
                    if href.startswith("/"):
                        href = "https://swatchon.com" + href
                    elif not href.startswith("http"):
                        href = "https://swatchon.com/" + href
                    res.append(href)
            except Exception:
                continue
        return res
    except Exception:
        return []


def collect_links_for_category(category_name: str, category_config: dict, target_count: int = 150, max_pages: int = 20) -> List[str]:
    all_links: Set[str] = set()
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
        page.set_default_timeout(random.randint(28000, 38000))

        current_page = 1
        cookies_accepted = False
        while len(all_links) < target_count and current_page <= max_pages:
            try:
                url = build_category_page_url(category_config["categoryIds"], current_page)
                resp = safe_goto(page, url)
                status = _get_status(resp)
                if not resp or (status is not None and status >= 400):
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
                links = collect_page_links(page, current_page, category_name)
                if not links:
                    break
                before = len(all_links)
                all_links.update(links)
                if len(all_links) == before:
                    # no new links; likely last page
                    pass
                if len(all_links) >= target_count:
                    break
                current_page += 1
                time.sleep(random.uniform(0.6, 1.4))
            except Exception:
                current_page += 1
                continue
        context.close()
        browser.close()
    return list(sorted(all_links))[:target_count]


def extract_numeric_id(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    seg = path.split("/")[-1] if path else ""
    m = re.search(r"(\d+)$", seg)
    if m:
        return m.group(1)
    # fallback: keep last segment sanitized
    return (
        seg.replace("\\", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace("*", "-")
        .replace("?", "-")
        .replace("\"", "'")
        .replace("<", "(")
        .replace(">", ")")
        .replace("|", "-")
    )


def run_detail_scrape(url: str, out_json: str) -> int:
    cmd = [sys.executable, os.path.join("scripts", "swatchon_scrape_detail.py"), url, "--out", out_json]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Print a compact error snippet for debugging
        err = (proc.stderr or "").strip()
        out = (proc.stdout or "").strip()
        if err:
            print("    stderr:", err.splitlines()[-1][:400])
        if out:
            print("    stdout:", out.splitlines()[-1][:400])
        return proc.returncode
    return 0


def main():
    parser = argparse.ArgumentParser(description="Scrape up to N=150 details per woven category, saving JSON+JPG per category folder")
    parser.add_argument("--limit", type=int, default=150, help="Max items per category (default 150)")
    parser.add_argument("--base-out", default=os.path.join("outputs", "category_details"), help="Base output directory")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between items")
    args = parser.parse_args()

    os.makedirs(args.base_out, exist_ok=True)

    # Order matters; iterate by the fixed list
    category_keys = [
        "Plain",
        "Twill_Weave",
        "Satin_Weave",
        "Jacquard_Weave",
        "Pile_Weave",
        "Dobby",
        "Double_Weave",
        "Eyelet",
        "Ripstop",
    ]

    for key in category_keys:
        cfg = CATEGORIES[key]
        disp = DISPLAY_NAME[key]
        out_dir = os.path.join(args.base_out, disp)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n===== {disp} =====")
        print(f"Collecting links (limit {args.limit})...")
        links = collect_links_for_category(disp, cfg, target_count=args.limit)
        print(f"Collected {len(links)} links for {disp}")

        for i, link in enumerate(links, 1):
            name = extract_numeric_id(link) or f"item_{i:03d}"
            out_json = os.path.join(out_dir, f"{name}.json")
            if os.path.exists(out_json):
                print(f"[{i}/{len(links)}] Skip existing {name}")
                continue
            print(f"[{i}/{len(links)}] {name}")
            rc = run_detail_scrape(link, out_json)
            if rc != 0:
                print(f"  -> FAILED: exit={rc}")
            else:
                print(f"  -> OK: {out_json} (+ JPG)")
            time.sleep(args.sleep)

    print("\nAll categories done.")


if __name__ == "__main__":
    sys.exit(main())

