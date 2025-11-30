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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# Load links from latest file under outputs/categories/<CategoryName>

def _find_latest_links_file(links_root: str, category_name: str) -> str | None:
    try:
        d = os.path.join(links_root, category_name)
        if not os.path.isdir(d):
            return None
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.json')]
        if not files:
            return None
        # For woven we prefer the largest file (older crawls often have more complete data)
        files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return files[0]
    except Exception:
        return None


def load_links_from_latest(links_root: str, category_name: str, limit: int = 150) -> List[str]:
    fpath = _find_latest_links_file(links_root, category_name)
    if not fpath:
        print(f"[warn] No link file found in {os.path.join(links_root, category_name)}")
        return []
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[warn] Failed to read links file: {fpath} -> {e}")
        return []
    links: List[str] = []
    if isinstance(obj, dict):
        cand = obj.get('all_links') or obj.get('links') or []
        if isinstance(cand, list):
            links = [str(x) for x in cand]
    elif isinstance(obj, list):
        links = [str(x) for x in obj]
    # de-duplicate preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq[:limit]


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


# Same retry wrapper as knit script

def process_link(link: str, out_json: str, sleep_sec: float, max_retries: int, jitter: float):
    """Run detail scrape with retries and jitter. Returns a tuple (status, name, out_json)."""
    name = os.path.splitext(os.path.basename(out_json))[0]
    if os.path.exists(out_json):
        return ("skip", name, out_json)
    attempt = 0
    while attempt <= max_retries:
        rc = run_detail_scrape(link, out_json)
        if rc == 0:
            return ("ok", name, out_json)
        # backoff + jitter
        delay = max(0.0, (2 ** attempt) * sleep_sec + random.uniform(0, max(0.0, jitter)))
        time.sleep(delay)
        attempt += 1
    return ("fail", name, out_json)


def main():
    parser = argparse.ArgumentParser(description="Scrape WOVEN category details with adaptive limits based on model performance")
    parser.add_argument("--limit", type=int, default=None, help="Override max items per category (default: use adaptive limits)")
    parser.add_argument("--base-out", default=os.path.join("outputs", "woven_category_details_hq"), help="Base output directory (HQ = High Quality)")
    parser.add_argument("--links-root", default=os.path.join("outputs", "categories"), help="Root directory where woven link JSONs are saved")
    parser.add_argument("--sleep", type=float, default=0.5, help="Base sleep seconds between items (used for backoff)")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel detail workers (default 3)")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per item on failure (default 2)")
    parser.add_argument("--jitter", type=float, default=0.5, help="Random jitter seconds added to backoff")
    args = parser.parse_args()

    os.makedirs(args.base_out, exist_ok=True)

    # Adaptive limits based on model performance analysis
    # P0 (极差): 400-500, P2 (中等): 250-300, Good: 150
    CATEGORY_LIMITS = {
        "Dobby": 500,            # P0: 21.7% accuracy - CRITICAL!!!
        "Double_Weave": 300,     # P2: 60.9% - moderate improvement needed
        "Jacquard_Weave": 300,   # P2: 60.9% - moderate improvement needed
        "Plain": 300,            # P2: 65.2% - moderate improvement needed
        "Satin_Weave": 300,      # P2: 65.2% - moderate improvement needed
        "Eyelet": 150,           # Excellent: 95.2% - maintain (补齐到150)
        "Pile_Weave": 150,       # Excellent: 95.7% - maintain current
        "Ripstop": 150,          # Good: 87.0% - maintain current
        "Twill_Weave": 150,      # Good: 87.0% - maintain current
    }

    # Order matters; iterate by priority (P0 first)
    category_keys = [
        "Dobby",           # P0 - HIGHEST PRIORITY!!!
        "Double_Weave",    # P2
        "Jacquard_Weave",  # P2
        "Plain",           # P2
        "Satin_Weave",     # P2
        "Eyelet",          # Good (补齐)
        "Pile_Weave",      # Excellent
        "Ripstop",         # Good
        "Twill_Weave",     # Good
    ]

    for key in category_keys:
        cfg = CATEGORIES[key]
        disp = DISPLAY_NAME[key]
        out_dir = os.path.join(args.base_out, disp)
        os.makedirs(out_dir, exist_ok=True)

        # Use adaptive limit or override
        category_limit = args.limit if args.limit is not None else CATEGORY_LIMITS.get(key, 150)

        print(f"\n===== {disp} (Target: {category_limit} images) =====")
        print(f"Loading links from {args.links_root}...")
        # NOTE: For woven, the links directory names follow the internal keys (e.g. "Double_Weave"),
        # not the human-readable display names (e.g. "Double Weave"). So we must use `key` here.
        links = load_links_from_latest(args.links_root, key, category_limit)
        print(f"Loaded {len(links)} links for {disp} (target: {category_limit})")

        total = len(links)
        if total == 0:
            print("No links found. Skipping.")
            continue

        # Submit tasks to thread pool (same as knit script)
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = []
            for i, link in enumerate(links, 1):
                name = extract_numeric_id(link) or f"item_{i:03d}"
                out_json = os.path.join(out_dir, f"{name}.json")
                if os.path.exists(out_json):
                    print(f"[{i}/{total}] Skip existing {name}")
                    continue
                futures.append(executor.submit(process_link, link, out_json, args.sleep, args.max_retries, args.jitter))

            done = 0
            for fut in as_completed(futures):
                status, name, out_path = fut.result()
                done += 1
                prefix = f"[{done}/{len(futures)}]"
                if status == "ok":
                    print(f"{prefix} {name} -> OK (+ JPG)")
                elif status == "skip":
                    print(f"{prefix} {name} -> SKIP")
                else:
                    print(f"{prefix} {name} -> FAILED")

    print("\nAll categories done.")


if __name__ == "__main__":
    sys.exit(main())

