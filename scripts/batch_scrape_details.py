import argparse
import json
import os
import subprocess
import sys
import time
from urllib.parse import urlsplit
import re


def extract_numeric_id(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    seg = path.split("/")[-1] if path else ""
    m = re.search(r"(\d+)$", seg)
    if m:
        return m.group(1)
    # fallback: sanitized last segment
    return seg.replace("\\", "-").replace("/", "-").replace(":", "-").replace("*", "-") \
              .replace("?", "-").replace("\"", "'").replace("<", "(").replace(">", ")").replace("|", "-")


def main():
    parser = argparse.ArgumentParser(description="Batch run swatchon_scrape_detail.py for a list of links")
    parser.add_argument("--input", default=os.path.join("outputs", "c_quality_links_page1.json"),
                        help="Path to JSON with a 'links' array (default: outputs/c_quality_links_page1.json)")
    parser.add_argument("--outdir", default=os.path.join("outputs", "details"),
                        help="Directory to write per-link JSON results (default: outputs/details)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between items (default: 0.5)")
    args = parser.parse_args()

    # Load links
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    links = data.get("links") or []
    if not isinstance(links, list) or not links:
        print("No links found in input JSON.")
        return 1

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Total links: {len(links)}")
    for i, link in enumerate(links, 1):
        try:
            name = extract_numeric_id(link) or f"item_{i:03d}"
            out_path = os.path.join(args.outdir, f"{name}.json")
            print(f"[{i}/{len(links)}] {name}")

            # Run the detail scraper as a subprocess so we don't depend on package-style imports
            cmd = [sys.executable, os.path.join("scripts", "swatchon_scrape_detail.py"), link, "--out", out_path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  -> FAILED (exit={proc.returncode})\n  stderr: {proc.stderr.strip()}")
            else:
                print(f"  -> OK -> {out_path}")
            time.sleep(args.sleep)
        except Exception as e:
            print(f"  -> ERROR: {e}")
            time.sleep(args.sleep)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

