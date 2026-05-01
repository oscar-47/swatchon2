#!/usr/bin/env python3
"""
Upload dataset images to Chenwei's FabricFlow_Dataset on Google Drive (shared team drive).

Usage:
  python scripts/upload_to_drive.py                     # upload all
  python scripts/upload_to_drive.py --only Tricot       # one class
  python scripts/upload_to_drive.py --dry-run           # preview
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Drive folder IDs (Chenwei's FabricFlow_Dataset on shared team drive)
UPLOAD_TARGETS = {
    "Tricot": {
        "local_dir": "FabricFlow_Dataset/KNIT/Tricot/swatchon",
        "drive_folder_id": "1pqgsndwbeduBiHl2an_erm5E_xsKozyJ",
    },
    "Ribbed_Poplin": {
        "local_dir": "FabricFlow_Dataset/WOVEN/Ribbed_Poplin/swatchon",
        "drive_folder_id": "13-iEmV8ewbjYSEAEb-kohaBlCqH0sUie",
    },
    "Leno_Gauze": {
        "local_dir": "FabricFlow_Dataset/WOVEN/Leno_Gauze/swatchon",
        "drive_folder_id": "1arIRUxxj082OyyY21XQBr78o0FXa_zX_",
    },
    "Interlock": {
        "local_dir": "outputs/fwd_interlock/cropped_patches",
        "drive_folder_id": "1awius_Vmwc6kLo9zkKboc0t_BZfTvwde",
    },
}


def upload_file(local_path: str, filename: str, drive_folder_id: str) -> bool:
    """Upload a single file to Google Drive shared team drive."""
    cmd = [
        "gws", "drive", "files", "create",
        "--json", json.dumps({
            "name": filename,
            "parents": [drive_folder_id],
        }),
        "--params", json.dumps({"supportsAllDrives": True}),
        "--upload", local_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            # Check if it's a quota/rate limit error
            if "rateLimitExceeded" in result.stderr or "userRateLimitExceeded" in result.stderr:
                time.sleep(5)
                # Retry once
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                return result.returncode == 0
            print(f"    ERROR: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT uploading {filename}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload images to Google Drive")
    parser.add_argument("--only", type=str, default="", help="Comma-separated class names")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--ext", type=str, default=".jpg", help="File extension to upload")
    args = parser.parse_args()

    selected = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()

    grand_total = 0
    grand_ok = 0

    for name, cfg in UPLOAD_TARGETS.items():
        if selected and name not in selected:
            continue

        local_dir = cfg["local_dir"]
        drive_id = cfg["drive_folder_id"]

        if not os.path.isdir(local_dir):
            print(f"\n[SKIP] {name}: directory not found: {local_dir}")
            continue

        files = sorted(f for f in os.listdir(local_dir) if f.endswith(args.ext))
        total = len(files)
        grand_total += total

        print(f"\n{'='*50}")
        print(f"  {name}: {total} files → {drive_id}")
        print(f"  Local: {local_dir}")
        print(f"{'='*50}")

        if args.dry_run:
            print(f"  (dry-run) Would upload {total} files")
            grand_ok += total
            continue

        ok = 0
        fail = 0
        for i, f in enumerate(files, 1):
            path = os.path.join(local_dir, f)
            success = upload_file(path, f, drive_id)
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  [{i}/{total}] FAIL: {f}")

            if i % 25 == 0:
                print(f"  [{i}/{total}] {ok} uploaded, {fail} failed")

        grand_ok += ok
        print(f"  Done: {ok}/{total} uploaded, {fail} failed")

    print(f"\n{'='*50}")
    print(f"  TOTAL: {grand_ok}/{grand_total} uploaded")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main() or 0)
