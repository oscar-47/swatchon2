#!/usr/bin/env python3
"""Upload Phase 2 images to Google Drive (professional sources only)."""
import subprocess, json, os, sys
from pathlib import Path

DATASET = Path("FabricFlow_Dataset")
SKIP = {"pexels", "unsplash"}

CLASS_IDS = {
    "Double_Jersey":  "1qEcwTnzCT2zvGaYrIeLtnnBTQ2TcVBPO",
    "Cable_Knit":     "1l6o9q7lfeIwonwGpfSZwg_5nrwKq7WDf",
    "Purl_Knit":      "1zmZAaXnuCfUmTMXZkI2-EXG3lJZeK_ks",
    "Intarsia":       "1oPS7yMN--5rbeH5Llo_kHFlOCBlhSYXu",
    "Raschel":        "1x0usTsjSQ5pbont978ovIJGYTHL19yax",
    "Basket_Hopsack": "168nChrHc9ebCg_otlHH9recCamOE3SYc",
}

L1 = {
    "Double_Jersey": "KNIT", "Cable_Knit": "KNIT", "Purl_Knit": "KNIT",
    "Intarsia": "KNIT", "Raschel": "KNIT", "Basket_Hopsack": "WOVEN",
}

def gws_create_folder(name, parent_id):
    r = subprocess.run([
        "gws", "drive", "files", "create",
        "--json", json.dumps({"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}),
        "--params", '{"supportsAllDrives": true}',
    ], capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
        return data.get("id", "")
    except:
        p(f"  ERROR creating folder {name}: {r.stderr}")
        return ""

def gws_upload(filepath, parent_id):
    r = subprocess.run([
        "gws", "drive", "files", "create",
        "--json", json.dumps({"name": os.path.basename(filepath), "parents": [parent_id]}),
        "--upload", str(filepath),
        "--params", '{"supportsAllDrives": true}',
    ], capture_output=True, text=True)
    return r.returncode == 0

def p(msg):
    print(msg, flush=True)

def main():
    grand = 0
    for cls, cls_drive_id in CLASS_IDS.items():
        l1 = L1[cls]
        cls_dir = DATASET / l1 / cls
        p(f"\n{'='*50}")
        p(f"  {cls} ({l1})")
        p(f"{'='*50}")

        if not cls_dir.exists():
            p("  [skip] dir not found")
            continue

        for source_dir in sorted(cls_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            src = source_dir.name
            if src in SKIP:
                p(f"  [skip] {src}")
                continue

            jpgs = sorted(source_dir.glob("*.jpg"))
            if not jpgs:
                continue

            p(f"\n  --- {src} ({len(jpgs)} images) ---")

            # Create source subfolder
            src_drive_id = gws_create_folder(src, cls_drive_id)
            if not src_drive_id:
                p(f"  ERROR: could not create folder")
                continue
            p(f"  Drive folder: {src_drive_id}")

            uploaded = 0
            for jpg in jpgs:
                if gws_upload(jpg, src_drive_id):
                    uploaded += 1
                if uploaded % 50 == 0:
                    p(f"    {uploaded}/{len(jpgs)}")
            p(f"  ✓ {uploaded}/{len(jpgs)} uploaded")
            grand += uploaded

    p(f"\n{'='*50}")
    p(f"  TOTAL: {grand} images uploaded")
    p(f"{'='*50}")

if __name__ == "__main__":
    main()
