#!/usr/bin/env python3
"""Upload core_set knit images to Google Drive.
Only uploads confirmed mappings; skips lacoste, plain_pique, skip_stitch."""

import subprocess, json, sys, os
from pathlib import Path

SRC = Path("/Users/oscar/Downloads/fabricflow/swatchon2/coredataset")

# core_set folder IDs on Drive
CORE_SET_IDS = {
    "Jersey":       "1Uatlww007Odx2MZ09jaeVE5KjJEBnhSF",
    "Rib_Knit":     "1WJ-oKC0cVtEgVCsesOsZ4OFjVgyIgvH8",
    "Double_Jersey":"1VRKJ6Ivp4mej3RMOq81cN050S1rzwfle",
    "Interlock":    "1g6_fYQn1ii7V8kRpnF3wckmKt4yDDO5t",
    "Purl_Knit":    "19NLG-9GwIp49MIn5m_5UzSEhElFeeozI",
    "French_Terry": "14jBhvYsYJZ1poN5c532xMK_zYSKv4nnF",
}

# filename prefix → class
FILE_MAP = {
    # Jersey
    "jersey": "Jersey",
    # Rib_Knit
    "1×1_rib_back": "Rib_Knit",
    "1×1_rib_front": "Rib_Knit",
    "2×2_rib_(2-in_1-out)_back": "Rib_Knit",
    "2×2_rib_(2-in_1-out)_front": "Rib_Knit",
    "2×2_rib_(2-in_2-out)_back": "Rib_Knit",
    "2×2_rib_(2-in_2-out)_front": "Rib_Knit",
    "simple_rib": "Rib_Knit",
    "board_rib": "Rib_Knit",
    "half_cardigan": "Rib_Knit",
    "full_cardigan": "Rib_Knit",
    "ottoman": "Rib_Knit",
    "racked_stitch": "Rib_Knit",
    "tubular_(split_welt)": "Rib_Knit",
    # Double_Jersey
    "double_jersey": "Double_Jersey",
    "double_face": "Double_Jersey",
    "ponte_di_roma": "Double_Jersey",
    "double_pique": "Double_Jersey",
    "full_milano": "Double_Jersey",
    "half_milano": "Double_Jersey",
    # Interlock
    "interlock": "Interlock",
    # Purl_Knit
    "links": "Purl_Knit",
    "seed_stitch": "Purl_Knit",
    "double_seed_stitch": "Purl_Knit",
    "moss_stitch": "Purl_Knit",
    # French_Terry
    "plush": "French_Terry",
}

import re
SUFFIX_RE = re.compile(r"_base_core_set_\d+x\.jpeg$")

uploaded = skipped = errors = 0

for f in sorted(SRC.glob("*.jpeg")):
    fname = f.name
    prefix = SUFFIX_RE.sub("", fname)
    cls = FILE_MAP.get(prefix)
    if not cls:
        print(f"SKIP: {fname}  (prefix={prefix})")
        skipped += 1
        continue

    folder_id = CORE_SET_IDS[cls]
    meta = json.dumps({"name": fname, "parents": [folder_id]})
    params = json.dumps({"includeItemsFromAllDrives": "true", "supportsAllDrives": "true"})

    print(f"UPLOAD: {fname} → {cls}/core_set ... ", end="", flush=True)
    try:
        r = subprocess.run(
            ["gws", "drive", "files", "create",
             "--json", meta, "--params", params,
             "--upload", str(f)],
            capture_output=True, text=True, timeout=60
        )
        resp = json.loads(r.stdout)
        if "id" in resp:
            print(resp["id"])
            uploaded += 1
        else:
            print(f"FAIL: {resp}")
            errors += 1
    except Exception as e:
        print(f"ERROR: {e}")
        errors += 1

print(f"\nDone: {uploaded} uploaded, {skipped} skipped, {errors} errors")
