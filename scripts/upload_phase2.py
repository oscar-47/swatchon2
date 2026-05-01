#!/usr/bin/env python3
"""Upload Phase 2 (6 new classes) images to Google Drive."""
import json, os, subprocess, sys, time

UPLOAD_MAP = {
    ("KNIT", "Cable_Knit", "pexels"): "1prYknIyrgGpCxsGPS8n4lKlNqIaqKqvk",
    ("KNIT", "Cable_Knit", "unsplash"): "1PEQXdzn_KW212ZDmyCLHLNZOSymnjM_6",
    ("WOVEN", "Basket_Hopsack", "pexels"): "1GI0pdELiosfYUmKiq_UluuY9fiaY74Cw",
    ("WOVEN", "Basket_Hopsack", "unsplash"): "1q9p5b-tNFcGWRU2-oSkB8NcIV0bJtndU",
    ("KNIT", "Purl_Knit", "pexels"): "13gpMAv1CYmTTWvoky1Yg9ghOhDOsKvHJ",
    ("KNIT", "Purl_Knit", "unsplash"): "1L_0USLM7q1Hrgf7oXwvI8LO582yKYtJF",
    ("KNIT", "Double_Jersey", "pexels"): "1s2kyvl6neUyURryLYJJOLM5LwimQDQ3b",
    ("KNIT", "Double_Jersey", "unsplash"): "1-W4O_nVzMAlAZCw8EGdo8IDtuur2FuAh",
    ("KNIT", "Intarsia", "pexels"): "19YbPPgjLXIFAttrYPc-mnCq3vc59UZPP",
    ("KNIT", "Raschel", "pexels"): "1P84A_m2iuigwYVIpfJPsXkAXnx8h8Fui",
}

def upload_file(path, name, folder_id):
    cmd = ["gws", "drive", "files", "create",
           "--json", json.dumps({"name": name, "parents": [folder_id]}),
           "--params", json.dumps({"supportsAllDrives": True}),
           "--upload", path]
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return True
        if "rateLimitExceeded" in r.stderr:
            time.sleep(5 * (attempt + 1))
        else:
            return False
    return False

grand_ok = 0
grand_total = 0

for (l1, cls, src), drive_id in UPLOAD_MAP.items():
    local_dir = f"FabricFlow_Dataset/{l1}/{cls}/{src}"
    if not os.path.isdir(local_dir):
        continue
    files = sorted(f for f in os.listdir(local_dir) if f.endswith(".jpg"))
    total = len(files)
    grand_total += total
    ok = fail = 0
    print(f"\n{'='*50}")
    print(f"  {cls}/{src}: {total} files")
    print(f"{'='*50}")
    for i, f in enumerate(files, 1):
        if upload_file(os.path.join(local_dir, f), f, drive_id):
            ok += 1
        else:
            fail += 1
            print(f"  [{i}/{total}] FAIL: {f}")
        if i % 50 == 0:
            print(f"  [{i}/{total}] {ok} ok, {fail} fail")
    grand_ok += ok
    print(f"  Done: {ok}/{total}")

print(f"\n{'='*50}")
print(f"  GRAND TOTAL: {grand_ok}/{grand_total}")
print(f"{'='*50}")
