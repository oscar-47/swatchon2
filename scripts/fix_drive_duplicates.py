#!/usr/bin/env python3
"""Fix duplicate Drive folders: move source subfolders from new class folders
into old class folders, then delete the new (empty) class folders."""
import subprocess, json, sys

def gws(params_dict):
    """Run gws drive files list with params."""
    r = subprocess.run(
        ["gws", "drive", "files", "list", "--params", json.dumps(params_dict)],
        capture_output=True, text=True
    )
    try:
        return json.loads(r.stdout).get("files", [])
    except:
        print(f"ERROR: {r.stderr}", flush=True)
        return []

def gws_update(file_id, body, params=None):
    """Update a file (move it)."""
    cmd = ["gws", "drive", "files", "update", "--id", file_id, "--json", json.dumps(body)]
    if params:
        cmd += ["--params", json.dumps(params)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0

def gws_delete(file_id):
    """Delete a file/folder."""
    r = subprocess.run(
        ["gws", "drive", "files", "delete", "--id", file_id,
         "--params", '{"supportsAllDrives": true}'],
        capture_output=True, text=True
    )
    return r.returncode == 0

def list_children(parent_id):
    """List immediate children of a folder."""
    return gws({
        "supportsAllDrives": True, "includeItemsFromAllDrives": True,
        "q": f"'{parent_id}' in parents and trashed=false",
        "fields": "files(id,name,mimeType,createdTime)",
        "pageSize": 200,
    })

def count_files(folder_id):
    """Count non-folder children."""
    files = gws({
        "supportsAllDrives": True, "includeItemsFromAllDrives": True,
        "q": f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false",
        "fields": "files(id)", "pageSize": 1,
    })
    return len(files)

p = lambda msg: print(msg, flush=True)

# Old (Feb 26) and New (Mar 27) class folder IDs
KNIT_ID = "1LPKz8vQhDHbb_kHs2Bbhw8EloVo1-Ya4"
WOVEN_ID = "1hTwPoHLaI0GtRUwEQGtLKJsa0UZDKrPC"

OLD_IDS = {
    "Cable_Knit":     "1VGUcPDI1Loq5I8LrfpYhFVa65LoTMuV_",
    "Double_Jersey":  "12WFHr_vmHyFrCzmpGyElrJXw4ZCYBfHM",
    "Intarsia":       "1O0DcIA7jcrpkla6H2Tlx_vkw61gArozc",
    "Purl_Knit":      "18lm0uYtG4i7d6KhaFrlBEbdipmgQfTw0",
    "Raschel":        "1JqO2DRPAhh_A-RTD2CMrcPvSCj2S6CVR",
}

NEW_IDS = {
    "Cable_Knit":     "1l6o9q7lfeIwonwGpfSZwg_5nrwKq7WDf",
    "Double_Jersey":  "1qEcwTnzCT2zvGaYrIeLtnnBTQ2TcVBPO",
    "Intarsia":       "1oPS7yMN--5rbeH5Llo_kHFlOCBlhSYXu",
    "Purl_Knit":      "1zmZAaXnuCfUmTMXZkI2-EXG3lJZeK_ks",
    "Raschel":        "1x0usTsjSQ5pbont978ovIJGYTHL19yax",
}

# Basket_Hopsack - check if old exists
WOVEN_CHILDREN = list_children(WOVEN_ID)
old_basket = [f for f in WOVEN_CHILDREN if f["name"] == "Basket_Hopsack"]
if len(old_basket) > 1:
    old_basket.sort(key=lambda x: x["createdTime"])
    OLD_IDS["Basket_Hopsack"] = old_basket[0]["id"]
    NEW_IDS["Basket_Hopsack"] = old_basket[1]["id"]
elif len(old_basket) == 1:
    # Only new one exists, rename approach
    p("Basket_Hopsack: only one folder found, checking...")
    NEW_IDS["Basket_Hopsack"] = "168nChrHc9ebCg_otlHH9recCamOE3SYc"
    # Check if there's an old one too
    for f in WOVEN_CHILDREN:
        if f["name"] == "Basket_Hopsack" and f["id"] != "168nChrHc9ebCg_otlHH9recCamOE3SYc":
            OLD_IDS["Basket_Hopsack"] = f["id"]

p("=" * 60)
p("Fix Drive Duplicate Folders")
p("=" * 60)

for cls in ["Double_Jersey", "Cable_Knit", "Purl_Knit", "Intarsia", "Raschel", "Basket_Hopsack"]:
    old_id = OLD_IDS.get(cls)
    new_id = NEW_IDS.get(cls)

    if not old_id or not new_id:
        p(f"\n[{cls}] Missing old or new ID, skip")
        continue

    p(f"\n{'='*50}")
    p(f"  {cls}")
    p(f"  Old: {old_id}")
    p(f"  New: {new_id}")
    p(f"{'='*50}")

    # List source subfolders in new class folder
    new_children = list_children(new_id)
    p(f"  New folder has {len(new_children)} children")

    # Group by name to find duplicates within new folder
    by_name = {}
    for child in new_children:
        name = child["name"]
        by_name.setdefault(name, []).append(child)

    for src_name, copies in by_name.items():
        p(f"\n  --- {src_name} ({len(copies)} copies) ---")

        if len(copies) > 1:
            # Keep the one with files, delete empty duplicates
            copies.sort(key=lambda x: x["createdTime"])
            # Check which has files
            for i, c in enumerate(copies):
                has_files = count_files(c["id"])
                p(f"    Copy {i}: {c['id']} created={c['createdTime'][:19]} hasFiles={has_files}")

            # Keep first one with files
            kept = None
            for c in copies:
                if count_files(c["id"]) > 0:
                    if kept is None:
                        kept = c
                    else:
                        # Duplicate with files — just delete it (images already in first)
                        p(f"    DELETE duplicate: {c['id']}")
                        gws_delete(c["id"])
                else:
                    p(f"    DELETE empty: {c['id']}")
                    gws_delete(c["id"])

            if kept is None:
                kept = copies[0]  # keep first if all empty
                p(f"    KEEP (fallback): {kept['id']}")
        else:
            kept = copies[0]

        # Move kept folder from new class → old class
        p(f"    MOVE {kept['id']} → old folder {old_id}")
        ok = gws_update(kept["id"], {},
            {"addParents": old_id, "removeParents": new_id, "supportsAllDrives": True})
        p(f"    {'✓' if ok else '✗'} moved")

    # Delete the now-empty new class folder
    p(f"\n  DELETE new class folder: {new_id}")
    gws_delete(new_id)
    p(f"  ✓ done with {cls}")

p(f"\n{'='*60}")
p("ALL DONE — duplicates cleaned up")
p(f"{'='*60}")
