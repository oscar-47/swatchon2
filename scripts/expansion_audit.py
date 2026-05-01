#!/usr/bin/env python3
"""
Expansion Audit Script
======================
Scans current scraped data vs target, identifies gaps, and produces an
actionable expansion plan.

Usage:
    python scripts/expansion_audit.py
    python scripts/expansion_audit.py --target 200
    python scripts/expansion_audit.py --json  # output machine-readable JSON
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent  # swatchon2 root

# New classification taxonomy (V3 - 10 Knit + 9 Woven)
NEW_TAXONOMY = {
    "Knit": {
        "Single":         {"detail_dir": "knit_category_details/Single",         "links_dir": "knit_categories/Single"},
        "Jacquard Knit":  {"detail_dir": "knit_category_details/Jacquard Knit",  "links_dir": "knit_categories/Jacquard Knit"},
        "Double":         {"detail_dir": "knit_category_details/Double",         "links_dir": "knit_categories/Double"},
        "Pile Knit":      {"detail_dir": "knit_category_details/Pile Knit",      "links_dir": "knit_categories/Pile Knit"},
        "Tricot":         {"detail_dir": "knit_category_details/Tricot",         "links_dir": "knit_categories/Tricot"},
        "Crepe Knit":     {"detail_dir": "knit_category_details/Crepe Knit",     "links_dir": "knit_categories/Crepe Knit"},
        "Pique":          {"detail_dir": "knit_category_details/Pique",          "links_dir": "knit_categories/Pique"},
        "Mesh":           {"detail_dir": "knit_category_details/Mesh",           "links_dir": "knit_categories/Mesh"},
        "Low Gauge Knit": {"detail_dir": "knit_category_details/Low Gauge Knit", "links_dir": "knit_categories/Low Gauge Knit"},
        "Lace Knit":      {"detail_dir": "knit_category_details/Lace Knit",      "links_dir": "knit_categories/Lace Knit"},
    },
    "Woven": {
        "Plain":          {"detail_dir": "woven_category_details/Plain",          "links_dir": "categories/Plain"},
        "Twill Weave":    {"detail_dir": "woven_category_details/Twill Weave",    "links_dir": "categories/Twill_Weave"},
        "Satin Weave":    {"detail_dir": "woven_category_details/Satin Weave",    "links_dir": "categories/Satin_Weave"},
        "Jacquard Weave": {"detail_dir": "woven_category_details/Jacquard Weave", "links_dir": "categories/Jacquard_Weave"},
        "Pile Weave":     {"detail_dir": "woven_category_details/Pile Weave",     "links_dir": "categories/Pile_Weave"},
        "Dobby":          {"detail_dir": "woven_category_details/Dobby",          "links_dir": "categories/Dobby"},
        "Double Weave":   {"detail_dir": "woven_category_details/Double Weave",   "links_dir": "categories/Double_Weave"},
        "Eyelet":         {"detail_dir": "woven_category_details/Eyelet",         "links_dir": "categories/Eyelet"},
        "Ripstop":        {"detail_dir": "woven_category_details/Ripstop",        "links_dir": "categories/Ripstop"},
    },
}

# Old model categories for migration reference
OLD_TAXONOMY = {
    "Knit": ["French_Terry", "Jacquard", "Mesh", "Rib", "Single_Jersey"],
    "Woven": ["Corduroy", "Jacquard", "Plain", "Satin", "Twill"],
}

MIGRATION_MAP = {
    # old_class -> (new_parent, new_class, notes)
    "French_Terry":  ("Knit",  "Pile Knit",  "French Terry is pile knit construction"),
    "Jacquard_K":    ("Knit",  "Jacquard Knit", "Direct mapping"),
    "Mesh":          ("Knit",  "Mesh",          "Direct mapping"),
    "Rib":           ("Knit",  "Single",        "Rib is a variant of single knit; could also be Double"),
    "Single_Jersey": ("Knit",  "Single",        "Direct mapping"),
    "Corduroy":      ("Woven", "Pile Weave",    "Corduroy is a pile weave variant"),
    "Jacquard_W":    ("Woven", "Jacquard Weave","Direct mapping"),
    "Plain":         ("Woven", "Plain",         "Direct mapping"),
    "Satin":         ("Woven", "Satin Weave",   "Direct mapping"),
    "Twill":         ("Woven", "Twill Weave",   "Direct mapping"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def count_images(detail_dir: Path) -> int:
    """Count .jpg files in a detail directory."""
    if not detail_dir.exists():
        return 0
    return len([f for f in detail_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])


def count_links(links_dir: Path) -> int:
    """Count links from the latest JSON in links directory."""
    if not links_dir.exists():
        return 0
    json_files = sorted(links_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        return 0
    try:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            links = data.get('all_links') or data.get('links') or []
        elif isinstance(data, list):
            links = data
        else:
            links = []
        return len(links)
    except Exception:
        return 0


def priority_label(deficit: int, current: int, target: int) -> str:
    """Assign priority based on deficit."""
    if deficit <= 0:
        return "✅ OK"
    ratio = current / target if target > 0 else 0
    if ratio < 0.4:
        return "🔴 CRITICAL"
    elif ratio < 0.7:
        return "🟠 HIGH"
    elif ratio < 0.9:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"


def source_recommendation(cat_name: str, current_images: int, total_links: int, target: int) -> str:
    """Recommend where to get more data."""
    deficit = target - current_images
    if deficit <= 0:
        return "No action needed"

    parts = []
    remaining_swatchon = total_links - current_images
    if remaining_swatchon > 0:
        can_fill = min(remaining_swatchon, deficit)
        parts.append(f"Swatchon: ~{can_fill} more (scrape remaining links)")
        deficit -= can_fill

    if deficit > 0:
        parts.append(f"External sources needed: ~{deficit} more")
        parts.append("  → Try: Fabric.com, MoodFabrics, Google Images")

    return " | ".join(parts)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_audit(target: int = 150, output_json: bool = False):
    outputs_dir = BASE / "outputs"

    results = {"timestamp": datetime.now().isoformat(), "target_per_category": target, "categories": {}}
    print_lines = []

    grand_total_images = 0
    grand_total_target = 0
    expansion_needed = []

    for parent_type, categories in NEW_TAXONOMY.items():
        print_lines.append(f"\n{'='*70}")
        print_lines.append(f"  {parent_type.upper()} ({len(categories)} categories)")
        print_lines.append(f"{'='*70}")
        print_lines.append(f"{'Category':<20} {'Images':>7} {'Links':>7} {'Target':>7} {'Deficit':>8} {'Priority':<15} Source")
        print_lines.append("-" * 110)

        for cat_name, paths in categories.items():
            detail_dir = outputs_dir / paths["detail_dir"]
            links_dir = outputs_dir / paths["links_dir"]

            n_images = count_images(detail_dir)
            n_links = count_links(links_dir)
            deficit = max(0, target - n_images)
            priority = priority_label(deficit, n_images, target)
            source = source_recommendation(cat_name, n_images, n_links, target)

            grand_total_images += n_images
            grand_total_target += target

            cat_key = f"{parent_type}/{cat_name}"
            results["categories"][cat_key] = {
                "parent_type": parent_type,
                "category": cat_name,
                "current_images": n_images,
                "total_links_available": n_links,
                "target": target,
                "deficit": deficit,
                "priority": priority,
                "source": source,
            }

            if deficit > 0:
                expansion_needed.append({
                    "parent_type": parent_type,
                    "category": cat_name,
                    "current": n_images,
                    "deficit": deficit,
                    "swatchon_remaining": max(0, n_links - n_images),
                    "external_needed": max(0, deficit - max(0, n_links - n_images)),
                })

            print_lines.append(
                f"{cat_name:<20} {n_images:>7} {n_links:>7} {target:>7} {deficit:>8} {priority:<15} {source}"
            )

    # Summary
    print_lines.append(f"\n{'='*70}")
    print_lines.append("  EXPANSION SUMMARY")
    print_lines.append(f"{'='*70}")
    print_lines.append(f"Total categories: {sum(len(c) for c in NEW_TAXONOMY.values())}")
    print_lines.append(f"Total images:     {grand_total_images} / {grand_total_target}")
    print_lines.append(f"Categories at target: {sum(1 for c in results['categories'].values() if c['deficit'] == 0)}")
    print_lines.append(f"Categories needing expansion: {len(expansion_needed)}")

    if expansion_needed:
        total_deficit = sum(e["deficit"] for e in expansion_needed)
        total_external = sum(e["external_needed"] for e in expansion_needed)
        total_swatchon = sum(min(e["deficit"], e["swatchon_remaining"]) for e in expansion_needed)

        print_lines.append(f"\nTotal images to collect: {total_deficit}")
        print_lines.append(f"  From Swatchon (remaining links): ~{total_swatchon}")
        print_lines.append(f"  From external sources:           ~{total_external}")

        print_lines.append(f"\n--- Expansion Tasks (sorted by deficit) ---")
        for e in sorted(expansion_needed, key=lambda x: -x["deficit"]):
            tag = "⚠️ EXTERNAL" if e["external_needed"] > 0 else "📥 Swatchon"
            print_lines.append(
                f"  [{tag}] {e['parent_type']}/{e['category']}: "
                f"{e['current']} → {target} (need {e['deficit']}, "
                f"swatchon: {e['swatchon_remaining']}, external: {e['external_needed']})"
            )

    # Old → New migration note
    print_lines.append(f"\n{'='*70}")
    print_lines.append("  OLD → NEW CATEGORY MIGRATION")
    print_lines.append(f"{'='*70}")
    for old, (parent, new, note) in MIGRATION_MAP.items():
        print_lines.append(f"  {old:<16} → {parent}/{new:<16} ({note})")

    results["expansion_needed"] = expansion_needed
    results["summary"] = {
        "total_images": grand_total_images,
        "total_target": grand_total_target,
        "complete_categories": sum(1 for c in results["categories"].values() if c["deficit"] == 0),
        "need_expansion": len(expansion_needed),
    }

    if output_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for line in print_lines:
            print(line)

    # Also save JSON report
    report_path = outputs_dir / "expansion_audit.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if not output_json:
        print(f"\n📁 Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit data expansion needs for V3 taxonomy")
    parser.add_argument("--target", type=int, default=150, help="Target images per category (default: 150)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()
    run_audit(target=args.target, output_json=args.json)
