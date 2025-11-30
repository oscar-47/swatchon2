#!/usr/bin/env python3
"""
Enhanced scraping script with adaptive limits and high-quality image download.

This script orchestrates the scraping of fabric images with:
1. Original/high-quality image URLs (not small thumbnails)
2. Adaptive limits based on model performance
3. Priority-based execution (P0 > P1 > P2)

Usage:
    python scripts/run_enhanced_scraping.py --mode all
    python scripts/run_enhanced_scraping.py --mode p0  # Only critical categories
    python scripts/run_enhanced_scraping.py --mode p1  # Only data-insufficient categories
    python scripts/run_enhanced_scraping.py --mode knit
    python scripts/run_enhanced_scraping.py --mode woven
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✅ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {description}")
        print(f"Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Enhanced fabric scraping with adaptive limits")
    parser.add_argument("--mode", choices=["all", "p0", "p1", "p2", "knit", "woven"], 
                        default="all", help="Scraping mode")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    # Define category groups
    P0_KNIT = ["Jacquard Knit"]  # 39.1% accuracy
    P0_WOVEN = ["Dobby"]         # 21.7% accuracy
    
    P1_KNIT = ["Crepe Knit", "Tricot", "Pique"]  # Data insufficient
    
    P2_KNIT = ["Double", "Low Gauge Knit", "Pile Knit"]
    P2_WOVEN = ["Double Weave", "Jacquard Weave", "Plain", "Satin Weave"]
    
    GOOD_KNIT = ["Single", "Mesh", "Lace Knit"]
    GOOD_WOVEN = ["Eyelet", "Pile Weave", "Ripstop", "Twill Weave"]

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   ENHANCED FABRIC SCRAPING - HIGH QUALITY                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Mode: {args.mode.upper()}
Dry Run: {args.dry_run}

Key Improvements:
✓ Original/Large quality images (not small thumbnails)
✓ Adaptive limits based on model performance
✓ Priority-based execution (P0 > P1 > P2)

Priority Levels:
🔴 P0 (CRITICAL): Dobby (21.7%), Jacquard Knit (39.1%)
🟡 P1 (DATA INSUFFICIENT): Crepe Knit (51 imgs), Tricot (80 imgs), Pique (83 imgs)
🟢 P2 (MODERATE): Double, Low Gauge Knit, Pile Knit, etc.
""")

    tasks = []
    
    # Determine which tasks to run
    if args.mode in ["all", "p0", "knit"]:
        tasks.append(("python", "scripts/scrape_knit_category_details.py", "Knit Categories (Adaptive Limits)"))
    
    if args.mode in ["all", "p0", "woven"]:
        tasks.append(("python", "scripts/scrape_woven_category_details.py", "Woven Categories (Adaptive Limits)"))

    if args.dry_run:
        print("\n🔍 DRY RUN - Commands that would be executed:\n")
        for task in tasks:
            print(f"  {' '.join(task[:-1])}")
        print("\nRun without --dry-run to execute.")
        return

    # Execute tasks
    results = []
    for task in tasks:
        cmd = task[:-1]
        description = task[-1]
        success = run_command(cmd, description)
        results.append((description, success))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for desc, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {desc}")
    
    total = len(results)
    succeeded = sum(1 for _, s in results if s)
    print(f"\nTotal: {succeeded}/{total} tasks succeeded")
    
    if succeeded == total:
        print("\n🎉 All tasks completed successfully!")
        print("\nNext steps:")
        print("1. Verify image quality: Check that images are high-resolution (>800px)")
        print("2. Review data counts per category")
        print("3. Retrain models with enhanced dataset")
    else:
        print("\n⚠️  Some tasks failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

