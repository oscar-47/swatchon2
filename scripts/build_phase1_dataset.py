#!/usr/bin/env python3
"""
Build FabricFlow phase-1 dataset slices from scraped SwatchOn details.

Pipeline:
1) Read class targets from config JSON.
2) Scan detail folders for JSON/JPG pairs (supports __vN variant outputs).
3) Class-wise near-duplicate filtering via dHash64 hamming distance.
4) Copy selected images into FabricFlow_Dataset/{L1}/{Class}/swatchon.
5) Write per-class manifest.csv and overall summary JSON.
"""

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image


@dataclass
class SampleRecord:
    json_path: Path
    image_path: Path
    detail_url: str
    variant_index: int
    variant_hint: str
    image_hash: Optional[int] = None


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_variant_index(path: Path) -> int:
    m = re.search(r"__v(\d+)$", path.stem)
    if m:
        return int(m.group(1))
    return 1


def discover_detail_dirs(source_root: Path, key: str) -> List[Path]:
    candidates = [
        source_root / "knit_category_details_hq" / key,
        source_root / "knit_category_details" / key,
        source_root / "woven_category_details_hq" / key,
        source_root / "woven_category_details" / key,
    ]
    return [p for p in candidates if p.is_dir()]


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def collect_records_from_dir(detail_dir: Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for json_path in sorted(detail_dir.glob("*.json")):
        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            continue

        payload = read_json(json_path)
        detail_url = str(payload.get("detail_url") or "")

        variant_index = parse_variant_index(json_path)
        variant_hint = ""
        images = payload.get("images")
        if isinstance(images, list) and images:
            first = images[0] if isinstance(images[0], dict) else {}
            if isinstance(first, dict):
                v_idx = first.get("variant_index")
                if isinstance(v_idx, int) and v_idx > 0:
                    variant_index = v_idx
                v_hint = first.get("variant_hint")
                if isinstance(v_hint, str):
                    variant_hint = v_hint.strip()

        records.append(
            SampleRecord(
                json_path=json_path,
                image_path=image_path,
                detail_url=detail_url,
                variant_index=variant_index,
                variant_hint=variant_hint,
            )
        )
    return records


def dhash64(image_path: Path) -> int:
    with Image.open(image_path) as img:
        resampling = getattr(Image, "Resampling", Image)
        lanczos = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
        img = img.convert("L").resize((9, 8), lanczos)
        pixels = list(img.getdata())
    rows = [pixels[i * 9:(i + 1) * 9] for i in range(8)]
    bits = 0
    for r in rows:
        for i in range(8):
            bits = (bits << 1) | (1 if r[i] > r[i + 1] else 0)
    return bits


def hamming_distance(a: int, b: int) -> int:
    xor = a ^ b
    try:
        return xor.bit_count()  # Python 3.8+
    except AttributeError:
        return bin(xor).count("1")


def select_unique_records(records: List[SampleRecord], target: int, threshold: int) -> Tuple[List[SampleRecord], int, int]:
    selected: List[SampleRecord] = []
    selected_hashes: List[int] = []
    hash_failures = 0
    duplicates = 0

    # Stable ordering: variant-first to distribute colors from same URL.
    records = sorted(records, key=lambda r: (r.variant_index, str(r.image_path)))

    for rec in records:
        if len(selected) >= target:
            break

        try:
            rec.image_hash = dhash64(rec.image_path)
        except Exception:
            hash_failures += 1
            continue

        is_dup = any(hamming_distance(rec.image_hash, h) <= threshold for h in selected_hashes)
        if is_dup:
            duplicates += 1
            continue

        selected.append(rec)
        selected_hashes.append(rec.image_hash)

    return selected, duplicates, hash_failures


def class_filename(dataset_class: str, seq: int) -> str:
    class_token = dataset_class.lower()
    return f"{class_token}_base_swatchon_{seq:04d}.jpg"


def write_manifest(manifest_path: Path, rows: List[Dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seq",
                "filename",
                "source_path",
                "detail_url",
                "variant_index",
                "variant_hint",
                "image_hash",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run(
    config_path: Path,
    source_root: Path,
    dataset_root: Path,
    threshold: int,
    dry_run: bool,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    class_plan = cfg.get("class_plan") or []
    summary: Dict[str, Any] = {
        "config": str(config_path),
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "threshold": threshold,
        "classes": [],
    }

    for item in class_plan:
        key = str(item.get("key") or "").strip()
        if not key:
            continue

        l1 = str(item.get("l1") or "").strip()
        dataset_class = str(item.get("dataset_class") or key).strip()
        target = int(item.get("target") or 0)

        detail_dirs = discover_detail_dirs(source_root, key)
        records: List[SampleRecord] = []
        for d in detail_dirs:
            records.extend(collect_records_from_dir(d))

        selected, duplicates, hash_failures = select_unique_records(records, target=target, threshold=threshold)

        class_dir = dataset_root / l1 / dataset_class
        swatchon_dir = class_dir / "swatchon"
        manifest_rows: List[Dict[str, Any]] = []

        for i, rec in enumerate(selected, start=1):
            fname = class_filename(dataset_class, i)
            dst = swatchon_dir / fname
            if not dry_run:
                swatchon_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rec.image_path, dst)
            manifest_rows.append(
                {
                    "seq": i,
                    "filename": fname,
                    "source_path": str(rec.image_path),
                    "detail_url": rec.detail_url,
                    "variant_index": rec.variant_index,
                    "variant_hint": rec.variant_hint,
                    "image_hash": rec.image_hash,
                }
            )

        manifest_path = class_dir / "manifest.csv"
        if not dry_run:
            write_manifest(manifest_path, manifest_rows)

        class_summary = {
            "key": key,
            "dataset_class": dataset_class,
            "target": target,
            "available_records": len(records),
            "selected": len(selected),
            "duplicates_filtered": duplicates,
            "hash_failures": hash_failures,
            "detail_dirs": [str(p) for p in detail_dirs],
            "manifest": str(manifest_path),
            "status": "ok" if len(selected) >= target else "insufficient",
        }
        summary["classes"].append(class_summary)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FabricFlow phase-1 dataset with dedup + quotas")
    parser.add_argument(
        "--config",
        default=os.path.join("scripts", "config", "targets_phase1_fabricflow.json"),
        help="Path to phase target config JSON",
    )
    parser.add_argument("--source-root", default="outputs", help="Root folder containing scraped outputs")
    parser.add_argument("--dataset-root", default="FabricFlow_Dataset", help="Target dataset root folder")
    parser.add_argument("--threshold", type=int, default=5, help="Near-duplicate hamming threshold")
    parser.add_argument("--dry-run", action="store_true", help="Do not copy files or write manifests")
    args = parser.parse_args()

    summary = run(
        config_path=Path(args.config),
        source_root=Path(args.source_root),
        dataset_root=Path(args.dataset_root),
        threshold=args.threshold,
        dry_run=args.dry_run,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.dry_run:
        summary_path = Path(args.dataset_root) / "phase1_build_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
