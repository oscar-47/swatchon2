#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise ValueError(f"{path} is not a JSON list")
    return obj


def _model_size_map(weights_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not weights_root.exists():
        return out
    for run_dir in weights_root.iterdir():
        if not run_dir.is_dir():
            continue
        best = run_dir / "best.pth"
        if not best.exists():
            continue
        model = run_dir.name
        stage_suffixes = ("_stage1", "_woven", "_knit")
        for suffix in stage_suffixes:
            if model.endswith(suffix):
                model = model[: -len(suffix)]
                break
        if model in out:
            continue
        size_bytes = best.stat().st_size
        out[model] = {
            "model_weight_bytes": size_bytes,
            "model_weight_mb": round(size_bytes / (1024 * 1024), 2),
            "model_weight_from_run": run_dir.name,
        }
    return out


def _model_param_estimate_map() -> dict[str, dict[str, Any]]:
    try:
        import torch
        from torchvision import models
    except Exception:
        return {}

    builders = {
        "efficientnet_v2_s": lambda: models.efficientnet_v2_s(weights=None),
        "convnext_base": lambda: models.convnext_base(weights=None),
        "maxvit_t": lambda: models.maxvit_t(weights=None),
        "regnet_y_8gf": lambda: models.regnet_y_8gf(weights=None),
        "densenet161": lambda: models.densenet161(weights=None),
        "resnext101_32x8d": lambda: models.resnext101_32x8d(weights=None),
    }

    out: dict[str, dict[str, Any]] = {}
    for model_name, build in builders.items():
        try:
            model = build()
            params = sum(p.numel() for p in model.parameters())
            est_bytes = int(params * 4)  # fp32 estimate
            out[model_name] = {
                "model_params_m": round(params / 1_000_000, 2),
                "model_param_est_mb": round(est_bytes / (1024 * 1024), 2),
            }
            del model
            gc.collect()
        except Exception:
            continue
    return out


def _normalize_finetune(
    rows: list[dict[str, Any]], source_name: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source": source_name,
                "run": row.get("run"),
                "model": row.get("model"),
                "strategy": row.get("strategy"),
                "task": row.get("task"),
                "test_acc": row.get("test_acc"),
                "test_macro_f1": row.get("test_macro_f1"),
                "roc_auc": row.get("roc_auc"),
                "best_epoch": row.get("best_epoch"),
                "peak_gpu_mb": row.get("peak_gpu_mb"),
                "status": row.get("status"),
            }
        )
    return out


def _normalize_matrix(
    rows: list[dict[str, Any]], source_name: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source": source_name,
                "run": row.get("run"),
                "model": row.get("model"),
                "strategy": None,
                "task": row.get("stage"),
                "test_acc": row.get("test_acc"),
                "test_macro_f1": row.get("test_macro_f1"),
                "roc_auc": row.get("roc_auc"),
                "best_epoch": row.get("best_epoch"),
                "peak_gpu_mb": None,
                "status": row.get("status"),
            }
        )
    return out


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("-inf")


def _rank_within_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task"]), []).append(row)
    ranked: list[dict[str, Any]] = []
    for task, items in grouped.items():
        items_sorted = sorted(
            items,
            key=lambda r: (
                _safe_float(r["test_macro_f1"]),
                _safe_float(r["test_acc"]),
                str(r["model"]),
                str(r["run"]),
            ),
            reverse=True,
        )
        for idx, row in enumerate(items_sorted, start=1):
            row_copy = dict(row)
            row_copy["rank_in_task"] = idx
            row_copy["task"] = task
            ranked.append(row_copy)
    return ranked


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge finetune and matrix run summaries into a single leaderboard."
    )
    ap.add_argument(
        "--finetune-summary",
        default="runs_finetune/finetune_summary.json",
        help="Path to finetune summary JSON.",
    )
    ap.add_argument(
        "--matrix-summary",
        default="runs20260203/multi_model_summary.json",
        help="Path to multi-model summary JSON.",
    )
    ap.add_argument(
        "--weights-root",
        default="runs20260203",
        help="Directory used to infer per-model weight size from best.pth.",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for generated leaderboard files.",
    )
    ap.add_argument(
        "--prefix",
        default="merged_leaderboard",
        help="Output filename prefix.",
    )
    ap.add_argument(
        "--include-non-ok",
        action="store_true",
        help="Include rows whose status is not OK.",
    )
    args = ap.parse_args()

    finetune_path = Path(args.finetune_summary)
    matrix_path = Path(args.matrix_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finetune_rows = _normalize_finetune(
        _load_json(finetune_path), source_name=finetune_path.parent.name
    )
    matrix_rows = _normalize_matrix(
        _load_json(matrix_path), source_name=matrix_path.parent.name
    )
    rows = finetune_rows + matrix_rows

    if not args.include_non_ok:
        rows = [row for row in rows if str(row.get("status", "")).upper() == "OK"]

    size_map = _model_size_map(Path(args.weights_root))
    param_map = _model_param_estimate_map()
    for row in rows:
        model_name = str(row.get("model"))
        size_info = size_map.get(str(row.get("model")))
        param_info = param_map.get(model_name)
        if size_info:
            row.update(size_info)
        else:
            row["model_weight_bytes"] = None
            row["model_weight_mb"] = None
            row["model_weight_from_run"] = None
        if param_info:
            row.update(param_info)
        else:
            row["model_params_m"] = None
            row["model_param_est_mb"] = None
        if row["model_weight_mb"] is None and row["model_param_est_mb"] is not None:
            row["model_weight_mb"] = row["model_param_est_mb"]
            row["model_weight_from_run"] = "estimated_fp32_params"

    ranked = _rank_within_task(rows)
    ranked_sorted = sorted(
        ranked,
        key=lambda r: (
            str(r["task"]),
            int(r["rank_in_task"]),
        ),
    )

    best_by_task: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for row in ranked_sorted:
        task = str(row["task"])
        if task in seen_tasks:
            continue
        seen_tasks.add(task)
        best_by_task.append(row)

    best_by_task_model: list[dict[str, Any]] = []
    seen_task_model: set[tuple[str, str]] = set()
    for row in ranked_sorted:
        key = (str(row["task"]), str(row["model"]))
        if key in seen_task_model:
            continue
        seen_task_model.add(key)
        best_by_task_model.append(row)

    strategy_task_avg: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ranked_sorted:
        strategy = row.get("strategy")
        if not strategy:
            continue
        key = (str(row["task"]), str(strategy))
        grouped.setdefault(key, []).append(row)
    for (task, strategy), items in sorted(grouped.items()):
        accs = [_safe_float(x["test_acc"]) for x in items]
        f1s = [_safe_float(x["test_macro_f1"]) for x in items]
        strategy_task_avg.append(
            {
                "task": task,
                "strategy": strategy,
                "n_runs": len(items),
                "avg_test_acc": _avg(accs),
                "avg_test_macro_f1": _avg(f1s),
            }
        )

    base_cols = [
        "rank_in_task",
        "task",
        "source",
        "run",
        "model",
        "strategy",
        "test_acc",
        "test_macro_f1",
        "roc_auc",
        "best_epoch",
        "peak_gpu_mb",
        "model_params_m",
        "model_param_est_mb",
        "model_weight_mb",
        "model_weight_bytes",
        "model_weight_from_run",
        "status",
    ]

    all_csv = out_dir / f"{args.prefix}_all.csv"
    best_task_csv = out_dir / f"{args.prefix}_best_by_task.csv"
    best_task_model_csv = out_dir / f"{args.prefix}_best_by_task_model.csv"
    strategy_task_avg_csv = out_dir / f"{args.prefix}_strategy_task_avg.csv"
    all_json = out_dir / f"{args.prefix}_all.json"

    _write_csv(all_csv, ranked_sorted, base_cols)
    _write_csv(best_task_csv, best_by_task, base_cols)
    _write_csv(best_task_model_csv, best_by_task_model, base_cols)
    _write_csv(
        strategy_task_avg_csv,
        strategy_task_avg,
        ["task", "strategy", "n_runs", "avg_test_acc", "avg_test_macro_f1"],
    )
    all_json.write_text(json.dumps(ranked_sorted, ensure_ascii=False, indent=2) + "\n")

    print(f"Total rows: {len(ranked_sorted)}")
    print(f"Tasks: {sorted({str(r['task']) for r in ranked_sorted})}")
    print(f"Wrote {all_csv}")
    print(f"Wrote {best_task_csv}")
    print(f"Wrote {best_task_model_csv}")
    print(f"Wrote {strategy_task_avg_csv}")
    print(f"Wrote {all_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
