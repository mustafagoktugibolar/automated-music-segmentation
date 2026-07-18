#!/usr/bin/env python
"""Run and compare label-classifier ablation experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_app_root = _here.parents[2]
EVAL_EXPERIMENTS_DIR = _app_root / "models" / "evaluation" / "experiments"

DEFAULT_EXPERIMENTS = [
    ("other", "full", "default"),
    ("other", "acoustic_context", "balanced"),
    ("other", "no_repetition", "balanced"),
    ("other", "no_context", "balanced"),
    ("other", "acoustic", "balanced"),
]
STRONG_REPEAT_FEATURE_SETS = ["full", "no_repetition"]


def _experiment_name(merge_mode: str, feature_set: str, preset: str) -> str:
    return f"{merge_mode}_{feature_set}_{preset}"


def _train_command(args, merge_mode: str, feature_set: str, preset: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_here / "train_label_classifier.py"),
        "--merge-mode", merge_mode,
        "--feature-set", feature_set,
        "--regularization-preset", preset,
        "--experiment-name", _experiment_name(merge_mode, feature_set, preset),
        "--backend", args.backend,
        "--seeds", *[str(s) for s in args.seeds],
    ]
    if args.no_multi_seed:
        cmd.append("--no-multi-seed")
    if args.extra_parquet:
        cmd.extend(["--extra-parquet", *args.extra_parquet])
    return cmd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    vals = {k: [] for k in rows[0] if k != "seed"}
    for row in rows:
        for key in vals:
            if row.get(key, "") != "":
                vals[key].append(float(row[key]))
    out: dict[str, float] = {}
    for key, numbers in vals.items():
        if numbers:
            mean = sum(numbers) / len(numbers)
            var = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = var ** 0.5
    return out


def build_summary() -> Path:
    EVAL_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp_dir in sorted(p for p in EVAL_EXPERIMENTS_DIR.iterdir() if p.is_dir()):
        meta = _read_json(exp_dir / "meta.json")
        if not meta:
            continue
        metrics = _read_metrics(exp_dir / "metrics_by_seed.csv")
        train = float(meta.get("train_macro_f1") or 0.0)
        test = float(meta.get("test_macro_f1") or 0.0)
        row = {
            "experiment_name": exp_dir.name,
            "merge_mode": meta.get("merge_mode"),
            "feature_set": meta.get("feature_set"),
            "regularization_preset": meta.get("regularization_preset"),
            "backend": meta.get("backend"),
            "val_macro_f1_mean": metrics.get("val_macro_f1_mean", (meta.get("multi_seed") or {}).get("val_macro_f1_mean")),
            "val_macro_f1_std": metrics.get("val_macro_f1_std", (meta.get("multi_seed") or {}).get("val_macro_f1_std")),
            "test_macro_f1_mean": metrics.get("test_macro_f1_mean", (meta.get("multi_seed") or {}).get("test_macro_f1_mean")),
            "test_macro_f1_std": metrics.get("test_macro_f1_std", (meta.get("multi_seed") or {}).get("test_macro_f1_std")),
            "val_test_gap_mean": (meta.get("multi_seed") or {}).get("val_test_gap_mean"),
            "train_macro_f1_primary": meta.get("train_macro_f1"),
            "val_macro_f1_primary": meta.get("val_macro_f1"),
            "test_macro_f1_primary": meta.get("test_macro_f1"),
            "train_test_gap_primary": round(train - test, 4),
        }
        rows.append(row)

    def sort_key(row: dict):
        score = row.get("test_macro_f1_mean") or row.get("test_macro_f1_primary") or 0.0
        gap = row.get("train_test_gap_primary") or 999.0
        return (-float(score), float(gap))

    rows.sort(key=sort_key)
    out_path = EVAL_EXPERIMENTS_DIR / "summary.csv"
    columns = [
        "experiment_name", "merge_mode", "feature_set", "regularization_preset", "backend",
        "val_macro_f1_mean", "val_macro_f1_std", "test_macro_f1_mean", "test_macro_f1_std",
        "val_test_gap_mean", "train_macro_f1_primary", "val_macro_f1_primary",
        "test_macro_f1_primary", "train_test_gap_primary",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved -> {out_path}")
    print("\nBest experiments:")
    for row in rows[:10]:
        score = row.get("test_macro_f1_mean") or row.get("test_macro_f1_primary")
        print(f"  {row['experiment_name']:<36} test_F1={score} gap={row['train_test_gap_primary']}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label classifier ablation experiments.")
    parser.add_argument("--extra-parquet", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 99])
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "sklearn", "xgboost"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-multi-seed", action="store_true")
    parser.add_argument("--include-strong", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    experiments = list(DEFAULT_EXPERIMENTS)
    if args.include_strong:
        experiments.extend(("other", fs, "strong") for fs in STRONG_REPEAT_FEATURE_SETS)

    if not args.summary_only:
        for merge_mode, feature_set, preset in experiments:
            cmd = _train_command(args, merge_mode, feature_set, preset)
            print("\n$ " + " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=_app_root, check=True)

    build_summary()


if __name__ == "__main__":
    main()
