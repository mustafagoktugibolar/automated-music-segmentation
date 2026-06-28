#!/usr/bin/env python
"""
Train + evaluate GBDT segment-label classifier.
================================================
Song-level grouped 60/20/20 splits across multiple random seeds.
Produces a full set of evaluation artifacts under models/evaluation/.

Usage
-----
    python scripts/label_training/train_label_classifier.py
    python scripts/label_training/train_label_classifier.py --seeds 42 7 99
    python scripts/label_training/train_label_classifier.py --no-multi-seed
    python scripts/label_training/train_label_classifier.py --merge-mode transition

Output
------
    models/segment_label_clf.joblib
    models/segment_label_clf.meta.json
    models/evaluation/
        metrics_by_seed.csv
        classification_report_train.json
        classification_report_val.json
        classification_report_test.json
        confusion_matrix_val.csv
        confusion_matrix_test.csv
        misclassifications_test.csv
        split_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PARQUET_PATH = os.path.join(_app_root, "data", "label_training", "segments.parquet")
MODELS_DIR   = os.path.join(_app_root, "models")
EVAL_DIR     = os.path.join(MODELS_DIR, "evaluation")
MODEL_JOBLIB = os.path.join(MODELS_DIR, "segment_label_clf.joblib")
MODEL_META   = os.path.join(MODELS_DIR, "segment_label_clf.meta.json")

# ── Label merge config ────────────────────────────────────────────────────────
# "none"       → original 9 classes unchanged
# "transition" → Pre-Chorus + Bridge → Transition
# "other"      → Pre-Chorus + Bridge → Other
_MERGE_MAPS: dict[str, dict[str, str]] = {
    "none":       {},
    "transition": {"Pre-Chorus": "Transition", "Bridge": "Transition"},
    "other":      {"Pre-Chorus": "Other",      "Bridge": "Other"},
}

DEFAULT_SEEDS    = [42, 123, 2024, 7, 99]
META_COLS        = {"song_id", "dataset", "segment_idx", "start", "end", "label"}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> "pd.DataFrame":
    import pandas as pd
    if not os.path.exists(path):
        print(f"[error] Training data not found: {path}")
        print("        Run scripts/label_training/prepare_label_dataset.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} segments from {df['song_id'].nunique()} songs.")
    return df


def apply_label_merge(df: "pd.DataFrame", merge_mode: str) -> "pd.DataFrame":
    mapping = _MERGE_MAPS.get(merge_mode, {})
    if not mapping:
        return df
    df = df.copy()
    n_before = df["label"].value_counts()
    df["label"] = df["label"].replace(mapping)
    n_after = df["label"].value_counts()
    print(f"\nLabel merge (mode='{merge_mode}'):")
    for src, dst in mapping.items():
        cnt = n_before.get(src, 0)
        print(f"  {src} → {dst}  ({cnt} segments)")
    print(f"Classes: {sorted(df['label'].unique())}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Splitting
# ═══════════════════════════════════════════════════════════════════════════════

def make_grouped_split(
    groups: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Song-level 3-way split. Returns (train_idx, val_idx, test_idx) as segment indices."""
    from sklearn.model_selection import GroupShuffleSplit

    n = len(groups)
    idx = np.arange(n)

    # Step 1: carve out test songs
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tv_idx, test_idx = next(gss1.split(idx, groups=groups))

    # Step 2: from remaining songs, carve out val
    val_frac = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    train_local, val_local = next(gss2.split(tv_idx, groups=groups[tv_idx]))

    train_idx = tv_idx[train_local]
    val_idx   = tv_idx[val_local]
    return train_idx, val_idx, test_idx


def check_split_integrity(
    train_ids: set, val_ids: set, test_ids: set
) -> None:
    """Raise if any song appears in more than one split."""
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    if tv or tt or vt:
        msg = "SPLIT LEAKAGE DETECTED:\n"
        if tv: msg += f"  train ∩ val  = {tv}\n"
        if tt: msg += f"  train ∩ test = {tt}\n"
        if vt: msg += f"  val   ∩ test = {vt}\n"
        raise RuntimeError(msg)
    print("  Split integrity: OK (no song appears in more than one split)")


def print_split_diagnostics(
    df: "pd.DataFrame",
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    diag: dict[str, Any] = {}

    for name, idx in splits.items():
        sub = df.iloc[idx]
        n_songs = sub["song_id"].nunique()
        n_segs  = len(sub)
        lc      = sub["label"].value_counts()
        lp      = (lc / n_segs * 100).round(1)
        avg_segs = n_segs / max(n_songs, 1)

        diag[name] = {
            "n_songs":    n_songs,
            "n_segments": n_segs,
            "avg_segs_per_song": round(avg_segs, 1),
            "label_counts": lc.to_dict(),
            "label_pct":    lp.to_dict(),
        }

        if "dataset" in sub.columns:
            diag[name]["dataset_distribution"] = sub["dataset"].value_counts().to_dict()

        if "end" in sub.columns and "start" in sub.columns:
            dur = sub["end"] - sub["start"]
            diag[name]["segment_duration"] = {
                "min": round(float(dur.min()), 2),
                "max": round(float(dur.max()), 2),
                "mean": round(float(dur.mean()), 2),
            }

        print(f"\n  ── {name.upper()} split ──")
        print(f"     Songs: {n_songs}  |  Segments: {n_segs}  |  Avg segs/song: {avg_segs:.1f}")
        print("     Label distribution:")
        for label in lc.index:
            print(f"       {label:<16} {lc[label]:>5}  ({lp[label]:>5.1f}%)")

    return diag


# ═══════════════════════════════════════════════════════════════════════════════
# Feature building
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(df: "pd.DataFrame"):
    """Return (X, y, groups, label_encoder, feature_cols)."""
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X      = df[feature_cols].values.astype(np.float32)
    groups = df["song_id"].values

    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)

    print(f"\nFeature matrix: {X.shape}  |  Classes: {list(le.classes_)}")
    return X, y, groups, le, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_weights(y: np.ndarray) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    backend: str,
):
    """Train GBDT with grouped early stopping on the external validation set.

    X_val is passed as eval_set for early stopping only — it is never included
    in the training rows.
    """
    sw_train = _sample_weights(y_train)
    sw_val   = _sample_weights(y_val)

    if backend == "lightgbm":
        try:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=2000, learning_rate=0.05,
                num_leaves=63, min_child_samples=20,
                reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            print("  External validation set used as LightGBM eval_set (not in training rows).")
            clf.fit(
                X_train, y_train, sample_weight=sw_train,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[sw_val],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            print(f"  Early stopping at iteration {clf.best_iteration_}.")
            return clf
        except ImportError:
            print("[warn] lightgbm not installed — falling back to HistGradientBoosting")

    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=1000, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                early_stopping_rounds=30, random_state=42, n_jobs=-1,
            )
            print("  External validation set used as XGBoost eval_set (not in training rows).")
            clf.fit(X_train, y_train, sample_weight=sw_train,
                    eval_set=[(X_val, y_val)], verbose=False)
            return clf
        except ImportError:
            print("[warn] xgboost not installed — falling back to HistGradientBoosting")

    from sklearn.ensemble import HistGradientBoostingClassifier

    print("  External validation set is not used for fitting.")
    print("  Early stopping uses an internal 10% split of the training set.")
    clf = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sw_train)
    print(f"  Early stopping at iteration {clf.n_iter_}.")
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    clf,
    le,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    n_songs: int,
) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score
    )

    y_pred = clf.predict(X)
    acc    = float(accuracy_score(y, y_pred))
    macro  = float(f1_score(y, y_pred, average="macro",    zero_division=0))
    weighted = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y, y_pred,
        labels=list(range(len(le.classes_))),
        target_names=list(le.classes_),
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=list(range(len(le.classes_)))).tolist()

    print(f"\n── {split_name} ({n_songs} songs) ──")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro-F1   : {macro:.4f}")
    print(f"  Weighted-F1: {weighted:.4f}")
    print(classification_report(
        y, y_pred,
        labels=list(range(len(le.classes_))),
        target_names=list(le.classes_),
        zero_division=0,
    ))

    return {
        "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
        "report": report, "confusion_matrix": cm, "y_pred": y_pred.tolist(),
    }


def top_misclassifications(
    cm: list[list[int]], classes: list[str], top_n: int = 15
) -> list[dict]:
    rows = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and cm[i][j] > 0:
                rows.append({"true": true_cls, "predicted": pred_cls, "count": cm[i][j]})
    rows.sort(key=lambda r: -r["count"])
    return rows[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# Artifact saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_evaluation_artifacts(
    classes:       list[str],
    train_result:  dict,
    val_result:    dict,
    test_result:   dict,
    split_diag:    dict,
    seed_rows:     list[dict],
    meta:          dict,
) -> None:
    import pandas as pd

    os.makedirs(EVAL_DIR, exist_ok=True)

    # ── Classification reports ────────────────────────────────────────────────
    for name, res in [("train", train_result), ("val", val_result), ("test", test_result)]:
        path = os.path.join(EVAL_DIR, f"classification_report_{name}.json")
        with open(path, "w") as f:
            json.dump(res["report"], f, indent=2)

    # ── Confusion matrices ────────────────────────────────────────────────────
    for name, res in [("val", val_result), ("test", test_result)]:
        cm_df = pd.DataFrame(res["confusion_matrix"], index=classes, columns=classes)
        cm_df.to_csv(os.path.join(EVAL_DIR, f"confusion_matrix_{name}.csv"))

    # ── Top misclassifications ────────────────────────────────────────────────
    mc = top_misclassifications(test_result["confusion_matrix"], classes)
    pd.DataFrame(mc).to_csv(
        os.path.join(EVAL_DIR, "misclassifications_test.csv"), index=False
    )
    print("\nTop misclassifications (test):")
    for row in mc[:10]:
        print(f"  {row['true']:<16} → {row['predicted']:<16} : {row['count']}")

    # ── Metrics by seed ───────────────────────────────────────────────────────
    if seed_rows:
        pd.DataFrame(seed_rows).to_csv(
            os.path.join(EVAL_DIR, "metrics_by_seed.csv"), index=False
        )

    # ── Split diagnostics ─────────────────────────────────────────────────────
    with open(os.path.join(EVAL_DIR, "split_diagnostics.json"), "w") as f:
        json.dump(split_diag, f, indent=2)

    # ── Model meta ────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts saved → {EVAL_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Single-seed run
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_seed(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    le, feature_cols: list[str],
    df: "pd.DataFrame",
    val_size: float, test_size: float,
    backend: str, seed: int,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

    train_idx, val_idx, test_idx = make_grouped_split(
        groups, val_size=val_size, test_size=test_size, random_state=seed
    )

    train_songs = set(groups[train_idx])
    val_songs   = set(groups[val_idx])
    test_songs  = set(groups[test_idx])

    check_split_integrity(train_songs, val_songs, test_songs)

    if verbose:
        print(f"\n  Train: {len(X[train_idx])} segs / {len(train_songs)} songs")
        print(f"  Val:   {len(X[val_idx])} segs / {len(val_songs)} songs")
        print(f"  Test:  {len(X[test_idx])} segs / {len(test_songs)} songs")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    t0 = time.perf_counter()
    clf = train_model(X_train, y_train, X_val, y_val, backend)
    if verbose:
        print(f"  Training done in {time.perf_counter() - t0:.1f}s.")

    train_res = evaluate_model(clf, le, X_train, y_train, "Train",      len(train_songs))
    val_res   = evaluate_model(clf, le, X_val,   y_val,   "Validation", len(val_songs))
    test_res  = evaluate_model(clf, le, X_test,  y_test,  "Test",       len(test_songs))

    return {
        "seed": seed,
        "clf": clf,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "train_songs": len(train_songs),
        "val_songs":   len(val_songs),
        "test_songs":  len(test_songs),
        "train_result": train_res,
        "val_result":   val_res,
        "test_result":  test_res,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_seed_evaluation(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    le, feature_cols: list[str],
    df: "pd.DataFrame",
    seeds: list[int],
    val_size: float, test_size: float,
    backend: str,
) -> list[dict]:
    print(f"\n{'#'*60}")
    print(f"  MULTI-SEED EVALUATION  (seeds: {seeds})")
    print(f"{'#'*60}")

    rows = []
    for seed in seeds:
        result = run_single_seed(
            X, y, groups, le, feature_cols, df,
            val_size, test_size, backend, seed, verbose=False
        )
        row = {
            "seed":            seed,
            "val_accuracy":    round(result["val_result"]["accuracy"], 4),
            "val_macro_f1":    round(result["val_result"]["macro_f1"], 4),
            "val_weighted_f1": round(result["val_result"]["weighted_f1"], 4),
            "test_accuracy":   round(result["test_result"]["accuracy"], 4),
            "test_macro_f1":   round(result["test_result"]["macro_f1"], 4),
            "test_weighted_f1":round(result["test_result"]["weighted_f1"], 4),
        }
        rows.append(row)
        print(
            f"  seed={seed:>5} | "
            f"val_acc={row['val_accuracy']:.3f}  val_F1={row['val_macro_f1']:.3f} | "
            f"test_acc={row['test_accuracy']:.3f}  test_F1={row['test_macro_f1']:.3f}"
        )

    vals  = np.array([r["val_macro_f1"]  for r in rows])
    tests = np.array([r["test_macro_f1"] for r in rows])
    print(f"\n  Val  Macro-F1 : {vals.mean():.3f} ± {vals.std():.3f}")
    print(f"  Test Macro-F1 : {tests.mean():.3f} ± {tests.std():.3f}")
    print(f"  Val–Test gap  : {(vals - tests).mean():.3f} ± {(vals - tests).std():.3f}")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train segment-label GBDT classifier.")
    parser.add_argument("--input",        default=PARQUET_PATH)
    parser.add_argument("--output-model", default=MODEL_JOBLIB)
    parser.add_argument("--val-size",     type=float, default=0.20)
    parser.add_argument("--test-size",    type=float, default=0.20)
    parser.add_argument("--backend",      default="lightgbm", choices=["lightgbm", "sklearn", "xgboost"])
    parser.add_argument("--merge-mode",   default="none", choices=list(_MERGE_MAPS))
    parser.add_argument("--seeds",        nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--no-multi-seed", action="store_true",
                        help="Skip multi-seed evaluation; run only seed=args.seeds[0]")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    import pandas as pd
    df = load_dataset(args.input)
    df = apply_label_merge(df, args.merge_mode)

    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")

    X, y, groups, le, feature_cols = build_features(df)

    # ── Multi-seed ────────────────────────────────────────────────────────────
    seed_rows: list[dict] = []
    if not args.no_multi_seed and len(args.seeds) > 1:
        seed_rows = run_multi_seed_evaluation(
            X, y, groups, le, feature_cols, df,
            args.seeds, args.val_size, args.test_size, args.backend,
        )

    # ── Full run with primary seed (diagnostics + artifact saving) ────────────
    primary_seed = args.seeds[0]
    print(f"\n{'#'*60}")
    print(f"  PRIMARY RUN  (seed={primary_seed}) — with full diagnostics")
    print(f"{'#'*60}")

    train_idx, val_idx, test_idx = make_grouped_split(
        groups, val_size=args.val_size, test_size=args.test_size,
        random_state=primary_seed,
    )

    train_songs = set(groups[train_idx])
    val_songs   = set(groups[val_idx])
    test_songs  = set(groups[test_idx])

    print("\nSplit integrity check:")
    check_split_integrity(train_songs, val_songs, test_songs)

    print("\nSplit diagnostics:")
    split_diag = print_split_diagnostics(df, train_idx, val_idx, test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"\nTraining {args.backend} (early stopping on validation set) …")
    t0 = time.perf_counter()
    clf = train_model(X_train, y_train, X_val, y_val, args.backend)
    print(f"Done in {time.perf_counter() - t0:.1f}s.")

    train_res = evaluate_model(clf, le, X_train, y_train, "Train",      len(train_songs))
    val_res   = evaluate_model(clf, le, X_val,   y_val,   "Validation", len(val_songs))
    test_res  = evaluate_model(clf, le, X_test,  y_test,  "Test",       len(test_songs))

    # ── Save model ────────────────────────────────────────────────────────────
    import joblib
    bundle = {
        "clf": clf, "label_encoder": le,
        "feature_names": feature_cols, "classes": list(le.classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(bundle, args.output_model)
    print(f"\nModel saved → {args.output_model}")

    # ── Build meta ────────────────────────────────────────────────────────────
    multi_summary: dict = {}
    if seed_rows:
        vals  = [r["val_macro_f1"]  for r in seed_rows]
        tests = [r["test_macro_f1"] for r in seed_rows]
        multi_summary = {
            "seeds": args.seeds,
            "val_macro_f1_mean":  round(float(np.mean(vals)),  4),
            "val_macro_f1_std":   round(float(np.std(vals)),   4),
            "test_macro_f1_mean": round(float(np.mean(tests)), 4),
            "test_macro_f1_std":  round(float(np.std(tests)),  4),
            "val_test_gap_mean":  round(float(np.mean(np.array(vals) - np.array(tests))), 4),
        }

    meta = {
        "trained_at":           datetime.now(timezone.utc).isoformat(),
        "dataset":              args.input,
        "backend":              args.backend,
        "merge_mode":           args.merge_mode,
        "classes":              list(le.classes_),
        "feature_count":        len(feature_cols),
        "primary_seed":         primary_seed,
        "grouped_split":        True,
        "leakage_check_passed": True,
        "val_size":             args.val_size,
        "test_size":            args.test_size,
        "n_train_songs":        len(train_songs),
        "n_val_songs":          len(val_songs),
        "n_test_songs":         len(test_songs),
        "n_train_segs":         int(len(X_train)),
        "n_val_segs":           int(len(X_val)),
        "n_test_segs":          int(len(X_test)),
        "train_accuracy":       round(train_res["accuracy"],    4),
        "train_macro_f1":       round(train_res["macro_f1"],    4),
        "train_weighted_f1":    round(train_res["weighted_f1"], 4),
        "val_accuracy":         round(val_res["accuracy"],      4),
        "val_macro_f1":         round(val_res["macro_f1"],      4),
        "val_weighted_f1":      round(val_res["weighted_f1"],   4),
        "test_accuracy":        round(test_res["accuracy"],     4),
        "test_macro_f1":        round(test_res["macro_f1"],     4),
        "test_weighted_f1":     round(test_res["weighted_f1"],  4),
        "multi_seed":           multi_summary if multi_summary else None,
        "per_class_f1_test": {
            k: round(v["f1-score"], 4)
            for k, v in test_res["report"].items()
            if k not in {"accuracy", "macro avg", "weighted avg"}
        },
        "confusion_matrix_test":   test_res["confusion_matrix"],
        "confusion_labels":        list(le.classes_),
    }

    save_evaluation_artifacts(
        classes      = list(le.classes_),
        train_result = train_res,
        val_result   = val_res,
        test_result  = test_res,
        split_diag   = split_diag,
        seed_rows    = seed_rows,
        meta         = meta,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Merge mode  : {args.merge_mode}")
    print(f"  Classes     : {list(le.classes_)}")
    print(f"  Features    : {len(feature_cols)}")
    print(f"  Primary seed: {primary_seed}")
    print(f"\n  {'Split':<10} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12}")
    print(f"  {'-'*44}")
    for name, res in [("Train", train_res), ("Val", val_res), ("Test", test_res)]:
        print(f"  {name:<10} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} {res['weighted_f1']:>12.4f}")

    if multi_summary:
        print(f"\n  Multi-seed test Macro-F1: "
              f"{multi_summary['test_macro_f1_mean']:.3f} ± {multi_summary['test_macro_f1_std']:.3f}  "
              f"(gap: {multi_summary['val_test_gap_mean']:.3f})")
        print("\n  ► Use multi-seed test Macro-F1 for reporting, not the single-seed val score.")


if __name__ == "__main__":
    main()
