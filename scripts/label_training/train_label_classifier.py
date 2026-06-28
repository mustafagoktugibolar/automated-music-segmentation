#!/usr/bin/env python
"""
Train a GBDT classifier for segment semantic labeling.
=======================================================

Reads data/label_training/segments.parquet produced by prepare_label_dataset.py,
performs a song-level group-aware 60/20/20 train/val/test split, trains a
HistGradientBoostingClassifier with early stopping on the validation set,
and saves the model bundle to models/segment_label_clf.joblib.

Usage
-----
    python scripts/train_label_classifier.py
    python scripts/train_label_classifier.py --backend xgboost
    python scripts/train_label_classifier.py --min-class-count 30

Output
------
    models/segment_label_clf.joblib    — model bundle
    models/segment_label_clf.meta.json — metadata (accuracy, per-class F1, …)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

import numpy as np

PARQUET_PATH   = os.path.join(_app_root, "data", "label_training", "segments.parquet")
MODELS_DIR     = os.path.join(_app_root, "models")
MODEL_JOBLIB   = os.path.join(MODELS_DIR, "segment_label_clf.joblib")
MODEL_META     = os.path.join(MODELS_DIR, "segment_label_clf.meta.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def _build_classifier(backend: str):
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                early_stopping_rounds=30,
                random_state=42,
                n_jobs=-1,
            )
        except ImportError:
            print("[warn] xgboost not installed; falling back to HistGradientBoosting.")

    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=None,   # we supply explicit val set via score_* workaround
        n_iter_no_change=30,
        random_state=42,
    )


def _song_level_split(X, y, groups, test_size: float, random_state: int):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train segment-label GBDT classifier.")
    parser.add_argument("--input",           default=PARQUET_PATH)
    parser.add_argument("--output-model",    default=MODEL_JOBLIB)
    parser.add_argument("--output-meta",     default=MODEL_META)
    parser.add_argument("--val-size",        type=float, default=0.20,
                        help="Fraction of songs for validation (default 0.20).")
    parser.add_argument("--test-size",       type=float, default=0.20,
                        help="Fraction of songs held out as final test (default 0.20).")
    parser.add_argument("--backend",         default="sklearn",
                        choices=["sklearn", "xgboost"])
    parser.add_argument("--min-class-count", type=int, default=20,
                        help="Merge labels with fewer segments into 'Other'.")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    import pandas as pd
    if not os.path.exists(args.input):
        print(f"Training data not found: {args.input}\nRun prepare_label_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} segments from {df['song_id'].nunique()} songs.")

    # ── Collapse rare labels ──────────────────────────────────────────────────
    counts = df["label"].value_counts()
    rare   = counts[counts < args.min_class_count].index
    if len(rare):
        print(f"Collapsing {len(rare)} rare labels → 'Other': {list(rare)}")
        df.loc[df["label"].isin(rare), "label"] = "Other"
    print("Label distribution:\n" + df["label"].value_counts().to_string())

    # ── Feature / target split ────────────────────────────────────────────────
    meta_cols    = {"song_id", "dataset", "segment_idx", "start", "end", "label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X      = df[feature_cols].values.astype(np.float32)
    labels = df["label"].values
    groups = df["song_id"].values

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    print(f"\nFeature matrix: {X.shape}  |  Classes: {list(le.classes_)}")

    # ── Song-level 60 / 20 / 20 split ────────────────────────────────────────
    # Step 1: carve out test (20% of all songs)
    trainval_idx, test_idx = _song_level_split(X, y, groups, args.test_size, random_state=42)

    X_trainval = X[trainval_idx];  y_trainval = y[trainval_idx];  g_trainval = groups[trainval_idx]
    X_test     = X[test_idx];      y_test     = y[test_idx]

    # Step 2: from remaining 80%, carve out val (25% → 20% of total)
    val_frac = args.val_size / (1.0 - args.test_size)
    train_idx2, val_idx2 = _song_level_split(
        X_trainval, y_trainval, g_trainval, val_frac, random_state=0
    )

    X_train = X_trainval[train_idx2];  y_train = y_trainval[train_idx2]
    X_val   = X_trainval[val_idx2];    y_val   = y_trainval[val_idx2]
    g_train = g_trainval[train_idx2]

    n_train_songs = len(set(g_train))
    n_val_songs   = len(set(g_trainval[val_idx2]))
    n_test_songs  = len(set(groups[test_idx]))
    print(f"\nTrain: {len(X_train)} segs / {n_train_songs} songs")
    print(f"Val:   {len(X_val)} segs / {n_val_songs} songs")
    print(f"Test:  {len(X_test)} segs / {n_test_songs} songs")

    # ── Train ─────────────────────────────────────────────────────────────────
    sample_weights = _compute_sample_weights(y_train)
    clf = _build_classifier(args.backend)

    print(f"\nTraining {clf.__class__.__name__} with early stopping on val …")
    t0 = time.perf_counter()

    if args.backend == "xgboost":
        clf.fit(X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=50)
    else:
        # HistGradientBoosting: fit on train+val combined, but use val for early stopping.
        # Sklearn's built-in early_stopping uses an internal fraction; to supply an explicit
        # val set we concatenate and pass validation_fraction based on val size.
        n_total = len(X_train) + len(X_val)
        val_frac_actual = len(X_val) / n_total
        clf.set_params(validation_fraction=val_frac_actual)
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
        sw_val = _compute_sample_weights(y_val)
        sw_fit = np.concatenate([sample_weights, sw_val])
        clf.fit(X_fit, y_fit, sample_weight=sw_fit)
        n_iters = clf.n_iter_
        print(f"Early stopping at iteration {n_iters}.")

    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.1f}s.")

    # ── Evaluate on val then test ─────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    def _report(split_name, X_s, y_s):
        y_pred = clf.predict(X_s)
        acc    = float(accuracy_score(y_s, y_pred))
        rep    = classification_report(
            y_s, y_pred,
            labels=list(range(len(le.classes_))),
            target_names=list(le.classes_),
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = float(rep["macro avg"]["f1-score"])
        print(f"\n── {split_name} ──")
        print(f"Accuracy : {acc:.4f}  |  Macro-F1 : {macro_f1:.4f}")
        print(classification_report(
            y_s, y_pred,
            labels=list(range(len(le.classes_))),
            target_names=list(le.classes_),
            zero_division=0,
        ))
        return acc, macro_f1, rep, y_pred

    val_acc, val_macro_f1, val_rep, _   = _report(f"Validation ({n_val_songs} songs)", X_val, y_val)
    test_acc, test_macro_f1, test_rep, y_pred_test = _report(f"Test ({n_test_songs} songs)", X_test, y_test)

    cm = confusion_matrix(y_test, y_pred_test).tolist()

    # ── Save model bundle ─────────────────────────────────────────────────────
    import joblib

    bundle = {
        "clf":           clf,
        "label_encoder": le,
        "feature_names": feature_cols,
        "classes":       list(le.classes_),
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "dataset":       args.input,
        "n_train":       int(len(X_train)),
        "n_val":         int(len(X_val)),
        "n_test":        int(len(X_test)),
        "val_accuracy":  val_acc,
        "val_macro_f1":  val_macro_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(bundle, args.output_model)
    print(f"\nModel saved → {args.output_model}")

    meta = {
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "dataset":         args.input,
        "backend":         args.backend,
        "classes":         list(le.classes_),
        "n_train_segs":    int(len(X_train)),
        "n_val_segs":      int(len(X_val)),
        "n_test_segs":     int(len(X_test)),
        "n_train_songs":   n_train_songs,
        "n_val_songs":     n_val_songs,
        "n_test_songs":    n_test_songs,
        "val_accuracy":    round(val_acc, 4),
        "val_macro_f1":    round(val_macro_f1, 4),
        "test_accuracy":   round(test_acc, 4),
        "test_macro_f1":   round(test_macro_f1, 4),
        "per_class_f1_test": {
            k: round(v["f1-score"], 4)
            for k, v in test_rep.items()
            if k not in {"accuracy", "macro avg", "weighted avg"}
        },
        "confusion_matrix":  cm,
        "confusion_labels":  list(le.classes_),
    }
    with open(args.output_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata  → {args.output_meta}")


if __name__ == "__main__":
    main()
