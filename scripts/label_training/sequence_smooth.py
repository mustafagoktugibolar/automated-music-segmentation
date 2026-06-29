#!/usr/bin/env python
"""
Viterbi-based sequence smoothing for the segment-label classifier.

Learns a label transition matrix from training sequences, then applies
Viterbi decoding using GBDT class probabilities as emission probabilities.
Never uses ground-truth test labels for decoding — only for evaluation.

Usage
-----
    python scripts/label_training/sequence_smooth.py [--merge-mode other]
    python scripts/label_training/sequence_smooth.py --all-modes
    python scripts/label_training/sequence_smooth.py --merge-mode none --seed 42

Reads:
    data/label_training/segments.parquet
    models/segment_label_clf.joblib

Outputs:
    models/evaluation/sequence_smooth_{merge_mode}.json
    models/evaluation/clean_baseline_comparison.csv  (with --all-modes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_here     = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, "..", ".."))
for _p in [_app_root, _here]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

PARQUET_PATH = os.path.join(_app_root, "data", "label_training", "segments.parquet")
MODEL_JOBLIB = os.path.join(_app_root, "models", "segment_label_clf.joblib")
EVAL_DIR     = os.path.join(_app_root, "models", "evaluation")

from train_label_classifier import (  # noqa: E402
    load_dataset, apply_label_merge, build_features_for_set,
    make_grouped_split, _MERGE_MAPS, DEFAULT_SEEDS, META_COLS,
)


# ── Viterbi ───────────────────────────────────────────────────────────────────

def learn_transition_matrix(
    label_seqs: list[list[int]], n_classes: int, alpha: float = 1.0,
) -> np.ndarray:
    """Return (K, K) log transition matrix with Laplace smoothing alpha."""
    trans = np.full((n_classes, n_classes), alpha, dtype=np.float64)
    for seq in label_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a, b] += 1.0
    trans /= trans.sum(axis=1, keepdims=True)
    return np.log(trans + 1e-12)


def viterbi_decode(log_emission: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """Standard Viterbi; uniform prior at t=0.

    Parameters
    ----------
    log_emission : (T, K) log probabilities from classifier
    log_transition : (K, K) log P(to | from)

    Returns
    -------
    states : (T,) int array of decoded labels
    """
    T, K = log_emission.shape
    dp      = np.full((T, K), -np.inf, dtype=np.float64)
    backptr = np.zeros((T, K), dtype=np.int32)

    dp[0] = log_emission[0]  # uniform prior

    for t in range(1, T):
        # scores[i, j] = dp[t-1, i] + log_trans[i, j]
        scores = dp[t - 1, :, None] + log_transition  # (K, K)
        backptr[t] = np.argmax(scores, axis=0)
        dp[t] = scores[backptr[t], np.arange(K)] + log_emission[t]

    states = np.zeros(T, dtype=np.int32)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states


# ── Evaluation ────────────────────────────────────────────────────────────────

def _song_sequences(df_sub, y_sub, test_idx_in_sub):
    """Yield (song_id, sorted_positions_in_sub) for each unique song."""
    song_ids = df_sub["song_id"].values
    for sid in np.unique(song_ids):
        mask = song_ids == sid
        pos  = np.where(mask)[0]
        if "segment_idx" in df_sub.columns:
            order = df_sub.iloc[pos]["segment_idx"].values.argsort()
            pos   = pos[order]
        yield sid, pos


def evaluate_smoothed(
    clf,
    le,
    df: "pd.DataFrame",
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_idx: np.ndarray,
    log_trans: np.ndarray,
    feature_cols: list[str],
) -> dict:
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    X_test = X[test_idx]
    y_test = y[test_idx]
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Pass named DataFrame to suppress LightGBM feature-name warning
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    proba = clf.predict_proba(X_test_df)
    log_emission = np.log(proba + 1e-12)

    y_raw    = clf.predict(X_test_df)
    y_smooth = np.empty_like(y_raw)

    for _sid, pos in _song_sequences(df_test, y_test, test_idx):
        if len(pos) == 1:
            y_smooth[pos] = y_raw[pos]
            continue
        states = viterbi_decode(log_emission[pos], log_trans)
        y_smooth[pos] = states

    def _report(y_true, y_pred, tag):
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cr  = classification_report(
            y_true, y_pred,
            labels=list(range(len(le.classes_))),
            target_names=list(le.classes_),
            zero_division=0,
        )
        print(f"\n── {tag} ──")
        print(f"  Accuracy : {acc:.4f}  Macro-F1 : {f1:.4f}")
        print(cr)
        return {"accuracy": acc, "macro_f1": f1,
                "per_class": {
                    c: round(f1_score(y_true, y_pred, labels=[i],
                                      average="macro", zero_division=0), 4)
                    for i, c in enumerate(le.classes_)
                }}

    raw_metrics    = _report(y_test, y_raw,    "Without smoothing")
    smooth_metrics = _report(y_test, y_smooth, "With Viterbi smoothing")

    return {"without_smoothing": raw_metrics, "with_smoothing": smooth_metrics}


# ── Single mode run ───────────────────────────────────────────────────────────

def run_mode(
    merge_mode: str,
    seed: int,
    val_size: float,
    test_size: float,
    extra_parquets: list[str] | None = None,
    model_path: str | None = None,
) -> dict:
    import joblib
    import pandas as pd

    chosen_model_path = model_path
    if chosen_model_path is None:
        mode_model_path = os.path.join(_app_root, "models", f"segment_label_clf_{merge_mode}.joblib")
        chosen_model_path = mode_model_path if os.path.exists(mode_model_path) else MODEL_JOBLIB
    print(f"Loading model: {chosen_model_path}")
    bundle = joblib.load(chosen_model_path)
    clf    = bundle["clf"]
    feature_set = bundle.get("feature_set", "full")

    df = load_dataset(PARQUET_PATH, extra_parquets=extra_parquets or [])
    df = apply_label_merge(df, merge_mode)

    group_col = "raw_track_id" if "raw_track_id" in df.columns else "song_id"
    X, y, groups, le, feature_cols = build_features_for_set(
        df, group_col=group_col, feature_set=feature_set
    )

    model_features = bundle.get("feature_names") or []
    if model_features and list(model_features) != list(feature_cols):
        missing = [c for c in model_features if c not in df.columns]
        if missing:
            raise RuntimeError(f"Model feature columns missing from data: {missing[:5]}")
        feature_cols = list(model_features)
        X = df[feature_cols].values.astype(np.float32)

    train_idx, val_idx, test_idx = make_grouped_split(
        groups, val_size=val_size, test_size=test_size, random_state=seed,
    )

    # ── Learn transition from training sequences ───────────────────────────────
    df_train    = df.iloc[train_idx].reset_index(drop=True)
    y_train     = y[train_idx]
    n_classes   = len(le.classes_)
    label_seqs  = []
    for _sid, pos in _song_sequences(df_train, y_train, train_idx):
        seq = y_train[pos].tolist()
        if len(seq) >= 2:
            label_seqs.append(seq)

    log_trans = learn_transition_matrix(label_seqs, n_classes)

    print(f"\nTransition matrix  (merge_mode={merge_mode}, seed={seed})")
    print("  " + "".join(f"{c:>16}" for c in le.classes_))
    for i, c in enumerate(le.classes_):
        row = np.exp(log_trans[i])
        print(f"  {c:>14}  " + "".join(f"{v:>16.3f}" for v in row))

    metrics = evaluate_smoothed(clf, le, df, X, y, groups, test_idx, log_trans, feature_cols)
    metrics["merge_mode"] = merge_mode
    metrics["seed"]       = seed
    metrics["model_path"] = chosen_model_path
    metrics["feature_set"] = feature_set
    metrics["extra_parquets"] = extra_parquets or []

    experiment_name = bundle.get("experiment_name")
    if experiment_name:
        eval_dir = os.path.join(EVAL_DIR, "experiments", str(experiment_name))
    else:
        eval_dir = EVAL_DIR
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, f"sequence_smooth_{merge_mode}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Update bundle with transition matrix ───────────────────────────────────
    bundle["transition_matrix"]  = np.exp(log_trans).tolist()
    bundle["transition_classes"] = list(le.classes_)
    joblib.dump(bundle, chosen_model_path)
    print(f"Bundle updated with transition_matrix -> {chosen_model_path}")

    return metrics


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_csv() -> None:
    import pandas as pd

    rows = []
    for mode in _MERGE_MAPS:
        meta_path   = os.path.join(EVAL_DIR, f"meta_{mode}.json")
        smooth_path = os.path.join(EVAL_DIR, f"sequence_smooth_{mode}.json")

        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        row = {
            "merge_mode":     mode,
            "n_classes":      len(meta.get("classes", [])),
            "n_features":     meta.get("feature_count", "?"),
            "train_macro_f1": meta.get("train_macro_f1"),
            "val_macro_f1":   meta.get("val_macro_f1"),
            "test_macro_f1":  meta.get("test_macro_f1"),
            "val_test_gap":   round(
                (meta.get("val_macro_f1", 0) or 0) -
                (meta.get("test_macro_f1", 0) or 0), 4
            ),
        }

        if meta.get("multi_seed"):
            ms = meta["multi_seed"]
            row["test_f1_mean"] = ms.get("test_macro_f1_mean")
            row["test_f1_std"]  = ms.get("test_macro_f1_std")
        else:
            row["test_f1_mean"] = row["test_macro_f1"]
            row["test_f1_std"]  = None

        for cls in meta.get("confusion_labels", []):
            pf = (meta.get("per_class_f1_test") or {}).get(cls)
            row[f"test_f1_{cls.lower()}"] = pf

        if os.path.exists(smooth_path):
            with open(smooth_path) as f:
                sm = json.load(f)
            row["smooth_test_macro_f1"] = sm.get("with_smoothing", {}).get("macro_f1")

        rows.append(row)

    if not rows:
        print("No meta files found. Run train_label_classifier.py for each merge mode first.")
        return

    df = pd.DataFrame(rows)
    out = os.path.join(EVAL_DIR, "clean_baseline_comparison.csv")
    df.to_csv(out, index=False)
    print(f"\nComparison table saved → {out}")
    print(df.to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Viterbi sequence smoothing for label classifier.")
    parser.add_argument("--merge-mode", default="other", choices=list(_MERGE_MAPS))
    parser.add_argument("--all-modes",  action="store_true",
                        help="Run for all merge modes and build comparison CSV.")
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--val-size",   type=float, default=0.20)
    parser.add_argument("--test-size",  type=float, default=0.20)
    parser.add_argument("--extra-parquet", nargs="*", default=[],
                        help="Additional parquet files to merge for smoothing evaluation.")
    parser.add_argument("--model-path", default="",
                        help="Explicit model bundle path, useful for experiments.")
    args = parser.parse_args()

    if args.all_modes:
        for mode in _MERGE_MAPS:
            meta_path = os.path.join(EVAL_DIR, f"meta_{mode}.json")
            if not os.path.exists(meta_path):
                print(f"\n[skip] No trained model found for merge_mode={mode}. "
                      f"Run train_label_classifier.py --merge-mode {mode} first.")
                continue
            print(f"\n{'='*60}")
            print(f"  merge_mode = {mode}")
            print(f"{'='*60}")
            run_mode(mode, args.seed, args.val_size, args.test_size, args.extra_parquet or [], None)
        build_comparison_csv()
    else:
        run_mode(
            args.merge_mode,
            args.seed,
            args.val_size,
            args.test_size,
            args.extra_parquet or [],
            args.model_path or None,
        )


if __name__ == "__main__":
    main()
