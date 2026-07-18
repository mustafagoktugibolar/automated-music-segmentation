#!/usr/bin/env python
"""
Evaluate ML label classifier versus the heuristic on held-out SALAMI songs.
============================================================================

Two evaluation modes:

(a) Clean classifier accuracy — run on GT segment boundaries.
    For each held-out song: feed GT segments → predict label → compare to GT.
    Measures how well the classifier knows the label vocabulary, free of
    segmentation noise.

(b) Pipeline accuracy — run our segmenter, then label with ML *and* heuristic.
    For each predicted segment, find the GT segment with maximum overlap and
    use its label as ground truth.  This captures real-world performance
    where boundary placement and labeling errors compound.

Usage
-----
    python scripts/eval_label_classifier.py
    python scripts/eval_label_classifier.py --mode clean pipeline --max-songs 30
    python scripts/eval_label_classifier.py --mode clean --split-parquet

The ``--split-parquet`` flag uses the same train/test split as the training
script (reads data/label_training/segments.parquet) so you evaluate on
exactly the held-out songs used during training.

Output
------
Prints a side-by-side table:
    ML vs Heuristic  |  accuracy  |  macro-F1  |  per-class F1
"""
from __future__ import annotations

import argparse
import os
import sys
# ── Path setup ────────────────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

import numpy as np  # noqa: F401 (used by sklearn internals)

from train_label_classifier import (  # noqa: E402
    DEFAULT_SEEDS,
    FEATURE_SETS,
    _MERGE_MAPS,
    apply_label_merge,
    build_features_for_set,
    load_dataset,
    make_grouped_split,
)

PARQUET_PATH    = os.path.join(_app_root, "data", "label_training", "segments.parquet")
ANNOTATIONS_DIR = os.path.join(_app_root, "data", "salami", "annotations")
AUDIO_CACHE_DIR = os.path.join(_app_root, "data", "audio_cache")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_gt_label(pred_seg: dict, gt_segs: list[dict]) -> str | None:
    """Return the GT label with the most overlap with pred_seg."""
    best_label = None
    best_ov    = 0.0
    for gt in gt_segs:
        ov = _overlap(pred_seg["start"], pred_seg["end"], gt["start"], gt["end"])
        if ov > best_ov:
            best_ov    = ov
            best_label = gt.get("label", "Unknown")
    return best_label if best_ov > 0.0 else None


def _print_report(title: str, y_true: list[str], y_pred: list[str]) -> None:
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(f"  Accuracy : {acc:.4f}  |  n_segments={len(y_true)}")
    print(classification_report(y_true, y_pred, zero_division=0))


# ── Mode (a): clean accuracy from parquet ────────────────────────────────────

def eval_clean_parquet(
    val_size: float,
    test_size: float,
    seed: int,
    merge_mode: str,
    feature_set: str,
    model_path: str,
    extra_parquets: list[str] | None = None,
) -> None:
    """Evaluate on the same grouped test split used by training."""
    import joblib
    import pandas as pd
    from segmentation.core.labeling.heuristic import assign_semantic_labels

    if not os.path.exists(PARQUET_PATH):
        print(f"Parquet not found: {PARQUET_PATH}. Run prepare_label_dataset.py first.")
        return
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train_label_classifier.py first.")
        return

    bundle = joblib.load(model_path)
    clf = bundle["clf"]
    le = bundle["label_encoder"]
    model_features = list(bundle.get("feature_names") or [])
    if not model_features:
        raise RuntimeError("bundle['feature_names'] is missing or empty.")

    bundle_merge_mode = bundle.get("merge_mode")
    if bundle_merge_mode and bundle_merge_mode != merge_mode:
        raise RuntimeError(
            f"Model was trained with merge_mode={bundle_merge_mode!r}; "
            f"eval received merge_mode={merge_mode!r}."
        )

    bundle_feature_set = bundle.get("feature_set")
    if bundle_feature_set and bundle_feature_set != feature_set:
        raise RuntimeError(
            f"Model was trained with feature_set={bundle_feature_set!r}; "
            f"eval received feature_set={feature_set!r}."
        )

    df = load_dataset(PARQUET_PATH, extra_parquets=extra_parquets or [])
    df = apply_label_merge(df, merge_mode)
    group_col = "raw_track_id" if "raw_track_id" in df.columns else "song_id"
    X, _y, groups, le_from_df, feature_cols = build_features_for_set(
        df, group_col=group_col, feature_set=feature_set
    )

    missing = [f for f in model_features if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"Parquet is missing {len(missing)} feature column(s) expected by the model. "
            f"Missing: {missing[:10]}"
        )
    if list(model_features) != list(feature_cols):
        X = df[model_features].values.astype(np.float32)
        feature_cols = model_features

    if list(le.classes_) != list(le_from_df.classes_):
        raise RuntimeError(
            "Model classes do not match eval labels after merge-mode transform: "
            f"model={list(le.classes_)} eval={list(le_from_df.classes_)}"
        )

    _, _, test_idx = make_grouped_split(
        groups, val_size=val_size, test_size=test_size, random_state=seed
    )
    df_test = df.iloc[test_idx]
    n_groups = df_test[group_col].nunique()
    print(
        f"Clean eval: {len(df_test)} segments from {n_groups} {group_col}s  "
        f"(seed={seed}, val_size={val_size}, test_size={test_size}, "
        f"merge_mode={merge_mode}, feature_set={feature_set}, features={len(feature_cols)})."
    )

    X_test = pd.DataFrame(X[test_idx], columns=feature_cols)
    preds = le.inverse_transform(clf.predict(X_test))
    y_true_ml = df_test["label"].tolist()
    y_pred_ml = preds.tolist()

    y_pred_heur: list[str] = []
    for _sid, grp in df_test.groupby("song_id"):
        grp = grp.sort_values("segment_idx")
        segments = [{"start": r["start"], "end": r["end"]} for _, r in grp.iterrows()]
        heur_segs = assign_semantic_labels(segments)
        y_pred_heur.extend(s.get("semantic_label", "Unknown") for s in heur_segs)

    _print_report("ML classifier (clean - GT boundaries, grouped test split)", y_true_ml, y_pred_ml)
    _print_report("Heuristic     (clean - GT boundaries, grouped test split)", y_true_ml, y_pred_heur)

def eval_pipeline(max_songs: int, test_size: float) -> None:
    """Run our segmenter on SALAMI audio, label with ML and heuristic, compare."""
    from segmentation.core.segmentation.salami_parser import parse_salami_annotation
    from segmentation.core.labeling.label_normalizer import normalize_label
    from segmentation.application.segmentation.custom_engine import _analyze_content
    from segmentation.core.labeling.ml import predict_semantic_labels
    from segmentation.core.labeling.heuristic import assign_semantic_labels

    if not os.path.isdir(ANNOTATIONS_DIR):
        print(f"Annotations dir not found: {ANNOTATIONS_DIR}")
        return

    song_ids = sorted(d.name for d in os.scandir(ANNOTATIONS_DIR) if d.is_dir())
    # Use the last test_size fraction as held-out (mirrors group-split logic).
    cut = max(1, int(len(song_ids) * (1 - test_size)))
    held_out = song_ids[cut:]
    if max_songs > 0:
        held_out = held_out[:max_songs]

    print(f"Pipeline eval: {len(held_out)} held-out songs …")

    y_true      : list[str] = []
    y_pred_ml   : list[str] = []
    y_pred_heur : list[str] = []

    for song_id in held_out:
        audio_path = os.path.join(AUDIO_CACHE_DIR, f"{song_id}.mp3")
        if not os.path.exists(audio_path):
            continue

        gt_segs = parse_salami_annotation(song_id, annotator=1)
        if not gt_segs:
            continue

        # Normalize GT labels.
        for seg in gt_segs:
            seg["label"] = normalize_label(seg["label"])

        try:
            with open(audio_path, "rb") as fh:
                audio_bytes = fh.read()
        except Exception:
            continue

        # Run segmenter (heuristic mode — get structural labels).
        try:
            result_h = _analyze_content(
                audio_bytes, "song.mp3", "audio/mpeg",
                params={"labeling_method": "heuristic", "return_diagnostics": False},
            )
            result_m = _analyze_content(
                audio_bytes, "song.mp3", "audio/mpeg",
                params={"labeling_method": "ml", "return_diagnostics": False},
            )
        except Exception as exc:
            print(f"  [skip] {song_id}: segmentation failed — {exc}")
            continue

        segs_h = result_h.get("segments", [])
        segs_m = result_m.get("segments", [])

        for pred_seg in segs_h:
            gt_label = _best_gt_label(pred_seg, gt_segs)
            if gt_label is None:
                continue
            y_true.append(gt_label)
            y_pred_heur.append(pred_seg.get("semantic_label", "Unknown"))

            # Find the corresponding ML-labeled segment (same index).
            idx = segs_h.index(pred_seg)
            ml_label = segs_m[idx].get("semantic_label", "Unknown") if idx < len(segs_m) else "Unknown"
            y_pred_ml.append(ml_label)

    if not y_true:
        print("No pipeline evaluation data collected.")
        return

    _print_report("ML classifier (pipeline — predicted boundaries)", y_true, y_pred_ml)
    _print_report("Heuristic     (pipeline — predicted boundaries)", y_true, y_pred_heur)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Evaluate segment-label classifier.")
    parser.add_argument("--mode", nargs="+", default=["clean"],
                        choices=["clean", "pipeline"],
                        help="Evaluation mode(s) to run.")
    parser.add_argument("--split-parquet", action="store_true", default=True,
                        help="Use parquet-based split for clean eval (default).")
    parser.add_argument("--val-size", type=float, default=0.20,
                        help="Validation fraction used before the grouped test split.")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Held-out test fraction (same as training script).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--merge-mode", default="none", choices=list(_MERGE_MAPS))
    parser.add_argument("--feature-set", default="full", choices=sorted(FEATURE_SETS))
    parser.add_argument("--model-path", default=os.path.join(_app_root, "models", "segment_label_clf.joblib"))
    parser.add_argument("--extra-parquet", nargs="*", default=[])
    parser.add_argument("--max-songs", type=int, default=20,
                        help="Max songs for pipeline mode (slow - runs segmenter).")
    args = parser.parse_args()

    if "clean" in args.mode:
        eval_clean_parquet(
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            merge_mode=args.merge_mode,
            feature_set=args.feature_set,
            model_path=args.model_path,
            extra_parquets=args.extra_parquet or [],
        )
    if "pipeline" in args.mode:
        eval_pipeline(max_songs=args.max_songs, test_size=args.test_size)


if __name__ == "__main__":
    main()
