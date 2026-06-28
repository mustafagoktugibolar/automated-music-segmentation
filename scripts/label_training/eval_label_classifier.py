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
_app_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

import numpy as np  # noqa: F401 (used by sklearn internals)

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

def eval_clean_parquet(test_size: float) -> None:
    """Evaluate using held-out rows from the training parquet (fast, no audio)."""
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    from shared.labeling.ml import predict_semantic_labels
    from shared.labeling.heuristic import assign_semantic_labels, build_segment_descriptors
    from workers.segmenters.llm.music_segmentation_agent.salami.label_normalizer import normalize_label

    if not os.path.exists(PARQUET_PATH):
        print(f"Parquet not found: {PARQUET_PATH}. Run prepare_label_dataset.py first.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    meta_cols    = {"song_id", "dataset", "segment_idx", "start", "end", "label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X      = df[feature_cols].values.astype(np.float32)
    labels = df["label"].values
    groups = df["song_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    _, test_idx = next(gss.split(X, labels, groups=groups))

    df_test = df.iloc[test_idx]
    print(f"Clean eval: {len(df_test)} segments from {df_test['song_id'].nunique()} songs.")

    # ML predictions: reconstruct segment dicts + descriptors per song.
    y_true_ml   : list[str] = []
    y_pred_ml   : list[str] = []
    y_pred_heur : list[str] = []

    from shared.labeling.features import build_segment_label_vectors
    import joblib, os as _os

    model_path = _os.path.join(_app_root, "models", "segment_label_clf.joblib")
    if not _os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train_label_classifier.py first.")
        return

    bundle = joblib.load(model_path)
    clf = bundle["clf"]
    le  = bundle["label_encoder"]

    for song_id, grp in df_test.groupby("song_id"):
        grp = grp.sort_values("segment_idx")
        segments = [
            {"start": r["start"], "end": r["end"],
             "structural_label": chr(65), "label": chr(65)}
            for _, r in grp.iterrows()
        ]
        descriptors = grp[feature_cols].values.astype(np.float32)

        # ML
        X_seg, _ = build_segment_label_vectors(segments, descriptors=descriptors)
        preds   = le.inverse_transform(clf.predict(X_seg))
        y_pred_ml.extend(preds.tolist())

        # Heuristic (no audio, no descriptor → positional only)
        heur_segs = assign_semantic_labels(segments)
        y_pred_heur.extend(s.get("semantic_label", "Unknown") for s in heur_segs)

        # Ground truth
        y_true_ml.extend(grp["label"].tolist())

    _print_report("ML classifier (clean — GT boundaries, parquet split)", y_true_ml, y_pred_ml)
    _print_report("Heuristic     (clean — GT boundaries, parquet split)", y_true_ml, y_pred_heur)


# ── Mode (b): pipeline accuracy (segment + label) ────────────────────────────

def eval_pipeline(max_songs: int, test_size: float) -> None:
    """Run our segmenter on SALAMI audio, label with ML and heuristic, compare."""
    from backend.services.salami_parser import parse_salami_annotation
    from workers.segmenters.llm.music_segmentation_agent.salami.label_normalizer import normalize_label
    from workers.segmenters.custom.segmentation_service import _analyze_content
    from shared.labeling.ml import predict_semantic_labels
    from shared.labeling.heuristic import assign_semantic_labels

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
    parser = argparse.ArgumentParser(description="Evaluate segment-label classifier.")
    parser.add_argument("--mode",        nargs="+", default=["clean"],
                        choices=["clean", "pipeline"],
                        help="Evaluation mode(s) to run.")
    parser.add_argument("--split-parquet", action="store_true", default=True,
                        help="Use parquet-based split for clean eval (default).")
    parser.add_argument("--test-size",   type=float, default=0.15,
                        help="Held-out fraction (same as training script).")
    parser.add_argument("--max-songs",   type=int, default=20,
                        help="Max songs for pipeline mode (slow — runs segmenter).")
    args = parser.parse_args()

    if "clean" in args.mode:
        eval_clean_parquet(test_size=args.test_size)
    if "pipeline" in args.mode:
        eval_pipeline(max_songs=args.max_songs, test_size=args.test_size)


if __name__ == "__main__":
    main()
