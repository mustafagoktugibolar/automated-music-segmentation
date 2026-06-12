"""
Boundary detection evaluation service.

Computes precision, recall, and F-measure for music segmentation boundaries
against ground truth annotations, using a configurable tolerance window.

Uses mir_eval.segment.detection when available; falls back to a manual
greedy matching implementation otherwise.
"""

from __future__ import annotations

import numpy as np


def _extract_boundaries(segments: list[dict], edge_margin: float = 2.0) -> np.ndarray:
    """
    Extract unique boundary timestamps from a list of segments.

    The start of each non-initial segment is a boundary. The first annotated
    start and near-edge starts are excluded because they represent track
    alignment/silence markers rather than internal structural changes.

    Args:
        segments: List of {start, end, label} dicts.

    Returns:
        Sorted numpy array of boundary timestamps (seconds).
    """
    ordered = sorted(
        segments,
        key=lambda seg: float(seg.get("start", 0) or 0),
    )
    if len(ordered) <= 1:
        return np.array([])

    track_end = max(float(seg.get("end", 0) or 0) for seg in ordered)
    boundaries = set()
    for idx, seg in enumerate(ordered):
        if idx == 0:
            continue
        start = float(seg.get("start", 0))
        if start <= edge_margin:
            continue
        if track_end > 0 and start >= track_end - edge_margin:
            continue
        if start > 0.0:
            boundaries.add(start)
    return np.array(sorted(boundaries))


def _greedy_match(ref: np.ndarray, est: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """
    Greedy bipartite matching within ±tolerance seconds.

    Returns:
        (n_true_positives, n_ref, n_est)
    """
    matched_ref = set()
    matched_est = set()
    tp = 0

    for i, r in enumerate(ref):
        for j, e in enumerate(est):
            if j in matched_est:
                continue
            if abs(r - e) <= tolerance:
                tp += 1
                matched_ref.add(i)
                matched_est.add(j)
                break

    return tp, len(ref), len(est)


def compute_segment_iou(
    ref_segments: list[dict],
    est_segments: list[dict],
) -> float:
    """
    Mean segment IoU (label-agnostic).

    For each reference segment, find the estimated segment with the greatest
    temporal overlap and compute IoU = overlap / union.  The mean over all
    reference segments is returned.

    This is the temporal analogue of mean-IoU used in image segmentation.
    """
    if not ref_segments or not est_segments:
        return 0.0

    ious: list[float] = []
    for ref in ref_segments:
        rs = float(ref.get("start", 0) or 0)
        re = float(ref.get("end", 0) or 0)
        ref_dur = re - rs
        if ref_dur <= 0:
            continue
        best_iou = 0.0
        for est in est_segments:
            es = float(est.get("start", 0) or 0)
            ee = float(est.get("end", 0) or 0)
            overlap = max(0.0, min(re, ee) - max(rs, es))
            union = ref_dur + (ee - es) - overlap
            if union > 0:
                best_iou = max(best_iou, overlap / union)
        ious.append(best_iou)

    return round(float(np.mean(ious)), 4) if ious else 0.0


def compute_boundary_metrics(
    ref_segments: list[dict],
    est_segments: list[dict],
    tolerance: float = 0.5,
) -> dict:
    """
    Compute boundary detection metrics between reference and estimated segments.

    Args:
        ref_segments: Ground truth segments [{start, end, label}, ...].
        est_segments: Algorithm output segments [{start, end, label}, ...].
        tolerance: Tolerance window in seconds (default ±0.5 s — MIREX standard,
                   FMP Section 4.5.4, Eq. 4.57).  The old default of 3.0 s was
                   too lenient and inflated scores on SALAMI evaluations.

    Returns:
        {
            precision: float,
            recall: float,
            f_measure: float,
            n_boundaries_ref: int,
            n_boundaries_est: int,
            tolerance_seconds: float,
        }
    """
    ref_boundaries = _extract_boundaries(ref_segments)
    est_boundaries = _extract_boundaries(est_segments)

    if len(ref_boundaries) == 0 and len(est_boundaries) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f_measure": 1.0,
            "n_boundaries_ref": 0,
            "n_boundaries_est": 0,
            "tolerance_seconds": tolerance,
        }

    try:
        import mir_eval

        # mir_eval expects intervals [[start, end], ...] in addition to boundaries.
        # For detection we only need boundary arrays.
        ref_intervals = np.column_stack([ref_boundaries[:-1], ref_boundaries[1:]]) if len(ref_boundaries) > 1 else np.array([[ref_boundaries[0], ref_boundaries[0] + 1.0]])
        est_intervals = np.column_stack([est_boundaries[:-1], est_boundaries[1:]]) if len(est_boundaries) > 1 else np.array([[est_boundaries[0], est_boundaries[0] + 1.0]])

        precision, recall, f_measure = mir_eval.segment.detection(
            ref_intervals,
            est_intervals,
            window=tolerance,
            beta=1.0,
        )
    except Exception:
        # Fallback: manual greedy matching
        tp, n_ref, n_est = _greedy_match(ref_boundaries, est_boundaries, tolerance)

        precision = tp / n_est if n_est > 0 else 0.0
        recall = tp / n_ref if n_ref > 0 else 0.0
        denom = precision + recall
        f_measure = 2 * precision * recall / denom if denom > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f_measure": round(float(f_measure), 4),
        "segment_iou": compute_segment_iou(ref_segments, est_segments),
        "n_boundaries_ref": int(len(ref_boundaries)),
        "n_boundaries_est": int(len(est_boundaries)),
        "tolerance_seconds": tolerance,
    }
