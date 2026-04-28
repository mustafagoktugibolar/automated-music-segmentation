"""
Boundary detection evaluation service.

Computes precision, recall, and F-measure for music segmentation boundaries
against ground truth annotations, using a configurable tolerance window.

Uses mir_eval.segment.detection when available; falls back to a manual
greedy matching implementation otherwise.
"""

from __future__ import annotations

import numpy as np


def _extract_boundaries(segments: list[dict]) -> np.ndarray:
    """
    Extract unique boundary timestamps from a list of segments.

    The start of each segment is a boundary. Boundaries at t=0 are excluded
    (they represent the implicit song start, not a structural change).

    Args:
        segments: List of {start, end, label} dicts.

    Returns:
        Sorted numpy array of boundary timestamps (seconds).
    """
    boundaries = set()
    for seg in segments:
        start = float(seg.get("start", 0))
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


def compute_boundary_metrics(
    ref_segments: list[dict],
    est_segments: list[dict],
    tolerance: float = 3.0,
) -> dict:
    """
    Compute boundary detection metrics between reference and estimated segments.

    Args:
        ref_segments: Ground truth segments [{start, end, label}, ...].
        est_segments: Algorithm output segments [{start, end, label}, ...].
        tolerance: Tolerance window in seconds (default ±3s, MIR convention).

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
        "n_boundaries_ref": int(len(ref_boundaries)),
        "n_boundaries_est": int(len(est_boundaries)),
        "tolerance_seconds": tolerance,
    }
