from __future__ import annotations

import numpy as np

from shared.segmentation.utils import segments_to_internal_boundaries, segments_to_intervals


def greedy_boundary_match(ref_boundaries: np.ndarray, est_boundaries: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    matched_est: set[int] = set()
    tp = 0
    for r in ref_boundaries:
        best_j: int | None = None
        best_dist = float("inf")
        for j, e in enumerate(est_boundaries):
            if j in matched_est:
                continue
            dist = abs(float(r) - float(e))
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            matched_est.add(best_j)
            tp += 1
    return tp, len(ref_boundaries), len(est_boundaries)


def compute_segment_iou(ref_segments: list[dict], est_segments: list[dict]) -> float:
    if not ref_segments or not est_segments:
        return 0.0
    ref_intervals = segments_to_intervals(ref_segments)
    est_intervals = segments_to_intervals(est_segments)
    if ref_intervals.size == 0 or est_intervals.size == 0:
        return 0.0

    ious: list[float] = []
    for rs, re in ref_intervals:
        ref_dur = re - rs
        if ref_dur <= 0:
            continue
        best_iou = 0.0
        for es, ee in est_intervals:
            overlap = max(0.0, min(float(re), float(ee)) - max(float(rs), float(es)))
            union = ref_dur + (ee - es) - overlap
            if union > 0:
                best_iou = max(best_iou, overlap / union)
        ious.append(best_iou)
    return round(float(np.mean(ious)), 4) if ious else 0.0


def compute_boundary_metrics_at_tolerance(
    ref_segments: list[dict],
    est_segments: list[dict],
    tolerance: float,
    edge_margin_seconds: float = 0.5,
) -> dict:
    ref_intervals = segments_to_intervals(ref_segments)
    est_intervals = segments_to_intervals(est_segments)
    ref_boundaries = np.asarray(segments_to_internal_boundaries(ref_segments, edge_margin_seconds), dtype=float)
    est_boundaries = np.asarray(segments_to_internal_boundaries(est_segments, edge_margin_seconds), dtype=float)

    if len(ref_boundaries) == 0 and len(est_boundaries) == 0:
        precision = recall = f_measure = 1.0
    else:
        try:
            import mir_eval

            precision, recall, f_measure = mir_eval.segment.detection(
                ref_intervals,
                est_intervals,
                window=tolerance,
                beta=1.0,
                trim=True,
            )
        except Exception:
            tp, n_ref, n_est = greedy_boundary_match(ref_boundaries, est_boundaries, tolerance)
            precision = tp / n_est if n_est > 0 else 0.0
            recall = tp / n_ref if n_ref > 0 else 0.0
            denom = precision + recall
            f_measure = 2.0 * precision * recall / denom if denom > 0 else 0.0

    n_ref = int(len(ref_boundaries))
    n_est = int(len(est_boundaries))
    ratio = float(n_est / n_ref) if n_ref > 0 else (0.0 if n_est == 0 else float("inf"))
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f_measure": round(float(f_measure), 4),
        "segment_iou": compute_segment_iou(ref_segments, est_segments),
        "n_boundaries_ref": n_ref,
        "n_boundaries_est": n_est,
        "n_reference_segments": int(len(ref_intervals)),
        "n_estimated_segments": int(len(est_intervals)),
        "n_reference_internal_boundaries": n_ref,
        "n_estimated_internal_boundaries": n_est,
        "over_under_segmentation_ratio": round(ratio, 4) if np.isfinite(ratio) else "inf",
        "tolerance_seconds": tolerance,
    }


def compute_multi_tolerance_metrics(
    ref_segments: list[dict],
    est_segments: list[dict],
    tolerances: tuple[float, ...] | list[float] = (0.5, 3.0),
) -> dict:
    metrics: dict = {}
    by_tol: dict[str, dict] = {}
    for tolerance in tolerances:
        single = compute_boundary_metrics_at_tolerance(ref_segments, est_segments, float(tolerance))
        suffix = str(tolerance).replace(".", "_")
        metrics[f"precision_{suffix}"] = single["precision"]
        metrics[f"recall_{suffix}"] = single["recall"]
        metrics[f"f1_{suffix}"] = single["f_measure"]
        by_tol[str(tolerance)] = single

    ref_intervals = segments_to_intervals(ref_segments)
    est_intervals = segments_to_intervals(est_segments)
    ref_boundaries = segments_to_internal_boundaries(ref_segments)
    est_boundaries = segments_to_internal_boundaries(est_segments)
    n_ref = len(ref_boundaries)
    n_est = len(est_boundaries)
    ratio = float(n_est / n_ref) if n_ref > 0 else (0.0 if n_est == 0 else float("inf"))
    metrics.update(
        {
            "n_reference_segments": int(len(ref_intervals)),
            "n_estimated_segments": int(len(est_intervals)),
            "n_reference_internal_boundaries": int(n_ref),
            "n_estimated_internal_boundaries": int(n_est),
            "over_under_segmentation_ratio": round(ratio, 4) if np.isfinite(ratio) else "inf",
            "segment_iou": compute_segment_iou(ref_segments, est_segments),
            "by_tolerance": by_tol,
        }
    )
    return metrics
