"""
Boundary detection evaluation service.

Primary metrics are boundary precision/recall/F1 computed from real segment
intervals, not artificial intervals reconstructed from boundary arrays.
"""

from __future__ import annotations

import numpy as np

from segmentation.core.segmentation.metrics import (
    compute_boundary_metrics_at_tolerance,
    compute_multi_tolerance_metrics,
    compute_segment_iou,
    greedy_boundary_match,
)
from segmentation.core.segmentation.utils import segments_to_internal_boundaries, segments_to_intervals


def _extract_boundaries(segments: list[dict], edge_margin: float = 0.5) -> np.ndarray:
    """Backward-compatible helper returning internal boundary timestamps."""
    return np.asarray(segments_to_internal_boundaries(segments, edge_margin_seconds=edge_margin), dtype=float)


def _greedy_match(ref: np.ndarray, est: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    return greedy_boundary_match(ref, est, tolerance)


def compute_boundary_metrics(
    ref_segments: list[dict],
    est_segments: list[dict],
    tolerance: float = 0.5,
) -> dict:
    """Compute boundary metrics at one tolerance using segment intervals."""
    return compute_boundary_metrics_at_tolerance(ref_segments, est_segments, tolerance)


def compute_boundary_metrics_multi(
    ref_segments: list[dict],
    est_segments: list[dict],
    tolerances: tuple[float, ...] | list[float] = (0.5, 3.0),
) -> dict:
    """Compute required 0.5s/3.0s style metrics in one call."""
    return compute_multi_tolerance_metrics(ref_segments, est_segments, tolerances)


__all__ = [
    "_extract_boundaries",
    "_greedy_match",
    "compute_segment_iou",
    "compute_boundary_metrics",
    "compute_boundary_metrics_multi",
    "segments_to_intervals",
]
