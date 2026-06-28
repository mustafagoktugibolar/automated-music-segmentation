"""
Candidate boundary filtering.

Applies duration-gap, confidence-threshold, and count-cap filters to a list of
CandidateBoundary objects. The output is sorted by time and ready for LLM decision.
"""

from __future__ import annotations

from .models import CandidateBoundary

from shared.logger import get_logger

logger = get_logger("filtering")


def filter_candidates(
    candidates: list[CandidateBoundary],
    min_duration_sec: float = 8.0,
    max_candidates: int = 20,
    min_confidence: float = 0.25,
) -> list[CandidateBoundary]:
    """
    Filter and cap a list of candidate boundaries.

    Filtering steps (applied in order):
      1. Remove candidates below *min_confidence*.
      2. Remove candidates that would create a segment shorter than
         *min_duration_sec* relative to their neighbours.
      3. If more than *max_candidates* remain, keep the top-N by confidence.

    Parameters
    ----------
    candidates       : Raw candidate list (any order).
    min_duration_sec : Minimum gap between consecutive boundaries (seconds).
    max_candidates   : Maximum number of boundaries to return.
    min_confidence   : Drop candidates below this confidence threshold.

    Returns
    -------
    Filtered list of CandidateBoundary, sorted by time_seconds.
    """
    if not candidates:
        return []

    # Step 1: confidence filter
    filtered = [c for c in candidates if c.confidence >= min_confidence]
    logger.debug(
        "Confidence filter: %d → %d (threshold=%.2f)",
        len(candidates),
        len(filtered),
        min_confidence,
    )

    if not filtered:
        return []

    # Sort by time for gap analysis.
    filtered.sort(key=lambda c: c.time_seconds)

    # Step 2: enforce minimum duration gap.
    passed: list[CandidateBoundary] = []
    for cand in filtered:
        if not passed:
            passed.append(cand)
            continue
        gap = cand.time_seconds - passed[-1].time_seconds
        if gap >= min_duration_sec:
            passed.append(cand)
        else:
            # Keep beat-grid anchors when a broad novelty peak lands a few
            # seconds before/after the musical downbeat. This is critical for
            # strict ±0.5s boundary evaluation.
            prev_is_grid = "beat_grid" in passed[-1].source
            cand_is_grid = "beat_grid" in cand.source
            if cand_is_grid and not prev_is_grid:
                passed[-1] = cand
            elif cand_is_grid == prev_is_grid and cand.confidence > passed[-1].confidence:
                passed[-1] = cand

    logger.debug(
        "Duration-gap filter (min=%.1fs): %d → %d", min_duration_sec, len(filtered), len(passed)
    )

    # Step 3: cap at max_candidates (keep highest confidence).
    if len(passed) > max_candidates:
        sorted_by_conf = sorted(passed, key=lambda c: c.confidence, reverse=True)
        kept = sorted_by_conf[:max_candidates]
        kept.sort(key=lambda c: c.time_seconds)
        logger.debug("Capped to %d candidates.", max_candidates)
        return kept

    return passed
