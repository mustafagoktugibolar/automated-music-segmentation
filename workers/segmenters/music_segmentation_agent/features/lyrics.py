"""
Lyrics-based boundary candidate extraction.

Takes a list of time-stamped lyric events and converts each lyric line into a
candidate boundary. Confidence is based on line length and local density of
lyric events (denser regions → section starts → higher confidence).

Handles empty input gracefully.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.models import CandidateBoundary
from shared.logger import get_logger

logger = get_logger("features.lyrics")


def extract_lyric_boundaries(
    timed_lyrics: list[dict] | None,
    total_duration: float,
    min_dist_sec: float = 4.0,
    base_confidence: float = 0.55,
) -> list[CandidateBoundary]:
    """
    Convert time-stamped lyric events into boundary candidates.

    Parameters
    ----------
    timed_lyrics    : List of dicts with keys:
                        "time_seconds" (float) and "text" (str).
                      None or empty list → returns [].
    total_duration  : Track duration in seconds (for edge filtering).
    min_dist_sec    : Minimum gap between consecutive lyric boundaries.
    base_confidence : Base confidence score for lyric events.

    Returns
    -------
    List of CandidateBoundary with source=["lyrics"], sorted by time.
    """
    if not timed_lyrics:
        return []

    # Validate and extract (time, text) pairs.
    events: list[tuple[float, str]] = []
    for item in timed_lyrics:
        try:
            t = float(item["time_seconds"])
            text = str(item.get("text", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if text and 0.0 < t < total_duration:
            events.append((t, text))

    if not events:
        return []

    events.sort(key=lambda x: x[0])
    times = np.array([e[0] for e in events], dtype=np.float64)

    # Local density: for each event, count how many others are within ±10s.
    density_window = 10.0
    densities = np.array(
        [np.sum(np.abs(times - t) <= density_window) for t in times],
        dtype=np.float64,
    )
    max_density = float(densities.max()) if densities.size > 0 else 1.0
    if max_density == 0:
        max_density = 1.0

    candidates: list[CandidateBoundary] = []
    last_t = -min_dist_sec

    for (t, text), density in zip(events, densities):
        if t - last_t < min_dist_sec:
            continue

        # Confidence: base + word-count bonus + density penalty (dense = less noteworthy).
        word_count = len(text.split())
        length_bonus = min(0.15, 0.015 * word_count)
        density_factor = 1.0 - 0.3 * (float(density) / max_density)
        conf = min(1.0, max(0.1, base_confidence * density_factor + length_bonus))

        candidates.append(
            CandidateBoundary(
                time_seconds=round(t, 3),
                source=["lyrics"],
                confidence=round(conf, 3),
            )
        )
        last_t = t

    logger.debug("Lyric boundaries: %d candidates from %d events.", len(candidates), len(events))
    return candidates
