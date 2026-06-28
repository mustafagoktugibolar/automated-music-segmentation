"""
RMS-based boundary candidate extraction.

Computes rolling RMS energy, applies first-order differentiation to find
energy change points, and returns them as CandidateBoundary objects tagged
with source="rms".
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..core.models import CandidateBoundary
from shared.logger import get_logger

logger = get_logger("features.rms")


def extract_rms_boundaries(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_dist_sec: float = 4.0,
    prominence: float = 0.12,
    smooth_sigma: float = 3.0,
) -> list[CandidateBoundary]:
    """
    Detect structural boundaries from RMS energy changes.

    Algorithm:
      1. Compute frame-level RMS and convert to dB.
      2. Gaussian-smooth the dB curve to suppress micro-variations.
      3. Differentiate and take absolute value → energy-change novelty.
      4. Normalise to [0, 1] and pick peaks.

    Parameters
    ----------
    y             : mono float32 audio waveform.
    sr            : sample rate.
    hop_length    : hop length for RMS frames.
    min_dist_sec  : minimum gap between consecutive boundaries (seconds).
    prominence    : scipy find_peaks minimum prominence on [0, 1].
    smooth_sigma  : Gaussian smoothing sigma applied to the dB curve.

    Returns
    -------
    List of CandidateBoundary with source=["rms"].
    """
    if y.size == 0:
        return []

    try:
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0].astype(np.float32)
        rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)

        if rms_db.size > 5:
            rms_db = gaussian_filter1d(rms_db, sigma=smooth_sigma)

        # First-order difference = energy change rate.
        novelty = np.abs(np.diff(rms_db, prepend=rms_db[0])).astype(np.float32)

        # Normalise to [0, 1].
        novelty -= float(novelty.min())
        max_v = float(novelty.max())
        if max_v > 0:
            novelty /= max_v

        frame_times = librosa.frames_to_time(
            np.arange(len(novelty)), sr=sr, hop_length=hop_length
        ).astype(np.float32)

        total_duration = float(frame_times[-1]) if frame_times.size > 0 else 0.0
        fps = float(sr) / hop_length
        min_dist_frames = max(1, int(min_dist_sec * fps))
        edge_margin = max(1, int(min_dist_sec * 0.5 * fps))

        peaks, _ = find_peaks(
            novelty, distance=min_dist_frames, prominence=prominence
        )
        peaks = peaks[(peaks >= edge_margin) & (peaks <= novelty.size - edge_margin)]

        candidates: list[CandidateBoundary] = []
        for p in peaks:
            t = float(frame_times[p])
            if t <= 0.0 or t >= total_duration:
                continue
            conf = round(float(np.clip(novelty[p], 0.0, 1.0)), 3)
            candidates.append(
                CandidateBoundary(
                    time_seconds=round(t, 3),
                    source=["rms"],
                    confidence=conf,
                )
            )

        logger.debug("RMS boundaries: %d candidates extracted.", len(candidates))
        return candidates

    except Exception as exc:
        logger.warning("RMS boundary extraction failed (%s).", exc)
        return []
