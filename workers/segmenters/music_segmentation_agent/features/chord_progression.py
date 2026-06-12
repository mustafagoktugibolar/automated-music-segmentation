"""
Chord-progression-based boundary candidate extraction.

Uses chroma-CENS features to compute frame-to-frame cosine distance (chord
change rate). Peaks in the smoothed distance curve indicate likely chord
progressions transitions, which often correlate with section boundaries.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..core.models import CandidateBoundary
from shared.logger import get_logger

logger = get_logger("features.chord_progression")


def extract_chord_boundaries(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_dist_sec: float = 4.0,
    prominence: float = 0.10,
    smooth_sigma: float = 2.0,
) -> list[CandidateBoundary]:
    """
    Detect harmonic / chord-change boundaries from chroma-CENS features.

    Algorithm:
      1. Compute chroma-CENS (L2-normalised per frame).
      2. Frame-to-frame cosine distance = 1 − <c_n, c_{n+1}> for adjacent frames.
      3. Gaussian-smooth the distance curve.
      4. Normalise to [0, 1] and pick peaks.

    Parameters
    ----------
    y             : mono float32 audio waveform.
    sr            : sample rate.
    hop_length    : hop length for chroma feature extraction.
    min_dist_sec  : minimum gap between consecutive boundary candidates.
    prominence    : minimum peak prominence on [0, 1].
    smooth_sigma  : Gaussian smoothing sigma for the chord-change curve.

    Returns
    -------
    List of CandidateBoundary with source=["chord_proxy"].
    """
    if y.size == 0:
        return []

    try:
        # Chroma-CENS (use CQT fallback if CENS fails).
        try:
            chroma = librosa.feature.chroma_cens(
                y=y, sr=sr, hop_length=hop_length, n_chroma=12
            ).astype(np.float32)
        except Exception:
            chroma = librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=hop_length
            ).astype(np.float32)

        if chroma.shape[1] < 2:
            return []

        # L2-normalise per frame.
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        chroma = chroma / norms

        # Frame-to-frame cosine similarity → distance.
        sims = np.sum(chroma[:, 1:] * chroma[:, :-1], axis=0)
        chord_change = np.concatenate([[0.0], 1.0 - np.clip(sims, -1.0, 1.0)]).astype(
            np.float32
        )

        if chord_change.size > 5:
            chord_change = gaussian_filter1d(chord_change, sigma=smooth_sigma)

        # Normalise.
        chord_change -= float(chord_change.min())
        max_v = float(chord_change.max())
        if max_v > 0:
            chord_change /= max_v

        frame_times = librosa.frames_to_time(
            np.arange(len(chord_change)), sr=sr, hop_length=hop_length
        ).astype(np.float32)

        total_duration = float(frame_times[-1]) if frame_times.size > 0 else 0.0
        fps = float(sr) / hop_length
        min_dist_frames = max(1, int(min_dist_sec * fps))
        edge_margin = max(1, int(min_dist_sec * 0.5 * fps))

        peaks, _ = find_peaks(
            chord_change, distance=min_dist_frames, prominence=prominence
        )
        peaks = peaks[(peaks >= edge_margin) & (peaks <= chord_change.size - edge_margin)]

        candidates: list[CandidateBoundary] = []
        for p in peaks:
            t = float(frame_times[p])
            if t <= 0.0 or t >= total_duration:
                continue
            conf = round(float(np.clip(chord_change[p], 0.0, 1.0)), 3)
            candidates.append(
                CandidateBoundary(
                    time_seconds=round(t, 3),
                    source=["chord_proxy"],
                    confidence=conf,
                )
            )

        logger.debug("Chord-proxy boundaries: %d candidates.", len(candidates))
        return candidates

    except Exception as exc:
        logger.warning("Chord boundary extraction failed (%s).", exc)
        return []
