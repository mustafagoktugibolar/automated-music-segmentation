"""
Beat-phrase boundary candidate extraction.

Places boundary candidates every N beats (configurable), aligned to the beat
grid with the strongest onset energy at that position. At 120 BPM, 16 beats ≈
8 seconds and 32 beats ≈ 16 seconds — both within the typical section length.

Mirrors the logic in multi_feature_fusion.beat_phrase_boundary_candidates but
produces CandidateBoundary objects with the new Pydantic schema.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..core.models import CandidateBoundary
from shared.logger import get_logger

logger = get_logger("features.beat_detection")


def extract_beat_boundaries(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    beats_per_boundary: int = 16,
    phrase_steps: tuple[int, ...] = (16, 24, 32, 48),
) -> list[CandidateBoundary]:
    """
    Generate phrase-level boundary candidates on a beat grid.

    For each phrase step in *phrase_steps*, the algorithm finds the global phase
    offset (first beat index) that maximises the mean onset strength at the
    candidate positions, then enumerates candidates at that offset.

    Parameters
    ----------
    y                  : mono float32 audio waveform.
    sr                 : sample rate.
    hop_length         : hop length for beat tracking and onset strength.
    beats_per_boundary : primary step size (kept for backward-compat; actual
                         steps come from *phrase_steps*).
    phrase_steps       : tuple of beat-count step sizes to try.

    Returns
    -------
    List of CandidateBoundary with source=["beat_phrase"].
    """
    if y.size == 0:
        return []

    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=hop_length
        ).astype(np.float32)

        if beat_times.size < 8:
            logger.debug("Too few beats (%d) for phrase boundaries.", beat_times.size)
            return []

        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length
        ).astype(np.float32)
        if onset_env.size > 5:
            onset_env = gaussian_filter1d(onset_env, sigma=5.0)

        onset_frame_times = librosa.frames_to_time(
            np.arange(len(onset_env)), sr=sr, hop_length=hop_length
        ).astype(np.float32)

        onset_at_beats = np.interp(
            beat_times, onset_frame_times, onset_env, left=0.0, right=0.0
        ).astype(np.float32)
        max_onset = float(np.max(onset_at_beats)) or 1.0

        total_dur = float(beat_times[-1])
        min_seg_dur = 6.0  # minimum segment duration in seconds

        out: list[CandidateBoundary] = []
        seen: set[float] = set()

        for step in phrase_steps:
            if beat_times.size < step:
                continue

            # Find the phase offset with the highest mean onset strength.
            best_offset = 0
            best_score = -1.0
            for offset in range(min(step, beat_times.size)):
                idx = np.arange(offset, beat_times.size, step)
                if idx.size < 2:
                    continue
                score = float(np.mean(onset_at_beats[idx]))
                if score > best_score:
                    best_score = score
                    best_offset = offset

            for idx in np.arange(best_offset, beat_times.size, step):
                t = float(beat_times[idx])
                if t <= min_seg_dur * 0.5 or t >= total_dur - min_seg_dur * 0.5:
                    continue
                t_key = round(t, 1)
                if t_key in seen:
                    continue
                seen.add(t_key)
                conf = 0.45 + 0.45 * min(1.0, float(onset_at_beats[idx]) / max_onset)
                out.append(
                    CandidateBoundary(
                        time_seconds=round(t, 3),
                        source=["beat_phrase"],
                        confidence=round(conf, 3),
                    )
                )

        # Dedup within 0.6s window (same logic as multi_feature_fusion).
        out.sort(key=lambda c: c.time_seconds)
        deduped: list[CandidateBoundary] = []
        for cand in out:
            if not deduped or (cand.time_seconds - deduped[-1].time_seconds) > 0.6:
                deduped.append(cand)
            elif cand.confidence > deduped[-1].confidence:
                deduped[-1] = cand

        logger.debug("Beat-phrase boundaries: %d candidates.", len(deduped))
        return deduped

    except Exception as exc:
        logger.warning("Beat boundary extraction failed (%s).", exc)
        return []
