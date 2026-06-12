"""
Tempo and beat detection.

Wraps librosa.beat.beat_track to return (estimated BPM, beat_times_seconds).
Beat times are used in two ways:
  1. Passed to AudioMetadata so the LLM sees valid snap targets.
  2. Consumed by extract_beat_boundaries() to place phrase-level candidates.
"""

from __future__ import annotations

import librosa
import numpy as np

from shared.logger import get_logger

logger = get_logger("features.tempo")


def extract_tempo(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> tuple[float, np.ndarray]:
    """
    Estimate global BPM and beat positions.

    Parameters
    ----------
    y          : mono float32 audio waveform.
    sr         : sample rate.
    hop_length : hop length for beat tracking.

    Returns
    -------
    (bpm, beat_times)
      bpm        : float, estimated beats per minute (≥ 0).
      beat_times : np.ndarray of float32, beat positions in seconds.
    """
    if y.size == 0:
        logger.warning("Empty audio passed to extract_tempo; returning defaults.")
        return 0.0, np.array([], dtype=np.float32)

    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length
        )
        # librosa ≥ 0.10 returns tempo as a 1-element array.
        bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=hop_length
        ).astype(np.float32)

        logger.debug("BPM=%.2f, beats=%d", bpm, len(beat_times))
        return round(bpm, 3), beat_times

    except Exception as exc:
        logger.warning("Tempo extraction failed (%s); returning defaults.", exc)
        return 0.0, np.array([], dtype=np.float32)
