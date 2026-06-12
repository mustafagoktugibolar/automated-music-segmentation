"""
RMS-based active region detection (silence removal).

Uses a dynamic RMS threshold (75th percentile of RMS-dB minus margin_db) to find
the start and end of the musically active portion of the track. Falls back to the
full track if the signal is too short or too quiet.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from shared.logger import get_logger

logger = get_logger("silence_removal")

_MARGIN_DB: float = 20.0
_MIN_REGION_S: float = 3.0


def detect_active_region(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = _MARGIN_DB,
) -> tuple[np.ndarray, float, float]:
    """
    Detect the active (non-silent) region of *y*.

    Parameters
    ----------
    y          : mono float32 waveform
    sr         : sample rate
    frame_length : RMS frame length
    hop_length   : RMS hop length
    top_db       : dynamic margin below the 75th-percentile RMS-dB level that
                   defines the silence threshold (analogous to librosa's top_db).

    Returns
    -------
    y_active   : trimmed waveform (view into y)
    start_sec  : start time of the active region in seconds
    end_sec    : end time of the active region in seconds

    The returned (start_sec, end_sec) are relative to the beginning of *y*.
    """
    total_dur = float(librosa.get_duration(y=y, sr=sr))

    if y.size == 0:
        logger.warning("Empty audio passed to detect_active_region.")
        return y, 0.0, total_dur

    try:
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0].astype(np.float32)

        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Smooth to avoid spurious silent frames inside dense sections.
        if rms_db.size > 5:
            rms_db = gaussian_filter1d(rms_db, sigma=2.0)

        # Dynamic threshold: P75 of the RMS-dB curve minus margin.
        threshold_db = float(np.percentile(rms_db, 75)) - float(top_db)
        active_mask = rms_db > threshold_db
        active_frames = np.where(active_mask)[0]

        if active_frames.size == 0:
            raise ValueError("No active frames found above threshold.")

        start_frame = int(active_frames[0])
        end_frame = int(active_frames[-1])

        start_sec = float(
            librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        )
        end_sec = float(
            librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
        )

        # Sanity check — region must be meaningful.
        if (end_sec - start_sec) < _MIN_REGION_S:
            raise ValueError(
                f"Active region too short: {end_sec - start_sec:.2f}s < {_MIN_REGION_S}s"
            )

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        y_active = y[start_sample:end_sample]

        logger.info(
            "Active region detected: %.2fs – %.2fs (%.2fs)",
            start_sec,
            end_sec,
            end_sec - start_sec,
        )
        return y_active, start_sec, end_sec

    except Exception as exc:
        logger.warning(
            "Active region detection failed (%s); using full track.", exc
        )
        return y, 0.0, total_dur
