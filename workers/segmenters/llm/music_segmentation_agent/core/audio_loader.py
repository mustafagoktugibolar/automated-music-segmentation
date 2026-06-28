"""
Audio loading utilities.

Loads audio from a file path using librosa. Returns (y, sr) as a mono float32
waveform at the default sample rate (22050 Hz). Metadata is assembled here
without BPM/beat info — those are computed later by the feature extraction step.
"""

from __future__ import annotations

import librosa
import numpy as np

from shared.logger import get_logger

logger = get_logger("audio_loader")

_DEFAULT_SR: int = 22050


def load_audio(
    file_path: str,
    sr: int = _DEFAULT_SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio from *file_path* and return (y, sr).

    Parameters
    ----------
    file_path : str
        Path to any audio format supported by librosa / soundfile / audioread.
    sr : int
        Target sample rate. Audio is resampled if necessary.
    mono : bool
        Downmix to mono if True.

    Returns
    -------
    y  : np.ndarray, shape (n_samples,), dtype float32
    sr : int  — actual sample rate (always equals the requested sr)
    """
    logger.info("Loading audio from: %s", file_path)
    try:
        y, sr_out = librosa.load(file_path, sr=sr, mono=mono)
        y = y.astype(np.float32)
        duration = float(librosa.get_duration(y=y, sr=sr_out))
        logger.info(
            "Loaded: sr=%d, duration=%.2fs, samples=%d", sr_out, duration, len(y)
        )
        return y, sr_out
    except Exception as exc:
        logger.error("Failed to load audio from %s: %s", file_path, exc, exc_info=True)
        raise


def get_duration(y: np.ndarray, sr: int) -> float:
    """Return duration of *y* in seconds."""
    return float(librosa.get_duration(y=y, sr=sr))
