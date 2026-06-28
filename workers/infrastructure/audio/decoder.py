"""
Unified audio decoder — single source of truth for audio loading.

Priority order for decoding:
  1. ffmpeg (fast C decoder, ~0.3-0.5s per 320s MP3)
  2. librosa.load fallback (slow Python/audioread, ~8-12s per 320s MP3)

Accepts both file paths (str) and in-memory bytes.  When bytes are given
ffmpeg reads from stdin (pipe:0) so no temporary files are created.

Usage
-----
    from workers.infrastructure.audio.decoder import load_audio

    y, sr = load_audio("/path/to/song.mp3")           # from path
    y, sr = load_audio(audio_bytes)                    # from bytes (MinIO download etc.)
    y, sr = load_audio("/path/to/song.mp3", sr=44100)  # custom sample rate
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess

import numpy as np

logger = logging.getLogger("audio_io")

_DEFAULT_SR = 22050

# --- ffmpeg discovery -----------------------------------------------------------
_EXTRA_PATHS = [
    "/opt/conda/envs/music-segmentation-worker-env/bin/ffmpeg",
    "/opt/conda/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
]


def find_ffmpeg() -> str | None:
    """Return the absolute path to ffmpeg, or None if not found."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in _EXTRA_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# Module-level detection — done once at import time.
FFMPEG_BIN: str | None = find_ffmpeg()
if FFMPEG_BIN:
    logger.debug("ffmpeg found: %s", FFMPEG_BIN)
else:
    logger.warning("ffmpeg not found — audio loading will use librosa (slow for large files).")


# --- Public API -----------------------------------------------------------------

def load_audio(
    src: "str | bytes",
    sr: int = _DEFAULT_SR,
) -> tuple[np.ndarray, int]:
    """Load audio from a file path or in-memory bytes.

    Parameters
    ----------
    src : str or bytes
        File path on disk, *or* raw audio file bytes (e.g. downloaded from
        MinIO).  The format is inferred by ffmpeg / librosa automatically.
    sr : int
        Target sample rate.  Audio is resampled if necessary.

    Returns
    -------
    y : np.ndarray, shape (N,), dtype float32
        Mono audio signal normalised to [-1, 1].
    sr : int
        Actual sample rate (equals *sr* parameter).
    """
    is_path = isinstance(src, str)
    if FFMPEG_BIN:
        return _load_ffmpeg(src, sr, is_path=is_path)
    return _load_librosa(src, sr, is_path=is_path)


# --- Internal helpers -----------------------------------------------------------

def _load_ffmpeg(
    src: "str | bytes",
    sr: int,
    *,
    is_path: bool,
) -> tuple[np.ndarray, int]:
    """Decode via ffmpeg subprocess.

    File path → -i <path> -nostdin (no stdin interaction)
    Bytes     → -i pipe:0 with input=<bytes> via stdin (no temp file)
    """
    if is_path:
        cmd = [
            FFMPEG_BIN, "-nostdin",
            "-i", src,
            "-f", "f32le", "-ar", str(sr), "-ac", "1",
            "-loglevel", "error", "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    else:
        # pipe:0 = stdin; do NOT use -nostdin when reading from stdin.
        cmd = [
            FFMPEG_BIN,
            "-i", "pipe:0",
            "-f", "f32le", "-ar", str(sr), "-ac", "1",
            "-loglevel", "error", "pipe:1",
        ]
        proc = subprocess.run(cmd, input=src, capture_output=True, timeout=180)

    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {err}")

    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size == 0:
        raise RuntimeError("ffmpeg produced empty output — audio may be corrupt or unreadable.")
    return y, sr


def _load_librosa(
    src: "str | bytes",
    sr: int,
    *,
    is_path: bool,
) -> tuple[np.ndarray, int]:
    """Fallback decoder using librosa / audioread."""
    try:
        import librosa as _librosa

        if is_path:
            y, sr_out = _librosa.load(src, sr=sr, mono=True)
        else:
            y, sr_out = _librosa.load(io.BytesIO(src), sr=sr, mono=True)
        return y.astype(np.float32), sr_out
    except Exception as exc:
        raise RuntimeError(f"librosa audio load failed: {exc}") from exc
