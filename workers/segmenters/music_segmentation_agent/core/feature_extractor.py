"""
Multi-feature extractor producing a unified feature dictionary.

Extracts Chroma-CENS, MFCC, RMS energy, onset envelope, and onset times at
the native hop rate, then median-pools to the target FPS for SSM computation.
All feature matrices are L2-normalised per frame before returning.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import median_filter

from shared.logger import get_logger

logger = get_logger("feature_extractor")


class FeatureExtractor:
    """
    Extracts and preprocesses audio features for segmentation.

    Parameters
    ----------
    hop_length : int
        Librosa hop length for raw feature extraction (~43 Hz at sr=22050).
    target_fps : float
        Target frames-per-second after median-pooling (default 10.0 → 100ms grid).
    n_mfcc : int
        Number of MFCC coefficients to compute.
    """

    def __init__(
        self,
        hop_length: int = 512,
        target_fps: float = 10.0,
        n_mfcc: int = 20,
    ) -> None:
        self.hop_length = hop_length
        self.target_fps = target_fps
        self.n_mfcc = n_mfcc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self, y: np.ndarray, sr: int) -> dict:
        """
        Run all feature extractors on *y* and return a unified dict.

        Returns
        -------
        dict with keys:
          "chroma"          : (12, N) float32, L2-normalised, at target_fps
          "mfcc"            : (n_mfcc, N) float32, L2-normalised, at target_fps
          "rms_curve"       : (N,) float32, RMS amplitude at target_fps
          "rms_times"       : (N,) float32, frame centre times in seconds
          "onset_envelope"  : (M,) float32, onset strength at raw hop rate
          "onset_times"     : (M,) float32, frame centre times in seconds
          "feature_rate"    : float, actual fps after pooling
        """
        fps_raw = float(sr) / self.hop_length
        pool_w = max(1, int(round(fps_raw / self.target_fps)))
        actual_fps = fps_raw / pool_w

        logger.debug(
            "feature_rate=%.2f Hz (pool_w=%d, raw=%.2f Hz)", actual_fps, pool_w, fps_raw
        )

        # --- Chroma-CENS ---
        chroma_raw = self._extract_chroma(y, sr)

        # --- MFCC ---
        mfcc_raw = self._extract_mfcc(y, sr)

        # --- RMS energy ---
        rms_raw = librosa.feature.rms(y=y, hop_length=self.hop_length)[0].astype(
            np.float32
        )
        raw_times = librosa.frames_to_time(
            np.arange(rms_raw.size), sr=sr, hop_length=self.hop_length
        ).astype(np.float32)

        # --- Onset strength ---
        try:
            onset_env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=self.hop_length
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("Onset strength failed (%s); using zeros.", exc)
            onset_env = np.zeros(rms_raw.size, dtype=np.float32)

        onset_times = librosa.frames_to_time(
            np.arange(onset_env.size), sr=sr, hop_length=self.hop_length
        ).astype(np.float32)

        # --- Median pool to target fps ---
        chroma_pooled = self._median_pool(chroma_raw, pool_w)
        mfcc_pooled = self._median_pool(mfcc_raw, pool_w)
        rms_pooled = self._median_pool_1d(rms_raw, pool_w)

        N = chroma_pooled.shape[1]
        frame_times = (np.arange(N, dtype=np.float32) + 0.5) / actual_fps

        # --- L2 normalise ---
        chroma_norm = self._l2_normalise(chroma_pooled)
        mfcc_norm = self._l2_normalise(mfcc_pooled)

        return {
            "chroma": chroma_norm,
            "mfcc": mfcc_norm,
            "rms_curve": rms_pooled,
            "rms_times": frame_times,
            "onset_envelope": onset_env,
            "onset_times": onset_times,
            "feature_rate": float(actual_fps),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            return librosa.feature.chroma_cens(
                y=y, sr=sr, hop_length=self.hop_length, n_chroma=12
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("chroma_cens failed (%s); falling back to chroma_cqt.", exc)
            try:
                return librosa.feature.chroma_cqt(
                    y=y, sr=sr, hop_length=self.hop_length
                ).astype(np.float32)
            except Exception as exc2:
                logger.error("chroma_cqt fallback also failed (%s).", exc2)
                n_frames = 1 + len(y) // self.hop_length
                return np.zeros((12, n_frames), dtype=np.float32)

    def _extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            return librosa.feature.mfcc(
                y=y, sr=sr, hop_length=self.hop_length, n_mfcc=self.n_mfcc
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("MFCC extraction failed (%s); using zeros.", exc)
            n_frames = 1 + len(y) // self.hop_length
            return np.zeros((self.n_mfcc, n_frames), dtype=np.float32)

    @staticmethod
    def _median_pool(feat: np.ndarray, pool_w: int) -> np.ndarray:
        """Median-pool a (D, T) feature matrix to (D, T//pool_w)."""
        if pool_w <= 1:
            return feat
        D, T = feat.shape
        n_pooled = T // pool_w
        if n_pooled == 0:
            return feat[:, :1]
        trimmed = feat[:, : n_pooled * pool_w]
        return np.median(
            trimmed.reshape(D, n_pooled, pool_w), axis=2
        ).astype(np.float32)

    @staticmethod
    def _median_pool_1d(arr: np.ndarray, pool_w: int) -> np.ndarray:
        """Median-pool a 1-D array."""
        if pool_w <= 1:
            return arr
        n_pooled = len(arr) // pool_w
        if n_pooled == 0:
            return arr[:1]
        trimmed = arr[: n_pooled * pool_w]
        return np.median(trimmed.reshape(n_pooled, pool_w), axis=1).astype(np.float32)

    @staticmethod
    def _l2_normalise(feat: np.ndarray) -> np.ndarray:
        """L2-normalise each column of a (D, N) matrix."""
        norms = np.linalg.norm(feat, axis=0, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (feat / norms).astype(np.float32)
