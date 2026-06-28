"""
Feature-based boundary candidate extraction from pre-computed feature dictionaries.

Combines RMS and onset envelope novelty curves into CandidateBoundary objects
with appropriate source tags and confidence scores. Designed to complement the
specialised per-feature extractors in workers/segmenters/music_segmentation_agent/features/.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .models import CandidateBoundary
from shared.logger import get_logger

logger = get_logger("feature_segmentation")


class FeatureSegmentationExtractor:
    """
    Derives CandidateBoundary objects from a features_dict produced by FeatureExtractor.

    This class is intentionally thin — it processes the already-computed feature
    dict rather than re-loading audio, keeping it cheap to call multiple times.
    """

    def __init__(
        self,
        min_dist_sec: float = 8.0,
        rms_prominence: float = 0.12,
        onset_prominence: float = 0.15,
        smooth_sigma: float = 2.0,
    ) -> None:
        self.min_dist_sec = min_dist_sec
        self.rms_prominence = rms_prominence
        self.onset_prominence = onset_prominence
        self.smooth_sigma = smooth_sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_features(
        self,
        features_dict: dict,
        sr: int,
        hop_length: int,
        total_duration: float,
    ) -> list[CandidateBoundary]:
        """
        Extract boundary candidates from a pre-computed feature dict.

        Parameters
        ----------
        features_dict  : dict returned by FeatureExtractor.extract_all().
        sr             : sample rate (for frame-to-time conversion).
        hop_length     : hop length used during feature extraction.
        total_duration : total duration of the active region in seconds.

        Returns
        -------
        List of CandidateBoundary sorted by time.
        """
        fps = float(features_dict.get("feature_rate", 10.0))
        rms_curve = features_dict.get("rms_curve", np.array([], dtype=np.float32))
        rms_times = features_dict.get("rms_times", np.array([], dtype=np.float32))
        onset_env = features_dict.get("onset_envelope", np.array([], dtype=np.float32))
        onset_times = features_dict.get("onset_times", np.array([], dtype=np.float32))

        candidates: list[CandidateBoundary] = []

        # --- RMS-derived candidates ---
        rms_candidates = self._extract_rms_candidates(
            rms_curve, rms_times, fps, total_duration
        )
        candidates.extend(rms_candidates)

        # --- Onset-derived candidates ---
        onset_candidates = self._extract_onset_candidates(
            onset_env, onset_times, rms_times, fps, total_duration
        )
        candidates.extend(onset_candidates)

        # Merge candidates at identical (or very close) times.
        candidates = self._merge_nearby(candidates, window_sec=0.5)

        candidates.sort(key=lambda c: c.time_seconds)
        logger.debug(
            "FeatureSegmentation: %d total candidates (rms=%d, onset=%d)",
            len(candidates),
            len(rms_candidates),
            len(onset_candidates),
        )
        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_rms_candidates(
        self,
        rms_curve: np.ndarray,
        frame_times: np.ndarray,
        fps: float,
        total_duration: float,
    ) -> list[CandidateBoundary]:
        """Find boundaries at RMS energy change points."""
        if rms_curve.size < 3 or frame_times.size < 3:
            return []

        try:
            rms_db = librosa.amplitude_to_db(
                rms_curve.astype(np.float32) + 1e-10, ref=np.max
            )
            if rms_db.size > 5:
                rms_db = gaussian_filter1d(rms_db, sigma=self.smooth_sigma)

            novelty = np.abs(np.diff(rms_db, prepend=rms_db[0]))
            novelty = self._normalise(novelty)

            return self._peaks_to_candidates(
                novelty, frame_times, fps, total_duration,
                source="rms", prominence=self.rms_prominence
            )
        except Exception as exc:
            logger.warning("RMS candidate extraction failed (%s).", exc)
            return []

    def _extract_onset_candidates(
        self,
        onset_env: np.ndarray,
        onset_times: np.ndarray,
        target_times: np.ndarray,
        fps: float,
        total_duration: float,
    ) -> list[CandidateBoundary]:
        """Find boundaries at onset-flux peaks."""
        if onset_env.size < 3 or onset_times.size < 3:
            return []

        try:
            smooth = gaussian_filter1d(onset_env.astype(np.float32), sigma=5.0)
            # Interpolate to the target grid (rms_times) for consistent fps.
            if target_times.size > 0:
                novelty = np.interp(
                    target_times, onset_times, smooth, left=0.0, right=0.0
                )
                times = target_times
            else:
                novelty = smooth
                times = onset_times
                fps = float(1.0 / np.mean(np.diff(onset_times))) if len(onset_times) > 1 else fps

            novelty = self._normalise(novelty)
            return self._peaks_to_candidates(
                novelty, times, fps, total_duration,
                source="onset_flux", prominence=self.onset_prominence
            )
        except Exception as exc:
            logger.warning("Onset candidate extraction failed (%s).", exc)
            return []

    def _peaks_to_candidates(
        self,
        novelty: np.ndarray,
        times: np.ndarray,
        fps: float,
        total_duration: float,
        source: str,
        prominence: float,
    ) -> list[CandidateBoundary]:
        """Convert novelty-curve peaks to CandidateBoundary objects."""
        min_dist_frames = max(1, int(self.min_dist_sec * fps))
        edge_margin = max(1, int(self.min_dist_sec * 0.5 * fps))

        try:
            peaks, props = find_peaks(
                novelty, distance=min_dist_frames, prominence=prominence
            )
        except Exception:
            return []

        peaks = peaks[(peaks >= edge_margin) & (peaks <= novelty.size - edge_margin)]
        out: list[CandidateBoundary] = []
        for p in peaks:
            t = float(times[p]) if p < len(times) else float(p) / fps
            if t <= 0.0 or t >= total_duration:
                continue
            conf = round(float(np.clip(novelty[p], 0.0, 1.0)), 3)
            out.append(
                CandidateBoundary(
                    time_seconds=round(t, 3),
                    source=[source],
                    confidence=conf,
                )
            )
        return out

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        arr -= float(arr.min())
        max_v = float(arr.max())
        if max_v > 0:
            arr /= max_v
        return arr

    @staticmethod
    def _merge_nearby(
        candidates: list[CandidateBoundary], window_sec: float
    ) -> list[CandidateBoundary]:
        """Merge candidates within *window_sec* of each other, keeping the highest confidence."""
        if not candidates:
            return []
        candidates.sort(key=lambda c: c.time_seconds)
        out: list[CandidateBoundary] = []
        for cand in candidates:
            if not out or (cand.time_seconds - out[-1].time_seconds) > window_sec:
                out.append(cand)
            else:
                prev = out[-1]
                merged_sources = sorted(set(prev.source) | set(cand.source))
                if cand.confidence >= prev.confidence:
                    out[-1] = CandidateBoundary(
                        time_seconds=cand.time_seconds,
                        source=merged_sources,
                        confidence=cand.confidence,
                    )
                else:
                    out[-1] = CandidateBoundary(
                        time_seconds=prev.time_seconds,
                        source=merged_sources,
                        confidence=prev.confidence,
                    )
        return out
