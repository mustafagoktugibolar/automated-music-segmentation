"""
SSM-based structure analysis: construction, enhancement, and novelty detection.

Follows the Fundamentals of Music Processing (FMP) framework:
  - Section 4.2: Self-similarity matrix construction and enhancement
  - Section 4.4.1: Gaussian checkerboard kernel novelty curve

All heavy computation is numpy-only for portability and speed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from shared.logger import get_logger

logger = get_logger("structure_analyzer")


class StructureAnalyzer:
    """
    Compute and analyse the self-similarity matrix (SSM) of a feature sequence.

    Methods are stateless — pass feature matrices directly to each method.
    """

    # ------------------------------------------------------------------
    # SSM construction
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ssm(feat_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity SSM from an L2-normalised (D, N) feature matrix.

        S[i, j] = <feat_i, feat_j>  (dot product of unit vectors = cosine sim.)

        Parameters
        ----------
        feat_matrix : (D, N) float32, each column should be L2-normalised.

        Returns
        -------
        S : (N, N) float32, values in [-1, 1].
        """
        F = feat_matrix.T.astype(np.float32)  # (N, D)
        S = (F @ F.T).astype(np.float32)
        np.clip(S, -1.0, 1.0, out=S)
        return S

    # ------------------------------------------------------------------
    # SSM enhancement
    # ------------------------------------------------------------------

    @staticmethod
    def enhance_ssm(
        S: np.ndarray,
        L: int = 14,
        rho: float = 0.2,
        penalty: float = -2.0,
        tempo_ratios: list[float] | None = None,
    ) -> np.ndarray:
        """
        Enhance the SSM via diagonal smoothing and global thresholding.

        FMP Section 4.2.2 — tempo-invariant enhancement:
          1. Smooth along diagonals at each tempo ratio θ (forward + backward).
          2. Cell-wise max over all smoothed versions.
          3. Global threshold: top ρ fraction normalised to [0, 1];
             sub-threshold cells → penalty δ.

        Parameters
        ----------
        S            : (N, N) SSM from compute_ssm().
        L            : diagonal smoothing window length (frames).
        rho          : fraction of cells to retain above threshold.
        penalty      : value assigned to sub-threshold cells.
        tempo_ratios : list of tempo ratios θ to smooth at.
                       Defaults to [0.66, 0.81, 1.0, 1.22, 1.50].

        Returns
        -------
        S_enhanced : (N, N) float32.
        """
        if tempo_ratios is None:
            tempo_ratios = [0.66, 0.81, 1.0, 1.22, 1.50]

        N = S.shape[0]
        if N == 0:
            return S.copy()

        versions: list[np.ndarray] = []
        for theta in tempo_ratios:
            Sf = StructureAnalyzer._diagonal_smooth(S, L, theta)
            Sb = StructureAnalyzer._diagonal_smooth(S.T, L, theta).T
            versions.append(np.maximum(Sf, Sb))

        S_smooth = np.max(np.stack(versions, axis=0), axis=0).astype(np.float32)
        np.fill_diagonal(S_smooth, 1.0)

        # Global threshold at (1 - rho) percentile.
        thresh = float(np.percentile(S_smooth, (1.0 - rho) * 100.0))
        denom = max(1.0 - thresh, 1e-8)
        S_enh = np.where(
            S_smooth >= thresh,
            (S_smooth - thresh) / denom,
            penalty,
        ).astype(np.float32)
        np.fill_diagonal(S_enh, 1.0)

        logger.debug(
            "SSM enhanced: shape=%s, threshold=%.3f, L=%d", S_enh.shape, thresh, L
        )
        return S_enh

    @staticmethod
    def _diagonal_smooth(S: np.ndarray, L: int, theta: float) -> np.ndarray:
        """
        Smooth S along direction (1, theta) — FMP Eq. 4.12.

        S_L[n, m] = (1/L) Σ_{l=0}^{L-1} S[n-l, m-round(l*theta)]
        """
        N = S.shape[0]
        S_out = np.zeros((N, N), dtype=np.float32)
        count = 0
        for l in range(L):
            r_sh = l
            c_sh = int(round(l * theta))
            if r_sh >= N or c_sh >= N:
                break
            S_out[r_sh:, c_sh:] += S[: N - r_sh, : N - c_sh]
            count += 1
        return S_out / max(count, 1)

    # ------------------------------------------------------------------
    # Novelty curve
    # ------------------------------------------------------------------

    @staticmethod
    def compute_novelty(
        S_enhanced: np.ndarray,
        L: int = 10,
        gamma: float = 10.0,
    ) -> np.ndarray:
        """
        Compute the Gaussian checkerboard kernel novelty curve.

        FMP Section 4.4.1, Eq. 4.38-4.43:
          For each frame n, inner-product of the (2L+1)×(2L+1) diagonal patch
          around (n, n) with the Gaussian-weighted checkerboard kernel K.

        Parameters
        ----------
        S_enhanced : (N, N) enhanced SSM.
        L          : half-size of the checkerboard kernel in frames.
        gamma      : controls Gaussian falloff rate.

        Returns
        -------
        novelty : (N,) float32, normalised to [0, 1].
        """
        N = S_enhanced.shape[0]
        if N == 0:
            return np.array([], dtype=np.float32)

        M = 2 * L + 1
        k = np.arange(-L, L + 1, dtype=np.float32)
        kk, ll = np.meshgrid(k, k, indexing="ij")

        # Checkerboard pattern weighted by Gaussian.
        K = (np.sign(kk) * np.sign(ll)).astype(np.float32)
        eps = gamma / max(L, 1)
        K *= np.exp(-(eps**2) * (kk**2 + ll**2)).astype(np.float32)
        abs_sum = float(np.sum(np.abs(K)))
        if abs_sum > 0:
            K /= abs_sum

        S_pad = np.pad(
            S_enhanced.astype(np.float32), L, mode="constant", constant_values=0.0
        )
        novelty = np.empty(N, dtype=np.float32)
        K_flat = K.ravel()
        for n in range(N):
            novelty[n] = float(
                np.dot(K_flat, S_pad[n : n + M, n : n + M].ravel())
            )

        novelty = np.maximum(novelty, 0.0)
        max_val = float(novelty.max())
        if max_val > 0:
            novelty /= max_val

        return novelty

    # ------------------------------------------------------------------
    # Peak picking
    # ------------------------------------------------------------------

    @staticmethod
    def pick_peaks(
        novelty: np.ndarray,
        feature_rate: float,
        min_dist_sec: float = 8.0,
        prominence: float = 0.18,
    ) -> np.ndarray:
        """
        Pick boundary frame indices from the novelty curve.

        Parameters
        ----------
        novelty      : (N,) novelty curve, normalised.
        feature_rate : frames per second.
        min_dist_sec : minimum time between consecutive peaks (seconds).
        prominence   : minimum peak prominence on [0, 1].

        Returns
        -------
        peak_frames : (K,) int array of boundary frame indices.
        """
        if novelty.size == 0 or feature_rate <= 0:
            return np.array([], dtype=np.int64)

        # Optionally smooth before picking.
        if novelty.size > 5:
            novelty_smooth = gaussian_filter1d(novelty, sigma=2.5)
            max_v = float(novelty_smooth.max())
            if max_v > 0:
                novelty_smooth = novelty_smooth / max_v
        else:
            novelty_smooth = novelty

        min_dist_frames = max(1, int(min_dist_sec * feature_rate))
        edge_margin = max(1, int(min_dist_sec * 0.5 * feature_rate))

        try:
            peaks, _ = find_peaks(
                novelty_smooth,
                distance=min_dist_frames,
                prominence=prominence,
            )
        except Exception as exc:
            logger.warning("find_peaks failed (%s).", exc)
            return np.array([], dtype=np.int64)

        # Remove peaks too close to the edges.
        peaks = peaks[
            (peaks >= edge_margin) & (peaks <= novelty.size - edge_margin)
        ]

        logger.debug(
            "Novelty peak picking: %d peaks found (min_dist=%.1fs, prominence=%.2f)",
            len(peaks),
            min_dist_sec,
            prominence,
        )
        return peaks.astype(np.int64)
