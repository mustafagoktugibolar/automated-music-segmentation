"""
Music segmentation service — deterministic multi-feature baseline.

Pipeline:
  Audio load (22050 Hz mono)
  → Active-region detection (RMS-based)
  → Shared feature grid: Chroma-CENS + MFCC, median-pooled to target FPS
  → Candidate extraction from RMS, onset/flux, tempo/beat, chord-change proxy,
    optional timed lyrics, and Chroma/MFCC self-similarity novelty
  → Weighted candidate filtering and fusion
  → Boundary snapping to strong onsets or beat positions
  → Segment clustering and section type assignment
  → Active-region offset correction back to the full-track timeline

The pipeline is intentionally deterministic. LLM-assisted decisions can be added
later on top of the grounded candidate boundary set without allowing fabricated
timestamps.
"""

from __future__ import annotations

import io
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

from shared.logger import get_logger
from workers.segmenters.multi_feature_fusion import (
    beat_phrase_boundary_candidates,
    candidates_from_boundaries,
    chord_proxy_boundary_candidates,
    find_boundaries,
    fuse_boundary_candidates,
    lyrics_boundary_candidates,
    normalise_feature_weights,
    onset_boundary_candidates,
    rms_boundary_candidates,
    snap_fused_boundaries,
    tempo_and_beats,
)

logger = get_logger()

# ---------------------------------------------------------------------------
# Module-level constants — all tunable via the params dict at call time.
# ---------------------------------------------------------------------------

_SR: int = 22050
_HOP_LENGTH: int = 512            # raw feature extraction hop (~43 Hz)
_TARGET_FPS: float = 10.0         # target feature rate after median-pooling (0.1s/frame)

# Diagonal-smoothing window in frames.  At 10 Hz, L=8 → 0.8s of smoothing,
# which is enough to reinforce repeating diagonal paths without shifting the
# apparent transition by more than ~0.4s (L/2 frames = 0.4s at 10 Hz).
# The old L=20 @ 5 Hz was a 4s window → ~2s systematic shift, killing ±0.5s F1.
_SMOOTHING_L: int = 14
_TEMPO_RATIOS: list[float] = [0.66, 0.81, 1.0, 1.22, 1.50]

_SSM_RHO: float = 0.20            # fraction of SSM cells kept (FMP Eq. 4.17)

_KERNEL_SECONDS: float = 8.0      # checkerboard kernel half-size in seconds (80 frames @ 10 Hz)
_KERNEL_VAR: float = 1.0          # Gaussian taper variance on normalised [-1,1] coords (FMP Eq. 4.40)
_NOVELTY_SIGMA: float = 2.5       # Gaussian smoothing σ applied to novelty curve
_PROMINENCE: float = 0.12         # scipy find_peaks min prominence on [0,1] novelty
                                  # (0.18 under-segmented: SALAMI sweep showed
                                  #  0.12 lifts recall with no precision cost)
_STRUCTURE_WEIGHT: float = 0.4    # share of structure-feature novelty in the combined curve

_MIN_SEG_DUR: float = 10.0        # minimum segment duration (seconds)
_N_CLUSTERS: int = 4
_MFCC_N: int = 20

_ACTIVE_MARGIN_DB: float = 20.0
_ACTIVE_MIN_S: float = 3.0

_MAX_SSM_FRAMES: int = 2000       # cap on frames (10 Hz × 4 min = 2400 frames — fits comfortably)

# Onset-snapping: after SSM peak picking, snap each boundary to the nearest
# strong onset within this window.  The SSM locates the right neighbourhood;
# onset strength gives sub-25ms precision at hop=512.  Kept well below the
# ±0.5s MIREX tolerance so a snap can never push a correct boundary outside
# the evaluation window.
_ONSET_SNAP_WINDOW: float = 0.25  # seconds


# ---------------------------------------------------------------------------
# Stage 0 — Audio loading
# ---------------------------------------------------------------------------

def _load_audio_from_bytes(content: bytes, sr: int = _SR) -> tuple[np.ndarray, int]:
    """Load audio from in-memory bytes, resample to sr, return mono float32."""
    try:
        y, sr_out = librosa.load(io.BytesIO(content), sr=sr, mono=True)
        return y.astype(np.float32), sr_out
    except Exception as exc:
        logger.error("Failed to load audio: %s", exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Stage 0b — Active-region detection
# ---------------------------------------------------------------------------

def _detect_active_region(
    y: np.ndarray,
    sr: int,
    hop_length: int = _HOP_LENGTH,
    margin_db: float = _ACTIVE_MARGIN_DB,
    min_region_s: float = _ACTIVE_MIN_S,
) -> tuple[float, float]:
    """
    Return (start_s, end_s) of the musically active region.

    Dynamic threshold = P75(RMS_dB) − margin_db.  Falls back to (0, duration).
    """
    if y.size == 0:
        return 0.0, 0.0
    try:
        rms_db = librosa.amplitude_to_db(
            librosa.feature.rms(y=y, hop_length=hop_length)[0], ref=np.max
        )
        if rms_db.size > 3:
            rms_db = gaussian_filter1d(rms_db, sigma=2.0)
        threshold = float(np.percentile(rms_db, 75)) - margin_db
        active = np.where(rms_db > threshold)[0]
        if active.size == 0:
            raise ValueError("no active frames")
        t0 = float(librosa.frames_to_time(active[0], sr=sr, hop_length=hop_length))
        t1 = float(librosa.frames_to_time(active[-1], sr=sr, hop_length=hop_length))
        if (t1 - t0) < min_region_s:
            raise ValueError("active region too short")
        return t0, t1
    except Exception:
        total = float(librosa.get_duration(y=y, sr=sr))
        return 0.0, total


# ---------------------------------------------------------------------------
# Stage 1 — Feature extraction + median-pool to target fps
# ---------------------------------------------------------------------------

def _median_pool(feat: np.ndarray, pw: int) -> np.ndarray:
    """Median-pool each pw consecutive columns → shape (D, n_raw//pw)."""
    if pw <= 1:
        return feat
    n_raw = feat.shape[1]
    n_pooled = n_raw // pw
    if n_pooled == 0:
        return feat[:, :1]
    trimmed = feat[:, : n_pooled * pw]           # (D, n_pooled*pw)
    return np.median(
        trimmed.reshape(feat.shape[0], n_pooled, pw), axis=2
    ).astype(np.float32)


def _extract_downsampled_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = _HOP_LENGTH,
    target_fps: float = _TARGET_FPS,
    n_mfcc: int = _MFCC_N,
    use_mfcc: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extract Chroma-CENS and MFCC at hop_length, then median-pool to ~target_fps.

    Using a uniform frame grid rather than beat-sync gives fixed 1/fps second
    spacing — worst-case boundary snap error = 0.5/fps = 0.1s at 5 Hz, always
    within the ±0.5s MIREX tolerance window.

    Returns
    -------
    chroma     : (12, N) float32, L2-normalised
    mfcc       : (n_mfcc, N) float32, L2-normalised  (zeros if use_mfcc=False)
    frame_times: (N,) seconds of each pooled frame
    fps        : actual frames per second after pooling
    """
    fps_raw = float(sr) / hop_length          # ~43 Hz
    pool_w = max(1, int(round(fps_raw / target_fps)))
    fps = fps_raw / pool_w                     # actual fps after pooling

    # --- Chroma-CENS ---
    try:
        chroma_raw = librosa.feature.chroma_cens(
            y=y, sr=sr, hop_length=hop_length, n_chroma=12
        ).astype(np.float32)
    except Exception as exc:
        logger.warning("chroma_cens failed (%s); using chroma_cqt fallback.", exc)
        chroma_raw = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length
        ).astype(np.float32)

    # --- MFCC ---
    # MFCC0 ≈ frame log-energy: its magnitude swamps the higher coefficients
    # under cosine similarity, turning the "timbre SSM" into a loudness SSM.
    # Request one extra coefficient and drop the first.
    if use_mfcc:
        try:
            mfcc_raw = librosa.feature.mfcc(
                y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc + 1
            )[1:].astype(np.float32)
        except Exception as exc:
            logger.warning("MFCC failed (%s); disabling.", exc)
            mfcc_raw = np.zeros((n_mfcc, chroma_raw.shape[1]), dtype=np.float32)
    else:
        mfcc_raw = np.zeros((n_mfcc, chroma_raw.shape[1]), dtype=np.float32)

    chroma = _median_pool(chroma_raw, pool_w)
    mfcc   = _median_pool(mfcc_raw,   pool_w)

    # L2-normalise per frame column (FMP Section 4.2.1)
    def _l2_norm(feat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(feat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return feat / norms

    chroma = _l2_norm(chroma)
    if use_mfcc:
        # Z-score per coefficient before cosine: MFCC dimensions have wildly
        # different scales, so without standardisation the low-order
        # coefficients dominate the inner product.
        mu  = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        mfcc = (mfcc - mu) / np.maximum(std, 1e-8)
        mfcc = _l2_norm(mfcc)

    N = chroma.shape[1]
    frame_times = (np.arange(N, dtype=np.float32) + 0.5) / fps

    logger.debug(
        "Features: pool_w=%d, fps=%.2f, n_frames=%d, chroma=%s, mfcc=%s",
        pool_w, fps, N, chroma.shape, mfcc.shape,
    )
    return chroma, mfcc, frame_times, fps


# ---------------------------------------------------------------------------
# Stage 2 — SSM construction
# ---------------------------------------------------------------------------

def _compute_ti_chroma_ssm(chroma: np.ndarray) -> np.ndarray:
    """
    Transposition-invariant chroma SSM — FMP Eq. 4.15.

    S_TI[n, m] = max_{c=0..11}  <roll_c(chroma_n), chroma_m>

    Captures repetitions at any transposition (modulation to a new key).
    Complexity: O(12 × N²) — ~12 × 600² = 4.3M ops for a 4-min song at 5 Hz.
    """
    N = chroma.shape[1]
    S_TI = np.full((N, N), -np.inf, dtype=np.float32)
    for c in range(12):
        S_c = (np.roll(chroma, c, axis=0).T @ chroma).astype(np.float32)
        np.maximum(S_TI, S_c, out=S_TI)
    return S_TI


def _compute_raw_ssm(feat: np.ndarray) -> np.ndarray:
    """Standard cosine SSM from L2-normalised (D, N) feature matrix."""
    F = feat.T.astype(np.float32)
    return (F @ F.T).astype(np.float32)


def _build_combined_ssm(
    chroma: np.ndarray,
    mfcc: np.ndarray,
    use_mfcc: bool,
    transposition_invariant: bool,
) -> np.ndarray:
    """
    Blend chroma (± TI) and MFCC SSMs equally.

    Chroma captures harmonic repetition (verse/chorus identity).
    MFCC captures timbral texture (instrumentation changes).
    Equal blend gives a balanced structural view.
    """
    S_chroma = _compute_ti_chroma_ssm(chroma) if transposition_invariant else _compute_raw_ssm(chroma)

    if use_mfcc and np.any(mfcc != 0):
        S = 0.5 * S_chroma + 0.5 * _compute_raw_ssm(mfcc)
    else:
        S = S_chroma

    return S.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 3 — SSM enhancement
# ---------------------------------------------------------------------------

def _diagonal_smooth_theta(
    S: np.ndarray, L: int, theta: float, forward: bool = True
) -> np.ndarray:
    """
    Smooth S along direction (1, theta) — FMP Eq. 4.12.

    forward=True  (trailing): S_L[n, m] = (1/L) Σ_{l} S[n-l, m-round(l·θ)]
    forward=False (leading):  S_L[n, m] = (1/L) Σ_{l} S[n+l, m+round(l·θ)]

    Both directions are needed: a single trailing average smears every block
    edge ~L/2 frames late, which shifts all detected boundaries by the same
    amount (≈1.3s at L=14, 5 Hz — outside the ±0.5s tolerance on its own).
    Taking the cell-wise max of the two keeps block edges anchored at the
    true transition.  theta=1.0 gives standard diagonal smoothing (Eq. 4.11).
    """
    N = S.shape[0]
    S_out = np.zeros((N, N), dtype=np.float32)
    count = 0
    for l in range(L):
        r_sh = l
        c_sh = int(round(l * theta))
        if r_sh >= N or c_sh >= N:
            break
        if forward:
            S_out[r_sh:, c_sh:] += S[: N - r_sh, : N - c_sh]
        else:
            S_out[: N - r_sh, : N - c_sh] += S[r_sh:, c_sh:]
        count += 1
    return S_out / max(count, 1)


def _smooth_ssm(
    S_raw: np.ndarray,
    L: int,
    tempo_ratios: list[float] = _TEMPO_RATIOS,
) -> np.ndarray:
    """
    Tempo-invariant diagonal smoothing — FMP Section 4.2.2, Eq. 4.12-4.13.

    Smooth at each θ ∈ Θ, forward + backward, then take the cell-wise max
    over all 2|Θ| versions.  The result keeps the cosine value range, which
    is what the checkerboard novelty kernel expects; thresholding with a
    penalty (Eq. 4.17) is applied separately and only where sparseness is
    wanted (structure features, segment labelling).
    """
    versions: list[np.ndarray] = []
    for theta in tempo_ratios:
        Sf = _diagonal_smooth_theta(S_raw, L, theta, forward=True)
        Sb = _diagonal_smooth_theta(S_raw, L, theta, forward=False)
        versions.append(np.maximum(Sf, Sb))

    S_smooth = np.max(np.stack(versions, axis=0), axis=0).astype(np.float32)
    np.fill_diagonal(S_smooth, 1.0)
    return S_smooth


def _threshold_ssm(
    S_smooth: np.ndarray,
    rho: float = _SSM_RHO,
    delta: float = 0.0,
) -> np.ndarray:
    """
    Global relative thresholding — FMP Eq. 4.17 with δ=0 (sparsify only).

    Keeps the top ρ cells rescaled to [0,1]; the rest are set to delta.
    A negative penalty δ is only useful for path-family DP scoring — for
    structure features and segment similarity a zero floor is correct.
    """
    thresh = float(np.percentile(S_smooth, (1.0 - rho) * 100.0))
    denom = max(1.0 - thresh, 1e-8)
    S_enh = np.where(
        S_smooth >= thresh,
        (S_smooth - thresh) / denom,
        delta,
    ).astype(np.float32)
    np.fill_diagonal(S_enh, 1.0)
    return S_enh


# ---------------------------------------------------------------------------
# Stage 4 — Novelty curve (Gaussian checkerboard kernel)
# ---------------------------------------------------------------------------

def _compute_novelty_ssm(S: np.ndarray, L: int, var: float = _KERNEL_VAR) -> np.ndarray:
    """
    Gaussian checkerboard kernel novelty — FMP Section 4.4.1, Eq. 4.38-4.43.

    For each frame n, inner-product of the (2L+1)×(2L+1) diagonal patch with
    kernel K.  Complexity: O(N × (2L+1)²).

    The Gaussian taper is defined on coordinates normalised to [-1, 1]
    (FMP Eq. 4.40): exp(-var·((k/L)² + (l/L)²)), so the taper scale follows
    the kernel size.  var=1.0 → weight e⁻² at the kernel corners.

    L (half-size in frames) controls the temporal scale of detected
    boundaries: L=80 frames @ 10 Hz → 8s per side → section-level
    transitions.
    """
    N = S.shape[0]
    if N < 3:
        return np.zeros(N, dtype=np.float32)
    # Reflect padding requires pad width < dimension size; clamp for very
    # short tracks where the requested kernel exceeds the SSM itself.
    L = min(L, N - 1)
    M = 2 * L + 1
    k = np.arange(-L, L + 1, dtype=np.float32) / max(L, 1)   # normalised [-1, 1]
    kk, ll = np.meshgrid(k, k, indexing='ij')

    K = (np.sign(kk) * np.sign(ll)).astype(np.float32)
    K *= np.exp(-var * (kk ** 2 + ll ** 2)).astype(np.float32)
    abs_sum = float(np.sum(np.abs(K)))
    if abs_sum > 0:
        K /= abs_sum

    # Reflect padding: a mirrored block is symmetric around the edge, so the
    # checkerboard response stays near zero there instead of spiking against
    # an artificial constant background.
    S_pad = np.pad(S.astype(np.float32), L, mode='reflect')

    # Diagonal patches as a zero-copy strided view: patches[n] = S_pad[n:n+M, n:n+M]
    s0, s1 = S_pad.strides
    patches = np.lib.stride_tricks.as_strided(
        S_pad, shape=(N, M, M), strides=(s0 + s1, s0, s1), writeable=False
    )
    novelty = np.einsum('nij,ij->n', patches, K).astype(np.float32)

    novelty = np.maximum(novelty, 0.0)
    max_val = float(novelty.max())
    if max_val > 0:
        novelty /= max_val
    return novelty


def _structure_feature_novelty(S_thresh: np.ndarray, sigma_time: float = 2.0) -> np.ndarray:
    """
    Structure-feature novelty — FMP Section 4.4.2, Eq. 4.44-4.46.

    Builds the circular time-lag matrix L◦(ℓ, n) = S((n+ℓ) mod N, n) from the
    thresholded SSM and takes the L2 distance between consecutive columns.
    Unlike the local checkerboard, each column encodes the *global* repetition
    context of frame n, so boundaries where the repetition pattern changes
    (e.g. verse→chorus in identical instrumentation) become visible.
    """
    N = S_thresh.shape[0]
    if N < 3:
        return np.zeros(N, dtype=np.float32)

    rows = (np.arange(N)[:, None] + np.arange(N)[None, :]) % N   # (ℓ, n) → (n+ℓ) mod N
    L_circ = S_thresh[rows, np.arange(N)[None, :]].astype(np.float32)

    # Light smoothing along time so the column difference reflects structural
    # change rather than frame-level sparsity noise in the thresholded SSM.
    if sigma_time > 0:
        L_circ = gaussian_filter1d(L_circ, sigma=sigma_time, axis=1)

    diff = np.linalg.norm(np.diff(L_circ, axis=1), axis=0)
    novelty = np.concatenate([[0.0], diff]).astype(np.float32)
    max_val = float(novelty.max())
    if max_val > 0:
        novelty /= max_val
    return novelty


# ---------------------------------------------------------------------------
# Stage 6 — Segment clustering and labelling
# ---------------------------------------------------------------------------

def _select_n_clusters(X: np.ndarray, max_k: int = 6) -> int:
    """Auto-select k via silhouette score over [2, min(max_k, n-1)]."""
    from sklearn.metrics import silhouette_score
    n = len(X)
    if n < 3:
        return min(2, n)
    upper = min(max_k, n - 1, 6)
    if upper < 2:
        return 2
    best_k, best_score = 2, -1.0
    for k in range(2, upper + 1):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=3, init='k-means++')
            labels = km.fit_predict(X)
            score = float(silhouette_score(X, labels))
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return best_k


def _ssm_segment_labels(
    S_thresh: np.ndarray,
    frame_times: np.ndarray,
    seg_spans: list[tuple[float, float]],
    fixed_k: int | None = None,
    max_k: int = 6,
) -> np.ndarray | None:
    """
    Label segments from pairwise SSM similarity (FMP-style grouping).

    Two segments belong to the same section type iff the SSM sub-block
    spanning them is dense (they repeat each other), so the mean of the
    thresholded SSM over [seg_i × seg_j] is a direct repetition measure —
    more reliable than KMeans on hand-crafted descriptor vectors when only
    ~10 samples exist.  Average-linkage agglomerative clustering on
    1 − similarity; k chosen by silhouette unless fixed_k is given.
    Returns None when the structure is degenerate (caller falls back).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    n = len(seg_spans)
    if n < 3 or S_thresh.shape[0] != frame_times.size:
        return None

    idxs: list[np.ndarray] = []
    for t0, t1 in seg_spans:
        mask = (frame_times >= t0) & (frame_times < t1)
        if not mask.any():
            mask = np.zeros(frame_times.size, dtype=bool)
            mask[int(np.argmin(np.abs(frame_times - (t0 + t1) / 2)))] = True
        idxs.append(np.where(mask)[0])

    sims = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            block = S_thresh[np.ix_(idxs[i], idxs[j])]
            sims[i, j] = sims[j, i] = float(block.mean())

    lo, hi = float(sims.min()), float(sims.max())
    if hi - lo < 1e-6:
        return None
    dist = 1.0 - (sims - lo) / (hi - lo)
    np.fill_diagonal(dist, 0.0)

    try:
        Z = linkage(squareform(dist, checks=False), method="average")
    except Exception as exc:
        logger.warning("SSM segment linkage failed (%s).", exc)
        return None

    if fixed_k is not None:
        labels = fcluster(Z, t=min(fixed_k, n), criterion="maxclust") - 1
        return labels if len(set(labels)) >= 1 else None

    best_labels, best_score = None, -1.0
    for k in range(2, min(max_k, n - 1) + 1):
        labels = fcluster(Z, t=k, criterion="maxclust") - 1
        if len(set(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(dist, labels, metric="precomputed"))
        except Exception:
            continue
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels


def _segment_feature_vector(feat: np.ndarray) -> np.ndarray:
    """
    Segment descriptor: mean ‖ delta_mean ‖ std.

    Combining mean, frame-to-frame delta and std makes clustering discriminative
    between segments that share a tonal centre but differ in dynamic texture
    (e.g., verse vs. pre-chorus with similar harmony but different density).
    """
    if feat.shape[1] < 2:
        mean = feat[:, 0] if feat.shape[1] == 1 else np.zeros(feat.shape[0])
        return np.concatenate([mean, np.zeros_like(mean), np.zeros_like(mean)])
    mean  = np.mean(feat, axis=1).astype(np.float32)
    delta = np.mean(np.abs(np.diff(feat, axis=1)), axis=1).astype(np.float32)
    std   = np.std(feat, axis=1).astype(np.float32)
    return np.concatenate([mean, delta, std])


def _enforce_min_segment_duration(
    segments: list[dict],
    min_dur: float,
    total_dur: float,
) -> list[dict]:
    """Iteratively merge short segments with their longer neighbour."""
    if not segments:
        return segments
    segs = [dict(s) for s in segments]

    def _merge(left: dict, right: dict, label_from: dict | None = None) -> dict:
        label_src = label_from or left
        sources = sorted(set(left.get("source_features", [])) | set(right.get("source_features", [])))
        confidence = round(float(np.mean([
            float(left.get("confidence", 0.5)),
            float(right.get("confidence", 0.5)),
        ])), 3)
        merged = {
            **label_src,
            "start": left["start"],
            "end": right["end"],
            "label": label_src["label"],
            "confidence": confidence,
            "source_features": sources,
        }
        return merged

    changed = True
    while changed:
        changed = False
        out: list[dict] = []
        i = 0
        while i < len(segs):
            seg = segs[i]
            dur = seg["end"] - seg["start"]
            if dur >= min_dur or len(segs) == 1:
                out.append(seg)
                i += 1
                continue
            if i == 0 and len(segs) > 1:
                nxt = segs[i + 1]
                out.append(_merge(seg, nxt, label_from=nxt))
                i += 2
            elif i == len(segs) - 1:
                out[-1]["end"] = seg["end"]
                out[-1]["confidence"] = round(float(np.mean([
                    float(out[-1].get("confidence", 0.5)),
                    float(seg.get("confidence", 0.5)),
                ])), 3)
                out[-1]["source_features"] = sorted(
                    set(out[-1].get("source_features", [])) | set(seg.get("source_features", []))
                )
                i += 1
            else:
                left  = out[-1]
                right = segs[i + 1]
                if (left["end"] - left["start"]) >= (right["end"] - right["start"]):
                    left["end"] = seg["end"]
                    left["confidence"] = round(float(np.mean([
                        float(left.get("confidence", 0.5)),
                        float(seg.get("confidence", 0.5)),
                    ])), 3)
                    left["source_features"] = sorted(
                        set(left.get("source_features", [])) | set(seg.get("source_features", []))
                    )
                    i += 1
                else:
                    out.append(_merge(seg, right, label_from=right))
                    i += 2
            changed = True
        segs = out
    if segs:
        segs[-1]["end"] = min(segs[-1]["end"], total_dur)
    return segs


def _assign_section_types(segments: list[dict], total_dur: float) -> list[dict]:
    """
    Heuristic type assignment:
      · Longest label → Chorus, second → Verse, third → Bridge, rest → Other
      · First segment (≥4 total) → Intro, last → Outro
    """
    if not segments:
        return segments
    dur_by_label: dict[str, float] = {}
    for s in segments:
        dur_by_label[s["label"]] = dur_by_label.get(s["label"], 0.0) + (s["end"] - s["start"])
    sorted_labels = [lbl for lbl, _ in sorted(dur_by_label.items(), key=lambda x: -x[1])]
    type_map = {lbl: ("Chorus" if i == 0 else "Verse" if i == 1 else "Bridge" if i == 2 else "Other")
                for i, lbl in enumerate(sorted_labels)}
    n = len(segments)
    out: list[dict] = []
    for i, seg in enumerate(segments):
        stype = type_map.get(seg["label"], "Other")
        if n >= 4:
            if i == 0:
                stype = "Intro"
            elif i == n - 1:
                stype = "Outro"
        out.append({**seg, "section_type": stype})
    return out


def _cluster_and_label_segments(
    chroma: np.ndarray,
    mfcc: np.ndarray,
    frame_times: np.ndarray,
    boundary_times: list[float],
    n_clusters: int,
    min_seg_dur: float,
    total_dur: float,
    auto_n_clusters: bool,
    use_mfcc: bool,
    energy_curve: np.ndarray | None = None,
    boundary_metadata: list[dict] | None = None,
    ssm_thresh: np.ndarray | None = None,
) -> list[dict]:
    """
    Build segments from boundary times, label by SSM repetition similarity
    (KMeans on feature descriptors as fallback), assign letter labels
    (A, B, C …), then assign section types.
    """
    fallback = [{"start": 0.0, "end": round(total_dur, 2), "label": "A", "section_type": "FullTrack"}]
    if frame_times.size == 0:
        return fallback

    feat_all = np.concatenate([chroma, mfcc], axis=0) if (use_mfcc and np.any(mfcc != 0)) else chroma
    if energy_curve is not None and energy_curve.size == frame_times.size:
        feat_all = np.concatenate([feat_all, energy_curve.reshape(1, -1)], axis=0)

    all_times = np.unique(np.clip(
        np.concatenate([[0.0], boundary_times, [total_dur]]), 0.0, total_dur
    ))

    seg_vecs: list[np.ndarray] = []
    seg_spans: list[tuple[float, float]] = []

    for j in range(len(all_times) - 1):
        t0, t1 = float(all_times[j]), float(all_times[j + 1])
        if t1 - t0 < 0.1:
            continue
        mask = (frame_times >= t0) & (frame_times < t1)
        if not mask.any():
            idx = int(np.argmin(np.abs(frame_times - (t0 + t1) / 2)))
            mask = np.zeros(frame_times.size, dtype=bool)
            mask[idx] = True
        seg_vecs.append(_segment_feature_vector(feat_all[:, mask]))
        seg_spans.append((t0, t1))

    if not seg_vecs:
        return fallback

    labels_arr: np.ndarray | None = None
    if ssm_thresh is not None:
        labels_arr = _ssm_segment_labels(
            ssm_thresh, frame_times, seg_spans,
            fixed_k=None if auto_n_clusters else min(n_clusters, len(seg_spans)),
        )
        if labels_arr is not None:
            logger.info("SSM-based labels: k=%d, n_segs=%d", len(set(labels_arr)), len(seg_spans))

    if labels_arr is None:
        X = np.array(seg_vecs, dtype=np.float32)
        k = _select_n_clusters(X, max_k=min(8, len(seg_vecs))) if auto_n_clusters else min(n_clusters, len(seg_vecs))
        logger.info("KMeans fallback: n_clusters=%d (auto=%s, n_segs=%d)", k, auto_n_clusters, len(seg_vecs))

        if k < 2 or len(seg_vecs) < 2:
            labels_arr = np.zeros(len(seg_vecs), dtype=int)
        else:
            try:
                km = KMeans(n_clusters=k, random_state=0, n_init=5, init='k-means++')
                labels_arr = km.fit_predict(X)
            except Exception as exc:
                logger.warning("KMeans failed (%s); single cluster.", exc)
                labels_arr = np.zeros(len(seg_vecs), dtype=int)

    counts = Counter(int(lbl) for lbl in labels_arr)
    id_to_char = {cid: chr(65 + i) for i, (cid, _) in enumerate(counts.most_common())}

    boundary_metadata = boundary_metadata or []

    def _boundary_context(t0: float, t1: float) -> tuple[float, list[str]]:
        nearby = [
            b for b in boundary_metadata
            if abs(float(b.get("time", -9999.0)) - t0) <= 0.35
            or abs(float(b.get("time", -9999.0)) - t1) <= 0.35
        ]
        if not nearby:
            return 0.5, []
        confidence = float(np.mean([float(b.get("confidence", 0.5)) for b in nearby]))
        sources = sorted({src for b in nearby for src in b.get("sources", [])})
        return round(confidence, 3), sources

    raw_segs = []
    for (t0, t1), lbl in zip(seg_spans, labels_arr):
        confidence, sources = _boundary_context(t0, t1)
        raw_segs.append({
            "start": round(t0, 2),
            "end": round(t1, 2),
            "label": id_to_char[int(lbl)],
            "confidence": confidence,
            "source_features": sources,
        })

    enforced = _enforce_min_segment_duration(raw_segs, min_dur=min_seg_dur, total_dur=total_dur)
    return _assign_section_types(enforced, total_dur)


# ---------------------------------------------------------------------------
# Dynamic weight adaptation
# ---------------------------------------------------------------------------

def _novelty_snr(curve: np.ndarray) -> float:
    """Confidence proxy: how peaked is a novelty curve (0=flat, 1=sharp peaks).

    Maps the 90th-percentile / mean ratio to [0, 1]:
      ratio ≤ 1  →  0.0  (completely flat)
      ratio = 2  →  0.5
      ratio ≥ 3  →  1.0  (very distinct peaks)
    """
    if curve.size < 4:
        return 0.0
    mean = float(np.mean(curve))
    if mean < 1e-8:
        return 0.0
    ratio = float(np.percentile(curve, 90)) / mean
    return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))


def _compute_dynamic_weights(
    static_weights: dict[str, float],
    rms_novelty: np.ndarray,
    onset_novelty: np.ndarray,
    chord_novelty: np.ndarray,
    ssm_novelty: np.ndarray,
    beat_times: np.ndarray,
    lyric_candidates: list[dict],
    core_dur: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Scale static weights by per-source signal quality, then renormalize.

    Returns (dynamic_weights, source_confidences) so diagnostics can log both.
    """
    conf: dict[str, float] = {
        "ssm":         _novelty_snr(ssm_novelty),
        "rms":         _novelty_snr(rms_novelty),
        "onset_flux":  _novelty_snr(onset_novelty),
        "chord_proxy": _novelty_snr(chord_novelty),
        "beat":        _beat_regularity(beat_times),
        "lyrics":      _lyrics_confidence(lyric_candidates, core_dur),
    }

    scaled = {k: static_weights.get(k, 0.0) * conf.get(k, 0.5) for k in static_weights}
    total = sum(scaled.values())
    if total <= 1e-8:
        return dict(static_weights), conf
    dynamic = {k: v / total for k, v in scaled.items()}
    return dynamic, conf


def _beat_regularity(beat_times: np.ndarray) -> float:
    """1 − (IBI std / IBI mean): 1 = perfect grid, 0 = chaotic timing."""
    if beat_times.size < 4:
        return 0.1
    ibi = np.diff(beat_times)
    mean = float(np.mean(ibi))
    if mean < 1e-8:
        return 0.0
    return float(np.clip(1.0 - np.std(ibi) / mean, 0.0, 1.0))


def _lyrics_confidence(lyric_candidates: list[dict], core_dur: float) -> float:
    """Scale with lyric density (candidates per 30 s), capped at 1."""
    if not lyric_candidates:
        return 0.0
    density = len(lyric_candidates) / max(1.0, core_dur / 30.0)
    return float(np.clip(density, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API — called by custom_worker.py
# ---------------------------------------------------------------------------

def process_file_path(file_path: str, params: dict | None = None) -> dict:
    """
    Worker entry point — reads audio from disk and runs the full pipeline.

    Returns
    -------
    {
        "filename": str,
        "duration_seconds": float,
        "segments": [{start, end, label, section_type}, ...]
    }
    """
    logger.info("Starting analysis for file: %s", file_path)
    try:
        with open(file_path, "rb") as fh:
            content = fh.read()
        filename = file_path.rsplit("/", 1)[-1]
        return _analyze_content(content, filename, content_type="audio/wav", params=params)
    except Exception:
        logger.error("Error processing %s", file_path, exc_info=True)
        raise


def _analyze_content(
    content: bytes,
    filename: str,
    content_type: str = "audio/wav",
    params: dict | None = None,
) -> dict:
    """
    Core segmentation pipeline.

    Accepted params keys (all optional):
      min_segment_duration_seconds  float  default 10.0
      novelty_kernel_size_seconds   float  default 8.0  (kernel half-size)
      target_fps                    float  default 10.0
      n_clusters                    int    default 4
      use_mfcc                      bool   default True
      mfcc_n_components             int    default 20   (MFCC0 dropped internally)
      auto_n_clusters               bool   default True
      smoothing_L                   int    default 14   (diagonal-smooth frames)
      transposition_invariant       bool   default True
      novelty_prominence            float  default 0.18
      use_beat_sync                 bool   default True (snap boundaries to beats)
      feature_weights               dict   optional multi-feature fusion weights
      timed_lyrics                  list   optional [{time_seconds, text}, ...]
      return_diagnostics            bool   default False
    """
    t_total = time.perf_counter()
    p = params or {}

    min_seg_dur  = float(p.get("min_segment_duration_seconds", _MIN_SEG_DUR))
    kernel_s     = float(p.get("novelty_kernel_size_seconds",  _KERNEL_SECONDS))
    target_fps   = float(p.get("target_fps",                   _TARGET_FPS))
    n_clusters   = int(  p.get("n_clusters",                   _N_CLUSTERS))
    use_mfcc     = bool( p.get("use_mfcc",                     True))
    n_mfcc       = int(  p.get("mfcc_n_components",            _MFCC_N))
    auto_k       = bool( p.get("auto_n_clusters",               True))
    smoothing_L  = int(  p.get("smoothing_L",                  _SMOOTHING_L))
    ti           = bool( p.get("transposition_invariant",       True))
    prominence   = float(p.get("novelty_prominence",            _PROMINENCE))
    use_beat_sync = bool(p.get("use_beat_sync",                 True))
    return_diagnostics = bool(p.get("return_diagnostics",       False))
    feature_weights = normalise_feature_weights(
        p.get("feature_weights"),
        p.get("spectral_flux_weight"),
    )

    # --- Stage 0: Load audio ---
    t0 = time.perf_counter()
    y, sr = _load_audio_from_bytes(content)
    original_dur = float(librosa.get_duration(y=y, sr=sr))
    logger.info("[%.2fs] Audio loaded: sr=%d, duration=%.2fs",
                time.perf_counter() - t0, sr, original_dur)

    # --- Stage 0b: Active region ---
    t0 = time.perf_counter()
    act_start, act_end = _detect_active_region(y, sr, hop_length=_HOP_LENGTH)
    core_dur = max(0.0, act_end - act_start)
    logger.info("[%.2fs] Active region: %.2fs–%.2fs (%.2fs)",
                time.perf_counter() - t0, act_start, act_end, core_dur)

    if core_dur <= 0.0:
        return _empty_result(filename, content_type, original_dur, "No active music region detected.")

    y_active = y[int(act_start * sr): int(act_end * sr)].astype(np.float32)

    # --- Stage 1: Feature extraction + downsampling ---
    t0 = time.perf_counter()
    chroma, mfcc, frame_times, fps = _extract_downsampled_features(
        y_active, sr,
        hop_length=_HOP_LENGTH,
        target_fps=target_fps,
        n_mfcc=n_mfcc,
        use_mfcc=use_mfcc,
    )
    logger.info("[%.2fs] Features: n_frames=%d, fps=%.2f", time.perf_counter() - t0, chroma.shape[1], fps)

    # Downsample further if track is very long.  Median-pool rather than
    # stride-decimate: plain `[::step]` aliases the already-pooled features.
    N = chroma.shape[1]
    if N > _MAX_SSM_FRAMES:
        step = int(np.ceil(N / _MAX_SSM_FRAMES))
        chroma = _median_pool(chroma, step)
        mfcc   = _median_pool(mfcc,   step)
        # Pooling breaks the unit norm required for cosine SSMs — re-normalise.
        for feat in (chroma, mfcc):
            norms = np.linalg.norm(feat, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            feat /= norms
        n_pooled = chroma.shape[1]
        frame_times = frame_times[: n_pooled * step].reshape(n_pooled, step).mean(axis=1)
        fps /= step
        logger.info("Further downsampled: n_frames=%d (step=%d)", chroma.shape[1], step)

    # --- Stage 2+3: Parallel candidate extraction + SSM build ---
    # All six tasks depend only on features from Stage 1 and are independent
    # of each other, so they run concurrently. Numpy/librosa release the GIL
    # during FFT and BLAS operations, so threads get real overlap.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as _pool:
        _fut_tempo  = _pool.submit(tempo_and_beats, y_active, sr, hop_length=_HOP_LENGTH)
        _fut_rms    = _pool.submit(
            rms_boundary_candidates,
            y_active, sr, frame_times, fps,
            min_seg_dur=min_seg_dur, total_dur=core_dur, hop_length=_HOP_LENGTH,
        )
        _fut_onset  = _pool.submit(
            onset_boundary_candidates,
            y_active, sr, frame_times, fps,
            min_seg_dur=min_seg_dur, total_dur=core_dur, hop_length=_HOP_LENGTH,
        )
        _fut_chord  = _pool.submit(
            chord_proxy_boundary_candidates,
            chroma, frame_times, fps,
            min_seg_dur=min_seg_dur, total_dur=core_dur,
        )
        _fut_lyrics = _pool.submit(
            lyrics_boundary_candidates,
            p.get("timed_lyrics"),
            active_start=act_start, total_dur=core_dur, min_seg_dur=min_seg_dur,
        )
        _fut_ssm    = _pool.submit(
            _build_combined_ssm, chroma, mfcc,
            use_mfcc=use_mfcc, transposition_invariant=ti,
        )

    estimated_bpm, beat_times                            = _fut_tempo.result()
    rms_candidates, rms_novelty                          = _fut_rms.result()
    onset_candidates, onset_novelty, onset_times, onset_env = _fut_onset.result()
    chord_candidates, chord_novelty                      = _fut_chord.result()
    lyric_candidates                                     = _fut_lyrics.result()
    S_raw                                                = _fut_ssm.result()

    all_candidates: list[dict] = (
        rms_candidates + onset_candidates + chord_candidates + lyric_candidates
    )
    logger.info(
        "[%.2fs] Parallel stage: bpm=%.2f rms=%d onset=%d chord=%d lyrics=%d ssm_shape=%s",
        time.perf_counter() - t0,
        estimated_bpm, len(rms_candidates), len(onset_candidates),
        len(chord_candidates), len(lyric_candidates), S_raw.shape,
    )

    # --- Stage 4: Enhance SSM ---
    # Checkerboard novelty runs on the *smoothed* SSM (kept in cosine range);
    # the thresholded/sparsified version is only used where sparsity helps:
    # structure features and segment-pair similarity (FMP Eq. 4.17 is a DP
    # scoring device, not a novelty preprocessing step).
    t0 = time.perf_counter()
    S_smooth = _smooth_ssm(S_raw, L=smoothing_L)
    S_thresh = _threshold_ssm(S_smooth, rho=_SSM_RHO, delta=0.0)
    logger.info("[%.2fs] SSM smoothed+thresholded (L=%d)", time.perf_counter() - t0, smoothing_L)

    # --- Stage 5: SSM novelty candidates ---
    t0 = time.perf_counter()
    kernel_L = max(8, int(kernel_s * fps))
    s_min, s_max = float(S_smooth.min()), float(S_smooth.max())
    S_unit = (S_smooth - s_min) / max(s_max - s_min, 1e-8)
    checkerboard_novelty = _compute_novelty_ssm(S_unit, L=kernel_L)
    structure_novelty = _structure_feature_novelty(S_thresh)
    ssm_novelty = (
        (1.0 - _STRUCTURE_WEIGHT) * checkerboard_novelty
        + _STRUCTURE_WEIGHT * structure_novelty
    )
    max_nov = float(ssm_novelty.max())
    if max_nov > 0:
        ssm_novelty = (ssm_novelty / max_nov).astype(np.float32)
    logger.info("[%.2fs] Novelty (kernel_L=%d frames = %.1fs, structure_w=%.2f)",
                time.perf_counter() - t0, kernel_L, kernel_L / fps, _STRUCTURE_WEIGHT)

    t0 = time.perf_counter()
    ssm_boundaries = find_boundaries(
        ssm_novelty, fps,
        min_segment_s=min_seg_dur,
        total_dur=core_dur,
        prominence=prominence,
        novelty_sigma=_NOVELTY_SIGMA,
    )
    ssm_candidates = candidates_from_boundaries(ssm_boundaries, "ssm", ssm_novelty, frame_times)
    all_candidates.extend(ssm_candidates)
    logger.info("[%.2fs] SSM candidates=%d", time.perf_counter() - t0, len(ssm_candidates))

    # Beat-phrase grid candidates: sections tend to start on 16/24/32-beat
    # phrase boundaries; the SSM novelty curve phase-locks the grid.
    beat_candidates = beat_phrase_boundary_candidates(
        beat_times, onset_times, onset_env,
        total_dur=core_dur, min_seg_dur=min_seg_dur,
        support_times=frame_times, support_curve=ssm_novelty,
    )
    all_candidates.extend(beat_candidates)
    logger.info("Beat-phrase candidates=%d", len(beat_candidates))

    # --- Stage 5b: Dynamic weight adaptation ---
    dynamic_weights, source_confidences = _compute_dynamic_weights(
        static_weights=feature_weights,
        rms_novelty=rms_novelty,
        onset_novelty=onset_novelty,
        chord_novelty=chord_novelty,
        ssm_novelty=ssm_novelty,
        beat_times=beat_times,
        lyric_candidates=lyric_candidates,
        core_dur=core_dur,
    )
    logger.info(
        "Dynamic weights — ssm=%.2f rms=%.2f onset=%.2f chord=%.2f beat=%.2f lyrics=%.2f",
        dynamic_weights.get("ssm", 0), dynamic_weights.get("rms", 0),
        dynamic_weights.get("onset_flux", 0), dynamic_weights.get("chord_proxy", 0),
        dynamic_weights.get("beat", 0), dynamic_weights.get("lyrics", 0),
    )

    # --- Stage 6: Filtering, fusion, and snapping ---
    t0 = time.perf_counter()
    # One boundary per ~9s on average: SALAMI coarse sections are mostly
    # 10-25s, but intros/outros and transitions are shorter — a 12s prior
    # capped recall hard.
    boundary_budget = max(1, int(core_dur / 9.0) - 1)
    fused_boundaries = fuse_boundary_candidates(
        all_candidates,
        weights=dynamic_weights,
        total_dur=core_dur,
        min_seg_dur=min_seg_dur,
        max_boundaries=boundary_budget,
    )
    snapped_boundaries = snap_fused_boundaries(
        fused_boundaries,
        onset_times=onset_times,
        onset_env=onset_env,
        beat_times=beat_times,
        use_beat_sync=use_beat_sync,
        onset_window_s=_ONSET_SNAP_WINDOW,
    )

    if not snapped_boundaries and ssm_boundaries:
        fallback = [{"time": t, "sources": ["ssm"], "confidence": 0.5} for t in ssm_boundaries]
        fallback = fallback[:boundary_budget]
        snapped_boundaries = snap_fused_boundaries(
            fallback,
            onset_times=onset_times,
            onset_env=onset_env,
            beat_times=beat_times,
            use_beat_sync=use_beat_sync,
            onset_window_s=_ONSET_SNAP_WINDOW,
        )

    boundary_times_core = [float(b["time"]) for b in snapped_boundaries]
    logger.info("[%.2fs] Fused boundaries=%d from candidates=%d",
                time.perf_counter() - t0, len(boundary_times_core), len(all_candidates))

    # --- Stage 7: Cluster & label ---
    t0 = time.perf_counter()
    segments_core = _cluster_and_label_segments(
        chroma, mfcc, frame_times,
        boundary_times=boundary_times_core,
        n_clusters=n_clusters,
        min_seg_dur=min_seg_dur,
        total_dur=core_dur,
        auto_n_clusters=auto_k,
        use_mfcc=use_mfcc,
        energy_curve=rms_novelty,
        boundary_metadata=snapped_boundaries,
        ssm_thresh=S_thresh,
    )
    logger.info("[%.2fs] %d segments after clustering", time.perf_counter() - t0, len(segments_core))

    # --- Stage 8: Shift back to full-track timeline ---
    for seg in segments_core:
        seg["start"] = round(seg["start"] + act_start, 2)
        seg["end"]   = round(seg["end"]   + act_start, 2)

    candidate_boundaries = []
    for item in snapped_boundaries:
        candidate_boundaries.append({
            "time": round(float(item["time"]) + act_start, 2),
            "source": item.get("sources", []),
            "confidence": item.get("confidence", 0.5),
        })

    logger.info("Total pipeline: %.2fs for %s", time.perf_counter() - t_total, filename)

    result = {
        "filename": filename,
        "content_type": content_type,
        "duration_seconds": round(original_dur, 2),
        "estimated_bpm": estimated_bpm,
        "candidate_boundaries": candidate_boundaries,
        "segments": segments_core,
        "status": "Segmentation complete.",
    }
    if return_diagnostics:
        result["diagnostics"] = {
            "active_region": {"start": round(act_start, 3), "end": round(act_end, 3)},
            "static_feature_weights": {k: round(v, 4) for k, v in feature_weights.items()},
            "source_confidences": {k: round(v, 4) for k, v in source_confidences.items()},
            "dynamic_feature_weights": {k: round(v, 4) for k, v in dynamic_weights.items()},
            "raw_candidate_counts": {
                "rms": len(rms_candidates),
                "onset_flux": len(onset_candidates),
                "chord_proxy": len(chord_candidates),
                "lyrics": len(lyric_candidates),
                "ssm": len(ssm_candidates),
            },
            "fused_boundary_count": len(snapped_boundaries),
        }
    return result


def _empty_result(filename: str, content_type: str, duration: float, msg: str) -> dict:
    return {
        "filename": filename,
        "content_type": content_type,
        "duration_seconds": round(duration, 2),
        "segments": [],
        "status": msg,
    }
