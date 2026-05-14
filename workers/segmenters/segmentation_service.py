"""
Music segmentation service — FMP Chapter 4 production implementation.

Pipeline (strictly in order):
  Audio load
  → Beat tracking (librosa.beat.beat_track)
  → Beat-synchronous feature extraction
      · Chroma-CENS  (harmonic — FMP Section 4.1.3)
      · MFCC         (timbral  — FMP Section 4.1.3)
  → L2 normalization per beat-frame
  → Enhanced SSM
      · Transposition-invariant (FMP Eq. 4.15): cell-wise max over 12 cyclic chroma shifts
      · Diagonal smoothing (FMP Eq. 4.11): length-L window along main diagonal
      · Tempo-invariant smoothing (FMP Eq. 4.12-4.13): Θ = {0.66, 0.81, 1.0, 1.22, 1.50}
      · Global thresholding with penalty (FMP Eq. 4.17): keep top ρ=20%, rest → δ=-2
  → Novelty curve (FMP Eq. 4.38-4.43)
      · Gaussian checkerboard kernel, sliding along diagonal only — O(N·L²)
      · Blend with spectral flux onset strength
  → Peak picking
      · Adaptive delta (60th-percentile-based)
      · Minimum distance = min_segment_duration_seconds
  → Segment clustering
      · Features per segment: mean + delta_mean + std
      · Auto k via silhouette score (2..8)
      · KMeans with k-means++ init
  → Section type assignment (Intro, Verse, Chorus, Bridge, Outro)
  → Active region detection + timeline offset correction
"""

from __future__ import annotations

import io
import time
from collections import Counter

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from shared.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Module-level constants — all tunable via the params dict at call time.
# ---------------------------------------------------------------------------

# Sample rate. 22050 Hz is sufficient for structural analysis; higher rates
# double frame count with no meaningful benefit for section-level features.
_SR: int = 22050

# Hop length for raw audio feature extraction (before beat sync).
# 512 samples @ 22050 Hz ≈ 23 ms/frame — standard librosa default.
_HOP_LENGTH: int = 512

# Default diagonal-smoothing window in beats (FMP Eq. 4.11).
# 10 beats ≈ 2.5 bars at 120 BPM.  15 was over-smoothing short boundaries.
_SMOOTHING_L: int = 10

# Tempo-invariant smoothing ratios Θ — covers ±50 % tempo variation.
# FMP Section 4.2.2.2, Eq. 4.13.
_TEMPO_RATIOS: list[float] = [0.66, 0.81, 1.0, 1.22, 1.50]

# Top fraction of SSM cells kept after thresholding (FMP Eq. 4.17).
# ρ=0.20 keeps the top 20 %; the rest get penalty δ=−2.
_SSM_RHO: float = 0.20
_SSM_PENALTY: float = -2.0

# Novelty kernel half-size in seconds.  4 s targets section-level boundaries.
_NOVELTY_KERNEL_SECONDS: float = 4.0

# Blend weight for spectral-flux novelty (0 = pure SSM, 1 = pure flux).
# 0.15: enough flux signal for harmonically uniform genres without flooding
# the novelty curve with onset-level false positives.
_SPECTRAL_FLUX_WEIGHT: float = 0.15

# Minimum segment duration (seconds) — prevents musically nonsensical micro-segments.
# 8 s ≈ 2 bars at 120 BPM; SALAMI sections are rarely shorter than this.
_MIN_SEG_DUR: float = 8.0

# Default cluster count for segment labelling.
_N_CLUSTERS: int = 4

# Number of MFCC coefficients (FMP Section 4.1.3).
_MFCC_N: int = 20

# Active-region detection margin and minimum length.
_ACTIVE_MARGIN_DB: float = 20.0
_ACTIVE_MIN_S: float = 3.0

# Hard cap on beats fed into the SSM.  Beat-sync already dramatically reduces
# frame count (120 BPM × 4 min ≈ 480 beats); cap handles very long tracks.
_MAX_SSM_BEATS: int = 2000


# ---------------------------------------------------------------------------
# Stage 0 — Audio loading
# ---------------------------------------------------------------------------

def _load_audio_from_bytes(content: bytes, sr: int = _SR) -> tuple[np.ndarray, int]:
    """Load audio from an in-memory byte buffer, resampling to `sr`."""
    try:
        y, sr_out = librosa.load(io.BytesIO(content), sr=sr, mono=True)
        return y.astype(np.float32), sr_out
    except Exception as exc:
        logger.error("Failed to load audio from bytes: %s", exc, exc_info=True)
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

    Uses frame-wise RMS → dB, dynamic threshold = P75 − margin_db.
    Falls back to (0.0, duration) when no clear active region is found.
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
# Stage 1 — Beat tracking + beat-synchronous feature extraction
# ---------------------------------------------------------------------------

def _extract_beat_sync_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = _HOP_LENGTH,
    n_mfcc: int = _MFCC_N,
    use_mfcc: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Beat-synchronous Chroma-CENS and (optionally) MFCC features.

    Beat-sync features compensate for intra-section tempo fluctuation —
    FMP Section 4.1.3, "beat-level feature sequences".

    Returns
    -------
    chroma_beat : (12, N_beats) float32, L2-normalised
    mfcc_beat   : (n_mfcc, N_beats) float32, L2-normalised  (zeros if use_mfcc=False)
    beat_times  : (N_beats,) seconds of each beat
    beat_rate   : beats per second (used to convert kernel_seconds → kernel_beats)
    """
    try:
        tempo_arr, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length, trim=False
        )
        if beat_frames.size < 4:
            raise ValueError(f"too few beats ({beat_frames.size})")
    except Exception as exc:
        logger.warning("Beat tracking failed (%s); falling back to uniform grid.", exc)
        # Fallback: 2 Hz uniform grid (≈ 120 BPM / 2 half-bars)
        target_fps = 2.0
        step = max(1, int(sr / hop_length / target_fps))
        n_frames = int(np.ceil(len(y) / hop_length))
        beat_frames = np.arange(0, n_frames, step, dtype=int)

    # Clamp beat_frames to valid range
    n_raw_frames = int(np.ceil(len(y) / hop_length))
    beat_frames = np.clip(beat_frames, 0, n_raw_frames - 1)
    beat_frames = np.unique(beat_frames).astype(int)

    beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).astype(np.float32)

    # --- Chroma-CENS (FMP Section 4.1.3 — robust to dynamics & timbre) ---
    try:
        chroma_raw = librosa.feature.chroma_cens(
            y=y, sr=sr, hop_length=hop_length, n_chroma=12
        ).astype(np.float32)
    except Exception as exc:
        logger.warning("chroma_cens failed (%s); using chroma_cqt fallback.", exc)
        chroma_raw = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length
        ).astype(np.float32)

    # librosa.util.sync(pad=True) prepends an extra segment [0, beat_frames[0]) ONLY
    # when beat_frames[0] > 0.  Mirror that logic so beat_times always has the same
    # length as the sync output's column count.
    chroma_beat = librosa.util.sync(chroma_raw, beat_frames, aggregate=np.median).astype(np.float32)
    if beat_frames[0] > 0:
        beat_times = np.concatenate([[0.0], beat_times_raw]).astype(np.float32)
    else:
        beat_times = beat_times_raw.astype(np.float32)
    # Safety clip — should be a no-op, but guards against librosa version quirks.
    beat_times = beat_times[: chroma_beat.shape[1]]

    # L2-normalise each beat-frame column  (FMP Section 4.2.1)
    norms = np.linalg.norm(chroma_beat, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    chroma_beat /= norms

    # --- MFCC (timbral) ---
    if use_mfcc:
        try:
            mfcc_raw = librosa.feature.mfcc(
                y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("MFCC extraction failed (%s); disabling MFCC.", exc)
            mfcc_raw = np.zeros((n_mfcc, n_raw_frames), dtype=np.float32)

        mfcc_beat = librosa.util.sync(mfcc_raw, beat_frames, aggregate=np.median).astype(np.float32)
        norms_m = np.linalg.norm(mfcc_beat, axis=0, keepdims=True)
        norms_m[norms_m == 0] = 1.0
        mfcc_beat /= norms_m
    else:
        mfcc_beat = np.zeros((n_mfcc, beat_times.size), dtype=np.float32)

    duration = float(librosa.get_duration(y=y, sr=sr))
    beat_rate = beat_times.size / duration if duration > 0 else 2.0

    logger.debug(
        "Beat sync: n_beats=%d, beat_rate=%.2f bps, chroma=%s, mfcc=%s",
        beat_times.size, beat_rate, chroma_beat.shape, mfcc_beat.shape,
    )
    return chroma_beat, mfcc_beat, beat_times, float(beat_rate)


# ---------------------------------------------------------------------------
# Stage 2 — SSM construction (raw cosine + transposition invariance)
# ---------------------------------------------------------------------------

def _compute_ti_chroma_ssm(chroma: np.ndarray) -> np.ndarray:
    """
    Transposition-invariant chroma SSM — FMP Section 4.2.2.3, Eq. 4.15.

    S_TI[n, m] = max_{c=0..11}  <roll_c(chroma_n),  chroma_m>

    Cyclic shift by c semitones captures repetitions in a different key.
    Complexity: O(12 × N²) — typically 12 × 500² = 3 M ops, fast enough.
    """
    N = chroma.shape[1]
    S_TI = np.full((N, N), -np.inf, dtype=np.float32)
    for c in range(12):
        chroma_shift = np.roll(chroma, c, axis=0)          # (12, N)
        S_c = (chroma_shift.T @ chroma).astype(np.float32)  # (N, N)
        np.maximum(S_TI, S_c, out=S_TI)
    return S_TI


def _compute_raw_ssm(feat: np.ndarray) -> np.ndarray:
    """
    Standard cosine SSM from an L2-normalised feature matrix.

    feat: (D, N) — already L2-normalised per column.
    Returns (N, N) float32 in [-1, 1].
    """
    F = feat.T.astype(np.float32)   # (N, D)
    return (F @ F.T).astype(np.float32)


def _build_combined_ssm(
    chroma: np.ndarray,
    mfcc: np.ndarray,
    use_mfcc: bool,
    transposition_invariant: bool,
) -> np.ndarray:
    """
    Build the combined SSM from chroma (± TI) and optional MFCC.

    Chroma captures harmonic repetition (verse/chorus identity).
    MFCC captures timbral texture (instrumentation, density).
    Equal-weight blend of the two gives a balanced view.
    """
    if transposition_invariant:
        S_chroma = _compute_ti_chroma_ssm(chroma)
    else:
        S_chroma = _compute_raw_ssm(chroma)

    if use_mfcc and mfcc is not None and np.any(mfcc != 0):
        S_mfcc = _compute_raw_ssm(mfcc)
        S = 0.5 * S_chroma + 0.5 * S_mfcc
    else:
        S = S_chroma

    return S.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 3 — SSM enhancement
# ---------------------------------------------------------------------------

def _diagonal_smooth_theta(S: np.ndarray, L: int, theta: float) -> np.ndarray:
    """
    Smooth S along the direction (1, theta) — FMP Eq. 4.12.

    S_L[n, m] = (1/L) * Σ_{l=0}^{L-1}  S[n-l,  m-floor(l*theta)]

    For theta=1.0 this reduces to standard diagonal smoothing (Eq. 4.11).
    Each l shifts both row and col back by (l, floor(l·theta)).
    """
    N = S.shape[0]
    S_out = np.zeros((N, N), dtype=np.float32)
    count = 0
    for l in range(L):
        r_sh = l
        c_sh = int(round(l * theta))
        if r_sh >= N or c_sh >= N:
            break
        r_len = N - r_sh
        c_len = N - c_sh
        # S_out[n, m] += S[n-r_sh, m-c_sh]  →  index shift in output coordinates:
        # S_out[r_sh:, c_sh:] += S[:r_len, :c_len]
        S_out[r_sh:, c_sh:] += S[:r_len, :c_len]
        count += 1
    return S_out / max(count, 1)


def _compute_enhanced_ssm(
    S_raw: np.ndarray,
    L: int,
    tempo_ratios: list[float] = _TEMPO_RATIOS,
    rho: float = _SSM_RHO,
    penalty: float = _SSM_PENALTY,
) -> np.ndarray:
    """
    Full SSM enhancement: diagonal smoothing → tempo-invariance → threshold.

    FMP Section 4.2.2:
      1. Smooth at each tempo ratio θ in Θ (Eq. 4.12-4.13).
      2. Also smooth the transposed SSM (captures reverse-direction paths).
      3. Take cell-wise max over all 2·|Θ| versions (Eq. 4.13).
      4. Global threshold: top ρ cells normalized to [0,1];
         remainder set to penalty δ (Eq. 4.17).
    """
    versions: list[np.ndarray] = []
    for theta in tempo_ratios:
        Sf = _diagonal_smooth_theta(S_raw, L, theta)
        Sb = _diagonal_smooth_theta(S_raw.T, L, theta).T   # reverse direction
        versions.append(np.maximum(Sf, Sb))

    S_smooth = np.max(np.stack(versions, axis=0), axis=0).astype(np.float32)
    np.fill_diagonal(S_smooth, 1.0)  # smoothing can erode the main diagonal

    # Global thresholding with penalty (FMP Eq. 4.17)
    thresh = float(np.percentile(S_smooth, (1.0 - rho) * 100.0))
    denom = max(1.0 - thresh, 1e-8)
    S_enh = np.where(
        S_smooth >= thresh,
        (S_smooth - thresh) / denom,
        penalty,
    ).astype(np.float32)
    np.fill_diagonal(S_enh, 1.0)

    return S_enh


# ---------------------------------------------------------------------------
# Stage 4 — Novelty curve (correct O(N·L²) implementation)
# ---------------------------------------------------------------------------

def _compute_novelty_ssm(
    S: np.ndarray,
    L: int,
    gamma: float = 10.0,
) -> np.ndarray:
    """
    Gaussian checkerboard kernel novelty — FMP Section 4.4.1, Eq. 4.38-4.43.

    For each frame n, extract the (2L+1)×(2L+1) patch centred on (n, n) and
    compute the inner product with kernel K.  Complexity: O(N × (2L+1)²) —
    not O(N⁴) as would result from correlate2d on the full NxN matrix followed
    by np.diag().

    Parameters
    ----------
    L     : kernel half-size in beats
    gamma : Gaussian tapering strength (ε = gamma/L in FMP Eq. 4.41)
    """
    M = 2 * L + 1
    k = np.arange(-L, L + 1, dtype=np.float32)
    kk, ll = np.meshgrid(k, k, indexing='ij')   # (M, M)

    # Checkerboard pattern (FMP Eq. 4.38)
    K = (np.sign(kk) * np.sign(ll)).astype(np.float32)

    # Gaussian tapering (FMP Eq. 4.40-4.41)
    eps = gamma / max(L, 1)
    K *= np.exp(-(eps ** 2) * (kk ** 2 + ll ** 2)).astype(np.float32)

    # Normalise (FMP Eq. 4.42)
    abs_sum = float(np.sum(np.abs(K)))
    if abs_sum > 0:
        K /= abs_sum

    N = S.shape[0]
    # Pad so the patch is always (M, M) regardless of position
    S_pad = np.pad(S.astype(np.float32), L, mode='constant', constant_values=0.0)

    novelty = np.empty(N, dtype=np.float32)
    for n in range(N):
        patch = S_pad[n: n + M, n: n + M]   # always (M, M)
        novelty[n] = float(np.dot(K.ravel(), patch.ravel()))

    # Rectify (FMP Eq. 4.43)
    novelty = np.maximum(novelty, 0.0)

    # Smooth slightly to reduce beat-level jitter
    if novelty.size > 5:
        novelty = gaussian_filter1d(novelty, sigma=1.0)

    max_val = float(novelty.max())
    if max_val > 0:
        novelty /= max_val

    return novelty


def _compute_spectral_flux_novelty(y: np.ndarray, sr: int, hop_length: int = _HOP_LENGTH) -> np.ndarray:
    """Onset strength envelope (spectral flux) normalised to [0, 1]."""
    try:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32)
        max_val = float(env.max())
        if max_val > 0:
            env /= max_val
        return env
    except Exception as exc:
        logger.warning("Spectral flux failed (%s); returning zeros.", exc)
        return np.zeros(int(np.ceil(len(y) / hop_length)), dtype=np.float32)


def _blend_novelty_curves(
    ssm_novelty: np.ndarray,
    flux_novelty: np.ndarray,
    beat_times: np.ndarray,
    y_len: int,
    sr: int,
    hop_length: int,
    flux_weight: float,
) -> np.ndarray:
    """
    Blend beat-indexed SSM novelty with raw-frame spectral-flux novelty.

    Flux novelty is resampled to beat grid via linear interpolation so both
    curves share the same length (N_beats).
    """
    if flux_weight <= 0.0 or flux_novelty.size == 0:
        return ssm_novelty

    # Time axis for flux novelty (raw frames)
    n_flux_frames = flux_novelty.size
    duration = y_len / sr
    flux_times = np.linspace(0.0, duration, n_flux_frames, endpoint=False, dtype=np.float32)

    # Resample flux to beat grid
    flux_resampled = np.interp(beat_times, flux_times, flux_novelty).astype(np.float32)

    blended = (1.0 - flux_weight) * ssm_novelty + flux_weight * flux_resampled
    max_val = float(blended.max())
    if max_val > 0:
        blended /= max_val
    return blended


# ---------------------------------------------------------------------------
# Stage 5 — Peak picking with musical constraints
# ---------------------------------------------------------------------------

def _find_boundaries(
    novelty: np.ndarray,
    beat_times: np.ndarray,
    min_segment_s: float,
    y_active: np.ndarray,
    sr: int,
    hop_length: int = _HOP_LENGTH,
) -> list[float]:
    """
    Pick boundary timestamps from the beat-indexed novelty curve.

    Adaptive delta: threshold = P60(novelty) × 0.10 — more sensitive than
    the previous P75 × 0.15 which was suppressing too many real boundaries.
    RMS-dip snapping removed: the ±1s snap window was consistently shifting
    boundaries 1–1.5s from reference (all mir_eval matches fell in that band).
    """
    if novelty.size == 0 or beat_times.size == 0:
        return []

    duration = float(beat_times[-1]) if beat_times.size > 0 else 0.0
    if duration == 0:
        return []

    beat_rate = beat_times.size / max(duration, 1.0)
    min_dist_beats = max(1, int(min_segment_s * beat_rate))

    # Narrow local-max window (0.5 s) → easier to qualify as a peak.
    # Local-avg window (3 s) — wider than the max window but not so wide that
    # it pulls the baseline up and suppresses real peaks.
    # wait = min_dist_beats enforces the hard minimum-segment constraint.
    pre_max  = max(1, int(beat_rate * 0.5))
    post_max = pre_max
    pre_avg  = max(4, int(beat_rate * 3.0))
    post_avg = pre_avg
    wait     = min_dist_beats

    # P75 × 0.20: peaks must stand 20% of the 75th-percentile value above their
    # local background. Stricter than the earlier P60×0.10 which produced
    # 3× too many boundaries on blues/jazz tracks.
    delta = max(0.05, float(np.percentile(novelty, 75)) * 0.20)

    try:
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait,
        )
    except Exception as exc:
        logger.warning("peak_pick failed (%s); no boundaries detected.", exc)
        return []

    if len(peaks) == 0:
        return []

    # Convert beat indices to seconds
    boundary_times = [float(beat_times[p]) for p in peaks if p < beat_times.size]

    # Deduplicate and enforce minimum distance between consecutive boundaries
    boundary_times = sorted(set(round(t, 3) for t in boundary_times))
    filtered: list[float] = []
    for t in boundary_times:
        if not filtered or (t - filtered[-1]) >= min_segment_s:
            filtered.append(t)

    # Drop boundaries only when they would create a segment shorter than half
    # the minimum — the full min_segment_s margin was removing legitimate
    # boundaries near track edges.
    track_end = float(beat_times[-1]) if beat_times.size > 0 else 0.0
    edge_margin = min_segment_s * 0.5
    filtered = [
        t for t in filtered
        if t >= edge_margin and (track_end - t) >= edge_margin
    ]

    return filtered


# ---------------------------------------------------------------------------
# Stage 6 — Segment clustering and labelling
# ---------------------------------------------------------------------------

def _select_n_clusters(X: np.ndarray, max_k: int = 8) -> int:
    """
    Auto-select k via silhouette score over range [2, min(max_k, n_samples-1)].

    Silhouette measures how well each point fits its cluster vs. the next-best;
    higher is better.  Cap at 6 — most popular music has ≤6 distinct sections.
    """
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


def _segment_feature_vector(feat: np.ndarray) -> np.ndarray:
    """
    Richer segment descriptor: mean ‖ delta_mean ‖ std.

    - mean      : average tonal/timbral colour of the segment
    - delta_mean: mean frame-to-frame change — distinguishes static vs. dynamic
    - std       : variance within the segment — dense vs. sparse texture

    FMP Section 4.4.3 motivation: mean-only representations lose dynamic
    information; delta + std makes clustering far more discriminative.
    """
    if feat.shape[1] < 2:
        mean = feat[:, 0] if feat.shape[1] == 1 else np.zeros(feat.shape[0])
        return np.concatenate([mean, np.zeros_like(mean), np.zeros_like(mean)])
    mean = np.mean(feat, axis=1).astype(np.float32)
    delta = np.mean(np.abs(np.diff(feat, axis=1)), axis=1).astype(np.float32)
    std = np.std(feat, axis=1).astype(np.float32)
    return np.concatenate([mean, delta, std])


def _merge_consecutive_same_labels(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments that share the same label."""
    if not segments:
        return segments
    merged: list[dict] = []
    for seg in segments:
        if merged and merged[-1]["label"] == seg["label"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def _enforce_min_segment_duration(
    segments: list[dict],
    min_dur: float,
    total_dur: float,
) -> list[dict]:
    """
    Merge short segments with their longer neighbour (iterative).
    """
    if not segments:
        return segments

    segs = [dict(s) for s in segments]
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
            # Too short — merge
            if i == 0 and len(segs) > 1:
                next_s = segs[i + 1]
                out.append({"start": seg["start"], "end": next_s["end"], "label": next_s["label"]})
                i += 2
            elif i == len(segs) - 1:
                out[-1]["end"] = seg["end"]
                i += 1
            else:
                left = out[-1]
                right = segs[i + 1]
                if (left["end"] - left["start"]) >= (right["end"] - right["start"]):
                    left["end"] = seg["end"]
                    i += 1
                else:
                    out.append({"start": seg["start"], "end": right["end"], "label": right["label"]})
                    i += 2
            changed = True
        segs = out

    if segs:
        segs[-1]["end"] = min(segs[-1]["end"], total_dur)
    return segs


def _assign_section_types(segments: list[dict], total_dur: float) -> list[dict]:
    """
    Heuristic section-type assignment.

    - Label with greatest cumulative duration → Chorus
    - Second → Verse
    - Others → Bridge / Other
    - First segment (when ≥4 total) → Intro
    - Last  segment (when ≥4 total) → Outro
    """
    if not segments:
        return segments

    dur_by_label: dict[str, float] = {}
    for s in segments:
        dur_by_label[s["label"]] = dur_by_label.get(s["label"], 0.0) + (s["end"] - s["start"])

    sorted_labels = [lbl for lbl, _ in sorted(dur_by_label.items(), key=lambda x: -x[1])]
    type_map: dict[str, str] = {}
    for i, lbl in enumerate(sorted_labels):
        if i == 0:
            type_map[lbl] = "Chorus"
        elif i == 1:
            type_map[lbl] = "Verse"
        elif i == 2:
            type_map[lbl] = "Bridge"
        else:
            type_map[lbl] = "Other"

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
    chroma_beat: np.ndarray,
    mfcc_beat: np.ndarray,
    beat_times: np.ndarray,
    boundary_times: list[float],
    n_clusters: int,
    min_seg_dur: float,
    total_dur: float,
    auto_n_clusters: bool,
    use_mfcc: bool,
) -> list[dict]:
    """
    Build segments from boundary times, cluster by rich feature vectors,
    assign letter labels (A, B, C …), then assign section types.
    """
    fallback = [{"start": 0.0, "end": round(total_dur, 2), "label": "A", "section_type": "FullTrack"}]

    if beat_times.size == 0:
        return fallback

    # Build boundary beat indices from boundary times
    all_times = np.concatenate([[0.0], boundary_times, [total_dur]])
    all_times = np.unique(np.clip(all_times, 0.0, total_dur))

    # Feature matrix to use for per-segment descriptors
    # Use chroma + MFCC (or just chroma) concatenated along feature axis
    if use_mfcc and np.any(mfcc_beat != 0):
        feat_all = np.concatenate([chroma_beat, mfcc_beat], axis=0)   # (12+n_mfcc, N)
    else:
        feat_all = chroma_beat   # (12, N)

    seg_vecs: list[np.ndarray] = []
    seg_spans: list[tuple[float, float]] = []

    for j in range(len(all_times) - 1):
        t0, t1 = float(all_times[j]), float(all_times[j + 1])
        if t1 - t0 < 0.1:
            continue

        # Beat indices that fall within [t0, t1)
        mask = (beat_times >= t0) & (beat_times < t1)
        if not mask.any():
            # If no beat falls in this slice (can happen at boundaries),
            # use the nearest single beat
            idx = int(np.argmin(np.abs(beat_times - (t0 + t1) / 2)))
            mask = np.zeros(beat_times.size, dtype=bool)
            mask[idx] = True

        seg_feat = feat_all[:, mask]
        seg_vecs.append(_segment_feature_vector(seg_feat))
        seg_spans.append((t0, t1))

    if not seg_vecs:
        return fallback

    X = np.array(seg_vecs, dtype=np.float32)

    if auto_n_clusters:
        k = _select_n_clusters(X, max_k=min(8, len(seg_vecs)))
        logger.info("Auto-selected n_clusters=%d via silhouette", k)
    else:
        k = min(n_clusters, len(seg_vecs))

    if k < 2 or len(seg_vecs) < 2:
        labels = np.zeros(len(seg_vecs), dtype=int)
    else:
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=5, init='k-means++')
            labels = km.fit_predict(X)
        except Exception as exc:
            logger.warning("KMeans failed (%s); assigning single cluster.", exc)
            labels = np.zeros(len(seg_vecs), dtype=int)

    # Map cluster ids to letters ordered by frequency (most common → A)
    counts = Counter(int(lbl) for lbl in labels)
    id_to_char = {cid: chr(65 + i) for i, (cid, _) in enumerate(counts.most_common())}

    raw_segs: list[dict] = []
    for (t0, t1), numeric_lbl in zip(seg_spans, labels):
        raw_segs.append({
            "start": round(t0, 2),
            "end": round(t1, 2),
            "label": id_to_char[int(numeric_lbl)],
        })

    # Intentionally NOT merging consecutive same-label segments here.
    # Merging destroys valid boundaries in harmonically uniform genres (blues,
    # folk) where KMeans assigns the same label to all verses — collapsing
    # 10+ correct boundaries into one giant segment and wrecking recall.
    enforced = _enforce_min_segment_duration(raw_segs, min_dur=min_seg_dur, total_dur=total_dur)
    return _assign_section_types(enforced, total_dur)


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

    Accepts a params dict with any subset of these keys:

    Existing keys (all preserved):
      min_segment_duration_seconds  float  default 6.0
      novelty_kernel_size_seconds   float  default 4.0
      n_clusters                    int    default 4
      use_mfcc                      bool   default True
      mfcc_n_components             int    default 20
      spectral_flux_weight          float  default 0.3
      auto_n_clusters               bool   default True

    New keys:
      use_beat_sync          bool  default True  — toggle beat-sync features
      smoothing_L            int   default 15    — diagonal smoothing window (beats)
      transposition_invariant bool default True  — toggle TI-SSM
    """
    t_total = time.perf_counter()

    # --- Parse params ---
    p = params or {}
    min_seg_dur  = float(p.get("min_segment_duration_seconds", _MIN_SEG_DUR))
    kernel_s     = float(p.get("novelty_kernel_size_seconds",  _NOVELTY_KERNEL_SECONDS))
    n_clusters   = int(  p.get("n_clusters",                   _N_CLUSTERS))
    use_mfcc     = bool( p.get("use_mfcc",                     True))
    n_mfcc       = int(  p.get("mfcc_n_components",             _MFCC_N))
    flux_weight  = float(p.get("spectral_flux_weight",          _SPECTRAL_FLUX_WEIGHT))
    flux_weight  = max(0.0, min(1.0, flux_weight))
    auto_k       = bool( p.get("auto_n_clusters",               True))
    use_beat_sync   = bool(p.get("use_beat_sync",               True))
    smoothing_L     = int( p.get("smoothing_L",                 _SMOOTHING_L))
    ti              = bool(p.get("transposition_invariant",      True))

    # --- Stage 0: Load audio ---
    t0 = time.perf_counter()
    y, sr = _load_audio_from_bytes(content)
    original_dur = float(librosa.get_duration(y=y, sr=sr))
    logger.info("[%.2fs] Audio loaded: sr=%d, duration=%.2fs", time.perf_counter() - t0, sr, original_dur)

    # --- Stage 0b: Active region ---
    t0 = time.perf_counter()
    act_start, act_end = _detect_active_region(y, sr, hop_length=_HOP_LENGTH)
    core_dur = max(0.0, act_end - act_start)
    logger.info("[%.2fs] Active region: %.2fs–%.2fs (%.2fs)",
                time.perf_counter() - t0, act_start, act_end, core_dur)

    if core_dur <= 0.0:
        return _empty_result(filename, content_type, original_dur, "No active music region detected.")

    y_active = y[int(act_start * sr): int(act_end * sr)].astype(np.float32)

    # --- Stage 1: Beat-sync features ---
    t0 = time.perf_counter()
    if use_beat_sync:
        chroma_beat, mfcc_beat, beat_times, beat_rate = _extract_beat_sync_features(
            y_active, sr, hop_length=_HOP_LENGTH, n_mfcc=n_mfcc, use_mfcc=use_mfcc
        )
    else:
        # Raw frame features (no beat sync) — used when audio is non-rhythmic
        chroma_raw = librosa.feature.chroma_cens(y=y_active, sr=sr, hop_length=_HOP_LENGTH).astype(np.float32)
        n_frames = chroma_raw.shape[1]
        norms = np.linalg.norm(chroma_raw, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        chroma_beat = chroma_raw / norms
        if use_mfcc:
            mfcc_raw = librosa.feature.mfcc(y=y_active, sr=sr, hop_length=_HOP_LENGTH, n_mfcc=n_mfcc).astype(np.float32)
            nm = np.linalg.norm(mfcc_raw, axis=0, keepdims=True)
            nm[nm == 0] = 1.0
            mfcc_beat = mfcc_raw / nm
        else:
            mfcc_beat = np.zeros((n_mfcc, chroma_beat.shape[1]), dtype=np.float32)
        beat_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=_HOP_LENGTH).astype(np.float32)
        beat_rate = sr / _HOP_LENGTH

    logger.info("[%.2fs] Features: n_beats=%d, beat_rate=%.2f bps",
                time.perf_counter() - t0, beat_times.size, beat_rate)

    # Downsample to cap if necessary (very long tracks)
    if beat_times.size > _MAX_SSM_BEATS:
        step = int(np.ceil(beat_times.size / _MAX_SSM_BEATS))
        chroma_beat = chroma_beat[:, ::step]
        mfcc_beat   = mfcc_beat[:, ::step]
        beat_times  = beat_times[::step]
        beat_rate  /= step
        logger.info("Downsampled beats to %d (step=%d)", beat_times.size, step)

    # --- Stage 2: Build raw SSM ---
    t0 = time.perf_counter()
    S_raw = _build_combined_ssm(chroma_beat, mfcc_beat, use_mfcc=use_mfcc, transposition_invariant=ti)
    logger.info("[%.2fs] SSM built: shape=%s, TI=%s", time.perf_counter() - t0, S_raw.shape, ti)

    # --- Stage 3: Enhance SSM ---
    t0 = time.perf_counter()
    S_enh = _compute_enhanced_ssm(S_raw, L=smoothing_L, rho=_SSM_RHO, penalty=_SSM_PENALTY)
    logger.info("[%.2fs] SSM enhanced (L=%d)", time.perf_counter() - t0, smoothing_L)

    # --- Stage 4: Novelty curve ---
    t0 = time.perf_counter()
    # Minimum 8 beats: at low beat-rates (e.g. 1-2 bps), a 4s kernel gives only
    # 4-8 beats — far too small for section-level detection.  8 beats covers
    # at least two bars at 120 BPM and provides a meaningful checkerboard.
    kernel_L = max(8, int(kernel_s * beat_rate))
    ssm_novelty = _compute_novelty_ssm(S_enh, L=kernel_L)

    if flux_weight > 0.0:
        flux_novelty = _compute_spectral_flux_novelty(y_active, sr, hop_length=_HOP_LENGTH)
        novelty = _blend_novelty_curves(
            ssm_novelty, flux_novelty, beat_times,
            y_len=len(y_active), sr=sr, hop_length=_HOP_LENGTH,
            flux_weight=flux_weight,
        )
    else:
        novelty = ssm_novelty

    logger.info("[%.2fs] Novelty curve (kernel_L=%d beats, flux_w=%.2f)",
                time.perf_counter() - t0, kernel_L, flux_weight)

    # --- Stage 5: Peak picking ---
    t0 = time.perf_counter()
    boundary_times_core = _find_boundaries(
        novelty, beat_times,
        min_segment_s=min_seg_dur,
        y_active=y_active, sr=sr, hop_length=_HOP_LENGTH,
    )
    logger.info("[%.2fs] %d boundaries detected", time.perf_counter() - t0, len(boundary_times_core))

    # --- Stage 6: Clustering & labelling ---
    t0 = time.perf_counter()
    segments_core = _cluster_and_label_segments(
        chroma_beat, mfcc_beat, beat_times,
        boundary_times=boundary_times_core,
        n_clusters=n_clusters,
        min_seg_dur=min_seg_dur,
        total_dur=core_dur,
        auto_n_clusters=auto_k,
        use_mfcc=use_mfcc,
    )
    logger.info("[%.2fs] %d segments after clustering", time.perf_counter() - t0, len(segments_core))

    # --- Stage 7: Shift times back to original timeline ---
    for seg in segments_core:
        seg["start"] = round(seg["start"] + act_start, 2)
        seg["end"]   = round(seg["end"]   + act_start, 2)

    logger.info("Total pipeline time: %.2fs for %s", time.perf_counter() - t_total, filename)

    return {
        "filename": filename,
        "content_type": content_type,
        "duration_seconds": round(original_dur, 2),
        "segments": segments_core,
        "status": "Segmentation complete.",
    }


def _empty_result(filename: str, content_type: str, duration: float, msg: str) -> dict:
    return {
        "filename": filename,
        "content_type": content_type,
        "duration_seconds": round(duration, 2),
        "segments": [],
        "status": msg,
    }
