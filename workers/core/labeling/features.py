"""
Segment-level feature vectors for ML-based label classification.

Feature layout (87 total)
--------------------------
[0:60]   acoustic   — chroma/MFCC/RMS/onset/beat/spectral (from descriptor)
[60:71]  context    — positional features, no label-derived values (11)
[71:79]  repetition — acoustic similarity-based repetition (8)
[79:87]  contrast   — local RMS/onset/centroid contrast (8)

The same function is called at both training time (ground-truth segments) and
inference time (predicted segments) so the feature space is identical.
"""
from __future__ import annotations

import numpy as np

# ── Acoustic descriptor layout ────────────────────────────────────────────────
#  chroma_mean  [0 :12]
#  chroma_std   [12:24]
#  mfcc_mean    [24:37]   (n_mfcc=13)
#  mfcc_std     [37:50]
#  rms_mean     [50]
#  rms_std      [51]
#  onset_density[52]
#  norm_duration[53]
#  tempo_norm   [54]
#  beat_density_norm [55]
#  beat_regularity   [56]
#  spectral_centroid_mean_norm [57]
#  spectral_centroid_std_norm  [58]
#  zcr_mean     [59]
_ACOUSTIC_DIM  = 60
_RMS_MEAN_IDX  = 50  # must stay in sync with workers.infrastructure.audio.features
_ONSET_IDX     = 52
_NORM_DUR_IDX  = 53
_CENTROID_IDX  = 57

# ── Context features (positional only, no label-derived values) ───────────────
_CONTEXT_NAMES: list[str] = [
    "normalized_start",   # 0
    "normalized_end",     # 1
    "position_center",    # 2
    "index_norm",         # 3  i / (n-1)
    "is_first",           # 4
    "is_last",            # 5
    "n_segments",         # 6
    "duration_s",         # 7
    "log_duration",       # 8  log1p(duration_s)
    "rms_energy_rank",    # 9  percentile rank among siblings
    "is_max_energy",      # 10 1 if highest-RMS segment
]
_CONTEXT_DIM = len(_CONTEXT_NAMES)  # 11

# ── Acoustic repetition features (Stage 5, non-leaky) ────────────────────────
_REPETITION_NAMES: list[str] = [
    "max_chroma_similarity",       # 0  max cosine sim to any other segment
    "mean_top3_chroma_similarity", # 1  mean of top-3
    "max_mfcc_similarity",         # 2
    "mean_top3_mfcc_similarity",   # 3
    "similar_count_chroma_080",    # 4  count of segments with chroma sim > 0.8
    "similar_count_mfcc_080",      # 5
    "nearest_similar_dist_norm",   # 6  norm position dist to nearest chroma-similar seg
    "is_repeated_acoustic",        # 7  1 if similar_count_chroma_080 >= 2
]
_REPETITION_DIM = len(_REPETITION_NAMES)  # 8

# ── Local contrast features (Stage 6, non-leaky) ─────────────────────────────
_CONTRAST_NAMES: list[str] = [
    "rms_vs_song_mean",    # 0
    "rms_vs_prev",         # 1
    "rms_vs_next",         # 2
    "onset_vs_song_mean",  # 3
    "onset_vs_prev",       # 4
    "onset_vs_next",       # 5
    "centroid_vs_song_mean", # 6
    "duration_vs_song_mean", # 7
]
_CONTRAST_DIM = len(_CONTRAST_NAMES)  # 8

_TOTAL_DIM = _ACOUSTIC_DIM + _CONTEXT_DIM + _REPETITION_DIM + _CONTRAST_DIM  # 87

# Minimum number of other segments with chroma similarity > 0.8 for a segment to be
# flagged as acoustically repeated. >= 1 means the section appears at least twice in
# the track (self is excluded from the count).
_ACOUSTIC_REPEAT_THRESHOLD = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine_sim_matrix(A: np.ndarray) -> np.ndarray:
    """Return NxN cosine similarity matrix for rows of A. Handles zero rows."""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    A_n = (A / norms).astype(np.float64)
    return np.clip(A_n @ A_n.T, -1.0, 1.0).astype(np.float32)


def _acoustic_repetition_features(acoustic: np.ndarray) -> np.ndarray:
    """Return (n, 8) acoustic repetition features. No label info used."""
    n = acoustic.shape[0]
    out = np.zeros((n, _REPETITION_DIM), dtype=np.float32)
    if n < 2:
        out[:, 6] = 1.0  # nearest_similar_dist_norm = max when no other segs
        return out

    chroma = acoustic[:, 0:12]    # chroma_mean
    mfcc   = acoustic[:, 24:37]   # mfcc_mean

    sim_c = _cosine_sim_matrix(chroma)
    sim_m = _cosine_sim_matrix(mfcc)

    positions = np.arange(n, dtype=np.float32) / max(n - 1, 1)

    for i in range(n):
        other = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        sc = sim_c[i, other]
        sm = sim_m[i, other]
        pos_others = positions[other]

        max_c = float(np.max(sc))
        max_m = float(np.max(sm))

        k = min(3, len(sc))
        mean_top3_c = float(np.mean(np.sort(sc)[-k:]))
        mean_top3_m = float(np.mean(np.sort(sm)[-k:]))

        cnt_c = int(np.sum(sc > 0.8))
        cnt_m = int(np.sum(sm > 0.8))

        sim_mask = sc > 0.8
        if np.any(sim_mask):
            nearest_dist = float(np.min(np.abs(pos_others[sim_mask] - positions[i])))
        else:
            nearest_dist = 1.0

        out[i] = [
            max_c, mean_top3_c, max_m, mean_top3_m,
            float(cnt_c), float(cnt_m), nearest_dist,
            1.0 if cnt_c >= _ACOUSTIC_REPEAT_THRESHOLD else 0.0,
        ]

    return out


def _local_contrast_features(acoustic: np.ndarray) -> np.ndarray:
    """Return (n, 8) local contrast features. No label info used."""
    n = acoustic.shape[0]
    out = np.zeros((n, _CONTRAST_DIM), dtype=np.float32)

    rms      = acoustic[:, _RMS_MEAN_IDX].astype(np.float64)
    onset    = acoustic[:, _ONSET_IDX].astype(np.float64)
    centroid = acoustic[:, _CENTROID_IDX].astype(np.float64)
    norm_dur = acoustic[:, _NORM_DUR_IDX].astype(np.float64)

    rms_mean      = float(np.mean(rms))
    onset_mean    = float(np.mean(onset))
    centroid_mean = float(np.mean(centroid))
    dur_mean      = float(np.mean(norm_dur))

    for i in range(n):
        out[i] = [
            rms[i] - rms_mean,
            rms[i] - rms[i - 1] if i > 0     else 0.0,
            rms[i] - rms[i + 1] if i < n - 1 else 0.0,
            onset[i] - onset_mean,
            onset[i] - onset[i - 1] if i > 0     else 0.0,
            onset[i] - onset[i + 1] if i < n - 1 else 0.0,
            centroid[i] - centroid_mean,
            norm_dur[i] - dur_mean,
        ]

    return out.astype(np.float32)


# ── Main function ─────────────────────────────────────────────────────────────

def build_segment_label_vectors(
    segments: list[dict],
    descriptors: np.ndarray | None = None,
    file_path: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return *(X, feature_names)* for *N* segments.

    Parameters
    ----------
    segments:
        List of segment dicts with at least ``start`` and ``end`` keys.
    descriptors:
        Pre-computed acoustic descriptor matrix, shape *(N, 60)*, from
        ``workers.infrastructure.audio.features.build_segment_descriptors``.
        When *None* and *file_path* is provided, descriptors are computed here.
    file_path:
        Path or bytes for audio. Used only when *descriptors* is *None*.

    Returns
    -------
    X : np.ndarray, shape *(N, 87)*, dtype float32
    feature_names : list[str], length 87
    """
    n = len(segments)
    if n == 0:
        return np.empty((0, _TOTAL_DIM), dtype=np.float32), feature_names()

    # ── Acoustic features ────────────────────────────────────────────────────
    if descriptors is None and file_path:
        from workers.infrastructure.audio.features import build_segment_descriptors as _bsd
        descriptors = _bsd(file_path, segments)

    if descriptors is not None and len(descriptors) == n:
        acoustic = np.nan_to_num(
            descriptors.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
    else:
        acoustic = np.zeros((n, _ACOUSTIC_DIM), dtype=np.float32)

    if acoustic.shape[1] < _ACOUSTIC_DIM:
        pad = np.zeros((n, _ACOUSTIC_DIM - acoustic.shape[1]), dtype=np.float32)
        acoustic = np.concatenate([acoustic, pad], axis=1)
    elif acoustic.shape[1] > _ACOUSTIC_DIM:
        acoustic = acoustic[:, :_ACOUSTIC_DIM]

    # ── Context features (positional only) ───────────────────────────────────
    total_dur = max(
        (float(s.get("end", 0.0) or 0.0) for s in segments), default=1.0
    )
    total_dur = max(total_dur, 1.0)

    rms_values = acoustic[:, _RMS_MEAN_IDX]
    rms_ranks  = (
        np.argsort(np.argsort(rms_values)).astype(np.float32) / max(n - 1, 1)
    )
    max_rms_idx = int(np.argmax(rms_values))

    ctx = np.zeros((n, _CONTEXT_DIM), dtype=np.float32)
    for i, seg in enumerate(segments):
        start    = float(seg.get("start", 0.0) or 0.0)
        end      = float(seg.get("end",   0.0) or 0.0)
        dur      = max(0.0, end - start)
        p_start  = start / total_dur
        p_end    = end   / total_dur
        p_center = (p_start + p_end) / 2.0

        ctx[i] = [
            p_start,
            p_end,
            p_center,
            float(i) / max(n - 1, 1),
            1.0 if i == 0     else 0.0,
            1.0 if i == n - 1 else 0.0,
            float(n),
            dur,
            float(np.log1p(dur)),
            rms_ranks[i],
            1.0 if i == max_rms_idx else 0.0,
        ]

    # ── Acoustic repetition features (Stage 5) ───────────────────────────────
    rep = _acoustic_repetition_features(acoustic)

    # ── Local contrast features (Stage 6) ────────────────────────────────────
    contrast = _local_contrast_features(acoustic)

    X = np.concatenate([acoustic, ctx, rep, contrast], axis=1)
    return X, feature_names()


def feature_names() -> list[str]:
    """Return the ordered list of all 87 feature names."""
    names: list[str] = []
    for d in range(12):
        names.append(f"chroma_mean_{d}")
    for d in range(12):
        names.append(f"chroma_std_{d}")
    for d in range(13):
        names.append(f"mfcc_mean_{d}")
    for d in range(13):
        names.append(f"mfcc_std_{d}")
    names += ["rms_mean", "rms_std", "onset_density", "norm_duration"]
    names += [
        "tempo_norm", "beat_density_norm", "beat_regularity",
        "spectral_centroid_mean_norm", "spectral_centroid_std_norm", "zcr_mean",
    ]
    names += _CONTEXT_NAMES      # 11
    names += _REPETITION_NAMES   # 8
    names += _CONTRAST_NAMES     # 8
    return names  # 60 acoustic + 11 context + 8 repetition + 8 contrast = 87
