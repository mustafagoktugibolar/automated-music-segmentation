"""
Segment-level feature vectors for ML-based label classification.

Wraps the existing ``build_segment_descriptors`` from ``shared.labeling``
(acoustic features: chroma/MFCC/RMS/onset/beat/spectral, 60-dim) and augments
the vector with contextual features that the heuristic rule-set uses implicitly
(position, duration, energy rank, repetition count).

The *same* function is called at both **training** time (ground-truth
segments) and **inference** time (predicted segments) so that the feature
space is identical in both settings.

Feature layout (73 total)
--------------------------
[0:54]  acoustic  — chroma_mean×12, chroma_std×12, mfcc_mean×13,
                    mfcc_std×13, rms_mean, rms_std, onset_density,
                    norm_duration
[54:60] rhythm    — tempo_norm, beat_density_norm, beat_regularity,
                    spectral_centroid_mean_norm, spectral_centroid_std_norm,
                    zcr_mean
[60:73] context   — positional / repetition features (see _CONTEXT_NAMES)
"""
from __future__ import annotations

import numpy as np
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# Descriptor layout produced by shared.labeling.build_segment_descriptors
# ──────────────────────────────────────────────────────────────────────────────
#  chroma_mean  [0 :12]
#  chroma_std   [12:24]
#  mfcc_mean    [24:37]   (n_mfcc=13)
#  mfcc_std     [37:50]
#  rms_mean     [50]
#  rms_std      [51]
#  onset_density[52]
#  norm_duration[53]
#  ─────────────────────
#  total                 54
_ACOUSTIC_DIM = 60
_RMS_MEAN_IDX = 50  # must stay in sync with shared.labeling._RMS_IDX

_CONTEXT_NAMES: list[str] = [
    "normalized_start",    # 0
    "normalized_end",      # 1
    "position_center",     # 2
    "index_norm",          # 3  index / (n-1)
    "is_first",            # 4
    "is_last",             # 5
    "n_segments",          # 6
    "duration_s",          # 7
    "log_duration",        # 8  log1p(duration_s)
    "rms_energy_rank",     # 9  percentile rank among siblings
    "is_max_energy",       # 10 1 if highest-RMS segment
    "repetition_count",    # 11 how many segments share the same structural label
    "is_repeated",         # 12 1 if repetition_count >= 2
]
_CONTEXT_DIM = len(_CONTEXT_NAMES)
_TOTAL_DIM = _ACOUSTIC_DIM + _CONTEXT_DIM  # 73


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
        ``structural_label`` / ``label`` are used for repetition counting.
    descriptors:
        Pre-computed acoustic descriptor matrix, shape *(N, 54)*, from
        ``shared.labeling.build_segment_descriptors``.  When *None* and
        *file_path* is provided the descriptors are computed here.
    file_path:
        Path to the audio file.  Used only when *descriptors* is *None*.

    Returns
    -------
    X : np.ndarray, shape *(N, 67)*, dtype float32
    feature_names : list[str], length 67
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

    # Pad or truncate to exactly _ACOUSTIC_DIM columns so the feature space
    # is stable even if the descriptor changes shape slightly.
    if acoustic.shape[1] < _ACOUSTIC_DIM:
        pad = np.zeros((n, _ACOUSTIC_DIM - acoustic.shape[1]), dtype=np.float32)
        acoustic = np.concatenate([acoustic, pad], axis=1)
    elif acoustic.shape[1] > _ACOUSTIC_DIM:
        acoustic = acoustic[:, :_ACOUSTIC_DIM]

    # ── Contextual features ──────────────────────────────────────────────────
    total_dur = max(
        (float(s.get("end", 0.0) or 0.0) for s in segments), default=1.0
    )
    total_dur = max(total_dur, 1.0)

    struct_labels = [
        str(s.get("structural_label") or s.get("label") or "").strip()
        for s in segments
    ]
    label_counts = Counter(struct_labels)

    rms_values = acoustic[:, _RMS_MEAN_IDX]

    # Percentile rank: 0 = quietest, 1 = loudest segment in this track.
    rms_ranks = (
        np.argsort(np.argsort(rms_values)).astype(np.float32) / max(n - 1, 1)
    )
    max_rms_idx = int(np.argmax(rms_values))

    ctx = np.zeros((n, _CONTEXT_DIM), dtype=np.float32)
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0.0) or 0.0)
        end   = float(seg.get("end",   0.0) or 0.0)
        dur   = max(0.0, end - start)
        p_start  = start / total_dur
        p_end    = end   / total_dur
        p_center = (p_start + p_end) / 2.0
        rep = label_counts[struct_labels[i]]

        ctx[i] = [
            p_start,                                  # 0
            p_end,                                    # 1
            p_center,                                 # 2
            float(i) / max(n - 1, 1),               # 3
            1.0 if i == 0 else 0.0,                  # 4
            1.0 if i == n - 1 else 0.0,              # 5
            float(n),                                 # 6
            dur,                                      # 7
            float(np.log1p(dur)),                    # 8
            rms_ranks[i],                             # 9
            1.0 if i == max_rms_idx else 0.0,        # 10
            float(rep),                               # 11
            1.0 if rep >= 2 else 0.0,                # 12
        ]

    X = np.concatenate([acoustic, ctx], axis=1)
    return X, feature_names()


def feature_names() -> list[str]:
    """Return the ordered list of all 67 feature names."""
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
    names += _CONTEXT_NAMES
    return names  # 60 acoustic + 13 context = 73
