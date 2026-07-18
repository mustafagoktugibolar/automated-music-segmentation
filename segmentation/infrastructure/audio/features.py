"""
Acoustic feature extraction from audio arrays.

Depends on librosa for feature computation and on segmentation.infrastructure.audio.decoder
for audio loading.  This module is NOT in core/ because it has librosa as an
external I/O dependency; callers in core receive descriptors as plain ndarrays.
"""
from __future__ import annotations

import numpy as np


def build_segment_descriptors_from_audio(
    y: np.ndarray,
    sr: int,
    segments: list[dict],
) -> "np.ndarray | None":
    """Build acoustic descriptors from a pre-loaded audio array.

    Returns an (N, 54) float32 array — one row per segment.
    """
    if not segments:
        return None
    try:
        import librosa

        duration = float(librosa.get_duration(y=y, sr=sr))
        if duration <= 0:
            return None

        hop_length = 512
        chroma    = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
        mfcc      = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        rms       = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        onset     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        zcr       = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        frame_times = librosa.frames_to_time(
            np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
        )

        # Tempo via autocorrelation on full onset envelope (no DP — stable on any length)
        _tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop_length)
        tempo_norm = float(np.atleast_1d(_tempo)[0]) / 200.0  # normalize to ~[0,1]

        # Beat times via PLP (Predominant Local Pulse) — tempo-aware, no Viterbi DP
        _pulse = librosa.beat.plp(onset_envelope=onset, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(
            np.flatnonzero(librosa.util.localmax(_pulse)), sr=sr, hop_length=hop_length
        )

        descriptors: list[np.ndarray] = []
        for seg in segments:
            start = float(seg.get("start", 0.0) or 0.0)
            end   = float(seg.get("end",   0.0) or 0.0)
            mask  = (frame_times >= start) & (frame_times < end)
            if not mask.any():
                idx  = int(np.argmin(np.abs(frame_times - ((start + end) / 2.0))))
                mask = np.zeros(frame_times.size, dtype=bool)
                mask[idx] = True

            chroma_block   = chroma[:, mask]
            mfcc_block     = mfcc[:, mask]
            rms_block      = rms[: frame_times.size][mask]
            onset_block    = onset[: frame_times.size][mask]
            centroid_block = centroid[: frame_times.size][mask]
            zcr_block      = zcr[: frame_times.size][mask]
            seg_dur        = max(0.01, end - start)

            onset_density = (
                float(np.mean(onset_block > np.percentile(onset, 75)))
                if onset_block.size else 0.0
            )

            # Per-segment beat features
            seg_beats = beat_times[(beat_times >= start) & (beat_times < end)]
            beat_density = float(len(seg_beats)) / seg_dur
            if len(seg_beats) >= 2:
                ibi = np.diff(seg_beats)
                beat_regularity = max(0.0, 1.0 - float(np.std(ibi)) / (float(np.mean(ibi)) + 1e-6))
            else:
                beat_regularity = 0.5

            descriptors.append(
                np.concatenate(
                    [
                        np.mean(chroma_block, axis=1),           # 12
                        np.std(chroma_block, axis=1),            # 12
                        np.mean(mfcc_block, axis=1),             # 13
                        np.std(mfcc_block, axis=1),              # 13
                        [float(np.mean(rms_block)) if rms_block.size else 0.0],   # 1
                        [float(np.std(rms_block))  if rms_block.size else 0.0],   # 1
                        [onset_density],                          # 1
                        [seg_dur / max(duration, 1.0)],           # 1  = 54 so far
                        # ── tempo + beat + spectral features ──────────────────
                        [tempo_norm],                             # 1
                        [beat_density / 5.0],                    # 1  ~[0,1] at 5 bps max
                        [beat_regularity],                        # 1
                        [float(np.mean(centroid_block)) / (sr / 2.0) if centroid_block.size else 0.0],  # 1
                        [float(np.std(centroid_block))  / (sr / 2.0) if centroid_block.size else 0.0],  # 1
                        [float(np.mean(zcr_block)) if zcr_block.size else 0.0],   # 1  = 60 total
                    ]
                ).astype(np.float32)
            )
        return np.vstack(descriptors)
    except Exception:
        return None


def build_segment_descriptors(
    src: "str | bytes",
    segments: list[dict],
) -> "np.ndarray | None":
    """Decode audio then extract per-segment descriptors.

    Parameters
    ----------
    src : str or bytes
        File path or raw audio bytes (MinIO download etc.).
    segments : list[dict]
        Segment dicts with ``start`` and ``end`` keys.

    Returns
    -------
    np.ndarray of shape (N, 54) or None on failure.
    """
    if not src or not segments:
        return None
    try:
        from segmentation.infrastructure.audio.decoder import load_audio
        y, sr = load_audio(src)
        return build_segment_descriptors_from_audio(y, sr, segments)
    except Exception:
        return None
