from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from shared.logger import get_logger

logger = get_logger()

DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "ssm": 0.42,
    "chord_proxy": 0.18,
    "onset_flux": 0.06,
    "rms": 0.06,
    "lyrics": 0.10,
    "beat": 0.02,
}


def find_boundaries(
    novelty: np.ndarray,
    fps: float,
    min_segment_s: float,
    total_dur: float,
    prominence: float,
    novelty_sigma: float,
) -> list[float]:
    """Detect boundary times from a frame-indexed novelty curve."""
    if novelty.size == 0 or fps <= 0:
        return []

    if novelty_sigma > 0 and novelty.size > 5:
        novelty = gaussian_filter1d(novelty, sigma=novelty_sigma)
        max_val = float(novelty.max())
        if max_val > 0:
            novelty = novelty / max_val

    min_dist_frames = max(1, int(min_segment_s * fps))
    edge_margin_frames = max(1, int(min_segment_s * 0.5 * fps))

    try:
        peaks, _ = find_peaks(novelty, distance=min_dist_frames, prominence=prominence)
    except Exception as exc:
        logger.warning("find_peaks failed (%s); no boundaries detected.", exc)
        return []

    peaks = peaks[
        (peaks >= edge_margin_frames) &
        (peaks <= novelty.size - edge_margin_frames)
    ]

    filtered: list[float] = []
    for t in sorted(float(p + 0.5) / fps for p in peaks):
        if t <= 0.0 or t >= total_dur:
            continue
        if not filtered or (t - filtered[-1]) >= min_segment_s:
            filtered.append(round(t, 3))
    return filtered


def normalise_curve(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    out = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = out - float(out.min())
    max_val = float(out.max())
    if max_val > 0:
        out = out / max_val
    return out.astype(np.float32)


def curve_confidence(curve: np.ndarray, frame_times: np.ndarray, t: float) -> float:
    if curve.size == 0 or frame_times.size == 0:
        return 0.5
    idx = int(np.argmin(np.abs(frame_times - t)))
    return round(float(np.clip(curve[idx], 0.0, 1.0)), 3)


def candidates_from_boundaries(
    boundaries: list[float],
    source: str,
    curve: np.ndarray,
    frame_times: np.ndarray,
    default_confidence: float = 0.5,
) -> list[dict]:
    return [
        {
            "time": round(float(t), 3),
            "source": source,
            "confidence": curve_confidence(curve, frame_times, t) if curve.size else default_confidence,
        }
        for t in boundaries
    ]


def rms_boundary_candidates(
    y: np.ndarray,
    sr: int,
    frame_times: np.ndarray,
    fps: float,
    min_seg_dur: float,
    total_dur: float,
    hop_length: int,
) -> tuple[list[dict], np.ndarray]:
    try:
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0].astype(np.float32)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        if rms_db.size > 5:
            rms_db = gaussian_filter1d(rms_db, sigma=3.0)
        novelty_raw = np.abs(np.diff(rms_db, prepend=rms_db[0]))
        raw_times = librosa.frames_to_time(np.arange(len(novelty_raw)), sr=sr, hop_length=hop_length)
        novelty = normalise_curve(np.interp(frame_times, raw_times, novelty_raw, left=0.0, right=0.0))
        boundaries = find_boundaries(novelty, fps, min_seg_dur, total_dur, prominence=0.12, novelty_sigma=2.0)
        return candidates_from_boundaries(boundaries, "rms", novelty, frame_times), novelty
    except Exception as exc:
        logger.warning("RMS boundary candidates failed (%s).", exc)
        return [], np.zeros(frame_times.size, dtype=np.float32)


def onset_boundary_candidates(
    y: np.ndarray,
    sr: int,
    frame_times: np.ndarray,
    fps: float,
    min_seg_dur: float,
    total_dur: float,
    hop_length: int,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32)
        if onset_env.size > 5:
            onset_env = gaussian_filter1d(onset_env, sigma=5.0)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length).astype(np.float32)
        novelty = normalise_curve(np.interp(frame_times, onset_times, onset_env, left=0.0, right=0.0))
        boundaries = find_boundaries(novelty, fps, min_seg_dur, total_dur, prominence=0.15, novelty_sigma=2.0)
        return candidates_from_boundaries(boundaries, "onset_flux", novelty, frame_times), novelty, onset_times, onset_env
    except Exception as exc:
        logger.warning("Onset boundary candidates failed (%s).", exc)
        return [], np.zeros(frame_times.size, dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)


def tempo_and_beats(y: np.ndarray, sr: int, hop_length: int) -> tuple[float, np.ndarray]:
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo_f = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).astype(np.float32)
        return round(tempo_f, 3), beat_times
    except Exception as exc:
        logger.warning("Tempo/beat detection failed (%s).", exc)
        return 0.0, np.array([], dtype=np.float32)


def beat_phrase_boundary_candidates(
    beat_times: np.ndarray,
    onset_times: np.ndarray,
    onset_env: np.ndarray,
    total_dur: float,
    min_seg_dur: float,
    support_times: np.ndarray | None = None,
    support_curve: np.ndarray | None = None,
    phrase_steps: tuple[int, ...] = (16, 24, 32, 48),
) -> list[dict]:
    """Generate globally phased beat-grid candidates for section-level timing."""
    if beat_times.size < 8 or onset_times.size == 0 or onset_env.size == 0:
        return []

    onset_at_beats = np.interp(beat_times, onset_times, onset_env, left=0.0, right=0.0)
    max_onset = float(np.max(onset_at_beats)) or 1.0
    if support_times is not None and support_curve is not None and support_times.size and support_curve.size:
        support_at_beats = np.interp(beat_times, support_times, support_curve, left=0.0, right=0.0)
        max_support = float(np.max(support_at_beats)) or 1.0
        support_at_beats = support_at_beats / max_support
    else:
        support_at_beats = np.zeros_like(onset_at_beats)
    out: list[dict] = []

    for step in phrase_steps:
        if beat_times.size < step:
            continue
        best_offset = 0
        best_score = -1.0
        for offset in range(min(step, beat_times.size)):
            idx = np.arange(offset, beat_times.size, step)
            if idx.size < 2:
                continue
            onset_score = float(np.mean(onset_at_beats[idx] / max_onset))
            support_score = float(np.mean(support_at_beats[idx]))
            score = 0.75 * support_score + 0.25 * onset_score
            if score > best_score:
                best_score = score
                best_offset = offset

        for idx in np.arange(best_offset, beat_times.size, step):
            t = float(beat_times[idx])
            if min_seg_dur * 0.5 <= t <= total_dur - min_seg_dur * 0.5:
                local_support = float(support_at_beats[idx])
                local_onset = min(1.0, float(onset_at_beats[idx]) / max_onset)
                confidence = 0.40 + 0.40 * local_support + 0.15 * local_onset
                # Source key must match DEFAULT_FEATURE_WEIGHTS ("beat"),
                # otherwise these candidates get zero weight in fusion.
                out.append({
                    "time": round(t, 3),
                    "source": "beat",
                    "confidence": round(confidence, 3),
                })

    return _dedupe_candidates(out, window_s=0.6)


def _dedupe_candidates(candidates: list[dict], window_s: float) -> list[dict]:
    deduped: list[dict] = []
    for item in sorted(candidates, key=lambda c: c["time"]):
        if not deduped or item["time"] - deduped[-1]["time"] > window_s:
            deduped.append(item)
        elif item["confidence"] > deduped[-1]["confidence"]:
            deduped[-1] = item
    return deduped


def chord_proxy_boundary_candidates(
    chroma: np.ndarray,
    frame_times: np.ndarray,
    fps: float,
    min_seg_dur: float,
    total_dur: float,
) -> tuple[list[dict], np.ndarray]:
    if chroma.shape[1] < 2:
        return [], np.zeros(frame_times.size, dtype=np.float32)
    try:
        # CENS is ~1s-smoothed, so adjacent frames at 10 Hz are nearly
        # identical and their difference is dominated by noise.  Compare
        # frames ±0.5s around each position instead (centred lag).
        n = chroma.shape[1]
        half = max(1, int(round(0.5 * fps)))
        chord_change = np.zeros(n, dtype=np.float32)
        if n > 2 * half:
            sims = np.sum(chroma[:, 2 * half:] * chroma[:, : n - 2 * half], axis=0)
            chord_change[half: n - half] = 1.0 - np.clip(sims, -1.0, 1.0)
        else:
            sims = np.sum(chroma[:, 1:] * chroma[:, :-1], axis=0)
            chord_change[1:] = 1.0 - np.clip(sims, -1.0, 1.0)
        if chord_change.size > 5:
            chord_change = gaussian_filter1d(chord_change, sigma=2.0)
        novelty = normalise_curve(chord_change)
        boundaries = find_boundaries(novelty, fps, min_seg_dur, total_dur, prominence=0.10, novelty_sigma=1.5)
        return candidates_from_boundaries(boundaries, "chord_proxy", novelty, frame_times), novelty
    except Exception as exc:
        logger.warning("Chord-proxy boundary candidates failed (%s).", exc)
        return [], np.zeros(frame_times.size, dtype=np.float32)


def lyrics_boundary_candidates(
    timed_lyrics: list[dict] | None,
    active_start: float,
    total_dur: float,
    min_seg_dur: float,
) -> list[dict]:
    if not timed_lyrics:
        return []
    out: list[dict] = []
    for item in timed_lyrics:
        try:
            text = str(item.get("text", "")).strip()
            t = float(item.get("time_seconds")) - active_start
        except Exception:
            continue
        if text and min_seg_dur * 0.5 <= t <= total_dur - min_seg_dur * 0.5:
            out.append({"time": round(t, 3), "source": "lyrics", "confidence": 0.55})
    return out


def normalise_feature_weights(params_weights: dict | None, spectral_flux_weight: float | None) -> dict[str, float]:
    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if params_weights:
        for key, value in params_weights.items():
            if key not in weights:
                continue
            try:
                weights[key] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
    if spectral_flux_weight is not None:
        weights["onset_flux"] = max(0.0, float(spectral_flux_weight))
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_FEATURE_WEIGHTS)
    return {k: v / total for k, v in weights.items()}


def fuse_feature_candidates(
    candidates: list[dict],
    weights: dict[str, float],
    total_dur: float,
    min_seg_dur: float,
    merge_window_s: float = 2.75,
    threshold: float = 0.30,
    max_boundaries: int | None = None,
) -> list[dict]:
    valid = [
        c for c in candidates
        if min_seg_dur * 0.5 <= float(c["time"]) <= total_dur - min_seg_dur * 0.5
    ]
    groups: list[list[dict]] = []
    for candidate in sorted(valid, key=lambda c: float(c["time"])):
        group_mean = float(np.mean([g["time"] for g in groups[-1]])) if groups else None
        if group_mean is None or abs(float(candidate["time"]) - group_mean) > merge_window_s:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)

    fused: list[dict] = []
    for group in groups:
        best_by_source: dict[str, dict] = {}
        for candidate in group:
            source = str(candidate["source"])
            if source not in best_by_source or candidate["confidence"] > best_by_source[source]["confidence"]:
                best_by_source[source] = candidate

        weighted_sum = 0.0
        for source, candidate in best_by_source.items():
            weight = weights.get(source, 0.0)
            confidence = float(candidate.get("confidence", 0.5))
            weighted_sum += weight * confidence

        sources = sorted(best_by_source)
        score = min(1.0, weighted_sum + min(0.15, 0.035 * max(0, len(sources) - 1)))
        # The SSM is the primary structural signal: a confident SSM-only
        # candidate must survive even when its weighted score falls below
        # the multi-source agreement threshold.
        ssm_cand = best_by_source.get("ssm")
        ssm_strong = ssm_cand is not None and float(ssm_cand.get("confidence", 0.0)) >= 0.5
        if score >= threshold or ssm_strong:
            anchor = _choose_boundary_anchor(best_by_source, weights)
            fused.append({
                "time": round(float(anchor["time"]), 3),
                "sources": sources,
                "confidence": round(score, 3),
            })

    kept: list[dict] = []
    for item in sorted(fused, key=lambda c: c["time"]):
        if not kept or item["time"] - kept[-1]["time"] >= min_seg_dur:
            kept.append(item)
        elif item["confidence"] > kept[-1]["confidence"]:
            kept[-1] = item

    if max_boundaries is not None and len(kept) > max_boundaries:
        strongest = sorted(kept, key=lambda c: c["confidence"], reverse=True)[:max_boundaries]
        kept = sorted(strongest, key=lambda c: c["time"])
    return kept


def fuse_boundary_candidates(*args, **kwargs):
    """Backward-compatible alias for feature-level candidate fusion."""
    return fuse_feature_candidates(*args, **kwargs)


def _choose_boundary_anchor(best_by_source: dict[str, dict], weights: dict[str, float]) -> dict:
    """Use structural features for timing; onset/beat are refinement signals."""
    for source in ("ssm", "chord_proxy", "lyrics"):
        if source in best_by_source:
            return best_by_source[source]
    return max(
        best_by_source.values(),
        key=lambda c: weights.get(str(c["source"]), 0.0) * float(c.get("confidence", 0.5)),
    )


def snap_fused_boundaries(
    fused: list[dict],
    onset_times: np.ndarray,
    onset_env: np.ndarray,
    beat_times: np.ndarray,
    use_beat_sync: bool,
    onset_window_s: float,
    beat_window_s: float = 0.25,
) -> list[dict]:
    onset_threshold = float(np.percentile(onset_env, 75)) if onset_env.size else 0.0
    snapped: list[dict] = []
    for item in fused:
        t = float(item["time"])
        snapped_time = t
        snap_source = None

        if use_beat_sync and beat_times.size:
            nearest_idx = int(np.argmin(np.abs(beat_times - t)))
            if abs(float(beat_times[nearest_idx]) - t) <= beat_window_s:
                snapped_time = float(beat_times[nearest_idx])
                snap_source = "beat"

        if snap_source is None and onset_times.size and onset_env.size:
            onset_mask = np.abs(onset_times - t) <= onset_window_s
            if onset_mask.any():
                local_env = np.where(onset_mask, onset_env, 0.0)
                best = int(np.argmax(local_env))
                if float(local_env[best]) >= onset_threshold:
                    snapped_time = float(onset_times[best])
                    snap_source = "onset_snap"

        sources = list(item["sources"])
        if snap_source and snap_source not in sources:
            sources.append(snap_source)
        snapped.append({**item, "time": round(snapped_time, 3), "sources": sorted(sources)})

    deduped: list[dict] = []
    for item in sorted(snapped, key=lambda c: c["time"]):
        if not deduped or abs(item["time"] - deduped[-1]["time"]) > 0.25:
            deduped.append(item)
        elif item["confidence"] > deduped[-1]["confidence"]:
            deduped[-1] = item
    return deduped
