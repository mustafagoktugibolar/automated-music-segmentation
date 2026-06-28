from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from shared.segmentation.models import AlgorithmResult, Boundary, Segment


CANONICAL_CUSTOM_ALGORITHM = "custom_librosa"
BASELINE_ALGORITHMS = ("custom_librosa", "foote", "cnmf", "scluster")
ALGORITHM_ALIASES = {
    "custom": CANONICAL_CUSTOM_ALGORITHM,
    "custom_librosa": CANONICAL_CUSTOM_ALGORITHM,
    "librosa": CANONICAL_CUSTOM_ALGORITHM,
    "foote": "foote",
    "cnmf": "cnmf",
    "scluster": "scluster",
    "fusion": "fusion",
    "llm": "llm",
}


def canonical_algorithm_name(value: Any) -> str:
    key = str(value or "").lower().strip()
    return ALGORITHM_ALIASES.get(key, key)


def get_audio_duration(file_path: str) -> float:
    try:
        import librosa

        try:
            return float(librosa.get_duration(path=file_path))
        except TypeError:
            return float(librosa.get_duration(filename=file_path))
    except Exception:
        try:
            import soundfile as sf

            info = sf.info(file_path)
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return 0.0


def _clean_time(value: Any) -> float | None:
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t) or t < 0:
        return None
    return t


def normalize_boundaries(
    boundaries: Iterable[Any],
    duration_seconds: float | None,
    min_gap_seconds: float = 0.1,
    include_edges: bool = True,
) -> list[float]:
    duration = _clean_time(duration_seconds)
    times: list[float] = []

    for item in boundaries or []:
        raw = item.get("time") if isinstance(item, dict) else item
        t = _clean_time(raw)
        if t is None:
            continue
        if duration is not None and duration > 0:
            if t > duration:
                continue
            t = min(max(0.0, t), duration)
        times.append(t)

    if include_edges:
        times.append(0.0)
        if duration is not None and duration > 0:
            times.append(duration)

    times = sorted(times)
    deduped: list[float] = []
    for t in times:
        if not deduped or abs(t - deduped[-1]) >= min_gap_seconds:
            deduped.append(t)
        elif duration is not None and abs(t - duration) <= min_gap_seconds:
            deduped[-1] = t

    if include_edges and duration is not None and duration > 0:
        if not deduped or deduped[0] > min_gap_seconds:
            deduped.insert(0, 0.0)
        else:
            deduped[0] = 0.0
        if duration - deduped[-1] > min_gap_seconds:
            deduped.append(duration)
        else:
            deduped[-1] = duration

    return [round(float(t), 3) for t in deduped]


def segments_to_intervals(segments: list[dict]) -> np.ndarray:
    intervals: list[tuple[float, float]] = []
    for seg in segments or []:
        start = _clean_time(seg.get("start") if isinstance(seg, dict) else None)
        end = _clean_time(seg.get("end") if isinstance(seg, dict) else None)
        if start is None or end is None:
            continue
        if end <= start:
            continue
        intervals.append((start, end))

    intervals.sort(key=lambda item: (item[0], item[1]))
    return np.asarray(intervals, dtype=float).reshape((-1, 2)) if intervals else np.empty((0, 2), dtype=float)


def segments_to_internal_boundaries(
    segments: list[dict],
    edge_margin_seconds: float = 0.5,
) -> list[float]:
    intervals = segments_to_intervals(segments)
    if len(intervals) <= 1:
        return []
    track_end = float(np.max(intervals[:, 1]))
    boundaries: list[float] = []
    for start in intervals[1:, 0]:
        t = float(start)
        if t <= edge_margin_seconds:
            continue
        if track_end > 0 and t >= track_end - edge_margin_seconds:
            continue
        boundaries.append(round(t, 3))
    return sorted(set(boundaries))


def boundaries_to_segments(
    boundaries: Iterable[Any],
    duration_seconds: float,
    labels: list[str] | None = None,
    min_gap_seconds: float = 0.1,
    boundary_metadata: list[dict] | None = None,
) -> list[dict]:
    times = normalize_boundaries(boundaries, duration_seconds, min_gap_seconds=min_gap_seconds, include_edges=True)
    if len(times) < 2:
        return []
    labels = labels or []
    boundary_metadata = boundary_metadata or []
    segments: list[dict] = []

    for idx, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if end <= start:
            continue
        label = str(labels[idx]) if idx < len(labels) and labels[idx] is not None else chr(65 + min(idx, 25))
        metadata = _nearest_boundary_metadata(start, end, boundary_metadata)
        segments.append(
            {
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "label": label,
                "structural_label": label,
                "confidence": metadata.get("confidence", 0.5),
                "source_features": metadata.get("sources", []),
            }
        )
    return segments


def _nearest_boundary_metadata(start: float, end: float, boundary_metadata: list[dict]) -> dict:
    nearby = [
        b for b in boundary_metadata
        if abs(float(b.get("time", -9999.0)) - start) <= 0.5
        or abs(float(b.get("time", -9999.0)) - end) <= 0.5
    ]
    if not nearby:
        return {}
    confidence = float(np.mean([float(b.get("confidence", 0.5)) for b in nearby]))
    sources_set: set[str] = set()
    for b in nearby:
        raw_sources = b.get("sources")
        if isinstance(raw_sources, list):
            sources_set.update(str(src) for src in raw_sources)
        elif raw_sources:
            sources_set.add(str(raw_sources))
        elif b.get("source"):
            sources_set.add(str(b["source"]))
    sources = sorted(sources_set)
    return {"confidence": round(confidence, 3), "sources": sources}


def enforce_min_segment_duration(
    segments: list[dict],
    min_duration_seconds: float,
    duration_seconds: float | None = None,
) -> list[dict]:
    if not segments:
        return []
    segs = [dict(s) for s in sorted(segments, key=lambda item: float(item.get("start", 0.0) or 0.0))]

    def merge(left: dict, right: dict) -> dict:
        label_source = right if (right.get("end", 0) - right.get("start", 0)) >= (left.get("end", 0) - left.get("start", 0)) else left
        sources = sorted(set(left.get("source_features", [])) | set(right.get("source_features", [])))
        return {
            **label_source,
            "start": left["start"],
            "end": right["end"],
            "label": label_source.get("label", "A"),
            "structural_label": label_source.get("structural_label") or label_source.get("label", "A"),
            "confidence": round(float(np.mean([float(left.get("confidence", 0.5)), float(right.get("confidence", 0.5))])), 3),
            "source_features": sources,
        }

    changed = True
    while changed and len(segs) > 1:
        changed = False
        out: list[dict] = []
        i = 0
        while i < len(segs):
            seg = segs[i]
            if float(seg["end"]) - float(seg["start"]) >= min_duration_seconds:
                out.append(seg)
                i += 1
                continue
            changed = True
            if i == 0:
                out.append(merge(seg, segs[i + 1]))
                i += 2
            elif i == len(segs) - 1:
                out[-1] = merge(out[-1], seg)
                i += 1
            else:
                left_dur = float(out[-1]["end"]) - float(out[-1]["start"])
                right_dur = float(segs[i + 1]["end"]) - float(segs[i + 1]["start"])
                if left_dur >= right_dur:
                    out[-1] = merge(out[-1], seg)
                    i += 1
                else:
                    out.append(merge(seg, segs[i + 1]))
                    i += 2
        segs = out

    if duration_seconds is not None and segs:
        segs[-1]["end"] = round(min(float(segs[-1]["end"]), float(duration_seconds)), 2)
    return segs


def make_boundary_dicts(
    boundaries: Iterable[Any],
    source: str,
    duration_seconds: float | None = None,
    min_gap_seconds: float = 0.1,
    confidence: float = 1.0,
    include_edges: bool = True,
) -> list[dict]:
    times = normalize_boundaries(
        boundaries,
        duration_seconds,
        min_gap_seconds=min_gap_seconds,
        include_edges=include_edges,
    )
    return [
        {
            "time": round(float(t), 3),
            "confidence": confidence,
            "source": source,
            "sources": [source],
        }
        for t in times
    ]


def normalize_algorithm_result(
    *,
    task_id: str,
    status: str,
    worker_type: str,
    algorithm: str,
    duration_seconds: float | None,
    boundaries: list[dict] | list[float] | None,
    segments: list[dict] | None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_algorithm = canonical_algorithm_name(algorithm)
    duration = _clean_time(duration_seconds)

    segment_dicts = [dict(s) for s in (segments or [])]
    if not segment_dicts and boundaries and duration:
        segment_dicts = boundaries_to_segments(boundaries, duration)

    if not boundaries and segment_dicts:
        internal = segments_to_internal_boundaries(segment_dicts, edge_margin_seconds=0.0)
        edges = [0.0, *internal]
        if duration:
            edges.append(duration)
        boundaries = edges

    boundary_dicts: list[Boundary] = []
    include_edges = bool(boundaries) or duration is not None
    normalized_times = normalize_boundaries(boundaries or [], duration, include_edges=include_edges)
    raw_boundary_items = [b for b in (boundaries or []) if isinstance(b, dict)]
    for t in normalized_times:
        raw = _closest_boundary_item(t, raw_boundary_items)
        raw_sources = raw.get("sources") if raw else None
        source = raw.get("source") if raw else canonical_algorithm
        if isinstance(source, list):
            source = source[0] if source else canonical_algorithm
        if isinstance(raw_sources, list):
            sources = [str(s) for s in raw_sources]
        elif raw_sources:
            sources = [str(raw_sources)]
        elif source:
            sources = [str(source)]
        else:
            sources = [canonical_algorithm]
        metadata = dict(raw.get("metadata") or {}) if raw else {}
        boundary_dicts.append(
            Boundary(
                time=round(float(t), 3),
                confidence=float(raw.get("confidence", 1.0)) if raw else 1.0,
                source=str(source) if source else canonical_algorithm,
                sources=sources,
                metadata=metadata,
            )
        )

    segment_models: list[Segment] = []
    for seg in segment_dicts:
        structural = str(seg.get("structural_label") or seg.get("label") or "A")
        semantic = seg.get("semantic_label") or seg.get("section_type")
        segment_models.append(
            Segment(
                start=round(float(seg.get("start", 0.0) or 0.0), 2),
                end=round(float(seg.get("end", 0.0) or 0.0), 2),
                label=structural,
                structural_label=structural,
                semantic_label=semantic,
                semantic_confidence=seg.get("semantic_confidence"),
                semantic_reason=seg.get("semantic_reason"),
                confidence=seg.get("confidence"),
                source_features=list(seg.get("source_features") or []),
                cluster_id=seg.get("cluster_id"),
                label_confidence=seg.get("label_confidence"),
                label_method=seg.get("label_method"),
            )
        )

    return AlgorithmResult(
        task_id=task_id,
        status=status,
        worker_type=worker_type,
        algorithm=canonical_algorithm,
        duration_seconds=round(duration, 3) if duration is not None else None,
        boundaries=boundary_dicts,
        segments=segment_models,
        diagnostics=diagnostics or {},
    ).to_dict()


def _closest_boundary_item(t: float, raw_items: list[dict], window_seconds: float = 0.25) -> dict | None:
    best: dict | None = None
    best_dist = float("inf")
    for item in raw_items:
        raw_t = _clean_time(item.get("time"))
        if raw_t is None:
            continue
        dist = abs(float(raw_t) - float(t))
        if dist <= window_seconds and dist < best_dist:
            best = item
            best_dist = dist
    return best


def extract_segments(result_or_segments: Any) -> list[dict]:
    if isinstance(result_or_segments, dict):
        if isinstance(result_or_segments.get("segments"), list):
            return result_or_segments["segments"]
        return []
    if isinstance(result_or_segments, list):
        return result_or_segments
    return []
