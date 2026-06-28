from __future__ import annotations

from typing import Any

import numpy as np

from workers.core.labeling.heuristic import apply_two_layer_labels
from shared.segmentation.utils import (
    BASELINE_ALGORITHMS,
    boundaries_to_segments,
    canonical_algorithm_name,
    enforce_min_segment_duration,
    extract_segments,
    normalize_algorithm_result,
    segments_to_internal_boundaries,
)


DEFAULT_ALGORITHM_WEIGHTS = {
    "custom_librosa": 0.35,
    "scluster": 0.30,
    "cnmf": 0.20,
    "foote": 0.15,
}


def _collect_leading_silence_ends(algorithm_results: dict[str, Any]) -> list[float]:
    """Return end-times of leading silence segments detected by any algorithm.

    Nearby times (within 2 s) are merged into a single median value so that
    small inter-algorithm disagreements don't create duplicate silence chunks.
    """
    raw: list[float] = []
    for result in algorithm_results.values():
        for seg in extract_segments(result):
            if seg.get("section_type") == "Silence" or seg.get("semantic_label") == "Silence":
                seg_start = float(seg.get("start", 0.0) or 0.0)
                t = round(float(seg.get("end", 0.0) or 0.0), 3)
                if seg_start < 2.0 and 0.5 < t < 30.0:
                    raw.append(t)
    if not raw:
        return []
    raw.sort()
    clusters: list[list[float]] = [[raw[0]]]
    for t in raw[1:]:
        if t - clusters[-1][-1] <= 2.0:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return [round(float(np.median(cluster)), 3) for cluster in clusters]


def _reinsert_silence_segments(segments: list[dict], silence_end_times: list[float]) -> list[dict]:
    """Re-split fused segments at leading-silence boundaries that were dropped during fusion."""
    if not silence_end_times or not segments:
        return segments
    result = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        splits = sorted(t for t in silence_end_times if start < t < end)
        if not splits:
            result.append(seg)
            continue
        boundaries = [start] + splits + [end]
        for i in range(len(boundaries) - 1):
            chunk = dict(seg)
            chunk["start"] = round(boundaries[i], 3)
            chunk["end"] = round(boundaries[i + 1], 3)
            if boundaries[i + 1] in splits:
                # Chunk ending at a silence-end boundary IS the silence portion
                chunk["section_type"] = "Silence"
                chunk["semantic_label"] = "Silence"
                chunk["semantic_confidence"] = 0.85
                chunk["semantic_reason"] = "Leading silence re-inserted from individual algorithm detection."
            result.append(chunk)
    return result


def _duration_from_results(algorithm_results: dict[str, Any]) -> float:
    durations = []
    for result in algorithm_results.values():
        if isinstance(result, dict) and result.get("duration_seconds"):
            durations.append(float(result["duration_seconds"]))
        segments = extract_segments(result)
        if segments:
            durations.append(max(float(s.get("end", 0.0) or 0.0) for s in segments))
    return round(max(durations), 3) if durations else 0.0


def _internal_boundaries_from_result(result: Any) -> list[dict]:
    if isinstance(result, dict) and isinstance(result.get("boundaries"), list):
        duration = float(result.get("duration_seconds") or 0.0)
        out = []
        for boundary in result["boundaries"]:
            try:
                t = float(boundary.get("time"))
            except (TypeError, ValueError):
                continue
            if t <= 0.5:
                continue
            if duration > 0 and t >= duration - 0.5:
                continue
            out.append(
                {
                    "time": round(t, 3),
                    "confidence": float(boundary.get("confidence", 1.0) or 1.0),
                }
            )
        if out:
            return out

    return [{"time": t, "confidence": 1.0} for t in segments_to_internal_boundaries(extract_segments(result))]


def _group_votes(votes: list[dict], merge_window_seconds: float) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for vote in sorted(votes, key=lambda item: float(item["time"])):
        if not groups:
            groups.append([vote])
            continue
        current_center = float(np.mean([v["time"] for v in groups[-1]]))
        if abs(float(vote["time"]) - current_center) <= merge_window_seconds:
            groups[-1].append(vote)
        else:
            groups.append([vote])
    return groups


def _choose_fused_time(
    group: list[dict],
    weights: dict[str, float],
    anchor_strategy: str,
    merge_window_seconds: float,
) -> float:
    if anchor_strategy == "custom_snap":
        custom_votes = [v for v in group if v["algorithm"] == "custom_librosa"]
        if custom_votes:
            raw_times = np.asarray([v["time"] for v in group], dtype=float)
            custom = max(custom_votes, key=lambda v: v.get("confidence", 1.0))
            if np.any(np.abs(raw_times - float(custom["time"])) <= merge_window_seconds):
                return round(float(custom["time"]), 3)

    numer = 0.0
    denom = 0.0
    for vote in group:
        weight = weights.get(vote["algorithm"], 0.0) * float(vote.get("confidence", 1.0) or 1.0)
        numer += weight * float(vote["time"])
        denom += weight
    if denom <= 0:
        return round(float(np.mean([v["time"] for v in group])), 3)
    return round(numer / denom, 3)


def fuse_algorithm_results(
    algorithm_results: dict[str, Any],
    *,
    task_id: str,
    params: dict | None = None,
    file_path: str | None = None,
) -> dict:
    params = params or {}
    weights = dict(DEFAULT_ALGORITHM_WEIGHTS)
    for key, value in (params.get("weights") or {}).items():
        algorithm = canonical_algorithm_name(key)
        if algorithm in weights:
            try:
                weights[algorithm] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue

    merge_window_seconds = float(params.get("merge_window_seconds", 2.5))
    threshold = float(params.get("threshold", 0.30))
    min_segment_duration = float(params.get("min_segment_duration_seconds", 5.0))
    anchor_strategy = str(params.get("anchor_strategy") or "custom_snap")
    required_vote_count = int(params.get("required_vote_count", 1))
    semantic_enabled = bool(params.get("semantic_labeling_enabled", True))

    normalized_inputs = {
        canonical_algorithm_name(algo): result
        for algo, result in (algorithm_results or {}).items()
    }
    duration = _duration_from_results(normalized_inputs)
    silence_end_times = _collect_leading_silence_ends(normalized_inputs)

    votes: list[dict] = []
    failed_algorithms: list[str] = list(params.get("failed_or_missing_algorithms") or [])
    for algorithm in BASELINE_ALGORITHMS:
        result = normalized_inputs.get(algorithm)
        if not result:
            if algorithm not in failed_algorithms:
                failed_algorithms.append(algorithm)
            continue
        for boundary in _internal_boundaries_from_result(result):
            votes.append(
                {
                    "algorithm": algorithm,
                    "time": float(boundary["time"]),
                    "confidence": float(boundary.get("confidence", 1.0) or 1.0),
                    "weight": weights.get(algorithm, 0.0),
                }
            )

    groups = _group_votes(votes, merge_window_seconds)
    accepted_boundaries: list[dict] = []
    diagnostics_groups: list[dict] = []

    for group in groups:
        best_by_algorithm: dict[str, dict] = {}
        for vote in group:
            algorithm = vote["algorithm"]
            if algorithm not in best_by_algorithm or vote["confidence"] > best_by_algorithm[algorithm]["confidence"]:
                best_by_algorithm[algorithm] = vote

        sources = sorted(best_by_algorithm)
        score = sum(weights.get(algorithm, 0.0) * float(vote.get("confidence", 1.0)) for algorithm, vote in best_by_algorithm.items())
        fused_time = _choose_fused_time(list(best_by_algorithm.values()), weights, anchor_strategy, merge_window_seconds)
        accepted = score >= threshold or len(sources) >= required_vote_count
        group_diag = {
            "fused_time": fused_time,
            "score": round(float(score), 4),
            "sources": sources,
            "raw_times": [
                {
                    "algorithm": vote["algorithm"],
                    "time": round(float(vote["time"]), 3),
                    "confidence": round(float(vote.get("confidence", 1.0)), 3),
                }
                for vote in group
            ],
            "accepted": accepted,
        }
        diagnostics_groups.append(group_diag)
        if accepted:
            accepted_boundaries.append(
                {
                    "time": fused_time,
                    "confidence": round(float(min(1.0, score)), 3),
                    "source": "algorithm_fusion",
                    "sources": sources,
                    "metadata": {"score": round(float(score), 4), "raw_times": group_diag["raw_times"]},
                }
            )

    accepted_boundaries = _dedupe_and_enforce_boundaries(
        accepted_boundaries,
        duration_seconds=duration,
        min_segment_duration=min_segment_duration,
    )
    segments = boundaries_to_segments(
        accepted_boundaries,
        duration,
        min_gap_seconds=0.1,
        boundary_metadata=accepted_boundaries,
    )
    segments = enforce_min_segment_duration(segments, min_segment_duration, duration_seconds=duration)
    segments = apply_two_layer_labels(
        segments,
        file_path=file_path,
        duration_seconds=duration,
        semantic_enabled=semantic_enabled,
        method_hint="fusion_boundary_voting",
    )
    segments = _reinsert_silence_segments(segments, silence_end_times)

    diagnostics = {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "merge_window_seconds": merge_window_seconds,
        "threshold": threshold,
        "min_segment_duration_seconds": min_segment_duration,
        "anchor_strategy": anchor_strategy,
        "required_vote_count": required_vote_count,
        "input_algorithms": list(BASELINE_ALGORITHMS),
        "available_algorithms": sorted(normalized_inputs),
        "failed_or_missing_algorithms": failed_algorithms,
        "boundary_groups": diagnostics_groups,
    }

    return normalize_algorithm_result(
        task_id=task_id,
        status="completed",
        worker_type="fusion",
        algorithm="fusion",
        duration_seconds=duration,
        boundaries=accepted_boundaries,
        segments=segments,
        diagnostics=diagnostics,
    )


def _dedupe_and_enforce_boundaries(
    boundaries: list[dict],
    *,
    duration_seconds: float,
    min_segment_duration: float,
) -> list[dict]:
    kept: list[dict] = []
    last_time = 0.0
    for boundary in sorted(boundaries, key=lambda item: float(item["time"])):
        t = float(boundary["time"])
        if t <= min_segment_duration:
            continue
        if duration_seconds > 0 and t >= duration_seconds - min_segment_duration:
            continue
        if kept and t - last_time < min_segment_duration:
            if float(boundary.get("confidence", 0.0)) > float(kept[-1].get("confidence", 0.0)):
                kept[-1] = boundary
                last_time = t
            continue
        kept.append(boundary)
        last_time = t
    return kept
