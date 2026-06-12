"""
LangChain tool definitions for the music segmentation agent.

Each tool wraps a deterministic audio-feature function. Tools always return
JSON strings (LangChain convention for tool outputs). Tools never invoke the
LLM — they are pure audio-processing wrappers.

Tool input shapes:
- Simple tools: take `file_path: str` directly.
- Complex tools: take a JSON string that is parsed internally.

All tools are available via get_all_tools().
"""

from __future__ import annotations

import json
import traceback

import numpy as np
from langchain_core.tools import tool

from ..core.audio_loader import load_audio, get_duration
from ..core.silence_removal import detect_active_region
from ..core.feature_extractor import FeatureExtractor
from ..core.filtering import filter_candidates
from ..core.fusion import BoundaryFusion
from ..core.evaluator import Evaluator
from ..core.models import CandidateBoundary, seconds_to_mmss
from ..features.rms import extract_rms_boundaries
from ..features.tempo import extract_tempo
from ..features.beat_detection import extract_beat_boundaries
from ..features.chord_progression import extract_chord_boundaries
from ..features.lyrics import extract_lyric_boundaries
from ..salami.annotation_loader import SalamiAnnotationLoader
from shared.logger import get_logger

logger = get_logger("agent.tools")


# ---------------------------------------------------------------------------
# Helper: serialise CandidateBoundary list to JSON-safe list of dicts
# ---------------------------------------------------------------------------

def _boundaries_to_dicts(candidates: list[CandidateBoundary]) -> list[dict]:
    return [
        {
            "time_seconds": c.time_seconds,
            "source": c.source,
            "confidence": c.confidence,
        }
        for c in candidates
    ]


def _safe_run(fn, *args, **kwargs) -> str:
    """Run *fn* and return JSON result, or a JSON error string on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Tool error: %s\n%s", error_msg, traceback.format_exc())
        return json.dumps({"error": error_msg})


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

@tool
def silence_removal_tool(file_path: str) -> str:
    """
    Detect the active (non-silent) region of an audio file.

    Returns JSON with: active_start_seconds, active_end_seconds, duration_seconds.
    """
    def _run(fp: str) -> str:
        y, sr = load_audio(fp)
        _, start_sec, end_sec = detect_active_region(y, sr)
        total_dur = get_duration(y, sr)
        return json.dumps({
            "active_start_seconds": round(start_sec, 3),
            "active_end_seconds": round(end_sec, 3),
            "duration_seconds": round(total_dur, 3),
        })
    return _safe_run(_run, file_path)


@tool
def rms_feature_tool(file_path: str) -> str:
    """
    Extract RMS-energy-based boundary candidates from an audio file.

    Returns JSON list of {time_seconds, source, confidence} objects.
    """
    def _run(fp: str) -> str:
        y, sr = load_audio(fp)
        y_active, start_sec, _ = detect_active_region(y, sr)
        candidates = extract_rms_boundaries(y_active, sr)
        # Shift times back to full-track timeline.
        shifted = [
            CandidateBoundary(
                time_seconds=round(c.time_seconds + start_sec, 3),
                source=c.source,
                confidence=c.confidence,
            )
            for c in candidates
        ]
        return json.dumps(_boundaries_to_dicts(shifted))
    return _safe_run(_run, file_path)


@tool
def tempo_feature_tool(file_path: str) -> str:
    """
    Estimate BPM and extract beat positions from an audio file.

    Returns JSON with: bpm (float), beat_times (list[float]).
    """
    def _run(fp: str) -> str:
        y, sr = load_audio(fp)
        y_active, start_sec, _ = detect_active_region(y, sr)
        bpm, beat_times = extract_tempo(y_active, sr)
        shifted = [round(float(t) + start_sec, 3) for t in beat_times]
        return json.dumps({"bpm": round(bpm, 2), "beat_times": shifted})
    return _safe_run(_run, file_path)


@tool
def beat_detection_tool(file_path: str) -> str:
    """
    Extract beat-phrase boundary candidates from an audio file.

    Returns JSON list of {time_seconds, source, confidence} objects.
    """
    def _run(fp: str) -> str:
        y, sr = load_audio(fp)
        y_active, start_sec, _ = detect_active_region(y, sr)
        candidates = extract_beat_boundaries(y_active, sr)
        shifted = [
            CandidateBoundary(
                time_seconds=round(c.time_seconds + start_sec, 3),
                source=c.source,
                confidence=c.confidence,
            )
            for c in candidates
        ]
        return json.dumps(_boundaries_to_dicts(shifted))
    return _safe_run(_run, file_path)


@tool
def chord_progression_tool(file_path: str) -> str:
    """
    Extract chord-change boundary candidates from an audio file using chroma-CENS.

    Returns JSON list of {time_seconds, source, confidence} objects.
    """
    def _run(fp: str) -> str:
        y, sr = load_audio(fp)
        y_active, start_sec, _ = detect_active_region(y, sr)
        candidates = extract_chord_boundaries(y_active, sr)
        shifted = [
            CandidateBoundary(
                time_seconds=round(c.time_seconds + start_sec, 3),
                source=c.source,
                confidence=c.confidence,
            )
            for c in candidates
        ]
        return json.dumps(_boundaries_to_dicts(shifted))
    return _safe_run(_run, file_path)


@tool
def lyrics_feature_tool(input_json: str) -> str:
    """
    Extract lyric-based boundary candidates.

    Input JSON: {"file_path": "...", "timed_lyrics": [{"time_seconds": float, "text": str}, ...]}.
    Returns JSON list of {time_seconds, source, confidence} objects.
    """
    def _run(raw: str) -> str:
        data = json.loads(raw)
        fp = data["file_path"]
        timed_lyrics = data.get("timed_lyrics", [])

        y, sr = load_audio(fp)
        total_dur = get_duration(y, sr)
        candidates = extract_lyric_boundaries(timed_lyrics, total_duration=total_dur)
        return json.dumps(_boundaries_to_dicts(candidates))
    return _safe_run(_run, input_json)


@tool
def filtering_tool(candidates_json: str) -> str:
    """
    Filter a list of candidate boundaries by duration gap and confidence.

    Input JSON: {"candidates": [...], "min_duration_sec": float, "max_candidates": int,
                 "min_confidence": float}.
    Returns filtered JSON list.
    """
    def _run(raw: str) -> str:
        data = json.loads(raw)
        raw_candidates = [
            CandidateBoundary(**c) for c in data.get("candidates", [])
        ]
        filtered = filter_candidates(
            raw_candidates,
            min_duration_sec=float(data.get("min_duration_sec", 8.0)),
            max_candidates=int(data.get("max_candidates", 20)),
            min_confidence=float(data.get("min_confidence", 0.25)),
        )
        return json.dumps(_boundaries_to_dicts(filtered))
    return _safe_run(_run, candidates_json)


@tool
def feature_segmentation_tool(file_path: str) -> str:
    """
    Run full feature extraction and derive RMS + onset-based boundary candidates.

    Returns JSON list of {time_seconds, source, confidence} objects.
    """
    def _run(fp: str) -> str:
        from ..core.feature_segmentation import FeatureSegmentationExtractor
        y, sr = load_audio(fp)
        y_active, start_sec, end_sec = detect_active_region(y, sr)
        total_dur = float(end_sec - start_sec)

        extractor = FeatureExtractor()
        features = extractor.extract_all(y_active, sr)

        seg_extractor = FeatureSegmentationExtractor()
        candidates = seg_extractor.extract_from_features(
            features, sr=sr, hop_length=512, total_duration=total_dur
        )
        shifted = [
            CandidateBoundary(
                time_seconds=round(c.time_seconds + start_sec, 3),
                source=c.source,
                confidence=c.confidence,
            )
            for c in candidates
        ]
        return json.dumps(_boundaries_to_dicts(shifted))
    return _safe_run(_run, file_path)


@tool
def fusion_tool(input_json: str) -> str:
    """
    Fuse multiple candidate lists using weighted scoring.

    Input JSON: {
      "candidate_lists": [[{time_seconds, source, confidence}, ...], ...],
      "weights": {"rms": 0.06, ...},
      "merge_window_sec": 1.75,
      "threshold": 0.30
    }.
    Returns fused JSON list.
    """
    def _run(raw: str) -> str:
        data = json.loads(raw)
        candidate_lists = [
            [CandidateBoundary(**c) for c in clist]
            for clist in data.get("candidate_lists", [])
        ]
        fuser = BoundaryFusion()
        fused = fuser.fuse(
            candidate_lists,
            weights=data.get("weights"),
            merge_window_sec=float(data.get("merge_window_sec", 1.75)),
            threshold=float(data.get("threshold", 0.30)),
        )
        return json.dumps(_boundaries_to_dicts(fused))
    return _safe_run(_run, input_json)


@tool
def salami_annotation_loader_tool(annotation_path: str) -> str:
    """
    Load and parse a SALAMI annotation file.

    Returns JSON list of {start, end, start_seconds, end_seconds, label} objects.
    """
    def _run(path: str) -> str:
        loader = SalamiAnnotationLoader()
        segments = loader.load(path)
        return json.dumps([s.model_dump() for s in segments])
    return _safe_run(_run, annotation_path)


@tool
def evaluation_tool(input_json: str) -> str:
    """
    Evaluate predicted segments against SALAMI ground truth.

    Input JSON: {
      "predicted_segments": [{start, end, start_seconds, end_seconds, label, ...}, ...],
      "ground_truth_segments": [{start, end, start_seconds, end_seconds, label}, ...],
      "tolerance_sec": 3.0
    }.
    Returns evaluation metrics JSON.
    """
    def _run(raw: str) -> str:
        from ..core.models import PredictedSegment, SalamiSegment
        data = json.loads(raw)
        predicted = [PredictedSegment(**s) for s in data.get("predicted_segments", [])]
        ground_truth = [SalamiSegment(**s) for s in data.get("ground_truth_segments", [])]
        tolerance = float(data.get("tolerance_sec", 3.0))

        ev = Evaluator()
        result = ev.evaluate(predicted, ground_truth, tolerance_sec=tolerance)
        return json.dumps(result.model_dump())
    return _safe_run(_run, input_json)


@tool
def report_generation_tool(input_json: str) -> str:
    """
    Generate a human-readable segmentation report.

    Input JSON: {
      "track_id": str | null,
      "duration_seconds": float,
      "bpm": float,
      "predicted_segments": [...],
      "evaluation": {...},
      "agent_explanation": str
    }.
    Returns a formatted text report as a JSON string {"report": "..."}.
    """
    def _run(raw: str) -> str:
        data = json.loads(raw)
        track_id = data.get("track_id", "Unknown")
        duration = data.get("duration_seconds", 0.0)
        bpm = data.get("bpm", 0.0)
        segments = data.get("predicted_segments", [])
        evaluation = data.get("evaluation", {})
        explanation = data.get("agent_explanation", "")

        lines = [
            "=" * 60,
            f"MUSIC SEGMENTATION REPORT",
            f"Track ID: {track_id}",
            f"Duration: {seconds_to_mmss(duration)} ({duration:.1f}s)",
            f"Estimated BPM: {bpm:.1f}",
            "=" * 60,
            "",
            "PREDICTED SEGMENTS",
            "-" * 40,
        ]
        for i, seg in enumerate(segments, start=1):
            lines.append(
                f"  {i:2d}. [{seg.get('start', '?')} – {seg.get('end', '?')}]  "
                f"{seg.get('label', '?'):15s}  conf={seg.get('confidence', 0):.2f}  "
                f"sources={seg.get('source_features', [])}"
            )
        lines += [
            "",
            "EVALUATION",
            "-" * 40,
            f"  Tolerance:  {evaluation.get('tolerance_seconds', 0):.1f}s",
            f"  Precision:  {evaluation.get('boundary_precision', 0):.4f}",
            f"  Recall:     {evaluation.get('boundary_recall', 0):.4f}",
            f"  F-Measure:  {evaluation.get('boundary_f_measure', 0):.4f}",
            f"  Label Acc:  {evaluation.get('label_accuracy', 0):.4f}",
            "",
            "AGENT EXPLANATION",
            "-" * 40,
            explanation,
            "=" * 60,
        ]
        report = "\n".join(lines)
        return json.dumps({"report": report})
    return _safe_run(_run, input_json)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def get_all_tools() -> list:
    """Return all registered segmentation agent tools."""
    return [
        silence_removal_tool,
        rms_feature_tool,
        tempo_feature_tool,
        beat_detection_tool,
        chord_progression_tool,
        lyrics_feature_tool,
        filtering_tool,
        feature_segmentation_tool,
        fusion_tool,
        salami_annotation_loader_tool,
        evaluation_tool,
        report_generation_tool,
    ]
