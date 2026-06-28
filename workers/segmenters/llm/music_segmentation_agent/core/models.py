"""
Pydantic v2 data models for the music segmentation agent.

All timestamps are represented both as seconds (float) and as "MM:SS" strings for
human-readable display. Confidence scores are always in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to 'MM:SS' string (truncated, not rounded)."""
    total_sec = int(max(0.0, seconds))
    minutes = total_sec // 60
    secs = total_sec % 60
    return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class AudioMetadata(BaseModel):
    """Metadata derived from audio loading and feature extraction."""

    file_path: str
    duration_seconds: float
    sample_rate: int
    estimated_bpm: float
    beat_times: list[float]   # seconds from start of active region
    active_start: float       # active region start (seconds from file start)
    active_end: float         # active region end (seconds from file start)


class CandidateBoundary(BaseModel):
    """A single boundary candidate produced by an audio-feature tool."""

    time_seconds: float
    source: list[str]         # e.g. ["rms", "onset_flux"]
    confidence: float = Field(ge=0.0, le=1.0)


class PredictedSegment(BaseModel):
    """A segment selected and labelled by the LLM decision step."""

    start: str                # "MM:SS"
    end: str                  # "MM:SS"
    start_seconds: float
    end_seconds: float
    label: str                # "Intro", "Verse", "Chorus", etc.
    confidence: float = Field(ge=0.0, le=1.0)
    source_features: list[str]
    reason: str


class SalamiSegment(BaseModel):
    """A ground-truth segment parsed from a SALAMI annotation file."""

    start: str
    end: str
    start_seconds: float
    end_seconds: float
    label: str


class EvaluationResult(BaseModel):
    """Boundary detection and label accuracy metrics."""

    tolerance_seconds: float
    boundary_precision: float
    boundary_recall: float
    boundary_f_measure: float
    label_accuracy: float
    segment_iou: float = 0.0
    over_segmentation_notes: list[str]
    under_segmentation_notes: list[str]


class SegmentationResult(BaseModel):
    """Top-level output returned by SegmentationAgent.run()."""

    track_id: Optional[str]
    duration: str             # "MM:SS"
    estimated_bpm: float
    candidate_boundaries: list[CandidateBoundary]
    predicted_segments: list[PredictedSegment]
    salami_ground_truth: list[SalamiSegment]
    evaluation: EvaluationResult
    agent_explanation: str
