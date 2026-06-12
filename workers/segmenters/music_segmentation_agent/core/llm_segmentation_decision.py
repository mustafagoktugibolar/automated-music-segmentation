"""
LLM-assisted boundary selection and segment labelling.

The LLM is shown a JSON array of audio-feature-derived candidate boundaries and
asked to SELECT which ones to keep (plus assign labels and reasons). It is
explicitly forbidden from fabricating timestamps.

After the LLM responds, every selected time is snapped to the nearest provided
candidate within a 2-second window. Selections that cannot be snapped are
discarded with a warning.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from pydantic import BaseModel, Field

from .models import AudioMetadata, CandidateBoundary, PredictedSegment, seconds_to_mmss
from ..agent.prompts import SEGMENTATION_DECISION_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from shared.logger import get_logger

logger = get_logger("llm_segmentation_decision")

_MAX_SNAP_SEC: float = 2.0


# ---------------------------------------------------------------------------
# Structured output schemas for the LLM
# ---------------------------------------------------------------------------

class LLMBoundarySelection(BaseModel):
    """One boundary selected (not invented) by the LLM."""

    time_seconds: float = Field(
        description="Must be one of the provided candidate times (or the nearest candidate within 2s)."
    )
    label: str = Field(description="Section label, e.g. 'Intro', 'Verse', 'Chorus'.")
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(description="Audio feature sources that support this boundary.")
    reason: str = Field(description="Brief explanation of why this boundary was selected.")


class LLMDecisionOutput(BaseModel):
    """Structured output from the segmentation decision LLM call."""

    selected_boundaries: list[LLMBoundarySelection] = Field(
        description="Subset of the provided candidate boundaries, with labels."
    )
    explanation: str = Field(description="Overall explanation of the segmentation decision.")


# ---------------------------------------------------------------------------
# Decision class
# ---------------------------------------------------------------------------

class LLMSegmentationDecision:
    """
    Wraps the LLM call that selects + labels boundaries from audio candidates.

    Critically:
    - The LLM sees only pre-computed candidate times — it cannot invent new ones.
    - Every output time is snapped to the nearest valid candidate.
    - Unsnappable times are silently discarded.
    """

    def __init__(self, llm) -> None:
        """
        Parameters
        ----------
        llm : A LangChain ChatAnthropic (or compatible) instance.
        """
        self.llm = llm
        self._structured_llm = llm.with_structured_output(LLMDecisionOutput)

    def decide(
        self,
        audio_metadata: AudioMetadata,
        candidates: list[CandidateBoundary],
        target_boundary_count: int | None = None,
        min_boundary_count: int | None = None,
        max_boundary_count: int | None = None,
    ) -> tuple[list[PredictedSegment], str]:
        """
        Ask the LLM to select and label boundaries.

        Parameters
        ----------
        audio_metadata : Track metadata including beat_times and duration.
        candidates     : Pre-filtered list of CandidateBoundary objects.
        target_boundary_count : Desired number of selected internal boundaries.
        min_boundary_count    : Minimum selected internal boundaries after
                                deterministic post-processing.
        max_boundary_count    : Maximum selected internal boundaries after
                                deterministic post-processing.

        Returns
        -------
        (predicted_segments, explanation_text)
          - predicted_segments are ordered by time, with t=0 implied as start.
          - explanation_text is the LLM's overall reasoning.
        """
        if not candidates:
            logger.warning("No candidates to present to LLM; returning empty.")
            return [], "No audio candidates were available for LLM selection."

        # Build the user prompt with candidates as JSON.
        bounds = self._resolve_boundary_count_bounds(
            audio_metadata=audio_metadata,
            candidates=candidates,
            target_boundary_count=target_boundary_count,
            min_boundary_count=min_boundary_count,
            max_boundary_count=max_boundary_count,
        )
        prompt = self._build_prompt(audio_metadata, candidates, bounds)

        try:
            llm_output: LLMDecisionOutput = self._structured_llm.invoke(
                [
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            # Fall back to using all candidates as boundaries.
            return self._fallback_segments(audio_metadata, candidates), (
                f"LLM call failed ({exc}). Using all audio candidates as boundaries."
            )

        # Validate and snap each selected boundary.
        valid_times = sorted({round(c.time_seconds, 3) for c in candidates})

        snapped_selections: list[LLMBoundarySelection] = []
        for sel in llm_output.selected_boundaries:
            snapped_t = self._snap_to_nearest(sel.time_seconds, valid_times)
            if snapped_t is None:
                logger.warning(
                    "LLM selected %.2fs but no valid candidate within ±%.1fs; discarding.",
                    sel.time_seconds,
                    _MAX_SNAP_SEC,
                )
                continue
            snapped_selections.append(
                LLMBoundarySelection(
                    time_seconds=snapped_t,
                    label=sel.label,
                    confidence=sel.confidence,
                    sources=sel.sources,
                    reason=sel.reason,
                )
            )

        # Remove duplicates (same snapped time).
        seen_times: set[float] = set()
        unique_selections: list[LLMBoundarySelection] = []
        for sel in sorted(snapped_selections, key=lambda s: s.time_seconds):
            key = round(sel.time_seconds, 2)
            if key not in seen_times:
                seen_times.add(key)
                unique_selections.append(sel)

        if not unique_selections:
            logger.warning("All LLM selections were invalid; falling back to candidates.")
            return self._fallback_segments(audio_metadata, candidates), (
                llm_output.explanation + " [All LLM selections were discarded after validation.]"
            )

        unique_selections, guard_note = self._stabilize_selection_count(
            selections=unique_selections,
            candidates=candidates,
            min_count=bounds["min"],
            max_count=bounds["max"],
        )

        predicted = self._build_segments(unique_selections, audio_metadata.duration_seconds)
        explanation = llm_output.explanation
        if guard_note:
            explanation = f"{explanation} [{guard_note}]"
        return predicted, explanation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        meta: AudioMetadata,
        candidates: list[CandidateBoundary],
        bounds: dict[str, int],
    ) -> str:
        """Render the segmentation decision prompt with candidate data."""
        candidates_json = json.dumps(
            [
                {
                    "time_seconds": round(c.time_seconds, 2),
                    "source": c.source,
                    "confidence": round(c.confidence, 3),
                }
                for c in sorted(candidates, key=lambda c: c.time_seconds)
            ],
            indent=2,
        )
        beat_sample = sorted(meta.beat_times)[:40]  # limit to first 40 beats for prompt brevity
        beat_json = json.dumps([round(t, 2) for t in beat_sample])

        return SEGMENTATION_DECISION_PROMPT.format(
            duration=round(meta.duration_seconds, 2),
            bpm=round(meta.estimated_bpm, 1),
            active_start=round(meta.active_start, 2),
            active_end=round(meta.active_end, 2),
            candidates_json=candidates_json,
            beat_times_json=beat_json,
            min_boundaries=bounds["min"],
            max_boundaries=bounds["max"],
            target_boundaries=bounds["target"],
        )

    @staticmethod
    def _resolve_boundary_count_bounds(
        audio_metadata: AudioMetadata,
        candidates: list[CandidateBoundary],
        target_boundary_count: int | None,
        min_boundary_count: int | None,
        max_boundary_count: int | None,
    ) -> dict[str, int]:
        candidate_count = len(candidates)
        active_dur = max(
            0.0,
            float(audio_metadata.active_end) - float(audio_metadata.active_start),
        )
        if target_boundary_count is None:
            # Pop/rock SALAMI-style structure is usually closer to 25-40s
            # section spans than the very sparse 45-60s spans LLMs tend to pick.
            target_boundary_count = max(3, round(active_dur / 32.0))

        target = max(0, min(candidate_count, int(target_boundary_count)))
        if min_boundary_count is None:
            min_boundary_count = max(0, target - 1)
        if max_boundary_count is None:
            max_boundary_count = max(target + 2, min_boundary_count)

        min_count = max(0, min(candidate_count, int(min_boundary_count)))
        max_count = max(min_count, min(candidate_count, int(max_boundary_count)))
        target = max(min_count, min(max_count, target))
        return {"min": min_count, "max": max_count, "target": target}

    @staticmethod
    def _snap_to_nearest(
        time_sec: float,
        valid_times: list[float],
        max_snap_sec: float = _MAX_SNAP_SEC,
    ) -> float | None:
        """
        Return the nearest valid time within *max_snap_sec* of *time_sec*.

        Returns None if no valid time is within the window.
        """
        if not valid_times:
            return None
        arr = np.array(valid_times, dtype=np.float64)
        dists = np.abs(arr - time_sec)
        idx = int(np.argmin(dists))
        if float(dists[idx]) <= max_snap_sec:
            return round(float(arr[idx]), 3)
        return None

    @staticmethod
    def _build_segments(
        selections: list[LLMBoundarySelection],
        total_duration: float,
    ) -> list[PredictedSegment]:
        """
        Convert a list of selected boundaries into consecutive PredictedSegment objects.

        The implicit first boundary is t=0 ("start of track") and the last is
        total_duration ("end of track").
        """
        # Build boundary list: 0 → [selected times] → total_duration.
        boundary_times = [0.0] + [s.time_seconds for s in selections] + [total_duration]
        boundary_times = sorted(set(round(t, 3) for t in boundary_times))

        # Map boundary time → LLM selection (for the START of each segment).
        label_map: dict[float, LLMBoundarySelection] = {}
        for sel in selections:
            label_map[round(sel.time_seconds, 3)] = sel

        segments: list[PredictedSegment] = []
        for i in range(len(boundary_times) - 1):
            start_t = boundary_times[i]
            end_t = boundary_times[i + 1]

            # Find the selection that opens this segment (may be start=0 → no selection).
            sel = label_map.get(round(start_t, 3))
            label = sel.label if sel else "Unknown"
            confidence = sel.confidence if sel else 0.5
            sources = sel.sources if sel else []
            reason = sel.reason if sel else "Boundary from audio feature candidates."

            segments.append(
                PredictedSegment(
                    start=seconds_to_mmss(start_t),
                    end=seconds_to_mmss(end_t),
                    start_seconds=start_t,
                    end_seconds=end_t,
                    label=label,
                    confidence=confidence,
                    source_features=sources,
                    reason=reason,
                )
            )

        return segments

    @staticmethod
    def _stabilize_selection_count(
        selections: list[LLMBoundarySelection],
        candidates: list[CandidateBoundary],
        min_count: int,
        max_count: int,
    ) -> tuple[list[LLMBoundarySelection], str]:
        """Keep boundary count in the expected range using candidate evidence."""
        if not selections:
            return selections, ""

        selected_by_time = {round(sel.time_seconds, 3): sel for sel in selections}
        notes: list[str] = []

        if len(selected_by_time) < min_count:
            ranked_candidates = sorted(
                candidates,
                key=LLMSegmentationDecision._candidate_rank,
                reverse=True,
            )
            added = 0
            for cand in ranked_candidates:
                key = round(cand.time_seconds, 3)
                if key in selected_by_time:
                    continue
                selected_by_time[key] = LLMBoundarySelection(
                    time_seconds=key,
                    label="Section",
                    confidence=cand.confidence,
                    sources=cand.source,
                    reason=(
                        "Added by deterministic guard because the LLM selected "
                        "too few structural boundaries for this track length."
                    ),
                )
                added += 1
                if len(selected_by_time) >= min_count:
                    break
            if added:
                notes.append(f"Added {added} high-confidence candidate boundary/boundaries.")

        if len(selected_by_time) > max_count:
            trimmed = sorted(
                selected_by_time.values(),
                key=lambda sel: (
                    float(sel.confidence),
                    len(sel.sources),
                    -float(sel.time_seconds),
                ),
                reverse=True,
            )[:max_count]
            removed = len(selected_by_time) - len(trimmed)
            selected_by_time = {round(sel.time_seconds, 3): sel for sel in trimmed}
            if removed:
                notes.append(f"Removed {removed} lowest-confidence boundary/boundaries.")

        stable = sorted(selected_by_time.values(), key=lambda sel: sel.time_seconds)
        return stable, " ".join(notes)

    @staticmethod
    def _candidate_rank(candidate: CandidateBoundary) -> tuple[float, int, int]:
        structural_sources = {"ssm", "chord_proxy", "beat_phrase", "lyrics"}
        structural_count = len(structural_sources.intersection(candidate.source))
        return (float(candidate.confidence), structural_count, len(candidate.source))

    @staticmethod
    def _fallback_segments(
        meta: AudioMetadata,
        candidates: list[CandidateBoundary],
    ) -> list[PredictedSegment]:
        """Use all candidates as boundaries with generic labels (LLM fallback)."""
        dummy_selections = [
            LLMBoundarySelection(
                time_seconds=c.time_seconds,
                label="Section",
                confidence=c.confidence,
                sources=c.source,
                reason="Audio-feature candidate (LLM fallback).",
            )
            for c in sorted(candidates, key=lambda c: c.time_seconds)
        ]
        return LLMSegmentationDecision._build_segments(dummy_selections, meta.duration_seconds)
