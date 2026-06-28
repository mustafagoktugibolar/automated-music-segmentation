"""
Orchestrator agent for LLM-assisted music segmentation.

Runs a deterministic pipeline to produce grounded boundary candidates, then
delegates the final selection and labelling decision to the LLM. The LLM can
only SELECT from the audio-derived candidates — it cannot invent timestamps.

Pipeline (all deterministic except the LLM decision step):
  1. Load audio
  2. Active-region detection (silence removal)
  3. Feature extraction (Chroma-CENS, MFCC, RMS, onset)
  4. Candidate extraction from RMS, tempo/beat, chord, lyrics
  5. Optional: SSM-based novelty candidates
  6. Boundary fusion
  7. Filtering
  8. LLM decision (select + label)
  9. Positional heuristics (Intro/Outro override)
  10. SALAMI ground truth loading (optional)
  11. Evaluation
  12. Build and return SegmentationResult
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ..core.audio_loader import load_audio, get_duration
from ..core.silence_removal import detect_active_region
from ..core.feature_extractor import FeatureExtractor
from ..core.feature_segmentation import FeatureSegmentationExtractor
from ..core.structure_analyzer import StructureAnalyzer
from ..core.fusion import BoundaryFusion
from ..core.filtering import filter_candidates
from ..core.llm_segmentation_decision import LLMSegmentationDecision
from ..core.evaluator import Evaluator
from ..core.models import (
    AudioMetadata,
    CandidateBoundary,
    PredictedSegment,
    SegmentationResult,
    SalamiSegment,
    seconds_to_mmss,
)
from ..features.rms import extract_rms_boundaries
from ..features.tempo import extract_tempo
from ..features.beat_detection import extract_beat_boundaries
from ..features.chord_progression import extract_chord_boundaries
from ..features.lyrics import extract_lyric_boundaries
from ..salami.annotation_loader import SalamiAnnotationLoader
from ..salami.label_normalizer import normalize_label
from shared.logger import get_logger

logger = get_logger("segmentation_agent")

# Default model name per provider.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}


def _build_llm(provider: str, model_name: str | None):
    """Instantiate a LangChain chat model for the requested provider."""
    provider = provider.lower().strip()
    model = model_name or _DEFAULT_MODELS.get(provider)
    if model is None:
        raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'.")

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required. Install with: pip install langchain-anthropic"
            ) from exc
        logger.info("Using Anthropic LLM: model=%s", model)
        return ChatAnthropic(model=model, temperature=0)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required. Install with: pip install langchain-openai"
            ) from exc
        logger.info("Using OpenAI LLM: model=%s", model)
        return ChatOpenAI(model=model, temperature=0)

    raise ValueError(f"Unsupported provider '{provider}'. Use 'anthropic' or 'openai'.")


# Default feature weights (mirror multi_feature_fusion defaults).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "ssm": 0.42,
    "beat_grid": 0.24,
    "chord_proxy": 0.18,
    "beat_phrase": 0.18,
    "onset_flux": 0.06,
    "rms": 0.06,
    "lyrics": 0.10,
    "beat": 0.02,
}

_EMPTY_EVAL = {
    "tolerance_seconds": 3.0,
    "boundary_precision": 0.0,
    "boundary_recall": 0.0,
    "boundary_f_measure": 0.0,
    "label_accuracy": 0.0,
    "segment_iou": 0.0,
    "over_segmentation_notes": [],
    "under_segmentation_notes": [],
}


class SegmentationAgent:
    """
    LLM-assisted music structure segmentation agent.

    The agent orchestrates deterministic audio-feature tools, fuses their
    outputs into a candidate list, and then asks the LLM to select and label
    the final segments.

    Parameters
    ----------
    llm        : Optional pre-constructed LangChain LLM. If provided,
                 *provider* and *model_name* are ignored.
    provider   : LLM provider — ``"anthropic"`` (default) or ``"openai"``.
    model_name : Model ID. Defaults to ``"claude-sonnet-4-6"`` for Anthropic
                 and ``"gpt-4o"`` for OpenAI.
    tools      : Optional list of LangChain tools (currently unused in the
                 direct-orchestration path, kept for future ReAct extension).
    """

    def __init__(
        self,
        llm=None,
        provider: str = "anthropic",
        model_name: str | None = None,
        tools: list | None = None,
    ) -> None:
        if llm is None:
            llm = _build_llm(provider, model_name)

        self.llm = llm
        self.tools = tools or []
        self._decision = LLMSegmentationDecision(llm)
        self._feature_extractor = FeatureExtractor()
        self._structure_analyzer = StructureAnalyzer()
        self._fuser = BoundaryFusion()
        self._evaluator = Evaluator()
        self._salami_loader = SalamiAnnotationLoader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        file_path: str,
        track_id: str | None = None,
        salami_annotation_path: str | None = None,
        timed_lyrics: list | None = None,
        params: dict | None = None,
    ) -> SegmentationResult:
        """
        Run the full segmentation pipeline.

        Parameters
        ----------
        file_path              : Path to the audio file.
        track_id               : Optional track identifier (for reporting).
        salami_annotation_path : Path to SALAMI annotation file for evaluation.
        timed_lyrics           : List of {time_seconds, text} dicts.
        params                 : Optional pipeline parameter overrides:
                                   min_segment_duration_seconds (float)
                                   feature_weights (dict)
                                   max_candidates (int)
                                   min_confidence (float)
                                   merge_window_sec (float)
                                   fusion_threshold (float)
                                   ssm_kernel_seconds (float)
                                   ssm_smoothing_L (int)

        Returns
        -------
        SegmentationResult
        """
        t_total = time.perf_counter()
        p = params or {}

        min_seg_dur = float(p.get("min_segment_duration_seconds", 8.0))
        max_candidates = int(p.get("max_candidates", 40))
        min_confidence = float(p.get("min_confidence", 0.25))
        merge_window = float(p.get("merge_window_sec", 1.75))
        fusion_threshold = float(p.get("fusion_threshold", 0.28))
        feature_weights = p.get("feature_weights", _DEFAULT_WEIGHTS)
        ssm_kernel_s = float(p.get("ssm_kernel_seconds", 8.0))
        smoothing_L = int(p.get("ssm_smoothing_L", 14))
        target_boundaries = p.get("target_boundaries")
        min_selected_boundaries = p.get("min_selected_boundaries")
        max_selected_boundaries = p.get("max_selected_boundaries")

        logger.info("SegmentationAgent.run: track_id=%s, file=%s", track_id, file_path)

        # ------------------------------------------------------------------
        # Stage 1: Load audio
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        y, sr = load_audio(file_path)
        total_dur = get_duration(y, sr)
        logger.info("[%.2fs] Audio loaded: duration=%.2fs, sr=%d", time.perf_counter() - t0, total_dur, sr)

        # ------------------------------------------------------------------
        # Stage 2: Active-region detection
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        y_active, active_start, active_end = detect_active_region(y, sr)
        active_dur = float(active_end - active_start)
        logger.info(
            "[%.2fs] Active region: %.2fs–%.2fs (%.2fs)",
            time.perf_counter() - t0, active_start, active_end, active_dur
        )

        if active_dur <= 0.0:
            logger.warning("No active audio region; returning empty result.")
            return self._empty_result(track_id, total_dur)

        # ------------------------------------------------------------------
        # Stage 3: Feature extraction
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        features = self._feature_extractor.extract_all(y_active, sr)
        fps = float(features["feature_rate"])
        logger.info("[%.2fs] Features extracted: fps=%.2f", time.perf_counter() - t0, fps)

        # ------------------------------------------------------------------
        # Stage 4: Tempo & beat detection
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        bpm, beat_times_active = extract_tempo(y_active, sr)
        beat_times_full = np.array(
            [round(float(t) + active_start, 3) for t in beat_times_active],
            dtype=np.float32,
        )
        logger.info("[%.2fs] BPM=%.2f, beats=%d", time.perf_counter() - t0, bpm, len(beat_times_full))

        # ------------------------------------------------------------------
        # Stage 5: Candidate extraction from all sources
        # ------------------------------------------------------------------
        all_candidate_lists: list[list[CandidateBoundary]] = []

        # RMS boundaries.
        rms_cands = extract_rms_boundaries(y_active, sr)
        rms_cands = self._shift_candidates(rms_cands, active_start)
        all_candidate_lists.append(rms_cands)
        logger.info("RMS candidates: %d", len(rms_cands))

        # Beat-phrase boundaries.
        beat_cands = extract_beat_boundaries(y_active, sr)
        beat_cands = self._shift_candidates(beat_cands, active_start)
        all_candidate_lists.append(beat_cands)
        logger.info("Beat-phrase candidates: %d", len(beat_cands))

        # Beat-grid structural candidates.
        #
        # SALAMI-style boundaries often land exactly on the beat grid even when
        # RMS/chroma/SSM novelty peaks occur 1-3s before or after the musical
        # section start. Give the LLM those beat-aligned times directly so the
        # strict ±0.5s evaluation has a chance to match.
        beat_grid_cands = self._beat_grid_candidates(
            beat_times_full=beat_times_full,
            active_start=active_start,
            active_end=active_end,
            min_seg_dur=min_seg_dur,
        )
        all_candidate_lists.append(beat_grid_cands)
        logger.info("Beat-grid candidates: %d", len(beat_grid_cands))

        # Chord-progression boundaries.
        chord_cands = extract_chord_boundaries(y_active, sr)
        chord_cands = self._shift_candidates(chord_cands, active_start)
        all_candidate_lists.append(chord_cands)
        logger.info("Chord candidates: %d", len(chord_cands))

        # Feature-segmentation (RMS + onset from feature dict).
        seg_extractor = FeatureSegmentationExtractor()
        fseg_cands = seg_extractor.extract_from_features(
            features, sr=sr, hop_length=512, total_duration=active_dur
        )
        fseg_cands = self._shift_candidates(fseg_cands, active_start)
        all_candidate_lists.append(fseg_cands)
        logger.info("Feature-segmentation candidates: %d", len(fseg_cands))

        # Lyrics boundaries (optional).
        if timed_lyrics:
            lyric_cands = extract_lyric_boundaries(
                timed_lyrics, total_duration=total_dur
            )
            all_candidate_lists.append(lyric_cands)
            logger.info("Lyric candidates: %d", len(lyric_cands))

        # SSM novelty boundaries.
        ssm_cands = self._compute_ssm_candidates(
            features, fps, active_dur, active_start, ssm_kernel_s, smoothing_L, min_seg_dur
        )
        all_candidate_lists.append(ssm_cands)
        logger.info("SSM candidates: %d", len(ssm_cands))

        # ------------------------------------------------------------------
        # Stage 6: Boundary fusion
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        fused = self._fuser.fuse(
            all_candidate_lists,
            weights=feature_weights,
            merge_window_sec=merge_window,
            threshold=fusion_threshold,
        )
        logger.info("[%.2fs] Fused boundaries: %d", time.perf_counter() - t0, len(fused))

        # ------------------------------------------------------------------
        # Stage 7: Filtering
        # ------------------------------------------------------------------
        filtered = filter_candidates(
            fused,
            min_duration_sec=min_seg_dur,
            max_candidates=max_candidates,
            min_confidence=min_confidence,
        )
        logger.info("Filtered candidates: %d", len(filtered))

        # ------------------------------------------------------------------
        # Stage 8: Build AudioMetadata for LLM
        # ------------------------------------------------------------------
        audio_metadata = AudioMetadata(
            file_path=file_path,
            duration_seconds=round(total_dur, 3),
            sample_rate=sr,
            estimated_bpm=bpm,
            beat_times=beat_times_full.tolist(),
            active_start=round(active_start, 3),
            active_end=round(active_end, 3),
        )

        # ------------------------------------------------------------------
        # Stage 9: LLM decision
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            predicted_segments, explanation = self._decision.decide(
                audio_metadata,
                filtered,
                target_boundary_count=(
                    int(target_boundaries) if target_boundaries is not None else None
                ),
                min_boundary_count=(
                    int(min_selected_boundaries)
                    if min_selected_boundaries is not None
                    else None
                ),
                max_boundary_count=(
                    int(max_selected_boundaries)
                    if max_selected_boundaries is not None
                    else None
                ),
            )
        except Exception as exc:
            logger.error("LLM decision failed: %s", exc, exc_info=True)
            predicted_segments = self._decision._fallback_segments(audio_metadata, filtered)
            explanation = f"LLM failed; using audio candidates directly. Error: {exc}"
        logger.info("[%.2fs] LLM decision: %d segments", time.perf_counter() - t0, len(predicted_segments))

        # ------------------------------------------------------------------
        # Stage 10: Positional heuristics (Intro / Outro override)
        # ------------------------------------------------------------------
        predicted_segments = self._apply_positional_heuristics(predicted_segments)

        # ------------------------------------------------------------------
        # Stage 11: SALAMI ground truth
        # ------------------------------------------------------------------
        ground_truth: list[SalamiSegment] = []
        if salami_annotation_path:
            try:
                raw_gt = self._salami_loader.load(salami_annotation_path)
                # Normalise SALAMI labels.
                ground_truth = [
                    SalamiSegment(
                        start=seg.start,
                        end=seg.end,
                        start_seconds=seg.start_seconds,
                        end_seconds=seg.end_seconds,
                        label=normalize_label(seg.label),
                    )
                    for seg in raw_gt
                ]
                logger.info("Loaded %d SALAMI segments.", len(ground_truth))
            except Exception as exc:
                logger.warning("Failed to load SALAMI annotations: %s", exc)

        # ------------------------------------------------------------------
        # Stage 12: Evaluation
        # ------------------------------------------------------------------
        evaluation = self._evaluator.evaluate(
            predicted_segments, ground_truth, tolerance_sec=3.0
        )

        # ------------------------------------------------------------------
        # Stage 13: Assemble result
        # ------------------------------------------------------------------
        logger.info(
            "Pipeline complete in %.2fs: %d segments, F=%.3f",
            time.perf_counter() - t_total,
            len(predicted_segments),
            evaluation.boundary_f_measure,
        )

        return SegmentationResult(
            track_id=track_id,
            duration=seconds_to_mmss(total_dur),
            estimated_bpm=round(bpm, 2),
            candidate_boundaries=filtered,
            predicted_segments=predicted_segments,
            salami_ground_truth=ground_truth,
            evaluation=evaluation,
            agent_explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ssm_candidates(
        self,
        features: dict,
        fps: float,
        active_dur: float,
        active_start: float,
        kernel_s: float,
        smoothing_L: int,
        min_seg_dur: float,
    ) -> list[CandidateBoundary]:
        """Compute SSM novelty curve and extract boundary candidates."""
        try:
            chroma = features["chroma"]
            mfcc = features["mfcc"]
            rms_times = features["rms_times"]

            # Blend chroma + MFCC SSMs.
            S_chroma = self._structure_analyzer.compute_ssm(chroma)
            if np.any(mfcc != 0):
                S_mfcc = self._structure_analyzer.compute_ssm(mfcc)
                S_raw = 0.5 * S_chroma + 0.5 * S_mfcc
            else:
                S_raw = S_chroma

            S_enh = self._structure_analyzer.enhance_ssm(
                S_raw, L=smoothing_L, rho=0.20, penalty=-2.0
            )

            kernel_L = max(8, int(kernel_s * fps))
            novelty = self._structure_analyzer.compute_novelty(S_enh, L=kernel_L)

            peak_frames = self._structure_analyzer.pick_peaks(
                novelty, feature_rate=fps,
                min_dist_sec=min_seg_dur,
                prominence=0.18,
            )

            candidates: list[CandidateBoundary] = []
            for frame in peak_frames:
                if frame < len(rms_times):
                    t = float(rms_times[frame]) + active_start
                else:
                    t = float(frame) / fps + active_start

                if t <= 0.0 or t >= active_dur + active_start:
                    continue

                conf = round(float(np.clip(novelty[frame], 0.0, 1.0)), 3)
                candidates.append(
                    CandidateBoundary(
                        time_seconds=round(t, 3),
                        source=["ssm"],
                        confidence=conf,
                    )
                )

            return candidates

        except Exception as exc:
            logger.warning("SSM candidate extraction failed (%s).", exc)
            return []

    @staticmethod
    def _shift_candidates(
        candidates: list[CandidateBoundary], offset: float
    ) -> list[CandidateBoundary]:
        """Shift all candidate times by *offset* (to convert active-region times to full-track times)."""
        return [
            CandidateBoundary(
                time_seconds=round(c.time_seconds + offset, 3),
                source=c.source,
                confidence=c.confidence,
            )
            for c in candidates
        ]

    @staticmethod
    def _beat_grid_candidates(
        beat_times_full: np.ndarray,
        active_start: float,
        active_end: float,
        min_seg_dur: float,
    ) -> list[CandidateBoundary]:
        """
        Add beat-aligned phrase candidates at a stable 16-beat phase.

        Librosa's first detected beat is often a pickup; index 1 tends to align
        with the first full bar/downbeat on SALAMI tracks. Keeping one phase
        avoids flooding the LLM with every beat while still exposing exact
        beat-grid boundary times.
        """
        if beat_times_full.size < 16:
            return []

        out: list[CandidateBoundary] = []
        for idx, t in enumerate(beat_times_full.tolist()):
            t = float(t)
            if idx % 16 != 1:
                continue
            if t <= active_start + min_seg_dur or t >= active_end - min_seg_dur:
                continue
            out.append(
                CandidateBoundary(
                    time_seconds=round(t, 3),
                    source=["beat_grid"],
                    confidence=0.64,
                )
            )
        return out

    @staticmethod
    def _apply_positional_heuristics(
        segments: list[PredictedSegment],
    ) -> list[PredictedSegment]:
        """
        Override first/last segment labels with Intro/Outro heuristics.

        Applied when there are 4 or more segments, consistent with the
        existing deterministic pipeline's logic.
        """
        if len(segments) < 4:
            return segments

        updated = list(segments)

        # First segment → Intro (unless it's already labelled as such).
        first = updated[0]
        if first.label not in {"Intro", "Silence"}:
            updated[0] = PredictedSegment(
                start=first.start,
                end=first.end,
                start_seconds=first.start_seconds,
                end_seconds=first.end_seconds,
                label="Intro",
                confidence=first.confidence,
                source_features=first.source_features,
                reason=first.reason + " [Positional heuristic: first segment → Intro.]",
            )
            logger.debug(
                "Overrode first segment label '%s' → 'Intro'", first.label
            )

        # Last segment → Outro (unless it's already labelled as such).
        last = updated[-1]
        if last.label not in {"Outro", "Silence"}:
            updated[-1] = PredictedSegment(
                start=last.start,
                end=last.end,
                start_seconds=last.start_seconds,
                end_seconds=last.end_seconds,
                label="Outro",
                confidence=last.confidence,
                source_features=last.source_features,
                reason=last.reason + " [Positional heuristic: last segment → Outro.]",
            )
            logger.debug(
                "Overrode last segment label '%s' → 'Outro'", last.label
            )

        return updated

    @staticmethod
    def _empty_result(track_id: str | None, total_dur: float) -> SegmentationResult:
        """Return a valid but empty SegmentationResult when audio is too short."""
        from ..core.models import EvaluationResult
        return SegmentationResult(
            track_id=track_id,
            duration=seconds_to_mmss(total_dur),
            estimated_bpm=0.0,
            candidate_boundaries=[],
            predicted_segments=[],
            salami_ground_truth=[],
            evaluation=EvaluationResult(**_EMPTY_EVAL),
            agent_explanation="No active audio region detected; segmentation was skipped.",
        )
