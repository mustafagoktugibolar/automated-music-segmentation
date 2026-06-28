"""
Segmentation evaluation using mir_eval.

Computes boundary precision/recall/F-measure and a simple label accuracy score
by comparing predicted segments to SALAMI ground-truth segments.

Boundary tolerance defaults to 3.0 seconds (generous production metric).
The strict MIREX tolerance is 0.5 seconds.
"""

from __future__ import annotations

import numpy as np

from .models import EvaluationResult, PredictedSegment, SalamiSegment
from shared.logger import get_logger

try:
    import mir_eval
    _MIR_EVAL_AVAILABLE = True
except ImportError:
    _MIR_EVAL_AVAILABLE = False

logger = get_logger("evaluator")


class Evaluator:
    """
    Evaluate predicted segments against SALAMI ground truth.

    Uses mir_eval for boundary detection metrics and computes a simple
    overlap-based label accuracy score.
    """

    def evaluate(
        self,
        predicted_segments: list[PredictedSegment],
        ground_truth_segments: list[SalamiSegment],
        tolerance_sec: float = 3.0,
    ) -> EvaluationResult:
        """
        Compute boundary and label accuracy metrics.

        Parameters
        ----------
        predicted_segments    : Output of LLMSegmentationDecision.decide().
        ground_truth_segments : SALAMI annotations (may be empty).
        tolerance_sec         : Boundary tolerance window in seconds.

        Returns
        -------
        EvaluationResult with precision, recall, F-measure and label accuracy.
        """
        if not ground_truth_segments:
            logger.info("No ground truth provided; returning zero metrics.")
            return EvaluationResult(
                tolerance_seconds=tolerance_sec,
                boundary_precision=0.0,
                boundary_recall=0.0,
                boundary_f_measure=0.0,
                label_accuracy=0.0,
                over_segmentation_notes=[],
                under_segmentation_notes=[],
            )

        if not predicted_segments:
            logger.warning("No predicted segments to evaluate against.")
            return EvaluationResult(
                tolerance_seconds=tolerance_sec,
                boundary_precision=0.0,
                boundary_recall=0.0,
                boundary_f_measure=0.0,
                label_accuracy=0.0,
                over_segmentation_notes=["No segments predicted."],
                under_segmentation_notes=[],
            )

        # ------------------------------------------------------------------
        # Build boundary arrays
        # ------------------------------------------------------------------
        pred_boundaries = self._segments_to_boundaries(
            [(s.start_seconds, s.end_seconds) for s in predicted_segments]
        )
        gt_boundaries = self._segments_to_boundaries(
            [(s.start_seconds, s.end_seconds) for s in ground_truth_segments]
        )

        # ------------------------------------------------------------------
        # Boundary metrics via mir_eval
        # ------------------------------------------------------------------
        if _MIR_EVAL_AVAILABLE:
            try:
                P, R, F = mir_eval.segment.detection(
                    gt_boundaries,
                    pred_boundaries,
                    window=tolerance_sec,
                    beta=1.0,
                    trim=True,
                )
                precision = round(float(P), 4)
                recall = round(float(R), 4)
                f_measure = round(float(F), 4)
            except Exception as exc:
                logger.warning("mir_eval boundary detection failed (%s); using zeros.", exc)
                precision, recall, f_measure = 0.0, 0.0, 0.0
        else:
            logger.warning("mir_eval not available; computing boundary metrics manually.")
            precision, recall, f_measure = self._manual_boundary_metrics(
                pred_boundaries, gt_boundaries, tolerance_sec
            )

        # ------------------------------------------------------------------
        # Label accuracy (overlap-based)
        # ------------------------------------------------------------------
        label_accuracy = self._compute_label_accuracy(
            predicted_segments, ground_truth_segments
        )

        # ------------------------------------------------------------------
        # Segment IoU (label-agnostic temporal IoU)
        # ------------------------------------------------------------------
        segment_iou = self._compute_segment_iou(predicted_segments, ground_truth_segments)

        # ------------------------------------------------------------------
        # Over/under segmentation notes
        # ------------------------------------------------------------------
        over_notes, under_notes = self._segmentation_notes(
            predicted_segments, ground_truth_segments, tolerance_sec
        )

        logger.info(
            "Evaluation (tol=%.1fs): P=%.3f R=%.3f F=%.3f label_acc=%.3f seg_iou=%.3f",
            tolerance_sec,
            precision,
            recall,
            f_measure,
            label_accuracy,
            segment_iou,
        )

        return EvaluationResult(
            tolerance_seconds=tolerance_sec,
            boundary_precision=precision,
            boundary_recall=recall,
            boundary_f_measure=f_measure,
            label_accuracy=label_accuracy,
            segment_iou=segment_iou,
            over_segmentation_notes=over_notes,
            under_segmentation_notes=under_notes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segments_to_boundaries(spans: list[tuple[float, float]]) -> np.ndarray:
        """Convert [(start, end), ...] to sorted unique boundary array."""
        times: list[float] = []
        for start, end in spans:
            times.append(float(start))
            times.append(float(end))
        return np.unique(np.array(times, dtype=np.float64))

    @staticmethod
    def _manual_boundary_metrics(
        pred: np.ndarray,
        ref: np.ndarray,
        tolerance: float,
    ) -> tuple[float, float, float]:
        """Fallback boundary metrics without mir_eval."""
        if len(ref) == 0 or len(pred) == 0:
            return 0.0, 0.0, 0.0

        n_correct_pred = sum(
            1 for p in pred if np.min(np.abs(ref - p)) <= tolerance
        )
        n_correct_ref = sum(
            1 for r in ref if np.min(np.abs(pred - r)) <= tolerance
        )

        precision = n_correct_pred / len(pred) if pred.size else 0.0
        recall = n_correct_ref / len(ref) if ref.size else 0.0
        if precision + recall > 0:
            f_measure = 2 * precision * recall / (precision + recall)
        else:
            f_measure = 0.0
        return round(precision, 4), round(recall, 4), round(float(f_measure), 4)

    @staticmethod
    def _compute_label_accuracy(
        predicted: list[PredictedSegment],
        ground_truth: list[SalamiSegment],
    ) -> float:
        """
        Overlap-based label accuracy.

        For each ground-truth segment, find the predicted segment with the
        greatest time overlap and check whether labels match. Accuracy =
        (matching overlap time) / (total ground-truth duration).
        """
        if not ground_truth or not predicted:
            return 0.0

        total_gt_dur = sum(s.end_seconds - s.start_seconds for s in ground_truth)
        if total_gt_dur <= 0:
            return 0.0

        matching_dur = 0.0
        for gt in ground_truth:
            best_overlap = 0.0
            best_match = False
            for pred in predicted:
                overlap_start = max(gt.start_seconds, pred.start_seconds)
                overlap_end = min(gt.end_seconds, pred.end_seconds)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    # Normalise labels to lowercase for comparison.
                    best_match = (
                        gt.label.strip().lower() == pred.label.strip().lower()
                    )
            if best_match:
                matching_dur += best_overlap

        return round(matching_dur / total_gt_dur, 4)

    @staticmethod
    def _compute_segment_iou(
        predicted: list[PredictedSegment],
        ground_truth: list[SalamiSegment],
    ) -> float:
        """
        Mean segment IoU (label-agnostic).

        For each GT segment, finds the predicted segment with the greatest
        temporal overlap and computes IoU = overlap / union.
        """
        if not ground_truth or not predicted:
            return 0.0

        ious: list[float] = []
        for gt in ground_truth:
            gt_dur = gt.end_seconds - gt.start_seconds
            if gt_dur <= 0:
                continue
            best_iou = 0.0
            for pred in predicted:
                overlap = max(0.0, min(gt.end_seconds, pred.end_seconds) - max(gt.start_seconds, pred.start_seconds))
                union = gt_dur + (pred.end_seconds - pred.start_seconds) - overlap
                if union > 0:
                    best_iou = max(best_iou, overlap / union)
            ious.append(best_iou)

        return round(float(np.mean(ious)), 4) if ious else 0.0

    @staticmethod
    def _segmentation_notes(
        predicted: list[PredictedSegment],
        ground_truth: list[SalamiSegment],
        tolerance: float,
    ) -> tuple[list[str], list[str]]:
        """
        Identify obvious over- and under-segmentation cases.

        Over-segmentation: a predicted boundary has no nearby GT boundary.
        Under-segmentation: a GT boundary has no nearby predicted boundary.
        """
        pred_times = [
            s.start_seconds for s in predicted if s.start_seconds > 0
        ] + [s.end_seconds for s in predicted]
        gt_times = [
            s.start_seconds for s in ground_truth if s.start_seconds > 0
        ] + [s.end_seconds for s in ground_truth]

        pred_arr = np.unique(np.array(pred_times, dtype=np.float64))
        gt_arr = np.unique(np.array(gt_times, dtype=np.float64))

        over_notes: list[str] = []
        for p in pred_arr:
            if gt_arr.size == 0 or np.min(np.abs(gt_arr - p)) > tolerance:
                over_notes.append(
                    f"Predicted boundary at {p:.2f}s has no GT match within ±{tolerance:.1f}s."
                )

        under_notes: list[str] = []
        for g in gt_arr:
            if pred_arr.size == 0 or np.min(np.abs(pred_arr - g)) > tolerance:
                under_notes.append(
                    f"GT boundary at {g:.2f}s is not covered by any predicted boundary within ±{tolerance:.1f}s."
                )

        return over_notes[:10], under_notes[:10]  # cap at 10 notes each
