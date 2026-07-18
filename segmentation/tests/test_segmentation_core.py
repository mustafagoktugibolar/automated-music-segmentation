import unittest

import numpy as np

from segmentation.core.segmentation.evaluation import compute_boundary_metrics, compute_boundary_metrics_multi
from segmentation.core.labeling.heuristic import apply_two_layer_labels
from segmentation.core.segmentation.utils import (
    boundaries_to_segments,
    normalize_algorithm_result,
    normalize_boundaries,
    segments_to_internal_boundaries,
    segments_to_intervals,
)
from segmentation.application.segmentation.fusion_engine import fuse_algorithm_results


class SegmentationUtilityTests(unittest.TestCase):
    def test_segments_to_intervals_uses_real_segment_extents(self):
        segments = [
            {"start": 0.0, "end": 14.2, "label": "A"},
            {"start": 14.2, "end": 42.8, "label": "B"},
        ]

        intervals = segments_to_intervals(segments)

        np.testing.assert_allclose(intervals, np.asarray([[0.0, 14.2], [14.2, 42.8]]))

    def test_internal_boundaries_exclude_edges(self):
        segments = [
            {"start": 0.0, "end": 10.0},
            {"start": 10.0, "end": 20.0},
            {"start": 20.0, "end": 30.0},
        ]

        self.assertEqual(segments_to_internal_boundaries(segments), [10.0, 20.0])

    def test_normalize_boundaries_adds_edges_deduplicates_and_drops_invalid(self):
        times = normalize_boundaries([20.0, -1.0, 10.0, 10.05, 999.0], 30.0, min_gap_seconds=0.25)

        self.assertEqual(times, [0.0, 10.0, 20.0, 30.0])

    def test_boundaries_to_segments(self):
        segments = boundaries_to_segments([10.0, 20.0], 30.0)

        self.assertEqual(
            [(s["start"], s["end"], s["label"]) for s in segments],
            [(0.0, 10.0, "A"), (10.0, 20.0, "B"), (20.0, 30.0, "C")],
        )


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_computed_from_segment_intervals(self):
        ref = [
            {"start": 0.0, "end": 10.0},
            {"start": 10.0, "end": 20.0},
            {"start": 20.0, "end": 30.0},
        ]
        est = [
            {"start": 0.0, "end": 10.4},
            {"start": 10.4, "end": 20.2},
            {"start": 20.2, "end": 30.0},
        ]

        metrics = compute_boundary_metrics(ref, est, tolerance=0.5)

        self.assertEqual(metrics["n_reference_internal_boundaries"], 2)
        self.assertEqual(metrics["n_estimated_internal_boundaries"], 2)
        self.assertEqual(metrics["f_measure"], 1.0)

    def test_multi_tolerance_metric_keys(self):
        ref = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}]
        est = [{"start": 0.0, "end": 12.0}, {"start": 12.0, "end": 20.0}]

        metrics = compute_boundary_metrics_multi(ref, est)

        self.assertIn("precision_0_5", metrics)
        self.assertIn("f1_3_0", metrics)
        self.assertEqual(metrics["f1_0_5"], 0.0)
        self.assertEqual(metrics["f1_3_0"], 1.0)


class ResultNormalizationTests(unittest.TestCase):
    def test_msaf_style_result_normalization_preserves_schema(self):
        result = normalize_algorithm_result(
            task_id="task-1",
            status="completed",
            worker_type="msaf",
            algorithm="foote",
            duration_seconds=30.0,
            boundaries=[{"time": 10.0, "confidence": 0.8, "source": "foote"}],
            segments=[{"start": 0.0, "end": 10.0, "label": "1"}, {"start": 10.0, "end": 30.0, "label": "2"}],
            diagnostics={"raw_est_times_count": 1},
        )

        self.assertEqual(result["algorithm"], "foote")
        self.assertEqual(result["duration_seconds"], 30.0)
        self.assertIn("boundaries", result)
        self.assertEqual(result["segments"][0]["structural_label"], "1")
        self.assertEqual(result["diagnostics"]["raw_est_times_count"], 1)

    def test_failed_result_without_duration_has_no_fabricated_boundary(self):
        result = normalize_algorithm_result(
            task_id="task-1",
            status="failed",
            worker_type="msaf",
            algorithm="foote",
            duration_seconds=None,
            boundaries=[],
            segments=[],
            diagnostics={"error": "boom"},
        )

        self.assertEqual(result["boundaries"], [])
        self.assertEqual(result["segments"], [])


class LabelingTests(unittest.TestCase):
    def test_semantic_labeling_is_conservative(self):
        segments = [
            {"start": 0.0, "end": 8.0, "label": "A"},
            {"start": 8.0, "end": 24.0, "label": "B"},
            {"start": 24.0, "end": 40.0, "label": "B"},
        ]

        labeled = apply_two_layer_labels(segments, duration_seconds=40.0)

        self.assertTrue(all(s["structural_label"] for s in labeled))
        self.assertEqual(labeled[0]["label"], labeled[0]["structural_label"])
        self.assertIn(labeled[0]["semantic_label"], {"Intro", "Unknown", "Other"})


class FusionTests(unittest.TestCase):
    def test_weighted_boundary_voting_accepts_multi_algorithm_group(self):
        base = {
            "custom_librosa": {
                "duration_seconds": 60.0,
                "segments": boundaries_to_segments([20.0, 40.0], 60.0),
            },
            "foote": {
                "duration_seconds": 60.0,
                "segments": boundaries_to_segments([20.7], 60.0),
            },
            "cnmf": {
                "duration_seconds": 60.0,
                "segments": boundaries_to_segments([39.5], 60.0),
            },
            "scluster": {
                "duration_seconds": 60.0,
                "segments": boundaries_to_segments([20.3, 40.4], 60.0),
            },
        }

        result = fuse_algorithm_results(base, task_id="task-1")

        self.assertEqual(result["algorithm"], "fusion")
        self.assertGreaterEqual(len(result["segments"]), 2)
        self.assertTrue(result["diagnostics"]["boundary_groups"])
        accepted = [g for g in result["diagnostics"]["boundary_groups"] if g["accepted"]]
        self.assertTrue(accepted)


if __name__ == "__main__":
    unittest.main()
