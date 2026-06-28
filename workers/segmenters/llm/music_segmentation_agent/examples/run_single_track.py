"""
Example: Segment a single audio track with optional SALAMI evaluation.

Run from the project root:
    PYTHONPATH=. python workers/segmenters/music_segmentation_agent/examples/run_single_track.py

Environment variables required:
    ANTHROPIC_API_KEY — Anthropic API key for Claude.

Optional environment variables:
    AUDIO_FILE          — Path to the audio file (default: see AUDIO_PATH below).
    SALAMI_ANNOTATION   — Path to SALAMI annotation file.
    TRACK_ID            — Track identifier string.
"""

from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so relative imports resolve.
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from workers.segmenters.llm.music_segmentation_agent import SegmentationService  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — override via environment variables.
# ---------------------------------------------------------------------------

AUDIO_PATH = os.environ.get("AUDIO_FILE", "/path/to/audio.mp3")
SALAMI_PATH = os.environ.get("SALAMI_ANNOTATION", None)
TRACK_ID = os.environ.get("TRACK_ID", "demo_track")

# Optional: provide timed lyrics as a list of {time_seconds, text} dicts.
TIMED_LYRICS = [
    # Example lyrics — replace with real data.
    # {"time_seconds": 16.5, "text": "Verse one begins here"},
    # {"time_seconds": 48.2, "text": "Chorus kicks in with the hook"},
]

# Pipeline parameter overrides (all optional).
PARAMS = {
    "min_segment_duration_seconds": 8.0,
    "max_candidates": 18,
    "min_confidence": 0.25,
    "merge_window_sec": 1.75,
    "fusion_threshold": 0.28,
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.isfile(AUDIO_PATH):
        print(f"[ERROR] Audio file not found: {AUDIO_PATH}")
        print("Set AUDIO_FILE environment variable to a valid audio path.")
        sys.exit(1)

    print(f"Segmenting: {AUDIO_PATH}")
    print(f"Track ID:   {TRACK_ID}")
    print(f"SALAMI:     {SALAMI_PATH or '(none)'}")
    print("-" * 60)

    service = SegmentationService(model_name="claude-sonnet-4-6")

    result = service.segment_audio(
        file_path=AUDIO_PATH,
        track_id=TRACK_ID,
        salami_annotation_path=SALAMI_PATH,
        timed_lyrics=TIMED_LYRICS or None,
        params=PARAMS,
    )

    # Pretty-print the full result.
    print(json.dumps(result.model_dump(), indent=2))

    # Summary.
    print("\n" + "=" * 60)
    print(f"Duration:   {result.duration}")
    print(f"BPM:        {result.estimated_bpm:.1f}")
    print(f"Segments:   {len(result.predicted_segments)}")
    print(f"Candidates: {len(result.candidate_boundaries)}")
    print(f"F-Measure:  {result.evaluation.boundary_f_measure:.4f} "
          f"(tol={result.evaluation.tolerance_seconds:.1f}s)")
    print(f"Label Acc:  {result.evaluation.label_accuracy:.4f}")
    print("=" * 60)
    print("\nSegments:")
    for seg in result.predicted_segments:
        print(
            f"  [{seg.start} – {seg.end}]  {seg.label:15s}  "
            f"conf={seg.confidence:.2f}  sources={seg.source_features}"
        )
    print("\nAgent Explanation:")
    print(result.agent_explanation)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Realistic example output (comment — does not execute)
# ---------------------------------------------------------------------------
#
# $ AUDIO_FILE=/data/salami/956/audio.mp3 \
#   SALAMI_ANNOTATION=/data/salami/956/annotations/annotator1.txt \
#   TRACK_ID=salami_956 \
#   python run_single_track.py
#
# {
#   "track_id": "salami_956",
#   "duration": "03:42",
#   "estimated_bpm": 121.8,
#   "candidate_boundaries": [
#     {"time_seconds": 14.32, "source": ["beat_phrase", "rms"], "confidence": 0.71},
#     {"time_seconds": 30.18, "source": ["chord_proxy", "ssm"], "confidence": 0.63},
#     {"time_seconds": 61.54, "source": ["beat_phrase", "chord_proxy", "ssm"], "confidence": 0.82},
#     {"time_seconds": 92.71, "source": ["rms", "ssm"], "confidence": 0.58},
#     {"time_seconds": 124.06, "source": ["beat_phrase", "chord_proxy"], "confidence": 0.77},
#     {"time_seconds": 155.23, "source": ["ssm"], "confidence": 0.51},
#     {"time_seconds": 186.49, "source": ["beat_phrase", "rms", "ssm"], "confidence": 0.79},
#     {"time_seconds": 210.14, "source": ["rms"], "confidence": 0.44}
#   ],
#   "predicted_segments": [
#     {
#       "start": "00:00", "end": "00:14",
#       "start_seconds": 0.0, "end_seconds": 14.32,
#       "label": "Intro", "confidence": 0.72,
#       "source_features": ["beat_phrase", "rms"],
#       "reason": "Opening low-energy section before the first verse. [Positional heuristic: first segment → Intro.]"
#     },
#     {
#       "start": "00:14", "end": "01:01",
#       "start_seconds": 14.32, "end_seconds": 61.54,
#       "label": "Verse", "confidence": 0.68,
#       "source_features": ["beat_phrase", "chord_proxy", "ssm"],
#       "reason": "Consistent harmonic progression with moderate energy. Primary narrative section."
#     },
#     {
#       "start": "01:01", "end": "02:04",
#       "start_seconds": 61.54, "end_seconds": 124.06,
#       "label": "Chorus", "confidence": 0.84,
#       "source_features": ["beat_phrase", "chord_proxy", "ssm"],
#       "reason": "Energy peak, strong beat emphasis, repeated chord pattern characteristic of the hook."
#     },
#     {
#       "start": "02:04", "end": "03:06",
#       "start_seconds": 124.06, "end_seconds": 186.49,
#       "label": "Verse", "confidence": 0.65,
#       "source_features": ["beat_phrase", "chord_proxy"],
#       "reason": "Return to verse-like harmonic and energy profile."
#     },
#     {
#       "start": "03:06", "end": "03:42",
#       "start_seconds": 186.49, "end_seconds": 222.0,
#       "label": "Outro", "confidence": 0.61,
#       "source_features": ["beat_phrase", "rms", "ssm"],
#       "reason": "Closing section with fading energy. [Positional heuristic: last segment → Outro.]"
#     }
#   ],
#   "salami_ground_truth": [
#     {"start": "00:00", "end": "00:15", "start_seconds": 0.0, "end_seconds": 15.0, "label": "Intro"},
#     {"start": "00:15", "end": "01:03", "start_seconds": 15.0, "end_seconds": 63.0, "label": "Section A"},
#     {"start": "01:03", "end": "02:07", "start_seconds": 63.0, "end_seconds": 127.0, "label": "Section B"},
#     {"start": "02:07", "end": "03:08", "start_seconds": 127.0, "end_seconds": 188.0, "label": "Section A"},
#     {"start": "03:08", "end": "03:42", "start_seconds": 188.0, "end_seconds": 222.0, "label": "Outro"}
#   ],
#   "evaluation": {
#     "tolerance_seconds": 3.0,
#     "boundary_precision": 0.8571,
#     "boundary_recall": 0.8571,
#     "boundary_f_measure": 0.8571,
#     "label_accuracy": 0.4921,
#     "over_segmentation_notes": [],
#     "under_segmentation_notes": [
#       "GT boundary at 30.18s is not covered by any predicted boundary within ±3.0s."
#     ]
#   },
#   "agent_explanation": "The track follows a compact Intro → Verse → Chorus → Verse → Outro structure. ..."
# }
#
# ============================================================
# Duration:   03:42
# BPM:        121.8
# Segments:   5
# Candidates: 8
# F-Measure:  0.8571 (tol=3.0s)
# Label Acc:  0.4921
# ============================================================
#
# Segments:
#   [00:00 – 00:14]  Intro            conf=0.72  sources=['beat_phrase', 'rms']
#   [00:14 – 01:01]  Verse            conf=0.68  sources=['beat_phrase', 'chord_proxy', 'ssm']
#   [01:01 – 02:04]  Chorus           conf=0.84  sources=['beat_phrase', 'chord_proxy', 'ssm']
#   [02:04 – 03:06]  Verse            conf=0.65  sources=['beat_phrase', 'chord_proxy']
#   [03:06 – 03:42]  Outro            conf=0.61  sources=['beat_phrase', 'rms', 'ssm']
