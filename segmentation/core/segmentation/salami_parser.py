"""
SALAMI annotation parser.

Reads SALAMI annotation files and converts them into the standard segment
format: [{start, end, label}, ...].

Preferred annotation file format (tab-separated):
    0.000000000    Silence
    0.464399092    Intro
    ...
    264.885215419  End

The parser prefers the SALAMI "parsed" function files when available, so labels
such as Intro, Verse, Chorus, Bridge, and Outro are preserved as the canonical
section names.
"""

import os
from typing import Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "salami", "annotations")


def _normalize_salami_label(raw_label: str) -> str:
    """
    Normalize a SALAMI label token into a section name.

    For parsed function files this preserves labels like Intro, Verse, Chorus,
    Bridge, Outro, and Silence. For raw annotation files, it falls back to the
    first meaningful token and prefers the first semantic token after the
    letter-based section marker.
    """
    raw_label = raw_label.strip()
    if not raw_label:
        return raw_label

    parts = [p.strip() for p in raw_label.split(",") if p.strip()]
    if len(parts) >= 2:
        second_token = parts[1]
        if second_token and len(second_token) > 1 and not second_token.isupper():
            return second_token

    if parts:
        return parts[0]

    return raw_label


def parse_salami_annotation(song_id: str, annotator: int = 1) -> Optional[list[dict]]:
    """
    Parse a SALAMI annotation file for the given song_id.

    Returns a list of segments: [{start: float, end: float, label: str}, ...]
    or None if the annotation file does not exist.

    Args:
        song_id: The SALAMI numeric song identifier (string).
        annotator: Which annotator file to read (1 or 2). Defaults to 1.
    """
    song_dir = os.path.join(ANNOTATIONS_DIR, str(song_id))
    parsed_annotation_path = os.path.join(
        song_dir, "parsed", f"textfile{annotator}_functions.txt"
    )
    raw_annotation_path = os.path.join(song_dir, f"textfile{annotator}.txt")

    annotation_path = (
        parsed_annotation_path if os.path.exists(parsed_annotation_path) else raw_annotation_path
    )

    if not os.path.exists(annotation_path):
        return None

    # Parse raw (timestamp, label) pairs
    raw_boundaries: list[tuple[float, str]] = []

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            try:
                timestamp = float(parts[0].strip())
            except ValueError:
                continue

            label = parts[1].strip()
            raw_boundaries.append((timestamp, label))

    if not raw_boundaries:
        return None

    # Find the End marker index
    end_idx = len(raw_boundaries)
    for i, (_, label) in enumerate(raw_boundaries):
        if label.strip().lower() == "end":
            end_idx = i
            break

    # Convert consecutive (timestamp, label) pairs into segments
    segments: list[dict] = []
    for i in range(end_idx):
        start = raw_boundaries[i][0]
        end = raw_boundaries[i + 1][0] if i + 1 < len(raw_boundaries) else raw_boundaries[end_idx][0]
        raw_label = raw_boundaries[i][1]

        section_label = _normalize_salami_label(raw_label)

        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "label": section_label,
        })

    return segments if segments else None


def get_annotation_dir_count() -> int:
    """Return the total number of song annotation directories available."""
    if not os.path.isdir(ANNOTATIONS_DIR):
        return 0
    return sum(1 for d in os.scandir(ANNOTATIONS_DIR) if d.is_dir())
