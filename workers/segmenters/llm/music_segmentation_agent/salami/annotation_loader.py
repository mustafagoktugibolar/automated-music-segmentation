"""
SALAMI annotation file parser.

Parses SALAMI-format annotation files (plain text, tab or space-separated) into
SalamiSegment objects. Handles both single-annotator files (annotator1.txt) and
raw segment files.

SALAMI annotation format:
  <time_seconds>\t<label>
  e.g.:
    0.000\tIntro
    16.325\tA
    32.651\tA'
    ...
    210.000\tEnd
"""

from __future__ import annotations

import os
import re

from ..core.models import SalamiSegment, seconds_to_mmss
from shared.logger import get_logger

logger = get_logger("salami.annotation_loader")


class SalamiAnnotationLoader:
    """
    Load and parse SALAMI annotation files.

    Supports:
    - Standard annotator files: `<time>\t<label>` per line.
    - Both tab-separated and multiple-space-separated formats.
    - Graceful handling of header lines, comment lines (#), and blank lines.
    """

    def load(self, annotation_path: str) -> list[SalamiSegment]:
        """
        Parse a SALAMI annotation file and return a list of SalamiSegment.

        Parameters
        ----------
        annotation_path : Path to the annotation file.
                          For SALAMI datasets, this is typically
                          ``/path/to/salami/<id>/parsed/textfile1_functions.txt``
                          or ``/path/to/salami/<id>/annotations/annotator1.txt``.

        Returns
        -------
        List of SalamiSegment with start/end in seconds and "MM:SS" strings.
        Empty list if the file cannot be parsed.
        """
        if not annotation_path:
            logger.warning("No annotation path provided.")
            return []

        # If a directory is given, try to find the best annotation file.
        if os.path.isdir(annotation_path):
            annotation_path = self._resolve_annotation_file(annotation_path)
            if annotation_path is None:
                logger.warning("No annotation file found in directory.")
                return []

        if not os.path.isfile(annotation_path):
            logger.warning("Annotation file not found: %s", annotation_path)
            return []

        try:
            raw_events = self._parse_file(annotation_path)
        except Exception as exc:
            logger.error("Failed to parse annotation file %s: %s", annotation_path, exc)
            return []

        if len(raw_events) < 2:
            logger.warning(
                "Annotation file has fewer than 2 events; cannot form segments: %s",
                annotation_path,
            )
            return []

        segments = self._events_to_segments(raw_events)
        logger.info(
            "Loaded %d segments from %s", len(segments), os.path.basename(annotation_path)
        )
        return segments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_annotation_file(directory: str) -> str | None:
        """
        Search *directory* for a valid annotation file, in priority order:
          annotator1.txt > annotator2.txt > any .txt file in the directory.
        """
        for name in ("annotator1.txt", "annotator2.txt"):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate

        # Recurse into common SALAMI sub-directories.
        for sub in ("parsed", "annotations", ""):
            sub_dir = os.path.join(directory, sub) if sub else directory
            if not os.path.isdir(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir)):
                if fname.endswith(".txt"):
                    return os.path.join(sub_dir, fname)
        return None

    @staticmethod
    def _parse_file(path: str) -> list[tuple[float, str]]:
        """
        Read the file and return [(time_seconds, label), ...] sorted by time.

        Handles:
        - Tab-separated lines: ``0.000\tIntro``
        - Space-separated lines: ``0.000 Intro``
        - Lines starting with '#' or blank lines (skipped).
        - Lines with 'silence', 'end', 'End', etc.
        """
        events: list[tuple[float, str]] = []

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                # Try tab split first, then whitespace.
                if "\t" in line:
                    parts = line.split("\t", maxsplit=1)
                else:
                    parts = line.split(maxsplit=1)

                if len(parts) < 2:
                    # Could be a time-only line (end marker).
                    try:
                        t = float(parts[0])
                        events.append((t, "End"))
                    except ValueError:
                        pass
                    continue

                try:
                    t = float(parts[0])
                except ValueError:
                    logger.debug("Line %d: cannot parse time '%s'; skipping.", line_no, parts[0])
                    continue

                label = parts[1].strip()
                if not label:
                    label = "?"

                events.append((t, label))

        events.sort(key=lambda e: e[0])
        return events

    @staticmethod
    def _events_to_segments(
        events: list[tuple[float, str]],
    ) -> list[SalamiSegment]:
        """
        Convert a list of (time, label) events into consecutive SalamiSegment objects.

        The last event is treated as an end marker (no segment beyond it is created).
        """
        segments: list[SalamiSegment] = []
        for i in range(len(events) - 1):
            start_sec, label = events[i]
            end_sec = events[i + 1][0]

            if end_sec <= start_sec:
                continue

            # Skip structural annotations used only as delimiters.
            if label.lower() in {"end", "silence", ""}:
                continue

            segments.append(
                SalamiSegment(
                    start=seconds_to_mmss(start_sec),
                    end=seconds_to_mmss(end_sec),
                    start_seconds=round(start_sec, 3),
                    end_seconds=round(end_sec, 3),
                    label=label,
                )
            )

        return segments
