#!/usr/bin/env python
"""
Standalone test / benchmark for the segmentation pipeline.

Usage
-----
# Basic run:
python scripts/test_segmentation.py --audio /path/to/track.mp3

# With SALAMI ground-truth evaluation:
python scripts/test_segmentation.py \
    --audio /path/to/track.mp3 \
    --salami-gt /path/to/salami/annotations/1234/parsed/textfile1_functions.txt

# Tweak params:
python scripts/test_segmentation.py --audio track.mp3 \
    --min-seg 8 --kernel 5 --no-mfcc --no-beat-sync --smoothing-L 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from repo root without installing the package
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_table(segments: list[dict]) -> None:
    """Print segments as a fixed-width table."""
    header = f"{'#':>3}  {'Start':>8}  {'End':>8}  {'Dur':>7}  {'Label':>5}  Section"
    print(header)
    print("-" * len(header))
    for i, seg in enumerate(segments):
        start = seg.get("start", 0.0)
        end   = seg.get("end",   0.0)
        dur   = end - start
        label = seg.get("label", "?")
        stype = seg.get("section_type", "")
        print(f"{i+1:>3}  {start:>8.2f}  {end:>8.2f}  {dur:>6.2f}s  {label:>5}  {stype}")


def _parse_salami_file(path: str) -> list[dict]:
    """
    Minimal SALAMI flat-file parser.  Returns [{start, end, label}, ...].
    Works for both raw textfiles and parsed function files.
    """
    rows: list[tuple[float, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            rows.append((t, parts[1].strip()))

    if not rows:
        return []

    # Find end marker
    end_idx = len(rows)
    for i, (_, lbl) in enumerate(rows):
        if lbl.lower() == "end":
            end_idx = i
            break

    segments: list[dict] = []
    for i in range(end_idx):
        t0 = rows[i][0]
        t1 = rows[i + 1][0] if i + 1 < len(rows) else rows[end_idx][0]
        lbl = rows[i][1].split(",")[0].strip()
        segments.append({"start": t0, "end": t1, "label": lbl})
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test music segmentation pipeline")
    parser.add_argument("--audio",       required=True, help="Path to audio file")
    parser.add_argument("--salami-gt",   default=None,  help="Path to SALAMI annotation file")
    parser.add_argument("--min-seg",     type=float, default=6.0,  help="Min segment duration (s)")
    parser.add_argument("--kernel",      type=float, default=4.0,  help="Novelty kernel size (s)")
    parser.add_argument("--n-clusters",  type=int,   default=4,    help="Number of clusters")
    parser.add_argument("--flux-weight", type=float, default=0.3,  help="Spectral flux blend weight")
    parser.add_argument("--smoothing-L", type=int,   default=15,   help="Diagonal smoothing window (beats)")
    parser.add_argument("--no-mfcc",      action="store_true", help="Disable MFCC features")
    parser.add_argument("--no-beat-sync", action="store_true", help="Disable beat-sync features")
    parser.add_argument("--no-ti",        action="store_true", help="Disable transposition-invariant SSM")
    parser.add_argument("--no-auto-k",    action="store_true", help="Disable auto cluster selection")
    parser.add_argument("--tolerance",   type=float, default=0.5,  help="Boundary eval tolerance (s)")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    params = {
        "min_segment_duration_seconds": args.min_seg,
        "novelty_kernel_size_seconds":  args.kernel,
        "n_clusters":                   args.n_clusters,
        "use_mfcc":                     not args.no_mfcc,
        "spectral_flux_weight":         args.flux_weight,
        "smoothing_L":                  args.smoothing_L,
        "use_beat_sync":                not args.no_beat_sync,
        "transposition_invariant":      not args.no_ti,
        "auto_n_clusters":              not args.no_auto_k,
    }

    print(f"\nAudio : {args.audio}")
    print(f"Params: {params}\n")

    # --- Run pipeline ---
    from workers.segmenters.custom.segmentation_service import process_file_path

    t_start = time.perf_counter()
    result  = process_file_path(args.audio, params=params)
    elapsed = time.perf_counter() - t_start

    segments = result.get("segments", [])
    print(f"Duration   : {result.get('duration_seconds', 0):.2f} s")
    print(f"Segments   : {len(segments)}")
    print(f"Wall time  : {elapsed:.2f} s\n")

    _print_table(segments)

    # --- Optional SALAMI evaluation ---
    if args.salami_gt:
        if not os.path.isfile(args.salami_gt):
            print(f"\nWARNING: SALAMI GT file not found: {args.salami_gt}", file=sys.stderr)
        else:
            from backend.services.evaluation_service import compute_boundary_metrics
            gt_segments = _parse_salami_file(args.salami_gt)
            if not gt_segments:
                print("\nWARNING: could not parse any GT segments from the file.")
            else:
                metrics = compute_boundary_metrics(
                    gt_segments, segments, tolerance=args.tolerance
                )
                print(f"\n--- Boundary evaluation (τ={args.tolerance}s) ---")
                print(f"  Precision : {metrics['precision']:.4f}")
                print(f"  Recall    : {metrics['recall']:.4f}")
                print(f"  F-measure : {metrics['f_measure']:.4f}")
                print(f"  Ref boundaries : {metrics['n_boundaries_ref']}")
                print(f"  Est boundaries : {metrics['n_boundaries_est']}")


if __name__ == "__main__":
    main()
