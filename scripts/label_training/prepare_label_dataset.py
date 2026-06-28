#!/usr/bin/env python
"""
Build a training dataset for the segment-label ML classifier.
=============================================================

Audio source priority:
  1. MinIO / S3 (S3_BUCKET_RAW, songs/<song_id>.mp3) — primary in Docker
  2. Local audio_cache (data/audio_cache/<song_id>.mp3) — local dev fallback

Audio decoding: ffmpeg (fast, ~0.3-0.5s/track) with librosa fallback.
Annotation source: SALAMI ground-truth via backend/services/salami_parser.

Run from repo root or inside the worker container:
    python scripts/prepare_label_dataset.py
    python scripts/prepare_label_dataset.py --max-songs 50
    python scripts/prepare_label_dataset.py --annotators 1 2

Output
------
    data/label_training/segments.parquet
        columns: song_id, raw_track_id, annotator_id, dataset,
                 segment_idx, start, end, label,
                 + 87 feature columns (acoustic + contextual + repetition + contrast)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from shared.labeling.features import build_segment_label_vectors
from workers.segmenters.llm.music_segmentation_agent.salami.label_normalizer import normalize_label

# ── Constants ─────────────────────────────────────────────────────────────────
ANNOTATIONS_DIR = os.path.join(_app_root, "data", "salami", "annotations")
AUDIO_CACHE_DIR = os.path.join(_app_root, "data", "audio_cache")
OUTPUT_DIR      = os.path.join(_app_root, "data", "label_training")
OUTPUT_PARQUET  = os.path.join(OUTPUT_DIR, "segments.parquet")

CANONICAL_LABELS = {
    "Intro", "Verse", "Pre-Chorus", "Chorus", "Post-Chorus",
    "Bridge", "Instrumental", "Outro", "Silence", "Spoken",
}

# ── Audio retrieval ────────────────────────────────────────────────────────────

def _get_audio_bytes(song_id: str) -> bytes | None:
    """Fetch audio for *song_id* from MinIO; fall back to local cache."""
    # Try MinIO first (primary source in Docker).
    try:
        from shared.storage.object_store import download
        data = download(song_id)
        if data:
            return data
    except Exception:
        pass

    # Local audio_cache fallback (useful for local dev / CI).
    local = os.path.join(AUDIO_CACHE_DIR, f"{song_id}.mp3")
    if os.path.exists(local):
        with open(local, "rb") as fh:
            return fh.read()

    return None


# ── SALAMI loader ─────────────────────────────────────────────────────────────

def _load_salami_entries(annotators: list[int], max_songs: int) -> list[dict]:
    """Return [{song_id, raw_track_id, annotator_id, dataset, segments}, ...].

    raw_track_id is annotator-independent (salami_{raw_id}); used for grouped
    splitting so the same audio track never appears in two different splits
    even when multiple annotators are present.
    """
    from backend.services.salami_parser import parse_salami_annotation

    if not os.path.isdir(ANNOTATIONS_DIR):
        print(f"[SALAMI] Annotations dir not found: {ANNOTATIONS_DIR}")
        return []

    # List annotated IDs from filesystem.
    ann_ids = sorted(
        d.name for d in os.scandir(ANNOTATIONS_DIR) if d.is_dir() and d.name.isdigit()
    )

    # Optionally intersect with MinIO-available IDs for efficiency.
    try:
        from shared.storage.object_store import list_song_ids
        minio_ids = set(list_song_ids())
        if minio_ids:
            ann_ids = [sid for sid in ann_ids if sid in minio_ids]
            print(f"[SALAMI] {len(ann_ids)} IDs have both annotations and MinIO audio.")
        else:
            print("[SALAMI] MinIO not available or empty — will fall back to local audio_cache.")
    except Exception:
        pass

    if max_songs > 0:
        ann_ids = ann_ids[:max_songs]

    entries: list[dict] = []
    for raw_id in ann_ids:
        for ann in annotators:
            segs = parse_salami_annotation(raw_id, annotator=ann)
            if segs:
                entries.append({
                    "song_id":      f"salami_{raw_id}_ann{ann}",
                    "raw_id":       raw_id,
                    "raw_track_id": f"salami_{raw_id}",
                    "annotator_id": ann,
                    "dataset":      "salami",
                    "segments":     segs,
                })
                break  # use the first available annotator

    print(f"[SALAMI] {len(entries)} songs with annotations to process.")
    return entries


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_rows(entry: dict) -> list[dict] | None:
    """Build one row per segment; return None when audio is unavailable."""
    song_id       = entry["song_id"]
    raw_id        = entry["raw_id"]
    raw_track_id  = entry["raw_track_id"]
    annotator_id  = entry["annotator_id"]
    segments      = entry["segments"]
    dataset       = entry["dataset"]

    audio_bytes = _get_audio_bytes(raw_id)
    if audio_bytes is None:
        print(f"  [skip] {song_id}: no audio (MinIO + local cache both empty)")
        return None

    # build_segment_descriptors now accepts bytes; audio_io.load_audio uses ffmpeg.
    from shared.labeling.heuristic import build_segment_descriptors
    descriptors = build_segment_descriptors(audio_bytes, segments)
    if descriptors is None:
        print(f"  [skip] {song_id}: descriptor extraction failed")
        return None

    X, feat_names = build_segment_label_vectors(segments, descriptors=descriptors)

    rows: list[dict] = []
    for idx, (seg, feat_vec) in enumerate(zip(segments, X)):
        raw_label = seg.get("label", "Unknown")
        canonical = normalize_label(raw_label)
        if canonical not in CANONICAL_LABELS or canonical.startswith(("Section ", "Motif ")):
            canonical = "Other"

        row: dict = {
            "song_id":      song_id,
            "raw_track_id": raw_track_id,
            "annotator_id": annotator_id,
            "dataset":      dataset,
            "segment_idx":  idx,
            "start":        round(float(seg.get("start", 0.0)), 3),
            "end":          round(float(seg.get("end",   0.0)), 3),
            "label":        canonical,
        }
        for fname, fval in zip(feat_names, feat_vec.tolist()):
            row[fname] = fval
        rows.append(row)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment-label training dataset.")
    parser.add_argument("--max-songs",  type=int, default=0,
                        help="Max SALAMI songs to process (0 = all).")
    parser.add_argument("--annotators", nargs="+", type=int, default=[1, 2],
                        help="SALAMI annotator indices to try (default: 1 2).")
    parser.add_argument("--output",     default=OUTPUT_PARQUET)
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallel worker processes (default: 4).")
    args = parser.parse_args()

    entries = _load_salami_entries(args.annotators, args.max_songs)
    if not entries:
        print("No songs found. Check annotations dir and MinIO/audio_cache.")
        sys.exit(1)

    print(f"\nExtracting features for {len(entries)} songs  (workers={args.workers}) …")
    all_rows: list[dict] = []
    t0 = time.perf_counter()

    from concurrent.futures import ThreadPoolExecutor as _Executor, as_completed
    import traceback as _tb
    with _Executor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract_rows, entry): entry for entry in entries}
        done = 0
        for fut in as_completed(futures):
            try:
                rows = fut.result()
            except Exception:
                _tb.print_exc()
                rows = []
            done += 1
            if rows:
                all_rows.extend(rows)
            if done % 10 == 0 or done == len(entries):
                print(
                    f"  {done}/{len(entries)}  "
                    f"segments: {len(all_rows)}  "
                    f"elapsed: {time.perf_counter() - t0:.1f}s"
                )

    if not all_rows:
        print("Feature extraction produced no rows.")
        sys.exit(1)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\nWrote {len(df)} rows  ({df['song_id'].nunique()} songs)  → {args.output}")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
