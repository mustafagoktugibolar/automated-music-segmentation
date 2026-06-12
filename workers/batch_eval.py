#!/usr/bin/env python
"""
SALAMI Batch Evaluation Script
================================
Evaluates the custom segmentation algorithm against the full (or a subset of)
SALAMI dataset and prints an actionable report.

Run inside worker-user-code container:
    docker exec worker-user-code python /app/workers/../scripts/batch_eval.py
    docker exec worker-user-code python /app/workers/../scripts/batch_eval.py --max-tracks 50
    docker exec worker-user-code python /app/workers/../scripts/batch_eval.py --max-tracks 0  # all tracks

Output:
    /app/data/eval_results.csv     -- per-track metrics
    /app/data/eval_summary.txt     -- human-readable report (also printed to stdout)
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

# ── Imports (require librosa, mir_eval — available in worker conda env) ───────
from workers.segmenters.segmentation_service import _analyze_content
from backend.services.salami_parser import parse_salami_annotation
from backend.services.evaluation_service import compute_boundary_metrics

# ── Constants ─────────────────────────────────────────────────────────────────
ANNOTATIONS_DIR = "/app/data/salami/annotations"
METADATA_CSV    = "/app/data/salami/metadata/id_index_internetarchive.csv"
RESULTS_CSV     = "/app/data/eval_results.csv"
SUMMARY_TXT     = "/app/data/eval_summary.txt"

DOWNLOAD_TIMEOUT_S = 90
WORKER_CONCURRENCY = 3

# MinIO / S3 config (read from env at runtime)
_S3_ENDPOINT  = os.getenv("S3_ENDPOINT")
_S3_KEY       = os.getenv("S3_ACCESS_KEY")
_S3_SECRET    = os.getenv("S3_SECRET_KEY")
_S3_BUCKET    = os.getenv("S3_BUCKET_RAW")
_S3_PREFIX    = os.getenv("DATASET_PREFIX", "").strip().strip("/")


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_metadata() -> dict[str, dict]:
    """Return {song_id: {title, artist, url, duration_s}} from metadata CSV."""
    meta: dict[str, dict] = {}
    if not os.path.exists(METADATA_CSV):
        return meta
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("SONG_ID", "")).strip()
            if sid:
                meta[sid] = {
                    "title":      row.get("TITLE", "?"),
                    "artist":     row.get("ARTIST", "?"),
                    "url":        row.get("URL", "").strip(),
                    "duration_s": int(row.get("SONG_DURATION", 0) or 0),
                }
    return meta


def list_annotated_song_ids() -> list[str]:
    """Return sorted list of song IDs that have at least one annotation file."""
    if not os.path.isdir(ANNOTATIONS_DIR):
        return []
    ids = []
    for name in os.listdir(ANNOTATIONS_DIR):
        if name.isdigit():
            # Check that at least one annotation file exists
            ann_dir = os.path.join(ANNOTATIONS_DIR, name)
            has_ann = (
                os.path.exists(os.path.join(ann_dir, "parsed", "textfile1_functions.txt"))
                or os.path.exists(os.path.join(ann_dir, "textfile1.txt"))
            )
            if has_ann:
                ids.append(name)
    return sorted(ids, key=int)


# ── MinIO audio source ────────────────────────────────────────────────────────

def _minio_available() -> bool:
    return bool(_S3_KEY and _S3_SECRET and _S3_BUCKET)


def _minio_client():
    import boto3
    session = boto3.session.Session()
    return session.client(
        "s3",
        aws_access_key_id=_S3_KEY,
        aws_secret_access_key=_S3_SECRET,
        endpoint_url=_S3_ENDPOINT or None,
    )


def list_minio_song_ids() -> list[str]:
    """Return numeric song IDs whose .mp3 exists in the configured MinIO bucket."""
    if not _minio_available():
        print("[WARNING] MinIO not configured — S3_BUCKET_RAW / S3_ACCESS_KEY missing.", flush=True)
        return []
    client = _minio_client()
    try:
        paginator = client.get_paginator("list_objects_v2")
        song_ids: set[str] = set()
        for page in paginator.paginate(Bucket=_S3_BUCKET):
            for obj in page.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name.endswith(".mp3"):
                    sid = name[:-4]
                    if sid.isdigit():
                        song_ids.add(sid)
        return sorted(song_ids, key=int)
    except Exception as exc:
        print(f"[ERROR] MinIO list failed: {exc}", flush=True)
        return []


def download_from_minio(song_id: str) -> Optional[bytes]:
    """Download audio bytes for song_id from MinIO. Returns None if not found."""
    if not _minio_available():
        return None
    client = _minio_client()
    candidates = [f"songs/{song_id}.mp3", f"{song_id}.mp3"]
    if _S3_PREFIX:
        candidates.append(f"{_S3_PREFIX}/songs/{song_id}.mp3")
    for key in candidates:
        try:
            resp = client.get_object(Bucket=_S3_BUCKET, Key=key)
            return resp["Body"].read()
        except Exception:
            continue
    return None


# ── Audio download (kept for fallback / testing) ──────────────────────────────

def download_audio(url: str, timeout: int = DOWNLOAD_TIMEOUT_S) -> Optional[bytes]:
    """Download audio from URL, return raw bytes or None on failure."""
    if not url:
        return None
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "music-segmentation-eval/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        print(f"    [download error] {exc}", flush=True)
        return None


# ── SALAMI annotation loading ─────────────────────────────────────────────────

def load_best_annotation(song_id: str) -> Optional[list[dict]]:
    """
    Load ground truth for song_id.
    Tries annotator 1 first, falls back to annotator 2.
    Returns None if neither exists.
    """
    for ann in (1, 2):
        segs = parse_salami_annotation(song_id, annotator=ann)
        if segs:
            return segs
    return None


def load_both_annotations(song_id: str) -> list[list[dict]]:
    """Return list of up to 2 annotation lists (one per annotator)."""
    result = []
    for ann in (1, 2):
        segs = parse_salami_annotation(song_id, annotator=ann)
        if segs:
            result.append(segs)
    return result


# ── Single track evaluation ───────────────────────────────────────────────────

def evaluate_track(
    song_id: str,
    title: str,
    tolerance: float,
) -> Optional[dict]:
    """
    Fetch audio from MinIO, run segmentation, evaluate against both SALAMI annotators.
    Returns metrics dict or None if track cannot be processed.
    """
    # Load annotations
    annotations = load_both_annotations(song_id)
    if not annotations:
        return None

    # Download audio from MinIO
    t_dl = time.perf_counter()
    audio_bytes = download_from_minio(song_id)
    dl_time = time.perf_counter() - t_dl
    if audio_bytes is None:
        return {"song_id": song_id, "title": title, "error": "minio_not_found", "dl_time_s": dl_time}

    # Run segmentation
    t_seg = time.perf_counter()
    try:
        output = _analyze_content(
            audio_bytes,
            filename=f"{song_id}.mp3",
            content_type="audio/mpeg",
        )
    except Exception as exc:
        return {"song_id": song_id, "title": title, "error": f"segmentation: {exc}"}
    seg_time = time.perf_counter() - t_seg

    est_segments = output.get("segments", [])

    # Evaluate against each annotator, take best F-measure (MIREX protocol)
    best: Optional[dict] = None
    for ann_idx, ref_segments in enumerate(annotations, 1):
        metrics = compute_boundary_metrics(ref_segments, est_segments, tolerance=tolerance)
        if best is None or metrics["f_measure"] > best["f_measure"]:
            best = metrics
            best["annotator"] = ann_idx

    if best is None:
        return None

    return {
        "song_id":          song_id,
        "title":            title,
        "duration_s":       output.get("duration_seconds", 0),
        "n_ref":            best["n_boundaries_ref"],
        "n_est":            best["n_boundaries_est"],
        "precision":        best["precision"],
        "recall":           best["recall"],
        "f_measure":        best["f_measure"],
        "annotator":        best["annotator"],
        "tolerance_s":      tolerance,
        "seg_time_s":       round(seg_time, 2),
        "dl_time_s":        round(dl_time, 2),
        "n_segments":       len(est_segments),
        "error":            "",
    }


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(rows: list[dict], tolerance: float) -> str:
    ok    = [r for r in rows if not r.get("error")]
    errs  = [r for r in rows if r.get("error")]

    lines = []
    lines.append("=" * 60)
    lines.append("  SALAMI Batch Evaluation — custom algorithm")
    lines.append("=" * 60)
    lines.append(f"  Tracks attempted : {len(rows)}")
    lines.append(f"  Tracks evaluated : {len(ok)}")
    lines.append(f"  Errors / skipped : {len(errs)}")
    lines.append(f"  Tolerance        : ±{tolerance}s")
    lines.append("")

    if not ok:
        lines.append("  No tracks evaluated successfully.")
        return "\n".join(lines)

    precision  = [r["precision"]  for r in ok]
    recall     = [r["recall"]     for r in ok]
    f_measure  = [r["f_measure"]  for r in ok]
    n_ref      = [r["n_ref"]      for r in ok]
    n_est      = [r["n_est"]      for r in ok]
    seg_times  = [r["seg_time_s"] for r in ok]

    def mean(xs):  return sum(xs) / len(xs) if xs else 0.0
    def stdev(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0

    lines.append("--- Macro-Averaged Metrics ---")
    lines.append(f"  Precision : {mean(precision):.3f}  (std {stdev(precision):.3f})")
    lines.append(f"  Recall    : {mean(recall):.3f}  (std {stdev(recall):.3f})")
    lines.append(f"  F-measure : {mean(f_measure):.3f}  (std {stdev(f_measure):.3f})")
    lines.append("")
    lines.append("--- Boundary Count Analysis ---")
    lines.append(f"  Avg ref  boundaries : {mean(n_ref):.1f}")
    lines.append(f"  Avg est  boundaries : {mean(n_est):.1f}")
    ratio = mean(n_est) / mean(n_ref) if mean(n_ref) > 0 else 0
    if ratio < 0.5:
        lines.append(f"  Ratio est/ref       : {ratio:.2f}  ← UNDER-SEGMENTING")
    elif ratio > 2.0:
        lines.append(f"  Ratio est/ref       : {ratio:.2f}  ← OVER-SEGMENTING")
    else:
        lines.append(f"  Ratio est/ref       : {ratio:.2f}  (ok)")
    lines.append(f"  Avg segmentation    : {mean(seg_times):.1f}s per track")
    lines.append("")

    # F=0 tracks
    zero_f = [r for r in ok if r["f_measure"] == 0.0]
    lines.append(f"--- Zero-F1 tracks : {len(zero_f)} / {len(ok)} ---")
    for r in zero_f[:10]:
        lines.append(f"  {r['song_id']:>6}  {r['title'][:40]:<40}  est={r['n_est']}  ref={r['n_ref']}")
    if len(zero_f) > 10:
        lines.append(f"  ... and {len(zero_f)-10} more")
    lines.append("")

    # Worst 10
    worst = sorted(ok, key=lambda r: r["f_measure"])[:10]
    lines.append("--- Worst 10 tracks (F1) ---")
    lines.append(f"  {'ID':>6}  {'Title':<36}  {'P':>6}  {'R':>6}  {'F1':>6}  {'est':>4}  {'ref':>4}")
    lines.append("  " + "-" * 72)
    for r in worst:
        lines.append(
            f"  {r['song_id']:>6}  {r['title'][:36]:<36}"
            f"  {r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f_measure']:>6.3f}"
            f"  {r['n_est']:>4}  {r['n_ref']:>4}"
        )
    lines.append("")

    # Best 10
    best10 = sorted(ok, key=lambda r: r["f_measure"], reverse=True)[:10]
    lines.append("--- Best 10 tracks (F1) ---")
    lines.append(f"  {'ID':>6}  {'Title':<36}  {'P':>6}  {'R':>6}  {'F1':>6}  {'est':>4}  {'ref':>4}")
    lines.append("  " + "-" * 72)
    for r in best10:
        lines.append(
            f"  {r['song_id']:>6}  {r['title'][:36]:<36}"
            f"  {r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f_measure']:>6.3f}"
            f"  {r['n_est']:>4}  {r['n_ref']:>4}"
        )
    lines.append("")

    # F1 distribution buckets
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for r in ok:
        f = r["f_measure"]
        if f < 0.2:   buckets["0.0-0.2"] += 1
        elif f < 0.4: buckets["0.2-0.4"] += 1
        elif f < 0.6: buckets["0.4-0.6"] += 1
        elif f < 0.8: buckets["0.6-0.8"] += 1
        else:         buckets["0.8-1.0"] += 1
    lines.append("--- F1 Distribution ---")
    for label, count in buckets.items():
        bar = "█" * count
        lines.append(f"  {label}  {count:>4}  {bar}")
    lines.append("")

    if errs:
        lines.append("--- Errors ---")
        for r in errs[:10]:
            lines.append(f"  {r['song_id']:>6}  {r.get('error', '?')}")
        if len(errs) > 10:
            lines.append(f"  ... and {len(errs)-10} more")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = [
        "song_id", "title", "duration_s", "n_ref", "n_est",
        "precision", "recall", "f_measure", "annotator",
        "tolerance_s", "seg_time_s", "dl_time_s", "n_segments", "error",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SALAMI batch evaluation for custom algorithm")
    parser.add_argument("--max-tracks", type=int, default=20,
                        help="Max tracks to evaluate (0 = all, default 20)")
    parser.add_argument("--tolerance",  type=float, default=0.5,
                        help="Boundary tolerance in seconds (default 0.5 — MIREX standard)")
    parser.add_argument("--output-csv", default=RESULTS_CSV,
                        help="Path for per-track CSV output")
    parser.add_argument("--concurrency", type=int, default=WORKER_CONCURRENCY,
                        help="Parallel download/eval workers")
    args = parser.parse_args()

    print(f"\nSALAMI Batch Evaluation", flush=True)
    print(f"  tolerance  : ±{args.tolerance}s", flush=True)
    print(f"  max tracks : {args.max_tracks or 'all'}", flush=True)
    print(f"  concurrency: {args.concurrency}", flush=True)
    print(flush=True)

    # Load metadata CSV for titles (optional — used only for display)
    meta = load_metadata()
    print(f"Metadata loaded: {len(meta)} entries", flush=True)

    # List songs available in MinIO
    minio_ids = list_minio_song_ids()
    print(f"MinIO songs    : {len(minio_ids)}", flush=True)

    if not minio_ids:
        print("ERROR: No songs found in MinIO. Check S3_BUCKET_RAW / S3_ACCESS_KEY env vars.", flush=True)
        return

    # Intersect with annotated song IDs
    annotated_ids = set(list_annotated_song_ids())
    print(f"Annotated      : {len(annotated_ids)} songs", flush=True)

    candidates = [sid for sid in minio_ids if sid in annotated_ids]
    print(f"Overlap        : {len(candidates)} tracks (in MinIO AND annotated)", flush=True)

    if args.max_tracks and args.max_tracks > 0:
        candidates = candidates[: args.max_tracks]

    print(f"Evaluating     : {len(candidates)} tracks\n", flush=True)

    # Evaluate in parallel
    rows: list[dict] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                evaluate_track,
                sid,
                meta.get(sid, {}).get("title", sid),
                args.tolerance,
            ): sid
            for sid in candidates
        }

        for fut in as_completed(futures):
            sid = futures[fut]
            completed += 1
            title = meta.get(sid, {}).get("title", sid)
            try:
                result = fut.result()
                if result:
                    rows.append(result)
                    status = result.get("error") or f"F1={result.get('f_measure', 0):.3f}"
                    print(f"  [{completed:>3}/{len(candidates)}] {sid:>6}  {status}", flush=True)
                else:
                    rows.append({"song_id": sid, "title": title, "error": "no_annotation"})
                    print(f"  [{completed:>3}/{len(candidates)}] {sid:>6}  skip (no annotation)", flush=True)
            except Exception as exc:
                rows.append({"song_id": sid, "title": title, "error": str(exc)})
                print(f"  [{completed:>3}/{len(candidates)}] {sid:>6}  ERROR: {exc}", flush=True)

    # Generate and print report
    report = generate_report(rows, args.tolerance)
    print("\n" + report, flush=True)

    # Save outputs
    save_csv(rows, args.output_csv)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved: {args.output_csv}", flush=True)
    print(f"Saved: {SUMMARY_TXT}", flush=True)


if __name__ == "__main__":
    main()
