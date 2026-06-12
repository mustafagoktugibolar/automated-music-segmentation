#!/usr/bin/env python
"""
Full-Dataset Segmentation Evaluation with Run Logging
=====================================================
Evaluates the custom segmentation algorithm over every SALAMI track that is
available locally (platform uploads + download cache) — no network needed —
and persists the results of each run so successive runs are comparable.

Audio sources (in priority order, deduped by song id):
    /app/media/uploads/<uuid>_<song_id>.mp3   (largest file wins on duplicates)
    /app/data/audio_cache/<song_id>.mp3

Outputs per run (under /app/data/eval_runs/<run_id>/):
    results.csv    -- per-track metrics
    summary.txt    -- human-readable report (same format as batch_eval)
    params.json    -- exact algorithm params used for the run
Plus one appended line in /app/data/eval_runs/history.csv with the macro
metrics, so the whole tuning history stays in one greppable file.

Run from the host via scripts/run_full_eval.sh, or directly:
    docker exec -e PYTHONPATH=/app worker-user-code \
        python /app/scripts/full_dataset_eval.py --label my-experiment
    ... --params '{"novelty_prominence": 0.10}' --max-tracks 50
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, ".."))
for p in (_app_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

from workers.segmenters.segmentation_service import _analyze_content
from backend.services.salami_parser import parse_salami_annotation
from backend.services.evaluation_service import compute_boundary_metrics
from batch_eval import generate_report, load_metadata  # reuse report + titles

UPLOADS_DIR = "/app/media/uploads"
CACHE_DIR   = "/app/data/audio_cache"
RUNS_DIR    = "/app/data/eval_runs"
HISTORY_CSV = os.path.join(RUNS_DIR, "history.csv")

CSV_FIELDS = [
    "song_id", "title", "duration_s", "n_ref", "n_est",
    "precision", "recall", "f_measure", "annotator",
    "tolerance_s", "seg_time_s", "n_segments", "audio_path", "error",
]


# ── Track discovery ───────────────────────────────────────────────────────────

def discover_audio() -> dict[str, str]:
    """Return {song_id: audio_path} for every locally available track."""
    by_sid: dict[str, str] = {}
    for path in glob.glob(os.path.join(UPLOADS_DIR, "*.mp3")):
        sid = os.path.basename(path).rsplit("_", 1)[-1][:-4]
        if not sid.isdigit():
            continue
        # Duplicate uploads of the same song: keep the largest file, which is
        # the most likely to be a complete copy.
        if sid not in by_sid or os.path.getsize(path) > os.path.getsize(by_sid[sid]):
            by_sid[sid] = path
    for path in glob.glob(os.path.join(CACHE_DIR, "*.mp3")):
        sid = os.path.basename(path)[:-4]
        if sid.isdigit():
            by_sid.setdefault(sid, path)
    return by_sid


def annotations_for(song_id: str) -> list[tuple[int, list[dict]]]:
    out = []
    for ann in (1, 2):
        segs = parse_salami_annotation(song_id, annotator=ann)
        if segs:
            out.append((ann, segs))
    return out


# ── Single-track evaluation ───────────────────────────────────────────────────

def evaluate_track(
    song_id: str,
    audio_path: str,
    title: str,
    params: dict,
    tolerance: float,
) -> Optional[dict]:
    anns = annotations_for(song_id)
    if not anns:
        return None

    try:
        with open(audio_path, "rb") as f:
            audio = f.read()
    except OSError as exc:
        return {"song_id": song_id, "title": title, "error": f"read: {exc}",
                "audio_path": audio_path}

    t0 = time.perf_counter()
    try:
        out = _analyze_content(audio, filename=os.path.basename(audio_path),
                               params=params or None)
    except Exception as exc:
        return {"song_id": song_id, "title": title, "error": f"segmentation: {exc}",
                "audio_path": audio_path}
    seg_time = time.perf_counter() - t0

    est = out.get("segments", [])
    best: Optional[dict] = None
    best_ann = 0
    for ann_idx, ref in anns:
        m = compute_boundary_metrics(ref, est, tolerance=tolerance)
        if best is None or m["f_measure"] > best["f_measure"]:
            best, best_ann = m, ann_idx

    return {
        "song_id":     song_id,
        "title":       title,
        "duration_s":  out.get("duration_seconds", 0),
        "n_ref":       best["n_boundaries_ref"],
        "n_est":       best["n_boundaries_est"],
        "precision":   best["precision"],
        "recall":      best["recall"],
        "f_measure":   best["f_measure"],
        "annotator":   best_ann,
        "tolerance_s": tolerance,
        "seg_time_s":  round(seg_time, 2),
        "n_segments":  len(est),
        "audio_path":  audio_path,
        "error":       "",
    }


# ── Run logging ───────────────────────────────────────────────────────────────

def append_history(run_id: str, label: str, rows: list[dict],
                   params: dict, tolerance: float) -> None:
    ok = [r for r in rows if not r.get("error")]
    n = len(ok)
    mean = lambda key: (sum(r[key] for r in ok) / n) if n else 0.0
    entry = {
        "run_id":     run_id,
        "label":      label,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tracks":   len(rows),
        "n_ok":       n,
        "precision":  round(mean("precision"), 4),
        "recall":     round(mean("recall"), 4),
        "f_measure":  round(mean("f_measure"), 4),
        "zero_f1":    sum(1 for r in ok if r["f_measure"] == 0.0),
        "tolerance":  tolerance,
        "params":     json.dumps(params, sort_keys=True),
    }
    write_header = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(entry))
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-dataset segmentation evaluation")
    parser.add_argument("--label", default="", help="Run label recorded in history.csv")
    parser.add_argument("--params", default="{}",
                        help='Algorithm params as JSON, e.g. \'{"novelty_prominence": 0.1}\'')
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--max-tracks", type=int, default=0, help="0 = all")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--verbose", action="store_true",
                        help="Keep worker INFO logs (very chatty)")
    args = parser.parse_args()

    if not args.verbose:
        logging.disable(logging.INFO)

    params = json.loads(args.params)
    run_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{args.label}" if args.label else "")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    meta = load_metadata()
    audio = discover_audio()
    tracks = sorted(
        (sid for sid in audio if annotations_for(sid)),
        key=int,
    )
    if args.max_tracks > 0:
        tracks = tracks[: args.max_tracks]

    print(f"Run {run_id}: {len(tracks)} tracks "
          f"(local audio: {len(audio)}), params={params or 'defaults'}", flush=True)

    rows: list[dict] = []
    done = 0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                evaluate_track, sid, audio[sid],
                meta.get(sid, {}).get("title", "?"), params, args.tolerance,
            ): sid
            for sid in tracks
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            done += 1
            try:
                row = fut.result()
            except Exception as exc:
                row = {"song_id": sid, "title": "?", "error": str(exc), "audio_path": ""}
            if row:
                rows.append(row)
                status = row.get("error") or f"F1={row['f_measure']:.3f}"
                print(f"  [{done:>3}/{len(tracks)}] {sid:>6}  {status}", flush=True)

    elapsed = time.perf_counter() - t_start
    report = generate_report(rows, args.tolerance)
    print("\n" + report, flush=True)
    print(f"\nTotal wall time: {elapsed/60:.1f} min", flush=True)

    rows.sort(key=lambda r: int(r["song_id"]))
    with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump({"params": params, "tolerance": args.tolerance,
                   "n_tracks": len(tracks)}, f, indent=2)
    append_history(run_id, args.label, rows, params, args.tolerance)

    print(f"Saved: {run_dir}/results.csv, summary.txt, params.json", flush=True)
    print(f"History: {HISTORY_CSV}", flush=True)


if __name__ == "__main__":
    main()
