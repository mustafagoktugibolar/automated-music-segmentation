#!/usr/bin/env python
"""Quick param-config comparison over locally cached SALAMI tracks.

Runs _analyze_content with several param sets on every track in
/app/data/audio_cache that has an annotation, prints macro P/R/F1 per config.
Usage: PYTHONPATH=/app python /app/scripts/exp_eval.py
"""
from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from workers.segmenters.segmentation_service import _analyze_content
from backend.services.salami_parser import parse_salami_annotation
from backend.services.evaluation_service import compute_boundary_metrics

CACHE_DIR = "/app/data/audio_cache"

CONFIGS = {
    "defaults":          {},
    "min8":              {"min_segment_duration_seconds": 8.0},
    "prom12":            {"novelty_prominence": 0.12},
    "min8_prom12":       {"min_segment_duration_seconds": 8.0, "novelty_prominence": 0.12},
}


def annotations_for(song_id: str) -> list[list[dict]]:
    out = []
    for ann in (1, 2):
        segs = parse_salami_annotation(song_id, annotator=ann)
        if segs:
            out.append(segs)
    return out


def main() -> None:
    tracks = []
    for name in sorted(os.listdir(CACHE_DIR)):
        if not name.endswith(".mp3"):
            continue
        sid = name[:-4]
        anns = annotations_for(sid)
        if anns:
            with open(os.path.join(CACHE_DIR, name), "rb") as f:
                tracks.append((sid, f.read(), anns))
    print(f"tracks with cached audio + annotation: {len(tracks)}", flush=True)

    for cfg_name, params in CONFIGS.items():
        rows = []
        for sid, audio, anns in tracks:
            try:
                out = _analyze_content(audio, filename=f"{sid}.mp3", params=params)
            except Exception as exc:
                print(f"  {sid}: ERROR {exc}", flush=True)
                continue
            est = out.get("segments", [])
            best = None
            for ref in anns:
                m = compute_boundary_metrics(ref, est, tolerance=0.5)
                if best is None or m["f_measure"] > best["f_measure"]:
                    best = m
            rows.append((sid, best))
        if not rows:
            continue
        mp = sum(r["precision"] for _, r in rows) / len(rows)
        mr = sum(r["recall"] for _, r in rows) / len(rows)
        mf = sum(r["f_measure"] for _, r in rows) / len(rows)
        per = "  ".join(f"{sid}:{r['f_measure']:.2f}" for sid, r in rows)
        print(f"[{cfg_name:10s}] P={mp:.3f} R={mr:.3f} F1={mf:.3f} (n={len(rows)})", flush=True)
        print(f"    {per}", flush=True)


if __name__ == "__main__":
    main()
