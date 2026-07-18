#!/usr/bin/env python
"""
Submit fusion segmentation for every unique song already in media/uploads.
Run inside the backend container:

    docker exec music_segmentation_backend \
        python /app/scripts/run_fusion_all.py [--algorithms custom_librosa foote scluster fusion] [--dry-run]

Workers pick up the tasks from RabbitMQ automatically.
Results land in the DB as usual.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_app = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _app not in sys.path:
    sys.path.insert(0, _app)

from segmentation.infrastructure.storage.db_models import SegmentationTask
from segmentation.infrastructure.storage.postgres import SessionLocal
from segmentation.infrastructure.logging import get_logger
from segmentation.infrastructure.messaging.rabbitmq import RabbitMQClient
from segmentation.core.segmentation.utils import BASELINE_ALGORITHMS, canonical_algorithm_name

logger = get_logger()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "media/uploads")

ALGO_ROUTING_KEYS = {
    "custom_librosa": "segmentation.custom",
    "foote":          "segmentation.foote",
    "cnmf":           "segmentation.cnmf",
    "scluster":       "segmentation.scluster",
    "fusion":         "segmentation.fusion",
}


def _scan_uploads(upload_dir: str) -> dict[str, str]:
    """Return {song_id: best_file_path} deduped by largest file."""
    pattern = re.compile(r"^[0-9a-f\-]+_(\d+)\.mp3$", re.IGNORECASE)
    best: dict[str, tuple[int, str]] = {}
    for name in os.listdir(upload_dir):
        m = pattern.match(name)
        if not m:
            continue
        song_id = m.group(1)
        full_path = os.path.join(upload_dir, name)
        size = os.path.getsize(full_path)
        if song_id not in best or size > best[song_id][0]:
            best[song_id] = (size, full_path)
    return {sid: path for sid, (_, path) in best.items()}


def _expand_algorithms(algorithms: list[str]) -> tuple[list[str], list[str]]:
    """Return (expected, dispatch) — fusion auto-adds all baseline algos to dispatch."""
    expected = [canonical_algorithm_name(a) for a in algorithms]
    dispatch = [a for a in expected if a != "fusion"]
    if "fusion" in expected:
        for base in BASELINE_ALGORITHMS:
            if base not in expected:
                expected.insert(0, base)
            if base not in dispatch:
                dispatch.append(base)
    return expected, dispatch


def _create_task(task_id: str, song_id: str, filename: str, expected: list[str]) -> None:
    db = SessionLocal()
    try:
        task = SegmentationTask(
            task_id=task_id,
            filename=filename,
            status="processing",
            results={},
            expected_algorithms=[a.lower() for a in expected],
            source_type="upload",
            source_song_id=song_id,
            requested_params={},
        )
        db.add(task)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("DB insert failed for %s: %s", task_id, exc)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithms", nargs="+",
        default=["custom_librosa", "foote", "scluster", "fusion"],
        help="Algorithms to run (cnmf omitted by default — it's slow).",
    )
    parser.add_argument("--max-songs", type=int, default=0, help="0 = all")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't publish.")
    parser.add_argument("--delay-ms", type=int, default=100,
                        help="Milliseconds between publishes to avoid overwhelming RabbitMQ.")
    args = parser.parse_args()

    songs = _scan_uploads(UPLOAD_DIR)
    if not songs:
        print(f"No songs found in {UPLOAD_DIR}")
        sys.exit(1)

    if args.max_songs > 0:
        songs = dict(list(songs.items())[: args.max_songs])

    expected, dispatch = _expand_algorithms(args.algorithms)

    print(f"Songs found:   {len(songs)}")
    print(f"Algorithms:    {dispatch}")
    print(f"Expected keys: {expected}")
    print(f"Dry run:       {args.dry_run}")

    if args.dry_run:
        for sid, path in list(songs.items())[:5]:
            print(f"  song_id={sid}  file={path}")
        print("  ... (dry run, not publishing)")
        return

    rabbitmq = RabbitMQClient(service_name="fusion-batch-runner")
    submitted = 0
    failed = 0

    for i, (song_id, file_path) in enumerate(songs.items(), 1):
        task_id = str(uuid.uuid4())
        filename = os.path.basename(file_path)

        _create_task(task_id, song_id, filename, expected)

        payload = {
            "task_id": task_id,
            "source_type": "upload",
            "original_filename": filename,
            "file_path": file_path,
            "content_type": "audio/mpeg",
            "algorithms": [a for a in expected if a != "fusion"],
            "params": {},
        }

        try:
            for algo in dispatch:
                routing_key = ALGO_ROUTING_KEYS[algo]
                rabbitmq.publish(
                    exchange="segmentation_topic",
                    routing_key=routing_key,
                    message=payload,
                )
            submitted += 1
            if i % 50 == 0 or i == len(songs):
                print(f"  [{i}/{len(songs)}] submitted {submitted} tasks, {failed} errors")
        except Exception as exc:
            logger.error("Publish failed for song %s: %s", song_id, exc)
            failed += 1

        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)

    print(f"\nDone. Submitted={submitted}  Errors={failed}")
    print("Workers will process tasks in the background.")
    print("Check progress: docker-compose logs -f worker-msaf-cnmf worker-msaf-foote worker-msaf-scluster worker-fusion")


if __name__ == "__main__":
    main()
