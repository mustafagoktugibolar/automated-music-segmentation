"""
Evaluation API.

Provides endpoints to:
- Run boundary detection evaluation for a completed segmentation task against a track's ground truth
- Compare multiple algorithms on the same track
- Retrieve stored evaluation results
- Run a batch evaluation across the full SALAMI dataset
"""

import asyncio
import csv
import json
import os
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from segmentation.infrastructure.storage.db_models import BatchEvalJob, DatasetTrack, EvaluationRun, SegmentationTask
from segmentation.infrastructure.storage.postgres import SessionLocal
from segmentation.core.segmentation.evaluation import compute_boundary_metrics, compute_boundary_metrics_multi
from segmentation.core.segmentation.salami_parser import parse_salami_annotation
from segmentation.application.orchestration.segmentation_orchestrator import SegmentationOrchestrator
from segmentation.infrastructure.logging import get_logger
from segmentation.core.segmentation.utils import BASELINE_ALGORITHMS, canonical_algorithm_name, extract_segments

logger = get_logger()
router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

UPLOAD_DIR        = os.getenv("UPLOAD_DIR", "media/uploads")
SALAMI_META_CSV   = "/app/data/salami/metadata/id_index_internetarchive.csv"
SALAMI_ANNOT_DIR  = "/app/data/salami/annotations"

# In-memory store for active batch jobs  {job_id: {lines, done, error, summary}}
_batch_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Batch eval helpers
# ---------------------------------------------------------------------------

def _load_salami_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not os.path.exists(SALAMI_META_CSV):
        return meta
    with open(SALAMI_META_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("SONG_ID", "")).strip()
            if sid:
                meta[sid] = {
                    "title":  row.get("TITLE", "?"),
                    "url":    row.get("URL", "").strip(),
                }
    return meta


def _list_annotated_song_ids() -> list[str]:
    if not os.path.isdir(SALAMI_ANNOT_DIR):
        return []
    ids = []
    for name in os.listdir(SALAMI_ANNOT_DIR):
        if name.isdigit():
            d = os.path.join(SALAMI_ANNOT_DIR, name)
            if (os.path.exists(os.path.join(d, "parsed", "textfile1_functions.txt"))
                    or os.path.exists(os.path.join(d, "textfile1.txt"))):
                ids.append(name)
    return sorted(ids, key=int)


def _download_audio(url: str, timeout: int = 90) -> Optional[bytes]:
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "music-segmentation/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _list_minio_song_ids() -> list[str]:
    from segmentation.infrastructure.storage.object_store import list_song_ids
    return list_song_ids()


def _download_from_minio(song_id: str) -> Optional[bytes]:
    from segmentation.infrastructure.storage.object_store import download
    return download(song_id)


def _dispatch_to_worker(
    audio_bytes: bytes,
    filename: str,
    algorithm: str | list[str] = "custom_librosa",
) -> str:
    """Save audio to disk, create DB task, publish to RabbitMQ. Returns task_id."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    task_id   = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{filename}")

    with open(file_path, "wb") as fh:
        fh.write(audio_bytes)

    requested_algorithms = algorithm if isinstance(algorithm, list) else [algorithm]
    orchestrator = SegmentationOrchestrator()
    algorithms = orchestrator._normalize_algorithms(requested_algorithms)
    expected_algorithms, dispatch_algorithms = orchestrator._expand_requested_algorithms(algorithms)
    requested_params = {}

    db = SessionLocal()
    try:
        db.add(SegmentationTask(
            task_id=task_id,
            filename=filename,
            status="processing",
            results={},
            expected_algorithms=expected_algorithms,
            source_type="upload",
            requested_params=requested_params,
        ))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    task_payload = {
        "task_id": task_id,
        "source_type": "upload",
        "original_filename": filename,
        "file_path": file_path,
        "content_type": "audio/mpeg",
        "algorithms": algorithms,
        "params": requested_params or {},
    }
    orchestrator._publish_tasks(task_payload, dispatch_algorithms)
    return task_id


def _wait_for_task(task_id: str, timeout: int = 240, poll_s: float = 3.0) -> Optional[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        db = SessionLocal()
        try:
            task = db.query(SegmentationTask).filter(
                SegmentationTask.task_id == task_id
            ).first()
            if task and task.status == "completed":
                return task.results or {}
            if task and task.status == "failed":
                return None
        finally:
            db.close()
        time.sleep(poll_s)
    return None


async def _wait_for_task_async(task_id: str, timeout: int = 240, poll_s: float = 3.0) -> Optional[dict]:
    """Non-blocking task poller — yields to event loop between polls."""
    def _query() -> Optional[dict] | str:
        db = SessionLocal()
        try:
            task = db.query(SegmentationTask).filter(
                SegmentationTask.task_id == task_id
            ).first()
            if task and task.status == "completed":
                return task.results or {}
            if task and task.status == "failed":
                return None
            return "pending"
        finally:
            db.close()

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await asyncio.to_thread(_query)
        if result != "pending":
            return result  # dict on success, None on failure
        await asyncio.sleep(poll_s)
    return None


def _batch_summary(rows: list[dict], tolerance: float) -> str:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return "No tracks evaluated successfully."

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    included = [r for r in ok if not r.get("is_outlier", False)]
    outlier_count = len(ok) - len(included)

    lines = [
        f"Tracks OK   : {len(ok)} / {len(rows)}",
        f"Outliers    : {outlier_count} (F1@3s < threshold, excluded from filtered avg)",
        f"Tolerance   : ±{tolerance}s",
        "",
        "Algorithm comparison (included / filtered):",
        "algorithm          avg_f1_0_5   avg_f1_3_0   filtered_f1   avg_precision_0_5   avg_recall_0_5   avg_est/ref",
    ]

    by_algorithm: dict[str, list[dict]] = {}
    for row in ok:
        by_algorithm.setdefault(row.get("algorithm", "unknown"), []).append(row)

    for algorithm, algo_rows in sorted(by_algorithm.items()):
        included_algo = [r for r in algo_rows if not r.get("is_outlier", False)]
        avg_f1_0_5   = mean([r.get("f1_0_5", r.get("f_measure", 0.0)) for r in algo_rows])
        avg_f1_3_0   = mean([r.get("f1_3_0", 0.0) for r in algo_rows])
        filtered_f1  = mean([r.get("f_measure", 0.0) for r in included_algo])
        avg_p_0_5    = mean([r.get("precision_0_5", r.get("precision", 0.0)) for r in algo_rows])
        avg_r_0_5    = mean([r.get("recall_0_5", r.get("recall", 0.0)) for r in algo_rows])
        r_ref  = mean([r["n_ref"] for r in algo_rows])
        r_est  = mean([r["n_est"] for r in algo_rows])
        ratio  = r_est / r_ref if r_ref > 0 else 0
        lines.append(
            f"{algorithm:<18} {avg_f1_0_5:>10.3f}   {avg_f1_3_0:>10.3f}   "
            f"{filtered_f1:>11.3f}   "
            f"{avg_p_0_5:>17.3f}   {avg_r_0_5:>14.3f}   {ratio:>10.2f}"
        )

    lines.extend([
        "",
        "Worst tracks:",
    ])
    for r in sorted(ok, key=lambda x: x["f_measure"])[:5]:
        lines.append(f"  {r['song_id']:>6}  {r.get('algorithm', '?'):<14}  {r['title'][:28]:<28}  F1={r['f_measure']:.3f}  est={r['n_est']}  ref={r['n_ref']}")
    lines.append("")
    lines.append("Best tracks:")
    for r in sorted(ok, key=lambda x: x["f_measure"], reverse=True)[:5]:
        lines.append(f"  {r['song_id']:>6}  {r.get('algorithm', '?'):<14}  {r['title'][:28]:<28}  F1={r['f_measure']:.3f}  est={r['n_est']}  ref={r['n_ref']}")
    return "\n".join(lines)


def _save_batch_job_result(job_id: str, rows: list[dict], summary: str | None, error: str | None) -> None:
    from sqlalchemy.orm.attributes import flag_modified
    ok_rows      = [r for r in rows if not r.get("error")]
    included     = [r for r in ok_rows if not r.get("is_outlier", False)]
    use_rows     = included if included else ok_rows
    avg_p = sum(r["precision"]  for r in use_rows) / len(use_rows) if use_rows else None
    avg_r = sum(r["recall"]     for r in use_rows) / len(use_rows) if use_rows else None
    avg_f = sum(r["f_measure"]  for r in use_rows) / len(use_rows) if use_rows else None

    db = SessionLocal()
    try:
        record = db.query(BatchEvalJob).filter(BatchEvalJob.job_id == job_id).first()
        if not record:
            logger.error("BatchEvalJob %s not found in DB — cannot save result.", job_id)
            return
        record.status        = "failed" if error else "completed"
        record.completed_at  = datetime.now(timezone.utc)
        record.summary       = summary
        record.rows          = list(rows)
        flag_modified(record, "rows")
        record.error         = error
        record.tracks_ok     = len(ok_rows)
        record.tracks_total  = len(rows)
        record.avg_precision = avg_p
        record.avg_recall    = avg_r
        record.avg_f1        = avg_f
        db.commit()
        logger.info("BatchEvalJob %s saved: status=%s tracks=%d/%d", job_id, record.status, len(ok_rows), len(rows))
    except Exception as exc:
        db.rollback()
        logger.error("Failed to persist BatchEvalJob %s: %s", job_id, exc, exc_info=True)
    finally:
        db.close()


async def _run_batch_eval_async(
    job_id: str,
    max_tracks: int,
    tolerance: float,
    concurrency: int,
    algorithms: list[str] | None = None,
    tolerances: list[float] | None = None,
    coverage_outlier_threshold: float = 0.20,
) -> None:
    job = _batch_jobs[job_id]
    sem = asyncio.Semaphore(concurrency)
    requested_algorithms = [canonical_algorithm_name(a) for a in (algorithms or ["custom_librosa"])]
    if "fusion" in requested_algorithms:
        algorithms_to_evaluate = list(dict.fromkeys([*BASELINE_ALGORITHMS, "fusion"]))
        algorithms_to_report = {"fusion"}
    else:
        algorithms_to_evaluate = requested_algorithms
        algorithms_to_report = set(requested_algorithms)
    tolerances = tolerances or [tolerance, 3.0]

    def log(line: str) -> None:
        job["lines"].append(line)

    async def dispatch_one(sid: str, title: str) -> dict:
        """Download audio and push one track to the worker queues."""
        ref = await asyncio.to_thread(
            lambda: parse_salami_annotation(sid, annotator=1)
                    or parse_salami_annotation(sid, annotator=2)
        )
        if not ref:
            return {"sid": sid, "title": title, "skip": "no_annotation", "ref": None}

        audio = await asyncio.to_thread(_download_from_minio, sid)
        if audio is None:
            return {"sid": sid, "title": title, "skip": "minio_not_found", "ref": None}

        try:
            task_id = await asyncio.to_thread(_dispatch_to_worker, audio, f"{sid}.mp3", requested_algorithms)
        except Exception as exc:
            logger.error("dispatch_one %s failed: %s", sid, exc)
            return {"sid": sid, "title": title, "skip": f"dispatch_failed: {exc}", "ref": None}
        return {"sid": sid, "title": title, "task_id": task_id, "ref": ref}

    async def collect_one(dispatch: dict, idx: int, total: int) -> list[dict] | dict:
        """Poll for one task result and compute its metrics."""
        sid, title = dispatch["sid"], dispatch["title"]
        if "skip" in dispatch:
            reason = dispatch["skip"]
            log(f"[{idx:>3}/{total}] {sid}  skip: {reason}")
            return {"song_id": sid, "title": title, "error": reason}
        try:
            results = await _wait_for_task_async(dispatch["task_id"])
            if not results:
                log(f"[{idx:>3}/{total}] {sid}  skip: timeout / worker failed")
                return {"song_id": sid, "title": title, "error": "timeout"}

            ref = dispatch["ref"]
            rows: list[dict] = []
            for algo in algorithms_to_evaluate:
                est = extract_segments(results.get(algo))
                if not est:
                    if algo in algorithms_to_report:
                        rows.append({"song_id": sid, "title": title, "algorithm": algo, "error": "missing_result"})
                    continue
                m2 = compute_boundary_metrics(ref, est, tolerance=tolerance)
                m_multi = compute_boundary_metrics_multi(ref, est, tolerances=tolerances)
                row = {
                    "song_id":   sid,
                    "title":     title,
                    "algorithm": algo,
                    "n_ref":     m2["n_boundaries_ref"],
                    "n_est":     m2["n_boundaries_est"],
                    "precision": m2["precision"],
                    "recall":    m2["recall"],
                    "f_measure": m2["f_measure"],
                    "error":     "",
                }
                row.update({k: v for k, v in m_multi.items() if k != "by_tolerance"})
                row["is_outlier"] = row.get("f1_3_0", 0.0) < coverage_outlier_threshold
                if algo in algorithms_to_report:
                    rows.append(row)

            ok_parts = [
                f"{r['algorithm']} F1={r.get('f_measure', 0):.3f}"
                for r in rows if not r.get("error")
            ]
            log(f"[{idx:>3}/{total}] {sid}  " + " | ".join(ok_parts))
            return rows
        except Exception as exc:
            log(f"[{idx:>3}/{total}] {sid}  EXCEPTION: {exc}")
            logger.error("collect_one %s failed", sid, exc_info=True)
            return {"song_id": sid, "title": title, "error": str(exc)}

    async def process_one(sid: str, title: str, idx: int, total: int) -> list[dict] | dict:
        """Keep one concurrency slot until the track is fully evaluated."""
        async with sem:
            dispatch = await dispatch_one(sid, title)
            return await collect_one(dispatch, idx, total)

    try:
        meta          = await asyncio.to_thread(_load_salami_metadata)
        minio_ids     = await asyncio.to_thread(_list_minio_song_ids)
        log(f"MinIO songs : {len(minio_ids)}")
        if not minio_ids:
            log("ERROR: No songs found in MinIO. Check S3_BUCKET_RAW / S3_ACCESS_KEY env vars.")
            return

        annotated_ids = set(await asyncio.to_thread(_list_annotated_song_ids))
        log(f"Annotated   : {len(annotated_ids)} songs")

        candidates = [sid for sid in minio_ids if sid in annotated_ids]
        log(f"Overlap     : {len(candidates)} tracks")
        if max_tracks > 0:
            candidates = candidates[:max_tracks]
        log(f"Evaluating  : {len(candidates)} tracks  (concurrency={concurrency})\n")

        # Keep at most `concurrency` tracks in flight. A slot is released only
        # after the worker result is collected, so queue wait time cannot consume
        # the per-task result timeout for hundreds of undispatched tracks.
        log(f"Processing in a bounded window of {concurrency} tracks...\n")
        gathered = list(await asyncio.gather(*[
            process_one(
                sid,
                meta.get(sid, {}).get("title", sid),
                i,
                len(candidates),
            )
            for i, sid in enumerate(candidates, 1)
        ]))
        rows: list[dict] = []
        for item in gathered:
            if isinstance(item, list):
                rows.extend(item)
            else:
                rows.append(item)

        summary = _batch_summary(rows, tolerance)
        job["summary"] = summary
        job["rows"]    = rows
        log("\n=== SUMMARY ===\n" + summary)

    except Exception as exc:
        import traceback
        job["error"] = str(exc)
        log(f"FATAL: {exc}\n{traceback.format_exc()}")
    finally:
        job["done"] = True
        try:
            await asyncio.shield(asyncio.to_thread(
                _save_batch_job_result,
                job_id,
                job.get("rows", []),
                job.get("summary"),
                job.get("error"),
            ))
        except asyncio.CancelledError:
            _save_batch_job_result(
                job_id,
                job.get("rows", []),
                job.get("summary"),
                job.get("error"),
            )
            raise


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class BatchEvalRequest(BaseModel):
    max_tracks: int  = Field(default=20, ge=0, le=500,
                             description="0 = all available tracks")
    tolerance_seconds: float = Field(default=0.5, gt=0, le=10)
    tolerances: list[float] = Field(default_factory=lambda: [0.5, 3.0])
    algorithms: list[str] = Field(default_factory=lambda: ["custom_librosa", "foote", "cnmf", "scluster", "fusion"])
    concurrency: int = Field(default=3, ge=1, le=10,
                             description="Parallel tracks (match your worker count)")
    coverage_outlier_threshold: float = Field(default=0.20, ge=0.0, le=1.0,
                                              description="Tracks whose F1@3s < threshold are flagged as outliers")


class EvaluationRunRequest(BaseModel):
    task_id: str = Field(min_length=1)
    track_id: str = Field(min_length=1)
    tolerance_seconds: float = Field(default=3.0, gt=0, le=30)


class CompareRequest(BaseModel):
    track_id: str = Field(min_length=1)
    algorithm_names: list[str] = Field(min_length=1)
    task_ids: dict[str, str] = Field(
        description="Mapping of algorithm_name → task_id for already-completed tasks"
    )
    tolerance_seconds: float = Field(default=3.0, gt=0, le=30)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/batch")
async def start_batch_eval(req: BatchEvalRequest):
    """Start a background SALAMI batch evaluation job. Returns job_id."""
    job_id = str(uuid.uuid4())
    algorithms = [canonical_algorithm_name(a) for a in req.algorithms]
    tolerances = req.tolerances or [req.tolerance_seconds, 3.0]
    _batch_jobs[job_id] = {
        "lines": [],
        "done": False,
        "error": None,
        "summary": None,
        "rows": [],
        "algorithms": algorithms,
        "tolerances": tolerances,
    }

    def _create_record():
        db = SessionLocal()
        try:
            db.add(BatchEvalJob(
                job_id=job_id,
                status="running",
                max_tracks=req.max_tracks,
                tolerance_seconds=req.tolerance_seconds,
                concurrency=req.concurrency,
                rows=[],
            ))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    await run_in_threadpool(_create_record)
    asyncio.create_task(
        _run_batch_eval_async(
            job_id,
            req.max_tracks,
            req.tolerance_seconds,
            req.concurrency,
            algorithms,
            tolerances,
            req.coverage_outlier_threshold,
        )
    )
    return {"job_id": job_id}


@router.get("/batch/history")
async def list_batch_eval_history(limit: int = Query(default=30, ge=1, le=100)):
    """Return list of past batch evaluation jobs, newest first."""
    def _get():
        db = SessionLocal()
        try:
            jobs = (
                db.query(BatchEvalJob)
                .order_by(BatchEvalJob.started_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "job_id":           j.job_id,
                    "status":           j.status,
                    "max_tracks":       j.max_tracks,
                    "tolerance_seconds": j.tolerance_seconds,
                    "concurrency":      j.concurrency,
                    "started_at":       j.started_at.isoformat() if j.started_at else None,
                    "completed_at":     j.completed_at.isoformat() if j.completed_at else None,
                    "tracks_ok":        j.tracks_ok,
                    "tracks_total":     j.tracks_total,
                    "avg_precision":    j.avg_precision,
                    "avg_recall":       j.avg_recall,
                    "avg_f1":           j.avg_f1,
                    "summary":          j.summary,
                    "rows":             j.rows or [],
                    "error":            j.error,
                }
                for j in jobs
            ]
        finally:
            db.close()
    return await run_in_threadpool(_get)


@router.get("/batch/{job_id}/stream")
async def stream_batch_eval(job_id: str):
    """SSE stream: yields progress lines, then a final done event with summary."""
    if job_id not in _batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")

    async def generate():
        cursor = 0
        while True:
            job   = _batch_jobs.get(job_id, {})
            lines = job.get("lines", [])
            while cursor < len(lines):
                yield f"data: {json.dumps({'line': lines[cursor]})}\n\n"
                cursor += 1
            if job.get("done"):
                yield f"data: {json.dumps({'done': True, 'summary': job.get('summary'), 'rows': job.get('rows', []), 'error': job.get('error')})}\n\n"
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/batch/{job_id}/result")
async def get_batch_result(job_id: str):
    """Return the final result of a completed batch eval job (memory then DB fallback)."""
    job = _batch_jobs.get(job_id)
    if job:
        if not job["done"]:
            raise HTTPException(status_code=202, detail="Job still running")
        return {"job_id": job_id, "summary": job.get("summary"), "rows": job.get("rows", []), "error": job.get("error")}

    def _from_db():
        db = SessionLocal()
        try:
            j = db.query(BatchEvalJob).filter(BatchEvalJob.job_id == job_id).first()
            return j
        finally:
            db.close()

    record = await run_in_threadpool(_from_db)
    if not record:
        raise HTTPException(status_code=404, detail="Batch job not found")
    if record.status == "running":
        raise HTTPException(status_code=202, detail="Job still running")
    return {
        "job_id":  record.job_id,
        "summary": record.summary,
        "rows":    record.rows or [],
        "error":   record.error,
    }


@router.post("/run")
async def run_evaluation(req: EvaluationRunRequest):
    """
    Evaluate a completed segmentation task against a track's ground truth.

    Fetches the task's results from DB, the track's ground truth from DB,
    computes boundary detection metrics per algorithm, and stores EvaluationRun
    records.

    Returns metrics for all algorithms present in the task results.
    """

    def _evaluate():
        db = SessionLocal()
        try:
            # Load task
            task = db.query(SegmentationTask).filter(SegmentationTask.task_id == req.task_id).first()
            if not task:
                raise ValueError(f"Task '{req.task_id}' not found")
            if task.status not in ("completed", "processing"):
                raise ValueError(f"Task status is '{task.status}' — need completed results")

            # Load track ground truth
            track = db.query(DatasetTrack).filter(DatasetTrack.track_id == req.track_id).first()
            if not track:
                raise ValueError(f"Track '{req.track_id}' not found")
            if not track.has_ground_truth or not track.ground_truth:
                raise ValueError("Track has no ground truth annotations")

            ref_segments = track.ground_truth
            results = task.results or {}

            if not results:
                raise ValueError("Task has no results yet")

            eval_results = {}
            for algo_name, raw_result in results.items():
                if "__" in str(algo_name):
                    continue
                segments = extract_segments(raw_result)
                if not segments:
                    continue

                metrics = compute_boundary_metrics(
                    ref_segments=ref_segments,
                    est_segments=segments,
                    tolerance=req.tolerance_seconds,
                )
                metrics.update(compute_boundary_metrics_multi(ref_segments, segments))

                # Store evaluation run
                eval_run = EvaluationRun(
                    eval_id=str(uuid.uuid4()),
                    algorithm_name=algo_name,
                    track_id=req.track_id,
                    task_id=req.task_id,
                    tolerance_seconds=req.tolerance_seconds,
                    metrics=metrics,
                )
                db.add(eval_run)
                eval_results[algo_name] = {
                    "eval_id": eval_run.eval_id,
                    "metrics": metrics,
                }

            db.commit()
            return {
                "task_id": req.task_id,
                "track_id": req.track_id,
                "tolerance_seconds": req.tolerance_seconds,
                "results": eval_results,
            }
        except ValueError:
            raise
        except Exception:
            db.rollback()
            logger.error("Evaluation run failed", exc_info=True)
            raise
        finally:
            db.close()

    try:
        return await run_in_threadpool(_evaluate)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_algorithms(req: CompareRequest):
    """
    Compare multiple algorithms on a single track using already-completed tasks.

    Accepts a mapping of algorithm_name → task_id. Each task must be completed.
    Computes boundary metrics for each algorithm and returns a side-by-side comparison.

    Use POST /segmentation/upload first to get task_ids, then call this endpoint
    once all tasks are completed.
    """

    def _compare():
        db = SessionLocal()
        try:
            # Load track ground truth
            track = db.query(DatasetTrack).filter(DatasetTrack.track_id == req.track_id).first()
            if not track:
                raise ValueError(f"Track '{req.track_id}' not found")
            if not track.has_ground_truth or not track.ground_truth:
                raise ValueError("Track has no ground truth annotations")

            ref_segments = track.ground_truth
            comparison = {}

            for algo_name, task_id in req.task_ids.items():
                task = db.query(SegmentationTask).filter(SegmentationTask.task_id == task_id).first()
                if not task:
                    comparison[algo_name] = {"error": f"Task '{task_id}' not found"}
                    continue
                if task.status != "completed":
                    comparison[algo_name] = {"error": f"Task status is '{task.status}'", "task_id": task_id}
                    continue

                results = task.results or {}
                canonical_algo = canonical_algorithm_name(algo_name)
                segments = extract_segments(results.get(canonical_algo) or results.get(algo_name))
                if not segments:
                    for key, value in results.items():
                        if "__" not in str(key):
                            segments = extract_segments(value)
                            if segments:
                                break

                if not segments:
                    comparison[algo_name] = {"error": "No segments in task results"}
                    continue

                metrics = compute_boundary_metrics(
                    ref_segments=ref_segments,
                    est_segments=segments,
                    tolerance=req.tolerance_seconds,
                )
                metrics.update(compute_boundary_metrics_multi(ref_segments, segments))

                # Store evaluation run
                eval_run = EvaluationRun(
                    eval_id=str(uuid.uuid4()),
                    algorithm_name=algo_name,
                    track_id=req.track_id,
                    task_id=task_id,
                    tolerance_seconds=req.tolerance_seconds,
                    metrics=metrics,
                )
                db.add(eval_run)
                comparison[algo_name] = {
                    "eval_id": eval_run.eval_id,
                    "task_id": task_id,
                    "segments": segments,
                    "metrics": metrics,
                }

            db.commit()
            return {
                "track_id": req.track_id,
                "tolerance_seconds": req.tolerance_seconds,
                "comparison": comparison,
            }
        except ValueError:
            raise
        except Exception:
            db.rollback()
            logger.error("Comparison failed", exc_info=True)
            raise
        finally:
            db.close()

    try:
        return await run_in_threadpool(_compare)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/track/{track_id}")
async def get_evaluations_for_track(
    track_id: str,
    tolerance: float | None = Query(default=None, gt=0, le=30),
):
    """
    Retrieve all stored evaluation runs for a track, grouped by algorithm.

    Optionally filter by tolerance window.
    """

    def _get():
        db = SessionLocal()
        try:
            q = db.query(EvaluationRun).filter(EvaluationRun.track_id == track_id)
            if tolerance is not None:
                q = q.filter(EvaluationRun.tolerance_seconds == tolerance)

            runs = q.order_by(EvaluationRun.created_at.desc()).all()

            grouped: dict[str, list] = {}
            for run in runs:
                entry = {
                    "eval_id": run.eval_id,
                    "task_id": run.task_id,
                    "tolerance_seconds": run.tolerance_seconds,
                    "metrics": run.metrics,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                }
                grouped.setdefault(run.algorithm_name, []).append(entry)

            return {"track_id": track_id, "evaluations": grouped}
        finally:
            db.close()

    return await run_in_threadpool(_get)


@router.get("/track/{track_id}/segmentations")
async def get_segmentations_for_track(track_id: str):
    """Return stored segmentation runs for a track, including raw segments and metrics.

    This is intended for the evaluation UI so users can inspect previous results.
    """

    def _get():
        db = SessionLocal()
        try:
            track = db.query(DatasetTrack).filter(DatasetTrack.track_id == track_id).first()
            if not track:
                raise ValueError(f"Track '{track_id}' not found")

            runs = (
                db.query(EvaluationRun, SegmentationTask)
                .outerjoin(SegmentationTask, EvaluationRun.task_id == SegmentationTask.task_id)
                .filter(EvaluationRun.track_id == track_id)
                .order_by(EvaluationRun.created_at.desc())
                .all()
            )

            items = []
            for eval_run, task in runs:
                results = task.results if task and task.results else {}
                segments = extract_segments(results.get(canonical_algorithm_name(eval_run.algorithm_name)))
                if not segments and results:
                    for key, value in results.items():
                        if "__" not in str(key):
                            segments = extract_segments(value)
                            if segments:
                                break

                items.append(
                    {
                        "eval_id": eval_run.eval_id,
                        "task_id": eval_run.task_id,
                        "algorithm_name": eval_run.algorithm_name,
                        "tolerance_seconds": eval_run.tolerance_seconds,
                        "metrics": eval_run.metrics,
                        "segments": segments or [],
                        "task_status": task.status if task else None,
                        "created_at": eval_run.created_at.isoformat() if eval_run.created_at else None,
                    }
                )

            return {"track_id": track_id, "segmentations": items}
        finally:
            db.close()

    try:
        return await run_in_threadpool(_get)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{eval_id}")
async def get_evaluation(eval_id: str):
    """Retrieve a single evaluation run by ID."""

    def _get():
        db = SessionLocal()
        try:
            run = db.query(EvaluationRun).filter(EvaluationRun.eval_id == eval_id).first()
            if not run:
                return None
            return {
                "eval_id": run.eval_id,
                "algorithm_name": run.algorithm_name,
                "track_id": run.track_id,
                "task_id": run.task_id,
                "tolerance_seconds": run.tolerance_seconds,
                "metrics": run.metrics,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
        finally:
            db.close()

    result = await run_in_threadpool(_get)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return result
