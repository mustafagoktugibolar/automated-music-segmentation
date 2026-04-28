"""
Evaluation API.

Provides endpoints to:
- Run boundary detection evaluation for a completed segmentation task against a track's ground truth
- Compare multiple algorithms on the same track
- Retrieve stored evaluation results
"""

import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.db.models import DatasetTrack, EvaluationRun, SegmentationTask
from backend.db.postgreSQL import SessionLocal
from backend.services.evaluation_service import compute_boundary_metrics
from shared.logger import get_logger

logger = get_logger()
router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


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
            for algo_name, segments in results.items():
                if not segments:
                    continue

                metrics = compute_boundary_metrics(
                    ref_segments=ref_segments,
                    est_segments=segments,
                    tolerance=req.tolerance_seconds,
                )

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

    Use POST /algorithms/{id}/test or POST /segmentation/upload first to get task_ids,
    then call this endpoint once all tasks are completed.
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

                # For user code tasks the result key is the algorithm_id; for built-in it's the algo name
                results = task.results or {}
                # Try to find segments: first by algo_name, then by first key
                segments = results.get(algo_name) or (list(results.values())[0] if results else None)

                if not segments:
                    comparison[algo_name] = {"error": "No segments in task results"}
                    continue

                metrics = compute_boundary_metrics(
                    ref_segments=ref_segments,
                    est_segments=segments,
                    tolerance=req.tolerance_seconds,
                )

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
                segments = results.get(eval_run.algorithm_name)
                if not segments and results:
                    segments = list(results.values())[0]

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
