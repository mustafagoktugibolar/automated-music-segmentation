"""
Algorithm management API.

Provides endpoints to:
- Save, list, retrieve, and soft-delete user-created segmentation algorithms
- Test an algorithm by dispatching a segmentation task via the user_code worker
"""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.db.models import Algorithm, DatasetTrack, SegmentationTask
from backend.db.postgreSQL import SessionLocal
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient

logger = get_logger()
router = APIRouter(prefix="/algorithms", tags=["Algorithms"])

UPLOAD_DIR_UPLOAD = "media/uploads"

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AlgorithmCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    code: str = Field(min_length=1)
    params_schema: dict | None = None


class AlgorithmTestRequest(BaseModel):
    audio_source: dict = Field(
        description=(
            "One of: "
            '{"type": "track_id", "value": "<track_id>"} or '
            '{"type": "salami", "value": "<song_id>"}'
        )
    )
    params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_syntax(code: str) -> str | None:
    """
    Return an error message if the code has a syntax error, else None.
    Also checks that a `segment` function is defined.
    """
    try:
        tree = compile(code, "<algorithm>", "exec")
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"

    import ast
    try:
        parsed = ast.parse(code)
    except Exception:
        return "Failed to parse code"

    func_names = {
        node.name
        for node in ast.walk(parsed)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "segment" not in func_names:
        return "Code must define a function named 'segment(audio_path, sr=22050, **params)'"

    return None


def _get_next_version(db, name: str) -> int:
    """Return the next version number for a given algorithm name."""
    from sqlalchemy import func as sa_func
    max_ver = (
        db.query(sa_func.max(Algorithm.version))
        .filter(Algorithm.name == name)
        .scalar()
    )
    return (max_ver or 0) + 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("")
async def create_algorithm(req: AlgorithmCreateRequest):
    """
    Save a new algorithm version.

    Validates syntax and checks that a `segment()` function is defined before
    persisting. Each save increments the version number for the given name.
    """
    syntax_error = _validate_syntax(req.code)
    if syntax_error:
        raise HTTPException(status_code=422, detail=f"Code validation failed: {syntax_error}")

    def _save():
        db = SessionLocal()
        try:
            version = _get_next_version(db, req.name)
            algo = Algorithm(
                algorithm_id=str(uuid.uuid4()),
                name=req.name,
                description=req.description,
                code=req.code,
                version=version,
                params_schema=req.params_schema,
                is_active=True,
            )
            db.add(algo)
            db.commit()
            db.refresh(algo)
            return {
                "algorithm_id": algo.algorithm_id,
                "name": algo.name,
                "version": algo.version,
                "description": algo.description,
                "created_at": algo.created_at.isoformat() if algo.created_at else None,
            }
        except Exception:
            db.rollback()
            logger.error("Failed to save algorithm", exc_info=True)
            raise
        finally:
            db.close()

    try:
        return await run_in_threadpool(_save)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_algorithms():
    """
    List all active algorithms (latest version per name).
    """

    def _list():
        db = SessionLocal()
        try:
            from sqlalchemy import func as sa_func

            # Subquery: max version per name among active algorithms
            subq = (
                db.query(Algorithm.name, sa_func.max(Algorithm.version).label("max_ver"))
                .filter(Algorithm.is_active == True)
                .group_by(Algorithm.name)
                .subquery()
            )
            algos = (
                db.query(Algorithm)
                .join(subq, (Algorithm.name == subq.c.name) & (Algorithm.version == subq.c.max_ver))
                .filter(Algorithm.is_active == True)
                .order_by(Algorithm.created_at.desc())
                .all()
            )
            return [
                {
                    "algorithm_id": a.algorithm_id,
                    "name": a.name,
                    "version": a.version,
                    "description": a.description,
                    "has_params_schema": a.params_schema is not None,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in algos
            ]
        finally:
            db.close()

    return await run_in_threadpool(_list)


@router.get("/{algorithm_id}")
async def get_algorithm(algorithm_id: str):
    """Get a single algorithm record including the full code."""

    def _get():
        db = SessionLocal()
        try:
            algo = db.query(Algorithm).filter(Algorithm.algorithm_id == algorithm_id).first()
            if not algo:
                return None
            return {
                "algorithm_id": algo.algorithm_id,
                "name": algo.name,
                "version": algo.version,
                "description": algo.description,
                "code": algo.code,
                "params_schema": algo.params_schema,
                "is_active": algo.is_active,
                "created_at": algo.created_at.isoformat() if algo.created_at else None,
                "updated_at": algo.updated_at.isoformat() if algo.updated_at else None,
            }
        finally:
            db.close()

    result = await run_in_threadpool(_get)
    if result is None:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return result


@router.get("/{name}/versions")
async def list_algorithm_versions(name: str):
    """List all versions of an algorithm by name."""

    def _list():
        db = SessionLocal()
        try:
            algos = (
                db.query(Algorithm)
                .filter(Algorithm.name == name)
                .order_by(Algorithm.version.desc())
                .all()
            )
            return [
                {
                    "algorithm_id": a.algorithm_id,
                    "name": a.name,
                    "version": a.version,
                    "description": a.description,
                    "is_active": a.is_active,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in algos
            ]
        finally:
            db.close()

    return await run_in_threadpool(_list)


@router.delete("/{algorithm_id}")
async def delete_algorithm(algorithm_id: str):
    """Soft-delete an algorithm (sets is_active=False)."""

    def _delete():
        db = SessionLocal()
        try:
            algo = db.query(Algorithm).filter(Algorithm.algorithm_id == algorithm_id).first()
            if not algo:
                return False
            algo.is_active = False
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    deleted = await run_in_threadpool(_delete)
    if not deleted:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return {"algorithm_id": algorithm_id, "deleted": True}


@router.post("/{algorithm_id}/test")
async def test_algorithm(algorithm_id: str, req: AlgorithmTestRequest):
    """
    Dispatch a test run for an algorithm via the user_code worker.

    Audio source types:
      - "track_id": uses a DatasetTrack's audio_url
      - "salami": uses a SALAMI song's Internet Archive URL (via song_id)

    Returns a task_id. Connect to GET /segmentation/stream/{task_id} for real-time results.
    """

    def _load_algo_and_dispatch():
        db = SessionLocal()
        try:
            algo = db.query(Algorithm).filter(
                Algorithm.algorithm_id == algorithm_id,
                Algorithm.is_active == True,
            ).first()
            if not algo:
                raise ValueError(f"Algorithm '{algorithm_id}' not found")

            # Resolve audio source
            source_type = req.audio_source.get("type", "")
            source_value = req.audio_source.get("value", "")
            audio_url = None
            file_path = None

            if source_type == "track_id":
                track = db.query(DatasetTrack).filter(DatasetTrack.track_id == source_value).first()
                if not track:
                    raise ValueError(f"Track '{source_value}' not found")
                if track.audio_url:
                    audio_url = track.audio_url
                elif track.audio_blob_name:
                    # Download from blob to temp path
                    from shared.blob_helper import AzureBlobCacheHelper
                    import os as _os
                    container = _os.getenv("AZURE_STORAGE_CONTAINER_RAW", "").strip()
                    if container:
                        helper = AzureBlobCacheHelper()
                        file_path = helper.download_to_cache(container, track.audio_blob_name)
                    else:
                        raise ValueError("No audio URL or blob storage configured for this track")
                else:
                    raise ValueError("Track has no audio source")

            elif source_type == "salami":
                from backend.services.dataset_worker import create_song_list
                songs = {s.song_id: s for s in create_song_list()}
                song = songs.get(str(source_value))
                if not song:
                    raise ValueError(f"SALAMI song '{source_value}' not found in metadata")
                audio_url = song.archive_path

            else:
                raise ValueError(f"Unknown audio_source type: '{source_type}'")

            # Create a SegmentationTask record for SSE tracking
            task_id = str(uuid.uuid4())
            new_task = SegmentationTask(
                task_id=task_id,
                filename=f"algo_test_{algorithm_id}",
                status="processing",
                results={},
                expected_algorithms=[algorithm_id],
                source_type="algo_test",
                source_song_id=source_value if source_type in ("track_id", "salami") else None,
            )
            db.add(new_task)
            db.commit()

            # Build worker payload
            if audio_url:
                payload = {
                    "task_id": task_id,
                    "algorithm_id": algorithm_id,
                    "algorithm_code": algo.code,
                    "algorithm_params": req.params,
                    "source_type": "upload_url",
                    "audio_url": audio_url,
                    # The worker will need to download this URL first.
                    # We reuse the upload source_type trick: set file_path to None
                    # and let the worker download. For now, we set source_type="upload_url"
                    # and the worker handles it.
                    "file_path": None,
                }
            else:
                payload = {
                    "task_id": task_id,
                    "algorithm_id": algorithm_id,
                    "algorithm_code": algo.code,
                    "algorithm_params": req.params,
                    "source_type": "upload",
                    "file_path": file_path,
                }

            # Publish to user_code worker
            rabbitmq = RabbitMQClient("algorithms_api")
            rabbitmq.publish(
                exchange="segmentation_topic",
                routing_key="segmentation.user_code",
                message=payload,
            )
            rabbitmq.close()

            return {"task_id": task_id, "algorithm_id": algorithm_id, "status": "processing"}

        except ValueError:
            raise
        except Exception:
            db.rollback()
            logger.error("Failed to dispatch algorithm test", exc_info=True)
            raise
        finally:
            db.close()

    try:
        return await run_in_threadpool(_load_algo_and_dispatch)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
