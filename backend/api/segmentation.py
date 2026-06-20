import json
import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from backend.api.schemas import SegmentationParams, StorageSegmentationRequest
from backend.db.models import SegmentationTask
from backend.db.postgreSQL import SessionLocal
from backend.services.segmentation_orchestrator import SegmentationOrchestrator
from backend.services.result_listener import register_sse_callback, unregister_sse_callback
from shared.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

orchestrator = SegmentationOrchestrator()


@router.get("/stream/{task_id}")
async def stream_task_status(task_id: str) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(data):
        # RabbitMQ consumer runs in a background thread; asyncio.Queue is not
        # thread-safe, so we must schedule the put via the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, data)

    # Register callback before the DB read to avoid missing a result that
    # arrives between the two. If the task is already terminal, pre-fill the
    # queue so the generator yields immediately without waiting for an event.
    register_sse_callback(task_id, callback)

    db = SessionLocal()
    try:
        task = db.query(SegmentationTask).filter(SegmentationTask.task_id == task_id).first()
        if task and task.status in ("completed", "failed"):
            queue.put_nowait({
                "task_id": task.task_id,
                "status": task.status,
                "filename": task.filename,
                "results": task.results,
            })
    finally:
        db.close()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("status") in ("completed", "failed"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'task_id': task_id, 'status': 'alive'})}\n\n"
        finally:
            unregister_sse_callback(task_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _parse_algorithms_json(raw_algorithms: str) -> list[str]:
    try:
        parsed = json.loads(raw_algorithms)
        if not isinstance(parsed, list) or not all(isinstance(a, str) for a in parsed):
            raise ValueError
        return parsed
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="algorithms field must be a valid JSON list of strings")


def _parse_params_json(raw_params: str | None) -> SegmentationParams | None:
    if raw_params is None or not raw_params.strip():
        return None

    try:
        data = json.loads(raw_params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="params field must be valid JSON")

    try:
        return SegmentationParams.model_validate(data)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())


@router.post("/upload")
async def upload_and_segment_audio(
    file: UploadFile = File(..., description="Audio file (e.g., WAV, MP3)"),
    algorithms: str = Form(default='["custom_librosa", "foote", "cnmf", "scluster"]'),
    params: str | None = Form(default=None, description="Optional JSON object of typed segmentation params"),
    webhook_url: str | None = Form(default=None, description="Optional webhook URL to call when task completes"),
):
    valid_extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    content_type = file.content_type or ""
    is_audio_mime = content_type.startswith("audio/")
    is_valid_ext = file.filename.lower().endswith(valid_extensions)

    if not is_audio_mime and not is_valid_ext:
        logger.warning(f"Invalid file received. Content-Type: {content_type}, Filename: {file.filename}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type ({content_type}). Please upload an audio file ({valid_extensions}).",
        )

    requested_algos = _parse_algorithms_json(algorithms)
    parsed_params = _parse_params_json(params)

    try:
        task_id = await orchestrator.process_upload(file, requested_algos, parsed_params, webhook_url)
        return {
            "message": "File uploaded and segmentation tasks dispatched.",
            "task_id": task_id,
            "status": "processing",
            "triggered_workers": requested_algos,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception:
        logger.error("Internal error during upload orchestration", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred")


@router.post("/from-storage")
async def segment_from_storage(req: StorageSegmentationRequest):
    try:
        task_id = await orchestrator.process_from_storage(req.song_id, req.algorithms, req.params)
        return {
            "message": "Storage song segmentation tasks dispatched.",
            "task_id": task_id,
            "status": "processing",
            "triggered_workers": req.algorithms,
            "song_id": req.song_id,
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception:
        logger.error("Internal error during storage orchestration", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred")


@router.get("/status/{task_id}")
def get_task_status(task_id: str):
    db = SessionLocal()
    try:
        task = db.query(SegmentationTask).filter(SegmentationTask.task_id == task_id).first()
    finally:
        db.close()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "filename": task.filename,
        "source_type": task.source_type,
        "source_song_id": task.source_song_id,
        "requested_params": task.requested_params,
        "created_at": task.created_at,
        "results": task.results,
    }
