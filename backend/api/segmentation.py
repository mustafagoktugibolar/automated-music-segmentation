import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from shared.logger import get_logger
from backend.db.models import SegmentationTask
from backend.db.postgreSQL import SessionLocal
from backend.services.segmentation_orchestrator import SegmentationOrchestrator

logger = get_logger()

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

orchestrator = SegmentationOrchestrator()

@router.post("/upload")
async def upload_and_segment_audio(
    file: UploadFile = File(..., description="Audio file (e.g., WAV, MP3)"),
    algorithms: str = Form(default='["custom", "foote", "cnmf", "scluster"]')
):
    valid_extensions = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    is_audio_mime = file.content_type.startswith("audio/")
    is_valid_ext = file.filename.lower().endswith(valid_extensions)

    if not is_audio_mime and not is_valid_ext:
        logger.warning(f"Invalid file received. Content-Type: {file.content_type}, Filename: {file.filename}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type ({file.content_type}). Please upload an audio file ({valid_extensions})."
        )
        
    try:
        requested_algos = json.loads(algorithms)
        if not isinstance(requested_algos, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="algorithms field must be a valid JSON list of strings")

    try:
        task_id = await orchestrator.process_upload(file, requested_algos)
        return {
            "message": "File uploaded and segmentation tasks dispatched.",
            "task_id": task_id,
            "status": "processing",
            "triggered_workers": requested_algos
        }

    except ValueError as ve:
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.get("/status/{task_id}")
def get_task_status(task_id: str):
    db = SessionLocal()
    task = db.query(SegmentationTask).filter(SegmentationTask.task_id == task_id).first()
    db.close()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "filename": task.filename,
        "created_at": task.created_at,
        "results": task.results
    }
