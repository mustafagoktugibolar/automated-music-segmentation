import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient

logger = get_logger()

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

# Define upload directory (mapped to shared volume)
UPLOAD_DIR = "/app/media/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize RabbitMQ Client
rabbitmq_client = RabbitMQClient(service_name="api-orchestrator")

@router.post("/upload", summary="Upload and distribute segmentation tasks")
async def upload_and_segment_audio(
    file: UploadFile = File(..., description="Audio file (e.g., WAV, MP3)"),
    algorithms: str = Form(default='["custom", "foote", "cnmf", "scluster"]', description='JSON list of algorithms to run. Options: "custom", "foote", "cnmf", "scluster"')
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
        
    # Parse requested algorithms
    import json
    try:
        requested_algos = json.loads(algorithms)
        if not isinstance(requested_algos, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="algorithms field must be a valid JSON list of strings")

    # Map algorithms to routing keys
    ALGO_TO_ROUTING_KEY = {
        "custom": "segmentation.custom",
        "foote": "segmentation.foote",
        "cnmf": "segmentation.cnmf",
        "scluster": "segmentation.scluster"
    }

    target_keys = []
    for algo in requested_algos:
        key = ALGO_TO_ROUTING_KEY.get(algo.lower())
        if key:
            target_keys.append(key)
        else:
            logger.warning(f"Unknown algorithm requested: {algo}")

    if not target_keys:
        raise HTTPException(status_code=400, detail="No valid algorithms specified.")

    task_id = str(uuid.uuid4())
    filename = f"{task_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        # 1. Save file to shared volume
        logger.info(f"Saving uploaded file to {file_path}")
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # Read in chunks
                await out_file.write(content)

        # 2. Construct the Task Payload
        task_payload = {
            "task_id": task_id,
            "original_filename": file.filename,
            "file_path": file_path,
            "content_type": file.content_type
        }

        # --- DB: Create Task Record ---
        try:
            from backend.db.models import SegmentationTask
            from backend.db.postgreSQL import SessionLocal
            
            db = SessionLocal()
            new_task = SegmentationTask(
                task_id=task_id, 
                filename=file.filename,
                status="processing",
                results={},
                expected_algorithms=requested_algos
            )
            db.add(new_task)
            db.commit()
            db.close()
        except Exception as db_e:
            logger.error("Failed to save initial task to DB", exc_info=True)
            # Proceed anyway or raise? Proceeding allows workers to run, but polling will fail.
            # Best to raise.
            raise HTTPException(status_code=500, detail="Database error during task creation")
        # -----------------------------

        # 3. Publish to RabbitMQ (Orchestrator Logic)
        logger.info(f"Distributing tasks for {task_id} to workers: {target_keys}")
        
        for key in target_keys:
            rabbitmq_client.publish(
                exchange="segmentation_topic",
                routing_key=key,
                message=task_payload
            )

        return {
            "message": "File uploaded and segmentation tasks dispatched.",
            "task_id": task_id,
            "status": "processing",
            "triggered_workers": requested_algos
        }

    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.get("/status/{task_id}", summary="Get segmentation status and results")
def get_task_status(task_id: str):
    from backend.db.models import SegmentationTask
    from backend.db.postgreSQL import SessionLocal
    
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
