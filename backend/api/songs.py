from fastapi import APIRouter, HTTPException

from backend.services.segmentation_orchestrator import SegmentationOrchestrator
from shared.logger import get_logger

logger = get_logger()
router = APIRouter(prefix="/songs", tags=["Songs"])

orchestrator = SegmentationOrchestrator()


@router.get("")
def list_songs():
    try:
        songs = orchestrator.list_available_songs()
        return {"songs": [{"song_id": s.song_id, "blob_name": s.blob_name} for s in songs]}
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception:
        logger.error("Failed to list songs from storage", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list songs from storage")
