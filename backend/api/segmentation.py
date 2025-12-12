from fastapi import APIRouter, UploadFile, File, HTTPException
from shared.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

@router.post("/upload", summary="Upload and analyze a music file")
async def upload_and_segment_audio(
    file: UploadFile = File(..., description="Audio file (e.g., WAV, MP3)")
):
    if not file.content_type.startswith("audio/"):
        logger.warning(f"Invalid content type received: {file.content_type}")
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload an audio file."
        )
        
    try:
        #result = await segmentation_service.analyze_and_segment_audio(file)
        return "" 
    except Exception as e:
        logger.error(f"An error occurred during file processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred: {str(e)}"
        )
