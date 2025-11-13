import librosa
import numpy as np
from fastapi import UploadFile
import io
from backend.core import logger

import librosa
import numpy as np
from fastapi import UploadFile
import io
from backend.core import logger

# --- Tunable Parameters ---
# Hop length for Chroma CQT. Affects the time resolution of the features.
CQT_HOP_LENGTH = 512


def _load_audio_from_bytes(content: bytes, sr: int = None) -> tuple[np.ndarray, int]:
    """Loads an audio waveform from an in-memory byte buffer."""
    try:
        audio_stream = io.BytesIO(content)
        y, sr = librosa.load(audio_stream, sr=sr)
        return y, sr
    except Exception as e:
        logger.error("Failed to load audio from bytes.", exception=e)
        raise


def _extract_chroma_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extracts Chroma CQT features from an audio waveform."""
    # Separate harmonic component for more stable chroma
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # Compute Chroma features
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr, 
        hop_length=CQT_HOP_LENGTH
    )
    return chroma


async def analyze_and_segment_audio(file: UploadFile):
    """
    Orchestrates the audio segmentation process.
    This is the main entry point for the segmentation service.
    """
    logger.info(f"Starting analysis for: {file.filename}")
    
    try:
        # Step 1: Load audio from uploaded file
        content = await file.read()
        y, sr = _load_audio_from_bytes(content)
        
        # Step 2: Extract features
        chroma_features = _extract_chroma_features(y, sr)
        
        # Step 3: (Future) Compute self-similarity matrix
        
        # Step 4: (Future) Find boundaries
        
        # Step 5: (Future) Cluster and label segments
        
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Successfully processed {file.filename}, duration: {duration:.2f}s")
        
        # Return dummy results for now
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "duration_seconds": round(duration, 2),
            "feature_shape": chroma_features.shape,
            "status": "Analysis successful (dummy output)"
        }
    except Exception as e:
        logger.error(f"Error processing file {file.filename}", exception=e)
        # Re-raise the exception to be caught by the API layer
        raise e

