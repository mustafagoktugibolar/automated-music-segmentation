import io
import urllib.request
import aiofiles
import requests
from fastapi.responses import StreamingResponse
from typing import List, Optional
import os


from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.services.segmentation_orchestrator import SegmentationOrchestrator
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient

logger = get_logger()
router = APIRouter(prefix="/songs", tags=["Songs"])

orchestrator = SegmentationOrchestrator()


class SegmentBatchRequest(BaseModel):
    song_ids: List[str]
    algorithms: Optional[List[str]] = ["custom_librosa", "foote", "cnmf", "scluster"]


@router.get("")
async def list_songs():
    def do_rpc():
        rabbitmq = RabbitMQClient("songs_api")
        return rabbitmq.rpc_call("dataset.list_musics", {}, timeout=15)

    try:
        response = await run_in_threadpool(do_rpc)
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        return response
    except Exception as e:
        logger.error("Failed to list songs from dataset worker", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-batch")
async def segment_batch(req: SegmentBatchRequest):
    results = []

    def fetch_location(song_id: str):
        rabbitmq = RabbitMQClient("songs_api")
        return rabbitmq.rpc_call("dataset.get_music", {"song_id": song_id}, timeout=15)

    for song_id in req.song_ids:
        try:
            # 1. RPC to get_music
            response = await run_in_threadpool(fetch_location, song_id)
            if "error" in response:
                results.append({"song_id": song_id, "status": "error", "error": response["error"]})
                continue
                
            location = response.get("location")
            blob_name = response.get("blob_name")
            provider = response.get("storage_provider")

            if not location and not blob_name:
                results.append({"song_id": song_id, "status": "error", "error": "No location returned"})
                continue
            
            # 2. Extract into memory based on location type
            audio_bytes = None
            if provider == "local" and blob_name:
                local_path = f"/app/data/audio/{blob_name}"
                logger.info(f"Reading from local dataset cache: {local_path}")
                async with aiofiles.open(local_path, "rb") as f:
                    audio_bytes = await f.read()
            elif location and (location.startswith("http://") or location.startswith("https://")):
                def download_url(url):
                    req_obj = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req_obj, timeout=60) as resp:
                        return resp.read()
                audio_bytes = await run_in_threadpool(download_url, location)
            else:
                async with aiofiles.open(location, "rb") as f:
                    audio_bytes = await f.read()

            # 3. Create a pseudo UploadFile
            file_obj = io.BytesIO(audio_bytes)
            upload_file = UploadFile(filename=f"{song_id}.mp3", file=file_obj)
            upload_file.content_type = "audio/mpeg"
            
            # 4. Process the "upload" internally
            task_id = await orchestrator.process_upload(file=upload_file, requested_algos=req.algorithms)
            results.append({"song_id": song_id, "status": "processing", "task_id": task_id})
            
        except Exception as e:
            logger.error(f"Failed to segment batch for song {song_id}", exc_info=True)
            results.append({"song_id": song_id, "status": "error", "error": str(e)})

    return {"results": results}


@router.get("/stream/{song_id}")
async def stream_song(song_id: str):
    # This endpoint only supports streaming from S3/MinIO. No external URL or local fallbacks.
    from shared.storage.object_store import get_client_and_bucket
    s3_client, s3_bucket = get_client_and_bucket()
    prefix = os.getenv("DATASET_PREFIX", "").strip().strip("/")

    if s3_client is None:
        raise HTTPException(status_code=500, detail="S3/MinIO storage is not configured for streaming")

    try:
        candidate_keys = [f"songs/{song_id}.mp3", f"{song_id}.mp3"]
        if prefix:
            candidate_keys.append(f"{prefix}/songs/{song_id}.mp3")

        found_key = None
        for key in candidate_keys:
            try:
                s3_client.head_object(Bucket=s3_bucket, Key=key)
                found_key = key
                logger.info(f"Found object in S3 for song {song_id}: {key}")
                break
            except Exception:
                continue

        if not found_key:
            raise HTTPException(status_code=404, detail="Song not found in MinIO/S3 storage")

        def s3_iter():
            resp = s3_client.get_object(Bucket=s3_bucket, Key=found_key)
            body = resp["Body"]
            chunk = body.read(65536)
            while chunk:
                yield chunk
                chunk = body.read(65536)

        try:
            head = s3_client.head_object(Bucket=s3_bucket, Key=found_key)
            content_type = head.get("ContentType")
        except Exception:
            content_type = "audio/mpeg"

        return StreamingResponse(s3_iter(), media_type=content_type or "audio/mpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stream song {song_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error while streaming song")


@router.get("/debug/{song_id}")
async def debug_song_rpc(song_id: str):
    """Return dataset RPC info for a song_id for debugging (blob_name, storage_provider, download_url)."""
    def fetch_location(sid: str):
        rabbitmq = RabbitMQClient("songs_api")
        return rabbitmq.rpc_call("dataset.get_music", {"song_id": sid}, timeout=15)

    try:
        response = await run_in_threadpool(fetch_location, song_id)
        if isinstance(response, dict) and "error" in response:
            raise HTTPException(status_code=404, detail=response["error"])
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch RPC info for song {song_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch RPC info")
