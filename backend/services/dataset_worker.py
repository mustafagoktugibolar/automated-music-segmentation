import os
import csv
import asyncio
from dataclasses import dataclass
from typing import List, Optional

from shared.logger import get_logger
from shared.blob_helper import AzureBlobCacheHelper

logger = get_logger()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
audio_metadata_path = os.path.join(
    BASE_DIR, "data", "salami", "metadata", "id_index_internetarchive.csv"
)

AZURE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER_RAW")
if not AZURE_CONTAINER:
    raise RuntimeError("AZURE_STORAGE_CONTAINER_RAW is not set")

DOWNLOAD_DIR = os.path.join(BASE_DIR, ".cache", "salami_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

helper = AzureBlobCacheHelper(
    cache_dir=os.path.join(BASE_DIR, ".cache", "salami_audio")
)

REQUIRED_COLUMNS = ("SONG_ID", "TITLE", "URL")


@dataclass
class AudioMetadata:
    song_id: str
    title: str
    archive_path: str

    blob_name: Optional[str] = None
    exists_in_blob: bool = False
    local_path: Optional[str] = None
    downloaded: bool = False
    uploaded: bool = False


def _safe_strip(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def _validate_csv_header(fieldnames: Optional[List[str]]) -> None:
    if not fieldnames:
        raise ValueError("CSV has no header (fieldnames is empty).")

    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}. Found: {fieldnames}")


def create_song_list() -> List[AudioMetadata]:
    song_list: List[AudioMetadata] = []
    skipped = 0

    try:
        with open(audio_metadata_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            _validate_csv_header(reader.fieldnames)

            for line_no, row in enumerate(reader, start=2):
                try:
                    song_id = _safe_strip(row.get("SONG_ID"))
                    title = _safe_strip(row.get("TITLE"))
                    url = _safe_strip(row.get("URL"))

                    if not song_id:
                        raise ValueError("SONG_ID is empty")
                    if not url:
                        raise ValueError("URL is empty")

                    song_list.append(AudioMetadata(song_id=song_id, title=title, archive_path=url))
                except Exception as e:
                    skipped += 1
                    logger.warning(f"Skipping invalid CSV row at line {line_no}: {e} | row={row}")

    except FileNotFoundError:
        logger.exception(f"Metadata file not found: {audio_metadata_path}")
        raise
    except UnicodeDecodeError:
        logger.exception(f"Encoding error while reading metadata file (expected utf-8): {audio_metadata_path}")
        raise
    except csv.Error:
        logger.exception(f"CSV parse error while reading: {audio_metadata_path}")
        raise
    except Exception:
        logger.exception("Failed to create song list")
        raise

    logger.info(f"Created song list with {len(song_list)} songs (skipped {skipped} invalid rows)")
    return song_list


def normalize_archive_paths(song_list: List[AudioMetadata]) -> None:
    normalized = 0
    skipped = 0

    for song in song_list:
        try:
            if not song.song_id:
                raise ValueError("song_id is empty")
            if not song.archive_path:
                raise ValueError("archive_path is empty")

            original = song.archive_path
            song.archive_path = (
                song.archive_path
                .replace("http://www.archive.org/download/", "http://archive.org/download/")
                .replace("https://www.archive.org/download/", "https://archive.org/download/")
                .replace("_vbr.mp3", ".mp3")
            )

            if original != song.archive_path:
                normalized += 1

        except Exception as e:
            skipped += 1
            logger.warning(f"Failed to normalize archive path for song_id={song.song_id!r}: {e}")

    logger.info(f"Normalized archive paths for {len(song_list)} songs (normalized {normalized}, skipped {skipped})")


def make_blob_name(song_id: str) -> str:
    if not song_id:
        raise ValueError("song_id is required for blob naming")
    return f"songs/{song_id}.mp3"


import httpx
import aiofiles
import uuid

async def _http_download(url: str, out_path: str, retries: int = 2) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Use unique temp file to avoid race conditions if duplicate IDs exist
    temp_path = f"{out_path}.{uuid.uuid4()}.tmp"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0, headers=headers) as client:
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 2):
            try:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(temp_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            await f.write(chunk)
                
                # Atomic move on success
                os.rename(temp_path, out_path)
                
                if os.path.getsize(out_path) == 0:
                    raise RuntimeError("Downloaded file is empty")
                
                return

            except Exception as e:
                last_err = e
                # Only log warnings for final failure or non-404/403 errors to reduce noise
                if "404" not in str(e) and "403" not in str(e):
                    logger.warning(f"Download attempt {attempt} failed for {url}: {e}")
                
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                
                if attempt < (retries + 1):
                    await asyncio.sleep(1.5 * attempt)

    # Re-raise the last error if it wasn't a standard 404 (to avoid crashing worker for expected missing files)
    if last_err and "404" in str(last_err):
         logger.warning(f"File not found on server: {url}")
         return # Treat 404 as "failed but handled"
    
    raise last_err if last_err else RuntimeError("Download failed")


async def process_one_song(song: AudioMetadata) -> None:
    try:
        # 1. Check if exists in Azure (Sync call wrapped in thread)
        blob_name = make_blob_name(song.song_id)
        exists = await asyncio.to_thread(helper.blob_exists, AZURE_CONTAINER, blob_name)
        if exists:
            # logger.info(f"Skipping {song.song_id}, already in Azure.")
            song.exists_in_blob = True
            return

        # 2. Download locally
        local_path = os.path.join(DOWNLOAD_DIR, f"{song.song_id}.mp3")
        await _http_download(song.archive_path, local_path, retries=3)
        song.local_path = local_path
        song.downloaded = True

        # 3. Upload to Azure (Sync call wrapped in thread)
        await asyncio.to_thread(
            helper.upload_file,
            local_path=local_path,
            container=AZURE_CONTAINER,
            blob_name=blob_name,
            overwrite=False,
            content_type="audio/mpeg"
        )
        song.uploaded = True
        song.exists_in_blob = True
        logger.info(f"Processed {song.song_id}: Downloaded & Uploaded")

        # 4. Cleanup local file to save space
        if os.path.exists(local_path):
            os.remove(local_path)

    except Exception as e:
        logger.warning(f"Failed to process song {song.song_id} ({song.archive_path}): {e}")
        # Clean up if download left a file
        local_part = os.path.join(DOWNLOAD_DIR, f"{song.song_id}.mp3")
        if os.path.exists(local_part):
            try:
                os.remove(local_part)
            except:
                pass

async def process_all_songs(song_list: List[AudioMetadata], concurrency: int = 5) -> None:
    logger.info(f"Starting concurrent processing for {len(song_list)} songs with concurrency={concurrency}...")
    sem = asyncio.Semaphore(concurrency)

    async def _worker(s):
        async with sem:
            await process_one_song(s)

    await asyncio.gather(*[_worker(s) for s in song_list])
    logger.info("All songs processed.")
