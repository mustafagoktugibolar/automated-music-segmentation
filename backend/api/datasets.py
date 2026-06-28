"""
Dataset management API.

Provides endpoints to:
- Import SALAMI dataset annotations into the database
- List and browse datasets and their tracks
- Upload custom audio tracks with optional ground truth
"""

import csv
import io
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.db.models import Dataset, DatasetTrack
from backend.db.postgreSQL import SessionLocal
from backend.services.salami_parser import (
    ANNOTATIONS_DIR,
    parse_salami_annotation,
)
from shared.logger import get_logger

logger = get_logger()
router = APIRouter(prefix="/datasets", tags=["Datasets"])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SALAMI_METADATA_CSV = os.path.join(
    BASE_DIR, "data", "salami", "metadata", "id_index_internetarchive.csv"
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DatasetCreateRequest(BaseModel):
    name: str
    description: str | None = None
    source_type: str = "custom"


# ---------------------------------------------------------------------------
# Helper: parse custom ground truth CSV uploaded by user
# ---------------------------------------------------------------------------


def _parse_ground_truth_csv(raw_bytes: bytes) -> list[dict]:
    """
    Parse a CSV with columns start,end,label into [{start, end, label}, ...].
    Raises ValueError on format errors.
    """
    text = raw_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    required = {"start", "end", "label"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise ValueError(f"Ground truth CSV must have columns: {required}. Got: {reader.fieldnames}")

    segments = []
    for i, row in enumerate(reader, start=2):
        try:
            segments.append({
                "start": round(float(row["start"]), 3),
                "end": round(float(row["end"]), 3),
                "label": str(row["label"]).strip(),
            })
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid row at line {i}: {e}")

    return segments


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/import-salami")
async def import_salami():
    """
    Import all SALAMI tracks from the local metadata CSV + annotation files.

    Creates (or reuses) a 'SALAMI' dataset record and inserts dataset_tracks
    rows for every song found in the metadata CSV. Ground truth is parsed
    from the annotation files when available.

    This is idempotent: re-running merges by song_id (skips existing tracks).
    """

    def _do_import():
        if not os.path.exists(SALAMI_METADATA_CSV):
            raise FileNotFoundError(f"SALAMI metadata CSV not found at {SALAMI_METADATA_CSV}")

        db = SessionLocal()
        try:
            # Find or create the SALAMI dataset record
            dataset = db.query(Dataset).filter(Dataset.name == "SALAMI").first()
            if not dataset:
                dataset = Dataset(
                    dataset_id=str(uuid.uuid4()),
                    name="SALAMI",
                    description="Structural Analysis of Large Amounts of Music Information dataset",
                    source_type="salami",
                    track_count=0,
                )
                db.add(dataset)
                db.flush()

            dataset_id = dataset.dataset_id

            imported = 0
            updated = 0
            skipped = 0

            with open(SALAMI_METADATA_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    song_id = (row.get("SONG_ID") or "").strip()
                    title = (row.get("TITLE") or "").strip()
                    url = (row.get("URL") or "").strip()

                    if not song_id or not url:
                        skipped += 1
                        continue

                    # Try to parse ground truth annotation
                    ground_truth = parse_salami_annotation(song_id, annotator=1)
                    has_gt = ground_truth is not None

                    track = (
                        db.query(DatasetTrack)
                        .filter(DatasetTrack.dataset_id == dataset_id)
                        .filter(DatasetTrack.song_id == song_id)
                        .first()
                    )

                    if track:
                        track.title = title or song_id
                        track.audio_url = url
                        track.has_ground_truth = has_gt
                        track.ground_truth = ground_truth
                        updated += 1
                    else:
                        track = DatasetTrack(
                            track_id=str(uuid.uuid4()),
                            dataset_id=dataset_id,
                            song_id=song_id,
                            title=title or song_id,
                            audio_url=url,
                            has_ground_truth=has_gt,
                            ground_truth=ground_truth,
                        )
                        db.add(track)
                        imported += 1

            # Update denormalized count
            total = (
                db.query(DatasetTrack)
                .filter(DatasetTrack.dataset_id == dataset_id)
                .count()
            )
            dataset.track_count = total
            db.commit()

            return {
                "dataset_id": dataset_id,
                "tracks_imported": imported,
                "tracks_updated": updated,
                "tracks_skipped": skipped,
                "total_tracks": total,
            }
        except Exception:
            db.rollback()
            logger.error("SALAMI import failed", exc_info=True)
            raise
        finally:
            db.close()

    try:
        result = await run_in_threadpool(_do_import)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("SALAMI import endpoint error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_dataset(req: DatasetCreateRequest):
    """Create a new custom dataset."""

    def _create():
        db = SessionLocal()
        try:
            existing = db.query(Dataset).filter(Dataset.name == req.name).first()
            if existing:
                raise ValueError(f"Dataset with name '{req.name}' already exists")

            dataset = Dataset(
                dataset_id=str(uuid.uuid4()),
                name=req.name,
                description=req.description,
                source_type=req.source_type,
                track_count=0,
            )
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            return {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "source_type": dataset.source_type,
                "track_count": 0,
            }
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    try:
        return await run_in_threadpool(_create)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_datasets():
    """List all datasets with track counts."""

    def _list():
        db = SessionLocal()
        try:
            datasets = db.query(Dataset).order_by(Dataset.created_at).all()
            return [
                {
                    "dataset_id": d.dataset_id,
                    "name": d.name,
                    "description": d.description,
                    "source_type": d.source_type,
                    "track_count": d.track_count,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in datasets
            ]
        finally:
            db.close()

    return await run_in_threadpool(_list)


@router.get("/{dataset_id}/tracks")
async def list_tracks(
    dataset_id: str,
    has_ground_truth: bool | None = Query(default=None),
):
    """List all tracks within a dataset, with optional ground truth filter."""

    def _list():
        db = SessionLocal()
        try:
            q = db.query(DatasetTrack).filter(DatasetTrack.dataset_id == dataset_id)
            if has_ground_truth is not None:
                q = q.filter(DatasetTrack.has_ground_truth == has_ground_truth)

            tracks = q.all()

            return {
                "total": len(tracks),
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "song_id": t.song_id,
                        "title": t.title,
                        "artist": t.artist,
                        "audio_url": t.audio_url,
                        "duration_seconds": t.duration_seconds,
                        "has_ground_truth": t.has_ground_truth,
                        "created_at": t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in tracks
                ],
            }
        finally:
            db.close()

    def _check_dataset():
        db = SessionLocal()
        try:
            return db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        finally:
            db.close()

    exists = await run_in_threadpool(_check_dataset)
    if not exists:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return await run_in_threadpool(_list)


@router.get("/{dataset_id}/tracks/{track_id}")
async def get_track(dataset_id: str, track_id: str):
    """Get a single track including ground truth segments."""

    def _get():
        db = SessionLocal()
        try:
            track = (
                db.query(DatasetTrack)
                .filter(
                    DatasetTrack.track_id == track_id,
                    DatasetTrack.dataset_id == dataset_id,
                )
                .first()
            )
            if not track:
                return None
            return {
                "track_id": track.track_id,
                "dataset_id": track.dataset_id,
                "song_id": track.song_id,
                "title": track.title,
                "artist": track.artist,
                "audio_url": track.audio_url,
                "audio_blob_name": track.audio_blob_name,
                "duration_seconds": track.duration_seconds,
                "has_ground_truth": track.has_ground_truth,
                "ground_truth": track.ground_truth,
                "created_at": track.created_at.isoformat() if track.created_at else None,
            }
        finally:
            db.close()

    result = await run_in_threadpool(_get)
    if result is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return result


@router.post("/{dataset_id}/tracks/upload")
async def upload_track(
    dataset_id: str,
    title: str = Form(default=""),
    artist: str = Form(default=""),
    file: UploadFile = File(...),
    ground_truth_csv: UploadFile | None = File(default=None),
):
    """
    Upload an audio file as a new track in a custom dataset.
    Optionally attach a ground truth CSV (columns: start, end, label).
    """
    # Validate dataset existence
    def _check_dataset():
        db = SessionLocal()
        try:
            return db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        finally:
            db.close()

    dataset = await run_in_threadpool(_check_dataset)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.source_type != "custom":
        raise HTTPException(status_code=400, detail="Cannot upload tracks to a built-in dataset")

    # Parse ground truth if provided
    ground_truth = None
    has_ground_truth = False
    if ground_truth_csv is not None:
        gt_bytes = await ground_truth_csv.read()
        try:
            ground_truth = _parse_ground_truth_csv(gt_bytes)
            has_ground_truth = True
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid ground truth CSV: {e}")

    # Save audio to Azure Blob Storage
    audio_bytes = await file.read()
    track_id = str(uuid.uuid4())
    blob_name = f"user-datasets/{dataset_id}/{track_id}/{file.filename}"

    try:
        from shared.storage.blob_helper import AzureBlobCacheHelper
        helper = AzureBlobCacheHelper()
        container = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "").strip()

        if container:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            def _upload():
                helper.upload_file(container, blob_name, tmp_path, content_type=file.content_type)
                os.unlink(tmp_path)

            await run_in_threadpool(_upload)
        else:
            # No Azure configured — store bytes locally (dev mode)
            blob_name = None
            logger.warning("AZURE_STORAGE_CONTAINER_RAW not set; audio file not persisted to blob storage")

    except Exception:
        logger.error("Failed to upload audio to blob storage", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store audio file")

    def _save_track():
        db = SessionLocal()
        try:
            track = DatasetTrack(
                track_id=track_id,
                dataset_id=dataset_id,
                title=title or file.filename,
                artist=artist or None,
                audio_blob_name=blob_name,
                has_ground_truth=has_ground_truth,
                ground_truth=ground_truth,
            )
            db.add(track)

            # Update denormalized count
            ds = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if ds:
                ds.track_count = ds.track_count + 1

            db.commit()
            return track_id
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    try:
        saved_id = await run_in_threadpool(_save_track)
        return {"track_id": saved_id, "blob_name": blob_name, "has_ground_truth": has_ground_truth}
    except Exception as e:
        logger.error("Failed to save track record", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
