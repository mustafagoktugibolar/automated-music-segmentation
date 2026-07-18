"""
Unified MinIO / S3 object storage adapter.

Single source of truth for all MinIO operations.  Previously duplicated in:
  - workers/batch_eval.py
  - backend/api/evaluation.py
  - backend/api/songs.py
  - scripts/label_training/prepare_label_dataset.py

All callers should import from here.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("object_store")

# Env-var config — evaluated at import time so changes to env before import
# are picked up, but the values are stable within a process lifetime.
_ENDPOINT = os.getenv("S3_ENDPOINT")
_KEY      = os.getenv("S3_ACCESS_KEY")
_SECRET   = os.getenv("S3_SECRET_KEY")
_BUCKET   = os.getenv("S3_BUCKET_RAW")
_PREFIX   = os.getenv("DATASET_PREFIX", "").strip().strip("/")


def is_available() -> bool:
    """Return True when the minimum MinIO config is present."""
    return bool(_KEY and _SECRET and _BUCKET)


def _client():
    """Create a boto3 S3 client pointed at the configured MinIO endpoint."""
    import boto3
    session = boto3.session.Session()
    return session.client(
        "s3",
        aws_access_key_id=_KEY,
        aws_secret_access_key=_SECRET,
        endpoint_url=_ENDPOINT or None,
    )


def list_song_ids() -> list[str]:
    """Return numeric song IDs whose .mp3 exists in the bucket.

    Scans all keys and returns those whose filename is a pure integer followed
    by `.mp3` — these correspond to SALAMI numeric IDs.
    """
    if not is_available():
        logger.warning("MinIO not configured — S3_BUCKET_RAW / S3_ACCESS_KEY missing.")
        return []
    client = _client()
    try:
        paginator = client.get_paginator("list_objects_v2")
        song_ids: set[str] = set()
        for page in paginator.paginate(Bucket=_BUCKET):
            for obj in page.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name.endswith(".mp3"):
                    sid = name[:-4]
                    if sid.isdigit():
                        song_ids.add(sid)
        return sorted(song_ids, key=int)
    except Exception as exc:
        logger.error("MinIO list failed: %s", exc)
        return []


def download(song_id: str) -> Optional[bytes]:
    """Download audio bytes for *song_id* from MinIO.

    Tries candidate key patterns in order:
      ``songs/<id>.mp3``  (primary)
      ``<id>.mp3``        (flat layout)
      ``<prefix>/songs/<id>.mp3``  (when DATASET_PREFIX is set)

    Returns None if the object is not found or MinIO is unreachable.
    """
    if not is_available():
        return None
    client = _client()
    candidates = [f"songs/{song_id}.mp3", f"{song_id}.mp3"]
    if _PREFIX:
        candidates.append(f"{_PREFIX}/songs/{song_id}.mp3")

    for key in candidates:
        try:
            resp = client.get_object(Bucket=_BUCKET, Key=key)
            data = resp["Body"].read()
            logger.debug("Downloaded %s bytes for song %s (key=%s)", len(data), song_id, key)
            return data
        except Exception:
            continue
    logger.warning("Audio not found in MinIO for song_id=%s", song_id)
    return None


def get_client_and_bucket():
    """Return *(boto3 S3 client, bucket_name)* for callers that need direct access.

    Returns *(None, None)* when MinIO is not configured.
    """
    if not is_available():
        return None, None
    return _client(), _BUCKET


def upload(song_id: str, data: bytes, content_type: str = "audio/mpeg") -> bool:
    """Upload *data* under ``songs/<song_id>.mp3``.  Returns True on success."""
    if not is_available():
        return False
    client = _client()
    key = f"songs/{song_id}.mp3"
    try:
        client.put_object(
            Bucket=_BUCKET, Key=key, Body=data, ContentType=content_type
        )
        logger.info("Uploaded %d bytes → %s/%s", len(data), _BUCKET, key)
        return True
    except Exception as exc:
        logger.error("MinIO upload failed for %s: %s", key, exc)
        return False


# ---------------------------------------------------------------------------
# Backward-compat aliases used by workers/batch_eval.py and scripts/
# ---------------------------------------------------------------------------

def list_minio_song_ids() -> list[str]:
    return list_song_ids()


def download_from_minio(song_id: str) -> Optional[bytes]:
    return download(song_id)
