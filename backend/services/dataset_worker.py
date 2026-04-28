import os
import csv
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from shared.logger import get_logger
from shared.config import AppSettings

logger = get_logger()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
audio_metadata_path = os.path.join(
    BASE_DIR, "data", "salami", "metadata", "id_index_internetarchive.csv"
)

REQUIRED_COLUMNS = ("SONG_ID", "TITLE", "URL")


@dataclass
class AudioMetadata:
    song_id: str
    title: str
    archive_path: str


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


def get_available_songs() -> List[AudioMetadata]:
    """Returns a list of all available songs based on the configured datasource."""
    if AppSettings.DATASOURCE_TYPE == 'local':
        songs = []
        search_dir = os.path.join(BASE_DIR, AppSettings.LOCAL_AUDIO_DIR)
        
        if not os.path.exists(search_dir):
            logger.warning(f"Local audio directory not found: {search_dir}")
            return []
            
        for p in Path(search_dir).rglob("*.mp3"):
            song_id = p.stem
            songs.append(AudioMetadata(song_id=song_id, title=song_id, archive_path=str(p)))
            
        logger.info(f"Loaded {len(songs)} songs from local directory {search_dir}")
        return songs
        
    elif AppSettings.DATASOURCE_TYPE == 'salami':
        return create_song_list()
        
    else:
        logger.warning(f"Unknown DATASOURCE_TYPE: {AppSettings.DATASOURCE_TYPE}")
        return []
