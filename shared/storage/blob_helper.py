import os
import shutil

from dataclasses import dataclass
from typing import List, Optional

import portalocker
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError, AzureError

from shared.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class BlobRef:
    container: str
    blob_name: str


class AzureBlobCacheHelper:
    def __init__(self, connection_string: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        self._connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not self._connection_string:
            raise ValueError(
                "Azure connection string is missing. "
                "Set AZURE_STORAGE_CONNECTION_STRING or pass connection_string."
            )

        self._blob_service = BlobServiceClient.from_connection_string(self._connection_string)

        self._cache_dir = cache_dir or os.path.join(os.getcwd(), ".cache", "azure_blobs")
        os.makedirs(self._cache_dir, exist_ok=True)

    def _container_client(self, container: str):
        if not container or not container.strip():
            raise ValueError("container name is empty")
        return self._blob_service.get_container_client(container)

    def _blob_client(self, container: str, blob_name: str):
        if not blob_name or not blob_name.strip():
            raise ValueError("blob_name is empty")
        return self._container_client(container).get_blob_client(blob_name)

    def _cache_path_for(self, container: str, blob_name: str) -> str:
        safe_container = container.replace("/", "_").replace("\\", "_")
        safe_name = blob_name.replace("\\", "/").lstrip("/")
        safe_name = os.path.normpath(safe_name).replace("\\", "/")
        if safe_name.startswith("../") or safe_name in ("..", "."):
            raise ValueError("blob_name is not safe")
        return os.path.join(self._cache_dir, safe_container, safe_name)

    def upload_file(
        self,
        local_path: str,
        container: str,
        blob_name: Optional[str] = None,
        overwrite: bool = False,
        content_type: Optional[str] = None,
    ) -> BlobRef:
        if not local_path or not local_path.strip():
            raise ValueError("local_path is empty")
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        blob_name = blob_name or os.path.basename(local_path)
        blob_client = self._blob_client(container, blob_name)

        try:
            extra = {}
            if content_type:
                extra["content_settings"] = ContentSettings(content_type=content_type)

            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=overwrite, **extra)

            logger.info(f"Uploaded file to Azure Blob: container={container} blob={blob_name}")
            return BlobRef(container=container, blob_name=blob_name)

        except AzureError:
            logger.exception(
                f"Azure upload failed: container={container} blob={blob_name} local_path={local_path}"
            )
            raise

    def download_to_cache(self, container: str, blob_name: str, force: bool = False) -> str:
        cache_path = self._cache_path_for(container, blob_name)
        lock_path = cache_path + ".lock"

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            with portalocker.Lock(lock_path, timeout=120):
                if not force and os.path.isfile(cache_path):
                    logger.info(f"Cache hit: {cache_path}")
                    return cache_path

                blob_client = self._blob_client(container, blob_name)
                downloader = blob_client.download_blob()
                with open(cache_path, "wb") as f:
                    downloader.readinto(f)

            logger.info(f"Downloaded blob to cache: container={container} blob={blob_name} -> {cache_path}")
            return cache_path

        except ResourceNotFoundError:
            logger.exception(f"Blob not found: container={container} blob={blob_name}")
            raise
        except AzureError:
            logger.exception(f"Azure download failed: container={container} blob={blob_name}")
            raise

    def delete_blob(self, container: str, blob_name: str) -> bool:
        blob_client = self._blob_client(container, blob_name)
        try:
            blob_client.delete_blob()
            logger.info(f"Deleted blob: container={container} blob={blob_name}")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Delete skipped (not found): container={container} blob={blob_name}")
            return False
        except AzureError:
            logger.exception(f"Azure delete failed: container={container} blob={blob_name}")
            raise

    def list_blobs(self, container: str, prefix: Optional[str] = None) -> List[str]:
        container_client = self._container_client(container)
        try:
            blobs = container_client.list_blobs(name_starts_with=prefix)
            result = [b.name for b in blobs]
            logger.info(f"Listed blobs: container={container} prefix={prefix!r} count={len(result)}")
            return result
        except AzureError:
            logger.exception(f"Azure list failed: container={container} prefix={prefix!r}")
            raise

    def blob_exists(self, container: str, blob_name: str) -> bool:
        blob_client = self._blob_client(container, blob_name)
        try:
            return bool(blob_client.exists())
        except AzureError:
            logger.exception(f"Azure exists check failed: container={container} blob={blob_name}")
            raise

    def remove_from_cache(self, container: str, blob_name: str) -> bool:
        cache_path = self._cache_path_for(container, blob_name)

        try:
            if os.path.isfile(cache_path):
                os.remove(cache_path)
                logger.info(f"Removed cached file: {cache_path}")
                self._cleanup_empty_dirs_upwards(os.path.dirname(cache_path))
                lock_path = cache_path + ".lock"
                if os.path.isfile(lock_path):
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass
                return True

            logger.warning(f"Cache file not found (skip): {cache_path}")
            return False

        except Exception:
            logger.exception(f"Failed to remove cached file: {cache_path}")
            raise

    def _cleanup_empty_dirs_upwards(self, start_dir: str) -> None:
        current = start_dir
        root = os.path.abspath(self._cache_dir)

        while current and os.path.abspath(current).startswith(root):
            try:
                if os.path.isdir(current) and not os.listdir(current):
                    os.rmdir(current)
                    current = os.path.dirname(current)
                else:
                    break
            except Exception:
                break

    def clear_cache(self) -> None:
        try:
            if os.path.isdir(self._cache_dir):
                shutil.rmtree(self._cache_dir)
            os.makedirs(self._cache_dir, exist_ok=True)
            logger.info(f"Cleared cache dir: {self._cache_dir}")
        except Exception:
            logger.exception(f"Failed to clear cache dir: {self._cache_dir}")
            raise