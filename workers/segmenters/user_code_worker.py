"""
User Code Worker.

Executes data-scientist-submitted Python segmentation algorithms in an isolated
subprocess with timeout and memory limits.

Security model (trusted internal users):
  - subprocess isolation: user code runs in a separate process
  - 30-second wall-clock timeout via subprocess.run(timeout=...)
  - 2 GB virtual memory cap via resource.setrlimit (Linux; skipped on macOS)
  - Import whitelist via a custom sys.meta_path finder injected into the harness
  - No network enforcement (trusted internal users)

Expected task payload:
    {
        "task_id": str,
        "algorithm_id": str,
        "algorithm_code": str,      # Python source string
        "algorithm_params": dict,   # Forwarded to segment(**params)
        "file_path": str | None,    # upload source
        "blob_name": str | None,    # storage source
        "source_type": "upload" | "storage",
    }

The user's code must define a function with this signature:
    def segment(audio_path: str, sr: int = 22050, **params) -> list[dict]:
        # Returns: [{"start": float, "end": float, "label": str}, ...]
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

from shared.logger import get_logger
from workers.BaseWorker import BaseWorker

logger = get_logger()

# Maximum execution time for user code (seconds)
EXECUTION_TIMEOUT = int(os.getenv("USER_CODE_TIMEOUT", "30"))

# Memory limit for user code subprocess (bytes). 2 GB default.
MEMORY_LIMIT_BYTES = int(os.getenv("USER_CODE_MEMORY_BYTES", str(2 * 1024 * 1024 * 1024)))

# Packages the user code may import (others are blocked)
ALLOWED_IMPORTS = {
    "librosa",
    "numpy",
    "np",
    "scipy",
    "sklearn",
    "torch",
    "torchaudio",
    "tensorflow",
    "keras",
    "mir_eval",
    "soundfile",
    "sf",
    "resampy",
    "audioread",
    "openl3",
    "collections",
    "math",
    "itertools",
    "functools",
    "pathlib",
    "os",
    "sys",
    "json",
    "re",
    "typing",
    "abc",
    "copy",
    "dataclasses",
    "enum",
    "io",
    "struct",
    "time",
    "datetime",
    "warnings",
}

# ---------------------------------------------------------------------------
# Harness template — injected with user code + runtime values
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = textwrap.dedent("""
import sys
import json
import traceback

# ── Import whitelist ──────────────────────────────────────────────────────────
_ALLOWED_IMPORTS = {allowed_imports!r}

class _ImportWhitelist:
    \"\"\"sys.meta_path finder that blocks non-whitelisted top-level imports.\"\"\"

    def find_module(self, fullname, path=None):
        top = fullname.split(".")[0]
        if top not in _ALLOWED_IMPORTS:
            raise ImportError(
                f"Import of '{{fullname}}' is not allowed. "
                f"Allowed packages: {{sorted(_ALLOWED_IMPORTS)}}"
            )
        return None  # let normal import machinery handle it

sys.meta_path.insert(0, _ImportWhitelist())

# ── User code ─────────────────────────────────────────────────────────────────
{user_code}

# ── Execution ─────────────────────────────────────────────────────────────────
try:
    _audio_path = {audio_path!r}
    _params = {params!r}
    _result = segment(audio_path=_audio_path, sr=22050, **_params)

    # Validate output shape
    if not isinstance(_result, list):
        raise TypeError(f"segment() must return a list, got {{type(_result).__name__}}")
    for i, seg in enumerate(_result):
        if not isinstance(seg, dict):
            raise TypeError(f"segment()[{{i}}] must be a dict, got {{type(seg).__name__}}")
        if "start" not in seg or "end" not in seg:
            raise ValueError(f"segment()[{{i}}] is missing 'start' or 'end' keys")

    print(json.dumps({{"ok": True, "segments": _result}}))

except Exception as _exc:
    print(json.dumps({{"ok": False, "error": str(_exc), "traceback": traceback.format_exc()}}))
""")


def _set_memory_limit():
    """Preexec function to cap virtual memory (Linux only)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, AttributeError, ValueError):
        # macOS or unsupported platform — skip
        pass


class UserCodeWorker(BaseWorker):
    def __init__(self):
        message_code = os.getenv("MESSAGE_CODE", "segmentation.user_code")

        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "user-code-worker"),
            queue_name=f"queue_{message_code}",
            routing_keys=[message_code],
        )

    def _resolve_file_path_extended(self, task: dict) -> tuple[str, bool]:
        """
        Extended file resolution that also handles 'upload_url' source type.
        Returns (file_path, is_temp) where is_temp=True means caller should delete the file.
        """
        source_type = str(task.get("source_type", "upload")).lower().strip()

        if source_type == "upload_url":
            audio_url = task.get("audio_url")
            if not audio_url:
                raise ValueError("audio_url is required for upload_url source")

            import tempfile
            import urllib.request
            suffix = ".mp3"
            for ext in (".wav", ".flac", ".ogg", ".m4a"):
                if audio_url.lower().endswith(ext):
                    suffix = ext
                    break

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.close()

            req = urllib.request.Request(
                audio_url,
                headers={"User-Agent": "Mozilla/5.0 (MusicSegmentation/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(tmp.name, "wb") as f:
                    f.write(resp.read())

            logger.info(f"[UserCodeWorker] Downloaded audio from URL to {tmp.name}")
            return tmp.name, True

        return self._resolve_file_path(task), False

    def process_task(self, task: dict) -> dict:
        task_id = task.get("task_id")
        algorithm_id = task.get("algorithm_id", "unknown")
        algorithm_code = task.get("algorithm_code", "")
        algorithm_params = task.get("algorithm_params") or {}

        if not algorithm_code.strip():
            raise ValueError("algorithm_code is empty")

        file_path, is_temp = self._resolve_file_path_extended(task)
        logger.info(f"[UserCodeWorker] Running algorithm '{algorithm_id}' on {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Build the harness script
        harness_source = _HARNESS_TEMPLATE.format(
            allowed_imports=ALLOWED_IMPORTS,
            user_code=algorithm_code,
            audio_path=file_path,
            params=algorithm_params,
        )

        # Write harness to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix=f"harness_{task_id}_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(harness_source)
            harness_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, harness_path],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT,
                preexec_fn=_set_memory_limit,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"[UserCodeWorker] Algorithm '{algorithm_id}' timed out after {EXECUTION_TIMEOUT}s")
            return self._error_result(
                task_id=task_id,
                algorithm_id=algorithm_id,
                error=f"Execution timed out after {EXECUTION_TIMEOUT} seconds",
            )
        finally:
            try:
                os.unlink(harness_path)
            except OSError:
                pass
            if is_temp:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass

        if result.returncode != 0 and not result.stdout.strip():
            error_msg = result.stderr.strip() or f"Process exited with code {result.returncode}"
            logger.error(f"[UserCodeWorker] Harness crashed: {error_msg}")
            return self._error_result(
                task_id=task_id,
                algorithm_id=algorithm_id,
                error=error_msg,
            )

        # Parse JSON output from harness
        stdout = result.stdout.strip()
        if not stdout:
            return self._error_result(
                task_id=task_id,
                algorithm_id=algorithm_id,
                error="Algorithm produced no output",
            )

        try:
            output = json.loads(stdout)
        except json.JSONDecodeError as e:
            logger.error(f"[UserCodeWorker] Failed to parse harness output: {e}\nstdout: {stdout[:500]}")
            return self._error_result(
                task_id=task_id,
                algorithm_id=algorithm_id,
                error=f"Output JSON parse error: {e}",
            )

        if not output.get("ok"):
            error = output.get("error", "Unknown error")
            tb = output.get("traceback", "")
            logger.warning(f"[UserCodeWorker] Algorithm raised exception: {error}\n{tb}")
            return self._error_result(
                task_id=task_id,
                algorithm_id=algorithm_id,
                error=error,
                traceback=tb,
            )

        segments = output.get("segments", [])
        logger.info(f"[UserCodeWorker] Algorithm '{algorithm_id}' produced {len(segments)} segments")

        return {
            "task_id": task_id,
            "status": "completed",
            "worker_type": "user_code",
            "algorithm": algorithm_id,
            "segments": segments,
        }

    @staticmethod
    def _error_result(task_id: str, algorithm_id: str, error: str, traceback: str = "") -> dict:
        return {
            "task_id": task_id,
            "status": "failed",
            "worker_type": "user_code",
            "algorithm": algorithm_id,
            "segments": [],
            "error": error,
            "traceback": traceback,
        }
