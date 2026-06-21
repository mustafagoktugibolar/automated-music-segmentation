const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

/**
 * @typedef {{ task_id: string, status: string, worker_type?: string, algorithm?: string, segments?: Array<Record<string, unknown>>, results?: Record<string, unknown>, error?: string }} TaskMessage
 * @typedef {{ file: File, groundTruthCsv?: File | null, title?: string | null, artist?: string | null }} UploadTrackOptions
 * @typedef {{ file: File, algorithms: string[], webhook_url?: string | null, params?: Record<string, unknown> | null }} UploadSegmentationOptions
 */

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * @param {string} path
 * @param {RequestInit} [options]
 */
async function apiFetch(path, options = {}) {
  const res = await fetch(`${BACKEND_URL}${path}`, options);
  const text = await res.text();
  if (!res.ok) {
    let detail = text.slice(0, 300);
    try {
      const j = JSON.parse(text);
      detail = j.detail ?? detail;
    } catch {}
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Invalid JSON response: ${text.slice(0, 200)}`);
  }
}

/**
 * @param {Record<string, unknown>} data
 */
function jsonBody(data) {
  return {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
}

// ── Datasets ──────────────────────────────────────────────────────────────────

export function listDatasets() {
  return apiFetch("/datasets");
}

/**
 * @param {{ name: string, description?: string | null }} options
 */
export function createDataset({ name, description = null }) {
  return apiFetch("/datasets", {
    method: "POST",
    ...jsonBody({ name, description }),
  });
}

export function importSalami() {
  return apiFetch("/datasets/import-salami", { method: "POST" });
}

/**
 * @param {string} datasetId
 * @param {{ page?: number, pageSize?: number, hasGroundTruth?: boolean | null }} [options]
 */
export function listDatasetTracks(datasetId, { page = 1, pageSize = 50, hasGroundTruth = null } = {}) {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (hasGroundTruth !== null) params.set("has_ground_truth", String(hasGroundTruth));
  return apiFetch(`/datasets/${datasetId}/tracks?${params}`);
}

/**
 * @param {string} datasetId
 * @param {string} trackId
 */
export function getDatasetTrack(datasetId, trackId) {
  return apiFetch(`/datasets/${datasetId}/tracks/${trackId}`);
}

// ── Segmentation / Songs helpers ───────────────────────────────────────────

/**
 * @param {string} songId
 * @param {string[]} [algorithms]
 */
export function segmentSongFromStorage(songId, algorithms = ["custom_librosa", "foote", "cnmf", "scluster"]) {
  return apiFetch("/segmentation/from-storage", {
    method: "POST",
    ...jsonBody({ song_id: songId, algorithms }),
  });
}

/**
 * @param {string[]} [songIds]
 * @param {string[]} [algorithms]
 */
export function segmentSongsBatch(songIds = [], algorithms = ["custom_librosa", "foote", "cnmf", "scluster"]) {
  return apiFetch("/songs/segment-batch", {
    method: "POST",
    ...jsonBody({ song_ids: songIds, algorithms }),
  });
}

/**
 * @param {string} songId
 */
export function getSongStreamUrl(songId) {
  return `${BACKEND_URL}/songs/stream/${encodeURIComponent(songId)}`;
}

/**
 * @param {string} datasetId
 * @param {UploadTrackOptions} options
 */
export async function uploadTrack(datasetId, { file, groundTruthCsv = null, title = null, artist = null }) {
  const form = new FormData();
  form.append("file", file);
  if (groundTruthCsv) form.append("ground_truth_csv", groundTruthCsv);
  if (title) form.append("title", title);
  if (artist) form.append("artist", artist);
  const res = await fetch(`${BACKEND_URL}/datasets/${datasetId}/tracks/upload`, {
    method: "POST",
    body: form,
  });
  const text = await res.text();
  if (!res.ok) {
    let detail = text.slice(0, 300);
    try { detail = JSON.parse(text).detail ?? detail; } catch {}
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return JSON.parse(text);
}

// ── Evaluation ────────────────────────────────────────────────────────────────

/**
 * @param {{ taskId: string, trackId: string, toleranceSeconds?: number }} options
 */
export function runEvaluation({ taskId, trackId, toleranceSeconds = 3.0 }) {
  return apiFetch("/evaluation/run", {
    method: "POST",
    ...jsonBody({ task_id: taskId, track_id: trackId, tolerance_seconds: toleranceSeconds }),
  });
}

/**
 * @param {{ trackId: string, algorithmNames?: string[], taskIds?: Record<string, string>, toleranceSeconds?: number }} options
 */
export function compareAlgorithms({ trackId, algorithmNames = [], taskIds = {}, toleranceSeconds = 3.0 }) {
  return apiFetch("/evaluation/compare", {
    method: "POST",
    ...jsonBody({
      track_id: trackId,
      algorithm_names: algorithmNames,
      task_ids: taskIds,
      tolerance_seconds: toleranceSeconds,
    }),
  });
}

/**
 * @param {string} trackId
 */
export function getEvaluationsForTrack(trackId) {
  return apiFetch(`/evaluation/track/${trackId}`);
}

/**
 * @param {string} trackId
 */
export function getSegmentationsForTrack(trackId) {
  return apiFetch(`/evaluation/track/${trackId}/segmentations`);
}

/**
 * @param {string} evalId
 */
export function getEvaluation(evalId) {
  return apiFetch(`/evaluation/${evalId}`);
}

/**
 * @param {{ maxTracks?: number, toleranceSeconds?: number, tolerances?: number[], algorithms?: string[], concurrency?: number, includeLLM?: boolean, llmMode?: string, coverageOutlierThreshold?: number }} options
 */
export function startBatchEval({
  maxTracks = 20,
  toleranceSeconds = 0.5,
  tolerances = [0.5, 3.0],
  algorithms = ["custom_librosa", "foote", "cnmf", "scluster", "fusion"],
  concurrency = 3,
  includeLLM = false,
  llmMode = "deterministic",
  coverageOutlierThreshold = 0.20,
} = {}) {
  return apiFetch("/evaluation/batch", {
    method: "POST",
    ...jsonBody({
      max_tracks: maxTracks,
      tolerance_seconds: toleranceSeconds,
      tolerances,
      algorithms,
      concurrency,
      include_llm: includeLLM,
      llm_mode: llmMode,
      coverage_outlier_threshold: coverageOutlierThreshold,
    }),
  });
}

/**
 * @param {{ limit?: number }} [options]
 */
export function listBatchEvalHistory({ limit = 30 } = {}) {
  return apiFetch(`/evaluation/batch/history?limit=${limit}`);
}

/**
 * @param {string} jobId
 */
export function getBatchEvalResult(jobId) {
  return apiFetch(`/evaluation/batch/${jobId}/result`);
}

/**
 * Subscribe to batch eval progress via SSE.
 * onLine(line: string) called for each log line.
 * onDone({ summary, rows, error }) called when complete.
 * Returns an unsubscribe function.
 *
 * @param {string} jobId
 * @param {(line: string) => void} onLine
 * @param {(result: { summary: string | null, rows: any[], error: string | null }) => void} onDone
 */
export function subscribeToBatchEval(jobId, onLine, onDone) {
  let closed = false;
  let pollTimer = null;

  function startPolling() {
    if (closed) return;
    pollTimer = setInterval(async () => {
      if (closed) { clearInterval(pollTimer); return; }
      try {
        const result = await getBatchEvalResult(jobId);
        if (result && !closed) {
          closed = true;
          clearInterval(pollTimer);
          onDone({ summary: result.summary ?? null, rows: result.rows ?? [], error: result.error ?? null });
        }
      } catch (_) {}
    }, 5000);
  }

  const es = new EventSource(`${BACKEND_URL}/evaluation/batch/${jobId}/stream`);
  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.done) {
        closed = true;
        clearInterval(pollTimer);
        onDone({ summary: data.summary ?? null, rows: data.rows ?? [], error: data.error ?? null });
        es.close();
      } else if (data.line !== undefined) {
        onLine(data.line);
      }
    } catch (e) {
      console.error("SSE parse error:", e);
    }
  };
  es.onerror = () => {
    es.close();
    startPolling();
  };
  return () => {
    closed = true;
    clearInterval(pollTimer);
    es.close();
  };
}

/**
 * @param {UploadSegmentationOptions} options
 */
export async function uploadSegmentation({ file, algorithms, webhook_url = null, params = null }) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("algorithms", JSON.stringify(algorithms));
  if (params) {
    formData.append("params", JSON.stringify(params));
  }
  if (webhook_url) {
    formData.append("webhook_url", webhook_url);
  }

  const res = await fetch(`${BACKEND_URL}/segmentation/upload`, {
    method: "POST",
    body: formData,
  });

  const text = await res.text();

  if (!res.ok) {
    throw new Error(
      `Upload failed (${res.status} ${res.statusText}): ${text.slice(0, 200)}`,
    );
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Upload returned invalid JSON: ${text.slice(0, 200)}`);
  }

  if (!data?.task_id) throw new Error("Upload response missing task_id");
  return data.task_id;
}

/**
 * @param {string} taskId
 */
export async function fetchStatus(taskId) {
  const res = await fetch(`${BACKEND_URL}/segmentation/status/${taskId}`);
  const text = await res.text();

  if (!res.ok) {
    throw new Error(
      `Status failed (${res.status} ${res.statusText}): ${text.slice(0, 200)}`,
    );
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Status returned invalid JSON: ${text.slice(0, 200)}`);
  }

  return data;
}

/**
 * @param {string} taskId
 * @param {(data: TaskMessage | Record<string, unknown>) => void} onMessage
 */
export function subscribeToTask(taskId, onMessage) {
  let terminated = false;

  function terminate() {
    if (!terminated) {
      terminated = true;
      eventSource.close();
    }
  }

  /** @param {Record<string, unknown>} data */
  function dispatch(data) {
    if (terminated) return;
    onMessage(data);
    if (data.status === "completed" || data.status === "failed") terminate();
  }

  // SSE — primary channel
  const eventSource = new EventSource(`${BACKEND_URL}/segmentation/stream/${taskId}`);

  eventSource.onmessage = (event) => {
    try { dispatch(JSON.parse(event.data)); }
    catch (e) { console.error("SSE parse error:", e); }
  };

  eventSource.onerror = () => {
    eventSource.close();
  };

  // Polling — runs in parallel from the start as a silent safety net.
  // Starts after 2 s to let SSE deliver first in the happy path.
  (async () => {
    await new Promise(r => setTimeout(r, 2000));
    let delay = 3000;
    while (!terminated) {
      await new Promise(r => setTimeout(r, delay));
      if (terminated) break;
      try {
        dispatch(await fetchStatus(taskId));
      } catch { /* backend unreachable — keep retrying */ }
      delay = Math.min(delay * 1.5, 10000);
    }
  })();

  return terminate;
}
