const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

/**
 * @typedef {{ task_id: string, status: string, worker_type?: string, algorithm?: string, segments?: Array<Record<string, unknown>>, results?: Record<string, unknown>, error?: string }} TaskMessage
 * @typedef {{ file: File, groundTruthCsv?: File | null, title?: string | null, artist?: string | null }} UploadTrackOptions
 * @typedef {{ file: File, algorithms: string[], webhook_url?: string | null }} UploadSegmentationOptions
 * @typedef {{ audioSource: { type: string, value: string }, params?: Record<string, unknown> }} TestAlgorithmOptions
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

// ── Algorithms ────────────────────────────────────────────────────────────────

export function listAlgorithms() {
  return apiFetch("/algorithms");
}

/**
 * @param {string} id
 */
export function getAlgorithm(id) {
  return apiFetch(`/algorithms/${id}`);
}

/**
 * @param {{ name: string, description?: string | null, code: string, params_schema?: Record<string, unknown> | null }} options
 */
export function saveAlgorithm({ name, description = null, code, params_schema = null }) {
  return apiFetch("/algorithms", {
    method: "POST",
    ...jsonBody({ name, description, code, params_schema }),
  });
}

/**
 * @param {string} name
 */
export function listAlgorithmVersions(name) {
  return apiFetch(`/algorithms/${encodeURIComponent(name)}/versions`);
}

/**
 * @param {string} id
 */
export function deleteAlgorithm(id) {
  return apiFetch(`/algorithms/${id}`, { method: "DELETE" });
}

/**
 * @param {string} id
 * @param {TestAlgorithmOptions} options
 */
export function testAlgorithm(id, { audioSource, params = {} }) {
  return apiFetch(`/algorithms/${id}/test`, {
    method: "POST",
    ...jsonBody({ audio_source: audioSource, params }),
  });
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
export function segmentSongFromStorage(songId, algorithms = ["custom", "foote", "cnmf", "scluster"]) {
  return apiFetch("/segmentation/from-storage", {
    method: "POST",
    ...jsonBody({ song_id: songId, algorithms }),
  });
}

/**
 * @param {string[]} [songIds]
 * @param {string[]} [algorithms]
 */
export function segmentSongsBatch(songIds = [], algorithms = ["custom", "foote", "cnmf", "scluster"]) {
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
 * @param {UploadSegmentationOptions} options
 */
export async function uploadSegmentation({ file, algorithms, webhook_url = null }) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("algorithms", JSON.stringify(algorithms));
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
  const eventSource = new EventSource(`${BACKEND_URL}/segmentation/stream/${taskId}`);
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
      if (data.status === "completed" || data.status === "failed") {
        eventSource.close();
      }
    } catch (e) {
      console.error("Failed to parse SSE message:", e);
    }
  };
  
  eventSource.onerror = () => {
    eventSource.close();
  };
  
  return () => eventSource.close();
}
