import { writable, get } from "svelte/store";
import { uploadSegmentation, subscribeToTask, getSongStreamUrl } from "./api.js";

export const BASELINE_ALGOS = ["custom_librosa", "foote", "cnmf", "scluster"];

export const ALL_ALGOS = [
  { id: "fusion", name: "Fusion", color: "var(--msp-alg-fusion)", desc: "Weighted combination of all algorithm outputs", isFusion: true },
  { id: "custom_librosa", name: "Custom Librosa", color: "var(--msp-alg-librosa)", desc: "Self-similarity, novelty, harmony and onset evidence" },
  { id: "foote", name: "Foote", color: "var(--msp-alg-foote)", desc: "Novelty-based baseline segmentation" },
  { id: "cnmf", name: "CNMF", color: "var(--msp-alg-cnmf)", desc: "Matrix factorization for recurring patterns" },
  { id: "scluster", name: "SCluster", color: "var(--msp-alg-scluster)", desc: "Spectral clustering of similar regions" },
];

function initialState() {
  return {
    // ── source ──────────────────────────────────────────────────────────
    file: null,
    sourceTrack: null, // dataset track picked from "Browse Library" (has ground_truth, song_id, ...)
    audioUrl: null,

    // ── config ──────────────────────────────────────────────────────────
    selectedAlgos: new Set(["fusion"]),
    labelingMethod: "heuristic",
    advanced: { threshold: 0.62, mergeWindowSeconds: 1.2, requiredVoteCount: 2 },

    // ── run ─────────────────────────────────────────────────────────────
    taskId: "",
    status: "idle", // idle | uploading | processing | completed | error
    errorMsg: "",
    requested: [],
    results: {},
    rawStatus: {},
    processingTimes: {}, // algo -> seconds (measured client-side if backend doesn't report)
    startedAt: null,

    // ── overview / compare selection ───────────────────────────────────
    selectedSegmentIndex: 0,
    viewAlgo: null, // algo id currently shown on Overview/Technical; null = auto (fusion, else first)
    visibleAlgos: {}, // algo id -> bool, for the Compare screen
  };
}

export const analysis = writable(initialState());

let unsubscribeTask = null;

export function setFile(file) {
  analysis.update((s) => ({
    ...s,
    file,
    sourceTrack: null,
    audioUrl: file ? URL.createObjectURL(file) : null,
  }));
}

export function setSourceTrack(track) {
  analysis.update((s) => ({
    ...s,
    sourceTrack: track,
    file: null,
    audioUrl: track?.song_id ? getSongStreamUrl(track.song_id) : track?.audio_url || null,
  }));
}

export function toggleAlgo(id) {
  analysis.update((s) => {
    const next = new Set(s.selectedAlgos);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return { ...s, selectedAlgos: next };
  });
}

export function setLabelingMethod(method) {
  analysis.update((s) => ({ ...s, labelingMethod: method }));
}

export function setAdvanced(partial) {
  analysis.update((s) => ({ ...s, advanced: { ...s.advanced, ...partial } }));
}

export function selectSegment(index) {
  analysis.update((s) => ({ ...s, selectedSegmentIndex: index }));
}

export function setViewAlgo(id) {
  analysis.update((s) => ({ ...s, viewAlgo: id, selectedSegmentIndex: 0 }));
}

export function toggleVisibleAlgo(id) {
  analysis.update((s) => ({
    ...s,
    visibleAlgos: { ...s.visibleAlgos, [id]: s.visibleAlgos[id] === false ? true : false },
  }));
}

export function resetRun() {
  if (unsubscribeTask) {
    unsubscribeTask();
    unsubscribeTask = null;
  }
  analysis.update((s) => ({
    ...s,
    taskId: "",
    status: "idle",
    errorMsg: "",
    requested: [],
    results: {},
    rawStatus: {},
    processingTimes: {},
    startedAt: null,
    selectedSegmentIndex: 0,
    viewAlgo: null,
  }));
}

async function resolveAudioFile(state) {
  if (state.file) return state.file;
  if (state.sourceTrack) {
    const url = state.sourceTrack.song_id
      ? getSongStreamUrl(state.sourceTrack.song_id)
      : state.sourceTrack.audio_url;
    if (!url) throw new Error("Selected track has no audio source.");
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch audio: ${resp.status} ${resp.statusText}`);
    const blob = await resp.blob();
    return new File(
      [blob],
      `${state.sourceTrack.song_id || state.sourceTrack.track_id || "track"}.mp3`,
      { type: resp.headers.get("content-type") || "audio/mpeg" },
    );
  }
  throw new Error("Pick an audio file first.");
}

/**
 * Kick off segmentation for the currently configured source + algorithms.
 * Streams progress into the store via SSE; resolves once the run reaches a
 * terminal state (completed/failed).
 */
export async function startAnalysis() {
  const state = get(analysis);
  const algos = Array.from(state.selectedAlgos);
  if (algos.length === 0) {
    analysis.update((s) => ({ ...s, status: "error", errorMsg: "Select at least one algorithm." }));
    return;
  }

  if (unsubscribeTask) unsubscribeTask();

  let file;
  try {
    file = await resolveAudioFile(state);
  } catch (err) {
    analysis.update((s) => ({ ...s, status: "error", errorMsg: err.message }));
    return;
  }

  const startedAt = Date.now();
  analysis.update((s) => ({
    ...s,
    status: "uploading",
    errorMsg: "",
    requested: algos,
    results: {},
    rawStatus: {},
    processingTimes: {},
    startedAt,
    selectedSegmentIndex: 0,
    viewAlgo: null,
  }));

  const customParams = { labeling_method: state.labelingMethod };
  const params = {
    custom: customParams,
    custom_librosa: customParams,
    fusion: {
      threshold: state.advanced.threshold,
      merge_window_seconds: state.advanced.mergeWindowSeconds,
      required_vote_count: state.advanced.requiredVoteCount,
    },
  };

  try {
    const taskId = await uploadSegmentation({ file, algorithms: algos, params });
    analysis.update((s) => ({ ...s, taskId, status: "processing" }));

    unsubscribeTask = subscribeToTask(taskId, (data) => {
      analysis.update((s) => {
        const newResults = data.results || {};
        const times = { ...s.processingTimes };
        const arrivalSecs = (Date.now() - startedAt) / 1000;
        for (const k of Object.keys(newResults)) {
          if (k.endsWith("__processing_time") && typeof newResults[k] === "number") {
            times[k.replace("__processing_time", "")] = newResults[k];
          } else if (!k.includes("__") && !(k in s.results) && times[k] == null) {
            times[k] = arrivalSecs;
          }
        }
        const results = { ...s.results, ...newResults };
        let status = s.status;
        let errorMsg = s.errorMsg;
        if (data.status === "completed") status = "completed";
        else if (data.status === "failed") {
          status = "error";
          errorMsg = data.error || "Backend reported failure.";
        }
        return { ...s, results, rawStatus: data, processingTimes: times, status, errorMsg };
      });
    });
  } catch (err) {
    analysis.update((s) => ({ ...s, status: "error", errorMsg: err.message }));
  }
}

/**
 * Full set of algorithm ids expected to produce a result for the current
 * run, expanding "fusion" into its baseline dependencies.
 */
export function expectedAlgos(state) {
  const exp = [...state.requested];
  if (state.requested.includes("fusion")) {
    BASELINE_ALGOS.forEach((a) => { if (!exp.includes(a)) exp.push(a); });
  }
  return exp;
}

/**
 * Per-algorithm run status derived from the raw results payload.
 * A failed worker still publishes an (empty) array under its own key, with
 * failure detail under `<algo>__diagnostics.error` or
 * `<algo>__result.status === "failed"` — there is no dedicated top-level
 * error field from the backend.
 */
export function algoRunStatus(state, algoId) {
  const { results, status } = state;
  const hasResult = Array.isArray(results[algoId]);
  const diagError = results[`${algoId}__diagnostics`]?.error;
  const resultStatus = results[`${algoId}__result`]?.status;
  if (hasResult) {
    if (diagError || resultStatus === "failed") {
      return { state: "failed", error: diagError || "Algorithm reported failure." };
    }
    return { state: "completed", error: null };
  }
  if (status === "processing" || status === "uploading") return { state: "running", error: null };
  return { state: "queued", error: null };
}

export function stopSubscription() {
  if (unsubscribeTask) {
    unsubscribeTask();
    unsubscribeTask = null;
  }
}
