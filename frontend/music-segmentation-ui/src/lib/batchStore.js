import { writable, get } from "svelte/store";
import { listDatasets, startBatchEval, subscribeToBatchEval } from "./api.js";

function initialState() {
  return {
    datasets: [],
    datasetId: "",
    runAllDataset: true,
    maxTracks: 20,
    concurrency: 3,
    tolerance: 0.5,
    algorithms: new Set(["custom_librosa", "foote", "cnmf", "scluster", "fusion"]),

    running: false,
    done: false,
    jobId: null,
    progress: { completed: 0, total: 0 },
    logLines: [],
    rows: [],
    summary: "",
    error: "",
  };
}

export const batch = writable(initialState());

let unsub = null;
const RE_PROGRESS = /\[\s*(\d+)\/\s*(\d+)\]/;

export async function loadDatasets() {
  try {
    const datasets = await listDatasets();
    batch.update((s) => ({ ...s, datasets }));
  } catch (e) {
    batch.update((s) => ({ ...s, error: e.message }));
  }
}

export function setField(partial) {
  batch.update((s) => ({ ...s, ...partial }));
}

export function toggleAlgorithm(id) {
  batch.update((s) => {
    const next = new Set(s.algorithms);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return { ...s, algorithms: next };
  });
}

export function resetBatch() {
  if (unsub) { unsub(); unsub = null; }
  batch.set(initialState());
}

export async function runBatch() {
  if (unsub) unsub();
  batch.update((s) => ({
    ...s, running: true, done: false, logLines: [], rows: [], summary: "", error: "",
    progress: { completed: 0, total: 0 },
  }));

  const state = get(batch);
  try {
    const { job_id } = await startBatchEval({
      maxTracks: state.runAllDataset ? 0 : Number(state.maxTracks),
      toleranceSeconds: Number(state.tolerance),
      tolerances: [Number(state.tolerance), 3.0],
      algorithms: Array.from(state.algorithms),
      concurrency: Number(state.concurrency),
    });
    batch.update((s) => ({ ...s, jobId: job_id }));

    unsub = subscribeToBatchEval(
      job_id,
      (line) => {
        batch.update((s) => {
          const logLines = s.logLines.length >= 300 ? [...s.logLines.slice(-299), line] : [...s.logLines, line];
          const m = RE_PROGRESS.exec(line);
          const progress = m ? { completed: parseInt(m[1]), total: parseInt(m[2]) } : s.progress;
          return { ...s, logLines, progress };
        });
      },
      ({ summary, rows, error }) => {
        batch.update((s) => ({
          ...s,
          rows: rows ?? [],
          summary: summary ?? "",
          error: error ?? "",
          running: false,
          done: true,
          progress: { completed: (rows ?? []).filter((r) => !r.error).length, total: (rows ?? []).length },
        }));
      },
    );
  } catch (e) {
    batch.update((s) => ({ ...s, error: e.message, running: false }));
  }
}
