<script>
  import { onMount } from "svelte";
  import {
    listAlgorithms,
    getAlgorithm,
    saveAlgorithm,
    testAlgorithm,
    listAlgorithmVersions,
    deleteAlgorithm,
    subscribeToTask,
    listDatasets,
    listDatasetTracks,
  } from "../lib/api.js";

  // ── State ────────────────────────────────────────────────────────────────
  let algorithms = [];
  let selectedAlgoId = null;
  let editorCode = defaultCode();
  let algoName = "";
  let algoDescription = "";

  let isSaving = false;
  let saveError = "";

  let datasets = [];
  let selectedDatasetId = "";
  let tracks = [];
  let selectedTrackId = "";
  let extraParams = "{}";

  let isRunning = false;
  let runError = "";
  let runResult = null;     // {segments: [...]}
  let taskId = "";
  let taskStatus = "";
  let unsubscribe = null;

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  onMount(async () => {
    await loadAlgorithms();
    await loadDatasets();
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  function defaultCode() {
    return `import numpy as np
import librosa

def segment(audio_path: str, sr: int = 22050, **params) -> list[dict]:
    """
    Your segmentation algorithm.
    Must return a list of dicts: [{start: float, end: float, label: str}, ...]
    """
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Example: single segment covering the whole track
    return [{"start": 0.0, "end": round(duration, 2), "label": "A"}]
`;
  }

  async function loadAlgorithms() {
    try {
      algorithms = await listAlgorithms();
    } catch (e) {
      console.error("Failed to load algorithms", e);
    }
  }

  async function loadDatasets() {
    try {
      datasets = await listDatasets();
    } catch (e) {
      console.error("Failed to load datasets", e);
    }
  }

  async function selectDataset(id) {
    selectedDatasetId = id;
    selectedTrackId = "";
    tracks = [];
    if (!id) return;
    try {
      const res = await listDatasetTracks(id, { page: 1, pageSize: 100 });
      tracks = res.tracks || [];
    } catch (e) {
      console.error("Failed to load tracks", e);
    }
  }

  async function selectAlgorithm(algo) {
    selectedAlgoId = algo.algorithm_id;
    algoName = algo.name;
    try {
      const full = await getAlgorithm(algo.algorithm_id);
      editorCode = full.code;
      algoDescription = full.description || "";
    } catch (e) {
      console.error("Failed to load algorithm code", e);
    }
  }

  function newAlgorithm() {
    selectedAlgoId = null;
    algoName = "";
    algoDescription = "";
    editorCode = defaultCode();
    runResult = null;
    taskId = "";
    taskStatus = "";
  }

  async function save() {
    if (!algoName.trim()) { saveError = "Algorithm name is required."; return; }
    isSaving = true;
    saveError = "";
    try {
      const res = await saveAlgorithm({
        name: algoName.trim(),
        description: algoDescription.trim() || null,
        code: editorCode,
      });
      selectedAlgoId = res.algorithm_id;
      await loadAlgorithms();
    } catch (e) {
      saveError = e.message;
    } finally {
      isSaving = false;
    }
  }

  async function runTest() {
    if (!selectedAlgoId) { runError = "Save the algorithm first."; return; }
    if (!selectedTrackId && !selectedDatasetId) {
      runError = "Select a track to test against.";
      return;
    }

    /** @type {Record<string, unknown>} */
    let parsedParams = {};
    try {
      parsedParams = JSON.parse(extraParams || "{}");
    } catch {
      runError = "Params must be valid JSON.";
      return;
    }

    isRunning = true;
    runError = "";
    runResult = null;
    taskId = "";
    taskStatus = "dispatching";

    if (unsubscribe) { unsubscribe(); unsubscribe = null; }

    try {
      const audioSource = selectedTrackId
        ? { type: "track_id", value: selectedTrackId }
        : { type: "salami", value: selectedDatasetId };

      const res = await testAlgorithm(selectedAlgoId, { audioSource, params: parsedParams });
      taskId = res.task_id;
      taskStatus = "processing";

      unsubscribe = subscribeToTask(taskId, /** @param {any} data */ (data) => {
        taskStatus = data.status || "processing";
        if (data.results && Object.keys(data.results).length > 0) {
          const firstKey = Object.keys(data.results)[0];
          runResult = data.results[firstKey];
        }
        if (data.status === "completed" || data.status === "failed") {
          isRunning = false;
          if (data.status === "failed") runError = "Task failed on worker.";
          if (unsubscribe) { unsubscribe(); unsubscribe = null; }
        }
      });
    } catch (e) {
      runError = e.message;
      isRunning = false;
    }
  }

  // ── SVG segment timeline ──────────────────────────────────────────────────
  function buildTimeline(segments) {
    if (!segments || segments.length === 0) return [];
    const totalDuration = Math.max(...segments.map(s => s.end));
    const width = 600;
    const colors = ["#818cf8","#34d399","#fb923c","#f472b6","#60a5fa","#a78bfa","#facc15"];
    const labels = [...new Set(segments.map(s => s.label))];
    return segments.map((seg, i) => ({
      x: (seg.start / totalDuration) * width,
      w: Math.max(2, ((seg.end - seg.start) / totalDuration) * width),
      color: colors[labels.indexOf(seg.label) % colors.length],
      label: seg.section_type || seg.label,
      start: seg.start,
      end: seg.end,
      section_type: seg.section_type,
    }));
  }
</script>

<div class="flex h-[calc(100vh-49px)] overflow-hidden text-zinc-100">
  <!-- Left: Algorithm list -->
  <aside class="w-56 shrink-0 border-r border-zinc-800 bg-zinc-900/50 flex flex-col">
    <div class="p-4 border-b border-zinc-800 flex items-center justify-between">
      <span class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Algorithms</span>
      <button
        class="rounded-lg bg-indigo-500/20 px-2 py-1 text-xs font-medium text-indigo-300 hover:bg-indigo-500/30"
        on:click={newAlgorithm}
      >+ New</button>
    </div>
    <div class="flex-1 overflow-y-auto p-2 space-y-1">
      {#each algorithms as algo}
        <button
          class={"w-full text-left rounded-xl px-3 py-2 text-sm transition-colors " +
            (selectedAlgoId === algo.algorithm_id
              ? "bg-indigo-500/20 text-indigo-200"
              : "text-zinc-300 hover:bg-zinc-800")}
          on:click={() => selectAlgorithm(algo)}
        >
          <div class="font-medium truncate">{algo.name}</div>
          <div class="text-[11px] text-zinc-500">v{algo.version}</div>
        </button>
      {/each}
      {#if algorithms.length === 0}
        <p class="text-xs text-zinc-500 px-3 py-2">No algorithms yet</p>
      {/if}
    </div>
  </aside>

  <!-- Center: Code editor -->
  <div class="flex-1 flex flex-col min-w-0 border-r border-zinc-800">
    <!-- Toolbar -->
    <div class="flex items-center gap-3 border-b border-zinc-800 bg-zinc-900/50 px-4 py-2">
      <input
        class="flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
        placeholder="Algorithm name…"
        bind:value={algoName}
      />
      <input
        class="w-48 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
        placeholder="Description (optional)"
        bind:value={algoDescription}
      />
      <button
        class="rounded-xl bg-indigo-500 px-4 py-1.5 text-sm font-semibold text-white hover:bg-indigo-400 disabled:opacity-50"
        on:click={save}
        disabled={isSaving}
      >
        {isSaving ? "Saving…" : "Save"}
      </button>
      {#if saveError}
        <span class="text-xs text-red-400">{saveError}</span>
      {/if}
    </div>

    <!-- Editor hint -->
    <div class="px-4 py-1.5 bg-zinc-950/40 border-b border-zinc-800 text-[11px] text-zinc-500">
      Define <code class="text-indigo-300">segment(audio_path, sr=22050, **params) → list[dict]</code> — each dict needs <code class="text-indigo-300">start</code>, <code class="text-indigo-300">end</code>, <code class="text-indigo-300">label</code>
    </div>

    <!-- Code textarea -->
    <textarea
      class="flex-1 resize-none bg-zinc-950 px-4 py-4 font-mono text-sm text-zinc-200 leading-relaxed focus:outline-none"
      spellcheck="false"
      autocomplete="off"
      bind:value={editorCode}
    ></textarea>
  </div>

  <!-- Right: Test panel -->
  <div class="w-80 shrink-0 flex flex-col bg-zinc-900/30 overflow-y-auto">
    <div class="p-4 border-b border-zinc-800">
      <h3 class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Test Panel</h3>
    </div>

    <div class="p-4 space-y-4">
      <!-- Dataset selector -->
      <div>
        <label class="text-xs font-medium text-zinc-400 block mb-1" for="ds-select">Dataset</label>
        <select
          id="ds-select"
          class="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
          bind:value={selectedDatasetId}
          on:change={(e) => selectDataset(e.currentTarget.value)}
        >
          <option value="">— select dataset —</option>
          {#each datasets as ds}
            <option value={ds.dataset_id}>{ds.name} ({ds.track_count})</option>
          {/each}
        </select>
      </div>

      <!-- Track selector -->
      {#if tracks.length > 0}
        <div>
          <label class="text-xs font-medium text-zinc-400 block mb-1" for="track-select">Track</label>
          <select
            id="track-select"
            class="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
            bind:value={selectedTrackId}
          >
            <option value="">— select track —</option>
            {#each tracks as t}
              <option value={t.track_id}>{t.title || t.song_id} {t.has_ground_truth ? "✓" : ""}</option>
            {/each}
          </select>
        </div>
      {/if}

      <!-- Params JSON -->
      <div>
        <label class="text-xs font-medium text-zinc-400 block mb-1" for="params-input">Params (JSON)</label>
        <textarea
          id="params-input"
          class="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-200 focus:border-indigo-500 focus:outline-none"
          rows="3"
          bind:value={extraParams}
        ></textarea>
      </div>

      <!-- Run button -->
      <button
        class="w-full rounded-2xl bg-emerald-600 py-2.5 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
        on:click={runTest}
        disabled={isRunning || !selectedAlgoId}
      >
        {#if isRunning}
          <span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/80 mr-2"></span>
          Running…
        {:else}
          ▶ Run Test
        {/if}
      </button>

      {#if runError}
        <div class="rounded-xl border border-red-900/60 bg-red-950/30 px-3 py-2 text-xs text-red-300">
          {runError}
        </div>
      {/if}

      <!-- Task status -->
      {#if taskId}
        <div class="rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs">
          <div class="text-zinc-500">Task: <span class="text-zinc-300 font-mono">{taskId.slice(0, 8)}…</span></div>
          <div class="mt-1 text-zinc-500">Status: <span class="text-zinc-200">{taskStatus}</span></div>
        </div>
      {/if}

      <!-- Results: segment timeline -->
      {#if runResult && runResult.length > 0}
        <div class="space-y-3">
          <div class="text-xs font-medium text-zinc-300">{runResult.length} segments</div>

          <!-- SVG timeline -->
          <svg viewBox="0 0 600 32" class="w-full rounded-xl overflow-hidden" style="height:32px">
            {#each buildTimeline(runResult) as bar}
              <rect
                x={bar.x}
                y={0}
                width={bar.w}
                height={32}
                fill={bar.color}
                opacity={0.8}
              />
              {#if bar.w > 24}
                <text
                  x={bar.x + bar.w / 2}
                  y={20}
                  text-anchor="middle"
                  font-size="10"
                  fill="white"
                  font-weight="600"
                >{bar.label}</text>
              {/if}
            {/each}
          </svg>

          <!-- Segment list -->
          <div class="space-y-1 max-h-48 overflow-y-auto">
            {#each runResult as seg}
              <div class="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1 text-xs">
                <span class="font-medium text-zinc-200">{seg.label}</span>
                <span class="text-zinc-500">{seg.start}s – {seg.end}s</span>
                {#if seg.section_type}
                  <span class="text-indigo-400">{seg.section_type}</span>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
