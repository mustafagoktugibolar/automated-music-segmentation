<script>
  import { onMount } from "svelte";
  import {
    listDatasets,
    listDatasetTracks,
    getDatasetTrack,
    compareAlgorithms,
    getEvaluationsForTrack,
    getSegmentationsForTrack,
    subscribeToTask,
    uploadSegmentation,
    getSongStreamUrl,
    startBatchEval,
    subscribeToBatchEval,
  } from "../lib/api.js";

  // ── State ────────────────────────────────────────────────────────────────
  let datasets = [];
  let selectedDatasetId = "";
  let tracks = [];
  let selectedTrack = null;

  let selectedAlgoIds = new Set();

  // Built-in algorithm names (always available for comparison)
  const BUILTIN_ALGOS = [
    { id: "custom_librosa", name: "custom_librosa (built-in)", isLLM: false },
    { id: "foote",    name: "foote (built-in)",     isLLM: false },
    { id: "cnmf",     name: "cnmf (built-in)",      isLLM: false },
    { id: "scluster", name: "scluster (built-in)",  isLLM: false },
    { id: "fusion",   name: "fusion (algorithm voting)", isLLM: false },
    { id: "llm",      name: "AI Agent (LLM)",        isLLM: true  },
  ];

  // Confirmation modal for LLM in evaluation
  let showLLMEvalConfirm = false;
  let llmMode = "deterministic";

  let toleranceSeconds = 3;

  let isRunning = false;
  let runError = "";
  let comparisonResults = null;  // { algo_name: { metrics: {...} } }

  let pastEvals = {};   // { algo_name: [ {...} ] }
  let segmentationHistory = [];
  let selectedHistoryItem = null;
  let timelineZoom = 1.25;
  let comparisonScrollEl = null;
  let comparisonDragState = null;
  let historyScrollEl = null;
  let historyDragState = null;

  const TIMELINE_WIDTH = 720;
  const TIMELINE_LEFT_COL = 140;
  const TIMELINE_ROW_H = 34;
  const TIMELINE_MIN_ZOOM = 0.5;
  const TIMELINE_MAX_ZOOM = 4;

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  onMount(async () => {
    await loadDatasets();
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  async function loadDatasets() {
    try { datasets = await listDatasets(); } catch (e) { console.error(e); }
  }

  async function selectDataset(id) {
    selectedDatasetId = id;
    selectedTrack = null;
    tracks = [];
    pastEvals = {};
    comparisonResults = null;
    if (!id) return;
    try {
      const res = await listDatasetTracks(id, { page: 1, pageSize: 200, hasGroundTruth: true });
      tracks = res.tracks || [];
    } catch (e) { console.error(e); }
  }

  async function selectTrack(track) {
    selectedTrack = track;
    comparisonResults = null;
    runError = "";
    try {
      if (selectedDatasetId && track?.track_id) {
        selectedTrack = await getDatasetTrack(selectedDatasetId, track.track_id);
      }
      const res = await getEvaluationsForTrack(track.track_id);
      pastEvals = res.evaluations || {};
      const historyRes = await getSegmentationsForTrack(track.track_id);
      segmentationHistory = historyRes.segmentations || [];
      selectedHistoryItem = segmentationHistory[0] || null;
    } catch (e) { console.error(e); }
  }

  function toggleAlgo(id) {
    const next = new Set(selectedAlgoIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    selectedAlgoIds = next;
  }

  // Run comparison: dispatch tasks for each selected algo, wait for completion, then evaluate
  async function runComparison() {
    if (!selectedTrack) { runError = "Select a track first."; return; }
    if (selectedAlgoIds.size === 0) { runError = "Select at least one algorithm."; return; }
    if (!selectedTrack.has_ground_truth) { runError = "Selected track has no ground truth."; return; }

    isRunning = true;
    runError = "";
    comparisonResults = null;

    try {
      if (!selectedTrack.audio_url && !selectedTrack.song_id) {
        runError = "Track has no audio source for built-in algorithms.";
        isRunning = false;
        return;
      }

      const sourceUrl = selectedTrack.song_id ? getSongStreamUrl(selectedTrack.song_id) : selectedTrack.audio_url;
      const resp = await fetch(sourceUrl);
      if (!resp.ok) throw new Error(`Failed to fetch audio: ${resp.status} ${resp.statusText}`);
      const blob = await resp.blob();
      const sharedBuiltInAudioFile = new File(
        [blob],
        `${selectedTrack.song_id || selectedTrack.track_id || "track"}.mp3`,
        { type: resp.headers.get("content-type") || "audio/mpeg" },
      );

      // Submit a single unified task for all selected algorithms.
      // This avoids uploading the same audio N times and prevents duplicate
      // queue messages (fusion would otherwise re-dispatch all base algorithms).
      const allAlgos = Array.from(selectedAlgoIds);
      const params = allAlgos.includes("llm") ? { llm_segmentation: { mode: llmMode } } : null;
      const unifiedTaskId = await uploadSegmentation({
        file: sharedBuiltInAudioFile,
        algorithms: allAlgos,
        params,
      });

      // All algorithms share the same task — the backend stores each
      // algorithm's result under its own key in task.results.
      /** @type {Record<string, string>} */
      const completedTasks = {};
      for (const algoId of selectedAlgoIds) {
        completedTasks[algoId] = unifiedTaskId;
      }

      // Wait for the unified task to reach a terminal state via SSE.
      await new Promise((resolve) => {
        const unsub = subscribeToTask(unifiedTaskId, (data) => {
          if (data.status === "completed" || data.status === "failed") {
            unsub();
            resolve();
          }
        });
      });

      // Run evaluation comparison
      const res = await compareAlgorithms({
        trackId: selectedTrack.track_id,
        algorithmNames: Array.from(selectedAlgoIds),
        taskIds: completedTasks,
        toleranceSeconds,
      });

      comparisonResults = res.comparison;

      // Refresh past evals
      const evRes = await getEvaluationsForTrack(selectedTrack.track_id);
      pastEvals = evRes.evaluations || {};
      const historyRes = await getSegmentationsForTrack(selectedTrack.track_id);
      segmentationHistory = historyRes.segmentations || [];
      selectedHistoryItem = segmentationHistory[0] || selectedHistoryItem;

    } catch (e) {
      runError = e.message;
    } finally {
      isRunning = false;
    }
  }

  // ── SVG timeline for boundary visualisation ──────────────────────────────
  function boundariesToSvgTicks(segments, width = 560) {
    if (!segments || segments.length === 0) return [];
    const totalDuration = Math.max(...segments.map((s) => s.end));
    return segments
      .map((s) => s.start)
      .filter((t) => t > 0)
      .map((t) => ({ x: (t / totalDuration) * width }));
  }

  function gtTicks() {
    return boundariesToSvgTicks(selectedTrack?.ground_truth);
  }

  function algoTicks(segments) {
    return boundariesToSvgTicks(segments);
  }

  // Colours per algorithm row
  const ROW_COLORS = [
    "#818cf8", "#34d399", "#fb923c", "#f472b6",
    "#60a5fa", "#a78bfa", "#facc15", "#6ee7b7",
  ];

  function fmtPct(v) {
    if (v == null) return "—";
    return (v * 100).toFixed(1) + "%";
  }

  function fmtF1Color(v) {
    if (v == null) return "text-zinc-400";
    if (v >= 0.7) return "text-emerald-300";
    if (v >= 0.5) return "text-amber-300";
    return "text-red-300";
  }

  function maxTimelineDuration() {
    const allSegments = [
      ...(selectedTrack?.ground_truth || []),
      ...Object.values(comparisonResults || {}).flatMap((result) => result?.segments || []),
    ];

    const fromSegments = allSegments.reduce((max, seg) => Math.max(max, Number(seg?.end || 0)), 0);
    return Math.max(fromSegments, 1);
  }

  function xForTime(t) {
    return (Number(t) / maxTimelineDuration()) * timelineWidth();
  }

  function timelineWidth() {
    return TIMELINE_WIDTH * timelineZoom;
  }

  function clampZoom(value) {
    return Math.max(TIMELINE_MIN_ZOOM, Math.min(TIMELINE_MAX_ZOOM, value));
  }

  function zoomIn() {
    timelineZoom = clampZoom(Number((timelineZoom + 0.25).toFixed(2)));
  }

  function zoomOut() {
    timelineZoom = clampZoom(Number((timelineZoom - 0.25).toFixed(2)));
  }

  function resetZoom() {
    timelineZoom = 1.25;
  }

  function segmentRects(segments) {
    return (segments || [])
      .filter((seg) => seg && Number.isFinite(Number(seg.start)) && Number.isFinite(Number(seg.end)))
      .map((seg) => ({
        x: xForTime(seg.start),
        width: Math.max(1, xForTime(seg.end) - xForTime(seg.start)),
        label: seg.section_type || seg.label || "—",
      }));
  }

  function timelineTicks() {
    const duration = maxTimelineDuration();
    const tickCount = 6;
    return Array.from({ length: tickCount + 1 }, (_, i) => {
      const t = (duration / tickCount) * i;
      return { x: xForTime(t), label: `${t.toFixed(0)}s` };
    });
  }

  function historyMaxDuration() {
    const allSegments = [
      ...(selectedTrack?.ground_truth || []),
      ...(selectedHistoryItem?.segments || []),
    ];
    return Math.max(allSegments.reduce((max, seg) => Math.max(max, Number(seg?.end || 0)), 0), 1);
  }

  function historyXForTime(t) {
    return (Number(t) / historyMaxDuration()) * historyTimelineWidth();
  }

  function historyTimelineWidth() {
    return TIMELINE_WIDTH * timelineZoom;
  }

  function setTimelineZoom(value) {
    timelineZoom = clampZoom(Number(value));
  }

  function resetTimelineZoom() {
    timelineZoom = 1.25;
  }

  function handleHistoryWheel(event) {
    if (!historyScrollEl) return;
    event.preventDefault();
    const direction = event.deltaY > 0 ? -0.12 : 0.12;
    timelineZoom = clampZoom(Number((timelineZoom + direction).toFixed(2)));
  }

  function handleComparisonWheel(event) {
    if (!comparisonScrollEl) return;
    if (event.metaKey || event.ctrlKey) {
      event.preventDefault();
      const direction = event.deltaY > 0 ? -0.12 : 0.12;
      timelineZoom = clampZoom(Number((timelineZoom + direction).toFixed(2)));
      return;
    }
    if (Math.abs(event.deltaY) > Math.abs(event.deltaX)) {
      comparisonScrollEl.scrollLeft += event.deltaY;
    }
  }

  function handleComparisonPointerDown(event) {
    if (!comparisonScrollEl) return;
    comparisonDragState = {
      startX: event.clientX,
      startScrollLeft: comparisonScrollEl.scrollLeft,
    };
    comparisonScrollEl.setPointerCapture?.(event.pointerId);
  }

  function handleComparisonPointerMove(event) {
    if (!comparisonScrollEl || !comparisonDragState) return;
    const deltaX = event.clientX - comparisonDragState.startX;
    comparisonScrollEl.scrollLeft = comparisonDragState.startScrollLeft - deltaX;
  }

  function endComparisonDrag() {
    comparisonDragState = null;
  }

  function handleHistoryPointerDown(event) {
    if (!historyScrollEl) return;
    historyDragState = {
      startX: event.clientX,
      startScrollLeft: historyScrollEl.scrollLeft,
    };
    historyScrollEl.setPointerCapture?.(event.pointerId);
  }

  function handleHistoryPointerMove(event) {
    if (!historyScrollEl || !historyDragState) return;
    const deltaX = event.clientX - historyDragState.startX;
    historyScrollEl.scrollLeft = historyDragState.startScrollLeft - deltaX;
  }

  function endHistoryDrag() {
    historyDragState = null;
  }

  function historySegmentRects(segments) {
    return (segments || [])
      .filter((seg) => seg && Number.isFinite(Number(seg.start)) && Number.isFinite(Number(seg.end)))
      .map((seg) => ({
        x: historyXForTime(seg.start),
        width: Math.max(1, historyXForTime(seg.end) - historyXForTime(seg.start)),
        label: seg.section_type || seg.label || "—",
      }));
  }

  function historyTicks() {
    const duration = historyMaxDuration();
    const tickCount = 6;
    return Array.from({ length: tickCount + 1 }, (_, i) => {
      const t = (duration / tickCount) * i;
      return { x: historyXForTime(t), label: `${t.toFixed(0)}s` };
    });
  }

  // ── Batch Eval ────────────────────────────────────────────────────────────
  let batchOpen        = false;
  let batchMaxTracks   = 20;
  let batchRunAllDataset = false;
  let batchConcurrency = 3;
  let batchTolerance   = 0.5;
  let batchRunning     = false;
  let batchLines       = [];
  let batchSummary     = null;
  let batchRows        = [];
  let batchError       = null;
  let batchLogEl       = null;
  let batchUnsub       = null;

  function openBatchPanel() {
    batchOpen    = true;
    batchLines   = [];
    batchSummary = null;
    batchRows    = [];
    batchError   = null;
  }

  async function runBatchEval() {
    if (batchRunning) return;
    batchRunning = true;
    batchLines   = [];
    batchSummary = null;
    batchRows    = [];
    batchError   = null;

    try {
      const { job_id } = await startBatchEval({
        maxTracks: batchRunAllDataset ? 0 : Number(batchMaxTracks),
        toleranceSeconds: Number(batchTolerance),
        concurrency: Number(batchConcurrency),
      });

      batchUnsub = subscribeToBatchEval(
        job_id,
        (line) => {
          batchLines = [...batchLines, line];
          // auto-scroll log
          if (batchLogEl) setTimeout(() => { batchLogEl.scrollTop = batchLogEl.scrollHeight; }, 0);
        },
        ({ summary, rows, error }) => {
          batchSummary = summary;
          batchRows    = rows ?? [];
          batchError   = error;
          batchRunning = false;
        },
      );
    } catch (e) {
      batchError   = e.message;
      batchRunning = false;
    }
  }

  function closeBatchPanel() {
    if (batchUnsub) { batchUnsub(); batchUnsub = null; }
    batchOpen = false;
  }

  function copyBatchReport() {
    const text = batchLines.join("\n") + (batchSummary ? "\n\n" + batchSummary : "");
    navigator.clipboard.writeText(text).catch(() => {});
    copied = true;
    setTimeout(() => { copied = false; }, 2000);
  }

  // ── Copy all results ──────────────────────────────────────────────────────
  let copied = false;

  function copyAllResults() {
    const lines = [];

    if (selectedTrack) {
      lines.push(`Track: ${selectedTrack.title || selectedTrack.song_id}`);
      lines.push(`Ground Truth: ${selectedTrack.ground_truth?.length || 0} segments`);
      lines.push('');
    }

    if (Object.keys(pastEvals).length > 0) {
      lines.push('=== Evaluation History ===');
      lines.push('Algorithm\tTolerance\tPrecision\tRecall\tF1\tDate');
      for (const [algoName, runs] of Object.entries(pastEvals)) {
        for (const run of runs) {
          lines.push([
            algoName,
            `±${run.tolerance_seconds}s`,
            fmtPct(run.metrics?.precision),
            fmtPct(run.metrics?.recall),
            fmtPct(run.metrics?.f_measure),
            run.created_at ? new Date(run.created_at).toLocaleDateString() : '—',
          ].join('\t'));
        }
      }
      lines.push('');
    }

    if (comparisonResults) {
      lines.push(`=== Comparison Results (tolerance ±${toleranceSeconds}s) ===`);
      lines.push('Algorithm\tPrecision\tRecall\tF1\t#Boundaries');
      for (const [name, result] of Object.entries(comparisonResults)) {
        if (result.error) {
          lines.push(`${name}\tERROR: ${result.error}`);
        } else {
          lines.push([
            name,
            fmtPct(result.metrics?.precision),
            fmtPct(result.metrics?.recall),
            fmtPct(result.metrics?.f_measure),
            `${result.metrics?.n_boundaries_est ?? '—'} / ${result.metrics?.n_boundaries_ref ?? '—'} ref`,
          ].join('\t'));
        }
      }
      lines.push('');
    }

    if (selectedHistoryItem) {
      lines.push('=== Selected Segmentation Run ===');
      lines.push(`Algorithm: ${selectedHistoryItem.algorithm_name}`);
      lines.push(`Task: ${selectedHistoryItem.task_id || '—'} · ${selectedHistoryItem.task_status || 'unknown'}`);
      if (selectedHistoryItem.metrics) {
        lines.push(`Precision: ${fmtPct(selectedHistoryItem.metrics?.precision)}`);
        lines.push(`Recall: ${fmtPct(selectedHistoryItem.metrics?.recall)}`);
        lines.push(`F1: ${fmtPct(selectedHistoryItem.metrics?.f_measure)}`);
        lines.push(`Tolerance: ±${selectedHistoryItem.tolerance_seconds}s`);
      }
      if (selectedHistoryItem.segments?.length > 0) {
        lines.push('');
        lines.push('Segments:');
        lines.push('Start\tEnd\tLabel\tType');
        for (const seg of selectedHistoryItem.segments) {
          lines.push(`${seg.start}s\t${seg.end}s\t${seg.label || '—'}\t${seg.section_type || '—'}`);
        }
      }
    }

    const text = lines.join('\n');
    navigator.clipboard.writeText(text).catch(() => {
      const el = document.createElement('textarea');
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
    });
    copied = true;
    setTimeout(() => { copied = false; }, 2000);
  }
</script>

<div class="flex h-[calc(100vh-49px)] overflow-hidden text-zinc-100">
  <!-- Left: configuration panel -->
  <aside class="w-72 shrink-0 border-r border-zinc-800 bg-zinc-900/50 flex flex-col overflow-y-auto">
    <div class="p-4 border-b border-zinc-800 flex items-center justify-between">
      <span class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Configuration</span>
      <button
        on:click={openBatchPanel}
        class="rounded-lg bg-indigo-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-indigo-500"
      >Batch Eval</button>
    </div>

    <div class="p-4 space-y-5">
      <!-- Dataset selector -->
      <div>
        <label class="text-xs font-medium text-zinc-400 block mb-1" for="eval-ds">Dataset</label>
        <select
          id="eval-ds"
          class="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
          bind:value={selectedDatasetId}
          on:change={(e) => selectDataset(e.currentTarget.value)}
        >
          <option value="">— select dataset —</option>
          {#each datasets as ds}
            <option value={ds.dataset_id}>{ds.name}</option>
          {/each}
        </select>
      </div>

      <!-- Track selector (with GT only) -->
      {#if tracks.length > 0}
        <div>
          <label class="text-xs font-medium text-zinc-400 block mb-1" for="eval-track">Track (with ground truth)</label>
          <select
            id="eval-track"
            class="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
            on:change={(e) => {
              const t = tracks.find((tr) => tr.track_id === e.currentTarget.value);
              if (t) selectTrack(t);
            }}
          >
            <option value="">— select track —</option>
            {#each tracks as t}
              <option value={t.track_id}>{t.title || t.song_id}</option>
            {/each}
          </select>
        </div>
      {:else if selectedDatasetId}
        <p class="text-xs text-zinc-500">No tracks with ground truth in this dataset. Run <strong>Import SALAMI</strong> first.</p>
      {/if}

      <!-- Algorithm multi-select -->
      <div>
        <p class="text-xs font-medium text-zinc-400 mb-2">Algorithms</p>
        <div class="space-y-1">
          {#each BUILTIN_ALGOS as a}
            <label class="flex items-center gap-2 cursor-pointer rounded-lg px-2 py-1.5 hover:bg-zinc-800 text-sm text-zinc-300">
              <input
                type="checkbox"
                class="accent-indigo-500"
                checked={selectedAlgoIds.has(a.id)}
                on:change={() => toggleAlgo(a.id)}
              />
              <span class="flex-1">{a.name}</span>
              {#if a.isLLM}
                <span class="rounded-full border border-amber-800/50 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-semibold text-amber-400">LLM</span>
              {/if}
            </label>
          {/each}
        </div>

        {#if selectedAlgoIds.has("llm")}
          <div class="mt-3 rounded-2xl border border-zinc-800 bg-zinc-950/60 px-3 py-3">
            <label class="text-[10px] font-medium uppercase tracking-wider text-zinc-500" for="eval-llm-mode">
              AI Agent mode
            </label>
            <select
              id="eval-llm-mode"
              bind:value={llmMode}
              class="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
            >
              <option value="deterministic">Deterministic</option>
              <option value="ai_generated">AI generated</option>
            </select>
            <p class="mt-2 text-[10px] text-zinc-500">Applies only to the AI Agent algorithm.</p>
          </div>
        {/if}
      </div>

      <!-- Tolerance slider -->
      <div>
        <label class="text-xs font-medium text-zinc-400 block mb-1" for="tolerance-slider">
          Tolerance window: <span class="text-zinc-200">{toleranceSeconds}s</span>
        </label>
        <input
          id="tolerance-slider"
          type="range"
          min="0.5"
          max="10"
          step="0.5"
          class="w-full accent-indigo-500"
          bind:value={toleranceSeconds}
        />
        <div class="flex justify-between text-[10px] text-zinc-600 mt-1">
          <span>0.5s</span><span>10s</span>
        </div>
      </div>

      <!-- Run button -->
      <button
        class="w-full rounded-2xl bg-indigo-500 py-2.5 text-sm font-semibold text-white hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed"
        on:click={() => selectedAlgoIds.has("llm") ? (showLLMEvalConfirm = true) : runComparison()}
        disabled={isRunning || !selectedTrack}
      >
        {#if isRunning}
          <span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/80 mr-2"></span>
          Running…
        {:else}
          ▶ Run Comparison
        {/if}
      </button>

      {#if runError}
        <div class="rounded-xl border border-red-900/60 bg-red-950/30 px-3 py-2 text-xs text-red-300">
          {runError}
        </div>
      {/if}
    </div>
  </aside>

  <!-- ── LLM eval confirmation modal ── -->
  {#if showLLMEvalConfirm}
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div class="w-full max-w-sm rounded-3xl border border-amber-800/50 bg-zinc-950 p-6 shadow-2xl shadow-black/40">
        <div class="flex items-start gap-3">
          <div class="shrink-0 rounded-xl border border-amber-800/40 bg-amber-500/10 p-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-amber-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
          </div>
          <div>
            <h3 class="text-sm font-semibold text-zinc-100">AI Agent selected</h3>
            <p class="mt-1.5 text-xs text-zinc-400 leading-relaxed">
              <strong class="text-zinc-200">AI Agent (LLM)</strong> has been included in the comparison.
              An <strong class="text-amber-300">LLM API call</strong> will be made for this track.
            </p>
          </div>
        </div>
        <div class="mt-5 flex gap-2">
          <button
            class="flex-1 rounded-2xl border border-zinc-800 bg-zinc-900 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800"
            on:click={() => (showLLMEvalConfirm = false)}
          >
            Cancel
          </button>
          <button
            class="flex-1 rounded-2xl bg-indigo-500 py-2 text-sm font-semibold text-white hover:bg-indigo-400"
            on:click={() => { showLLMEvalConfirm = false; runComparison(); }}
          >
            Yes, continue
          </button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Batch Eval sliding panel -->
  {#if batchOpen}
    <div class="fixed inset-0 z-40 flex" role="dialog" aria-modal="true">
      <!-- backdrop -->
      <button class="absolute inset-0 bg-black/60" on:click={closeBatchPanel} aria-label="Close batch eval"></button>

      <!-- panel -->
      <div class="relative ml-auto w-full max-w-2xl h-full bg-zinc-950 border-l border-zinc-800 flex flex-col shadow-2xl z-50">
        <!-- header -->
        <div class="flex items-center justify-between px-5 py-4 border-b border-zinc-800 shrink-0">
          <div>
            <h2 class="text-base font-semibold text-zinc-100">Batch Evaluation</h2>
            <p class="text-xs text-zinc-500 mt-0.5">Run selected algorithms on multiple SALAMI tracks</p>
          </div>
          <button on:click={closeBatchPanel} class="text-zinc-500 hover:text-zinc-200 text-xl leading-none">✕</button>
        </div>

        <!-- config -->
        <div class="px-5 py-4 border-b border-zinc-800 shrink-0 space-y-4">
          <div class="flex flex-wrap items-end gap-4">
            <div>
              <label for="batch-max-tracks" class="text-xs font-medium text-zinc-400 block mb-1">Dataset scope</label>
              <div class="mb-2 grid w-48 grid-cols-2 gap-1 rounded-lg border border-zinc-800 bg-zinc-950 p-1">
                <button
                  type="button"
                  on:click={() => (batchRunAllDataset = false)}
                  disabled={batchRunning}
                  class={"rounded-md px-2 py-1 text-[10px] font-semibold transition-colors disabled:opacity-50 " +
                    (!batchRunAllDataset ? "bg-zinc-800 text-zinc-100" : "text-zinc-500 hover:text-zinc-300")}
                >
                  Limited
                </button>
                <button
                  type="button"
                  on:click={() => (batchRunAllDataset = true)}
                  disabled={batchRunning}
                  class={"rounded-md px-2 py-1 text-[10px] font-semibold transition-colors disabled:opacity-50 " +
                    (batchRunAllDataset ? "bg-indigo-500 text-white" : "text-zinc-500 hover:text-zinc-300")}
                >
                  All dataset
                </button>
              </div>
              <input
                id="batch-max-tracks"
                type="number"
                min="1" max="500"
                bind:value={batchMaxTracks}
                disabled={batchRunning || batchRunAllDataset}
                class="w-24 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none disabled:opacity-50"
              />
            </div>
            <div>
              <label for="batch-concurrency" class="text-xs font-medium text-zinc-400 block mb-1">Concurrency</label>
              <input
                id="batch-concurrency"
                type="number"
                min="1" max="10"
                bind:value={batchConcurrency}
                disabled={batchRunning}
                class="w-24 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none disabled:opacity-50"
              />
              <div class="mt-1.5 flex gap-1">
                {#each [2, 3, 4] as preset}
                  <button
                    type="button"
                    on:click={() => (batchConcurrency = preset)}
                    disabled={batchRunning}
                    class="rounded-md border border-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:border-zinc-700 hover:text-zinc-300 disabled:opacity-50"
                  >
                    {preset}
                  </button>
                {/each}
              </div>
            </div>
            <div>
              <label for="batch-tolerance" class="text-xs font-medium text-zinc-400 block mb-1">
                Tolerance: <span class="text-zinc-200">±{batchTolerance}s</span>
              </label>
              <input
                id="batch-tolerance"
                type="range" min="0.5" max="3" step="0.5"
                bind:value={batchTolerance}
                disabled={batchRunning}
                class="w-40 accent-indigo-500"
              />
            </div>
            <button
              on:click={runBatchEval}
              disabled={batchRunning}
              class="ml-auto rounded-xl bg-indigo-500 px-5 py-2 text-sm font-semibold text-white hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {#if batchRunning}
                <span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/80"></span>
                Running…
              {:else if batchRunAllDataset}
                Run All Dataset
              {:else}
                ▶ Run
              {/if}
            </button>
          </div>
        </div>

        <!-- log -->
        <div
          bind:this={batchLogEl}
          class="flex-1 overflow-y-auto p-4 font-mono text-[11px] text-zinc-300 bg-zinc-950 space-y-0.5"
        >
          {#if batchLines.length === 0 && !batchRunning}
            <p class="text-zinc-600">Configure and press Run to start batch evaluation.</p>
          {/if}
          {#each batchLines as line}
            <div class={line.startsWith('  P=') ? 'text-emerald-400' : line.startsWith('  skip') || line.startsWith('FATAL') ? 'text-red-400' : line.startsWith('[') ? 'text-zinc-200' : 'text-zinc-400'}>
              {line || ' '}
            </div>
          {/each}
          {#if batchError && !batchRunning}
            <div class="text-red-400 mt-2">Error: {batchError}</div>
          {/if}
        </div>

        <!-- summary table (when done) -->
        {#if batchRows.length > 0 && !batchRunning}
          <div class="border-t border-zinc-800 shrink-0">
            <div class="px-5 py-3 flex items-center justify-between">
              <h3 class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Results</h3>
              <button
                on:click={copyBatchReport}
                class="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800/60 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-700"
              >
                {#if copied}
                  <span class="text-emerald-400">Copied!</span>
                {:else}
                  Copy report
                {/if}
              </button>
            </div>
            <div class="overflow-x-auto max-h-52">
              <table class="w-full text-xs">
                <thead class="border-b border-zinc-800 sticky top-0 bg-zinc-950">
                  <tr>
                    <th class="px-4 py-2 text-left text-zinc-400 font-medium">ID</th>
                    <th class="px-4 py-2 text-left text-zinc-400 font-medium">Title</th>
                    <th class="px-4 py-2 text-right text-zinc-400 font-medium">P</th>
                    <th class="px-4 py-2 text-right text-zinc-400 font-medium">R</th>
                    <th class="px-4 py-2 text-right text-zinc-400 font-medium">F1</th>
                    <th class="px-4 py-2 text-right text-zinc-400 font-medium">est/ref</th>
                  </tr>
                </thead>
                <tbody>
                  {#each batchRows.filter(r => !r.error) as row}
                    <tr class="border-b border-zinc-800/40">
                      <td class="px-4 py-1.5 text-zinc-500">{row.song_id}</td>
                      <td class="px-4 py-1.5 text-zinc-300 max-w-[160px] truncate">{row.title}</td>
                      <td class="px-4 py-1.5 text-right text-zinc-300">{fmtPct(row.precision)}</td>
                      <td class="px-4 py-1.5 text-right text-zinc-300">{fmtPct(row.recall)}</td>
                      <td class={"px-4 py-1.5 text-right font-bold " + fmtF1Color(row.f_measure)}>{fmtPct(row.f_measure)}</td>
                      <td class="px-4 py-1.5 text-right text-zinc-500">{row.n_est}/{row.n_ref}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Right: results -->
  <div class="flex-1 overflow-y-auto p-6 space-y-6">
    {#if !selectedTrack}
      <div class="flex min-h-[calc(100vh-120px)] items-center justify-center">
        <div class="w-full max-w-3xl rounded-3xl border border-zinc-800 bg-zinc-900/50 p-6 shadow-2xl shadow-black/20">
          <div class="flex items-start justify-between gap-6">
            <div>
              <p class="text-xs font-semibold uppercase tracking-wider text-zinc-500">Evaluation Dashboard</p>
              <h2 class="mt-2 text-2xl font-semibold text-zinc-100">Pick a dataset and track</h2>
              <p class="mt-2 max-w-xl text-sm text-zinc-400">
                Once you choose a SALAMI track with ground truth, this screen will show past
                segmentation runs, their metrics, and the timeline comparison panel.
              </p>
            </div>
            <div class="rounded-2xl border border-zinc-800 bg-zinc-950/80 px-4 py-3 text-right">
              <div class="text-[10px] uppercase tracking-widest text-zinc-500">Status</div>
              <div class="mt-1 text-sm font-medium text-zinc-200">Waiting for selection</div>
            </div>
          </div>

          <div class="mt-6 grid gap-4 md:grid-cols-2">
            <div class="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
              <div class="mb-3 flex items-center gap-2 text-sm font-medium text-zinc-200">
                <span class="h-2 w-2 rounded-full bg-indigo-500"></span>
                What you will see
              </div>
              <ul class="space-y-2 text-sm text-zinc-400">
                <li>• Comparison metrics for each algorithm</li>
                <li>• Ground truth vs segmentation timeline</li>
                <li>• Saved segmentation history for the track</li>
              </ul>
            </div>

            <div class="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
              <div class="mb-3 text-sm font-medium text-zinc-200">Preview layout</div>
              <div class="space-y-2">
                <div class="h-3 w-full rounded-full bg-zinc-800"></div>
                <div class="h-3 w-5/6 rounded-full bg-zinc-800"></div>
                <div class="h-3 w-2/3 rounded-full bg-zinc-800"></div>
                <div class="mt-4 h-20 rounded-2xl border border-dashed border-zinc-700 bg-zinc-900/60"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    {:else}
      <!-- Track header -->
      <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-5 py-4 flex items-center justify-between gap-4">
        <div>
          <h2 class="text-base font-semibold text-zinc-100">{selectedTrack.title || selectedTrack.song_id}</h2>
          <p class="text-xs text-zinc-400 mt-1">
            Ground truth: {selectedTrack.ground_truth?.length || 0} segments
            {#if selectedTrack.audio_url}
              · <a href={selectedTrack.audio_url} target="_blank" rel="noopener" class="text-indigo-400 hover:underline">audio ↗</a>
            {/if}
          </p>
        </div>
        <button
          on:click={copyAllResults}
          class="shrink-0 flex items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-800/60 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
          title="Copy all results to clipboard"
        >
          {#if copied}
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5 text-emerald-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 00-1.414 0L8 12.586 4.707 9.293a1 1 0 00-1.414 1.414l4 4a1 1 0 001.414 0l8-8a1 1 0 000-1.414z" clip-rule="evenodd" />
            </svg>
            <span class="text-emerald-400">Copied!</span>
          {:else}
            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
              <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
            </svg>
            Copy Results
          {/if}
        </button>
      </div>

      <!-- Ground truth preview -->
      <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div class="px-5 py-3 border-b border-zinc-800 flex items-center justify-between">
          <h3 class="text-sm font-semibold text-zinc-200">Ground Truth Preview</h3>
          <span class="text-xs text-zinc-400">{selectedTrack.ground_truth?.length || 0} segments</span>
        </div>

        <div class="px-5 py-4 overflow-x-auto">
          {#if selectedTrack.ground_truth?.length > 0}
            <svg
              viewBox={`0 0 ${TIMELINE_LEFT_COL + TIMELINE_WIDTH} 60`}
              class="w-full rounded-2xl border border-zinc-800 bg-zinc-950/80"
              style="min-width: 860px"
            >
              {#each timelineTicks() as tick}
                <line x1={TIMELINE_LEFT_COL + tick.x} y1="18" x2={TIMELINE_LEFT_COL + tick.x} y2="60" stroke="#27272a" stroke-width="1" stroke-dasharray="3 4" />
                <text x={TIMELINE_LEFT_COL + tick.x} y="14" text-anchor="middle" fill="#71717a" font-size="10">{tick.label}</text>
              {/each}

              <text x="16" y="40" fill="#a1a1aa" font-size="11">Ground Truth</text>
              <rect x={TIMELINE_LEFT_COL} y="18" width={TIMELINE_WIDTH} height="18" rx="9" fill="#18181b" stroke="#27272a" />
              {#each segmentRects(selectedTrack.ground_truth || []) as seg}
                <rect x={TIMELINE_LEFT_COL + seg.x} y="19" width={seg.width} height="16" rx="7" fill="#4f46e5" opacity="0.95" />
                {#if seg.width > 42}
                  <text x={TIMELINE_LEFT_COL + seg.x + 6} y="31" fill="#f8fafc" font-size="10" font-weight="600">{seg.label}</text>
                {/if}
              {/each}
            </svg>
          {:else}
            <div class="rounded-2xl border border-dashed border-zinc-800 bg-zinc-950/60 p-4 text-sm text-zinc-500">
              Ground truth is not loaded for this track.
            </div>
          {/if}
        </div>
      </div>

      <!-- Current comparison results -->
      {#if comparisonResults}
        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
          <div class="px-5 py-3 border-b border-zinc-800 flex items-center justify-between">
            <h3 class="text-sm font-semibold text-zinc-200">Comparison Results</h3>
            <span class="text-xs text-zinc-400">tolerance ±{toleranceSeconds}s</span>
          </div>

          <!-- Metrics table -->
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead class="border-b border-zinc-800">
                <tr>
                  <th class="px-5 py-2 text-left text-xs font-medium text-zinc-400">Algorithm</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">Precision</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">Recall</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400 font-bold">F1</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">#Boundaries</th>
                </tr>
              </thead>
              <tbody>
                {#each Object.entries(comparisonResults) as [name, result], i}
                  <tr class="border-b border-zinc-800/50">
                    <td class="px-5 py-2.5">
                      <div class="flex items-center gap-2">
                        <div class="h-2.5 w-2.5 rounded-full" style="background: {ROW_COLORS[i % ROW_COLORS.length]}"></div>
                        <span class="text-zinc-200 font-medium">{name}</span>
                      </div>
                    </td>
                    {#if result.error}
                      <td colspan="4" class="px-5 py-2.5 text-xs text-red-400">{result.error}</td>
                    {:else}
                      <td class="px-5 py-2.5 text-right text-zinc-300">{fmtPct(result.metrics?.precision)}</td>
                      <td class="px-5 py-2.5 text-right text-zinc-300">{fmtPct(result.metrics?.recall)}</td>
                      <td class={"px-5 py-2.5 text-right font-bold " + fmtF1Color(result.metrics?.f_measure)}>
                        {fmtPct(result.metrics?.f_measure)}
                      </td>
                      <td class="px-5 py-2.5 text-right text-zinc-400 text-xs">
                        {result.metrics?.n_boundaries_est ?? "—"} / {result.metrics?.n_boundaries_ref ?? "—"} ref
                      </td>
                    {/if}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>

          <!-- Segmentation map -->
          <div class="px-5 py-4 space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p class="text-xs font-medium text-zinc-400">Segmentation Map</p>
                <p class="text-[10px] text-zinc-500">drag to pan, Ctrl/⌘ + wheel to zoom</p>
              </div>
              <div class="flex items-center gap-2">
                <button
                  type="button"
                  class="h-7 w-7 rounded-md border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
                  on:click={zoomOut}
                  disabled={timelineZoom <= TIMELINE_MIN_ZOOM}
                  title="Zoom out"
                >−</button>
                <input
                  class="w-28 accent-indigo-500"
                  type="range"
                  min={TIMELINE_MIN_ZOOM}
                  max={TIMELINE_MAX_ZOOM}
                  step="0.05"
                  value={timelineZoom}
                  on:input={(e) => setTimelineZoom(e.currentTarget.value)}
                  title="Timeline zoom"
                />
                <button
                  type="button"
                  class="h-7 w-7 rounded-md border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
                  on:click={zoomIn}
                  disabled={timelineZoom >= TIMELINE_MAX_ZOOM}
                  title="Zoom in"
                >+</button>
                <button
                  type="button"
                  class="h-7 rounded-md border border-zinc-700 bg-zinc-900 px-2 text-[11px] text-zinc-300 hover:bg-zinc-800"
                  on:click={resetZoom}
                >{timelineZoom.toFixed(2)}x</button>
              </div>
            </div>

            <div
              class={"overflow-x-auto rounded-2xl border border-zinc-800 bg-zinc-950/80 " + (comparisonDragState ? "cursor-grabbing" : "cursor-grab")}
              bind:this={comparisonScrollEl}
              on:wheel={handleComparisonWheel}
              on:pointerdown={handleComparisonPointerDown}
              on:pointermove={handleComparisonPointerMove}
              on:pointerup={endComparisonDrag}
              on:pointercancel={endComparisonDrag}
              on:pointerleave={endComparisonDrag}
            >
              <svg
                viewBox={`0 0 ${TIMELINE_LEFT_COL + timelineWidth()} ${Math.max(180, (Object.keys(comparisonResults).length + 1) * TIMELINE_ROW_H + 42)}`}
                class="block"
                style={`width: ${TIMELINE_LEFT_COL + timelineWidth()}px; min-width: 100%; height: ${Math.max(180, (Object.keys(comparisonResults).length + 1) * TIMELINE_ROW_H + 42)}px`}
              >
                {#each timelineTicks() as tick}
                  <line x1={TIMELINE_LEFT_COL + tick.x} y1="22" x2={TIMELINE_LEFT_COL + tick.x} y2="100%" stroke="#27272a" stroke-width="1" stroke-dasharray="3 4" />
                  <text x={TIMELINE_LEFT_COL + tick.x} y="16" text-anchor="middle" fill="#71717a" font-size="10">{tick.label}</text>
                {/each}

                <!-- Ground truth row -->
                <text x="16" y={49} fill="#a1a1aa" font-size="11" font-weight="600">Ground Truth</text>
                <rect x={TIMELINE_LEFT_COL} y={26} width={timelineWidth()} height="20" rx="7" fill="#18181b" stroke="#27272a" />
                {#each segmentRects(selectedTrack.ground_truth || []) as seg}
                  <rect
                    x={TIMELINE_LEFT_COL + seg.x}
                    y="27"
                    width={seg.width}
                    height="18"
                    rx="6"
                    fill="#4f46e5"
                    opacity="0.9"
                  />
                  {#if seg.width > 34}
                    <text x={TIMELINE_LEFT_COL + seg.x + 6} y="40" fill="#f8fafc" font-size="10" font-weight="600">{seg.label}</text>
                  {/if}
                {/each}

                {#each Object.entries(comparisonResults) as [name, result], i}
                  {#if !result.error}
                    <text x="16" y={49 + (i + 1) * TIMELINE_ROW_H} fill="#a1a1aa" font-size="11" font-weight="600">{name}</text>
                    <rect x={TIMELINE_LEFT_COL} y={26 + (i + 1) * TIMELINE_ROW_H} width={timelineWidth()} height="20" rx="7" fill="#18181b" stroke="#27272a" />
                    {#each segmentRects(result.segments || []) as seg}
                      <rect
                        x={TIMELINE_LEFT_COL + seg.x}
                        y={27 + (i + 1) * TIMELINE_ROW_H}
                        width={seg.width}
                        height="18"
                        rx="6"
                        fill={ROW_COLORS[i % ROW_COLORS.length]}
                        opacity="0.88"
                      />
                      {#if seg.width > 34}
                        <text x={TIMELINE_LEFT_COL + seg.x + 6} y={40 + (i + 1) * TIMELINE_ROW_H} fill="#fafafa" font-size="10" font-weight="600">{seg.label}</text>
                      {/if}
                    {/each}
                  {/if}
                {/each}
              </svg>
            </div>

            <div class="flex items-center gap-4 text-[10px] text-zinc-500">
              <span class="inline-flex items-center gap-1"><span class="h-2 w-2 rounded-full bg-indigo-500"></span> ground truth</span>
              {#each Object.entries(comparisonResults) as [name, result], i}
                {#if !result.error}
                  <span class="inline-flex items-center gap-1"><span class="h-2 w-2 rounded-full" style="background: {ROW_COLORS[i % ROW_COLORS.length]}"></span> {name}</span>
                {/if}
              {/each}
            </div>
          </div>
        </div>
      {/if}

      <!-- Past evaluation history -->
      {#if Object.keys(pastEvals).length > 0}
        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
          <div class="px-5 py-3 border-b border-zinc-800">
            <h3 class="text-sm font-semibold text-zinc-200">Evaluation History</h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead class="border-b border-zinc-800">
                <tr>
                  <th class="px-5 py-2 text-left text-xs font-medium text-zinc-400">Algorithm</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">Tolerance</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">Precision</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">Recall</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">F1</th>
                  <th class="px-5 py-2 text-right text-xs font-medium text-zinc-400">Date</th>
                </tr>
              </thead>
              <tbody>
                {#each Object.entries(pastEvals) as [algoName, runs]}
                  {#each runs as run}
                    <tr class="border-b border-zinc-800/40">
                      <td class="px-5 py-2 text-zinc-300 font-medium">{algoName}</td>
                      <td class="px-5 py-2 text-right text-zinc-400 text-xs">±{run.tolerance_seconds}s</td>
                      <td class="px-5 py-2 text-right text-zinc-300">{fmtPct(run.metrics?.precision)}</td>
                      <td class="px-5 py-2 text-right text-zinc-300">{fmtPct(run.metrics?.recall)}</td>
                      <td class={"px-5 py-2 text-right font-bold " + fmtF1Color(run.metrics?.f_measure)}>
                        {fmtPct(run.metrics?.f_measure)}
                      </td>
                      <td class="px-5 py-2 text-right text-zinc-500 text-xs">
                        {run.created_at ? new Date(run.created_at).toLocaleDateString() : "—"}
                      </td>
                    </tr>
                  {/each}
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {/if}

      <!-- Segmentation history -->
      {#if segmentationHistory.length > 0}
        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
          <div class="px-5 py-3 border-b border-zinc-800 flex items-center justify-between">
            <h3 class="text-sm font-semibold text-zinc-200">Segmentation History</h3>
            <span class="text-xs text-zinc-500">stored runs for this track</span>
          </div>

          <div class="grid gap-0 lg:grid-cols-[300px_1fr]">
            <div class="border-b lg:border-b-0 lg:border-r border-zinc-800 max-h-[420px] overflow-y-auto">
              {#each segmentationHistory as item}
                <button
                  class={'w-full text-left px-4 py-3 border-b border-zinc-800/60 hover:bg-zinc-800/40 ' +
                    (selectedHistoryItem?.eval_id === item.eval_id ? 'bg-indigo-500/10' : '')}
                  on:click={() => (selectedHistoryItem = item)}
                >
                  <div class="flex items-center justify-between gap-3">
                    <div>
                      <div class="text-sm font-medium text-zinc-100">{item.algorithm_name}</div>
                      <div class="text-[11px] text-zinc-500">{item.created_at ? new Date(item.created_at).toLocaleString() : '—'}</div>
                    </div>
                    <div class="text-right text-[11px] text-zinc-400">
                      <div>{fmtPct(item.metrics?.f_measure)}</div>
                      <div>±{item.tolerance_seconds}s</div>
                    </div>
                  </div>
                </button>
              {/each}
            </div>

            <div class="p-4 space-y-3">
              {#if selectedHistoryItem}
                <div class="flex items-center justify-between gap-3">
                  <div>
                    <h4 class="text-sm font-semibold text-zinc-100">{selectedHistoryItem.algorithm_name}</h4>
                    <p class="text-[11px] text-zinc-500">task {selectedHistoryItem.task_id || '—'} · {selectedHistoryItem.task_status || 'unknown'}</p>
                  </div>
                  {#if selectedHistoryItem.metrics}
                    <div class="text-right text-xs text-zinc-300">
                      <div>Precision: {fmtPct(selectedHistoryItem.metrics?.precision)}</div>
                      <div>Recall: {fmtPct(selectedHistoryItem.metrics?.recall)}</div>
                      <div class={fmtF1Color(selectedHistoryItem.metrics?.f_measure)}>F1: {fmtPct(selectedHistoryItem.metrics?.f_measure)}</div>
                    </div>
                  {/if}
                </div>

                <div
                  bind:this={historyScrollEl}
                  class="overflow-x-auto cursor-grab active:cursor-grabbing select-none"
                  on:wheel|preventDefault={handleHistoryWheel}
                  on:pointerdown={handleHistoryPointerDown}
                  on:pointermove={handleHistoryPointerMove}
                  on:pointerup={endHistoryDrag}
                  on:pointercancel={endHistoryDrag}
                  on:pointerleave={endHistoryDrag}
                >
                  <svg
                    viewBox={`0 0 ${TIMELINE_LEFT_COL + historyTimelineWidth()} ${Math.max(120, 3 * TIMELINE_ROW_H)}`}
                    class="rounded-2xl border border-zinc-800 bg-zinc-950/80"
                    style={`min-width: ${TIMELINE_LEFT_COL + historyTimelineWidth()}px`}
                  >
                    {#each historyTicks() as tick}
                      <line x1={TIMELINE_LEFT_COL + tick.x} y1="18" x2={TIMELINE_LEFT_COL + tick.x} y2="100%" stroke="#27272a" stroke-width="1" stroke-dasharray="3 4" />
                      <text x={TIMELINE_LEFT_COL + tick.x} y="14" text-anchor="middle" fill="#71717a" font-size="10">{tick.label}</text>
                    {/each}

                    <text x="16" y="46" fill="#a1a1aa" font-size="11">Ground Truth</text>
                    <rect x={TIMELINE_LEFT_COL} y="24" width={TIMELINE_WIDTH} height="18" rx="9" fill="#18181b" stroke="#27272a" />
                    {#each historySegmentRects(selectedTrack.ground_truth || []) as seg}
                      <rect x={TIMELINE_LEFT_COL + seg.x} y="25" width={seg.width} height="16" rx="7" fill="#4f46e5" opacity="0.9" />
                      {#if seg.width > 42}
                        <text x={TIMELINE_LEFT_COL + seg.x + 6} y="37" fill="#f8fafc" font-size="10" font-weight="600">{seg.label}</text>
                      {/if}
                    {/each}

                    <text x="16" y={46 + TIMELINE_ROW_H} fill="#a1a1aa" font-size="11">{selectedHistoryItem.algorithm_name}</text>
                    <rect x={TIMELINE_LEFT_COL} y={24 + TIMELINE_ROW_H} width={TIMELINE_WIDTH} height="18" rx="9" fill="#18181b" stroke="#27272a" />
                    {#each historySegmentRects(selectedHistoryItem.segments || []) as seg}
                      <rect x={TIMELINE_LEFT_COL + seg.x} y={25 + TIMELINE_ROW_H} width={seg.width} height="16" rx="7" fill={ROW_COLORS[0]} opacity="0.92" />
                      {#if seg.width > 42}
                        <text x={TIMELINE_LEFT_COL + seg.x + 6} y={37 + TIMELINE_ROW_H} fill="#fafafa" font-size="10" font-weight="600">{seg.label}</text>
                      {/if}
                    {/each}
                  </svg>
                </div>
                <div class="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
                  <button class="rounded-lg border border-zinc-700 px-3 py-1.5 hover:bg-zinc-800" on:click={zoomOut}>Zoom -</button>
                  <button class="rounded-lg border border-zinc-700 px-3 py-1.5 hover:bg-zinc-800" on:click={zoomIn}>Zoom +</button>
                  <button class="rounded-lg border border-zinc-700 px-3 py-1.5 hover:bg-zinc-800" on:click={resetTimelineZoom}>Reset</button>
                  <label class="ml-2 flex items-center gap-2">
                    <span>Scale</span>
                    <input
                      type="range"
                      min={TIMELINE_MIN_ZOOM}
                      max={TIMELINE_MAX_ZOOM}
                      step="0.05"
                      value={timelineZoom}
                      on:input={(e) => setTimelineZoom(e.currentTarget.value)}
                      class="w-40 accent-indigo-500"
                    />
                    <span class="w-12 text-right text-zinc-300">{timelineZoom.toFixed(2)}x</span>
                  </label>
                  <span class="text-zinc-500">Use the horizontal scroll to pan</span>
                </div>
              {:else}
                <div class="text-sm text-zinc-500">Select a past run on the left to inspect its segmentation result.</div>
              {/if}
            </div>
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>
