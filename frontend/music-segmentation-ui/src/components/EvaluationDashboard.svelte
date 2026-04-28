<script>
  import { onMount } from "svelte";
  import {
    listDatasets,
    listDatasetTracks,
    getDatasetTrack,
    listAlgorithms,
    runEvaluation,
    compareAlgorithms,
    getEvaluationsForTrack,
    getSegmentationsForTrack,
    subscribeToTask,
    uploadSegmentation,
    testAlgorithm,
    getSongStreamUrl,
  } from "../lib/api.js";

  // ── State ────────────────────────────────────────────────────────────────
  let datasets = [];
  let selectedDatasetId = "";
  let tracks = [];
  let selectedTrack = null;

  let algorithms = [];
  let selectedAlgoIds = new Set();

  // Built-in algorithm names (always available for comparison)
  const BUILTIN_ALGOS = [
    { id: "custom", name: "custom (built-in)" },
    { id: "foote", name: "foote (built-in)" },
    { id: "cnmf", name: "cnmf (built-in)" },
    { id: "scluster", name: "scluster (built-in)" },
  ];

  let toleranceSeconds = 3;

  let isRunning = false;
  let runError = "";
  let comparisonResults = null;  // { algo_name: { metrics: {...} } }

  let pastEvals = {};   // { algo_name: [ {...} ] }
  let segmentationHistory = [];
  let selectedHistoryItem = null;
  let timelineZoom = 1.25;
  let historyScrollEl = null;
  let historyDragState = null;

  const TIMELINE_WIDTH = 720;
  const TIMELINE_LEFT_COL = 140;
  const TIMELINE_ROW_H = 34;
  const TIMELINE_MIN_ZOOM = 0.5;
  const TIMELINE_MAX_ZOOM = 4;

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  onMount(async () => {
    await Promise.all([loadDatasets(), loadAlgorithms()]);
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  async function loadDatasets() {
    try { datasets = await listDatasets(); } catch (e) { console.error(e); }
  }

  async function loadAlgorithms() {
    try { algorithms = await listAlgorithms(); } catch (e) { console.error(e); }
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
      const taskIds = {};
      let sharedBuiltInAudioFile = null;

      const hasBuiltIn = Array.from(selectedAlgoIds).some((algoId) =>
        BUILTIN_ALGOS.some((algo) => algo.id === algoId)
      );

      if (hasBuiltIn) {
        if (!selectedTrack.audio_url && !selectedTrack.song_id) {
          runError = "Track has no audio source for built-in algorithms.";
          isRunning = false;
          return;
        }

        const sourceUrl = selectedTrack.song_id ? getSongStreamUrl(selectedTrack.song_id) : selectedTrack.audio_url;
        const resp = await fetch(sourceUrl);
        if (!resp.ok) throw new Error(`Failed to fetch audio: ${resp.status} ${resp.statusText}`);
        const blob = await resp.blob();
        sharedBuiltInAudioFile = new File(
          [blob],
          `${selectedTrack.song_id || selectedTrack.track_id || "track"}.mp3`,
          { type: resp.headers.get("content-type") || "audio/mpeg" },
        );
      }

      // Dispatch tasks for each algorithm
      for (const algoId of selectedAlgoIds) {
        const isBuiltin = BUILTIN_ALGOS.some((a) => a.id === algoId);

        if (isBuiltin) {
          const taskId = await uploadSegmentation({ file: sharedBuiltInAudioFile, algorithms: [algoId] });
          taskIds[algoId] = taskId;
        } else {
          // User algorithm — dispatch via test endpoint
          const res = await testAlgorithm(algoId, {
            audioSource: { type: "track_id", value: selectedTrack.track_id },
            params: {},
          });
          taskIds[algoId] = res.task_id;
        }
      }

      // Wait for all tasks to complete via SSE
      /** @type {Record<string, string>} */
      const completedTasks = {};
      await Promise.all(
        Object.entries(taskIds).map(([algoId, taskId]) =>
          new Promise((resolve) => {
            const unsub = subscribeToTask(taskId, (data) => {
              if (data.status === "completed" || data.status === "failed") {
                completedTasks[algoId] = taskId;
                unsub();
                resolve();
              }
            });
          })
        )
      );

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
</script>

<div class="flex h-[calc(100vh-49px)] overflow-hidden text-zinc-100">
  <!-- Left: configuration panel -->
  <aside class="w-72 shrink-0 border-r border-zinc-800 bg-zinc-900/50 flex flex-col overflow-y-auto">
    <div class="p-4 border-b border-zinc-800">
      <span class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Configuration</span>
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
              {a.name}
            </label>
          {/each}
          {#if algorithms.length > 0}
            <p class="text-xs text-zinc-500 pt-1 pl-2">User algorithms:</p>
            {#each algorithms as a}
              <label class="flex items-center gap-2 cursor-pointer rounded-lg px-2 py-1.5 hover:bg-zinc-800 text-sm text-zinc-300">
                <input
                  type="checkbox"
                  class="accent-indigo-500"
                  checked={selectedAlgoIds.has(a.algorithm_id)}
                  on:change={() => toggleAlgo(a.algorithm_id)}
                />
                {a.name} v{a.version}
              </label>
            {/each}
          {/if}
        </div>
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
        on:click={runComparison}
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
      <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-5 py-4">
        <h2 class="text-base font-semibold text-zinc-100">{selectedTrack.title || selectedTrack.song_id}</h2>
        <p class="text-xs text-zinc-400 mt-1">
          Ground truth: {selectedTrack.ground_truth?.length || 0} segments
          {#if selectedTrack.audio_url}
            · <a href={selectedTrack.audio_url} target="_blank" rel="noopener" class="text-indigo-400 hover:underline">audio ↗</a>
          {/if}
        </p>
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
            <div class="flex items-center justify-between">
              <p class="text-xs font-medium text-zinc-400">Segmentation Map</p>
              <p class="text-[10px] text-zinc-500">time axis, segments shown as bars</p>
            </div>

            <div class="overflow-x-auto">
              <svg
                viewBox={`0 0 ${TIMELINE_LEFT_COL + TIMELINE_WIDTH} ${Math.max(160, (Object.keys(comparisonResults).length + 1) * TIMELINE_ROW_H + 28)}`}
                class="w-full rounded-2xl border border-zinc-800 bg-zinc-950/80"
                style="min-width: 860px"
              >
                {#each timelineTicks() as tick}
                  <line x1={TIMELINE_LEFT_COL + tick.x} y1="18" x2={TIMELINE_LEFT_COL + tick.x} y2="100%" stroke="#27272a" stroke-width="1" stroke-dasharray="3 4" />
                  <text x={TIMELINE_LEFT_COL + tick.x} y="14" text-anchor="middle" fill="#71717a" font-size="10">{tick.label}</text>
                {/each}

                <!-- Ground truth row -->
                <text x="16" y={46} fill="#a1a1aa" font-size="11">Ground Truth</text>
                <rect x={TIMELINE_LEFT_COL} y={24} width={TIMELINE_WIDTH} height="18" rx="9" fill="#18181b" stroke="#27272a" />
                {#each segmentRects(selectedTrack.ground_truth || []) as seg}
                  <rect
                    x={TIMELINE_LEFT_COL + seg.x}
                    y="25"
                    width={seg.width}
                    height="16"
                    rx="7"
                    fill="#4f46e5"
                    opacity="0.9"
                  />
                  {#if seg.width > 42}
                    <text x={TIMELINE_LEFT_COL + seg.x + 6} y="37" fill="#f8fafc" font-size="10" font-weight="600">{seg.label}</text>
                  {/if}
                {/each}

                {#each Object.entries(comparisonResults) as [name, result], i}
                  {#if !result.error}
                    <text x="16" y={46 + (i + 1) * TIMELINE_ROW_H} fill="#a1a1aa" font-size="11">{name}</text>
                    <rect x={TIMELINE_LEFT_COL} y={24 + (i + 1) * TIMELINE_ROW_H} width={TIMELINE_WIDTH} height="18" rx="9" fill="#18181b" stroke="#27272a" />
                    {#each segmentRects(result.segments || []) as seg}
                      <rect
                        x={TIMELINE_LEFT_COL + seg.x}
                        y={25 + (i + 1) * TIMELINE_ROW_H}
                        width={seg.width}
                        height="16"
                        rx="7"
                        fill={ROW_COLORS[i % ROW_COLORS.length]}
                        opacity="0.88"
                      />
                      {#if seg.width > 42}
                        <text x={TIMELINE_LEFT_COL + seg.x + 6} y={37 + (i + 1) * TIMELINE_ROW_H} fill="#fafafa" font-size="10" font-weight="600">{seg.label}</text>
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
