<script>
  import { onMount, onDestroy } from "svelte";
  import * as XLSX from "xlsx";
  import {
    listDatasets,
    createDataset,
    importSalami,
    listDatasetTracks,
    uploadTrack,
    segmentSongFromStorage,
    segmentSongsBatch,
    getSongStreamUrl,
    startBatchEval,
    subscribeToBatchEval,
  } from "../lib/api.js";

  // ── State ────────────────────────────────────────────────────────────────
  let datasets = [];
  let selectedDataset = null;
  let tracks = [];
  let tracksTotal = 0;
  let page = 1;
  const PAGE_SIZE = 476;

  let filterGT = false;
  let isImporting = false;
  let importResult = null;
  let importError = "";

  let isCreating = false;
  let newDatasetName = "";
  let newDatasetDesc = "";
  let createError = "";

  let isUploadingTrack = false;
  let uploadTitle = "";
  let uploadArtist = "";
  let uploadFile = null;
  let uploadGTFile = null;
  let uploadError = "";
  let uploadSuccess = "";

  let selectedTrack = null;
  let isSegmenting = false;
  let segmentMessage = "";
  let player = null;

  // ── Batch Eval panel ──────────────────────────────────────────────────────
  let batchPanelOpen = false;
  let batchRunning = false;
  let batchDone = false;
  let batchJobId = null;
  let batchUnsub = null;
  let batchProgress = { completed: 0, total: 0 };
  let batchLogLines = /** @type {string[]} */ ([]);
  let batchRows = /** @type {any[]} */ ([]);
  let batchSummary = "";
  let batchError = "";
  let batchStartError = "";

  const RE_PROGRESS = /\[\s*(\d+)\/\s*(\d+)\]/;

  $: batchSuccessRows = batchRows.filter(r => !r.error);
  $: batchOutlierCount = batchSuccessRows.filter(r => r.is_outlier).length;
  $: batchIncludedRows = batchSuccessRows.filter(r => !r.is_outlier);
  $: batchAvgF1Raw = avg(batchSuccessRows, "f_measure");
  $: batchAvgF1Adj = avg(batchIncludedRows, "f_measure");
  $: batchSorted = [...batchSuccessRows].sort((a, b) => (b.f_measure ?? 0) - (a.f_measure ?? 0));

  function avg(rows, key) {
    const vals = rows.map(r => r[key]).filter(v => v != null && !isNaN(v));
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  }

  function fmtPct(v) {
    return v == null ? "—" : (v * 100).toFixed(1) + "%";
  }

  async function runBatchEval() {
    if (batchRunning) return;
    batchUnsub?.();
    batchRunning = true;
    batchDone = false;
    batchRows = [];
    batchLogLines = [];
    batchProgress = { completed: 0, total: 0 };
    batchSummary = "";
    batchError = "";
    batchStartError = "";
    try {
      const { job_id } = await startBatchEval({
        maxTracks: 0,
        toleranceSeconds: 0.5,
        tolerances: [0.5, 3.0],
        concurrency: 3,
        coverageOutlierThreshold: 0.20,
      });
      batchJobId = job_id;
      batchUnsub = subscribeToBatchEval(
        job_id,
        (line) => {
          batchLogLines = batchLogLines.length >= 200 ? [...batchLogLines.slice(-199), line] : [...batchLogLines, line];
          const m = RE_PROGRESS.exec(line);
          if (m) batchProgress = { completed: parseInt(m[1]), total: parseInt(m[2]) };
        },
        ({ summary, rows, error }) => {
          batchRows = rows ?? [];
          batchSummary = summary ?? "";
          batchError = error ?? "";
          batchDone = true;
          batchRunning = false;
          batchProgress = { completed: batchRows.filter(r => !r.error).length, total: batchRows.length };
        },
      );
    } catch (e) {
      batchStartError = e.message;
      batchRunning = false;
    }
  }

  function exportBatchExcel() {
    if (!batchRows.length) return;
    const data = [
      ["ID", "Algorithm", "Title", "Precision", "Recall", "F1", "F1@3s", "n_est", "n_ref", "Outlier"],
      ...batchRows.filter(r => !r.error).map(r => [
        r.song_id, r.algorithm ?? "—", r.title ?? "—",
        r.precision != null ? +(r.precision * 100).toFixed(2) : "",
        r.recall    != null ? +(r.recall    * 100).toFixed(2) : "",
        r.f_measure != null ? +(r.f_measure * 100).toFixed(2) : "",
        r.f1_3_0    != null ? +(r.f1_3_0   * 100).toFixed(2) : "",
        r.n_est ?? "", r.n_ref ?? "",
        r.is_outlier ? "YES" : "no",
      ]),
    ];
    const ws = XLSX.utils.aoa_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Batch Eval");
    const dsName = (selectedDataset?.name ?? "dataset").replace(/[^a-z0-9]/gi, "_");
    XLSX.writeFile(wb, `batch_eval_${dsName}.xlsx`);
  }

  onDestroy(() => { batchUnsub?.(); });

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  onMount(async () => {
    await loadDatasets();
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  async function loadDatasets() {
    try {
      datasets = await listDatasets();
    } catch (e) {
      console.error("Failed to load datasets", e);
    }
  }

  async function selectDataset(ds) {
    selectedDataset = ds;
    selectedTrack = null;
    page = 1;
    await loadTracks();
  }

  async function loadTracks() {
    if (!selectedDataset) return;
    try {
      const res = await listDatasetTracks(selectedDataset.dataset_id, {
        page,
        pageSize: PAGE_SIZE,
        hasGroundTruth: filterGT ? true : null,
      });
      tracks = res.tracks || [];
      tracksTotal = res.total || 0;
    } catch (e) {
      console.error("Failed to load tracks", e);
    }
  }

  async function doImportSalami() {
    isImporting = true;
    importError = "";
    importResult = null;
    try {
      importResult = await importSalami();
      await loadDatasets();
      // Auto-select SALAMI dataset
      const salami = datasets.find((d) => d.name === "SALAMI");
      if (salami) await selectDataset(salami);
    } catch (e) {
      importError = e.message;
    } finally {
      isImporting = false;
    }
  }

  async function doCreateDataset() {
    if (!newDatasetName.trim()) { createError = "Name is required."; return; }
    isCreating = true;
    createError = "";
    try {
      await createDataset({ name: newDatasetName.trim(), description: newDatasetDesc.trim() || null });
      newDatasetName = "";
      newDatasetDesc = "";
      await loadDatasets();
    } catch (e) {
      createError = e.message;
    } finally {
      isCreating = false;
    }
  }

  async function doUploadTrack() {
    if (!uploadFile) { uploadError = "Select an audio file."; return; }
    isUploadingTrack = true;
    uploadError = "";
    uploadSuccess = "";
    try {
      const res = await uploadTrack(selectedDataset.dataset_id, {
        file: uploadFile,
        groundTruthCsv: uploadGTFile,
        title: uploadTitle.trim() || uploadFile.name,
        artist: uploadArtist.trim() || null,
      });
      uploadSuccess = `Track uploaded (ID: ${res.track_id})`;
      uploadFile = null;
      uploadGTFile = null;
      uploadTitle = "";
      uploadArtist = "";
      await loadTracks();
      // refresh dataset counts
      await loadDatasets();
      selectedDataset = datasets.find((d) => d.dataset_id === selectedDataset.dataset_id) || selectedDataset;
    } catch (e) {
      uploadError = e.message;
    } finally {
      isUploadingTrack = false;
    }
  }

  async function segmentTrack() {
    if (!selectedTrack || !selectedTrack.song_id) return;
    isSegmenting = true;
    segmentMessage = "";
    try {
      const resp = await segmentSongFromStorage(selectedTrack.song_id);
      segmentMessage = `Task dispatched: ${resp.task_id}`;
    } catch (e) {
      segmentMessage = `Failed: ${e.message}`;
    } finally {
      isSegmenting = false;
    }
  }

  async function segmentAll() {
    const ids = tracks.map((t) => t.song_id).filter(Boolean);
    if (ids.length === 0) { segmentMessage = "No song IDs available to segment."; return; }
    isSegmenting = true;
    segmentMessage = "";
    try {
      const res = await segmentSongsBatch(ids);
      segmentMessage = `Batch dispatch: ${res.results?.length || ids.length} items`;
    } catch (e) {
      segmentMessage = `Failed: ${e.message}`;
    } finally {
      isSegmenting = false;
    }
  }

  function changePage(delta) {
    page = Math.max(1, page + delta);
    loadTracks();
  }

  const totalPages = () => Math.ceil(tracksTotal / PAGE_SIZE);
</script>

<div class="flex h-[calc(100vh-49px)] overflow-hidden text-zinc-100">
  <!-- Left: Dataset list -->
  <aside class="w-64 shrink-0 border-r border-zinc-800 bg-zinc-900/50 flex flex-col">
    <div class="p-4 border-b border-zinc-800">
      <span class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Datasets</span>
    </div>

    <!-- SALAMI import -->
    <div class="p-4 border-b border-zinc-800 space-y-2">
      <button
        class="w-full rounded-xl bg-indigo-500/20 border border-indigo-500/30 py-2 text-sm font-medium text-indigo-300 hover:bg-indigo-500/30 disabled:opacity-50"
        on:click={doImportSalami}
        disabled={isImporting}
      >
        {isImporting ? "Importing…" : "⬇ Import SALAMI"}
      </button>
      {#if importResult}
        <div class="text-xs text-emerald-400">
          Imported {importResult.tracks_imported} tracks (total: {importResult.total_tracks})
        </div>
      {/if}
      {#if importError}
        <div class="text-xs text-red-400">{importError}</div>
      {/if}
    </div>

    <!-- New custom dataset -->
    <div class="p-4 border-b border-zinc-800 space-y-2">
      <p class="text-xs font-medium text-zinc-400">New custom dataset</p>
      <input
        class="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
        placeholder="Name"
        bind:value={newDatasetName}
      />
      <input
        class="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
        placeholder="Description (optional)"
        bind:value={newDatasetDesc}
      />
      <button
        class="w-full rounded-xl bg-zinc-700 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-600 disabled:opacity-50"
        on:click={doCreateDataset}
        disabled={isCreating}
      >{isCreating ? "Creating…" : "Create"}</button>
      {#if createError}
        <div class="text-xs text-red-400">{createError}</div>
      {/if}
    </div>

    <!-- Dataset list -->
    <div class="flex-1 overflow-y-auto p-2 space-y-1">
      {#each datasets as ds}
        <button
          class={"w-full text-left rounded-xl px-3 py-2 text-sm transition-colors " +
            (selectedDataset?.dataset_id === ds.dataset_id
              ? "bg-indigo-500/20 text-indigo-200"
              : "text-zinc-300 hover:bg-zinc-800")}
          on:click={() => selectDataset(ds)}
        >
          <div class="font-medium truncate">{ds.name}</div>
          <div class="text-[11px] text-zinc-500">{ds.track_count} tracks · {ds.source_type}</div>
        </button>
      {/each}
      {#if datasets.length === 0}
        <p class="text-xs text-zinc-500 px-3 py-2">No datasets yet</p>
      {/if}
    </div>
  </aside>

  <!-- Right: Track list + detail -->
  <div class="flex-1 flex flex-col min-w-0">
    {#if selectedDataset}
      <!-- Dataset header -->
      <div class="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/50 px-6 py-3">
        <div>
          <h2 class="text-base font-semibold text-zinc-100">{selectedDataset.name}</h2>
          <p class="text-xs text-zinc-400">{selectedDataset.track_count} tracks · {selectedDataset.source_type}</p>
        </div>

        <div class="flex items-center gap-3">
          <!-- GT filter -->
          <label class="flex items-center gap-2 cursor-pointer text-sm text-zinc-300">
            <input
              type="checkbox"
              class="accent-indigo-500"
              bind:checked={filterGT}
              on:change={() => { page = 1; loadTracks(); }}
            />
            Ground truth only
          </label>

          <!-- Batch Eval toggle -->
          <button
            class={"rounded-xl border px-3 py-1.5 text-xs font-semibold transition-colors " +
              (batchPanelOpen
                ? "border-indigo-600 bg-indigo-500/20 text-indigo-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-zinc-600")}
            on:click={() => (batchPanelOpen = !batchPanelOpen)}
          >
            {batchRunning ? "⏳ Running…" : batchDone ? "✓ Batch Eval" : "Run Batch Eval"}
          </button>
        </div>
      </div>

      <div class="flex flex-1 min-h-0">
        <!-- Track table -->
        <div class="flex-1 flex flex-col overflow-hidden">
          <div class="flex-1 overflow-y-auto">
            <table class="w-full text-sm">
              <thead class="sticky top-0 bg-zinc-900/90 backdrop-blur">
                <tr class="border-b border-zinc-800">
                  <th class="px-4 py-2 text-left text-xs font-medium text-zinc-400">ID</th>
                  <th class="px-4 py-2 text-left text-xs font-medium text-zinc-400">Title</th>
                  <th class="px-4 py-2 text-left text-xs font-medium text-zinc-400">Artist</th>
                  <th class="px-4 py-2 text-left text-xs font-medium text-zinc-400">Duration</th>
                  <th class="px-4 py-2 text-left text-xs font-medium text-zinc-400">GT</th>
                </tr>
              </thead>
              <tbody>
                {#each tracks as t}
                  <tr
                    class={"border-b border-zinc-800/50 cursor-pointer hover:bg-zinc-800/30 " +
                      (selectedTrack?.track_id === t.track_id ? "bg-indigo-500/10" : "")}
                    on:click={() => (selectedTrack = t)}
                  >
                    <td class="px-4 py-2 font-mono text-xs text-zinc-400">{t.song_id || t.track_id.slice(0, 8)}</td>
                    <td class="px-4 py-2 text-zinc-200 truncate max-w-[200px]">{t.title || "—"}</td>
                    <td class="px-4 py-2 text-zinc-400 truncate max-w-[120px]">{t.artist || "—"}</td>
                    <td class="px-4 py-2 text-zinc-400 text-xs">{t.duration_seconds ? `${t.duration_seconds.toFixed(0)}s` : "—"}</td>
                    <td class="px-4 py-2">
                      {#if t.has_ground_truth}
                        <span class="rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs text-emerald-300">✓</span>
                      {:else}
                        <span class="text-zinc-600 text-xs">—</span>
                      {/if}
                    </td>
                  </tr>
                {/each}
                {#if tracks.length === 0}
                  <tr><td colspan="5" class="px-4 py-8 text-center text-sm text-zinc-500">No tracks found</td></tr>
                {/if}
              </tbody>
            </table>
          </div>

          <!-- Pagination -->
          <div class="flex items-center justify-between border-t border-zinc-800 bg-zinc-900/50 px-4 py-2">
            <span class="text-xs text-zinc-400">
              {tracksTotal} total · page {page} / {totalPages() || 1}
            </span>
            <div class="flex gap-2">
              <button
                class="rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
                on:click={() => changePage(-1)}
                disabled={page <= 1}
              >← Prev</button>
              <button
                class="rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
                on:click={() => changePage(1)}
                disabled={page >= totalPages()}
              >Next →</button>
            </div>
          </div>
        </div>

        <!-- ── Batch Eval Panel ───────────────────────────────────────────── -->
        {#if batchPanelOpen}
          <div class="border-t border-zinc-800 bg-zinc-950/60 flex flex-col" style="max-height: 55vh;">
            <!-- Panel header -->
            <div class="flex items-center justify-between px-5 py-3 border-b border-zinc-800 shrink-0">
              <div class="flex items-center gap-3">
                <span class="text-xs font-semibold text-zinc-200 uppercase tracking-wider">Batch Evaluation</span>
                {#if batchDone}
                  <span class="text-[10px] text-zinc-500">
                    {batchSuccessRows.length} tracks ·
                    <span class="text-zinc-300 font-semibold">Raw F1 {fmtPct(batchAvgF1Raw)}</span>
                    · Adj {fmtPct(batchAvgF1Adj)}
                    {#if batchOutlierCount > 0}
                      · <span class="text-amber-400">{batchOutlierCount} outliers excluded</span>
                    {/if}
                  </span>
                {/if}
              </div>
              <div class="flex items-center gap-2">
                {#if batchDone && batchRows.length > 0}
                  <button
                    class="flex items-center gap-1.5 rounded-lg border border-emerald-800/50 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-300 hover:bg-emerald-500/20 transition-colors"
                    on:click={exportBatchExcel}
                  >⬇ Export Excel</button>
                {/if}
                <button
                  class="rounded-xl bg-indigo-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
                  on:click={runBatchEval}
                  disabled={batchRunning}
                >
                  {#if batchRunning}
                    <span class="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white/20 border-t-white/80"></span>
                    Running…
                  {:else}
                    {batchDone ? "Run Again" : "▶ Run All"}
                  {/if}
                </button>
              </div>
            </div>

            {#if batchStartError}
              <p class="px-5 py-2 text-xs text-red-400">{batchStartError}</p>
            {/if}

            <!-- Progress bar -->
            {#if batchRunning || (batchProgress.total > 0)}
              <div class="px-5 py-2 shrink-0">
                <div class="flex items-center justify-between mb-1 text-[10px] text-zinc-500">
                  <span>{batchProgress.completed} / {batchProgress.total || "?"} tracks</span>
                  {#if batchProgress.total > 0}
                    <span>{Math.round(batchProgress.completed / batchProgress.total * 100)}%</span>
                  {/if}
                </div>
                <div class="h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
                  <div
                    class="h-full rounded-full bg-indigo-500 transition-all duration-300"
                    style="width: {batchProgress.total > 0 ? (batchProgress.completed / batchProgress.total * 100) : 0}%"
                  ></div>
                </div>
              </div>
            {/if}

            <!-- Results table (scrollable) -->
            {#if batchSorted.length > 0}
              <div class="flex-1 overflow-y-auto min-h-0">
                <table class="w-full text-xs">
                  <thead class="sticky top-0 bg-zinc-900/95 backdrop-blur border-b border-zinc-800">
                    <tr>
                      <th class="px-3 py-2 text-left font-medium text-zinc-500">ID</th>
                      <th class="px-3 py-2 text-left font-medium text-zinc-500">Algorithm</th>
                      <th class="px-3 py-2 text-left font-medium text-zinc-500">Title</th>
                      <th class="px-3 py-2 text-right font-medium text-zinc-500">Prec</th>
                      <th class="px-3 py-2 text-right font-medium text-zinc-500">Rec</th>
                      <th class="px-3 py-2 text-right font-medium text-zinc-500">F1</th>
                      <th class="px-3 py-2 text-right font-medium text-zinc-500">F1@3s</th>
                      <th class="px-3 py-2 text-center font-medium text-zinc-500">Flag</th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each batchSorted as row (`${row.song_id}-${row.algorithm}`)}
                      <tr class={"border-b border-zinc-800/30 " + (row.is_outlier ? "bg-amber-500/5" : "hover:bg-zinc-800/20")}>
                        <td class="px-3 py-1.5 font-mono text-zinc-500">{row.song_id}</td>
                        <td class="px-3 py-1.5 font-mono text-zinc-400">{row.algorithm ?? "—"}</td>
                        <td class="px-3 py-1.5 text-zinc-300 max-w-[160px] truncate" title={row.title}>{row.title ?? "—"}</td>
                        <td class="px-3 py-1.5 text-right text-zinc-400 tabular-nums">{fmtPct(row.precision)}</td>
                        <td class="px-3 py-1.5 text-right text-zinc-400 tabular-nums">{fmtPct(row.recall)}</td>
                        <td class="px-3 py-1.5 text-right font-bold tabular-nums {row.f_measure >= 0.5 ? 'text-emerald-400' : row.f_measure >= 0.3 ? 'text-amber-400' : 'text-red-400'}">
                          {fmtPct(row.f_measure)}
                        </td>
                        <td class="px-3 py-1.5 text-right tabular-nums text-zinc-500">{fmtPct(row.f1_3_0)}</td>
                        <td class="px-3 py-1.5 text-center">
                          {#if row.is_outlier}
                            <span title="F1@3s < 20% — excluded from aggregate" class="text-amber-400 text-[11px]">⚠</span>
                          {:else}
                            <span class="text-zinc-700">—</span>
                          {/if}
                        </td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {:else if batchRunning}
              <div class="px-5 py-6 text-xs text-zinc-600">Results will appear as tracks complete…</div>
            {:else if !batchDone}
              <div class="px-5 py-6 text-xs text-zinc-600">Press ▶ Run All to evaluate all algorithms against ground truth.</div>
            {/if}
          </div>
        {/if}

        <!-- Track detail sidebar -->
        {#if selectedTrack}
          <div class="w-72 shrink-0 border-l border-zinc-800 bg-zinc-900/30 overflow-y-auto p-4 space-y-4">
            <div>
              <h3 class="text-sm font-semibold text-zinc-100">{selectedTrack.title || selectedTrack.song_id}</h3>
              {#if selectedTrack.artist}
                <p class="text-xs text-zinc-400">{selectedTrack.artist}</p>
              {/if}
              {#if selectedTrack.song_id}
                <div class="mt-2 space-y-2">
                  <a
                    href={getSongStreamUrl(selectedTrack.song_id)}
                    target="_blank"
                    rel="noopener"
                    class="block text-xs text-indigo-400 hover:underline truncate"
                  >Stream URL (open)</a>

                  <div class="flex items-center gap-2">
                    <audio bind:this={player} class="w-full" controls preload="none"></audio>
                    <button
                      class="rounded px-2 py-1 bg-indigo-600 text-white text-xs"
                      on:click={async () => {
                        if (!player) return;
                        try {
                          player.src = getSongStreamUrl(selectedTrack.song_id);
                          await player.play();
                        } catch (e) {
                          console.error('Playback failed', e);
                        }
                      }}
                    >Play</button>
                  </div>
                </div>
              {:else if selectedTrack.audio_url}
                <a
                  href={selectedTrack.audio_url}
                  target="_blank"
                  rel="noopener"
                  class="mt-2 block text-xs text-indigo-400 hover:underline truncate"
                >{selectedTrack.audio_url}</a>
              {/if}
            </div>

            {#if selectedTrack.has_ground_truth}
              <div>
                <p class="text-xs font-medium text-zinc-300 mb-2">Ground Truth ({selectedTrack.ground_truth?.length || 0} segments)</p>
                {#if selectedTrack.ground_truth}
                  <div class="space-y-1 max-h-64 overflow-y-auto">
                    {#each selectedTrack.ground_truth as seg}
                      <div class="flex justify-between rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1 text-xs">
                        <span class="font-medium text-indigo-300">{seg.section_type || seg.label}</span>
                        <span class="text-zinc-500">{seg.start}s – {seg.end}s</span>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {:else}
              <p class="text-xs text-zinc-500">No ground truth annotations</p>
            {/if}
          </div>
        {/if}
      </div>

      <!-- Upload track panel (custom datasets only) -->
      {#if selectedDataset.source_type === "custom"}
        <div class="border-t border-zinc-800 bg-zinc-900/50 px-6 py-4 space-y-3">
          <p class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Upload Track</p>
          <div class="flex flex-wrap gap-3 items-end">
            <input
              class="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
              placeholder="Title"
              bind:value={uploadTitle}
            />
            <input
              class="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
              placeholder="Artist"
              bind:value={uploadArtist}
            />
            <label class="text-xs text-zinc-400">
              Audio file
              <input
                type="file"
                accept=".mp3,.wav,.flac,.ogg,.m4a"
                class="block mt-1 text-xs text-zinc-300 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-3 file:py-1 file:text-xs file:text-zinc-200 hover:file:bg-zinc-700"
                on:change={(e) => (uploadFile = e.currentTarget.files?.[0] ?? null)}
              />
            </label>
            <label class="text-xs text-zinc-400">
              Ground truth CSV (optional)
              <input
                type="file"
                accept=".csv"
                class="block mt-1 text-xs text-zinc-300 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-3 file:py-1 file:text-xs file:text-zinc-200 hover:file:bg-zinc-700"
                on:change={(e) => (uploadGTFile = e.currentTarget.files?.[0] ?? null)}
              />
            </label>
            <button
              class="rounded-xl bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
              on:click={doUploadTrack}
              disabled={isUploadingTrack || !uploadFile}
            >{isUploadingTrack ? "Uploading…" : "Upload"}</button>
          </div>
          {#if uploadError}
            <p class="text-xs text-red-400">{uploadError}</p>
          {/if}
          {#if uploadSuccess}
            <p class="text-xs text-emerald-400">{uploadSuccess}</p>
          {/if}
        </div>
      {/if}

      <!-- Segment controls (for storage-backed datasets) -->
      {#if selectedTrack}
        <div class="border-t border-zinc-800 bg-zinc-900/50 px-6 py-4 space-y-3">
          <p class="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Segmentation</p>
          <div class="flex gap-2">
            <button
              class="rounded-xl bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
              on:click={async () => { await segmentTrack(); }}
              disabled={isSegmenting || !selectedTrack.song_id}
            >{isSegmenting ? 'Segmenting…' : 'Segment Track'}</button>
            <button
              class="rounded-xl bg-indigo-500/20 border border-indigo-500/30 px-3 py-1.5 text-sm font-medium text-indigo-300 hover:bg-indigo-500/30 disabled:opacity-50"
              on:click={async () => { await segmentAll(); }}
              disabled={isSegmenting || tracks.length === 0}
            >{isSegmenting ? 'Segmenting…' : 'Segment All (page)'}</button>
          </div>
          {#if segmentMessage}
            <p class="text-xs text-emerald-400">{segmentMessage}</p>
          {/if}
        </div>
      {/if}

    {:else}
      <div class="flex-1 flex items-center justify-center text-zinc-500 text-sm">
        Select a dataset from the left panel
      </div>
    {/if}
  </div>
</div>
