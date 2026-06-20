<script>
  import { onDestroy, onMount } from "svelte";
  import * as XLSX from "xlsx";
  import { startBatchEval, subscribeToBatchEval, listBatchEvalHistory } from "../lib/api.js";

  // ── Config ────────────────────────────────────────────────────────────────
  let maxTracks = 20;
  let runAllDataset = false;
  let tolerance = 0.5;
  let concurrency = 3;
  let includeLLM = false;
  let llmMode = "deterministic";
  let showLLMBatchConfirm = false;
  const WORKER_PRESETS = [2, 3, 4];

  // ── Run state ─────────────────────────────────────────────────────────────
  let running = false;
  let jobId = null;
  let unsub = null;

  // ── Live data (accumulates during streaming) ──────────────────────────────
  let liveRows = [];       // { song_id, f_measure } — parsed from log lines
  let logLines = [];
  let progressCompleted = 0;
  let progressTotal = 0;

  // ── Final data (from done event) ──────────────────────────────────────────
  let finalRows = [];      // full row objects with P/R/F1/etc.
  let summary = null;
  let doneError = null;
  let isDone = false;

  // ── UI state ──────────────────────────────────────────────────────────────
  let logExpanded = true;
  let logEl = null;
  let sortDir = "desc";    // "asc" | "desc"
  let startError = null;
  let copied = false;

  // ── History ───────────────────────────────────────────────────────────────
  let historyItems = [];
  let historyLoading = false;
  let viewingHistoryId = null;   // job_id of the history run currently displayed

  // ── Regex for parsing SSE log lines ──────────────────────────────────────
  const RE_F1       = /\[\s*(\d+)\/\s*(\d+)\]\s+(\d+).*?F1=([\d.]+)/;
  const RE_PROGRESS = /\[\s*(\d+)\/\s*(\d+)\]/;

  // ── Derived state ─────────────────────────────────────────────────────────
  $: displayRows = isDone ? finalRows : liveRows;

  $: successRows = displayRows.filter((r) => !r.error);

  $: outlierRows  = successRows.filter((r) => r.is_outlier);
  $: includedRows = successRows.filter((r) => !r.is_outlier);

  $: errorCount = isDone
    ? finalRows.filter((r) => r.error).length
    : logLines.filter((l) => /skip|download_failed|FATAL/i.test(l)).length;

  $: avgPrecision = computeAvg(successRows, "precision");
  $: avgRecall    = computeAvg(successRows, "recall");
  $: avgF1        = computeAvg(successRows, "f_measure");
  $: adjF1        = computeAvg(includedRows, "f_measure");

  $: sortedRows = [...successRows].sort((a, b) => {
    const af = a.f_measure ?? 0;
    const bf = b.f_measure ?? 0;
    return sortDir === "desc" ? bf - af : af - bf;
  });

  $: distribution = computeDistribution(successRows);

  $: progressPct = progressTotal > 0
    ? Math.round((progressCompleted / progressTotal) * 100)
    : 0;

  // ── Helpers ───────────────────────────────────────────────────────────────
  function computeAvg(rows, key) {
    const vals = rows.map((r) => r[key]).filter((v) => v != null && !isNaN(v));
    if (vals.length === 0) return null;
    return vals.reduce((s, v) => s + v, 0) / vals.length;
  }

  const DIST_BUCKETS = [
    { label: "0.0 – 0.2", min: 0.0,  max: 0.2,  color: "bg-red-500" },
    { label: "0.2 – 0.4", min: 0.2,  max: 0.4,  color: "bg-orange-500" },
    { label: "0.4 – 0.6", min: 0.4,  max: 0.6,  color: "bg-amber-500" },
    { label: "0.6 – 0.8", min: 0.6,  max: 0.8,  color: "bg-lime-500" },
    { label: "0.8 – 1.0", min: 0.8,  max: 1.01, color: "bg-emerald-500" },
  ];

  function computeDistribution(rows) {
    const counts = DIST_BUCKETS.map(() => 0);
    for (const r of rows) {
      const f = r.f_measure ?? 0;
      for (let i = 0; i < DIST_BUCKETS.length; i++) {
        if (f >= DIST_BUCKETS[i].min && f < DIST_BUCKETS[i].max) {
          counts[i]++;
          break;
        }
      }
    }
    const max = Math.max(...counts, 1);
    return DIST_BUCKETS.map((b, i) => ({ ...b, count: counts[i], pct: counts[i] / max }));
  }

  function fmtPct(v) {
    if (v == null || isNaN(v)) return "—";
    return (v * 100).toFixed(1) + "%";
  }

  function f1ColorClass(v) {
    if (v == null) return "text-zinc-400";
    if (v >= 0.7)  return "text-emerald-400";
    if (v >= 0.5)  return "text-amber-400";
    return "text-red-400";
  }

  function logLineColor(line) {
    if (/F1=/i.test(line))                         return "text-emerald-400";
    if (/skip|download_failed|FATAL|error/i.test(line)) return "text-red-400";
    if (/warning/i.test(line))                     return "text-amber-400";
    return "text-zinc-400";
  }

  function scrollLogToBottom() {
    if (logEl) setTimeout(() => { logEl.scrollTop = logEl.scrollHeight; }, 0);
  }

  // ── Parse an SSE log line ─────────────────────────────────────────────────
  function parseLine(line) {
    const mF1 = RE_F1.exec(line);
    if (mF1) {
      const completed = parseInt(mF1[1], 10);
      const total     = parseInt(mF1[2], 10);
      const songId    = mF1[3];
      const f1        = parseFloat(mF1[4]);
      progressCompleted = completed;
      progressTotal     = total;
      // Upsert live row
      const idx = liveRows.findIndex((r) => String(r.song_id) === String(songId));
      if (idx >= 0) {
        liveRows[idx] = { song_id: songId, f_measure: f1 };
      } else {
        liveRows = [...liveRows, { song_id: songId, f_measure: f1 }];
      }
      return;
    }
    const mProg = RE_PROGRESS.exec(line);
    if (mProg) {
      progressCompleted = parseInt(mProg[1], 10);
      progressTotal     = parseInt(mProg[2], 10);
    }
  }

  // ── Run / reset ───────────────────────────────────────────────────────────
  async function runBatch() {
    if (running) return;

    // Reset everything
    unsub?.();
    unsub              = null;
    running            = true;
    isDone             = false;
    viewingHistoryId   = null;
    liveRows           = [];
    finalRows          = [];
    logLines           = [];
    progressCompleted  = 0;
    progressTotal      = 0;
    summary            = null;
    doneError          = null;
    startError         = null;

    try {
      const { job_id } = await startBatchEval({
        maxTracks: runAllDataset ? 0 : Number(maxTracks),
        toleranceSeconds: Number(tolerance),
        concurrency: Number(concurrency),
        includeLLM,
        llmMode,
      });
      jobId = job_id;

      unsub = subscribeToBatchEval(
        job_id,
        (line) => {
          logLines = logLines.length >= 100
            ? [...logLines.slice(-99), line]
            : [...logLines, line];
          parseLine(line);
          scrollLogToBottom();
        },
        ({ summary: s, rows, error }) => {
          summary   = s;
          finalRows = rows ?? [];
          doneError = error ?? null;
          isDone    = true;
          running   = false;
          unsub     = null;
          // Sync progress to final count
          progressCompleted = finalRows.filter((r) => !r.error).length;
          progressTotal     = finalRows.length;
          // Refresh history list to include this run
          loadHistory();
        },
      );
    } catch (e) {
      startError = e.message;
      running    = false;
    }
  }

  function toggleSort() {
    sortDir = sortDir === "desc" ? "asc" : "desc";
  }

  function setWorkerPreset(count) {
    if (running) return;
    concurrency = count;
  }

  function copyReport() {
    const lines = [];
    if (summary) lines.push(summary, "");
    lines.push("ID\tTitle\tPrecision\tRecall\tF1\tSeg time");
    for (const r of finalRows.filter((r) => !r.error)) {
      lines.push([
        r.song_id,
        r.title ?? "—",
        fmtPct(r.precision),
        fmtPct(r.recall),
        fmtPct(r.f_measure),
        r.seg_time_s != null ? r.seg_time_s.toFixed(2) + "s" : "—",
      ].join("\t"));
    }
    navigator.clipboard.writeText(lines.join("\n")).catch(() => {});
    copied = true;
    setTimeout(() => { copied = false; }, 2000);
  }

  function exportExcel() {
    const rows = finalRows.filter(r => !r.error);
    if (!rows.length) return;
    const data = [
      ["ID", "Algorithm", "Title", "Precision%", "Recall%", "F1%", "F1@3s%", "n_est", "n_ref", "Seg time (s)", "Outlier"],
      ...rows.map(r => [
        r.song_id, r.algorithm ?? "—", r.title ?? "—",
        r.precision != null ? +(r.precision * 100).toFixed(2) : "",
        r.recall    != null ? +(r.recall    * 100).toFixed(2) : "",
        r.f_measure != null ? +(r.f_measure * 100).toFixed(2) : "",
        r.f1_3_0    != null ? +(r.f1_3_0   * 100).toFixed(2) : "",
        r.n_est ?? "", r.n_ref ?? "",
        r.seg_time_s != null ? +r.seg_time_s.toFixed(3) : "",
        r.is_outlier ? "YES" : "no",
      ]),
    ];
    const ws = XLSX.utils.aoa_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Batch Eval");
    XLSX.writeFile(wb, `batch_eval_${jobId ?? "run"}.xlsx`);
  }

  // ── History helpers ───────────────────────────────────────────────────────
  async function loadHistory() {
    historyLoading = true;
    try {
      historyItems = await listBatchEvalHistory({ limit: 30 });
    } catch (_) {
      historyItems = [];
    } finally {
      historyLoading = false;
    }
  }

  function viewHistoryRun(item) {
    if (running) return;
    unsub?.();
    unsub            = null;
    viewingHistoryId = item.job_id;
    isDone           = true;
    running          = false;
    finalRows        = item.rows ?? [];
    summary          = item.summary ?? null;
    doneError        = item.error ?? null;
    liveRows         = [];
    logLines         = [];
    progressCompleted = finalRows.filter((r) => !r.error).length;
    progressTotal     = finalRows.length;
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
      + " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
  onMount(() => { loadHistory(); });

  onDestroy(() => {
    unsub?.();
  });
</script>

<!-- ── LLM batch confirmation modal ── -->
{#if showLLMBatchConfirm}
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
    <div class="w-full max-w-md rounded-3xl border border-amber-800/50 bg-zinc-950 p-6 shadow-2xl shadow-black/40">
      <div class="flex items-start gap-3">
        <div class="shrink-0 rounded-xl border border-amber-800/40 bg-amber-500/10 p-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-amber-400" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
          </svg>
        </div>
        <div>
          <h3 class="text-sm font-semibold text-zinc-100">Batch Eval with AI Agent</h3>
          <p class="mt-2 text-xs text-zinc-400 leading-relaxed">
            The AI Agent will run for <strong class="text-amber-300">{runAllDataset ? "all available" : maxTracks} tracks</strong>.
            Each track makes a separate <strong class="text-zinc-200">LLM API call</strong>, which may take time and incur charges.
          </p>
          <p class="mt-1.5 text-[11px] text-zinc-500">Do you want to continue?</p>
        </div>
      </div>
      <div class="mt-5 flex gap-2">
        <button
          class="flex-1 rounded-2xl border border-zinc-800 bg-zinc-900 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800"
          on:click={() => (showLLMBatchConfirm = false)}
        >
          Cancel
        </button>
        <button
          class="flex-1 rounded-2xl bg-amber-500 py-2 text-sm font-semibold text-white hover:bg-amber-400"
          on:click={() => { showLLMBatchConfirm = false; runBatch(); }}
        >
          Yes, start
        </button>
      </div>
    </div>
  </div>
{/if}

<!-- ── Full-page layout ────────────────────────────────────────────────────── -->
<div class="flex h-[calc(100vh-49px)] overflow-hidden text-zinc-100 bg-zinc-950">

  <!-- ── LEFT SIDEBAR (280px) ──────────────────────────────────────────────── -->
  <aside class="w-[280px] shrink-0 border-r border-zinc-800 bg-zinc-900/50 flex flex-col overflow-y-auto">

    <!-- Sidebar header -->
    <div class="px-5 py-4 border-b border-zinc-800">
      <p class="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">Batch Evaluation</p>
      <h2 class="mt-0.5 text-sm font-semibold text-zinc-100">Configuration</h2>
    </div>

    <div class="flex-1 p-5 space-y-6">

      <!-- Max tracks -->
      <div>
        <label for="batch-max-tracks" class="text-xs font-medium text-zinc-400 block mb-1.5">
          Dataset scope
        </label>
        <div class="grid grid-cols-2 gap-1.5 rounded-xl border border-zinc-800 bg-zinc-950 p-1">
          <button
            type="button"
            on:click={() => (runAllDataset = false)}
            disabled={running}
            class={"rounded-lg px-2 py-1.5 text-[11px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 " +
              (!runAllDataset ? "bg-zinc-800 text-zinc-100" : "text-zinc-500 hover:text-zinc-300")}
          >
            Limited run
          </button>
          <button
            type="button"
            on:click={() => (runAllDataset = true)}
            disabled={running}
            class={"rounded-lg px-2 py-1.5 text-[11px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 " +
              (runAllDataset ? "bg-indigo-500 text-white" : "text-zinc-500 hover:text-zinc-300")}
          >
            All dataset
          </button>
        </div>
        <input
          id="batch-max-tracks"
          type="number"
          min="1"
          max="500"
          bind:value={maxTracks}
          disabled={running || runAllDataset}
          class="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <p class="mt-1 text-[10px] text-zinc-600">
          {runAllDataset ? "Runs every available SALAMI track, no cap" : "Use a smaller cap for smoke runs"}
        </p>
      </div>

      <!-- Concurrency -->
      <div>
        <label for="batch-concurrency" class="text-xs font-medium text-zinc-400 block mb-1.5">
          Concurrency
        </label>
        <input
          id="batch-concurrency"
          type="number"
          min="1"
          max="10"
          bind:value={concurrency}
          disabled={running}
          class="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div class="mt-2 grid grid-cols-3 gap-1.5">
          {#each WORKER_PRESETS as preset}
            <button
              type="button"
              on:click={() => setWorkerPreset(preset)}
              disabled={running}
              class={"rounded-lg border px-2 py-1.5 text-[10px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 " +
                (Number(concurrency) === preset
                  ? "border-indigo-500 bg-indigo-500/20 text-indigo-200"
                  : "border-zinc-800 bg-zinc-950 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300")}
            >
              {preset} workers
            </button>
          {/each}
        </div>
        <p class="mt-1 text-[10px] text-zinc-600">Parallel tracks — match available worker capacity</p>
      </div>

      <!-- Tolerance slider -->
      <div>
        <label for="batch-tolerance" class="text-xs font-medium text-zinc-400 block mb-1.5">
          Tolerance: <span class="text-zinc-200 font-semibold">±{tolerance}s</span>
        </label>
        <input
          id="batch-tolerance"
          type="range"
          min="0.5"
          max="3.0"
          step="0.5"
          bind:value={tolerance}
          disabled={running}
          class="w-full accent-indigo-500"
        />
        <div class="flex justify-between mt-1 text-[10px] text-zinc-600">
          <span>0.5s</span><span>3.0s</span>
        </div>
      </div>

      <!-- AI Agent toggle -->
      <div class="rounded-2xl border border-zinc-800 bg-zinc-950/60 px-3 py-3">
        <label class="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            bind:checked={includeLLM}
            disabled={running}
            class="accent-indigo-500 h-4 w-4 shrink-0"
          />
          <div class="min-w-0">
            <p class="text-xs font-medium text-zinc-200 flex items-center gap-1.5">
              AI Agent (LLM)
              <span class="rounded-full border border-amber-800/50 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-semibold text-amber-400">PAID</span>
            </p>
            <p class="text-[10px] text-zinc-500 mt-0.5">Makes an LLM API call for each track</p>
          </div>
        </label>

        {#if includeLLM}
          <div class="mt-3 border-t border-zinc-800 pt-3">
            <label class="text-[10px] font-medium uppercase tracking-wider text-zinc-500" for="batch-llm-mode">
              AI Agent mode
            </label>
            <select
              id="batch-llm-mode"
              bind:value={llmMode}
              class="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
            >
              <option value="deterministic">Deterministic</option>
              <option value="ai_generated">AI generated</option>
            </select>
            <p class="mt-2 text-[10px] text-zinc-500">Applies to AI Agent batch runs.</p>
          </div>
        {/if}
      </div>

      <!-- Run button -->
      <button
        on:click={() => includeLLM ? (showLLMBatchConfirm = true) : runBatch()}
        disabled={running}
        class="w-full rounded-2xl bg-indigo-500 py-2.5 text-sm font-semibold text-white hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
      >
        {#if running}
          <span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/80"></span>
          Running…
        {:else if isDone}
          Run Again
        {:else if runAllDataset}
          Run All Dataset
        {:else}
          Run Batch Eval
        {/if}
      </button>

      {#if startError}
        <div class="rounded-xl border border-red-900/60 bg-red-950/30 px-3 py-2 text-xs text-red-300">
          {startError}
        </div>
      {/if}

      <!-- Progress bar -->
      {#if progressTotal > 0 || running}
        <div>
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs text-zinc-400">Progress</span>
            <span class="text-xs font-medium text-zinc-200">
              {progressCompleted} / {progressTotal > 0 ? progressTotal : "?"}
            </span>
          </div>
          <div class="h-2 w-full rounded-full bg-zinc-800 overflow-hidden">
            <div
              class="h-full rounded-full bg-indigo-500 transition-all duration-300"
              style="width: {progressPct}%"
            ></div>
          </div>
          {#if progressTotal > 0}
            <p class="mt-1 text-[10px] text-zinc-600 text-right">{progressPct}%</p>
          {/if}
        </div>
      {/if}

      <!-- Collapsible log -->
      <div>
        <button
          on:click={() => (logExpanded = !logExpanded)}
          class="flex w-full items-center justify-between text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          <span>Live Log</span>
          <span class="text-zinc-600 font-mono text-[10px]">{logExpanded ? "▲" : "▼"}</span>
        </button>

        {#if logExpanded}
          <div
            bind:this={logEl}
            class="mt-2 h-52 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-950 p-3 font-mono text-[10px] space-y-0.5 scroll-smooth"
          >
            {#if logLines.length === 0}
              <p class="text-zinc-600">
                {running ? "Waiting for output…" : "No log yet. Press Run to start."}
              </p>
            {/if}
            {#each logLines as line}
              <div class={logLineColor(line)}>{line || " "}</div>
            {/each}
            {#if doneError}
              <div class="text-red-400 mt-1">Fatal: {doneError}</div>
            {/if}
          </div>
        {/if}
      </div>

      <!-- History -->
      <div>
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-medium text-zinc-400">Run History</span>
          <button
            on:click={loadHistory}
            disabled={historyLoading}
            class="text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-40"
            title="Refresh history"
          >↻</button>
        </div>

        {#if historyLoading}
          <p class="text-[10px] text-zinc-600">Loading…</p>
        {:else if historyItems.length === 0}
          <p class="text-[10px] text-zinc-600">No past runs yet.</p>
        {:else}
          <div class="space-y-1.5 max-h-64 overflow-y-auto pr-0.5">
            {#each historyItems as item (item.job_id)}
              <button
                on:click={() => viewHistoryRun(item)}
                class="w-full text-left rounded-xl border px-3 py-2 text-[11px] transition-colors
                  {viewingHistoryId === item.job_id
                    ? 'border-indigo-700 bg-indigo-500/10 text-indigo-200'
                    : 'border-zinc-800 bg-zinc-950/60 text-zinc-300 hover:border-zinc-700 hover:bg-zinc-800/40'}"
              >
                <div class="flex items-center justify-between gap-1 mb-0.5">
                  <span class="font-mono text-[10px] text-zinc-500">{fmtDate(item.started_at)}</span>
                  <span class="rounded-full px-1.5 py-0.5 text-[9px] font-semibold
                    {item.status === 'completed' ? 'bg-emerald-500/15 text-emerald-400' :
                     item.status === 'failed'    ? 'bg-red-500/15 text-red-400' :
                                                   'bg-indigo-500/15 text-indigo-400'}">
                    {item.status}
                  </span>
                </div>
                <div class="flex items-center justify-between gap-1">
                  <span class="text-zinc-400">
                    {item.tracks_ok ?? "?"}/{item.tracks_total ?? "?"} tracks · ±{item.tolerance_seconds}s
                  </span>
                  {#if item.avg_f1 != null}
                    <span class="font-semibold tabular-nums
                      {item.avg_f1 >= 0.4 ? 'text-emerald-400' : item.avg_f1 >= 0.25 ? 'text-amber-400' : 'text-red-400'}">
                      F1 {(item.avg_f1 * 100).toFixed(1)}%
                    </span>
                  {/if}
                </div>
              </button>
            {/each}
          </div>
        {/if}
      </div>

    </div>
  </aside>

  <!-- ── RIGHT CONTENT AREA ──────────────────────────────────────────────── -->
  <div class="flex-1 overflow-y-auto p-6 space-y-6">

    {#if viewingHistoryId}
      <div class="flex items-center gap-3 rounded-2xl border border-indigo-800/40 bg-indigo-500/5 px-4 py-2.5">
        <span class="text-xs text-indigo-300">Viewing saved run</span>
        <span class="font-mono text-[10px] text-indigo-500">{viewingHistoryId}</span>
        <button
          on:click={() => { viewingHistoryId = null; isDone = false; finalRows = []; summary = null; }}
          class="ml-auto text-[10px] text-indigo-400 hover:text-indigo-200 transition-colors"
        >✕ Clear</button>
      </div>
    {/if}

    <!-- Empty state -->
    {#if !running && !isDone && liveRows.length === 0 && logLines.length === 0}
      <div class="flex min-h-[calc(100vh-120px)] items-center justify-center">
        <div class="w-full max-w-2xl rounded-3xl border border-zinc-800 bg-zinc-900/50 p-8 shadow-2xl shadow-black/20">
          <div class="flex items-start justify-between gap-6">
            <div>
              <p class="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">Batch Evaluation</p>
              <h2 class="mt-2 text-2xl font-semibold text-zinc-100">No results yet</h2>
              <p class="mt-2 max-w-md text-sm text-zinc-400">
                Configure max tracks and tolerance in the sidebar, then press
                <span class="text-zinc-200 font-medium">{runAllDataset ? "Run All Dataset" : "Run Batch Eval"}</span> to start evaluating
                selected algorithms against the full SALAMI dataset.
              </p>
            </div>
            <div class="rounded-2xl border border-zinc-800 bg-zinc-950/80 px-4 py-3 text-right shrink-0">
              <div class="text-[10px] uppercase tracking-widest text-zinc-500">Status</div>
              <div class="mt-1 text-sm font-medium text-zinc-200">Idle</div>
            </div>
          </div>

          <div class="mt-8 grid gap-4 sm:grid-cols-2">
            <div class="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
              <div class="mb-3 flex items-center gap-2 text-sm font-medium text-zinc-200">
                <span class="h-2 w-2 rounded-full bg-indigo-500"></span>
                What you will see
              </div>
              <ul class="space-y-2 text-sm text-zinc-400">
                <li>• Live F1 metrics as tracks complete</li>
                <li>• F1 distribution across the dataset</li>
                <li>• Full per-track results table</li>
              </ul>
            </div>
            <div class="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
              <div class="mb-3 text-sm font-medium text-zinc-200">Summary stats</div>
              <div class="space-y-2">
                <div class="h-3 w-full rounded-full bg-zinc-800"></div>
                <div class="h-3 w-5/6 rounded-full bg-zinc-800"></div>
                <div class="h-3 w-3/4 rounded-full bg-zinc-800"></div>
                <div class="mt-4 h-16 rounded-2xl border border-dashed border-zinc-700 bg-zinc-900/60"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

    {:else}

      <!-- ── STATS BAR ──────────────────────────────────────────────────── -->
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">

        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-4 py-4 flex flex-col gap-1">
          <span class="text-2xl font-bold tabular-nums text-zinc-100">{fmtPct(avgPrecision)}</span>
          <span class="text-[10px] uppercase tracking-wider text-zinc-500">Avg Precision</span>
        </div>

        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-4 py-4 flex flex-col gap-1">
          <span class="text-2xl font-bold tabular-nums text-zinc-100">{fmtPct(avgRecall)}</span>
          <span class="text-[10px] uppercase tracking-wider text-zinc-500">Avg Recall</span>
        </div>

        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-4 py-4 flex flex-col gap-1">
          <span class="text-2xl font-bold tabular-nums {f1ColorClass(avgF1)}">{fmtPct(avgF1)}</span>
          <span class="text-[10px] uppercase tracking-wider text-zinc-500">Avg F1 (raw)</span>
        </div>

        <div class="rounded-2xl border border-indigo-800/40 bg-indigo-500/5 px-4 py-4 flex flex-col gap-1">
          <span class="text-2xl font-bold tabular-nums {f1ColorClass(adjF1)}">{fmtPct(adjF1)}</span>
          <span class="text-[10px] uppercase tracking-wider text-indigo-400">Adj F1 (–outliers)</span>
        </div>

        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-4 py-4 flex flex-col gap-1">
          <span class="text-2xl font-bold tabular-nums text-zinc-100">
            {successRows.length}{progressTotal > 0 ? "/" + progressTotal : ""}
          </span>
          <span class="text-[10px] uppercase tracking-wider text-zinc-500">Tracks Evaluated</span>
        </div>

        <div class="rounded-2xl border border-amber-800/30 bg-amber-500/5 px-4 py-4 flex flex-col gap-1">
          <span class="text-2xl font-bold tabular-nums {outlierRows.length > 0 ? 'text-amber-400' : 'text-zinc-400'}">{outlierRows.length}</span>
          <span class="text-[10px] uppercase tracking-wider text-amber-500/70">Outliers ⚠</span>
        </div>

        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-4 py-4 flex flex-col gap-1">
          <span class="text-2xl font-bold tabular-nums {errorCount > 0 ? 'text-red-400' : 'text-zinc-100'}">{errorCount}</span>
          <span class="text-[10px] uppercase tracking-wider text-zinc-500">Errors / Skipped</span>
        </div>

      </div>

      <!-- ── F1 DISTRIBUTION ────────────────────────────────────────────── -->
      <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div class="px-5 py-3 border-b border-zinc-800 flex items-center justify-between">
          <h3 class="text-sm font-semibold text-zinc-200">F1 Distribution</h3>
          <span class="text-xs text-zinc-500">{successRows.length} tracks</span>
        </div>
        <div class="px-5 py-4 space-y-2.5">
          {#each distribution as bucket}
            <div class="flex items-center gap-3">
              <span class="w-[80px] shrink-0 text-[11px] text-zinc-500 font-mono">{bucket.label}</span>
              <div class="flex-1 h-5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  class="h-full rounded-full transition-all duration-500 {bucket.color}"
                  style="width: {(bucket.pct * 100).toFixed(1)}%"
                ></div>
              </div>
              <span class="w-8 shrink-0 text-right text-xs text-zinc-300 tabular-nums">{bucket.count}</span>
            </div>
          {/each}
          {#if successRows.length === 0}
            <p class="text-xs text-zinc-600 py-2">Waiting for results…</p>
          {/if}
        </div>
      </div>

      <!-- ── PER-TRACK RESULTS TABLE ─────────────────────────────────────── -->
      <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div class="px-5 py-3 border-b border-zinc-800 flex items-center justify-between gap-3 flex-wrap">
          <div class="flex items-center gap-3">
            <h3 class="text-sm font-semibold text-zinc-200">Per-Track Results</h3>
            {#if running}
              <span class="inline-flex items-center gap-1.5 rounded-full border border-indigo-800/60 bg-indigo-500/10 px-2 py-0.5 text-[10px] font-medium text-indigo-300">
                <span class="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse"></span>
                Live
              </span>
            {/if}
            {#if errorCount > 0}
              <span class="rounded-full bg-red-500/10 border border-red-900/40 px-2 py-0.5 text-[10px] font-medium text-red-400">
                {errorCount} error{errorCount !== 1 ? "s" : ""}
              </span>
            {/if}
          </div>
          {#if isDone && finalRows.length > 0}
            <div class="flex items-center gap-2">
              <button
                on:click={exportExcel}
                class="flex items-center gap-1.5 rounded-xl border border-emerald-800/50 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300 hover:bg-emerald-500/20 transition-colors"
              >
                ⬇ Export Excel
              </button>
              <button
                on:click={copyReport}
                class="flex items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-800/60 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
              >
                {#if copied}
                  <span class="text-emerald-400">Copied!</span>
                {:else}
                  Copy TSV Report
                {/if}
              </button>
            </div>
          {/if}
        </div>

        {#if sortedRows.length === 0}
          <div class="px-5 py-10 text-center">
            <div class="text-sm text-zinc-600">
              {running ? "Results will appear as tracks complete…" : "No successful results to display."}
            </div>
          </div>
        {:else}
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead class="border-b border-zinc-800 bg-zinc-900/80 sticky top-0">
                <tr>
                  <th class="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 whitespace-nowrap">ID</th>
                  <th class="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 whitespace-nowrap">Algorithm</th>
                  <th class="px-4 py-2.5 text-left text-xs font-medium text-zinc-400">Title</th>
                  {#if isDone}
                    <th class="px-4 py-2.5 text-right text-xs font-medium text-zinc-400 whitespace-nowrap">Precision</th>
                    <th class="px-4 py-2.5 text-right text-xs font-medium text-zinc-400 whitespace-nowrap">Recall</th>
                  {/if}
                  <th class="px-4 py-2.5 text-right whitespace-nowrap">
                    <button
                      on:click={toggleSort}
                      class="text-xs font-medium text-zinc-400 hover:text-indigo-400 transition-colors flex items-center gap-1 ml-auto"
                    >
                      F1
                      <span class="text-[10px] text-zinc-600">{sortDir === "desc" ? "▼" : "▲"}</span>
                    </button>
                  </th>
                  {#if isDone}
                    <th class="px-4 py-2.5 text-right text-xs font-medium text-zinc-400 whitespace-nowrap">F1@3s</th>
                    <th class="px-4 py-2.5 text-right text-xs font-medium text-zinc-400 whitespace-nowrap">est / ref</th>
                    <th class="px-4 py-2.5 text-right text-xs font-medium text-zinc-400 whitespace-nowrap">Seg time</th>
                    <th class="px-4 py-2.5 text-center text-xs font-medium text-amber-500/70 whitespace-nowrap">⚠</th>
                  {/if}
                </tr>
              </thead>
              <tbody>
                {#each sortedRows as row (`${row.song_id}-${row.algorithm ?? "default"}`)}
                  <tr class={"border-b border-zinc-800/40 transition-colors " + (row.is_outlier ? "bg-amber-500/5 hover:bg-amber-500/10" : "hover:bg-zinc-800/20")}>
                    <td class="px-4 py-2 text-xs text-zinc-500 font-mono whitespace-nowrap">{row.song_id}</td>
                    <td class="px-4 py-2 text-xs text-zinc-300 font-mono whitespace-nowrap">{row.algorithm ?? "custom_librosa"}</td>
                    <td class="px-4 py-2 text-zinc-300 max-w-[180px]">
                      <span class="block truncate" title={row.title ?? "—"}>{row.title ?? "—"}</span>
                    </td>
                    {#if isDone}
                      <td class="px-4 py-2 text-right text-zinc-300 tabular-nums text-xs whitespace-nowrap">{fmtPct(row.precision)}</td>
                      <td class="px-4 py-2 text-right text-zinc-300 tabular-nums text-xs whitespace-nowrap">{fmtPct(row.recall)}</td>
                    {/if}
                    <td class="px-4 py-2 text-right font-bold tabular-nums text-xs whitespace-nowrap {f1ColorClass(row.f_measure)}">
                      {fmtPct(row.f_measure)}
                    </td>
                    {#if isDone}
                      <td class="px-4 py-2 text-right text-zinc-500 tabular-nums text-xs whitespace-nowrap">{fmtPct(row.f1_3_0)}</td>
                      <td class="px-4 py-2 text-right text-zinc-500 text-xs whitespace-nowrap">
                        {row.n_est ?? "—"} / {row.n_ref ?? "—"}
                      </td>
                      <td class="px-4 py-2 text-right text-zinc-500 text-xs whitespace-nowrap">
                        {row.seg_time_s != null ? row.seg_time_s.toFixed(2) + "s" : "—"}
                      </td>
                      <td class="px-4 py-2 text-center text-xs">
                        {#if row.is_outlier}
                          <span title="F1@3s < 20% — excluded from adjusted F1" class="text-amber-400">⚠</span>
                        {:else}
                          <span class="text-zinc-800">—</span>
                        {/if}
                      </td>
                    {/if}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </div>

      <!-- ── SUMMARY BLOCK (shown after done) ───────────────────────────── -->
      {#if isDone && summary}
        <div class="rounded-2xl border border-zinc-800 bg-zinc-900/50 px-5 py-4">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-2">Summary</h3>
          <pre class="font-mono text-xs text-zinc-300 whitespace-pre-wrap">{summary}</pre>
        </div>
      {/if}

      <!-- ── DONE ERROR ─────────────────────────────────────────────────── -->
      {#if doneError}
        <div class="rounded-2xl border border-red-900/60 bg-red-950/30 px-5 py-4 text-sm text-red-300">
          <span class="font-semibold">Job error:</span> {doneError}
        </div>
      {/if}

    {/if}
  </div>
</div>
