<script>
  import { onDestroy } from "svelte";
  import { uploadSegmentation, subscribeToTask } from "./lib/api";
  import DatasetManager from "./components/DatasetManager.svelte";
  import EvaluationDashboard from "./components/EvaluationDashboard.svelte";
  import BatchEvalDashboard from "./components/BatchEvalDashboard.svelte";

  let currentPage = "segmentation";

  const NAV_ITEMS = [
    { id: "segmentation", label: "Segmentation" },
    { id: "datasets", label: "Datasets" },
    { id: "evaluation", label: "Evaluation" },
    { id: "batch-eval", label: "Batch Eval" },
  ];
  import { Button } from "src/lib/components/ui/button";
  import * as Card from "src/lib/components/ui/card";
  import {
    Alert,
    AlertTitle,
    AlertDescription,
  } from "src/lib/components/ui/alert";
  import { Badge } from "src/lib/components/ui/badge";
  import { Label } from "src/lib/components/ui/label";
  import { Checkbox } from "src/lib/components/ui/checkbox";
  import { Separator } from "src/lib/components/ui/separator";

  const ALL_ALGOS = [
    { id: "custom_librosa", label: "Custom Librosa", hint: "Deterministic feature fusion", isLLM: false },
    { id: "foote",    label: "Foote",     hint: "MSAF algorithm",              isLLM: false },
    { id: "cnmf",     label: "CNMF",      hint: "MSAF algorithm",              isLLM: false },
    { id: "scluster", label: "S-Cluster", hint: "MSAF algorithm",              isLLM: false },
    { id: "fusion",   label: "Fusion",    hint: "Algorithm-level voting",      isLLM: false },
    { id: "llm",      label: "AI Agent",  hint: "LangChain · LLM calls billed", isLLM: true },
  ];

  const BASELINE_ALGOS = ["custom_librosa", "foote", "cnmf", "scluster"];

  let file = null;
  let selected = new Set(["custom_librosa"]);
  let llmMode = "deterministic";
  let labelingMethod = "heuristic";

  // Confirmation modal for LLM
  let showLLMConfirm = false;

  let isUploading = false;
  let taskId = "";
  let status = "idle"; // idle | uploading | processing | completed | error | timeout
  let statusText = "";
  let errorMsg = "";

  let requested = [];
  let results = {};
  let rawStatus = {};
  let algoStartTimes = {};
  let algoEndTimes = {};
  let expandedAlgos = new Set();

  let unsubscribe = null;

  function toggleAlgo(id) {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    selected = next;
  }

  function resetRunState() {
    if (unsubscribe) {
      unsubscribe();
      unsubscribe = null;
    }
    taskId = "";
    status = "idle";
    statusText = "";
    errorMsg = "";
    requested = [];
    results = {};
    rawStatus = {};
    algoStartTimes = {};
    algoEndTimes = {};
    expandedAlgos = new Set();
    isUploading = false;
  }

  function allRequestedResultsPresent() {
    if (requested.length === 0) return false;
    return requested.every((a) =>
      Object.prototype.hasOwnProperty.call(results, a),
    );
  }

  function algoChipState(algoId) {
    const isFusionSub = requested.includes("fusion") && BASELINE_ALGOS.includes(algoId);
    if (!requested.includes(algoId) && !isFusionSub) return "neutral";
    if (results?.[algoId]) return "done";
    if (status === "processing" || status === "uploading") return "pending";
    return "neutral";
  }

  $: allExpectedAlgos = (() => {
    if (requested.length === 0) return Object.keys(results).filter(k => !k.includes("__"));
    let exp = [...requested];
    if (requested.includes("fusion")) {
      BASELINE_ALGOS.forEach(a => { if (!exp.includes(a)) exp.push(a); });
    }
    Object.keys(results).filter(k => !k.includes("__") && !exp.includes(k)).forEach(k => exp.push(k));
    return exp;
  })();

  function handleStartClick() {
    if (selected.has("llm") && !showLLMConfirm) {
      showLLMConfirm = true;
    } else {
      showLLMConfirm = false;
      startProcess();
    }
  }

  async function startProcess() {
    errorMsg = "";

    if (!file) {
      status = "error";
      errorMsg = "Pick an audio file first.";
      return;
    }

    const algos = Array.from(selected);
    if (algos.length === 0) {
      status = "error";
      errorMsg = "Select at least one algorithm.";
      return;
    }

    if (unsubscribe) {
      unsubscribe();
    }
    results = {};
    rawStatus = {};
    algoStartTimes = {};
    algoEndTimes = {};
    requested = algos;

    const taskStart = Date.now();
    algos.forEach(a => { algoStartTimes[a] = taskStart; });
    if (algos.includes("fusion")) {
      BASELINE_ALGOS.forEach(a => { algoStartTimes[a] = taskStart; });
    }
    algoStartTimes = { ...algoStartTimes };

    const customParams = { labeling_method: labelingMethod };
    const params = selected.has("llm")
      ? { llm_segmentation: { mode: llmMode }, custom: customParams, custom_librosa: customParams }
      : { custom: customParams, custom_librosa: customParams };

    isUploading = true;
    status = "uploading";
    statusText = "Uploading…";

    try {
      taskId = await uploadSegmentation({ file, algorithms: algos, params });

      isUploading = false;
      status = "processing";
      statusText = `Task ${taskId} — processing (waiting for results)`;

      // Use SSE instead of polling
      console.log("Subscribing to SSE for task:", taskId);
      unsubscribe = subscribeToTask(taskId, /** @param {{status?: string, results?: Record<string, unknown>, error?: string}} data */ (data) => {
        console.log("SSE received data:", data);
        rawStatus = data;

        const arrivalTime = Date.now();
        const newResults = data.results || {};
        Object.keys(newResults).forEach(k => {
          if (!k.includes('__') && !algoEndTimes[k]) {
            algoEndTimes[k] = arrivalTime;
          }
        });
        algoEndTimes = { ...algoEndTimes };
        results = { ...results, ...newResults };

        if (data.status === "completed" || allRequestedResultsPresent()) {
          console.log("Task completed!");
          status = "completed";
          statusText = "Completed.";
          if (unsubscribe) {
            unsubscribe();
            unsubscribe = null;
          }
        } else if (data.status === "failed") {
          console.log("Task failed!");
          status = "error";
          errorMsg = data.error || "Backend reported failure.";
          statusText = "Failed.";
          if (unsubscribe) {
            unsubscribe();
            unsubscribe = null;
          }
        }
      });
      console.log("SSE subscribed, waiting for results...");
    } catch (err) {
      isUploading = false;
      status = "error";
      errorMsg = err.message;
      statusText = "Error.";
    }
  }

  onDestroy(() => {
    if (unsubscribe) unsubscribe();
  });

  function prettyBytes(bytes) {
    if (!bytes && bytes !== 0) return "";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let n = bytes;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i++;
    }
    return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }
</script>

<!-- App shell -->
<main class="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-indigo-500/40">
  <!-- soft glow background -->
  <div class="pointer-events-none fixed inset-0 -z-10">
    <div
      class="absolute left-1/2 top-[-200px] h-[520px] w-[520px] -translate-x-1/2 rounded-full bg-indigo-500/20 blur-3xl"
    ></div>
    <div
      class="absolute right-[-120px] top-[40%] h-[420px] w-[420px] rounded-full bg-fuchsia-500/10 blur-3xl"
    ></div>
    <div
      class="absolute left-[-120px] top-[55%] h-[420px] w-[420px] rounded-full bg-cyan-500/10 blur-3xl"
    ></div>
  </div>

  <!-- Top navigation bar -->
  <nav class="sticky top-0 z-30 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur">
    <div class="mx-auto flex max-w-7xl items-center gap-1 px-4 py-2">
      <span class="mr-4 text-sm font-semibold text-zinc-200 tracking-tight">MusicSeg</span>
      {#each NAV_ITEMS as item}
        <button
          class={"rounded-xl px-3 py-1.5 text-sm font-medium transition-colors " +
            (currentPage === item.id
              ? "bg-indigo-500/20 text-indigo-300"
              : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800")}
          on:click={() => (currentPage = item.id)}
        >
          {item.label}
        </button>
      {/each}
    </div>
  </nav>

  <!-- Page router -->
  {#if currentPage === "datasets"}
    <DatasetManager />
  {:else if currentPage === "evaluation"}
    <EvaluationDashboard />
  {:else if currentPage === "batch-eval"}
    <BatchEvalDashboard />
  {:else}

  <div class="mx-auto max-w-screen-2xl px-6 py-8">
    <!-- Header row -->
    <header class="mb-6 flex items-center justify-between gap-4">
      <div>
        <h1 class="text-2xl font-semibold tracking-tight">Music Segmentation</h1>
        <p class="mt-0.5 text-sm text-zinc-400">Upload an audio file, run multiple algorithms, compare results side-by-side.</p>
      </div>
      <div class="flex items-center gap-3">
        {#if taskId}
          <div class="hidden sm:flex flex-col items-end">
            <span class="text-[10px] text-zinc-500">Task ID</span>
            <span class="font-mono text-xs text-zinc-300">{taskId}</span>
          </div>
        {/if}
        {#if statusText}
          <div class="flex items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/50 px-3 py-2">
            {#if status === "processing" || status === "uploading"}
              <span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/80"></span>
            {:else if status === "completed"}
              <span class="inline-flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500/15 text-[11px] text-emerald-300">✓</span>
            {:else if status === "timeout"}
              <span class="inline-flex h-4 w-4 items-center justify-center rounded-full bg-amber-500/15 text-[11px] text-amber-300">!</span>
            {:else if status === "error"}
              <span class="inline-flex h-4 w-4 items-center justify-center rounded-full bg-red-500/15 text-[11px] text-red-300">×</span>
            {/if}
            <span class="text-xs text-zinc-400">Status:</span>
            <span class="text-xs text-zinc-200">{statusText}</span>
            <button
              class="ml-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-300 hover:border-zinc-600 disabled:opacity-40"
              on:click={resetRunState}
              disabled={isUploading}
            >Reset</button>
          </div>
        {/if}
      </div>
    </header>

    {#if errorMsg}
      <div class="mb-4 rounded-2xl border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-200">
        <span class="font-semibold">Error:</span> {errorMsg}
      </div>
    {/if}

    <!-- Main layout: narrow controls sidebar + wide results -->
    <div class="flex gap-5 items-start">

      <!-- Left sidebar: Controls -->
      <aside class="w-64 shrink-0 rounded-3xl border border-zinc-800 bg-zinc-900/50 p-5 backdrop-blur sticky top-16">
        <h2 class="text-xs font-semibold uppercase tracking-widest text-zinc-400">Run segmentation</h2>

        <!-- File -->
        <div class="mt-4">
          <label class="text-[11px] font-medium text-zinc-400" for="audio-file">Audio file</label>
          <div class="mt-1.5 rounded-xl border border-zinc-800 bg-zinc-950 p-2.5">
            <input
              id="audio-file"
              class="block w-full cursor-pointer text-xs text-zinc-200 file:mr-3 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-zinc-100 hover:file:bg-zinc-700"
              type="file"
              accept=".mp3,.wav,.flac,.ogg,.m4a"
              on:change={(e) => (file = e.currentTarget.files?.[0] ?? null)}
            />
            {#if file}
              <div class="mt-2 flex items-center justify-between gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-2.5 py-1.5">
                <div class="min-w-0">
                  <div class="truncate text-[11px] font-medium text-zinc-200">{file.name}</div>
                  <div class="text-[10px] text-zinc-500">{prettyBytes(file.size)}</div>
                </div>
                <button
                  class="shrink-0 rounded-md border border-zinc-800 bg-zinc-950 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:border-zinc-700"
                  on:click={() => (file = null)} type="button"
                >Clear</button>
              </div>
            {:else}
              <p class="mt-1.5 text-[10px] text-zinc-600">mp3 · wav · flac · ogg · m4a</p>
            {/if}
          </div>
        </div>

        <!-- Algorithms -->
        <div class="mt-4">
          <p class="text-[11px] font-medium text-zinc-400">Algorithms</p>
          <div class="mt-2 space-y-1">
            {#each ALL_ALGOS as a}
              <button
                type="button"
                class="group flex w-full items-center justify-between rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-left hover:border-zinc-700"
                on:click={() => toggleAlgo(a.id)}
              >
                <div class="flex items-center gap-2.5">
                  <div class={"h-3.5 w-3.5 rounded border shrink-0 " + (selected.has(a.id) ? "border-indigo-400 bg-indigo-400/30" : "border-zinc-700 bg-transparent")}></div>
                  <div>
                    <div class="text-xs font-medium text-zinc-200">{a.label}</div>
                    <div class="text-[10px] text-zinc-600">{a.hint}</div>
                  </div>
                </div>
                {#if requested.length > 0}
                  {#if algoChipState(a.id) === "done"}
                    <span class="shrink-0 rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">Done</span>
                  {:else if algoChipState(a.id) === "pending"}
                    <span class="shrink-0 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">…</span>
                  {/if}
                {/if}
              </button>
            {/each}
          </div>

          <!-- Labeling method selector (always visible) -->
          <div class="mt-3 rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2.5">
            <label class="text-[11px] font-medium text-zinc-400" for="labeling-method">Segment labeling</label>
            <select
              id="labeling-method"
              class="mt-1.5 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-100 focus:border-indigo-500 focus:outline-none"
              bind:value={labelingMethod}
            >
              <option value="heuristic">Heuristic (fast)</option>
              <option value="ml">ML — Gradient Boosted Trees</option>
            </select>
          </div>

          {#if selected.has("llm")}
            <div class="mt-3 rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2.5">
              <label class="text-[11px] font-medium text-zinc-400" for="llm-mode">AI Agent mode</label>
              <select
                id="llm-mode"
                class="mt-1.5 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-100 focus:border-indigo-500 focus:outline-none"
                bind:value={llmMode}
              >
                <option value="deterministic">Deterministic</option>
                <option value="ai_generated">AI generated</option>
              </select>
            </div>
          {/if}
        </div>

        <!-- Start button -->
        <div class="mt-4">
          <button
            class="w-full inline-flex items-center justify-center rounded-xl bg-indigo-500 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-zinc-700"
            on:click={handleStartClick}
            disabled={isUploading || status === "processing"}
          >
            {#if isUploading}
              <span class="mr-2 inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/80"></span>
              Uploading…
            {:else if status === "processing"}
              <span class="mr-2 inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/20 border-t-white/80"></span>
              Processing…
            {:else}
              Start segmentation
            {/if}
          </button>
          <p class="mt-2 text-[10px] text-zinc-600 leading-snug">Shorter clips give faster feedback while testing.</p>
        </div>
      </aside>

      <!-- Right: Results — full width, stacked rows -->
      <section class="flex-1 min-w-0 rounded-3xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur">
        <div class="flex items-center justify-between mb-5">
          <div>
            <h2 class="text-sm font-semibold text-zinc-200">Results</h2>
            <p class="mt-0.5 text-xs text-zinc-500">All timelines share the same scale — compare boundaries directly.</p>
          </div>
          <span class="rounded-full bg-zinc-800 px-2 py-1 text-xs text-zinc-400">SSE</span>
        </div>

        {#if allExpectedAlgos.length === 0}
          <div class="rounded-2xl border border-dashed border-zinc-800 bg-zinc-950/40 p-16 text-center">
            <div class="mx-auto mb-3 h-10 w-10 rounded-2xl border border-zinc-800 bg-zinc-900/60"></div>
            <div class="text-sm font-medium text-zinc-200">No results yet</div>
            <div class="mt-1 text-xs text-zinc-500">Run segmentation to see outputs stream in.</div>
          </div>
        {:else}
          {@const globalMaxDur = Math.max(...allExpectedAlgos.map(a => { const s = results[a] || []; return s.length ? Math.max(...s.map(x => x.end)) : 0; }), 1)}
          {@const palette = ['#6366f1','#10b981','#f59e0b','#ec4899','#06b6d4','#f97316','#8b5cf6','#14b8a6']}

          <!-- Shared time axis -->
          <div class="mb-3 flex items-center gap-0" style="padding-left: 220px;">
            <div class="flex-1 flex justify-between text-[10px] text-zinc-600 font-mono tabular-nums">
              <span>0:00</span>
              <span>{String(Math.floor(globalMaxDur/4/60)).padStart(2,'0')}:{String(Math.floor(globalMaxDur/4%60)).padStart(2,'0')}</span>
              <span>{String(Math.floor(globalMaxDur/2/60)).padStart(2,'0')}:{String(Math.floor(globalMaxDur/2%60)).padStart(2,'0')}</span>
              <span>{String(Math.floor(globalMaxDur*3/4/60)).padStart(2,'0')}:{String(Math.floor(globalMaxDur*3/4%60)).padStart(2,'0')}</span>
              <span>{String(Math.floor(globalMaxDur/60)).padStart(2,'0')}:{String(Math.floor(globalMaxDur%60)).padStart(2,'0')}</span>
            </div>
          </div>

          <!-- Algorithm rows -->
          <div class="space-y-2">
            {#each allExpectedAlgos as algo}
              {@const isLlm = algo === 'llm'}
              {@const segs = results[algo] || []}
              {@const isPending = segs.length === 0 && (status === "processing" || status === "uploading")}
              {@const workerTime = results[algo + '__processing_time']}
              {@const uniqueLabels = [...new Set(segs.map(s => s.label))]}
              {@const colorMap = Object.fromEntries(uniqueLabels.map((l, i) => [l, palette[i % palette.length]]))}

              <div class={"rounded-2xl border " + (isLlm ? "border-indigo-800/50 bg-zinc-950" : "border-zinc-800 bg-zinc-950")}>
                <!-- Row: label col + timeline bar -->
                <button
                  type="button"
                  disabled={isPending}
                  class={"w-full flex items-center gap-0 px-4 py-3 rounded-2xl transition-colors " + (isPending ? "cursor-default opacity-60" : "cursor-pointer " + (isLlm ? "hover:bg-indigo-500/5" : "hover:bg-zinc-900/60")) + (expandedAlgos.has(algo) ? " rounded-b-none border-b " + (isLlm ? "border-indigo-800/30" : "border-zinc-800/60") : "")}
                  on:click={() => { if (isPending) return; const next = new Set(expandedAlgos); next.has(algo) ? next.delete(algo) : next.add(algo); expandedAlgos = next; }}
                >
                  <!-- Label column: fixed 220px -->
                  <div class="w-[220px] shrink-0 flex items-center gap-3 pr-4">
                    <div class={"h-2 w-2 rounded-full shrink-0 " + (isLlm ? "bg-indigo-400" : "bg-indigo-400/70")}></div>
                    <div class="min-w-0 text-left">
                      <div class="flex items-center gap-2">
                        <span class="text-sm font-semibold text-zinc-200">{isLlm ? 'AI Agent' : algo.toUpperCase()}</span>
                        {#if isLlm}<span class="rounded-full border border-indigo-800/60 bg-indigo-500/10 px-1.5 py-0.5 text-[9px] font-medium text-indigo-300">LLM</span>{/if}
                      </div>
                      <div class="flex items-center gap-2 mt-0.5">
                        {#if isPending}
                          <span class="text-[11px] text-zinc-600 italic">waiting…</span>
                        {:else}
                          <span class="text-[11px] text-zinc-500">{segs.length} seg</span>
                          {#if workerTime != null}
                            <span class="font-mono text-[11px] text-zinc-500 tabular-nums">{workerTime.toFixed(1)}s</span>
                          {:else if algoEndTimes[algo] && algoStartTimes[algo]}
                            <span class="font-mono text-[11px] text-zinc-600 tabular-nums">{((algoEndTimes[algo] - algoStartTimes[algo]) / 1000).toFixed(1)}s</span>
                          {/if}
                        {/if}
                        {#if isLlm && results["llm__evaluation"]?.boundary_f_measure != null}
                          <span class="text-[11px] font-semibold tabular-nums {results['llm__evaluation'].boundary_f_measure >= 0.7 ? 'text-emerald-400' : results['llm__evaluation'].boundary_f_measure >= 0.5 ? 'text-amber-400' : 'text-red-400'}">
                            F1 {(results["llm__evaluation"].boundary_f_measure * 100).toFixed(1)}%
                          </span>
                        {/if}
                      </div>
                    </div>
                  </div>

                  <!-- Timeline bar: flex-1 -->
                  <div class="flex-1 min-w-0">
                    <div class="relative h-10 rounded-xl overflow-hidden bg-zinc-900/80 w-full">
                      {#if isPending}
                        <div class="absolute inset-0 flex items-center justify-center gap-2">
                          <span class="inline-block h-3 w-3 animate-spin rounded-full border-2 border-zinc-700 border-t-zinc-400"></span>
                          <span class="text-[10px] text-zinc-600">processing</span>
                        </div>
                      {/if}
                      {#each segs as seg}
                        {@const left = (seg.start / globalMaxDur) * 100}
                        {@const width = Math.max((seg.end - seg.start) / globalMaxDur * 100, 0.2)}
                        <div
                          class="absolute inset-y-0 flex items-center justify-center overflow-hidden"
                          style="left: {left}%; width: {width}%; background-color: {colorMap[seg.label]}40; border-right: 1px solid {colorMap[seg.label]}55;"
                          title="{seg.label}: {seg.start.toFixed(1)}s – {seg.end.toFixed(1)}s"
                        >
                          {#if width > 4}
                            <span class="px-1 text-[10px] font-bold truncate" style="color: {colorMap[seg.label]}">{seg.label}</span>
                          {/if}
                        </div>
                      {/each}
                      <!-- Boundary lines -->
                      {#each segs as seg, i}
                        {#if i > 0}
                          <div class="absolute inset-y-0 w-px bg-white/20 z-20 pointer-events-none" style="left: {(seg.start / globalMaxDur) * 100}%"></div>
                        {/if}
                      {/each}
                    </div>
                  </div>

                  <!-- Chevron -->
                  <div class="ml-3 shrink-0">
                    <svg class="h-3.5 w-3.5 text-zinc-600 transition-transform {expandedAlgos.has(algo) ? 'rotate-180' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/></svg>
                  </div>
                </button>

                <!-- Boundary timestamps row — mirrors button's internal structure -->
                <div class="flex items-start gap-0 px-4 pb-1.5">
                  <div class="w-[220px] shrink-0"></div>
                  <div class="flex-1 min-w-0 relative h-4">
                    {#each segs as seg, i}
                      {#if i > 0}
                        {@const pct = (seg.start / globalMaxDur) * 100}
                        <div class="absolute top-0 flex flex-col items-center" style="left: {pct}%; transform: translateX(-50%)">
                          <div class="h-1.5 w-px bg-zinc-700/70"></div>
                          <span class="text-[8px] font-mono tabular-nums text-zinc-600 leading-none whitespace-nowrap">{String(Math.floor(seg.start/60)).padStart(2,'0')}:{String(Math.floor(seg.start%60)).padStart(2,'0')}</span>
                        </div>
                      {/if}
                    {/each}
                  </div>
                  <div class="ml-3 w-3.5 shrink-0"></div>
                </div>

                <!-- Expanded detail rows -->
                {#if expandedAlgos.has(algo)}
                  {#if isLlm && results["llm__explanation"]}
                    <div class="px-4 py-3 border-b border-zinc-800/40 bg-indigo-500/5">
                      <p class="text-[10px] font-semibold uppercase tracking-widest text-indigo-400 mb-1">Agent Explanation</p>
                      <p class="text-sm text-zinc-300 leading-relaxed">{results["llm__explanation"]}</p>
                    </div>
                  {/if}
                  <div class="px-4 pb-4 pt-2 space-y-1">
                    {#each segs as seg}
                      <div class={"rounded-xl border px-3 py-2.5 hover:bg-zinc-900/60 transition-colors " + (isLlm ? "border-indigo-800/20 bg-zinc-900/30" : "border-zinc-800/60 bg-zinc-900/30")}>
                        <div class="flex items-center gap-2">
                          <div class="h-2 w-2 shrink-0 rounded-sm" style="background-color: {colorMap[seg.label]}"></div>
                          <span class="font-mono text-[11px] text-zinc-500 w-28 shrink-0 tabular-nums">
                            {String(Math.floor(seg.start/60)).padStart(2,'0')}:{String(Math.floor(seg.start%60)).padStart(2,'0')}
                            –
                            {String(Math.floor(seg.end/60)).padStart(2,'0')}:{String(Math.floor(seg.end%60)).padStart(2,'0')}
                          </span>
                          <span class="text-xs font-bold" style="color: {colorMap[seg.label]}">{seg.label}</span>
                          {#if seg.structural_label && seg.structural_label !== seg.label}
                            <span class="rounded-full border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400">{seg.structural_label}</span>
                          {/if}
                          {#if seg.section_type}
                            <span class="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{seg.section_type}</span>
                          {/if}
                          {#if seg.confidence != null}
                            <span class="ml-auto text-[10px] tabular-nums font-mono shrink-0 {seg.confidence >= 0.7 ? 'text-emerald-400' : seg.confidence >= 0.5 ? 'text-amber-400' : 'text-zinc-500'}">
                              {(seg.confidence * 100).toFixed(0)}%
                            </span>
                          {/if}
                        </div>
                        {#if seg.semantic_label || seg.semantic_reason || seg.reason}
                          <div class="mt-1.5 flex items-start gap-2 pl-4">
                            {#if seg.semantic_label}
                              <span class="shrink-0 rounded-full bg-indigo-500/10 px-2 py-0.5 text-[10px] font-medium text-indigo-300 border border-indigo-500/20">
                                {seg.semantic_label}{seg.semantic_confidence != null ? ` · ${(seg.semantic_confidence * 100).toFixed(0)}%` : ''}
                              </span>
                            {/if}
                            <span class="text-[11px] text-zinc-500 leading-snug">{seg.semantic_reason || seg.reason || ''}</span>
                          </div>
                        {/if}
                        {#if seg.source_features?.length || seg.label_method || seg.label_confidence != null}
                          <div class="mt-1.5 flex flex-wrap items-center gap-1.5 pl-4">
                            {#each seg.source_features || [] as feat}
                              <span class="rounded bg-zinc-800/80 px-1.5 py-0.5 text-[10px] font-mono text-zinc-500">{feat}</span>
                            {/each}
                            {#if seg.label_method}<span class="ml-auto text-[10px] text-zinc-600 font-mono">{seg.label_method}</span>{/if}
                            {#if seg.label_confidence != null}<span class="text-[10px] text-zinc-600 tabular-nums">lconf {(seg.label_confidence * 100).toFixed(0)}%</span>{/if}
                          </div>
                        {/if}
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}

        <!-- Debug payload -->
        {#if Object.keys(rawStatus).length > 0}
          <details class="mt-5 rounded-2xl border border-zinc-800 bg-zinc-950/40 p-4">
            <summary class="cursor-pointer text-xs font-medium text-zinc-400">▶ Debug: last status payload</summary>
            <pre class="mt-3 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-xs text-zinc-200">{JSON.stringify(rawStatus, null, 2)}</pre>
          </details>
        {/if}
      </section>
    </div>

    <footer class="mt-8 text-center text-xs text-zinc-600">
      Built with Svelte + Tailwind.
    </footer>
  </div>
  {/if}

  <!-- ── LLM confirmation modal (rendered outside the page div, works because position:fixed) ── -->
  {#if showLLMConfirm}
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
              <strong class="text-zinc-200">AI Agent (LLM)</strong> was selected. This algorithm makes an <strong class="text-amber-300">API call</strong> each time it runs and may incur charges.
            </p>
            <p class="mt-1 text-xs text-zinc-500">Provider: <span class="text-zinc-300">LLM_PROVIDER env</span></p>
          </div>
        </div>
        <div class="mt-5 flex gap-2">
          <button
            class="flex-1 rounded-2xl border border-zinc-800 bg-zinc-900 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800"
            on:click={() => (showLLMConfirm = false)}
          >
            Cancel
          </button>
          <button
            class="flex-1 rounded-2xl bg-indigo-500 py-2 text-sm font-semibold text-white hover:bg-indigo-400"
            on:click={() => { showLLMConfirm = false; startProcess(); }}
          >
            Yes, continue
          </button>
        </div>
      </div>
    </div>
  {/if}
</main>
