<script>
  import { onDestroy } from "svelte";
  import { uploadSegmentation, fetchStatus, subscribeToTask } from "./lib/api";
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
    { id: "custom", label: "Custom", hint: "Optimized baseline" },
    { id: "foote", label: "Foote", hint: "MSAF algorithm" },
    { id: "cnmf", label: "CNMF", hint: "MSAF algorithm" },
    { id: "scluster", label: "S-Cluster", hint: "MSAF algorithm" },
  ];

  let file = null;
  let selected = new Set(["custom", "foote", "cnmf", "scluster"]);

  let isUploading = false;
  let taskId = "";
  let status = "idle"; // idle | uploading | processing | completed | error | timeout
  let statusText = "";
  let errorMsg = "";

  let requested = [];
  let results = {};
  let rawStatus = {};

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
    isUploading = false;
  }

  function allRequestedResultsPresent() {
    if (requested.length === 0) return false;
    return requested.every((a) =>
      Object.prototype.hasOwnProperty.call(results, a)
    );
  }

  function algoChipState(algoId) {
    if (!requested.includes(algoId)) return "neutral";
    if (results?.[algoId]) return "done";
    if (status === "processing" || status === "uploading") return "pending";
    return "neutral";
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
    requested = algos;

    isUploading = true;
    status = "uploading";
    statusText = "Uploading…";

    try {
      taskId = await uploadSegmentation({ file, algorithms: algos });

      isUploading = false;
      status = "processing";
      statusText = `Task ${taskId} — processing (waiting for results)`;

      // Use SSE instead of polling
      console.log("Subscribing to SSE for task:", taskId);
      unsubscribe = subscribeToTask(taskId, (data) => {
        console.log("SSE received data:", data);
        rawStatus = data;
        results = { ...results, ...(data.results || {}) };

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

  <div class="mx-auto max-w-5xl px-4 py-10">
    <!-- Header -->
    <header class="mb-8 flex flex-col gap-2">
      <div class="flex items-center justify-between gap-4">
        <div>
          <h1 class="text-3xl font-semibold tracking-tight">
            Music Segmentation
          </h1>
          <p class="mt-1 text-sm text-zinc-400">
            Upload an audio file, run multiple algorithms, and inspect results
            live.
          </p>
        </div>

        {#if taskId}
          <div class="hidden sm:flex flex-col items-end">
            <span class="text-xs text-zinc-400">Task ID</span>
            <span class="font-mono text-xs text-zinc-200">{taskId}</span>
          </div>
        {/if}
      </div>

      <!-- Status strip -->
      {#if statusText}
        <div
          class="mt-3 flex items-center justify-between rounded-2xl border border-zinc-800 bg-zinc-900/50 px-4 py-3"
        >
          <div class="flex items-center gap-3">
            {#if status === "processing" || status === "uploading"}
              <span
                class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white/80"
              ></span>
            {:else if status === "completed"}
              <span
                class="inline-flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-300"
                >✓</span
              >
            {:else if status === "timeout"}
              <span
                class="inline-flex h-5 w-5 items-center justify-center rounded-full bg-amber-500/15 text-amber-300"
                >!</span
              >
            {:else if status === "error"}
              <span
                class="inline-flex h-5 w-5 items-center justify-center rounded-full bg-red-500/15 text-red-300"
                >×</span
              >
            {/if}

            <div class="text-sm text-zinc-200">
              <span class="text-zinc-400">Status:</span>
              {statusText}
            </div>
          </div>

          <button
            class="rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs font-medium text-zinc-200 hover:border-zinc-700 disabled:opacity-50"
            on:click={resetRunState}
            disabled={isUploading}
          >
            Reset
          </button>
        </div>
      {/if}

      {#if errorMsg}
        <div
          class="mt-3 rounded-2xl border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-200"
        >
          <span class="font-semibold">Error:</span>
          {errorMsg}
        </div>
      {/if}
    </header>

    <!-- Main grid -->
    <div class="grid gap-6 lg:grid-cols-[360px_1fr]">
      <!-- Left: Controls -->
      <section
        class="rounded-3xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur"
      >
        <h2 class="text-sm font-semibold text-zinc-200">Run segmentation</h2>
        <p class="mt-1 text-sm text-zinc-400">
          Choose file + algorithms. Results stream in as workers finish.
        </p>

        <!-- File -->
        <div class="mt-6">
          <label class="text-xs font-medium text-zinc-300">Audio file</label>
          <div class="mt-2 rounded-2xl border border-zinc-800 bg-zinc-950 p-3">
            <input
              class="block w-full cursor-pointer text-sm text-zinc-200 file:mr-4 file:rounded-xl file:border-0 file:bg-zinc-800 file:px-4 file:py-2 file:text-xs file:font-semibold file:text-zinc-100 hover:file:bg-zinc-700"
              type="file"
              accept=".mp3,.wav,.flac,.ogg,.m4a"
              on:change={(e) => (file = e.currentTarget.files?.[0] ?? null)}
            />
            {#if file}
              <div
                class="mt-3 flex items-center justify-between gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2"
              >
                <div class="min-w-0">
                  <div class="truncate text-xs font-medium text-zinc-200">
                    {file.name}
                  </div>
                  <div class="text-[11px] text-zinc-400">
                    {prettyBytes(file.size)}
                  </div>
                </div>
                <button
                  class="shrink-0 rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-300 hover:border-zinc-700"
                  on:click={() => (file = null)}
                  type="button"
                >
                  Clear
                </button>
              </div>
            {:else}
              <p class="mt-2 text-[11px] text-zinc-500">
                Supported: mp3, wav, flac, ogg, m4a
              </p>
            {/if}
          </div>
        </div>

        <!-- Algorithms -->
        <div class="mt-6">
          <label class="text-xs font-medium text-zinc-300">Algorithms</label>

          <div class="mt-3 grid gap-2">
            {#each ALL_ALGOS as a}
              <button
                type="button"
                class="group flex w-full items-center justify-between rounded-2xl border border-zinc-800 bg-zinc-950 px-4 py-3 text-left hover:border-zinc-700"
                on:click={() => toggleAlgo(a.id)}
              >
                <div class="flex items-center gap-3">
                  <div
                    class={"h-4 w-4 rounded border " +
                      (selected.has(a.id)
                        ? "border-indigo-400 bg-indigo-400/30"
                        : "border-zinc-700 bg-transparent")}
                  ></div>
                  <div>
                    <div class="text-sm font-medium text-zinc-200">
                      {a.label}
                    </div>
                    <div class="text-[11px] text-zinc-500">{a.hint}</div>
                  </div>
                </div>

                <!-- Chip -->
                {#if requested.length > 0}
                  {#if algoChipState(a.id) === "done"}
                    <span
                      class="rounded-full bg-emerald-500/15 px-2 py-1 text-xs text-emerald-300"
                      >Done</span
                    >
                  {:else if algoChipState(a.id) === "pending"}
                    <span
                      class="rounded-full bg-amber-500/15 px-2 py-1 text-xs text-amber-300"
                      >Pending</span
                    >
                  {:else}
                    <span
                      class="rounded-full bg-zinc-800 px-2 py-1 text-xs text-zinc-300"
                      >—</span
                    >
                  {/if}
                {/if}
              </button>
            {/each}
          </div>

          <div class="mt-3 text-[11px] text-zinc-500">
            Selected: {Array.from(selected).join(", ") || "none"}
          </div>
        </div>

        <!-- Actions -->
        <div class="mt-6 grid gap-2">
          <button
            class="inline-flex items-center justify-center rounded-2xl bg-indigo-500 px-5 py-3 text-sm font-semibold text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-zinc-700"
            on:click={startProcess}
            disabled={isUploading || status === "processing"}
          >
            {#if isUploading}
              <span
                class="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white/80"
              ></span>
              Uploading…
            {:else if status === "processing"}
              <span
                class="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white/80"
              ></span>
              Processing…
            {:else}
              Start segmentation
            {/if}
          </button>

          <p class="text-[11px] text-zinc-500">
            Tip: use a shorter audio sample while testing so you can verify
            results quickly.
          </p>
        </div>
      </section>

      <!-- Right: Results -->
      <section
        class="rounded-3xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur"
      >
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-semibold text-zinc-200">Results</h2>
            <p class="mt-1 text-sm text-zinc-400">
              JSON output per algorithm. Expand and inspect.
            </p>
          </div>

          <div class="hidden sm:flex items-center gap-2 text-xs text-zinc-400">
            <span class="rounded-full bg-zinc-800 px-2 py-1">SSE</span>
          </div>
        </div>

        {#if Object.keys(results).length === 0}
          <div
            class="mt-6 rounded-2xl border border-dashed border-zinc-800 bg-zinc-950/40 p-10 text-center"
          >
            <div
              class="mx-auto mb-3 h-10 w-10 rounded-2xl border border-zinc-800 bg-zinc-900/60"
            ></div>
            <div class="text-sm font-medium text-zinc-200">No results yet</div>
            <div class="mt-1 text-sm text-zinc-500">
              Run segmentation to see outputs stream in.
            </div>
          </div>
        {:else}
          <div class="mt-6 grid gap-4 lg:grid-cols-2">
            {#each Object.keys(results) as algo}
              <details
                class="group self-start rounded-2xl border border-zinc-800 bg-zinc-950"
              >
                <summary
                  class="flex self-start cursor-pointer list-none items-center justify-between gap-3 px-4 py-3"
                >
                  <div class="flex items-center gap-3">
                    <div class="h-2 w-2 rounded-full bg-indigo-400/80"></div>
                    <div class="text-sm font-semibold text-zinc-200">
                      {algo.toUpperCase()}
                    </div>
                  </div>
                  <div class="text-xs text-zinc-500 group-open:hidden">
                    Click to expand
                  </div>
                  <div class="text-xs text-zinc-500 hidden group-open:block">
                    Click to collapse
                  </div>
                </summary>

                <div class="border-t border-zinc-800 px-4 py-4">
                  <pre
                    class="max-h-[420px] overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-xs text-zinc-200">
{JSON.stringify(results[algo], null, 2)}
                  </pre>
                </div>
              </details>
            {/each}
          </div>
        {/if}

        <!-- Debug payload -->
        {#if Object.keys(rawStatus).length > 0}
          <details
            class="mt-6 rounded-2xl border border-zinc-800 bg-zinc-950/40 p-4"
          >
            <summary class="cursor-pointer text-sm font-medium text-zinc-200"
              >Debug: last status payload</summary
            >
            <pre
              class="mt-3 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-xs text-zinc-200">
{JSON.stringify(rawStatus, null, 2)}
            </pre>
          </details>
        {/if}
      </section>
    </div>

    <footer class="mt-10 text-center text-xs text-zinc-500">
      Built with Svelte + Tailwind.
    </footer>
  </div>
</main>
