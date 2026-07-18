<script>
  import { onMount } from "svelte";
  import { batch, loadDatasets, setField, toggleAlgorithm, runBatch } from "../../lib/batchStore.js";
  import { ALL_ALGOS } from "../../lib/analysisStore.js";

  export let goTo;

  onMount(() => {
    if ($batch.datasets.length === 0) loadDatasets();
  });

  $: total = $batch.progress.total || 0;
  $: completedCount = $batch.progress.completed || 0;
  $: activeCount = $batch.running ? Math.max(0, Math.min(Number($batch.concurrency), total - completedCount)) : 0;
  $: queuedCount = Math.max(0, total - completedCount - activeCount);

  $: if ($batch.done && !$batch.running) {
    goTo(8);
  }

  async function start() {
    await runBatch();
  }
</script>

<div style="display: grid; grid-template-columns: 340px 1fr; gap: 24px;">
  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; height: fit-content;">
    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint); margin-bottom: 12px;">Batch configuration</div>

    <div style="margin-bottom: 14px;">
      <div style="font-size: 12px; font-weight: 700; margin-bottom: 6px;">Dataset</div>
      <select
        bind:value={$batch.datasetId}
        on:change={(e) => setField({ datasetId: e.currentTarget.value })}
        disabled={$batch.running}
        style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 9px 12px; border-radius: 8px; font-size: 12.5px;"
      >
        <option value="">All available tracks with ground truth</option>
        {#each $batch.datasets as ds}<option value={ds.dataset_id}>{ds.name}</option>{/each}
      </select>
    </div>

    <div style="margin-bottom: 14px;">
      <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
        <span style="font-weight: 700;">Tracks</span>
        <label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
          <input type="checkbox" bind:checked={$batch.runAllDataset} disabled={$batch.running} />
          <span style="color: var(--msp-text-faint);">all</span>
        </label>
      </div>
      {#if !$batch.runAllDataset}
        <input type="number" min="1" max="500" bind:value={$batch.maxTracks} disabled={$batch.running}
          style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 8px 10px; border-radius: 8px; font-size: 12.5px;" />
      {/if}
    </div>

    <div style="margin-bottom: 14px;">
      <div style="font-size: 12px; font-weight: 700; margin-bottom: 8px;">Algorithms</div>
      <div style="display: flex; flex-wrap: wrap; gap: 6px;">
        {#each ALL_ALGOS as a}
          <button
            type="button" disabled={$batch.running}
            on:click={() => toggleAlgorithm(a.id)}
            style="font-size: 11px; font-weight: 700; padding: 5px 10px; border-radius: 6px; cursor: pointer; border: 1px solid {$batch.algorithms.has(a.id) ? 'var(--msp-border-strong)' : 'var(--msp-border)'}; background: {$batch.algorithms.has(a.id) ? 'var(--msp-accent-bg)' : 'transparent'}; color: {$batch.algorithms.has(a.id) ? 'var(--msp-accent)' : 'var(--msp-text-faint)'}; font-family: inherit;"
          >{a.name}</button>
        {/each}
      </div>
    </div>

    <div style="margin-bottom: 14px;">
      <div style="font-size: 12px; font-weight: 700; margin-bottom: 6px;">Tolerance: <span class="msp-mono">±{$batch.tolerance}s</span></div>
      <input type="range" min="0.5" max="3" step="0.5" bind:value={$batch.tolerance} disabled={$batch.running} style="width: 100%; accent-color: var(--msp-accent);" />
    </div>

    <div style="margin-bottom: 18px;">
      <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;"><span style="font-weight: 700;">Concurrency</span><span class="msp-mono" style="color: var(--msp-text-faint);">{$batch.concurrency} workers</span></div>
      <input type="range" min="1" max="10" step="1" bind:value={$batch.concurrency} disabled={$batch.running} style="width: 100%; accent-color: var(--msp-accent);" />
    </div>

    <button
      type="button" on:click={start} disabled={$batch.running}
      style="width: 100%; background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 11px; border-radius: 8px; font-size: 13px; font-weight: 800; cursor: pointer; font-family: inherit; margin-bottom: 8px; opacity: {$batch.running ? 0.6 : 1};"
    >{$batch.running ? "Running…" : "Run batch evaluation"}</button>

    {#if $batch.error}
      <div style="font-size: 11.5px; color: var(--msp-danger); margin-top: 8px;">{$batch.error}</div>
    {/if}
  </div>

  <div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 18px;">
      <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 12px; background: var(--msp-panel);"><div style="font-size: 10.5px; color: var(--msp-text-faint); font-weight: 700; text-transform: uppercase;">Completed</div><div style="font-size: 20px; font-weight: 800; color: var(--msp-ok);">{completedCount}{total ? ` / ${total}` : ""}</div></div>
      <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 12px; background: var(--msp-panel);"><div style="font-size: 10.5px; color: var(--msp-text-faint); font-weight: 700; text-transform: uppercase;">Active</div><div style="font-size: 20px; font-weight: 800; color: var(--msp-accent);">{activeCount}</div></div>
      <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 12px; background: var(--msp-panel);"><div style="font-size: 10.5px; color: var(--msp-text-faint); font-weight: 700; text-transform: uppercase;">Queued</div><div style="font-size: 20px; font-weight: 800; color: var(--msp-text-dim);">{queuedCount}</div></div>
      <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 12px; background: var(--msp-panel);"><div style="font-size: 10.5px; color: var(--msp-text-faint); font-weight: 700; text-transform: uppercase;">Job</div><div class="msp-mono" style="font-size: 12px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{$batch.jobId || "—"}</div></div>
    </div>

    <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint); margin-bottom: 10px;">Log</div>
    <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-bg); padding: 14px; height: 320px; overflow-y: auto;" class="msp-mono">
      {#if $batch.logLines.length === 0}
        <div style="font-size: 11.5px; color: var(--msp-text-faint);">Configure and run to start streaming progress.</div>
      {/if}
      {#each $batch.logLines as line}
        <div style="font-size: 11px; color: {line.includes('skip') || line.includes('EXCEPTION') ? 'var(--msp-danger)' : 'var(--msp-text-dim)'}; line-height: 1.6;">{line || " "}</div>
      {/each}
    </div>
  </div>
</div>
