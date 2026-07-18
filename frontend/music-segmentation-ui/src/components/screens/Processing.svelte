<script>
  import { analysis, expectedAlgos, algoRunStatus, ALL_ALGOS } from "../../lib/analysisStore.js";

  export let goTo;

  $: algos = expectedAlgos($analysis);
  $: runs = algos.map((id) => {
    const def = ALL_ALGOS.find((a) => a.id === id) || { name: id, color: "var(--msp-text-faint)" };
    const rs = algoRunStatus($analysis, id);
    const time = $analysis.processingTimes[id];
    return { id, name: def.name, color: def.color, ...rs, time };
  });

  $: doneCount = runs.filter((r) => r.state === "completed" || r.state === "failed").length;
  $: allDone = runs.length > 0 && doneCount === runs.length;
  $: stages = [
    { label: "Uploading audio", done: $analysis.status !== "idle" && $analysis.status !== "uploading", running: $analysis.status === "uploading" },
    ...runs.map((r) => ({ label: `Running ${r.name}`, done: r.state === "completed" || r.state === "failed", running: r.state === "running" })),
    { label: "Preparing results", done: $analysis.status === "completed", running: allDone && $analysis.status !== "completed" },
  ];
  $: stagesDone = stages.filter((s) => s.done).length;

  $: if ($analysis.status === "completed" && allDone) {
    goTo(3);
  }
</script>

<div style="max-width: 780px; margin: 0 auto;">
  <div style="border: 1px solid var(--msp-border); border-radius: 12px; padding: 22px; background: var(--msp-panel); margin-bottom: 22px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
      <span style="font-size: 13.5px; font-weight: 700;">Overall progress</span>
      <span class="msp-mono" style="font-size: 12px; color: var(--msp-text-faint);">{stagesDone} / {stages.length} stages</span>
    </div>
    {#each stages as st}
      <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0;">
        <div style="width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:800; flex:none; background:{st.done ? 'var(--msp-ok)' : st.running ? 'var(--msp-accent)' : 'var(--msp-panel-2)'}; color:{st.done || st.running ? 'var(--msp-accent-ink)' : 'var(--msp-text-faint)'}; {st.running ? 'animation: msp-pulse 1.2s ease-in-out infinite;' : ''}">
          {st.done ? "✓" : st.running ? "●" : ""}
        </div>
        <span style="font-size:12.5px; font-weight:{st.done || st.running ? '700' : '500'}; color:{st.done ? 'var(--msp-text)' : st.running ? 'var(--msp-text)' : 'var(--msp-text-faint)'};">{st.label}</span>
      </div>
    {/each}
  </div>

  <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--msp-text-faint); margin-bottom: 10px;">Per-algorithm status</div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
    {#each runs as run}
      <div style="padding: 14px; border: 1px solid var(--msp-border); border-radius: 10px; background: var(--msp-panel);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
          <div style="display: flex; align-items: center; gap: 7px;">
            <div style="width: 8px; height: 8px; border-radius: 2px; background: {run.color};"></div>
            <span style="font-size: 12.5px; font-weight: 700;">{run.name}</span>
          </div>
          <span style="font-size:10px; font-weight:700; padding:3px 9px; border-radius:5px; background:{run.state === 'completed' ? 'var(--msp-panel-2)' : run.state === 'running' ? 'var(--msp-accent-bg)' : run.state === 'failed' ? 'var(--msp-danger)' : 'var(--msp-panel-2)'}; color:{run.state === 'completed' ? 'var(--msp-ok)' : run.state === 'running' ? 'var(--msp-accent)' : run.state === 'failed' ? 'var(--msp-accent-ink)' : 'var(--msp-text-faint)'};">
            {run.state === "completed" ? "Completed" : run.state === "running" ? "Running" : run.state === "failed" ? "Failed" : "Queued"}
          </span>
        </div>
        <div style="height: 5px; border-radius: 3px; background: var(--msp-panel-2); overflow: hidden;">
          <div style="width: {run.state === 'completed' || run.state === 'failed' ? 100 : run.state === 'running' ? 55 : 0}%; height: 100%; background: {run.state === 'failed' ? 'var(--msp-danger)' : run.color};"></div>
        </div>
        {#if run.time != null}
          <div class="msp-mono" style="font-size: 10.5px; color: var(--msp-text-faint); margin-top: 6px;">{run.time.toFixed(1)}s</div>
        {/if}
        {#if run.error}
          <div style="font-size: 10.5px; color: var(--msp-danger); margin-top: 6px;">{run.error}</div>
        {/if}
      </div>
    {/each}
  </div>
  <div style="margin-top: 18px; font-size: 11.5px; color: var(--msp-text-faint); line-height: 1.6;">A failure in one algorithm will not hide successful results from the others — Fusion recalculates using the algorithms that completed.</div>

  {#if $analysis.status === "error"}
    <div style="margin-top: 18px; padding: 12px 16px; border-radius: 10px; background: var(--msp-danger); color: var(--msp-accent-ink); font-size: 12.5px;">{$analysis.errorMsg}</div>
    <button type="button" on:click={() => goTo(1)} style="margin-top: 10px; background: transparent; border: 1px solid var(--msp-border-strong); color: var(--msp-text); padding: 8px 14px; border-radius: 8px; font-size: 12px; font-weight: 700; cursor: pointer; font-family: inherit;">Back to configuration</button>
  {/if}

  {#if allDone && $analysis.status !== "completed"}
    <button type="button" on:click={() => goTo(3)} style="margin-top: 18px; width: 100%; background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 12px; border-radius: 10px; font-size: 13px; font-weight: 800; cursor: pointer; font-family: inherit;">View results</button>
  {/if}
</div>
