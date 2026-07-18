<script>
  import {
    analysis, ALL_ALGOS, toggleAlgo, setLabelingMethod, setAdvanced, startAnalysis,
  } from "../../lib/analysisStore.js";

  export let goTo;

  $: sourceName = $analysis.file
    ? $analysis.file.name
    : $analysis.sourceTrack
      ? ($analysis.sourceTrack.title || $analysis.sourceTrack.song_id)
      : "No source selected";
  $: sourceMeta = $analysis.file
    ? `${prettyBytes($analysis.file.size)}`
    : $analysis.sourceTrack
      ? `${$analysis.sourceTrack.has_ground_truth ? "ground truth available" : "no ground truth"}`
      : "";

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

  function handleAnalyzeClick() {
    runAndAdvance();
  }

  async function runAndAdvance() {
    goTo(2);
    await startAnalysis();
  }
</script>

{#if !$analysis.file && !$analysis.sourceTrack}
  <div style="max-width: 480px; margin: 60px auto; text-align: center;">
    <div style="font-size: 14px; color: var(--msp-text-dim); margin-bottom: 14px;">No song selected yet.</div>
    <button
      type="button"
      on:click={() => goTo(0)}
      style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 10px 20px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; font-family: inherit;"
    >Upload a song</button>
  </div>
{:else}
<div style="max-width: 880px; margin: 0 auto;">
  <div style="display: flex; align-items: center; gap: 14px; padding: 16px; border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); margin-bottom: 22px;">
    <div style="width: 44px; height: 44px; border-radius: 9px; background: var(--msp-panel-2); flex: none; display: flex; align-items: center; justify-content: center; font-size: 18px;">♪</div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14.5px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{sourceName}</div>
      <div class="msp-mono" style="font-size: 12px; color: var(--msp-text-faint);">{sourceMeta}</div>
    </div>
    <span role="button" tabindex="0" on:click={() => goTo(0)} on:keydown={(e) => e.key === "Enter" && goTo(0)} style="font-size: 11px; color: var(--msp-accent); font-weight: 700; cursor: pointer;">Replace</span>
  </div>

  <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--msp-text-faint); margin-bottom: 10px;">Algorithms</div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 22px;">
    {#each ALL_ALGOS as alg}
      <button
        type="button"
        on:click={() => toggleAlgo(alg.id)}
        style="display: flex; align-items: flex-start; gap: 10px; padding: 12px 14px; border: 1px solid var(--msp-border); border-radius: 10px; background: var(--msp-panel); text-align: left; cursor: pointer; font-family: inherit;"
      >
        <div style="width:16px; height:16px; border-radius:4px; border:1.5px solid {$analysis.selectedAlgos.has(alg.id) ? 'var(--msp-accent)' : 'var(--msp-border-strong)'}; background:{$analysis.selectedAlgos.has(alg.id) ? 'var(--msp-accent)' : 'transparent'}; flex:none; margin-top:2px;"></div>
        <div>
          <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 8px; height: 8px; border-radius: 2px; background: {alg.color};"></div>
            <span style="font-size: 13px; font-weight: 700;">{alg.name}</span>
            {#if alg.isFusion}<span style="font-size: 9.5px; font-weight: 800; padding: 1px 6px; border-radius: 4px; background: var(--msp-accent-bg); color: var(--msp-accent); text-transform: uppercase; letter-spacing: .04em;">Recommended</span>{/if}
          </div>
          <div style="font-size: 11.5px; color: var(--msp-text-faint); margin-top: 3px; line-height: 1.5;">{alg.desc}</div>
        </div>
      </button>
    {/each}
  </div>

  <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 14px 16px; margin-bottom: 16px;">
    <div style="font-size: 13px; font-weight: 700; margin-bottom: 10px;">Segment labeling</div>
    <select
      bind:value={$analysis.labelingMethod}
      on:change={(e) => setLabelingMethod(e.currentTarget.value)}
      style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 8px 10px; border-radius: 8px; font-size: 12.5px;"
    >
      <option value="heuristic">Heuristic (fast)</option>
      <option value="ml">ML — Gradient Boosted Trees</option>
    </select>
  </div>

  <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 14px 16px; margin-bottom: 24px;">
    <div style="font-size: 13px; font-weight: 700; margin-bottom: 14px;">Advanced parameters (Fusion)</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;"><span style="font-weight: 600;">Fusion threshold</span><span class="msp-mono" style="color: var(--msp-text-faint);">{$analysis.advanced.threshold.toFixed(2)}</span></div>
        <input type="range" min="0" max="1" step="0.01" bind:value={$analysis.advanced.threshold} on:input={(e) => setAdvanced({ threshold: Number(e.currentTarget.value) })} style="width: 100%; accent-color: var(--msp-accent);" />
        <div style="font-size: 10.5px; color: var(--msp-text-faint); margin-top: 5px;">Minimum weighted vote score to accept a boundary</div>
      </div>
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;"><span style="font-weight: 600;">Boundary merge window</span><span class="msp-mono" style="color: var(--msp-text-faint);">{$analysis.advanced.mergeWindowSeconds.toFixed(1)}s</span></div>
        <input type="range" min="0.1" max="10" step="0.1" bind:value={$analysis.advanced.mergeWindowSeconds} on:input={(e) => setAdvanced({ mergeWindowSeconds: Number(e.currentTarget.value) })} style="width: 100%; accent-color: var(--msp-accent);" />
        <div style="font-size: 10.5px; color: var(--msp-text-faint); margin-top: 5px;">Merges nearby boundaries within this window</div>
      </div>
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;"><span style="font-weight: 600;">Minimum vote count</span><span class="msp-mono" style="color: var(--msp-text-faint);">{$analysis.advanced.requiredVoteCount} of 4</span></div>
        <input type="range" min="1" max="4" step="1" bind:value={$analysis.advanced.requiredVoteCount} on:input={(e) => setAdvanced({ requiredVoteCount: Number(e.currentTarget.value) })} style="width: 100%; accent-color: var(--msp-accent);" />
        <div style="font-size: 10.5px; color: var(--msp-text-faint); margin-top: 5px;">Algorithms that must agree to keep a boundary</div>
      </div>
    </div>
  </div>

  {#if $analysis.errorMsg}
    <div style="margin-bottom: 14px; padding: 10px 14px; border-radius: 8px; background: var(--msp-danger); color: var(--msp-accent-ink); font-size: 12.5px;">{$analysis.errorMsg}</div>
  {/if}

  <button
    type="button"
    on:click={handleAnalyzeClick}
    style="width: 100%; background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 14px; border-radius: 10px; font-size: 14px; font-weight: 800; cursor: pointer; font-family: inherit;"
  >Analyze Song</button>
</div>
{/if}
