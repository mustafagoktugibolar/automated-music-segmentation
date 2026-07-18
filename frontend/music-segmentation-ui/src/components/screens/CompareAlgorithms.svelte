<script>
  import { analysis, expectedAlgos, ALL_ALGOS, toggleVisibleAlgo } from "../../lib/analysisStore.js";
  import { secToLabel, boundaryTimes, maxDuration } from "../../lib/segmentUtils.js";

  export let goTo;

  const GT = { id: "groundtruth", name: "Ground Truth", color: "var(--msp-alg-gt)" };

  $: algoIds = expectedAlgos($analysis).filter((id) => Array.isArray($analysis.results[id]));
  $: hasGT = !!$analysis.sourceTrack?.ground_truth?.length;
  $: rowDefs = [
    ...algoIds.map((id) => ALL_ALGOS.find((a) => a.id === id) || { id, name: id, color: "var(--msp-text-faint)" }),
    ...(hasGT ? [GT] : []),
  ];
  $: segmentsFor = (id) => id === "groundtruth" ? ($analysis.sourceTrack?.ground_truth || []) : ($analysis.results[id] || []);
  $: duration = maxDuration(rowDefs.map((r) => segmentsFor(r.id)));
  $: rows = rowDefs.map((r) => ({
    ...r,
    visible: $analysis.visibleAlgos[r.id] !== false,
    boundaries: boundaryTimes(segmentsFor(r.id)),
  }));

  let selectedBoundary = null;

  function nearestMatch(boundaries, t, windowSec = 3) {
    let best = null;
    let bestDist = Infinity;
    for (const b of boundaries) {
      const d = Math.abs(b - t);
      if (d < bestDist) { bestDist = d; best = b; }
    }
    return best != null && bestDist <= windowSec ? { time: best, dist: bestDist } : null;
  }

  $: matchDetail = selectedBoundary == null ? null : rows.map((r) => ({
    name: r.name,
    match: nearestMatch(r.boundaries, selectedBoundary),
  }));
</script>

{#if rowDefs.length === 0}
  <div style="max-width: 480px; margin: 60px auto; text-align: center;">
    <div style="font-size: 14px; color: var(--msp-text-dim); margin-bottom: 14px;">No results to compare yet.</div>
    <button type="button" on:click={() => goTo(0)} style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 10px 20px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; font-family: inherit;">Upload a song</button>
  </div>
{:else}
<div>
  <div style="display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap;">
    {#each rows as r}
      <button
        type="button"
        on:click={() => toggleVisibleAlgo(r.id)}
        style="display: flex; align-items: center; gap: 7px; padding: 7px 13px; border-radius: 8px; font-size: 12px; font-weight: 700; cursor: pointer; border: 1px solid {r.visible ? 'var(--msp-border-strong)' : 'var(--msp-border)'}; background: {r.visible ? 'var(--msp-panel-2)' : 'transparent'}; color: {r.visible ? 'var(--msp-text)' : 'var(--msp-text-faint)'}; font-family: inherit;"
      >
        <div style="width: 7px; height: 7px; border-radius: 2px; background: {r.color};"></div>
        {r.name}
      </button>
    {/each}
  </div>

  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 20px;">
    <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--msp-text-faint); margin-bottom: 10px; padding-left: 132px;" class="msp-mono">
      <span>0:00</span><span>{secToLabel(duration / 2)}</span><span>{secToLabel(duration)}</span>
    </div>
    {#each rows.filter((r) => r.visible) as row}
      <div style="display: flex; align-items: center; gap: 14px; padding: 10px 0; border-top: 1px solid var(--msp-border);">
        <div style="width: 118px; flex: none; display: flex; align-items: center; gap: 7px;">
          <div style="width: 8px; height: 8px; border-radius: 2px; background: {row.color};"></div>
          <span style="font-size: 12px; font-weight: 700;">{row.name}</span>
        </div>
        <div style="flex: 1; height: 26px; position: relative; background: var(--msp-panel-2); border-radius: 6px;">
          {#each row.boundaries as t}
            <button
              type="button"
              title={secToLabel(t)}
              on:click={() => (selectedBoundary = t)}
              style="position: absolute; top: 2px; bottom: 2px; left: {(t/duration)*100}%; width: 3px; background: {row.color}; border-radius: 1px; border: none; padding: 0; cursor: pointer;"
            ></button>
          {/each}
        </div>
      </div>
    {/each}
  </div>

  {#if matchDetail}
    <div style="margin-top: 20px; border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; max-width: 460px;">
      <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint); margin-bottom: 10px;">Boundary detail · {secToLabel(selectedBoundary)}</div>
      <div style="display: flex; flex-direction: column; gap: 7px; font-size: 12px;" class="msp-mono">
        {#each matchDetail as m}
          <div style="display: flex; justify-content: space-between;">
            <span style="color: var(--msp-text-dim);">{m.name}</span>
            {#if m.match}
              <span>{secToLabel(m.match.time)} {m.match.dist < 0.05 ? "✓" : `(±${m.match.dist.toFixed(1)}s)`}</span>
            {:else}
              <span style="color: var(--msp-text-faint);">no boundary</span>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>
{/if}
