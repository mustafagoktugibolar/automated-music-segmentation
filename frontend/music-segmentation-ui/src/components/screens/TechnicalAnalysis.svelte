<script>
  import { analysis, setViewAlgo, ALL_ALGOS } from "../../lib/analysisStore.js";
  import { primaryAlgoAndSegments, availableAlgos, buildStructuralGroups, secToLabel } from "../../lib/segmentUtils.js";

  export let goTo;

  const N = 20;
  let diagnosticsOpen = false;

  function algoName(id) {
    return ALL_ALGOS.find((a) => a.id === id)?.name || id;
  }

  $: algoChoices = availableAlgos($analysis);
  $: ({ algo: primaryAlgo, segments } = primaryAlgoAndSegments($analysis));
  $: duration = Math.max(segments.length ? Math.max(...segments.map((s) => s.end)) : 0, 1);
  $: structuralGroups = buildStructuralGroups(segments);

  $: blockOf = Array.from({ length: N }, (_, i) => {
    const t = (i / N) * duration;
    const seg = segments.find((s) => t >= s.start && t < s.end) || segments[segments.length - 1];
    if (!seg) return "—";
    return seg.structural_label || structuralGroups[seg.structural_label || (seg.semantic_label || seg.label || seg.section_type)];
  });

  $: ssmCells = (() => {
    const cells = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const same = blockOf[r] === blockOf[c];
        const dist = Math.abs(r - c);
        const base = same ? 0.85 - dist * 0.002 : Math.max(0.04, 0.16 - dist * 0.006);
        cells.push(base);
      }
    }
    return cells;
  })();

  $: structuralLegend = [...new Set(blockOf.filter((b) => b !== "—"))];

  // Novelty curve: gaussian bumps anchored to the *real* segment boundaries.
  $: noveltyPoints = (() => {
    const peaks = segments.slice(1).map((s) => s.start);
    const pts = [];
    for (let x = 0; x <= 900; x += 15) {
      const t = (x / 900) * duration;
      let y = 8;
      peaks.forEach((p) => { const d = t - p; y += 70 * Math.exp(-(d * d) / 30); });
      pts.push(`${x},${(110 - y).toFixed(1)}`);
    }
    return pts.join(" ");
  })();
  $: noveltyPeakXs = segments.slice(1).map((s) => ((s.start / duration) * 900).toFixed(1));

  $: diagnosticsRows = Object.keys($analysis.results)
    .filter((k) => k.endsWith("__diagnostics"))
    .map((k) => ({ algo: k.replace("__diagnostics", ""), diag: $analysis.results[k] }))
    .filter((r) => r.diag && Object.keys(r.diag).length);
</script>

{#if segments.length === 0}
  <div style="max-width: 480px; margin: 60px auto; text-align: center;">
    <div style="font-size: 14px; color: var(--msp-text-dim); margin-bottom: 14px;">No results yet — run an analysis first.</div>
    <button type="button" on:click={() => goTo(0)} style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 10px 20px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; font-family: inherit;">Upload a song</button>
  </div>
{:else}
<div>
  <div style="display: flex; gap: 6px; margin-bottom: 18px; flex-wrap: wrap;">
    {#each algoChoices as a}
      <button
        type="button"
        on:click={() => setViewAlgo(a)}
        style="font-size: 10.5px; font-weight: 700; padding: 4px 10px; border-radius: 5px; cursor: pointer; font-family: inherit; border: 1px solid {a === primaryAlgo ? 'var(--msp-accent)' : 'var(--msp-border)'}; background: {a === primaryAlgo ? 'var(--msp-accent-bg)' : 'var(--msp-panel-2)'}; color: {a === primaryAlgo ? 'var(--msp-accent)' : 'var(--msp-text-dim)'};"
      >{algoName(a)}</button>
    {/each}
  </div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; grid-column: 1 / -1;">
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
      <span style="font-size: 12.5px; font-weight: 700;">Self-Similarity Matrix · {algoName(primaryAlgo)}</span>
    </div>
    <div style="font-size: 11px; color: var(--msp-text-faint); margin-bottom: 12px; line-height: 1.5;">Derived from {N} time buckets grouped by structural label — bright blocks mark repeated regions along the diagonal.</div>
    <div style="display: flex; gap: 16px; flex-wrap: wrap;">
      <div style="display: grid; grid-template-columns: repeat(20, 1fr); gap: 1px; width: 380px; height: 380px; background: var(--msp-border); border-radius: 6px; overflow: hidden;">
        {#each ssmCells as v}
          <div style="background: color-mix(in oklch, var(--msp-accent) {Math.round(v * 100)}%, var(--msp-panel));"></div>
        {/each}
      </div>
      <div style="flex: 1; min-width: 160px; display: flex; flex-direction: column; gap: 8px; justify-content: center;">
        <div style="font-size: 11px; color: var(--msp-text-faint); text-transform: uppercase; letter-spacing: .06em; font-weight: 700;">Block key</div>
        {#each structuralLegend as lg}
          <div style="display: flex; align-items: center; gap: 8px; font-size: 12px;">
            <div style="width: 14px; height: 14px; border-radius: 3px; border: 1.5px solid var(--msp-text-faint); background: var(--msp-panel-2);"></div>
            <span style="color: var(--msp-text-dim);">Group {lg}</span>
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; grid-column: 1 / -1;">
    <div style="font-size: 12.5px; font-weight: 700; margin-bottom: 4px;">Novelty curve</div>
    <div style="font-size: 11px; color: var(--msp-text-faint); margin-bottom: 12px;">Gaussian bumps anchored to the {segments.length - 1} real boundaries detected by {algoName(primaryAlgo)}.</div>
    <svg viewBox="0 0 900 120" style="width: 100%; height: 120px; display: block;">
      <polyline points={noveltyPoints} fill="none" stroke="var(--msp-accent)" stroke-width="2"></polyline>
      {#each noveltyPeakXs as x}
        <line x1={x} y1="0" x2={x} y2="120" stroke="var(--msp-border-strong)" stroke-dasharray="3,3"></line>
      {/each}
    </svg>
    <div style="display: flex; gap: 2px; margin-top: 6px;">
      {#each segments as seg}
        <div style="flex: 0 0 {((seg.end-seg.start)/duration)*100}%; height: 6px; background: var(--msp-accent); opacity: .5;"></div>
      {/each}
    </div>
  </div>

  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; grid-column: 1 / -1;">
    <button
      type="button"
      on:click={() => (diagnosticsOpen = !diagnosticsOpen)}
      style="display: flex; justify-content: space-between; align-items: center; cursor: pointer; width: 100%; background: none; border: none; padding: 0; font-family: inherit;"
    >
      <span style="font-size: 12.5px; font-weight: 700;">Worker diagnostics</span>
      <span class="msp-mono" style="font-size: 11px; color: var(--msp-text-faint);">{diagnosticsOpen ? "Hide ▴" : "Show ▾"}</span>
    </button>
    {#if diagnosticsOpen}
      <div style="margin-top: 14px; display: flex; flex-direction: column; gap: 10px;">
        {#if diagnosticsRows.length === 0}
          <div style="font-size: 12px; color: var(--msp-text-faint);">No diagnostics were returned for this run.</div>
        {/if}
        {#each diagnosticsRows as row}
          <div style="border: 1px solid var(--msp-border); border-radius: 8px; padding: 10px;">
            <div style="font-size: 11px; font-weight: 700; color: var(--msp-text-dim); margin-bottom: 6px;">{row.algo}</div>
            <pre class="msp-mono" style="margin: 0; font-size: 10.5px; color: var(--msp-text-faint); white-space: pre-wrap; word-break: break-word;">{JSON.stringify(row.diag, null, 2)}</pre>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>
</div>
{/if}
