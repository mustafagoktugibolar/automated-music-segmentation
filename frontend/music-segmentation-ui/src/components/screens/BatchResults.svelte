<script>
  import { batch } from "../../lib/batchStore.js";
  import { ALL_ALGOS } from "../../lib/analysisStore.js";

  export let goTo;

  function algoColor(id) {
    return ALL_ALGOS.find((a) => a.id === id)?.color || "var(--msp-text-faint)";
  }

  function avg(rows, key) {
    const vals = rows.map((r) => r[key]).filter((v) => v != null && !Number.isNaN(v));
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  }

  $: okRows = $batch.rows.filter((r) => !r.error);
  $: includedRows = okRows.filter((r) => !r.is_outlier);

  $: byAlgo = (() => {
    const groups = {};
    for (const r of includedRows) {
      const a = r.algorithm || "unknown";
      (groups[a] ||= []).push(r);
    }
    return Object.entries(groups)
      .map(([id, rows]) => ({
        id, name: ALL_ALGOS.find((a) => a.id === id)?.name || id, color: algoColor(id),
        f1: avg(rows, "f_measure"), precision: avg(rows, "precision"), recall: avg(rows, "recall"),
      }))
      .sort((a, b) => (b.f1 ?? 0) - (a.f1 ?? 0));
  })();
  $: maxF1 = Math.max(...byAlgo.map((r) => r.f1 ?? 0), 0.01);

  $: f1Histogram = (() => {
    const bins = Array.from({ length: 10 }, () => 0);
    for (const r of includedRows) {
      if (r.f_measure == null) continue;
      const idx = Math.min(9, Math.floor(r.f_measure * 10));
      bins[idx]++;
    }
    return bins;
  })();
  $: maxBin = Math.max(...f1Histogram, 1);

  function fmtPct(v) {
    return v == null ? "—" : (v * 100).toFixed(1) + "%";
  }

  function exportCsv() {
    const header = ["song_id", "title", "algorithm", "precision", "recall", "f_measure", "f1_3_0", "n_est", "n_ref", "is_outlier"];
    const lines = [header.join(",")];
    for (const r of okRows) {
      lines.push(header.map((k) => {
        const v = r[k];
        if (v == null) return "";
        const s = String(v).replace(/"/g, '""');
        return /[,"\n]/.test(s) ? `"${s}"` : s;
      }).join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "batch_eval_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

{#if $batch.rows.length === 0}
  <div style="max-width: 480px; margin: 60px auto; text-align: center;">
    <div style="font-size: 14px; color: var(--msp-text-dim); margin-bottom: 14px;">No batch results yet.</div>
    <button type="button" on:click={() => goTo(7)} style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 10px 20px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; font-family: inherit;">Run a batch evaluation</button>
  </div>
{:else}
<div>
  <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint); margin-bottom: 10px;">Algorithm ranking · avg. across {includedRows.length ? new Set(includedRows.map((r) => r.song_id)).size : 0} tracks</div>
  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; margin-bottom: 20px;">
    {#each byAlgo as row}
      <div style="display: flex; align-items: center; gap: 14px; padding: 8px 0;">
        <span style="width: 130px; font-size: 12.5px; font-weight: 700; display: flex; align-items: center; gap: 6px;"><div style="width: 8px; height: 8px; border-radius: 2px; background: {row.color};"></div>{row.name}</span>
        <div style="flex: 1; height: 16px; background: var(--msp-panel-2); border-radius: 4px; overflow: hidden;"><div style="width: {((row.f1 ?? 0) / maxF1) * 100}%; height: 100%; background: {row.color};"></div></div>
        <span style="width: 50px; text-align: right;" class="msp-mono">{fmtPct(row.f1)}</span>
      </div>
    {/each}
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
    <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px;">
      <div style="font-size: 12.5px; font-weight: 700; margin-bottom: 12px;">Precision–recall</div>
      <svg viewBox="0 0 220 180" style="width: 100%; height: 180px;">
        <line x1="30" y1="10" x2="30" y2="160" stroke="var(--msp-border-strong)"></line>
        <line x1="30" y1="160" x2="210" y2="160" stroke="var(--msp-border-strong)"></line>
        {#each byAlgo as pt}
          {#if pt.recall != null && pt.precision != null}
            <circle cx={30 + pt.recall * 180} cy={160 - pt.precision * 150} r="5" fill={pt.color}></circle>
          {/if}
        {/each}
      </svg>
    </div>
    <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px;">
      <div style="font-size: 12.5px; font-weight: 700; margin-bottom: 12px;">F1 distribution</div>
      <div style="display: flex; align-items: flex-end; gap: 5px; height: 140px;">
        {#each f1Histogram as c}
          <div style="flex: 1; height: {Math.round((c / maxBin) * 130)}px; background: var(--msp-accent); opacity: {c ? 0.5 + (c / maxBin) * 0.5 : 0.15}; border-radius: 3px 3px 0 0;"></div>
        {/each}
      </div>
    </div>
  </div>

  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
    <span style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint);">Per-track results</span>
    <button type="button" on:click={exportCsv} style="background: var(--msp-panel-2); border: 1px solid var(--msp-border); color: var(--msp-text); padding: 7px 12px; border-radius: 7px; font-size: 11.5px; font-weight: 700; cursor: pointer; font-family: inherit;">Export CSV</button>
  </div>
  <div style="border: 1px solid var(--msp-border); border-radius: 12px; overflow: hidden; max-height: 420px; overflow-y: auto;">
    <div style="display: grid; grid-template-columns: 1.6fr 1fr .8fr .8fr .8fr .6fr; padding: 10px 16px; background: var(--msp-panel-2); font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: var(--msp-text-faint); position: sticky; top: 0;">
      <span>Track</span><span>Algorithm</span><span>F1</span><span>F1 @3s</span><span>Segments</span><span>Status</span>
    </div>
    {#each okRows as r}
      <div style="display: grid; grid-template-columns: 1.6fr 1fr .8fr .8fr .8fr .6fr; padding: 10px 16px; align-items: center; border-top: 1px solid var(--msp-border); background: var(--msp-panel); font-size: 12px;">
        <span style="font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{r.title || r.song_id}</span>
        <span style="color: var(--msp-text-dim);">{r.algorithm}</span>
        <span class="msp-mono" style="font-weight: 700;">{fmtPct(r.f_measure)}</span>
        <span class="msp-mono" style="color: var(--msp-text-dim);">{fmtPct(r.f1_3_0)}</span>
        <span style="color: var(--msp-text-dim);">{r.n_est ?? "—"}/{r.n_ref ?? "—"}</span>
        {#if r.is_outlier}
          <span style="font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 5px; background: var(--msp-warn); color: var(--msp-accent-ink); width: fit-content;">Outlier</span>
        {:else}
          <span style="font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 5px; background: var(--msp-panel-2); color: var(--msp-ok); width: fit-content;">OK</span>
        {/if}
      </div>
    {/each}
  </div>
</div>
{/if}
