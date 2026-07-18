<script>
  import { onMount } from "svelte";
  import {
    listDatasets, listDatasetTracks, getDatasetTrack, compareAlgorithms,
    uploadSegmentation, subscribeToTask, getSongStreamUrl,
  } from "../../lib/api.js";
  import { ALL_ALGOS } from "../../lib/analysisStore.js";
  import { secToLabel, boundaryTimes } from "../../lib/segmentUtils.js";

  let datasets = [];
  let selectedDatasetId = "";
  let tracks = [];
  let selectedTrack = null;

  let selectedAlgoIds = new Set(["fusion", "custom_librosa", "foote", "cnmf", "scluster"]);
  let toleranceSeconds = 0.5;

  let isRunning = false;
  let runError = "";
  let comparisonResults = null; // { algo: { metrics, segments? } }

  onMount(async () => {
    try { datasets = await listDatasets(); } catch (e) { runError = e.message; }
  });

  async function selectDataset(id) {
    selectedDatasetId = id;
    selectedTrack = null;
    tracks = [];
    comparisonResults = null;
    if (!id) return;
    try {
      const res = await listDatasetTracks(id, { hasGroundTruth: true });
      tracks = res.tracks || [];
    } catch (e) { runError = e.message; }
  }

  async function selectTrack(track) {
    comparisonResults = null;
    runError = "";
    try {
      selectedTrack = selectedDatasetId ? await getDatasetTrack(selectedDatasetId, track.track_id) : track;
    } catch (e) { runError = e.message; }
  }

  function toggleAlgo(id) {
    const next = new Set(selectedAlgoIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    selectedAlgoIds = next;
  }

  function triggerRun() {
    runComparison();
  }

  async function runComparison() {
    if (!selectedTrack) { runError = "Select a track first."; return; }
    if (selectedAlgoIds.size === 0) { runError = "Select at least one algorithm."; return; }

    isRunning = true;
    runError = "";
    comparisonResults = null;

    try {
      const sourceUrl = selectedTrack.song_id ? getSongStreamUrl(selectedTrack.song_id) : selectedTrack.audio_url;
      if (!sourceUrl) throw new Error("Track has no audio source.");
      const resp = await fetch(sourceUrl);
      if (!resp.ok) throw new Error(`Failed to fetch audio: ${resp.status} ${resp.statusText}`);
      const blob = await resp.blob();
      const file = new File([blob], `${selectedTrack.song_id || selectedTrack.track_id}.mp3`, { type: resp.headers.get("content-type") || "audio/mpeg" });

      const algos = Array.from(selectedAlgoIds);
      const taskId = await uploadSegmentation({ file, algorithms: algos, params: null });

      const taskIds = {};
      algos.forEach((a) => (taskIds[a] = taskId));

      await new Promise((resolve) => {
        const unsub = subscribeToTask(taskId, (data) => {
          if (data.status === "completed" || data.status === "failed") { unsub(); resolve(); }
        });
      });

      const res = await compareAlgorithms({
        trackId: selectedTrack.track_id,
        algorithmNames: algos,
        taskIds,
        toleranceSeconds,
      });
      comparisonResults = res?.comparison ?? null;
      if (!comparisonResults) runError = "Server returned no comparison data.";
    } catch (e) {
      runError = e.message;
    } finally {
      isRunning = false;
    }
  }

  function fmtPct(v) {
    return v == null ? "—" : (v * 100).toFixed(0) + "%";
  }

  $: primaryName = comparisonResults ? (comparisonResults.fusion ? "fusion" : Object.keys(comparisonResults)[0]) : null;
  $: primary = primaryName ? comparisonResults[primaryName] : null;

  $: matchMarks = (() => {
    if (!primary?.metrics || !selectedTrack?.ground_truth) return [];
    const gt = boundaryTimes(selectedTrack.ground_truth);
    const est = boundaryTimes(primary.segments || []);
    return gt.map((t) => {
      const matched = est.some((e) => Math.abs(e - t) <= toleranceSeconds);
      return { t, matched };
    });
  })();
</script>

<div style="display: grid; grid-template-columns: 300px 1fr; gap: 24px; align-items: start;">
  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px;">
    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint); margin-bottom: 12px;">Configuration</div>

    <div style="margin-bottom: 14px;">
      <div style="font-size: 12px; font-weight: 700; margin-bottom: 6px;">Dataset</div>
      <select
        bind:value={selectedDatasetId}
        on:change={(e) => selectDataset(e.currentTarget.value)}
        style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 8px 10px; border-radius: 8px; font-size: 12.5px;"
      >
        <option value="">— select dataset —</option>
        {#each datasets as ds}<option value={ds.dataset_id}>{ds.name}</option>{/each}
      </select>
    </div>

    {#if tracks.length > 0}
      <div style="margin-bottom: 14px;">
        <div style="font-size: 12px; font-weight: 700; margin-bottom: 6px;">Track (with ground truth)</div>
        <select
          on:change={(e) => { const t = tracks.find((tr) => tr.track_id === e.currentTarget.value); if (t) selectTrack(t); }}
          style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 8px 10px; border-radius: 8px; font-size: 12.5px;"
        >
          <option value="">— select track —</option>
          {#each tracks as t}<option value={t.track_id}>{t.title || t.song_id}</option>{/each}
        </select>
      </div>
    {:else if selectedDatasetId}
      <p style="font-size: 11.5px; color: var(--msp-text-faint);">No tracks with ground truth in this dataset.</p>
    {/if}

    <div style="margin-bottom: 14px;">
      <div style="font-size: 12px; font-weight: 700; margin-bottom: 8px;">Algorithms</div>
      <div style="display: flex; flex-wrap: wrap; gap: 6px;">
        {#each ALL_ALGOS as a}
          <button
            type="button"
            on:click={() => toggleAlgo(a.id)}
            style="font-size: 11px; font-weight: 700; padding: 5px 10px; border-radius: 6px; cursor: pointer; border: 1px solid {selectedAlgoIds.has(a.id) ? 'var(--msp-border-strong)' : 'var(--msp-border)'}; background: {selectedAlgoIds.has(a.id) ? 'var(--msp-accent-bg)' : 'transparent'}; color: {selectedAlgoIds.has(a.id) ? 'var(--msp-accent)' : 'var(--msp-text-faint)'}; font-family: inherit;"
          >{a.name}</button>
        {/each}
      </div>
    </div>

    <div style="margin-bottom: 18px;">
      <div style="font-size: 12px; font-weight: 700; margin-bottom: 6px;">Tolerance: <span class="msp-mono">±{toleranceSeconds}s</span></div>
      <input type="range" min="0.5" max="10" step="0.5" bind:value={toleranceSeconds} style="width: 100%; accent-color: var(--msp-accent);" />
    </div>

    <button
      type="button"
      on:click={triggerRun}
      disabled={isRunning || !selectedTrack}
      style="width: 100%; background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 11px; border-radius: 8px; font-size: 13px; font-weight: 800; cursor: pointer; font-family: inherit; opacity: {isRunning || !selectedTrack ? 0.5 : 1};"
    >{isRunning ? "Running…" : "Run comparison"}</button>

    {#if runError}
      <div style="margin-top: 12px; font-size: 11.5px; color: var(--msp-danger);">{runError}</div>
    {/if}
  </div>

  <div>
    {#if !comparisonResults}
      <div style="border: 1px dashed var(--msp-border-strong); border-radius: 12px; padding: 40px; text-align: center; color: var(--msp-text-faint); font-size: 13px;">
        Pick a dataset track and run a comparison to see precision, recall, and F1 against ground truth.
      </div>
    {:else}
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 14px; background: var(--msp-panel);">
          <div style="font-size: 11px; font-weight: 700; color: var(--msp-text-faint); text-transform: uppercase; letter-spacing: .05em;">Precision</div>
          <div style="font-size: 22px; font-weight: 800; margin-top: 8px;">{fmtPct(primary?.metrics?.precision)}</div>
        </div>
        <div style="border: 1px solid var(--msp-border); border-radius: 10px; padding: 14px; background: var(--msp-panel);">
          <div style="font-size: 11px; font-weight: 700; color: var(--msp-text-faint); text-transform: uppercase; letter-spacing: .05em;">Recall</div>
          <div style="font-size: 22px; font-weight: 800; margin-top: 8px;">{fmtPct(primary?.metrics?.recall)}</div>
        </div>
        <div style="border: 1px solid var(--msp-accent); border-radius: 10px; padding: 14px; background: var(--msp-accent-bg);">
          <div style="font-size: 11px; font-weight: 700; color: var(--msp-accent); text-transform: uppercase; letter-spacing: .05em;">F1 Score ({primaryName})</div>
          <div style="font-size: 22px; font-weight: 800; margin-top: 8px; color: var(--msp-accent);">{fmtPct(primary?.metrics?.f_measure)}</div>
        </div>
      </div>

      <div style="border: 1px solid var(--msp-border); border-radius: 12px; overflow: hidden; margin-bottom: 20px;">
        <div style="display: grid; grid-template-columns: 1.3fr .8fr .8fr .8fr .8fr; padding: 10px 16px; background: var(--msp-panel-2); font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--msp-text-faint);">
          <span>Algorithm</span><span>F1</span><span>Precision</span><span>Recall</span><span>Predicted / Ref</span>
        </div>
        {#each Object.entries(comparisonResults) as [name, result]}
          <div style="display: grid; grid-template-columns: 1.3fr .8fr .8fr .8fr .8fr; padding: 12px 16px; align-items: center; border-top: 1px solid var(--msp-border); background: var(--msp-panel); font-size: 12.5px;">
            <span style="font-weight: 700;">{name}</span>
            {#if result.error}
              <span style="grid-column: 2 / -1; color: var(--msp-danger); font-size: 11.5px;">{result.error}</span>
            {:else}
              <span class="msp-mono" style="font-weight: 700;">{fmtPct(result.metrics?.f_measure)}</span>
              <span class="msp-mono" style="color: var(--msp-text-dim);">{fmtPct(result.metrics?.precision)}</span>
              <span class="msp-mono" style="color: var(--msp-text-dim);">{fmtPct(result.metrics?.recall)}</span>
              <span class="msp-mono" style="color: var(--msp-text-dim);">{result.metrics?.n_boundaries_est ?? "—"} / {result.metrics?.n_boundaries_ref ?? "—"}</span>
            {/if}
          </div>
        {/each}
      </div>

      {#if matchMarks.length}
        <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px;">
          <div style="font-size: 12.5px; font-weight: 700; margin-bottom: 12px;">Boundary matching · {primaryName} vs. ground truth</div>
          <div style="position: relative; height: 44px; background: var(--msp-panel-2); border-radius: 6px;">
            {#each matchMarks as mm}
              <div
                title="{secToLabel(mm.t)} — {mm.matched ? 'matched' : 'missed'}"
                style="position: absolute; top: 6px; bottom: 6px; left: {(mm.t / Math.max(...boundaryTimes(selectedTrack.ground_truth), mm.t, 1)) * 100}%; width: 3px; border-radius: 2px; background: {mm.matched ? 'var(--msp-ok)' : 'var(--msp-danger)'};"
              ></div>
            {/each}
          </div>
          <div style="display: flex; gap: 18px; margin-top: 12px; font-size: 11.5px; color: var(--msp-text-dim);">
            <span style="display:flex;align-items:center;gap:6px;"><div style="width:9px;height:9px;border-radius:2px;background:var(--msp-ok);"></div>Matched ({matchMarks.filter((m) => m.matched).length})</span>
            <span style="display:flex;align-items:center;gap:6px;"><div style="width:9px;height:9px;border-radius:2px;background:var(--msp-danger);"></div>Missed ({matchMarks.filter((m) => !m.matched).length})</span>
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>
