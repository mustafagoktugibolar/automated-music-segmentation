<script>
  import { onMount } from "svelte";
  import {
    listDatasets, createDataset, importSalami, listDatasetTracks, uploadTrack, segmentSongFromStorage,
  } from "../../lib/api.js";
  import { setSourceTrack } from "../../lib/analysisStore.js";

  export let goTo;

  let datasets = [];
  let selectedDataset = null;
  let tracks = [];

  let isImporting = false;
  let importError = "";

  let showNewDataset = false;
  let newName = "";
  let newDesc = "";
  let createError = "";

  let showUpload = false;
  let uploadFile = null;
  let uploadGTFile = null;
  let uploadTitle = "";
  let uploadArtist = "";
  let uploadError = "";
  let isUploading = false;

  let segmentMessage = "";

  onMount(loadDatasets);

  async function loadDatasets() {
    try { datasets = await listDatasets(); } catch (e) { importError = e.message; }
  }

  async function selectDataset(ds) {
    selectedDataset = ds;
    tracks = [];
    try {
      const res = await listDatasetTracks(ds.dataset_id);
      tracks = res.tracks || [];
    } catch (e) { importError = e.message; }
  }

  async function doImportSalami() {
    isImporting = true;
    importError = "";
    try {
      const res = await importSalami();
      await loadDatasets();
      const salami = datasets.find((d) => d.name === "SALAMI");
      if (salami) await selectDataset(salami);
      importError = `Imported ${res.tracks_imported} tracks.`;
    } catch (e) {
      importError = e.message;
    } finally {
      isImporting = false;
    }
  }

  async function doCreateDataset() {
    if (!newName.trim()) { createError = "Name is required."; return; }
    createError = "";
    try {
      await createDataset({ name: newName.trim(), description: newDesc.trim() || null });
      newName = ""; newDesc = ""; showNewDataset = false;
      await loadDatasets();
    } catch (e) { createError = e.message; }
  }

  async function doUploadTrack() {
    if (!uploadFile || !selectedDataset) { uploadError = "Select an audio file."; return; }
    isUploading = true;
    uploadError = "";
    try {
      await uploadTrack(selectedDataset.dataset_id, {
        file: uploadFile, groundTruthCsv: uploadGTFile,
        title: uploadTitle.trim() || uploadFile.name, artist: uploadArtist.trim() || null,
      });
      uploadFile = null; uploadGTFile = null; uploadTitle = ""; uploadArtist = ""; showUpload = false;
      await selectDataset(selectedDataset);
      await loadDatasets();
    } catch (e) {
      uploadError = e.message;
    } finally {
      isUploading = false;
    }
  }

  function analyzeTrack(t) {
    setSourceTrack(t);
    goTo(1);
  }

  async function segmentInPlace(t) {
    if (!t.song_id) return;
    segmentMessage = "";
    try {
      const res = await segmentSongFromStorage(t.song_id);
      segmentMessage = `Task dispatched: ${res.task_id}`;
    } catch (e) {
      segmentMessage = `Failed: ${e.message}`;
    }
  }

  function fmtDuration(sec) {
    if (sec == null) return "—";
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }
</script>

<div>
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
    <span style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint);">Datasets</span>
    <div style="display: flex; gap: 8px;">
      <button type="button" on:click={doImportSalami} disabled={isImporting}
        style="background: var(--msp-panel-2); border: 1px solid var(--msp-border); color: var(--msp-text); padding: 8px 14px; border-radius: 7px; font-size: 12px; font-weight: 700; cursor: pointer; font-family: inherit; opacity: {isImporting ? 0.6 : 1};"
      >{isImporting ? "Importing…" : "Import SALAMI"}</button>
      <button type="button" on:click={() => (showNewDataset = !showNewDataset)}
        style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 8px 14px; border-radius: 7px; font-size: 12px; font-weight: 700; cursor: pointer; font-family: inherit;"
      >New dataset</button>
    </div>
  </div>

  {#if importError}<div style="font-size: 11.5px; color: var(--msp-text-dim); margin-bottom: 10px;">{importError}</div>{/if}

  {#if showNewDataset}
    <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 14px; margin-bottom: 16px; display: flex; gap: 10px; align-items: flex-end;">
      <div style="flex: 1;">
        <div style="font-size: 11px; color: var(--msp-text-faint); margin-bottom: 4px;">Name</div>
        <input bind:value={newName} style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 7px 10px; border-radius: 7px; font-size: 12.5px;" />
      </div>
      <div style="flex: 1;">
        <div style="font-size: 11px; color: var(--msp-text-faint); margin-bottom: 4px;">Description</div>
        <input bind:value={newDesc} style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 7px 10px; border-radius: 7px; font-size: 12.5px;" />
      </div>
      <button type="button" on:click={doCreateDataset} style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 8px 14px; border-radius: 7px; font-size: 12px; font-weight: 700; cursor: pointer; font-family: inherit;">Create</button>
    </div>
    {#if createError}<div style="font-size: 11.5px; color: var(--msp-danger); margin-bottom: 10px;">{createError}</div>{/if}
  {/if}

  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 26px;">
    {#each datasets as ds}
      <button
        type="button" on:click={() => selectDataset(ds)}
        style="text-align: left; border: 1px solid {selectedDataset?.dataset_id === ds.dataset_id ? 'var(--msp-accent)' : 'var(--msp-border)'}; border-radius: 12px; background: var(--msp-panel); padding: 14px; cursor: pointer; font-family: inherit;"
      >
        <div style="font-size: 13px; font-weight: 700; margin-bottom: 6px;">{ds.name}</div>
        <div style="font-size: 11.5px; color: var(--msp-text-faint);">{ds.track_count ?? "—"} songs</div>
      </button>
    {/each}
  </div>

  {#if selectedDataset}
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
      <span style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint);">{selectedDataset.name} · songs</span>
      <div style="display: flex; align-items: center; gap: 10px;">
        {#if segmentMessage}<span style="font-size: 11px; color: var(--msp-text-faint);">{segmentMessage}</span>{/if}
        <button type="button" on:click={() => (showUpload = !showUpload)}
          style="background: var(--msp-panel-2); border: 1px solid var(--msp-border); color: var(--msp-text); padding: 6px 12px; border-radius: 7px; font-size: 11.5px; font-weight: 700; cursor: pointer; font-family: inherit;"
        >{showUpload ? "Cancel" : "Upload track"}</button>
      </div>
    </div>

    {#if showUpload}
      <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 14px; margin-bottom: 16px;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
          <input placeholder="Title" bind:value={uploadTitle} style="border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 7px 10px; border-radius: 7px; font-size: 12.5px;" />
          <input placeholder="Artist" bind:value={uploadArtist} style="border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 7px 10px; border-radius: 7px; font-size: 12.5px;" />
        </div>
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
          <label style="font-size: 11.5px; color: var(--msp-text-faint);">Audio
            <input type="file" accept=".mp3,.wav,.flac,.ogg,.m4a" on:change={(e) => (uploadFile = e.currentTarget.files?.[0] ?? null)} />
          </label>
          <label style="font-size: 11.5px; color: var(--msp-text-faint);">Ground truth CSV (optional)
            <input type="file" accept=".csv" on:change={(e) => (uploadGTFile = e.currentTarget.files?.[0] ?? null)} />
          </label>
        </div>
        {#if uploadError}<div style="font-size: 11.5px; color: var(--msp-danger); margin-bottom: 8px;">{uploadError}</div>{/if}
        <button type="button" on:click={doUploadTrack} disabled={isUploading}
          style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 8px 16px; border-radius: 7px; font-size: 12px; font-weight: 700; cursor: pointer; font-family: inherit; opacity: {isUploading ? 0.6 : 1};"
        >{isUploading ? "Uploading…" : "Upload"}</button>
      </div>
    {/if}

    <div style="border: 1px solid var(--msp-border); border-radius: 12px; overflow: hidden;">
      <div style="display: grid; grid-template-columns: 1.6fr .7fr 1fr 1fr 1fr; padding: 10px 16px; background: var(--msp-panel-2); font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: var(--msp-text-faint);">
        <span>Song</span><span>Duration</span><span>Annotation</span><span></span><span>Actions</span>
      </div>
      {#each tracks as t}
        <div style="display: grid; grid-template-columns: 1.6fr .7fr 1fr 1fr 1fr; padding: 11px 16px; align-items: center; border-top: 1px solid var(--msp-border); background: var(--msp-panel); font-size: 12.5px;">
          <span style="font-weight: 600;">{t.title || t.song_id}</span>
          <span class="msp-mono" style="color: var(--msp-text-dim);">{fmtDuration(t.duration_seconds)}</span>
          <span style="font-size: 10.5px; font-weight: 700; color: {t.has_ground_truth ? 'var(--msp-ok)' : 'var(--msp-warn)'};">{t.has_ground_truth ? "Verified" : "No annotation"}</span>
          <span></span>
          <div style="display: flex; gap: 6px;">
            <button type="button" on:click={() => analyzeTrack(t)} style="background: var(--msp-panel-2); border: 1px solid var(--msp-border); color: var(--msp-text); padding: 5px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; cursor: pointer; font-family: inherit;">Analyze</button>
            <button type="button" on:click={() => segmentInPlace(t)} style="background: transparent; border: 1px solid var(--msp-border); color: var(--msp-text-dim); padding: 5px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; cursor: pointer; font-family: inherit;">Segment</button>
          </div>
        </div>
      {/each}
      {#if tracks.length === 0}
        <div style="padding: 20px; text-align: center; font-size: 12px; color: var(--msp-text-faint);">No tracks in this dataset yet.</div>
      {/if}
    </div>
  {/if}
</div>
