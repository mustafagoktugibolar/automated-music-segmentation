<script>
  import { setFile, setSourceTrack } from "../../lib/analysisStore.js";
  import { listDatasets, listDatasetTracks } from "../../lib/api.js";

  export let goTo;

  let dragOver = false;
  let libraryOpen = false;
  let datasets = [];
  let selectedDatasetId = "";
  let tracks = [];
  let loadingTracks = false;
  let loadError = "";

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

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    goTo(1);
  }

  function onFileInput(e) {
    pickFile(e.currentTarget.files?.[0] ?? null);
  }

  function onDrop(e) {
    e.preventDefault();
    dragOver = false;
    pickFile(e.dataTransfer?.files?.[0] ?? null);
  }

  async function openLibrary() {
    libraryOpen = !libraryOpen;
    if (libraryOpen && datasets.length === 0) {
      try {
        datasets = await listDatasets();
      } catch (e) {
        loadError = e.message;
      }
    }
  }

  async function selectDataset(id) {
    selectedDatasetId = id;
    tracks = [];
    loadError = "";
    if (!id) return;
    loadingTracks = true;
    try {
      const res = await listDatasetTracks(id, { pageSize: 200 });
      tracks = res.tracks || [];
    } catch (e) {
      loadError = e.message;
    } finally {
      loadingTracks = false;
    }
  }

  function pickTrack(track) {
    setSourceTrack(track);
    goTo(1);
  }
</script>

<div style="max-width: 760px; margin: 24px auto 0;">
  <div style="text-align: center; margin-bottom: 28px;">
    <div style="font-size: 21px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px;">Upload a song to detect its structure</div>
    <div style="font-size: 13.5px; color: var(--msp-text-dim); max-width: 480px; margin: 0 auto; line-height: 1.6;">Detect structural boundaries, repeated sections, and possible Verse, Chorus, Bridge, and Outro regions — automatically.</div>
  </div>

  <button
    type="button"
    on:dragover={(e) => { e.preventDefault(); dragOver = true; }}
    on:dragleave={() => (dragOver = false)}
    on:drop={onDrop}
    on:click={() => document.getElementById("upload-song-input").click()}
    style="width: 100%; border: 2px dashed {dragOver ? 'var(--msp-accent)' : 'var(--msp-border-strong)'}; border-radius: 16px; padding: 56px 32px; text-align: center; background: var(--msp-panel); transition: border-color .15s; cursor: pointer;"
  >
    <div style="width: 56px; height: 56px; border-radius: 14px; background: var(--msp-accent-bg); display: flex; align-items: center; justify-content: center; margin: 0 auto 18px;">
      <div style="width: 20px; height: 24px; border: 2px solid var(--msp-accent); border-radius: 3px; position: relative;">
        <div style="position: absolute; left: 50%; top: -8px; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 8px solid var(--msp-accent); transform: translateX(-50%);"></div>
      </div>
    </div>
    <div style="font-size: 14.5px; font-weight: 700; margin-bottom: 4px;">Drag and drop an audio file</div>
    <div style="font-size: 12px; color: var(--msp-text-faint); margin-bottom: 18px;">MP3, WAV, FLAC, OGG, M4A</div>
    <span style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 10px 20px; border-radius: 8px; font-size: 13px; font-weight: 700; display: inline-block;">Choose Audio File</span>
  </button>
  <input id="upload-song-input" type="file" accept=".mp3,.wav,.flac,.ogg,.m4a" style="display: none;" on:change={onFileInput} />

  <div style="display: flex; align-items: center; gap: 12px; margin: 22px 0;">
    <div style="flex: 1; height: 1px; background: var(--msp-border);"></div>
    <span style="font-size: 11px; color: var(--msp-text-faint); font-weight: 600;">OR</span>
    <div style="flex: 1; height: 1px; background: var(--msp-border);"></div>
  </div>

  <div style="border: 1px solid var(--msp-border); border-radius: 12px; padding: 16px 18px; background: var(--msp-panel);">
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <div>
        <div style="font-size: 13.5px; font-weight: 700; margin-bottom: 2px;">Select an existing song</div>
        <div style="font-size: 12px; color: var(--msp-text-faint);">Choose a track from a dataset already in storage</div>
      </div>
      <button
        type="button"
        on:click={openLibrary}
        style="background: transparent; color: var(--msp-text); border: 1px solid var(--msp-border-strong); padding: 8px 16px; border-radius: 8px; font-size: 12.5px; font-weight: 700; cursor: pointer; font-family: inherit;"
      >{libraryOpen ? "Hide" : "Browse Library"}</button>
    </div>

    {#if libraryOpen}
      <div style="margin-top: 14px; border-top: 1px solid var(--msp-border); padding-top: 14px;">
        <select
          bind:value={selectedDatasetId}
          on:change={(e) => selectDataset(e.currentTarget.value)}
          style="width: 100%; border: 1px solid var(--msp-border-strong); background: var(--msp-panel-2); color: var(--msp-text); padding: 8px 10px; border-radius: 8px; font-size: 12.5px; margin-bottom: 10px;"
        >
          <option value="">— select dataset —</option>
          {#each datasets as ds}
            <option value={ds.dataset_id}>{ds.name}</option>
          {/each}
        </select>

        {#if loadError}
          <div style="font-size: 12px; color: var(--msp-danger);">{loadError}</div>
        {/if}
        {#if loadingTracks}
          <div style="font-size: 12px; color: var(--msp-text-faint);">Loading tracks…</div>
        {:else if selectedDatasetId && tracks.length === 0}
          <div style="font-size: 12px; color: var(--msp-text-faint);">No tracks in this dataset.</div>
        {/if}

        <div style="max-height: 220px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px;">
          {#each tracks as t}
            <button
              type="button"
              on:click={() => pickTrack(t)}
              style="text-align: left; display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; border-radius: 7px; border: 1px solid var(--msp-border); background: var(--msp-panel-2); cursor: pointer; font-family: inherit;"
            >
              <span style="font-size: 12.5px; font-weight: 600; color: var(--msp-text);">{t.title || t.song_id}</span>
              <span style="font-size: 10.5px; color: var(--msp-text-faint);">{t.has_ground_truth ? "ground truth ✓" : ""}</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </div>
</div>
