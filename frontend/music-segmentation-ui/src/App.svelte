<script>
  import { onMount } from "svelte";
  import { analysis } from "./lib/analysisStore.js";

  import UploadSong from "./components/screens/UploadSong.svelte";
  import Configuration from "./components/screens/Configuration.svelte";
  import Processing from "./components/screens/Processing.svelte";
  import Overview from "./components/screens/Overview.svelte";
  import CompareAlgorithms from "./components/screens/CompareAlgorithms.svelte";
  import TechnicalAnalysis from "./components/screens/TechnicalAnalysis.svelte";
  import Evaluation from "./components/screens/Evaluation.svelte";
  import BatchEvaluation from "./components/screens/BatchEvaluation.svelte";
  import BatchResults from "./components/screens/BatchResults.svelte";
  import DatasetManagerScreen from "./components/screens/DatasetManagerScreen.svelte";

  const SCREENS = [
    UploadSong, Configuration, Processing, Overview, CompareAlgorithms,
    TechnicalAnalysis, Evaluation, BatchEvaluation, BatchResults, DatasetManagerScreen,
  ];

  const NAV_GROUPS = [
    { label: "Analyze", items: [{ id: 0, label: "Upload Song" }, { id: 1, label: "Configuration" }, { id: 2, label: "Processing" }] },
    { label: "Results", items: [{ id: 3, label: "Overview" }, { id: 4, label: "Compare Algorithms" }, { id: 5, label: "Technical Analysis" }, { id: 6, label: "Evaluation" }] },
    { label: "Research", items: [{ id: 7, label: "Batch Evaluation" }, { id: 8, label: "Batch Results" }, { id: 9, label: "Dataset Manager" }] },
  ];

  const TITLES = [
    "Analyze Song", "Analysis Configuration", "Processing", "Analysis Result", "Algorithm Comparison",
    "Technical Analysis", "Evaluation", "Batch Evaluation", "Batch Results", "Dataset Manager",
  ];
  const SUBTITLES = [
    "Upload a song or select one from your library", "Choose algorithms and parameters before running analysis",
    "Running selected algorithms", "Fusion recommended result",
    "Aligned boundaries across all algorithms", "Self-similarity, novelty, and expert feature views",
    "Precision, recall, and F1 against ground truth", "Configure and monitor a multi-track run",
    "Aggregate performance across the dataset", "Manage datasets, songs, and reference annotations",
  ];

  let screen = 0;
  let theme = "dark";

  onMount(() => {
    theme = localStorage.getItem("msp-theme") || "dark";
    document.documentElement.setAttribute("data-theme", theme);
  });

  function toggleTheme() {
    theme = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("msp-theme", theme);
  }

  function goTo(id) {
    screen = id;
  }

  $: status = $analysis.status;
  $: workersActive = status === "processing" || status === "uploading" ? 1 : 0;

  $: CurrentScreen = SCREENS[screen];
</script>

<div class="msp-app" style="min-height: 100vh; display: flex;">
  <!-- SIDEBAR -->
  <div style="width: 232px; flex: none; background: var(--msp-bg-elevated); border-right: 1px solid var(--msp-border); display: flex; flex-direction: column; position: sticky; top: 0; height: 100vh;">
    <div style="padding: 20px 20px 16px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid var(--msp-border);">
      <div style="width: 30px; height: 30px; border-radius: 8px; background: var(--msp-accent); display: flex; align-items: center; justify-content: center; flex: none;">
        <div style="width: 3px; height: 14px; background: var(--msp-accent-ink); border-radius: 2px;"></div>
        <div style="width: 3px; height: 20px; background: var(--msp-accent-ink); border-radius: 2px; margin: 0 2px;"></div>
        <div style="width: 3px; height: 10px; background: var(--msp-accent-ink); border-radius: 2px;"></div>
      </div>
      <div>
        <div style="font-size: 13px; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2;">Segmentation Lab</div>
        <div class="msp-mono" style="font-size: 10.5px; color: var(--msp-text-faint);">MIR analysis platform</div>
      </div>
    </div>

    <div style="flex: 1; overflow-y: auto; padding: 14px 12px;">
      {#each NAV_GROUPS as grp}
        <div style="margin-bottom: 18px;">
          <div style="font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--msp-text-faint); padding: 0 8px 6px;">{grp.label}</div>
          {#each grp.items as item}
            <div
              role="button" tabindex="0"
              on:click={() => goTo(item.id)}
              on:keydown={(e) => e.key === "Enter" && goTo(item.id)}
              style="display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 7px; cursor: pointer; margin-bottom: 2px; font-size: 12.5px; font-weight: 600;"
              class="msp-nav-item"
            >
              <div style="width: 5px; height: 5px; border-radius: 50%; background: {screen === item.id ? 'var(--msp-accent)' : 'transparent'};"></div>
              <span style="color: {screen === item.id ? 'var(--msp-text)' : 'var(--msp-text-dim)'};">{item.label}</span>
            </div>
          {/each}
        </div>
      {/each}
    </div>

    <div style="padding: 14px 20px; border-top: 1px solid var(--msp-border); display: flex; align-items: center; justify-content: space-between;">
      <span class="msp-mono" style="font-size: 11px; color: var(--msp-text-faint);">v1.0.0</span>
      <button
        type="button"
        on:click={toggleTheme}
        aria-label="Toggle theme"
        style="width: 40px; height: 22px; border-radius: 999px; background: var(--msp-panel-2); border: 1px solid var(--msp-border-strong); position: relative; cursor: pointer; padding: 0;"
      >
        <div style="position: absolute; top: 2px; left: {theme === 'dark' ? '20px' : '2px'}; width: 16px; height: 16px; border-radius: 50%; background: var(--msp-accent); transition: left .15s;"></div>
      </button>
    </div>
  </div>

  <!-- MAIN -->
  <div style="flex: 1; min-width: 0;">
    <div style="padding: 16px 32px; border-bottom: 1px solid var(--msp-border); display: flex; align-items: center; justify-content: space-between; background: var(--msp-bg-elevated); position: sticky; top: 0; z-index: 5;">
      <div>
        <div style="font-size: 16px; font-weight: 700; letter-spacing: -0.01em;">{TITLES[screen]}</div>
        <div style="font-size: 12px; color: var(--msp-text-dim); margin-top: 2px;">{SUBTITLES[screen]}</div>
      </div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 7px; background: var(--msp-panel-2); border: 1px solid var(--msp-border);">
          <div style="width: 6px; height: 6px; border-radius: 50%; background: {workersActive ? 'var(--msp-accent)' : 'var(--msp-ok)'};"></div>
          <span class="msp-mono" style="font-size: 11.5px; color: var(--msp-text-dim);">workers {workersActive ? "busy" : "idle"} · {workersActive}/4</span>
        </div>
      </div>
    </div>

    <div style="padding: 28px 32px 60px;">
      <svelte:component this={CurrentScreen} {goTo} />
    </div>
  </div>
</div>

<style>
  :global(.msp-nav-item:hover) {
    background: var(--msp-panel-2);
  }
  :global(body) {
    background: var(--msp-bg);
  }
</style>
