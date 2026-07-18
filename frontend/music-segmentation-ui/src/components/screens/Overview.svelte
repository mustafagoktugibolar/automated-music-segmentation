<script>
  import { analysis, selectSegment, setViewAlgo, ALL_ALGOS } from "../../lib/analysisStore.js";
  import { primaryAlgoAndSegments, availableAlgos, buildStructuralGroups, segLabel, secToLabel } from "../../lib/segmentUtils.js";

  export let goTo;

  const PALETTE = [
    "var(--msp-sem-intro)", "var(--msp-sem-verse)", "var(--msp-sem-prechorus)", "var(--msp-sem-chorus)",
    "var(--msp-sem-bridge)", "var(--msp-sem-instrumental)", "var(--msp-sem-outro)", "var(--msp-sem-unknown)",
  ];

  let audioEl;
  let currentTime = 0;
  let audioDuration = 0;
  let isPlaying = false;
  let waveformPeaks = [];
  let decodedUrl = "";

  $: results = $analysis.results;
  $: algoChoices = availableAlgos($analysis);
  $: ({ algo: primaryAlgo, segments } = primaryAlgoAndSegments($analysis));
  $: duration = Math.max(audioDuration, segments.length ? Math.max(...segments.map((s) => s.end)) : 0, 1);

  $: uniqueLabels = [...new Set(segments.map((s) => segLabel(s)))];
  $: colorMap = Object.fromEntries(uniqueLabels.map((l, i) => [l, PALETTE[i % PALETTE.length]]));
  $: structuralGroups = buildStructuralGroups(segments);

  $: songTitle = $analysis.file ? $analysis.file.name.replace(/\.[^.]+$/, "") : ($analysis.sourceTrack?.title || $analysis.sourceTrack?.song_id || "Untitled");
  $: songSub = $analysis.sourceTrack?.artist ? $analysis.sourceTrack.artist : ($analysis.file ? $analysis.file.type || "audio" : "");

  $: sel = segments[$analysis.selectedSegmentIndex] || segments[0] || null;

  function algoName(id) {
    return ALL_ALGOS.find((a) => a.id === id)?.name || id;
  }

  function segStruct(s) {
    return s.structural_label || structuralGroups[s.structural_label || segLabel(s)] || "—";
  }

  function similarSegments(target) {
    if (!target) return [];
    const key = target.structural_label || segLabel(target);
    return segments
      .filter((s) => s !== target && (s.structural_label || segLabel(s)) === key)
      .map((s) => secToLabel(s.start));
  }

  function pick(i) {
    selectSegment(i);
    if (audioEl) {
      audioEl.currentTime = segments[i]?.start ?? 0;
    }
  }

  function seekTo(fraction) {
    if (!audioEl) return;
    audioEl.currentTime = fraction * duration;
  }

  function togglePlay() {
    if (!audioEl) return;
    if (audioEl.paused) audioEl.play();
    else audioEl.pause();
  }

  async function decodeWaveform(url) {
    if (!url || url === decodedUrl) return;
    decodedUrl = url;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      const buf = await (await fetch(url)).arrayBuffer();
      const audioBuffer = await ctx.decodeAudioData(buf);
      const channel = audioBuffer.getChannelData(0);
      const buckets = 80;
      const blockSize = Math.max(1, Math.floor(channel.length / buckets));
      const peaks = [];
      for (let i = 0; i < buckets; i++) {
        let max = 0;
        const start = i * blockSize;
        for (let j = start; j < start + blockSize && j < channel.length; j++) {
          const v = Math.abs(channel[j]);
          if (v > max) max = v;
        }
        peaks.push(max);
      }
      const peakMax = Math.max(...peaks, 0.01);
      waveformPeaks = peaks.map((p) => Math.max(0.08, p / peakMax));
      ctx.close();
    } catch (e) {
      waveformPeaks = Array.from({ length: 80 }, () => 0.25 + Math.random() * 0.3);
    }
  }

  $: if ($analysis.audioUrl) decodeWaveform($analysis.audioUrl);
</script>

{#if segments.length === 0}
  <div style="max-width: 480px; margin: 60px auto; text-align: center;">
    <div style="font-size: 14px; color: var(--msp-text-dim); margin-bottom: 14px;">No results yet — run an analysis first.</div>
    <button type="button" on:click={() => goTo(0)} style="background: var(--msp-accent); color: var(--msp-accent-ink); border: none; padding: 10px 20px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer; font-family: inherit;">Upload a song</button>
  </div>
{:else}
<div>
  <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 20px;">
    <div style="width: 68px; height: 68px; border-radius: 10px; background: var(--msp-panel-2); flex: none; display: flex; align-items: center; justify-content: center; font-size: 26px;">♪</div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 19px; font-weight: 800; letter-spacing: -0.01em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{songTitle}</div>
      <div class="msp-mono" style="font-size: 12.5px; color: var(--msp-text-dim); margin-top: 3px;">{songSub} · {secToLabel(duration)}</div>
      <div style="display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap;">
        {#each algoChoices as a}
          <button
            type="button"
            on:click={() => setViewAlgo(a)}
            style="font-size: 10.5px; font-weight: 700; padding: 4px 10px; border-radius: 5px; cursor: pointer; font-family: inherit; border: 1px solid {a === primaryAlgo ? 'var(--msp-accent)' : 'var(--msp-border)'}; background: {a === primaryAlgo ? 'var(--msp-accent-bg)' : 'var(--msp-panel-2)'}; color: {a === primaryAlgo ? 'var(--msp-accent)' : 'var(--msp-text-dim)'};"
          >{algoName(a)}</button>
        {/each}
      </div>
    </div>
    <div style="display: flex; gap: 8px;">
      <button type="button" on:click={() => goTo(1)} style="background: transparent; border: 1px solid var(--msp-border-strong); color: var(--msp-text); padding: 9px 14px; border-radius: 8px; font-size: 12px; font-weight: 700; cursor: pointer; font-family: inherit;">Re-run</button>
      <button type="button" on:click={() => goTo(4)} style="background: var(--msp-panel-2); border: 1px solid var(--msp-border); color: var(--msp-text); padding: 9px 14px; border-radius: 8px; font-size: 12px; font-weight: 700; cursor: pointer; font-family: inherit;">Compare ▸</button>
    </div>
  </div>

  {#if $analysis.audioUrl}
    <audio
      bind:this={audioEl}
      src={$analysis.audioUrl}
      on:timeupdate={() => (currentTime = audioEl.currentTime)}
      on:loadedmetadata={() => (audioDuration = audioEl.duration || 0)}
      on:play={() => (isPlaying = true)}
      on:pause={() => (isPlaying = false)}
      style="display: none;"
    ></audio>

    <div style="display: flex; align-items: center; gap: 16px; padding: 12px 18px; border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); margin-bottom: 18px;">
      <button
        type="button" on:click={togglePlay}
        style="width: 34px; height: 34px; border-radius: 50%; background: var(--msp-accent); border: none; display: flex; align-items: center; justify-content: center; cursor: pointer; flex: none;"
      >
        {#if isPlaying}
          <div style="width: 10px; height: 10px; display: flex; gap: 3px;"><div style="width:3px;height:10px;background:var(--msp-accent-ink);"></div><div style="width:3px;height:10px;background:var(--msp-accent-ink);"></div></div>
        {:else}
          <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 9px solid var(--msp-accent-ink); margin-left: 2px;"></div>
        {/if}
      </button>
      <span class="msp-mono" style="font-size: 12px; color: var(--msp-text-dim); width: 46px;">{secToLabel(currentTime)}</span>
      <button
        type="button"
        aria-label="Seek"
        style="flex: 1; height: 4px; background: var(--msp-panel-2); border-radius: 2px; position: relative; border: none; padding: 0; cursor: pointer;"
        on:click={(e) => { const r = e.currentTarget.getBoundingClientRect(); seekTo((e.clientX - r.left) / r.width); }}
      >
        <div style="width: {(currentTime / duration) * 100}%; height: 100%; background: var(--msp-accent); border-radius: 2px;"></div>
        <div style="position: absolute; left: {(currentTime / duration) * 100}%; top: 50%; width: 11px; height: 11px; background: var(--msp-accent); border: 2px solid var(--msp-bg-elevated); border-radius: 50%; transform: translate(-50%,-50%);"></div>
      </button>
      <span class="msp-mono" style="font-size: 12px; color: var(--msp-text-faint); width: 46px;">{secToLabel(duration)}</span>
    </div>
  {/if}

  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; margin-bottom: 16px;">
    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint); margin-bottom: 10px;">Waveform</div>
    <button
      type="button"
      style="position: relative; height: 96px; width: 100%; background: transparent; border: none; padding: 0; cursor: pointer;"
      on:click={(e) => { const r = e.currentTarget.getBoundingClientRect(); seekTo((e.clientX - r.left) / r.width); }}
    >
      <div style="position: absolute; inset: 0; display: flex; align-items: center; gap: 2px;">
        {#each waveformPeaks as h}
          <div style="flex:1; height:{Math.round(h * 90)}px; background: var(--msp-border-strong); border-radius: 2px; opacity: .7;"></div>
        {/each}
      </div>
      {#each segments as seg}
        <div style="position:absolute; top:0; bottom:0; left:{(seg.start/duration)*100}%; width:{((seg.end-seg.start)/duration)*100}%; background:{colorMap[segLabel(seg)]}; opacity:.14;"></div>
      {/each}
      {#each segments.slice(1) as seg}
        <div style="position:absolute; top:0; bottom:0; left:{(seg.start/duration)*100}%; width:1px; background: var(--msp-border-strong);"></div>
      {/each}
      <div style="position: absolute; top: 0; bottom: 0; left: {(currentTime/duration)*100}%; width: 2px; background: var(--msp-accent-ink); box-shadow: 0 0 8px var(--msp-accent);"></div>
    </button>
  </div>

  <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; margin-bottom: 24px;">
    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint); margin-bottom: 10px;">Structural timeline · {algoName(primaryAlgo)}</div>
    <div style="display: flex; gap: 2px; height: 46px; border-radius: 8px; overflow: hidden;">
      {#each segments as seg, i}
        <button
          type="button"
          on:click={() => pick(i)}
          style="flex: 0 0 {((seg.end-seg.start)/duration)*100}%; min-width: 0; background: {colorMap[segLabel(seg)]}; display: flex; align-items: center; justify-content: center; cursor: pointer; opacity: {i === $analysis.selectedSegmentIndex ? 1 : .82}; outline: {i === $analysis.selectedSegmentIndex ? '2px solid var(--msp-text)' : 'none'}; outline-offset: -2px; overflow: hidden; border: none; padding: 0;"
        >
          <span style="font-size: 10.5px; font-weight: 700; color: var(--msp-accent-ink); text-shadow: 0 1px 2px rgba(0,0,0,.15); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 0 4px;">{segLabel(seg)}</span>
        </button>
      {/each}
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 10px; color: var(--msp-text-faint);" class="msp-mono">
      <span>0:00</span><span>{secToLabel(duration)}</span>
    </div>
  </div>

  <div style="display: grid; grid-template-columns: 1fr 320px; gap: 20px; align-items: start;">
    <div>
      <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--msp-text-faint); margin-bottom: 10px;">All segments</div>
      <div style="border: 1px solid var(--msp-border); border-radius: 12px; overflow: hidden;">
        {#each segments as seg, i}
          <button
            type="button"
            on:click={() => pick(i)}
            style="width: 100%; text-align: left; display: flex; align-items: center; gap: 12px; padding: 11px 14px; border-top: {i === 0 ? '0' : '1px'} solid var(--msp-border); background: {i === $analysis.selectedSegmentIndex ? 'var(--msp-panel-2)' : 'var(--msp-panel)'}; cursor: pointer; border-left: none; border-right: none; border-bottom: none; font-family: inherit;"
          >
            <div style="width:9px; height:9px; border-radius:2px; background:{colorMap[segLabel(seg)]};"></div>
            <span class="msp-mono" style="font-size: 11.5px; color: var(--msp-text-faint); width: 90px;">{secToLabel(seg.start)}–{secToLabel(seg.end)}</span>
            <span style="font-size: 10.5px; font-weight: 800; width: 20px; height: 20px; border-radius: 50%; border: 1.5px solid var(--msp-text-faint); display: inline-flex; align-items: center; justify-content: center; color: var(--msp-text-dim);">{segStruct(seg)}</span>
            <span style="font-size: 12.5px; font-weight: 700; flex: 1;">{segLabel(seg)}</span>
            {#if seg.confidence != null}
              <span class="msp-mono" style="font-size: 11px; color: var(--msp-text-faint);">{Math.round(seg.confidence * 100)}% conf.</span>
            {/if}
          </button>
        {/each}
      </div>
    </div>

    {#if sel}
      <div style="border: 1px solid var(--msp-border); border-radius: 12px; background: var(--msp-panel); padding: 18px; position: sticky; top: 90px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--msp-text-faint);">Segment {$analysis.selectedSegmentIndex + 1}</span>
          <span style="font-size: 10.5px; font-weight: 800; width: 20px; height: 20px; border-radius: 50%; border: 1.5px solid var(--msp-text-faint); display: inline-flex; align-items: center; justify-content: center; color: var(--msp-text-dim);">{segStruct(sel)}</span>
        </div>
        <div style="font-size: 18px; font-weight: 800; margin-bottom: 2px;">{segLabel(sel)}</div>
        <div class="msp-mono" style="font-size: 12px; color: var(--msp-text-faint); margin-bottom: 14px;">{secToLabel(sel.start)} – {secToLabel(sel.end)} · {secToLabel(sel.end - sel.start)}</div>

        {#if sel.confidence != null}
          <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; font-size: 11.5px; margin-bottom: 5px;"><span style="font-weight: 600; color: var(--msp-text-dim);">Confidence</span><span class="msp-mono">{Math.round(sel.confidence * 100)}%</span></div>
            <div style="height: 5px; border-radius: 3px; background: var(--msp-panel-2);"><div style="width: {Math.round(sel.confidence*100)}%; height: 100%; border-radius: 3px; background: {sel.confidence >= 0.8 ? 'var(--msp-ok)' : sel.confidence >= 0.65 ? 'var(--msp-accent)' : 'var(--msp-warn)'};"></div></div>
          </div>
        {/if}

        {#if sel.semantic_reason || sel.reason}
          <div style="font-size: 12px; line-height: 1.6; color: var(--msp-text-dim); background: var(--msp-panel-2); border-radius: 8px; padding: 12px; margin-bottom: 14px;">{sel.semantic_reason || sel.reason}</div>
        {/if}

        {#if similarSegments(sel).length}
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--msp-text-faint); margin-bottom: 8px;">Similar segments</div>
          <div style="display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;">
            {#each similarSegments(sel) as t}
              <span class="msp-mono" style="font-size: 11px; padding: 4px 8px; border-radius: 6px; background: var(--msp-panel-2); color: var(--msp-text-dim);">{t}</span>
            {/each}
          </div>
        {/if}

        {#if sel.source_features?.length}
          <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;">
            {#each sel.source_features as feat}
              <span class="msp-mono" style="border-radius: 5px; background: var(--msp-panel-2); padding: 3px 7px; font-size: 10px; color: var(--msp-text-faint);">{feat}</span>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>
{/if}
