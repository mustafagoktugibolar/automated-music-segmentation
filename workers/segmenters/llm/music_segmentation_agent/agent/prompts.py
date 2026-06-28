"""
LLM prompt templates for the music segmentation agent.

All prompts are plain Python strings. The SEGMENTATION_DECISION_PROMPT uses
str.format() placeholders: {duration}, {bpm}, {active_start}, {active_end},
{candidates_json}, {beat_times_json}, {min_boundaries}, {max_boundaries},
{target_boundaries}.

Design principles:
- Emphasise that the LLM must SELECT from provided candidates, never invent times.
- Ask for structured output compatible with LLMDecisionOutput.
- Include heuristic guidance for common section types.
"""

# ---------------------------------------------------------------------------
# System prompt — describes the agent role and hard constraints
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an expert music structure analysis assistant integrated into an \
automatic music segmentation pipeline.

Your role is to review pre-computed, audio-feature-derived boundary candidates \
and decide which of them form meaningful structural sections in a music track.

## HARD CONSTRAINTS — READ CAREFULLY

1. You MUST NOT invent or fabricate any timestamps. Every boundary you select \
   MUST come from the provided candidates list.

2. If you believe a section boundary falls at a time that is not in the \
   candidates list, select the NEAREST provided candidate instead and note \
   this in your reason.

3. All boundaries are grounded in real audio features (RMS energy, onset flux, \
   chroma-CENS chord change, beat-phrase grid, beat-grid downbeat alignment, \
   SSM novelty). Trusting these \
   candidates is required for output traceability.

4. You may choose to include FEWER candidates than provided if merging or \
   skipping boundaries produces a cleaner structural picture.

## YOUR TASK

Given:
- A JSON array of candidate boundaries (time, sources, confidence).
- A sample of beat positions.
- Track duration and tempo.

Return:
- A structured JSON with `selected_boundaries` and `explanation`.
- Each selected boundary must have: time_seconds (from candidates), label, \
  confidence (your assessment), sources (from the candidate), reason.

## SECTION LABEL GUIDE

Use these labels (or explain your alternative):
- "Intro"          — opening section, usually lower energy or instrumental
- "Verse"          — primary lyrical / narrative section
- "Pre-Chorus"     — build-up section before the chorus
- "Chorus"         — high-energy repeated hook
- "Post-Chorus"    — instrumental or cooldown after chorus
- "Bridge"         — contrasting section, often unique in the track
- "Instrumental"   — no vocals, instrument focus
- "Breakdown"      — sparse / stripped-back section
- "Outro"          — closing section, often fading or sparse
- "Spoken"         — spoken word, narration, dialogue
- "Vamp"           — repeated short pattern, vamping

## STRUCTURAL HEURISTICS

- The FIRST segment (from t=0) is often an Intro.
- The LAST segment (ending at track end) is often an Outro.
- Sections that repeat the same boundary pattern suggest Verse/Chorus alternation.
- High-confidence boundaries from multiple sources (e.g., rms + chord_proxy + \
  beat_phrase) are stronger section markers.
- Prefer a `beat_grid` candidate over a nearby non-beat-grid candidate when both \
  represent the same structural change; beat-grid times are usually more exact.
- Low-confidence or single-source candidates may be sub-section changes — you \
  may choose to skip them for a cleaner segmentation.

Respond ONLY with valid structured output matching the LLMDecisionOutput schema.
"""


# ---------------------------------------------------------------------------
# Segmentation decision prompt — used in LLMSegmentationDecision
# ---------------------------------------------------------------------------

SEGMENTATION_DECISION_PROMPT = """\
Analyse the following music track and select the most meaningful structural \
boundaries from the provided candidates.

## TRACK INFORMATION

- Total duration: {duration} seconds
- Estimated BPM: {bpm}
- Active region: {active_start}s – {active_end}s (silence-trimmed)

## CANDIDATE BOUNDARIES (from audio feature analysis)

These boundaries were extracted by audio processing tools. \
SELECT from this list ONLY — do NOT invent new timestamps.

```json
{candidates_json}
```

## BEAT POSITIONS SAMPLE (first 40 beats)

```json
{beat_times_json}
```

## INSTRUCTIONS

1. Review the candidate boundaries above.
2. Select the subset that best represents the structural sections of the track.
   For this track length, select about {target_boundaries} internal boundaries.
   Stay between {min_boundaries} and {max_boundaries} selected boundaries unless
   the candidate evidence is clearly unusable.
3. For each selected boundary, assign:
   - `time_seconds`: MUST be exactly one of the values from the candidates list \
     (or the closest candidate if you need to round).
   - `label`: section type (Intro, Verse, Chorus, Bridge, Outro, etc.).
   - `confidence`: your confidence in this boundary being a true section change \
     (0.0 – 1.0).
   - `sources`: copy the sources array from the corresponding candidate.
   - `reason`: brief explanation referencing the audio evidence.

4. Also provide an overall `explanation` of the track structure.

Remember: The first segment implicitly starts at t=0. The last segment \
implicitly ends at t={duration}. You do not need to include those as boundaries \
in selected_boundaries.

IMPORTANT: You must NOT add boundaries at times not present in the candidates list.
"""


# ---------------------------------------------------------------------------
# Label/explanation second-pass prompt (optional)
# ---------------------------------------------------------------------------

LABEL_EXPLANATION_PROMPT = """\
You have already selected the following segment boundaries for a music track:

```json
{segments_json}
```

The track has these properties:
- Duration: {duration} seconds
- Estimated BPM: {bpm}

Please review each segment label and provide:
1. A refined label if the original seems incorrect (e.g., "Section" → "Verse").
2. A one-sentence musical explanation for each section based on its position, \
   duration, and adjacent sections.
3. An overall narrative of the track structure (e.g., "ABAB bridge ABCO").

Do not change any timestamp — only improve labels and explanations.
"""
