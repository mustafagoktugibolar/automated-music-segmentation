"use strict";

const pptxgen = require("pptxgenjs");
const pres = new pptxgen();

pres.layout = "LAYOUT_WIDE";
pres.title = "Automated Music Structure Segmentation";
pres.author = "Capstone Team";

// ─── COLOR PALETTE: Navy & Copper ──────────────────────────────────────────
const C = {
  navy:      "1B2A4A",
  navyMid:   "243556",
  copper:    "B87333",
  copperLt:  "D4944A",
  slate:     "4A5568",
  lightBg:   "F4F6FA",
  white:     "FFFFFF",
  offWhite:  "F8F9FC",
  gray:      "718096",
  grayLt:    "CBD5E0",
  green:     "2D6A4F",
  greenLt:   "52B788",
  amber:     "B45309",
  amberLt:   "F59E0B",
  red:       "9B1C1C",
  teal:      "0D9488",
};

// ─── PRESENTER ASSIGNMENTS ──────────────────────────────────────────────────
// 3 people, ~7 slides each
const PRESENTERS = {
  1:  "Presenter 1",
  2:  "Presenter 1",
  3:  "Presenter 1",
  4:  "Presenter 1",
  5:  "Presenter 1",
  6:  "Presenter 1",
  7:  "Presenter 1",
  8:  "Presenter 2",
  9:  "Presenter 2",
  10: "Presenter 2",
  11: "Presenter 2",
  12: "Presenter 2",
  13: "Presenter 2",
  14: "Presenter 3",
  15: "Presenter 3",
  16: "Presenter 3",
  17: "Presenter 3",
  18: "Presenter 3",
  19: "Presenter 3",
  20: "Presenter 3",
};

// ─── HELPERS ────────────────────────────────────────────────────────────────

function presenterBadge(slide, slideNum) {
  const who = PRESENTERS[slideNum];
  const colorMap = {
    "Presenter 1": C.copper,
    "Presenter 2": C.teal,
    "Presenter 3": C.green,
  };
  const bg = colorMap[who] || C.slate;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 10.5, y: 6.9, w: 2.6, h: 0.38,
    fill: { color: bg }, line: { color: bg }, rounding: 0.1,
  });
  slide.addText(`🎤 ${who}`, {
    x: 10.5, y: 6.9, w: 2.6, h: 0.38,
    fontSize: 9, color: C.white, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
}

function slideNumLabel(slide, n) {
  slide.addText(`${n} / 20`, {
    x: 0.2, y: 6.95, w: 1, h: 0.25,
    fontSize: 8, color: C.gray, align: "left", margin: 0,
  });
}

// Standard navy header bar + white title
function headerBar(slide, title, subtitle) {
  slide.background = { color: C.offWhite };
  // top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 13.3, h: 1.15,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  // copper accent line
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 1.15, w: 13.3, h: 0.06,
    fill: { color: C.copper }, line: { color: C.copper },
  });
  slide.addText(title, {
    x: 0.4, y: 0.08, w: 11.5, h: 0.65,
    fontSize: 26, color: C.white, bold: true, fontFace: "Georgia",
    align: "left", valign: "middle", margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.4, y: 0.73, w: 11.5, h: 0.38,
      fontSize: 13, color: C.copperLt, bold: false, fontFace: "Calibri",
      align: "left", valign: "middle", margin: 0,
    });
  }
}

// Section divider (full navy slide)
function sectionDivider(slide, section, desc) {
  slide.background = { color: C.navy };
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 2.8, w: 13.3, h: 0.07,
    fill: { color: C.copper }, line: { color: C.copper },
  });
  slide.addText(section, {
    x: 1, y: 1.8, w: 11.3, h: 1.1,
    fontSize: 38, color: C.white, bold: true, fontFace: "Georgia",
    align: "center", valign: "middle",
  });
  if (desc) {
    slide.addText(desc, {
      x: 1.5, y: 3.2, w: 10.3, h: 0.9,
      fontSize: 16, color: C.copperLt, fontFace: "Calibri",
      align: "center", valign: "middle",
    });
  }
}

function bulletBox(slide, items, opts) {
  const { x, y, w, h, fontSize = 14, color = C.slate, bg, radius } = opts || {};
  if (bg) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w, h,
      fill: { color: bg },
      line: { color: bg },
      rounding: radius || 0.08,
    });
  }
  const richItems = [];
  items.forEach((item, i) => {
    richItems.push({
      text: item,
      options: {
        bullet: true,
        breakLine: i < items.length - 1,
        fontSize,
        color,
        fontFace: "Calibri",
      },
    });
  });
  slide.addText(richItems, { x, y, w, h, valign: "top", margin: [8, 10, 8, 10] });
}

function infoCard(slide, titleText, bodyItems, x, y, w, h, accentColor) {
  const ac = accentColor || C.copper;
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.white },
    line: { color: C.grayLt, width: 1.2 },
    shadow: { type: "outer", color: "000000", blur: 5, offset: 2, angle: 135, opacity: 0.08 },
  });
  // accent top border
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: w, h: 0.07,
    fill: { color: ac }, line: { color: ac },
  });
  slide.addText(titleText, {
    x: x + 0.12, y: y + 0.1, w: w - 0.24, h: 0.36,
    fontSize: 12, bold: true, color: C.navy, fontFace: "Georgia",
    align: "left", valign: "middle", margin: 0,
  });
  const richItems = [];
  bodyItems.forEach((item, i) => {
    richItems.push({
      text: item,
      options: {
        bullet: true,
        breakLine: i < bodyItems.length - 1,
        fontSize: 11,
        color: C.slate,
        fontFace: "Calibri",
      },
    });
  });
  slide.addText(richItems, {
    x: x + 0.12, y: y + 0.5, w: w - 0.24, h: h - 0.6,
    valign: "top", margin: [4, 4, 4, 4],
  });
}

function codeBox(slide, codeText, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: "1E2A3A" },
    line: { color: "2D3F55" },
    rounding: 0.05,
  });
  slide.addText(codeText, {
    x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h - 0.2,
    fontSize: 10, color: "7EC8E3", fontFace: "Courier New",
    align: "left", valign: "top", margin: 0,
  });
}

function formulaBox(slide, formulaText, x, y, w, h) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: "FFF8F0" },
    line: { color: C.copper, width: 1.5 },
    rounding: 0.05,
  });
  slide.addText(formulaText, {
    x: x + 0.15, y: y + 0.08, w: w - 0.3, h: h - 0.16,
    fontSize: 11, color: C.navy, fontFace: "Courier New",
    align: "left", valign: "middle", margin: 0,
  });
}

// ─── SLIDE 1: TITLE SLIDE ───────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  // decorative shapes
  s.addShape(pres.shapes.OVAL, {
    x: 10.5, y: -0.5, w: 4, h: 4,
    fill: { color: "1F3460" }, line: { color: "1F3460" },
  });
  s.addShape(pres.shapes.OVAL, {
    x: -1.2, y: 5.2, w: 3, h: 3,
    fill: { color: "162440" }, line: { color: "162440" },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.55, w: 13.3, h: 0.07,
    fill: { color: C.copper }, line: { color: C.copper },
  });

  s.addText("Automated Music", {
    x: 0.7, y: 1.0, w: 11, h: 1.0,
    fontSize: 46, color: C.white, bold: true, fontFace: "Georgia",
    align: "center",
  });
  s.addText("Structure Segmentation", {
    x: 0.7, y: 1.95, w: 11, h: 1.0,
    fontSize: 46, color: C.copperLt, bold: true, fontFace: "Georgia",
    align: "center",
  });
  s.addText("Distributed Multi-Algorithm Boundary Detection with Two-Level Fusion", {
    x: 1, y: 3.0, w: 11.3, h: 0.55,
    fontSize: 15, color: C.grayLt, fontFace: "Calibri",
    align: "center",
  });
  s.addText("Capstone Presentation  •  Spring 2026", {
    x: 1, y: 3.65, w: 11.3, h: 0.45,
    fontSize: 13, color: C.gray, fontFace: "Calibri",
    align: "center",
  });

  // 3 presenter badges
  const presenters = ["Presenter 1", "Presenter 2", "Presenter 3"];
  const colors = [C.copper, C.teal, C.green];
  presenters.forEach((p, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 2.4 + i * 3.2, y: 4.85, w: 2.8, h: 0.4,
      fill: { color: colors[i] }, line: { color: colors[i] }, rounding: 0.1,
    });
    s.addText(p, {
      x: 2.4 + i * 3.2, y: 4.85, w: 2.8, h: 0.4,
      fontSize: 12, color: C.white, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });

  slideNumLabel(s, 1);
}

// ─── SLIDE 2: AGENDA ────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Presentation Agenda", "What we will cover in the next 30 minutes");
  presenterBadge(s, 2);
  slideNumLabel(s, 2);

  const sections = [
    ["01", "Problem & Motivation", "Why music segmentation is hard", C.copper],
    ["02", "System Architecture", "Distributed workers, RabbitMQ, PostgreSQL", C.teal],
    ["03", "Segmentation Methods", "Custom Librosa pipeline + MSAF baselines", C.green],
    ["04", "Two-Level Fusion", "Feature fusion & algorithm-level voting", C.copper],
    ["05", "Evaluation", "SALAMI dataset, precision/recall/F1", C.teal],
    ["06", "Demo & Conclusion", "Live demo, limitations, future work", C.green],
  ];

  sections.forEach(([num, title, desc, color], i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.35 + col * 6.5;
    const y = 1.45 + row * 1.75;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 6.1, h: 1.5,
      fill: { color: C.white },
      line: { color: C.grayLt },
      shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.07 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.06, h: 1.5,
      fill: { color }, line: { color },
    });
    s.addText(num, {
      x: x + 0.18, y: y + 0.08, w: 0.55, h: 0.55,
      fontSize: 22, color, bold: true, fontFace: "Georgia",
      align: "center", margin: 0,
    });
    s.addText(title, {
      x: x + 0.8, y: y + 0.08, w: 5.1, h: 0.45,
      fontSize: 14, color: C.navy, bold: true, fontFace: "Georgia",
      align: "left", valign: "middle", margin: 0,
    });
    s.addText(desc, {
      x: x + 0.8, y: y + 0.58, w: 5.1, h: 0.75,
      fontSize: 11, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "top", margin: 0,
    });
  });
}

// ─── SLIDE 3: PROBLEM DEFINITION ────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Problem Definition", "Detecting structural boundaries in music automatically");
  presenterBadge(s, 3);
  slideNumLabel(s, 3);

  // timeline graphic (simplified)
  const timelineY = 1.8;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: timelineY + 0.55, w: 12.2, h: 0.12,
    fill: { color: C.grayLt }, line: { color: C.grayLt },
  });
  const sections = [
    { label: "Intro", x: 0.5, w: 2.0, color: C.copper },
    { label: "Verse", x: 2.6, w: 2.8, color: C.teal },
    { label: "Chorus", x: 5.5, w: 2.5, color: C.green },
    { label: "Verse", x: 8.1, w: 2.3, color: C.teal },
    { label: "Outro", x: 10.5, w: 2.2, color: C.amber },
  ];
  sections.forEach(({ label, x, w, color }) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: timelineY, w, h: 0.7,
      fill: { color }, line: { color },
    });
    s.addText(label, {
      x, y: timelineY, w, h: 0.7,
      fontSize: 12, color: C.white, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });
  const boundaries = [2.5, 5.4, 7.95, 10.45];
  boundaries.forEach((bx) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx + 0.45, y: timelineY - 0.3, w: 0.04, h: 1.55,
      fill: { color: "DC2626" }, line: { color: "DC2626" },
    });
    s.addText("▲ boundary", {
      x: bx, y: timelineY + 1.25, w: 1.0, h: 0.3,
      fontSize: 8, color: "DC2626", align: "center",
    });
  });

  s.addText("0 s", { x: 0.4, y: timelineY + 0.7, w: 0.5, h: 0.25, fontSize: 9, color: C.gray, align: "left" });
  s.addText("~240 s", { x: 12.3, y: timelineY + 0.7, w: 0.8, h: 0.25, fontSize: 9, color: C.gray, align: "right" });

  // Key definitions
  const defs = [
    ["Boundary", "The exact timestamp where one section ends and another begins"],
    ["Segment", "The time interval between two consecutive boundaries"],
    ["Structural Label", "A/B/C grouping — which segments sound similar to each other"],
    ["Semantic Label", "Intro/Verse/Chorus — human-readable musical role (heuristic)"],
  ];
  defs.forEach(([term, def], i) => {
    const x = 0.35 + (i % 2) * 6.5;
    const y = 3.1 + Math.floor(i / 2) * 0.9;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 6.1, h: 0.75,
      fill: { color: C.white }, line: { color: C.grayLt },
    });
    s.addText(`${term}:`, {
      x: x + 0.12, y, w: 1.5, h: 0.75,
      fontSize: 12, color: C.navy, bold: true, fontFace: "Georgia",
      align: "left", valign: "middle",
    });
    s.addText(def, {
      x: x + 1.6, y, w: 4.35, h: 0.75,
      fontSize: 11, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "middle",
    });
  });
}

// ─── SLIDE 4: WHY IS IT HARD? ───────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Why Music Segmentation Is Difficult", "No single signal reveals all structural boundaries");
  presenterBadge(s, 4);
  slideNumLabel(s, 4);

  const challenges = [
    { title: "Harmonic Change", desc: "Chord progression shifts, but energy may stay constant", icon: "🎵", color: C.copper },
    { title: "Energy Change", desc: "RMS jump can be a drum fill, not a section change", icon: "⚡", color: C.teal },
    { title: "Timbre Change", desc: "Guitar → full band entry marks transition, not pitch", icon: "🎸", color: C.green },
    { title: "Rhythm / Onset", desc: "Onset density changes may be note events, not boundaries", icon: "🥁", color: C.amber },
    { title: "Structural Repetition", desc: "Repeated Chorus must be detected via SSM patterns", icon: "🔁", color: C.navy },
    { title: "Annotation Uncertainty", desc: "Human annotators disagree on exact milliseconds", icon: "❓", color: "9B1C1C" },
  ];

  challenges.forEach(({ title, desc, icon, color }, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.3 + col * 4.35;
    const y = 1.45 + row * 2.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 4.05, h: 2.15,
      fill: { color: C.white },
      line: { color: C.grayLt },
      shadow: { type: "outer", color: "000000", blur: 5, offset: 1, angle: 135, opacity: 0.08 },
      rounding: 0.08,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.05, h: 0.06,
      fill: { color }, line: { color },
    });
    s.addText(icon, {
      x: x + 0.15, y: y + 0.15, w: 0.55, h: 0.55,
      fontSize: 22, align: "center", valign: "middle",
    });
    s.addText(title, {
      x: x + 0.75, y: y + 0.15, w: 3.15, h: 0.55,
      fontSize: 12, color, bold: true, fontFace: "Georgia",
      align: "left", valign: "middle",
    });
    s.addText(desc, {
      x: x + 0.15, y: y + 0.75, w: 3.75, h: 1.25,
      fontSize: 11, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "top",
    });
  });

  s.addText("→  This motivates multi-feature, multi-algorithm, and tolerance-based evaluation", {
    x: 0.3, y: 5.98, w: 12.7, h: 0.3,
    fontSize: 11, color: C.copper, bold: true, fontFace: "Calibri",
    align: "center",
  });
}

// ─── SLIDE 5: HIGH-LEVEL ARCHITECTURE ───────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "System Architecture", "Distributed, asynchronous, modular design");
  presenterBadge(s, 5);
  slideNumLabel(s, 5);

  // Architecture diagram using shapes
  const boxes = [
    { label: "Svelte\nFrontend", x: 0.3, y: 1.5, w: 2.0, h: 0.9, color: C.teal, textColor: C.white },
    { label: "FastAPI\nBackend", x: 2.9, y: 1.5, w: 2.0, h: 0.9, color: C.navy, textColor: C.white },
    { label: "Segmentation\nOrchestrator", x: 5.5, y: 1.5, w: 2.2, h: 0.9, color: C.navyMid, textColor: C.white },
    { label: "RabbitMQ\nTopic Exchange", x: 8.3, y: 1.5, w: 2.2, h: 0.9, color: C.amber, textColor: C.white },
    { label: "PostgreSQL\nTask Store", x: 5.5, y: 4.0, w: 2.2, h: 0.9, color: C.slate, textColor: C.white },
    { label: "Result\nListener", x: 2.9, y: 4.0, w: 2.0, h: 0.9, color: C.green, textColor: C.white },
  ];

  boxes.forEach(({ label, x, y, w, h, color, textColor }) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w, h, fill: { color }, line: { color }, rounding: 0.1,
    });
    s.addText(label, {
      x, y, w, h,
      fontSize: 11, color: textColor, bold: true, fontFace: "Calibri",
      align: "center", valign: "middle",
    });
  });

  // Workers column
  const workers = [
    { label: "custom_librosa\nworker", color: C.copper },
    { label: "MSAF Foote\nworker", color: "6B7280" },
    { label: "MSAF CNMF\nworker", color: "6B7280" },
    { label: "MSAF SCluster\nworker", color: "6B7280" },
    { label: "Fusion\nworker", color: "7C3AED" },
  ];
  workers.forEach(({ label, color }, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 11.0, y: 1.35 + i * 1.0, w: 2.1, h: 0.8,
      fill: { color }, line: { color }, rounding: 0.08,
    });
    s.addText(label, {
      x: 11.0, y: 1.35 + i * 1.0, w: 2.1, h: 0.8,
      fontSize: 9, color: C.white, bold: true,
      align: "center", valign: "middle",
    });
    // arrow from RabbitMQ
    s.addShape(pres.shapes.RECTANGLE, {
      x: 10.5, y: 1.7 + i * 1.0, w: 0.5, h: 0.03,
      fill: { color: C.grayLt }, line: { color: C.grayLt },
    });
  });

  // Arrows between top boxes
  const arrows = [
    [2.3, 1.95, 0.6],   // Frontend → FastAPI
    [4.9, 1.95, 0.6],   // FastAPI → Orchestrator
    [7.7, 1.95, 0.6],   // Orchestrator → RabbitMQ
  ];
  arrows.forEach(([x, y, w]) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h: 0.03,
      fill: { color: C.copper }, line: { color: C.copper },
    });
    s.addText("▶", { x: x + w - 0.1, y: y - 0.1, w: 0.25, h: 0.25, fontSize: 9, color: C.copper });
  });

  // Down arrow RabbitMQ → Workers area
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.35, y: 2.4, w: 0.03, h: 1.1,
    fill: { color: C.copper }, line: { color: C.copper },
  });

  // segmentation.result → Listener
  s.addText("segmentation.result", {
    x: 4.5, y: 3.45, w: 2.8, h: 0.28,
    fontSize: 9, color: C.copper, bold: true, align: "center",
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.65, y: 3.68, w: 0.03, h: 0.32,
    fill: { color: C.copper }, line: { color: C.copper },
  });

  // Listener → PostgreSQL
  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.9, y: 4.43, w: 0.6, h: 0.03,
    fill: { color: C.copper }, line: { color: C.copper },
  });

  // Frontend ← result (bottom)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 4.5, w: 2.6, h: 0.03,
    fill: { color: C.teal }, line: { color: C.teal },
  });
  s.addText("SSE / status polling", {
    x: 0.3, y: 4.6, w: 2.6, h: 0.25,
    fontSize: 8, color: C.teal, align: "center",
  });

  // Key insight box
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.3, y: 5.85, w: 12.7, h: 0.4,
    fill: { color: "EFF6FF" }, line: { color: C.teal }, rounding: 0.05,
  });
  s.addText("Key Design Decision: API never does audio processing — it dispatches tasks to independent workers via RabbitMQ", {
    x: 0.5, y: 5.85, w: 12.3, h: 0.4,
    fontSize: 11, color: C.navy, bold: true, fontFace: "Calibri",
    align: "center", valign: "middle",
  });
}

// ─── SLIDE 6: REQUEST LIFECYCLE ──────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "End-to-End Request Lifecycle", "From user upload to completed segmentation result");
  presenterBadge(s, 6);
  slideNumLabel(s, 6);

  const steps = [
    ["01", "User uploads audio or selects from storage (song_id)", C.copper],
    ["02", "FastAPI validates file type, algorithm names, and params via Pydantic schemas", C.teal],
    ["03", "Orchestrator normalizes algorithm names, generates UUID task_id", C.green],
    ["04", "Task inserted into PostgreSQL with status=processing, expected_algorithms, results={}", C.amber],
    ["05", "Base worker messages published to RabbitMQ with per-algorithm routing keys", C.copper],
    ["06", "Workers run in parallel — audio analysis, DSP, boundary detection", C.teal],
    ["07", "Each worker publishes result to segmentation.result routing key", C.green],
    ["08", "ResultListener normalizes result, stores in DB, checks fusion readiness", C.amber],
    ["09", "When all expected results arrive, task status → completed", C.copper],
    ["10", "Frontend reads final result via SSE stream or status polling", C.teal],
  ];

  steps.forEach(([num, text, color], i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.3 + col * 6.55;
    const y = 1.42 + row * 1.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 0.55, h: 0.75,
      fill: { color }, line: { color }, rounding: 0.08,
    });
    s.addText(num, {
      x, y, w: 0.55, h: 0.75,
      fontSize: 14, color: C.white, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.55, y: y + 0.08, w: 5.8, h: 0.6,
      fill: { color: C.white }, line: { color: C.grayLt },
    });
    s.addText(text, {
      x: x + 0.7, y: y + 0.08, w: 5.65, h: 0.6,
      fontSize: 11, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "middle", margin: [4, 6, 4, 6],
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.3, y: 6.7, w: 12.7, h: 0.38,
    fill: { color: "FFF7ED" }, line: { color: C.copper }, rounding: 0.05,
  });
  s.addText("expected vs dispatch: fusion is expected but NOT dispatched immediately — it waits for all base results first", {
    x: 0.5, y: 6.7, w: 12.3, h: 0.38,
    fontSize: 10, color: C.amber, bold: true, fontFace: "Calibri",
    align: "center", valign: "middle",
  });
}

// ─── SLIDE 7: COMMON RESULT SCHEMA ──────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Common Result Schema", "All algorithms speak the same language for fusion & evaluation");
  presenterBadge(s, 7);
  slideNumLabel(s, 7);

  // Left: schema objects
  infoCard(s, "Boundary", [
    "time — timestamp in seconds",
    "confidence — 0–1 evidence strength",
    "source — which algorithm/feature",
    "sources — list if fusion contributed",
    "metadata — score, raw votes",
  ], 0.3, 1.42, 3.9, 2.7, C.copper);

  infoCard(s, "Segment", [
    "start, end — time interval",
    "structural_label — A/B/C similarity group",
    "semantic_label — Intro/Verse/Chorus (heuristic)",
    "label_confidence, semantic_confidence",
    "sources — contributing algorithms",
  ], 0.3, 4.22, 3.9, 2.6, C.teal);

  infoCard(s, "AlgorithmResult", [
    "task_id, status — completed | failed",
    "worker_type — custom, msaf, fusion",
    "algorithm — canonical name",
    "duration_seconds — track length",
    "boundaries[], segments[], diagnostics{}",
  ], 4.3, 1.42, 4.1, 2.7, C.green);

  // Right: JSON example
  codeBox(s, `{
  "task_id": "abc-123",
  "status": "completed",
  "worker_type": "msaf",
  "algorithm": "foote",
  "duration_seconds": 180.2,
  "boundaries": [
    { "time": 31.4,
      "confidence": 1.0,
      "source": "foote" }
  ],
  "segments": [
    { "start": 0.0,
      "end": 31.4,
      "structural_label": "A",
      "semantic_label": "Intro" }
  ]
}`, 8.5, 1.42, 4.6, 5.4);

  s.addText("Why schema matters: fusion & evaluation never need algorithm-specific parsers", {
    x: 0.3, y: 6.88, w: 12.7, h: 0.3,
    fontSize: 11, color: C.copper, bold: true, fontFace: "Calibri",
    align: "center",
  });
}

// ─── SLIDE 8: SECTION DIVIDER ────────────────────────────────────────────────
{
  const s = pres.addSlide();
  sectionDivider(s, "Segmentation Methods", "Custom Librosa Pipeline  •  MSAF Baselines: Foote, CNMF, SCluster");
  presenterBadge(s, 8);
  slideNumLabel(s, 8);
}

// ─── SLIDE 9: CUSTOM LIBROSA PIPELINE ───────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Custom Librosa Pipeline", "Deterministic multi-feature segmentation with feature-level fusion");
  presenterBadge(s, 9);
  slideNumLabel(s, 9);

  // Pipeline flow diagram
  const stages = [
    { label: "Audio Load\n(ffmpeg, mono,\n22050 Hz)", color: C.slate },
    { label: "Active Region\nDetection\n(RMS threshold)", color: C.teal },
    { label: "Feature\nExtraction\n(Chroma-CENS\n+ MFCC)", color: C.green },
    { label: "Self-Similarity\nMatrix (SSM)\n+ Novelty", color: C.copper },
    { label: "Multi-Feature\nCandidates\n(RMS, Onset,\nChord, Beat)", color: C.amber },
    { label: "Feature-Level\nFusion\n(Weighted score\n+ snapping)", color: "7C3AED" },
    { label: "Segments\n+ Labels\n(A/B/C structural\n+ semantic)", color: C.navy },
  ];

  stages.forEach(({ label, color }, i) => {
    const x = 0.3 + i * 1.84;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.5, w: 1.65, h: 2.6,
      fill: { color }, line: { color }, rounding: 0.1,
    });
    s.addText(label, {
      x, y: 1.5, w: 1.65, h: 2.6,
      fontSize: 9.5, color: C.white, bold: true, fontFace: "Calibri",
      align: "center", valign: "middle",
    });
    if (i < stages.length - 1) {
      s.addText("→", {
        x: x + 1.65, y: 2.55, w: 0.19, h: 0.4,
        fontSize: 14, color: C.copper, bold: true,
        align: "center",
      });
    }
  });

  // Key feature explanations
  const feats = [
    ["Chroma-CENS", "12 pitch classes, octave-invariant, captures harmonic repetition", C.green],
    ["MFCC", "Spectral envelope (timbre) — detects instrument/texture changes", C.teal],
    ["SSM (Self-Similarity Matrix)", "Each frame vs every other frame — reveals repeated sections as bright blocks", C.copper],
    ["Active Region Detection", "Crops leading/trailing silence using RMS+dB; timestamps restored at end", C.slate],
  ];

  feats.forEach(([name, desc, color], i) => {
    const x = 0.3 + (i % 2) * 6.55;
    const y = 4.5 + Math.floor(i / 2) * 0.95;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.06, h: 0.75,
      fill: { color }, line: { color },
    });
    s.addText(name, {
      x: x + 0.2, y, w: 6.1, h: 0.3,
      fontSize: 11, color, bold: true, fontFace: "Georgia",
      align: "left", valign: "middle", margin: 0,
    });
    s.addText(desc, {
      x: x + 0.2, y: y + 0.3, w: 6.1, h: 0.45,
      fontSize: 10, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "top", margin: 0,
    });
  });
}

// ─── SLIDE 10: MSAF BASELINES ───────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "MSAF Baseline Algorithms", "Three complementary mathematical perspectives on structure");
  presenterBadge(s, 10);
  slideNumLabel(s, 10);

  const algos = [
    {
      name: "Foote",
      color: "5B21B6",
      math: "Checkerboard Novelty",
      desc: "Applies a checkerboard kernel to the SSM diagonal. High response where before/after similarity structure changes sharply. Detects local transitions.",
      strength: "Fast, explainable, detects sharp transitions",
      risk: "Sensitive to non-structural transients (e.g., drum fills)",
    },
    {
      name: "CNMF",
      color: C.teal,
      math: "Convex Non-negative Matrix Factorization",
      desc: "Factorizes the feature matrix into recurring latent components. Sections emerge as regions dominated by consistent activation patterns.",
      strength: "Captures repeated latent patterns across the track",
      risk: "Factorization rank choice affects boundary granularity",
    },
    {
      name: "SCluster",
      color: C.green,
      math: "Spectral Clustering on Affinity Graph",
      desc: "Treats the similarity structure as a graph and applies spectral clustering. Uses global organization to determine section boundaries.",
      strength: "Global structure-aware; handles repeated sections well",
      risk: "Cluster count parameter influences segment number",
    },
  ];

  algos.forEach(({ name, color, math, desc, strength, risk }, i) => {
    const x = 0.3 + i * 4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.42, w: 4.05, h: 5.55,
      fill: { color: C.white }, line: { color: C.grayLt },
      shadow: { type: "outer", color: "000000", blur: 5, offset: 1, angle: 135, opacity: 0.08 },
      rounding: 0.1,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.42, w: 4.05, h: 0.52,
      fill: { color }, line: { color },
    });
    s.addText(name, {
      x, y: 1.42, w: 4.05, h: 0.52,
      fontSize: 20, color: C.white, bold: true, fontFace: "Georgia",
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(math, {
      x: x + 0.15, y: 2.02, w: 3.75, h: 0.38,
      fontSize: 10, color, bold: true, fontFace: "Calibri",
      align: "center",
    });
    s.addText(desc, {
      x: x + 0.15, y: 2.42, w: 3.75, h: 2.1,
      fontSize: 11, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "top",
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.15, y: 4.58, w: 3.75, h: 0.03,
      fill: { color: C.grayLt }, line: { color: C.grayLt },
    });
    s.addText("✓ " + strength, {
      x: x + 0.15, y: 4.65, w: 3.75, h: 0.4,
      fontSize: 10, color: C.green, fontFace: "Calibri", align: "left", valign: "top",
    });
    s.addText("⚠ " + risk, {
      x: x + 0.15, y: 5.08, w: 3.75, h: 0.55,
      fontSize: 10, color: C.amber, fontFace: "Calibri", align: "left", valign: "top",
    });
  });

  s.addText("Diversity of error profiles is the key value — this is exactly why algorithm-level fusion is needed", {
    x: 0.3, y: 7.1, w: 12.7, h: 0.28,
    fontSize: 11, color: C.copper, bold: true, fontFace: "Calibri", align: "center",
  });
}

// ─── SLIDE 11: FEATURE-LEVEL FUSION SOURCES ─────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Feature-Level Fusion: Candidate Sources", "Six independent evidence streams inside the custom_librosa pipeline");
  presenterBadge(s, 11);
  slideNumLabel(s, 11);

  const sources = [
    { name: "SSM Novelty", weight: "0.42", desc: "Checkerboard response on Self-Similarity Matrix — structural change evidence. Can pass acceptance alone if confidence ≥ 0.5.", color: C.copper },
    { name: "Chord Proxy", weight: "0.18", desc: "Cosine similarity drop between Chroma-CENS frames ~0.5 s apart — harmonic change without full chord recognition.", color: C.teal },
    { name: "Lyrics", weight: "0.10", desc: "Timed lyric line boundaries as secondary evidence. Pipeline remains fully deterministic if lyrics absent.", color: C.green },
    { name: "Onset Flux", weight: "0.06", desc: "Spectral flux novelty curve — measures new note/attack density changes across frames.", color: C.amber },
    { name: "RMS Energy", weight: "0.06", desc: "Absolute RMS level is NOT used — the derivative of RMS in dB is used to detect energy change events.", color: "7C3AED" },
    { name: "Beat / Phrase", weight: "0.02", desc: "16/24/32/48-beat phrase-grid candidates. Provides rhythmic alignment support and snapping anchor.", color: C.navy },
  ];

  // Bar chart visual
  s.addText("Default Source Weights", {
    x: 0.3, y: 1.38, w: 4.5, h: 0.35,
    fontSize: 12, color: C.navy, bold: true, fontFace: "Georgia",
    align: "left",
  });
  const maxW = 4.0;
  sources.forEach(({ name, weight, color }, i) => {
    const barW = parseFloat(weight) * maxW / 0.42;
    const y = 1.8 + i * 0.73;
    s.addText(name, {
      x: 0.3, y, w: 1.7, h: 0.55,
      fontSize: 10, color: C.slate, fontFace: "Calibri",
      align: "right", valign: "middle",
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 2.1, y: y + 0.08, w: barW, h: 0.4,
      fill: { color }, line: { color },
    });
    s.addText(weight, {
      x: 2.15 + barW, y: y + 0.08, w: 0.5, h: 0.4,
      fontSize: 10, color, bold: true,
      align: "left", valign: "middle",
    });
  });

  // Right column: card per source with desc
  sources.forEach(({ name, desc, color }, i) => {
    const x = 7.0;
    const y = 1.42 + i * 0.98;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.05, h: 0.75,
      fill: { color }, line: { color },
    });
    s.addText(name + ":", {
      x: x + 0.15, y, w: 2.1, h: 0.75,
      fontSize: 10, color, bold: true, fontFace: "Georgia",
      align: "left", valign: "middle",
    });
    s.addText(desc, {
      x: x + 2.3, y, w: 3.7, h: 0.75,
      fontSize: 10, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "middle",
    });
  });
}

// ─── SLIDE 12: FEATURE-LEVEL FUSION FORMULA ─────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Feature-Level Fusion: Scoring & Acceptance", "Weighted score + agreement bonus → acceptance decision");
  presenterBadge(s, 12);
  slideNumLabel(s, 12);

  // Steps
  const steps = [
    { n: "1", title: "Temporal Grouping", desc: "Nearby candidates (within merge_window_s = 2.75 s) are placed in the same group — they vote on the same structural transition." },
    { n: "2", title: "One Vote per Source", desc: "Each feature source contributes only its highest-confidence candidate per group — prevents noisy sources from dominating." },
    { n: "3", title: "Weighted Score", desc: "Score = Σ (source_weight × candidate_confidence)  +  agreement_bonus\nagreement_bonus = min(0.15, 0.035 × (source_count − 1))" },
    { n: "4", title: "Acceptance", desc: "Accept if score ≥ threshold (default 0.30), OR if strong SSM candidate exists (confidence ≥ 0.5) — SSM can pass alone." },
    { n: "5", title: "Snapping", desc: "Accepted boundary is snapped to nearest strong onset or beat within a limited window for precise timing alignment." },
  ];

  steps.forEach(({ n, title, desc }, i) => {
    const y = 1.42 + i * 1.05;
    s.addShape(pres.shapes.OVAL, {
      x: 0.3, y: y + 0.1, w: 0.52, h: 0.52,
      fill: { color: C.navy }, line: { color: C.navy },
    });
    s.addText(n, {
      x: 0.3, y: y + 0.1, w: 0.52, h: 0.52,
      fontSize: 13, color: C.white, bold: true,
      align: "center", valign: "middle",
    });
    s.addText(title, {
      x: 0.95, y, w: 3.3, h: 0.38,
      fontSize: 12, color: C.navy, bold: true, fontFace: "Georgia",
      align: "left", valign: "middle",
    });
    s.addText(desc, {
      x: 0.95, y: y + 0.38, w: 5.5, h: 0.62,
      fontSize: 10, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "top",
    });
  });

  // Numeric example
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.0, y: 1.42, w: 6.0, h: 5.85,
    fill: { color: C.white }, line: { color: C.grayLt },
    shadow: { type: "outer", color: "000000", blur: 5, offset: 1, angle: 135, opacity: 0.08 },
    rounding: 0.1,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.0, y: 1.42, w: 6.0, h: 0.45,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText("Numeric Example — Boundary at ~31 s", {
    x: 7.1, y: 1.42, w: 5.8, h: 0.45,
    fontSize: 12, color: C.white, bold: true, fontFace: "Georgia",
    align: "center", valign: "middle",
  });

  codeBox(s, `SSM:    weight=0.42 × conf=0.80 = 0.336
Chord:  weight=0.18 × conf=0.70 = 0.126
RMS:    weight=0.06 × conf=0.60 = 0.036

weighted_sum           = 0.498
agreement_bonus (3 src)= min(0.15, 0.035×2) = 0.070
─────────────────────────────────────────
score                  = 0.568

threshold = 0.30   →   ACCEPTED ✓

Anchor: SSM at 30.82 s
Strong onset nearby: 30.67 s
Snapped boundary: 30.67 s`, 7.1, 1.92, 5.8, 5.25);
}

// ─── SLIDE 13: SECTION DIVIDER — ALGORITHM FUSION ───────────────────────────
{
  const s = pres.addSlide();
  sectionDivider(s, "Algorithm-Level Fusion", "Combining four independent segmenters with weighted voting");
  presenterBadge(s, 13);
  slideNumLabel(s, 13);
}

// ─── SLIDE 14: WHY ALGORITHM FUSION ─────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Why Algorithm-Level Fusion?", "Different algorithms have different error profiles — consensus reduces mistakes");
  presenterBadge(s, 14);
  slideNumLabel(s, 14);

  // Timeline comparison
  const timelines = [
    { label: "custom_librosa", boundaries: [20, 48, 80, 112], color: C.copper },
    { label: "foote", boundaries: [22, 49, 81, 110], color: "5B21B6" },
    { label: "cnmf", boundaries: [19, 47, 85, 113], color: C.teal },
    { label: "scluster", boundaries: [21, 50, 80, 115], color: C.green },
    { label: "FUSION", boundaries: [21, 49, 81, 112], color: C.navy },
  ];

  const totalDur = 130;
  const trackW = 10.5;
  const startX = 1.5;

  timelines.forEach(({ label, boundaries, color }, i) => {
    const y = 1.65 + i * 1.0;
    const isFusion = label === "FUSION";
    // track bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: startX, y: y + 0.25, w: trackW, h: isFusion ? 0.45 : 0.35,
      fill: { color: isFusion ? "EFF6FF" : "F0F4F8" },
      line: { color: color, width: isFusion ? 2 : 1 },
    });
    s.addText(label, {
      x: 0.05, y: y + 0.1, w: 1.4, h: 0.55,
      fontSize: isFusion ? 12 : 10, color, bold: isFusion, fontFace: "Calibri",
      align: "right", valign: "middle",
    });
    boundaries.forEach((b) => {
      const bx = startX + (b / totalDur) * trackW;
      s.addShape(pres.shapes.RECTANGLE, {
        x: bx - 0.015, y: y + 0.1, w: 0.04, h: isFusion ? 0.65 : 0.55,
        fill: { color }, line: { color },
      });
    });
  });

  // 0s / end label
  s.addText("0 s", { x: startX, y: 6.7, w: 0.5, h: 0.25, fontSize: 9, color: C.gray, align: "left" });
  s.addText("130 s", { x: startX + trackW - 0.5, y: 6.7, w: 0.6, h: 0.25, fontSize: 9, color: C.gray, align: "right" });

  // Why not average?
  const reasons = [
    "Algorithms produce different numbers of boundaries",
    "Same transition may be predicted at slightly different timestamps",
    "Algorithm reliability is not equal — custom pipeline is more informed",
    "Confidence values carry additional evidence strength per boundary",
  ];
  s.addText("Why not simple averaging?", {
    x: 7.5, y: 1.65, w: 5.5, h: 0.38,
    fontSize: 13, color: C.navy, bold: true, fontFace: "Georgia", align: "left",
  });
  reasons.forEach((r, i) => {
    s.addShape(pres.shapes.OVAL, {
      x: 7.5, y: 2.1 + i * 0.7, w: 0.22, h: 0.22,
      fill: { color: C.copper }, line: { color: C.copper },
    });
    s.addText(r, {
      x: 7.8, y: 2.06 + i * 0.7, w: 5.2, h: 0.55,
      fontSize: 11, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "middle",
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.5, y: 4.95, w: 5.5, h: 0.85,
    fill: { color: "F0FDF4" }, line: { color: C.green }, rounding: 0.08,
  });
  s.addText("→ Weighted voting groups nearby votes, deduplicates per-algorithm, and computes:\n     score = Σ (algorithm_weight × boundary_confidence)", {
    x: 7.65, y: 4.98, w: 5.2, h: 0.79,
    fontSize: 10, color: C.green, fontFace: "Calibri", align: "left", valign: "middle",
  });
}

// ─── SLIDE 15: FUSION ORCHESTRATION ─────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Fusion Orchestration & Readiness", "ResultListener coordinates the two-phase dispatch sequence");
  presenterBadge(s, 15);
  slideNumLabel(s, 15);

  // State machine flow
  const states = [
    { label: "Fusion\nRequested", color: C.slate },
    { label: "Base Workers\nDispatched\n(4 parallel)", color: C.teal },
    { label: "Base Results\nCollected", color: C.green },
    { label: "Readiness\nChecked\n(all 4 resolved?)", color: C.amber },
    { label: "Fusion\nDispatched", color: C.copper },
    { label: "Task\nCompleted", color: C.navy },
  ];

  states.forEach(({ label, color }, i) => {
    const x = 0.3 + i * 2.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.55, w: 1.9, h: 1.55,
      fill: { color }, line: { color }, rounding: 0.1,
    });
    s.addText(label, {
      x, y: 1.55, w: 1.9, h: 1.55,
      fontSize: 10, color: C.white, bold: true, fontFace: "Calibri",
      align: "center", valign: "middle",
    });
    if (i < states.length - 1) {
      s.addText("→", {
        x: x + 1.9, y: 2.15, w: 0.25, h: 0.4,
        fontSize: 14, color: C.copper, bold: true, align: "center",
      });
    }
  });

  // _maybe_dispatch_fusion() logic
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 3.5, w: 12.7, h: 0.42,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText("_maybe_dispatch_fusion()  —  ResultListener guard conditions", {
    x: 0.3, y: 3.5, w: 12.7, h: 0.42,
    fontSize: 12, color: C.white, bold: true, fontFace: "Georgia",
    align: "center", valign: "middle",
  });

  const guards = [
    ["Guard 1", "Fusion not requested → skip", C.teal],
    ["Guard 2", "Fusion result already received → skip (no duplicate dispatch)", C.green],
    ["Guard 3", "fusion__dispatched flag already set → skip", C.amber],
    ["Condition", "All 4 baselines resolved (completed OR failed) → check success count", C.copper],
    ["Success ≥ 2", "Dispatch fusion worker with available algorithm_results payload", C.green],
    ["Success < 2", "Produce failed fusion result directly — no worker needed", "9B1C1C"],
  ];

  guards.forEach(([label, desc, color], i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.3 + col * 6.55;
    const y = 4.05 + row * 0.88;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 1.5, h: 0.65,
      fill: { color }, line: { color }, rounding: 0.08,
    });
    s.addText(label, {
      x, y, w: 1.5, h: 0.65,
      fontSize: 9, color: C.white, bold: true,
      align: "center", valign: "middle",
    });
    s.addText(desc, {
      x: x + 1.58, y, w: 4.8, h: 0.65,
      fontSize: 10, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "middle",
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.3, y: 6.75, w: 12.7, h: 0.4,
    fill: { color: "FFF7ED" }, line: { color: C.amber }, rounding: 0.05,
  });
  s.addText("resolved = worker responded (completed or failed)  •  successful = completed with non-empty segments", {
    x: 0.5, y: 6.75, w: 12.3, h: 0.4,
    fontSize: 10, color: C.amber, bold: true, fontFace: "Calibri",
    align: "center", valign: "middle",
  });
}

// ─── SLIDE 16: WEIGHTED VOTING ───────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Algorithm-Level Weighted Voting", "How fusion decides which boundaries survive");
  presenterBadge(s, 16);
  slideNumLabel(s, 16);

  // Weights table
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.42, w: 4.0, h: 0.4,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText("Default Algorithm Weights", {
    x: 0.3, y: 1.42, w: 4.0, h: 0.4,
    fontSize: 12, color: C.white, bold: true, fontFace: "Georgia",
    align: "center", valign: "middle",
  });
  const weightData = [
    ["custom_librosa", "0.35", C.copper],
    ["scluster", "0.30", C.green],
    ["cnmf", "0.20", C.teal],
    ["foote", "0.15", "5B21B6"],
  ];
  weightData.forEach(([alg, w, color], i) => {
    const y = 1.85 + i * 0.68;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y, w: 4.0, h: 0.58,
      fill: { color: i % 2 === 0 ? C.white : "F8FAFC" },
      line: { color: C.grayLt },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.32, y: y + 0.08, w: parseFloat(w) * 8, h: 0.4,
      fill: { color }, line: { color },
    });
    s.addText(`${alg}  ${w}`, {
      x: 0.42, y, w: 3.8, h: 0.58,
      fontSize: 11, color: C.white, bold: true, fontFace: "Calibri",
      align: "left", valign: "middle",
    });
  });

  // Steps
  const steps = [
    "Collect all internal boundary votes from each algorithm result",
    "Remove start (≤0.5 s) and end-edge boundaries from voting",
    "Group votes within merge_window_seconds = 2.5 s into boundary groups",
    "Per group: each algorithm contributes at most ONE vote (highest confidence)",
    "Compute: score = Σ(algorithm_weight × boundary_confidence)",
    "Accept if: score ≥ threshold (0.45) OR unique_algorithm_count ≥ required_vote_count (2)",
  ];

  s.addText("Voting Steps", {
    x: 4.6, y: 1.42, w: 8.4, h: 0.38,
    fontSize: 13, color: C.navy, bold: true, fontFace: "Georgia", align: "left",
  });
  steps.forEach((step, i) => {
    s.addShape(pres.shapes.OVAL, {
      x: 4.6, y: 1.88 + i * 0.7, w: 0.28, h: 0.28,
      fill: { color: C.copper }, line: { color: C.copper },
    });
    s.addText(step, {
      x: 4.98, y: 1.85 + i * 0.7, w: 8.0, h: 0.62,
      fontSize: 11, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "middle",
    });
  });

  // Numeric example box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 4.6, w: 12.7, h: 0.4,
    fill: { color: C.navyMid }, line: { color: C.navyMid },
  });
  s.addText("Numeric Example", {
    x: 0.3, y: 4.6, w: 12.7, h: 0.4,
    fontSize: 12, color: C.white, bold: true, fontFace: "Georgia",
    align: "center", valign: "middle",
  });
  codeBox(s, `Group at ~60 s:
  custom_librosa: time=60.2 s, confidence=0.90  →  0.35 × 0.90 = 0.315
  scluster:       time=61.0 s, confidence=0.80  →  0.30 × 0.80 = 0.240
  ──────────────────────────────────────────────────────────────────────
  group_score = 0.315 + 0.240 = 0.555     threshold = 0.45   →   ACCEPTED ✓
  anchor_strategy=custom_snap  →  fused_time = 60.2 s  (custom's snapped timestamp)`, 0.3, 5.05, 12.7, 2.1);
}

// ─── SLIDE 17: DIAGNOSTICS & FAILURE ────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Fusion Diagnostics & Failure Handling", "Explainability is built into every boundary decision");
  presenterBadge(s, 17);
  slideNumLabel(s, 17);

  // Left: diagnostics JSON
  codeBox(s, `// Each fused boundary carries full decision trace:
{
  "time": 60.2,
  "confidence": 0.555,
  "source": "algorithm_fusion",
  "sources": ["custom_librosa", "scluster"],
  "metadata": {
    "score": 0.555,
    "raw_times": [
      { "algorithm": "custom_librosa",
        "time": 60.2, "confidence": 0.90 },
      { "algorithm": "scluster",
        "time": 61.0, "confidence": 0.80 }
    ],
    "accepted": true,
    "fused_time": 60.2
  }
}`, 0.3, 1.42, 5.8, 5.55);

  // Right: failure handling
  s.addText("Failure Handling", {
    x: 6.5, y: 1.42, w: 6.5, h: 0.38,
    fontSize: 14, color: C.navy, bold: true, fontFace: "Georgia", align: "left",
  });

  const cases = [
    { title: "Worker Exception", desc: "BaseWorker catches exception, publishes a failed normalized result — marks algorithm as resolved-but-failed", color: "9B1C1C" },
    { title: "< 2 Successful Results", desc: "ResultListener generates a failed fusion result directly without dispatching the fusion worker", color: C.amber },
    { title: "Rejected Boundary Group", desc: "Group score below threshold AND vote count insufficient → logged in diagnostics as rejected, not returned", color: C.teal },
    { title: "Worker Disappears (Known Limit)", desc: "If a worker dies without publishing any result, _maybe_dispatch_fusion() never sees all 4 resolved → task stays in processing. Watchdog/timeout needed (future work).", color: C.copper },
  ];

  cases.forEach(({ title, desc, color }, i) => {
    const y = 1.9 + i * 1.38;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.5, y, w: 6.5, h: 1.2,
      fill: { color: C.white }, line: { color: C.grayLt },
      shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.07 },
      rounding: 0.07,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.5, y, w: 0.06, h: 1.2,
      fill: { color }, line: { color },
    });
    s.addText(title, {
      x: 6.65, y: y + 0.1, w: 6.2, h: 0.35,
      fontSize: 12, color, bold: true, fontFace: "Georgia",
      align: "left", valign: "middle",
    });
    s.addText(desc, {
      x: 6.65, y: y + 0.48, w: 6.2, h: 0.65,
      fontSize: 10, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "top",
    });
  });
}

// ─── SLIDE 18: STRUCTURAL vs SEMANTIC LABELS ─────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Structural vs Semantic Labeling", "Two-layer approach: similarity claim vs musical role claim");
  presenterBadge(s, 18);
  slideNumLabel(s, 18);

  // Example timeline
  const exSegs = [
    { label: "Struct: A\nSemantic: Intro", color: C.copper },
    { label: "Struct: B\nSemantic: Verse", color: C.teal },
    { label: "Struct: C\nSemantic: Chorus", color: C.green },
    { label: "Struct: B\nSemantic: Verse", color: C.teal },
    { label: "Struct: C\nSemantic: Chorus", color: C.green },
    { label: "Struct: D\nSemantic: Bridge", color: C.amber },
    { label: "Struct: A\nSemantic: Outro", color: C.copper },
  ];

  exSegs.forEach(({ label, color }, i) => {
    const segW = 13.3 / exSegs.length;
    s.addShape(pres.shapes.RECTANGLE, {
      x: i * segW, y: 1.42, w: segW - 0.04, h: 1.1,
      fill: { color }, line: { color },
    });
    s.addText(label, {
      x: i * segW, y: 1.42, w: segW - 0.04, h: 1.1,
      fontSize: 9, color: C.white, bold: true, fontFace: "Calibri",
      align: "center", valign: "middle",
    });
  });

  // Two columns explanation
  infoCard(s, "Structural Label (A/B/C)", [
    "Based on audio descriptor similarity (Chroma, MFCC, RMS, onset density)",
    "Agglomerative clustering; best silhouette score selects k",
    "A = most frequent cluster, B = second, etc.",
    "Does NOT imply Verse/Chorus — only similarity",
    "This is the reliable, scientifically defensible layer",
  ], 0.3, 2.75, 6.1, 3.25, C.copper);

  infoCard(s, "Semantic Label (Intro/Verse/Chorus…)", [
    "Heuristic inference layer — weaker claim",
    "Intro: first non-silence segment in early 20% of track",
    "Outro: last non-silence in final 25%",
    "Chorus: repeated cluster with higher RMS energy",
    "Verse: other repeated clusters",
    "Bridge: unique, middle, long-enough segment",
    "Does NOT overwrite structural label",
  ], 6.6, 2.75, 6.4, 3.25, C.teal);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.3, y: 6.22, w: 12.7, h: 0.46,
    fill: { color: "FFF8F0" }, line: { color: C.copper }, rounding: 0.05,
  });
  s.addText("Semantic labels are heuristic — confidence & reason fields are always stored so the strength of the claim is transparent", {
    x: 0.5, y: 6.22, w: 12.3, h: 0.46,
    fontSize: 11, color: C.amber, bold: true, fontFace: "Calibri",
    align: "center", valign: "middle",
  });
}

// ─── SLIDE 19: EVALUATION ───────────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Evaluation with SALAMI Dataset", "Measuring boundary detection quality at two tolerance levels");
  presenterBadge(s, 19);
  slideNumLabel(s, 19);

  // Dataset info
  infoCard(s, "SALAMI Dataset", [
    "Human-annotated structural intervals for music tracks",
    "Multiple annotators per track → annotation uncertainty is real",
    "Used as reference for boundary detection evaluation",
    "Evaluation uses mir_eval.segment.detection(trim=True)",
  ], 0.3, 1.42, 5.8, 2.3, C.navy);

  // Metrics explanation
  infoCard(s, "Evaluation Metrics", [
    "Precision = TP / (TP + FP)  — how many predicted boundaries are correct?",
    "Recall = TP / (TP + FN)  — how many ground-truth boundaries were found?",
    "F1 = 2 × P × R / (P + R)  — harmonic mean, penalizes imbalance",
    "over-segmentation → low Precision;  under-segmentation → low Recall",
  ], 0.3, 3.92, 5.8, 2.65, C.teal);

  // Tolerances column
  s.addText("Two Tolerance Windows", {
    x: 6.5, y: 1.42, w: 6.5, h: 0.38,
    fontSize: 14, color: C.navy, bold: true, fontFace: "Georgia",
  });

  const tols = [
    { tol: "±0.5 s", label: "Strict", desc: "Tests exact timestamp localization. Low F1@0.5 means algorithm finds the right region but is imprecise about when.", color: C.copper },
    { tol: "±3.0 s", label: "Lenient", desc: "Tests whether the correct structural region is detected. High F1@3.0 but low F1@0.5 suggests good region detection, weak timing.", color: C.teal },
  ];

  tols.forEach(({ tol, label, desc, color }, i) => {
    const y = 1.95 + i * 2.2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.5, y, w: 6.5, h: 2.0,
      fill: { color: C.white }, line: { color: C.grayLt },
      shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.07 },
      rounding: 0.08,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.5, y, w: 6.5, h: 0.5,
      fill: { color }, line: { color },
    });
    s.addText(`${tol}  —  ${label}`, {
      x: 6.5, y, w: 6.5, h: 0.5,
      fontSize: 16, color: C.white, bold: true, fontFace: "Georgia",
      align: "center", valign: "middle",
    });
    s.addText(desc, {
      x: 6.65, y: y + 0.58, w: 6.2, h: 1.35,
      fontSize: 12, color: C.slate, fontFace: "Calibri",
      align: "left", valign: "top",
    });
  });

  // Metrics table placeholder
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 6.68, w: 12.7, h: 0.4,
    fill: { color: "F0FDF4" }, line: { color: C.green },
  });
  s.addText("Batch evaluation runs across all SALAMI tracks with concurrency — averages P / R / F1 per algorithm for rigorous comparison", {
    x: 0.5, y: 6.68, w: 12.3, h: 0.4,
    fontSize: 10, color: C.green, bold: true, fontFace: "Calibri",
    align: "center", valign: "middle",
  });
}

// ─── SLIDE 20: CONCLUSION ───────────────────────────────────────────────────
{
  const s = pres.addSlide();
  headerBar(s, "Contributions, Limitations & Future Work", "What we built, what we learned, and where it can go");
  presenterBadge(s, 20);
  slideNumLabel(s, 20);

  const cols = [
    {
      title: "Contributions",
      icon: "✓",
      color: C.green,
      items: [
        "Distributed multi-algorithm pipeline with RabbitMQ",
        "Common AlgorithmResult schema for interoperability",
        "Feature-level fusion inside custom_librosa (6 sources)",
        "Algorithm-level weighted voting fusion worker",
        "SALAMI-based multi-tolerance evaluation (0.5 s / 3.0 s)",
        "Full decision diagnostics for explainability",
        "Structural / semantic two-layer labeling separation",
      ],
    },
    {
      title: "Limitations",
      icon: "⚠",
      color: C.amber,
      items: [
        "Confidence values not fully calibrated across algorithms",
        "Fusion weights are static — not learned from data",
        "No watchdog / timeout for disappeared workers",
        "Semantic labels are heuristic — not benchmark-validated",
        "anchor_strategy schema/service default mismatch",
        "Genre-specific minimum segment duration not tuned",
      ],
    },
    {
      title: "Future Work",
      icon: "→",
      color: C.teal,
      items: [
        "Dataset-based automatic weight optimization",
        "Learned confidence calibration per algorithm",
        "Genre-adaptive fusion weights",
        "Periodic watchdog + dead-letter queue + retry",
        "Learned meta-classifier for boundary groups",
        "Richer frontend: vote timeline visualization",
        "New algorithm workers via existing worker interface",
      ],
    },
  ];

  cols.forEach(({ title, icon, color, items }, i) => {
    const x = 0.3 + i * 4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.42, w: 4.05, h: 5.95,
      fill: { color: C.white }, line: { color: C.grayLt },
      shadow: { type: "outer", color: "000000", blur: 5, offset: 1, angle: 135, opacity: 0.08 },
      rounding: 0.1,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.42, w: 4.05, h: 0.6,
      fill: { color }, line: { color },
    });
    s.addText(`${icon}  ${title}`, {
      x, y: 1.42, w: 4.05, h: 0.6,
      fontSize: 14, color: C.white, bold: true, fontFace: "Georgia",
      align: "center", valign: "middle",
    });
    const richItems = [];
    items.forEach((item, j) => {
      richItems.push({
        text: item,
        options: {
          bullet: true,
          breakLine: j < items.length - 1,
          fontSize: 10.5,
          color: C.slate,
          fontFace: "Calibri",
        },
      });
    });
    s.addText(richItems, {
      x: x + 0.15, y: 2.1, w: 3.75, h: 5.15,
      valign: "top", margin: [4, 6, 4, 6],
    });
  });

  // Closing statement
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 7.35, w: 13.3, h: 0.4,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText(
    "\"The contribution is not one new algorithm — it is the integration of multi-feature and multi-algorithm decisions into a distributed, explainable, evaluable system.\"",
    {
      x: 0.3, y: 7.35, w: 12.7, h: 0.4,
      fontSize: 9.5, color: C.copperLt, fontFace: "Calibri",
      align: "center", valign: "middle",
    }
  );
}

// ─── WRITE FILE ──────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "docs/MusicSegmentation_FinalPresentation.pptx" })
  .then(() => console.log("✅  Presentation created: docs/MusicSegmentation_FinalPresentation.pptx"))
  .catch((err) => { console.error("❌  Error:", err); process.exit(1); });
