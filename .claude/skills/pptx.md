---
name: pptx
description: Create, edit, and analyze PowerPoint presentations. Triggers on "deck", "slides", "presentation", or .pptx file mentions. Creates from scratch with PptxGenJS or edits existing files by unpacking/repacking XML.
---

# PPTX Skill

Handle all PowerPoint presentation work—creation, editing, analysis, and content extraction.

## Core Workflows

**Reading content:** Use `python -m markitdown presentation.pptx` for text extraction or `thumbnail.py` for visual overview.

**Editing existing files:** Unpack → modify slides → repack (see Editing section below).

**Creating from scratch:** Use PptxGenJS when no template exists (see PptxGenJS section below).

## Design Principles

Avoid plain text-only slides. Instead:
- Select topic-specific color palettes with one dominant color (60-70%), supporting tones, and an accent
- Pair contrasting fonts (e.g., Georgia with Calibri)
- Include visual elements on every slide—images, charts, icons, or shapes
- Vary layouts across slides; left-align body text; center only titles
- Use 36-44pt for titles, 14-16pt for body, with 0.5" minimum margins

10 example color schemes: Midnight Executive, Forest & Moss, Ocean & Sand, Ruby & Slate, Violet & Gold, Arctic Blue, Terracotta & Cream, Charcoal & Teal, Burgundy & Ivory, Navy & Copper.

## Quality Assurance

Always assume issues exist. Conduct content QA using markitdown, then use subagents for visual inspection by converting slides to JPEG images and checking for overlaps, text overflow, contrast problems, and alignment issues. Complete at least one fix-and-verify cycle before declaring success.

---

## Editing Existing Presentations

### Workflow (7 phases — order is mandatory)

1. **Analyze** — `python thumbnail.py presentation.pptx` + `python -m markitdown presentation.pptx`
2. **Plan** — Map content to slides, prioritize layout variety
3. **Unpack** — `python unpack.py presentation.pptx ./unpacked/`
4. **Build** — Delete, duplicate (`python add_slide.py`), reorder slides. All structural changes before content edits.
5. **Edit** — Modify `unpacked/ppt/slides/slide*.xml` files (parallelizable with subagents)
6. **Clean** — `python clean.py ./unpacked/` to remove orphaned references
7. **Pack** — `python pack.py ./unpacked/ output.pptx`

### XML Editing Rules

- Quotation marks: use `&#x201C;` and `&#x201D;` (never raw `"`)
- Multi-item content: separate `<a:p>` elements, never concatenated strings
- Bold: `b="1"` attribute on the run properties element
- Use the Edit tool for XML changes, not sed/awk

### Layout Diversity

Monotonous presentations are a common failure mode. Seek varied layouts:
- Multi-column designs
- Image + text combinations
- Full-bleed image slides
- Quote slides
- Section dividers
- Icon grids

---

## Creating from Scratch with PptxGenJS

### Setup

```javascript
const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';  // LAYOUT_16x9, LAYOUT_16x10, LAYOUT_4x3, LAYOUT_WIDE
pres.author = 'Your Name';
pres.title = 'Presentation Title';

let slide = pres.addSlide();
pres.writeFile({ fileName: "Presentation.pptx" });
```

### Layout Dimensions

| Layout | Width | Height |
|--------|-------|--------|
| `LAYOUT_16x9` | 10" | 5.625" (default) |
| `LAYOUT_16x10` | 10" | 6.25" |
| `LAYOUT_4x3` | 10" | 7.5" |
| `LAYOUT_WIDE` | 13.3" | 7.5" |

### Text

```javascript
// Basic
slide.addText("Simple Text", {
  x: 1, y: 1, w: 8, h: 2, fontSize: 24, fontFace: "Arial",
  color: "363636", bold: true, align: "center", valign: "middle"
});

// Character spacing
slide.addText("SPACED", { x: 1, y: 1, w: 8, h: 1, charSpacing: 6 });

// Rich text array
slide.addText([
  { text: "Bold ", options: { bold: true } },
  { text: "Italic", options: { italic: true } }
], { x: 1, y: 3, w: 8, h: 1 });

// Multi-line (breakLine required)
slide.addText([
  { text: "Line 1", options: { breakLine: true } },
  { text: "Line 2", options: { breakLine: true } },
  { text: "Line 3" }
], { x: 0.5, y: 0.5, w: 8, h: 2 });

// margin: 0 when aligning text precisely with shapes
slide.addText("Title", { x: 0.5, y: 0.3, w: 9, h: 0.6, margin: 0 });
```

### Lists & Bullets

```javascript
// Correct: multiple bullets
slide.addText([
  { text: "First item", options: { bullet: true, breakLine: true } },
  { text: "Second item", options: { bullet: true, breakLine: true } },
  { text: "Third item", options: { bullet: true } }
], { x: 0.5, y: 0.5, w: 8, h: 3 });

// Sub-items and numbered lists
{ text: "Sub-item", options: { bullet: true, indentLevel: 1 } }
{ text: "First", options: { bullet: { type: "number" }, breakLine: true } }
```

Never use unicode bullets like `"•"` — creates double bullets.

### Shapes

```javascript
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.8, w: 1.5, h: 3.0,
  fill: { color: "FF0000" }, line: { color: "000000", width: 2 }
});

// With shadow
slide.addShape(pres.shapes.RECTANGLE, {
  x: 1, y: 1, w: 3, h: 2,
  fill: { color: "FFFFFF" },
  shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.15 }
});
```

Shadow options: `type` ("outer"/"inner"), `color` (6-char hex, no `#`), `blur` (0-100 pt), `offset` (≥0), `angle` (0-359°), `opacity` (0.0-1.0). To cast shadow upward use `angle: 270` with positive offset.

Available shapes: `RECTANGLE`, `OVAL`, `LINE`, `ROUNDED_RECTANGLE`

### Images

```javascript
// From path
slide.addImage({ path: "images/chart.png", x: 1, y: 1, w: 5, h: 3 });

// From URL
slide.addImage({ path: "https://example.com/image.jpg", x: 1, y: 1, w: 5, h: 3 });

// From base64 (faster)
slide.addImage({ data: "image/png;base64,iVBORw0KGgo...", x: 1, y: 1, w: 5, h: 3 });

// Options
slide.addImage({ path: "img.png", x: 1, y: 1, w: 5, h: 3,
  rotate: 45, rounding: true, transparency: 50, flipH: false });

// Preserve aspect ratio
const origW = 1978, origH = 923, maxH = 3.0;
const calcW = maxH * (origW / origH);
slide.addImage({ path: "img.png", x: (10 - calcW) / 2, y: 1.2, w: calcW, h: maxH });
```

Sizing modes: `contain` (fit inside), `cover` (fill, may crop), `crop` (cut portion).

### Icons (react-icons → PNG)

```javascript
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaCheckCircle } = require("react-icons/fa");

async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

const iconData = await iconToBase64Png(FaCheckCircle, "#4472C4", 256);
slide.addImage({ data: iconData, x: 1, y: 1, w: 0.5, h: 0.5 });
```

Install: `npm install -g react-icons react react-dom sharp`  
Icon sets: `react-icons/fa` (Font Awesome), `react-icons/md` (Material), `react-icons/hi` (Heroicons), `react-icons/bi` (Bootstrap)

### Slide Backgrounds

```javascript
slide.background = { color: "F1F1F1" };
slide.background = { path: "https://example.com/bg.jpg" };
slide.background = { data: "image/png;base64,..." };
```

### Tables

```javascript
slide.addTable([
  ["Header 1", "Header 2"],
  ["Cell 1", "Cell 2"]
], { x: 1, y: 1, w: 8, h: 2, border: { pt: 1, color: "999999" } });

// With merged cells
let tableData = [
  [{ text: "Header", options: { fill: { color: "6699CC" }, color: "FFFFFF", bold: true } }, "Cell"],
  [{ text: "Merged", options: { colspan: 2 } }]
];
slide.addTable(tableData, { x: 1, y: 3.5, w: 8, colW: [4, 4] });
```

### Charts

```javascript
// Bar
slide.addChart(pres.charts.BAR, [{ name: "Sales", labels: ["Q1","Q2"], values: [4500,5500] }], {
  x: 0.5, y: 0.6, w: 6, h: 3, barDir: 'col', showTitle: true, title: 'Sales'
});

// Line
slide.addChart(pres.charts.LINE, [{ name: "Temp", labels: ["Jan","Feb"], values: [32,35] }], {
  x: 0.5, y: 4, w: 6, h: 3, lineSmooth: true
});

// Pie
slide.addChart(pres.charts.PIE, [{ name: "Share", labels: ["A","B"], values: [35,65] }], {
  x: 7, y: 1, w: 5, h: 4, showPercent: true
});
```

Modern chart styling:

```javascript
slide.addChart(pres.charts.BAR, chartData, {
  chartColors: ["0D9488", "14B8A6"],
  chartArea: { fill: { color: "FFFFFF" }, roundedCorners: true },
  catAxisLabelColor: "64748B", valAxisLabelColor: "64748B",
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelColor: "1E293B", showLegend: false
});
```

Available charts: `BAR`, `LINE`, `PIE`, `DOUGHNUT`, `SCATTER`, `BUBBLE`, `RADAR`

### Slide Masters

```javascript
pres.defineSlideMaster({
  title: 'TITLE_SLIDE', background: { color: '283A5E' },
  objects: [{ placeholder: { options: { name: 'title', type: 'title', x: 1, y: 2, w: 8, h: 2 } } }]
});
let titleSlide = pres.addSlide({ masterName: "TITLE_SLIDE" });
titleSlide.addText("My Title", { placeholder: "title" });
```

---

## Common Pitfalls

1. **Never use `#` with hex colors** — corrupts file. Use `"FF0000"` not `"#FF0000"`.
2. **Never encode opacity in color string** — `"00000020"` corrupts file. Use `opacity: 0.12` property.
3. **Never use unicode bullets** (`"•"`) — creates double bullets. Use `bullet: true`.
4. **Always use `breakLine: true`** between array text items.
5. **Avoid `lineSpacing` with bullets** — use `paraSpaceAfter` instead.
6. **Never reuse `pptxgen()` instances** across presentations.
7. **Never reuse option objects across calls** — PptxGenJS mutates them in-place. Use a factory function: `const makeShadow = () => ({ ... })`.
8. **Don't use `ROUNDED_RECTANGLE` with rectangular accent overlays** — corners won't align. Use `RECTANGLE` instead.
9. **Shadow `offset` must be non-negative** — use `angle: 270` for upward shadows, not negative offset.
