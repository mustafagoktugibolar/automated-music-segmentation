export function secToLabel(s) {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}

export function segLabel(s) {
  return s.semantic_label || s.label || s.section_type || "—";
}

/** Groups segments sharing the same structural identity into letters A, B, C… in first-seen order. */
export function buildStructuralGroups(segments) {
  const map = {};
  let next = 0;
  const letters = "ABCDEFGHIJ";
  segments.forEach((s) => {
    const key = s.structural_label || segLabel(s);
    if (!(key in map)) map[key] = letters[next++ % letters.length];
  });
  return map;
}

export function segStruct(seg, structuralGroups) {
  return seg.structural_label || structuralGroups[seg.structural_label || segLabel(seg)] || "—";
}

/**
 * Picks the algorithm to drive a results-derived view: the user's explicit
 * `viewAlgo` choice if it has results, else fusion, else any requested algo
 * with array results.
 */
export function primaryAlgoAndSegments(state) {
  const { results, requested, viewAlgo } = state;
  const algo = (viewAlgo && Array.isArray(results[viewAlgo]))
    ? viewAlgo
    : results.fusion
      ? "fusion"
      : (requested.find((a) => Array.isArray(results[a])) ||
         Object.keys(results).find((k) => !k.includes("__") && Array.isArray(results[k])));
  return { algo, segments: (algo && Array.isArray(results[algo])) ? results[algo] : [] };
}

/** All algorithm ids with actual (array) results available to view. */
export function availableAlgos(state) {
  const { results, requested } = state;
  const ids = new Set(requested.filter((a) => Array.isArray(results[a])));
  for (const k of Object.keys(results)) {
    if (!k.includes("__") && Array.isArray(results[k])) ids.add(k);
  }
  return Array.from(ids);
}

/** Interior boundary times (excludes t=0 start) for a segment list. */
export function boundaryTimes(segments) {
  return (segments || []).map((s) => s.start).filter((t) => t > 0);
}

export function maxDuration(segmentLists) {
  const ends = segmentLists.flat().map((s) => s.end).filter((v) => Number.isFinite(v));
  return Math.max(...ends, 1);
}
