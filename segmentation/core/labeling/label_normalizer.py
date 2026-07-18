"""
SALAMI label normalisation.

Maps raw SALAMI annotation labels (which can be a mix of structural symbols like
'A', 'B', 'A'' and natural-language labels like 'intro', 'verse', 'chorus') to
a consistent set of human-readable section type names.

SALAMI uses two annotation tiers:
  - Coarse (upper-case letters + functional labels): 'Intro', 'Verse', 'Chorus', etc.
  - Fine-grained (lower-case): 'a', 'a'', 'b', etc. (motivic sub-sections)

Normalisation strategy:
  - If a label matches a known functional name → map to canonical form.
  - Upper-case letters (A, B, C, …) → treat as distinct sections (keep as-is
    with a "Section " prefix for readability).
  - Lower-case letters (a, b, a', …) → "Motif <X>" to distinguish from coarse.
  - Unknown labels → returned as-is with a note.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_FUNCTIONAL_MAP: dict[str, str] = {
    # Intro variants
    "intro": "Intro",
    "introduction": "Intro",
    "opening": "Intro",
    "prelude": "Intro",
    # Verse
    "verse": "Verse",
    "v": "Verse",
    "verses": "Verse",
    "rap_verse": "Verse",
    # Pre-chorus / build
    "pre-chorus": "Pre-Chorus",
    "prechorus": "Pre-Chorus",
    "pre_chorus": "Pre-Chorus",
    "buildup": "Pre-Chorus",
    "build": "Pre-Chorus",
    # Chorus
    "chorus": "Chorus",
    "hook": "Chorus",
    "refrain": "Chorus",
    # Post-chorus
    "post-chorus": "Post-Chorus",
    "postchorus": "Post-Chorus",
    # Bridge
    "bridge": "Bridge",
    "c": "Bridge",
    "transition": "Bridge",
    "trans": "Bridge",
    # Instrumental
    "instrumental": "Instrumental",
    "solo": "Instrumental",
    "guitar_solo": "Instrumental",
    "interlude": "Instrumental",
    "break": "Instrumental",
    # Outro / Coda
    "outro": "Outro",
    "coda": "Outro",
    "ending": "Outro",
    "end": "Outro",
    "fade": "Outro",
    "fade-out": "Outro",
    "fadeout": "Outro",
    "fade_out": "Outro",
    # Silence / noise
    "silence": "Silence",
    "noise": "Silence",
    "applause": "Silence",
    # Miscellaneous
    "spoken": "Spoken",
    "spoken_word": "Spoken",
    "dialogue": "Spoken",
    "narration": "Spoken",
    "ad_lib": "Ad-Lib",
    "adlib": "Ad-Lib",
    "vamp": "Vamp",
}

# Section letters A–Z (upper-case) → "Section A", "Section B", etc.
_UPPER_LETTER_RE = re.compile(r"^([A-Z])('*)\s*$")

# Motivic sub-section letters a–z (lower-case) → "Motif a", "Motif a'", etc.
_LOWER_LETTER_RE = re.compile(r"^([a-z])('*)\s*$")


def normalize_label(raw_label: str) -> str:
    """
    Normalise a raw SALAMI annotation label to a human-readable section type.

    Parameters
    ----------
    raw_label : Raw label string from the annotation file, e.g. "A", "a'",
                "chorus", "Verse", "silence", "bridge".

    Returns
    -------
    Normalised label string.

    Examples
    --------
    >>> normalize_label("verse")
    'Verse'
    >>> normalize_label("A")
    'Section A'
    >>> normalize_label("a'")
    "Motif a'"
    >>> normalize_label("Intro")
    'Intro'
    >>> normalize_label("Z1")
    'Z1'
    """
    label = raw_label.strip()
    if not label:
        return "Unknown"

    # 1. Direct lookup in functional map (case-insensitive).
    lower = label.lower().replace("-", "_")
    if lower in _FUNCTIONAL_MAP:
        return _FUNCTIONAL_MAP[lower]

    # Also try stripping trailing punctuation / numbers.
    clean = re.sub(r"[\d_\s]+$", "", lower).strip()
    if clean in _FUNCTIONAL_MAP:
        return _FUNCTIONAL_MAP[clean]

    # 2. Upper-case structural letter: "A", "B", "A'", etc.
    m = _UPPER_LETTER_RE.match(label)
    if m:
        letter = m.group(1)
        primes = m.group(2)
        return f"Section {letter}{primes}"

    # 3. Lower-case motivic letter: "a", "b", "a'", etc.
    m = _LOWER_LETTER_RE.match(label)
    if m:
        letter = m.group(1)
        primes = m.group(2)
        return f"Motif {letter}{primes}"

    # 4. Capitalised first letter (e.g. "Verse", "Chorus" — already clean).
    if label[0].isupper() and len(label) > 1 and label[1:].islower():
        return label

    # 5. Unknown — return as-is to preserve information.
    return label
