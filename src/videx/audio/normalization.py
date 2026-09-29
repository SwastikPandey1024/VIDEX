"""Conservative transcript normalization for speech recognition results in VIDEX.

Preserves raw ASR transcript strings while generating canonical normalized text
suitable for temporal fusion, search indexing, and evidence reconciliation.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_transcript"]

# Explicit zero-width characters to strip (BOM and zero-width spaces)
_ZERO_WIDTH_CHARS_TO_STRIP = {
    "\ufeff",  # Zero-width no-break space / BOM
    "\u200b",  # Zero-width space
    "\u2060",  # Word joiner
}

# Regex to collapse multiple whitespace characters
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_transcript(
    text: str,
    lowercase: bool = False,
    preserve_indic_zw: bool = True,
) -> str:
    """Conservatively normalize a speech transcript string.

    Rules applied:
    1. Unicode NFKC normalization.
    2. Strips zero-width space (U+200B), BOM (U+FEFF), and word-joiners (U+2060).
    3. Preserves Zero-Width Joiner (U+200D) and Zero-Width Non-Joiner (U+200C)
       crucial for correct Hindi and Devanagari script conjunct rendering.
    4. Collapses multiple spaces, tabs, and newlines into single spaces.
    5. Strips leading and trailing whitespace.
    6. Optionally applies case-folding (lowercase).

    Args:
        text: Raw transcript string.
        lowercase: Whether to lowercase Latin characters. Defaults to False to
                   preserve proper nouns and acronyms.
        preserve_indic_zw: Whether to preserve ZWJ (\\u200d) and ZWNJ (\\u200c).
                           Defaults to True.

    Returns:
        Conservatively normalized transcript string.
    """
    if not text:
        return ""

    # Step 1: Unicode NFKC normalization
    norm = unicodedata.normalize("NFKC", text)

    # Step 2: Strip specific zero-width and invisible artifacts
    filtered_chars: list[str] = []
    for char in norm:
        if char in _ZERO_WIDTH_CHARS_TO_STRIP:
            continue
        if not preserve_indic_zw and char in ("\u200c", "\u200d"):
            continue
        filtered_chars.append(char)

    norm = "".join(filtered_chars)

    # Step 3: Collapse whitespace
    norm = _WHITESPACE_RE.sub(" ", norm).strip()

    # Step 4: Optional lowercasing
    if lowercase:
        norm = norm.lower()

    return norm
