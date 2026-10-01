"""Conservative text normalization for OCR observations.

Preserves raw engine output faithfully while providing clean, canonical
representations for indexing, deduplication, and temporal fusion.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode ranges for Indic scripts (Devanagari: 0900–097F)
_DEVANAGARI_RANGE = (0x0900, 0x097F)
_ZWNJ = "\u200c"  # Zero-width non-joiner
_ZWJ = "\u200d"  # Zero-width joiner


def _is_devanagari_char(char: str) -> bool:
    """Return True if character falls within the Devanagari Unicode block."""
    if not char:
        return False
    cp = ord(char[0])
    return _DEVANAGARI_RANGE[0] <= cp <= _DEVANAGARI_RANGE[1]


def normalize_text(
    text: str,
    *,
    lowercase: bool = False,
    normalize_unicode: bool = True,
    preserve_indic_joiners: bool = True,
) -> str:
    """Conservatively normalize raw OCR text.

    Rules applied:
    1. Unicode Normalization: Form NFKC to canonicalize composed characters,
       ligatures, and full-width numbers/punctuation.
    2. Whitespace Normalization: Collapse tabs, newlines, non-breaking spaces,
       and multiple spaces into single standard ASCII spaces. Strip leading/trailing.
    3. Invisible Character Sanitization: Remove zero-width spaces (\\u200b) and BOM (\\ufeff),
       while strictly preserving ZWJ (\\u200d) and ZWNJ (\\u200c) when adjacent to
       Devanagari characters (vital for Hindi conjunct consonants like क्या, पत्ता).
    4. Optional Lowercase: Off by default to avoid altering acronyms, proper nouns,
       or casing-sensitive identifiers.

    Args:
        text: Raw text string from OCR engine.
        lowercase: Whether to convert text to lowercase (default False).
        normalize_unicode: Whether to apply unicodedata.normalize('NFKC') (default True).
        preserve_indic_joiners: Whether to retain ZWJ/ZWNJ in Indic words (default True).

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    s = text

    # 1. Unicode normalization (NFKC)
    if normalize_unicode:
        s = unicodedata.normalize("NFKC", s)

    # 2. Invisible character filtering
    # Always strip BOM and zero-width space
    s = s.replace("\ufeff", "").replace("\u200b", "")

    if not preserve_indic_joiners:
        s = s.replace(_ZWNJ, "").replace(_ZWJ, "")
    else:
        # Sanitize ZWJ/ZWNJ only if isolated or surrounded by non-Indic characters
        chars = list(s)
        filtered_chars: list[str] = []
        n = len(chars)
        for i, ch in enumerate(chars):
            if ch in (_ZWJ, _ZWNJ):
                has_prev_indic = i > 0 and _is_devanagari_char(chars[i - 1])
                has_next_indic = i + 1 < n and _is_devanagari_char(chars[i + 1])
                if has_prev_indic or has_next_indic:
                    filtered_chars.append(ch)
            else:
                filtered_chars.append(ch)
        s = "".join(filtered_chars)

    # 3. Collapse whitespace and strip
    s = re.sub(r"[\s\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+", " ", s)
    s = s.strip()

    # 4. Casing
    if lowercase:
        s = s.lower()

    return s
