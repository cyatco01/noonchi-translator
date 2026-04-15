"""
Stage 4: Confidence filtering.

Removes low-quality pairs before training. Approximately 40–60% of raw pairs
are discarded to ensure label reliability.

Filters applied:
  - No EF tag found (fragment or incomplete sentence)
  - Ambiguous or unknown ending (not matched by suffix rules)
  - Length outliers (< 3 tokens or > 150 tokens)
  - Misaligned pairs (sentence-length ratio heuristic)
"""

MIN_TOKENS = 3
MAX_TOKENS = 150
MAX_LENGTH_RATIO = 4.0


def is_valid_pair(en: str, ko: str, formality: str | None) -> bool:
    """Return True if the pair passes all quality filters."""
    if formality is None:
        return False

    en_len = len(en.split())
    ko_len = len(ko.split())

    if en_len < MIN_TOKENS or en_len > MAX_TOKENS:
        return False
    if ko_len < MIN_TOKENS or ko_len > MAX_TOKENS:
        return False

    ratio = max(en_len, ko_len) / max(min(en_len, ko_len), 1)
    if ratio > MAX_LENGTH_RATIO:
        return False

    return True


def filter_pairs(
    pairs: list[tuple[str, str, str | None]]
) -> list[tuple[str, str, str]]:
    """Filter labeled (en, ko, formality) triples. Returns only valid pairs."""
    return [
        (en, ko, formality)
        for en, ko, formality in pairs
        if is_valid_pair(en, ko, formality)
    ]
