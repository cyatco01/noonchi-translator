"""
Stage 4: Confidence filtering.

Removes low-quality pairs before training. Approximately 40–60% of raw pairs
are discarded to ensure label reliability.

Filters applied:
  - No EF tag found (fragment or incomplete sentence)
  - Ambiguous or unknown ending (not matched by suffix rules)
  - Length outliers (below min or above per-class max)
  - Misaligned pairs (sentence-length ratio heuristic)
"""

from .label import get_mecab

# Per-class bounds derived from morpheme-count distribution of train.tsv (338K rows).
# EN bounds: word count at p1/p99+headroom. KO bounds: Mecab morpheme count at p1/p99+headroom.
# Formal p99: EN=21 words, KO=27 morphemes. Polite/casual p99: EN=24 words, KO=30-31 morphemes.
LENGTH_BOUNDS: dict[str, dict[str, tuple[int, int]]] = {
    "formal": {"en": (3, 25), "ko": (3, 30)},
    "polite": {"en": (3, 28), "ko": (3, 35)},
    "casual": {"en": (3, 28), "ko": (3, 35)},
}
MAX_LENGTH_RATIO = 4.0


def is_valid_pair(en: str, ko: str, formality: str | None) -> bool:
    """Return True if the pair passes all quality filters."""
    if formality is None:
        return False

    bounds = LENGTH_BOUNDS.get(formality)
    if bounds is None:
        return False

    if not en.strip() or not ko.strip():
        return False

    en_len = len(en.split())
    ko_len = len(get_mecab().morphs(ko))

    if not (bounds["en"][0] <= en_len <= bounds["en"][1]):
        return False
    if not (bounds["ko"][0] <= ko_len <= bounds["ko"][1]):
        return False

    # Length ratio uses morpheme count for KO, word count for EN — both already computed above.
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
