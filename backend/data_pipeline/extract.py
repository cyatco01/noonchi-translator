"""
Stage 1: Corpus extraction.

Downloads and aligns English–Korean sentence pairs from:
  - OPUS OpenSubtitles: high-volume, colloquial register
  - Tatoeba EN–KR: curated, broader register distribution
"""


def load_opus_opensubtitles(data_dir: str) -> list[tuple[str, str]]:
    """Load EN–KR pairs from OPUS OpenSubtitles."""
    raise NotImplementedError


def load_tatoeba(data_dir: str) -> list[tuple[str, str]]:
    """Load EN–KR pairs from Tatoeba."""
    raise NotImplementedError


def load_corpus(data_dir: str) -> list[tuple[str, str]]:
    """Load and combine all corpora. Returns list of (en, ko) pairs."""
    pairs = []
    pairs.extend(load_opus_opensubtitles(data_dir))
    pairs.extend(load_tatoeba(data_dir))
    return pairs
