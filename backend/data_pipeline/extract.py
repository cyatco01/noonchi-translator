"""
Stage 1: Corpus extraction.

Loads and aligns English–Korean sentence pairs from:
  - Tatoeba EN–KR: curated, broader register distribution
  - OPUS OpenSubtitles: high-volume, colloquial register

Expected directory layout (files must be downloaded manually):
  data/raw/tatoeba/sentences.csv   — tab-separated: id, lang, text
  data/raw/tatoeba/links.csv       — tab-separated: sentence_id, translation_id
  data/raw/opus/en-ko.tmx (or OpenSubtitles.en-ko.tmx.gz)  — TMX (XML), gzipped or plain
"""

import gzip
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def load_tatoeba(data_dir: str) -> list[tuple[str, str]]:
    """
    Load EN–KR pairs from Tatoeba.

    Reads sentences.csv to build a mapping of sentence ID → (lang, text),
    then walks links.csv to find aligned EN–KR pairs.
    """
    tatoeba_dir = Path(data_dir) / "raw" / "tatoeba"
    links_path = tatoeba_dir / "links.csv"

    # Accept either per-language files (preferred, smaller) or the full sentences.csv
    eng_path = tatoeba_dir / "eng_sentences.tsv"
    kor_path = tatoeba_dir / "kor_sentences.tsv"
    combined_path = tatoeba_dir / "sentences.csv"

    use_split = eng_path.exists() and kor_path.exists()
    use_combined = combined_path.exists()

    if not use_split and not use_combined:
        raise FileNotFoundError(
            f"Tatoeba sentence files not found in {tatoeba_dir}\n"
            "Download per-language files (smaller):\n"
            "  eng_sentences.tsv.bz2 and kor_sentences.tsv.bz2\n"
            "from https://tatoeba.org/en/downloads → 'Download per language'"
        )
    if not links_path.exists():
        raise FileNotFoundError(
            f"Tatoeba links file not found: {links_path}\n"
            "Download links.csv.bz2 from https://tatoeba.org/en/downloads"
        )

    # Step 1: Load sentences into id → text (lang already known from filename)
    sentences: dict[str, tuple[str, str]] = {}
    if use_split:
        for path, lang in [(eng_path, "eng"), (kor_path, "kor")]:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        sid, text = parts[0], parts[-1]
                        sentences[sid] = (lang, text.strip())
    else:
        with open(combined_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    sid, lang, text = parts
                    if lang in ("eng", "kor"):
                        sentences[sid] = (lang, text.strip())

    # Step 2: Build sets of English and Korean sentence IDs for fast lookup
    eng_ids = {sid for sid, (lang, _) in sentences.items() if lang == "eng"}
    kor_ids = {sid for sid, (lang, _) in sentences.items() if lang == "kor"}

    # Step 3: Walk links to find aligned EN–KR pairs
    pairs: list[tuple[str, str]] = []
    with open(links_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                a, b = parts
                if a in eng_ids and b in kor_ids:
                    pairs.append((sentences[a][1], sentences[b][1]))
                elif a in kor_ids and b in eng_ids:
                    pairs.append((sentences[b][1], sentences[a][1]))

    logger.info(f"Tatoeba: loaded {len(pairs):,} EN–KR pairs")
    return pairs


def load_opus_opensubtitles(data_dir: str) -> list[tuple[str, str]]:
    """
    Load EN–KR pairs from OPUS OpenSubtitles (gzipped TMX format).

    Uses iterparse to stream the file rather than loading it all into memory —
    essential for the ~500MB compressed file. elem.clear() frees each parsed
    element immediately to avoid OOM.
    """
    opus_dir = Path(data_dir) / "raw" / "opus"
    tmx_path = next(
        (p for p in [
            opus_dir / "OpenSubtitles.en-ko.tmx.gz",
            opus_dir / "OpenSubtitles.en-ko.tmx",
            opus_dir / "en-ko.tmx.gz",
            opus_dir / "en-ko.tmx",
        ] if p.exists()),
        None,
    )

    if tmx_path is None:
        raise FileNotFoundError(
            f"OPUS OpenSubtitles file not found in {opus_dir}\n"
            "Download OpenSubtitles.en-ko.tmx.gz from https://opus.nlpl.eu/OpenSubtitles "
            "and save to data/raw/opus/"
        )

    pairs: list[tuple[str, str]] = []
    open_fn = gzip.open if str(tmx_path).endswith(".gz") else open

    with open_fn(tmx_path, "rt", encoding="utf-8") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag == "tu":
                en_text = ko_text = None
                for tuv in elem:
                    lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                    seg = tuv.find("seg")
                    if seg is not None and seg.text:
                        if lang.startswith("en"):
                            en_text = seg.text.strip()
                        elif lang.startswith("ko"):
                            ko_text = seg.text.strip()
                if en_text and ko_text:
                    pairs.append((en_text, ko_text))
                elem.clear()  # free memory — critical for large files

    logger.info(f"OPUS OpenSubtitles: loaded {len(pairs):,} EN–KR pairs")
    return pairs


def load_corpus(data_dir: str) -> list[tuple[str, str]]:
    """Load and combine all corpora. Returns list of (en, ko) pairs."""
    pairs: list[tuple[str, str]] = []
    pairs.extend(load_tatoeba(data_dir))
    pairs.extend(load_opus_opensubtitles(data_dir))
    logger.info(f"Total corpus: {len(pairs):,} EN–KR pairs")
    return pairs
