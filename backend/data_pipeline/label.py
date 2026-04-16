"""
Stage 2–3: Morphological analysis and formality labeling.

Uses KoNLPy/Mecab to extract the sentence-final EF (Eojeol Final) morpheme
and maps it to a formality token via suffix pattern rules.

Falls back to raw-text suffix matching for cases where Mecab mislabels a
sentence-final ending (e.g. 가요 → JX instead of EF due to lexical ambiguity).
"""

import re
from konlpy.tag import Mecab

_mecab = None

# Suffix → formality token mapping (ordered most-specific first)
SUFFIX_RULES: list[tuple[tuple[str, ...], str]] = [
    (("-습니다", "-ㅂ니다", "-습니까", "-ㅂ니까", "-시오", "-십시오"), "formal"),
    (("-아요", "-어요", "-여요", "-죠", "-네요", "-군요", "-나요", "-래요", "-대요"), "polite"),
    (("-아", "-어", "-냐", "-지", "-구나", "-네", "-군", "-야", "-을래", "-ㄹ래"), "casual"),
]

# Fallback: raw-text regex for when Mecab tags -요 as JX instead of EF
_POLITE_FALLBACK = re.compile(r"요[.!?~\s]*$")


def get_mecab() -> Mecab:
    global _mecab
    if _mecab is None:
        _mecab = Mecab()
    return _mecab


def extract_ef_morpheme(sentence: str) -> str | None:
    """Extract the sentence-final EF morpheme from a Korean sentence."""
    tags = get_mecab().pos(sentence)
    ef_morphemes = [morph for morph, tag in tags if tag == "EF" or tag.endswith("+EF")]
    return ef_morphemes[-1] if ef_morphemes else None


def label_formality(ef_morpheme: str) -> str | None:
    """Map an EF morpheme to a formality label. Returns None if ambiguous."""
    for suffixes, label in SUFFIX_RULES:
        if any(ef_morpheme.endswith(s.lstrip("-")) for s in suffixes):
            return label
    return None


def label_sentence(ko_sentence: str) -> str | None:
    """
    Label a Korean sentence with its formality token.

    Primary path: Mecab EF morpheme extraction + suffix matching.
    Fallback: if no EF is found but the sentence ends in -요, label as polite
    (handles Mecab lexical ambiguity, e.g. 가요 parsed as 가/JKS + 요/JX).

    Returns None if the ending is ambiguous or unrecognized.
    """
    ef = extract_ef_morpheme(ko_sentence)
    if ef is not None:
        return label_formality(ef)

    # Fallback for -요 sentences misparse by Mecab
    if _POLITE_FALLBACK.search(ko_sentence):
        return "polite"

    return None
