"""
Stage 2–3: Morphological analysis and formality labeling.

Uses KoNLPy/Mecab to extract the sentence-final EF (Eojeol Final) morpheme
and maps it to a formality token via suffix pattern rules.
"""

from konlpy.tag import Mecab

_mecab = None

# Suffix → formality token mapping
SUFFIX_RULES: list[tuple[tuple[str, ...], str]] = [
    (("-습니다", "-ㅂ니다", "-습니까", "-시오", "-십시오"), "formal"),
    (("-아요", "-어요", "-여요"), "polite"),
    (("-아", "-어", "-냐", "-지", "-구나"), "casual"),
]


def get_mecab() -> Mecab:
    global _mecab
    if _mecab is None:
        _mecab = Mecab()
    return _mecab


def extract_ef_morpheme(sentence: str) -> str | None:
    """Extract the sentence-final EF morpheme from a Korean sentence."""
    tags = get_mecab().pos(sentence)
    ef_morphemes = [morph for morph, tag in tags if tag == "EF"]
    return ef_morphemes[-1] if ef_morphemes else None


def label_formality(ef_morpheme: str) -> str | None:
    """Map an EF morpheme to a formality label. Returns None if ambiguous."""
    for suffixes, label in SUFFIX_RULES:
        if any(ef_morpheme.endswith(s) for s in suffixes):
            return label
    return None


def label_sentence(ko_sentence: str) -> str | None:
    """
    Label a Korean sentence with its formality token.
    Returns None if no EF morpheme is found or the ending is ambiguous.
    """
    ef = extract_ef_morpheme(ko_sentence)
    if ef is None:
        return None
    return label_formality(ef)
