"""
Stage 5: Class balancing via synthetic augmentation.

Natural corpora skew heavily toward polite register (해요체 dominates subtitle
data). Two strategies balance the class distribution:

  1. Suffix substitution — algorithmically replace terminal morphemes on
     polite-labeled examples to generate formal and casual variants.
     Semantic content is preserved; only the speech level morphology changes.

  2. LLM-assisted generation — prompt a language model to generate additional
     EN–KR pairs in underrepresented registers, verified against the
     morphological labeling rules before inclusion.
"""

# Polite → Formal substitution pairs (terminal morpheme swap)
POLITE_TO_FORMAL: dict[str, str] = {
    "아요": "습니다",
    "어요": "습니다",
    "여요": "습니다",
}

# Polite → Casual substitution pairs
POLITE_TO_CASUAL: dict[str, str] = {
    "아요": "아",
    "어요": "어",
    "여요": "여",
}


def substitute_suffix(ko_sentence: str, substitutions: dict[str, str]) -> str | None:
    """
    Replace the terminal morpheme in a Korean sentence using a substitution map.
    Returns None if no substitution applies.
    """
    for source, target in substitutions.items():
        if ko_sentence.endswith(source):
            return ko_sentence[: -len(source)] + target
    return None


def augment_by_substitution(
    pairs: list[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    """
    Generate formal and casual variants from polite-labeled examples
    via suffix substitution.
    """
    augmented = []
    for en, ko, formality in pairs:
        if formality != "polite":
            continue

        formal_ko = substitute_suffix(ko, POLITE_TO_FORMAL)
        if formal_ko:
            augmented.append((en, formal_ko, "formal"))

        casual_ko = substitute_suffix(ko, POLITE_TO_CASUAL)
        if casual_ko:
            augmented.append((en, casual_ko, "casual"))

    return augmented


def augment_by_llm(target_label: str, target_count: int) -> list[tuple[str, str, str]]:
    """
    Generate additional EN–KR pairs in underrepresented registers using
    LLM-assisted generation. Generated pairs are verified against the
    morphological labeling pipeline before inclusion.
    """
    raise NotImplementedError
