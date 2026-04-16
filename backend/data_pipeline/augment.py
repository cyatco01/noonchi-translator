"""
Stage 5: Class balancing via synthetic augmentation.

Natural corpora skew heavily toward polite register (해요체 dominates subtitle
data). Two strategies balance the class distribution:

  1. Suffix substitution — algorithmically replace terminal morphemes on
     polite-labeled examples to generate formal and casual variants.
     Semantic content is preserved; only the speech level morphology changes.

  2. LLM-assisted generation — prompt Claude to generate additional EN–KR
     pairs in underrepresented registers. Generated pairs are morphologically
     verified before inclusion.
"""

import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

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

Formality = Literal["formal", "polite", "casual"]

_FORMALITY_DESCRIPTIONS = {
    "formal": (
        "하십시오체 (formal/honorific). "
        "Sentence-final endings: -습니다, -ㅂ니다, -습니까, -십시오. "
        "Used in business presentations, news broadcasts, public announcements."
    ),
    "polite": (
        "해요체 (polite/standard). "
        "Sentence-final endings: -아요, -어요, -여요, -죠. "
        "Used in everyday conversation with strangers, colleagues, service workers."
    ),
    "casual": (
        "해체 (casual/informal). "
        "Sentence-final endings: -아, -어, -냐, -지, -야. "
        "Used with close friends, family, and same-age peers."
    ),
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


def augment_by_llm(
    target_label: Formality,
    target_count: int,
    batch_size: int = 20,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 200,
) -> list[tuple[str, str, str]]:
    """
    Generate EN–KR pairs in an underrepresented register using Claude.

    Prompts the model in batches of `batch_size` pairs per request, verifies
    each Korean sentence morphologically, and saves progress to a checkpoint
    file every `checkpoint_every` pairs so the run can be resumed after
    interruption.

    Uses claude-haiku-4-5 for cost efficiency — bulk training data generation
    is a legitimate high-volume, lower-stakes task where speed/cost dominates.

    Args:
        target_label:      Formality level ("formal", "polite", "casual").
        target_count:      Number of verified pairs to produce.
        batch_size:        Pairs to request per API call.
        checkpoint_dir:    Directory for checkpoint files (default: data/checkpoints/).
        checkpoint_every:  Save progress after every N verified pairs.

    Returns:
        List of (en, ko, formality) tuples, all morphologically verified.
    """
    import json
    import anthropic
    from dotenv import load_dotenv

    from .label import label_sentence

    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Cannot run LLM augmentation."
        )

    # Checkpoint setup
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (
        Path(__file__).parent.parent.parent / "data" / "checkpoints"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"llm_augment_{target_label}.json"

    # Resume from checkpoint if it exists
    collected: list[tuple[str, str, str]] = []
    if ckpt_file.exists():
        with open(ckpt_file, encoding="utf-8") as f:
            saved = json.load(f)
        collected = [tuple(p) for p in saved]  # type: ignore[misc]
        logger.info(
            f"Resuming [{target_label}] from checkpoint: "
            f"{len(collected)}/{target_count} pairs already collected"
        )
    else:
        logger.info(f"Starting LLM augmentation [{target_label}]: target {target_count:,} pairs")

    if len(collected) >= target_count:
        logger.info(f"[{target_label}] checkpoint already complete — skipping API calls")
        return collected[:target_count]

    client = anthropic.Anthropic(api_key=api_key)
    description = _FORMALITY_DESCRIPTIONS[target_label]

    system_prompt = (
        "You are a Korean linguistics expert generating parallel English–Korean sentence "
        "pairs for a machine translation training corpus.\n\n"
        "OUTPUT FORMAT — respond with ONLY a numbered list, one pair per line:\n"
        "1. English sentence | Korean sentence\n"
        "2. English sentence | Korean sentence\n"
        "...\n\n"
        "RULES:\n"
        "- Each Korean sentence MUST end with the correct speech-level morpheme.\n"
        "- Vary the topics: everyday life, work, food, travel, emotions, requests.\n"
        "- Keep sentences short to medium length (5–15 words in English).\n"
        "- No romanization, no parenthetical notes, no explanations.\n"
        "- The pipe character | separates English from Korean.\n"
    )

    consecutive_empty = 0
    max_consecutive_empty = 5
    last_checkpoint_at = len(collected)

    try:
        while len(collected) < target_count:
            actual_batch = batch_size

            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=2048,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate exactly {actual_batch} English–Korean sentence pairs in "
                        f"{target_label} register ({description}).\n\n"
                        "Each Korean sentence must end with the appropriate speech-level morpheme."
                    ),
                }],
            )

            text = next((b.text for b in response.content if b.type == "text"), "")
            batch_verified = _parse_and_verify(text, target_label, label_sentence)

            if not batch_verified:
                consecutive_empty += 1
                logger.warning(
                    f"[{target_label}] batch produced 0 verified pairs "
                    f"({consecutive_empty}/{max_consecutive_empty} consecutive empties)"
                )
                if consecutive_empty >= max_consecutive_empty:
                    logger.error(f"[{target_label}] too many empty batches — stopping early")
                    break
                continue

            consecutive_empty = 0
            collected.extend(batch_verified)
            logger.info(f"[{target_label}] {len(collected):,}/{target_count:,} verified pairs")

            # Save checkpoint periodically
            if len(collected) - last_checkpoint_at >= checkpoint_every:
                with open(ckpt_file, "w", encoding="utf-8") as f:
                    json.dump(collected, f, ensure_ascii=False)
                last_checkpoint_at = len(collected)
                logger.info(f"[{target_label}] checkpoint saved ({len(collected):,} pairs)")

    except KeyboardInterrupt:
        logger.warning(f"[{target_label}] interrupted — saving checkpoint and exiting")
    finally:
        # Always save on exit (interrupt or normal completion)
        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False)
        logger.info(f"[{target_label}] checkpoint saved: {len(collected):,} pairs in {ckpt_file}")

    result = collected[:target_count]
    logger.info(f"LLM augmentation complete: {len(result):,} {target_label} pairs")
    return result


def _parse_and_verify(
    text: str,
    expected_label: str,
    label_fn,
) -> list[tuple[str, str, str]]:
    """
    Parse numbered list output from the LLM and verify each Korean sentence
    with the morphological labeler. Returns only verified pairs.
    """
    verified = []
    for line in text.splitlines():
        # Strip leading number + dot/paren: "1. ", "1) ", etc.
        line = line.strip()
        if not line:
            continue
        for sep in (". ", ") ", "- "):
            if line[0].isdigit() and sep in line:
                line = line.split(sep, 1)[1]
                break

        if "|" not in line:
            continue

        parts = line.split("|", 1)
        if len(parts) != 2:
            continue

        en, ko = parts[0].strip(), parts[1].strip()
        if not en or not ko:
            continue

        label = label_fn(ko)
        if label == expected_label:
            verified.append((en, ko, label))

    return verified
