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
import re
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
# '여요' is excluded: stripping it leaves bare '여', which is not a valid casual
# sentence-final ending. Correct casual copula form would be '야', but that
# requires knowing the preceding syllable — safer to skip and let LLM augmentation
# handle 이에요/이여요 verbs.
POLITE_TO_CASUAL: dict[str, str] = {
    "아요": "아",
    "어요": "어",
}

_JONGSEONG_RIEUL = 8   # ㄹ 종성 index in Unicode Hangul syllable encoding
_JONGSEONG_BIEUP = 17  # ㅂ 종성 index — replaces ㄹ in ㄹ-drop rule

# ㄷ-irregular stems: extracted stem (after stripping 어요) → correct formal stem.
# These verbs conjugate ㄷ→ㄹ before vowels; the formal form restores ㄷ.
_DT_IRREGULAR_STEMS: dict[str, str] = {
    "들": "듣",  # 듣다 — listen
    "걸": "걷",  # 걷다 — walk
    "물": "묻",  # 묻다 — ask/inquire
    "실": "싣",  # 싣다 — load
}

# ㅅ-irregular stems: extracted stem (after stripping 아요/어요) → correct formal stem.
# These verbs drop ㅅ before vowels; the formal form restores ㅅ.
# Only includes cases where collision risk is low:
#   나 (낫다) — 나아요 almost never comes from 나다 (which contracts to 나요)
#   지 (짓다) — 지어요 almost never comes from 지다 (which contracts to 져요)
#   부 (붓다) — 부어요 has no common competing verb
# Excluded: 이→잇 (이어요 collides with copula 이다), 그→긋 (긋다 is rare)
_ST_IRREGULAR_STEMS: dict[str, str] = {
    "나": "낫",  # 낫다 — get better / recover
    "지": "짓",  # 짓다 — build / compose / commit
    "부": "붓",  # 붓다 — pour / swell
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


def _to_formal(stem: str) -> str:
    """
    Convert a stripped polite stem to its 합쇼체 formal form.

    Handles two irregular patterns before falling back to regular 습니다:
    - ㄷ-irregular: lookup corrects the stem (들→듣) then appends 습니다
    - ㄹ-irregular: ㄹ종성 drops before ㅂ via Hangul Unicode arithmetic
      e.g. 알→압니다, 살→삽니다, 만들→만듭니다
    """
    last = stem[-1] if stem else ""
    # Match ㄷ-irregular against the full last eojeol (word), not just the last
    # character — 들 at the end of 만들 is ㄹ-regular (만들다), but 들 as a
    # standalone word is ㄷ-irregular (듣다).
    last_word = stem.rsplit(None, 1)[-1] if stem.strip() else stem

    # ㄷ-irregular: only when the entire last word matches the irregular stem
    if last_word in _DT_IRREGULAR_STEMS:
        return stem[:-len(last_word)] + _DT_IRREGULAR_STEMS[last_word] + "습니다"

    # ㅅ-irregular: same last-word matching; ㅅ is restored before 습니다
    if last_word in _ST_IRREGULAR_STEMS:
        return stem[:-len(last_word)] + _ST_IRREGULAR_STEMS[last_word] + "습니다"

    # ㄹ-irregular: swap ㄹ종성 (index 8) for ㅂ종성 (index 17) in the syllable block
    code = ord(last) - 0xAC00
    if 0 <= code <= 11171 and code % 28 == _JONGSEONG_RIEUL:
        new_last = chr(ord(last) - _JONGSEONG_RIEUL + _JONGSEONG_BIEUP)
        return stem[:-1] + new_last + "니다"

    return stem + "습니다"


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

    Trailing punctuation (? . !) is stripped before suffix matching and
    reattached to the output so that punctuated sentences are not silently
    skipped (e.g. '드세요?' → strips '?' → matches '어요' → '드십시오?').
    """
    augmented = []
    for en, ko, formality in pairs:
        if formality != "polite":
            continue

        punct_match = _TRAILING_PUNCT.search(ko)
        punct = punct_match.group(1) if punct_match else ""
        ko_base = ko[: punct_match.start()] if punct_match else ko

        for source in POLITE_TO_FORMAL:
            if ko_base.endswith(source):
                augmented.append((en, _to_formal(ko_base[: -len(source)]) + punct, "formal"))
                break

        casual_ko = substitute_suffix(ko_base, POLITE_TO_CASUAL)
        if casual_ko:
            augmented.append((en, casual_ko + punct, "casual"))

    return augmented


_LIST_PREFIX = re.compile(r"^\d+[.)]\s+")
_MIN_EN_WORDS = 3
_TRAILING_PUNCT = re.compile(r"([.?!~]+)$")


def augment_by_llm(
    target_label: Formality,
    target_count: int,
    batch_size: int = 20,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 200,
    corpus_ko: set[str] | None = None,
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
        corpus_ko:         Optional set of Korean sentences already in the original corpus.
                           Synthetic sentences matching any entry here are skipped, preventing
                           exact-duplicate KO sentences with different English sources.

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

    seen_ko: set[str] = {ko for _, ko, _ in collected}
    if corpus_ko:
        seen_ko |= corpus_ko

    client = anthropic.Anthropic(api_key=api_key)
    description = _FORMALITY_DESCRIPTIONS[target_label]

    # Cache the system prompt — it's identical on every call, so Haiku reuses it
    # after the first request, cutting input token cost on subsequent calls.
    system_prompt = [
        {
            "type": "text",
            "text": (
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
                "- Every sentence must be unique — do not repeat sentences from previous batches.\n"
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    consecutive_parse_failures = 0
    max_parse_failures = 5
    consecutive_all_dupes = 0
    max_consecutive_all_dupes = 10
    last_checkpoint_at = len(collected)
    # Track recent acceptance rate to right-size each request — avoids paying
    # for pairs that get discarded as duplicates.
    recent_verified: list[int] = []
    recent_unique: list[int] = []

    try:
        while len(collected) < target_count:
            # Adaptive batch size: request just enough to yield ~batch_size unique pairs.
            # Uses rolling acceptance rate from last 5 batches; clamp between 1x and 4x.
            if recent_verified and sum(recent_verified):
                acceptance_rate = sum(recent_unique) / sum(recent_verified)
                actual_batch = min(
                    max(batch_size, round(batch_size / max(acceptance_rate, 0.25))),
                    batch_size * 4,
                )
            else:
                actual_batch = batch_size

            # Scale max_tokens to actual request size (~60 tokens per pair is a safe upper bound)
            max_tokens = min(actual_batch * 60, 4096)

            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=max_tokens,
                system=system_prompt,  # type: ignore[arg-type]
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate exactly {actual_batch} English–Korean sentence pairs in "
                        f"{target_label} register ({description}).\n\n"
                        "Each Korean sentence must end with the appropriate speech-level morpheme. "
                        "Every sentence must be unique and cover a different topic or situation."
                    ),
                }],
                timeout=60.0,
            )

            text = next((b.text for b in response.content if b.type == "text"), "")
            batch_verified = _parse_and_verify(text, target_label, label_sentence)

            if not batch_verified:
                consecutive_parse_failures += 1
                logger.warning(
                    f"[{target_label}] batch produced 0 verified pairs "
                    f"({consecutive_parse_failures}/{max_parse_failures} consecutive failures)"
                )
                if consecutive_parse_failures >= max_parse_failures:
                    logger.error(f"[{target_label}] too many parse failures — stopping early")
                    break
                continue

            consecutive_parse_failures = 0

            # Deduplicate against all previously seen Korean sentences
            unique_batch = [(en, ko, lbl) for en, ko, lbl in batch_verified if ko not in seen_ko]
            for _, ko, _ in unique_batch:
                seen_ko.add(ko)

            # Update rolling acceptance rate (keep last 5 batches)
            recent_verified.append(len(batch_verified))
            recent_unique.append(len(unique_batch))
            if len(recent_verified) > 5:
                recent_verified.pop(0)
                recent_unique.pop(0)

            dupe_count = len(batch_verified) - len(unique_batch)
            if dupe_count:
                logger.debug(f"[{target_label}] filtered {dupe_count} duplicate(s) from batch")

            if not unique_batch:
                consecutive_all_dupes += 1
                if consecutive_all_dupes >= max_consecutive_all_dupes:
                    logger.error(
                        f"[{target_label}] {max_consecutive_all_dupes} consecutive batches "
                        "yielded only duplicates — stopping early"
                    )
                    break
            else:
                consecutive_all_dupes = 0

            collected.extend(unique_batch)
            logger.info(f"[{target_label}] {len(collected):,}/{target_count:,} unique verified pairs")

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


def augment_by_triplets(
    pairs: list[tuple[str, str, str]],
    target_triplets: int,
    batch_size: int = 10,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 100,
) -> list[tuple[str, str, str]]:
    """
    Generate contrastive triplets: the same English sentence paired with all three
    formality-level Korean translations.

    Triplets give the model an explicit signal that the conditioning token controls
    register — not sentence content. Only complete triplets (all 3 formality levels
    verified) are added; partial matches are discarded.

    Args:
        pairs:            Existing (en, ko, formality) training pairs used as the
                          English candidate pool.
        target_triplets:  Number of complete triplets to generate (each produces
                          3 training rows).
        batch_size:       English sentences per API call.
        checkpoint_dir:   Directory for checkpoint file (default: data/checkpoints/).
        checkpoint_every: Save after every N complete triplets.

    Returns:
        List of (en, ko, formality) tuples — 3 per accepted triplet.
    """
    import json
    import random
    import anthropic
    from dotenv import load_dotenv

    from .label import label_sentence

    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Cannot run triplet augmentation.")

    # Checkpoint setup
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (
        Path(__file__).parent.parent.parent / "data" / "checkpoints"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "triplets.json"

    collected: list[tuple[str, str, str]] = []
    collected_triplet_count = 0
    if ckpt_file.exists():
        with open(ckpt_file, encoding="utf-8") as f:
            saved = json.load(f)
        collected = [tuple(p) for p in saved]  # type: ignore[misc]
        collected_triplet_count = len(collected) // 3
        logger.info(
            f"Resuming triplets from checkpoint: "
            f"{collected_triplet_count}/{target_triplets} triplets already collected"
        )
    else:
        logger.info(f"Starting triplet augmentation: target {target_triplets:,} triplets")

    if collected_triplet_count >= target_triplets:
        logger.info("Triplet checkpoint already complete — skipping API calls")
        return collected[: target_triplets * 3]

    # Build candidate pool: short English sentences not already in all 3 formality classes.
    # Sentences already covered by substitution augmentation (present as formal, polite, and casual)
    # would produce near-duplicate triplets — exclude them so triplets add genuine contrastive signal.
    seen_en: set[str] = {en for en, _, _ in collected}
    en_formalities: dict[str, set[str]] = {}
    for en, _, formality in pairs:
        if en not in en_formalities:
            en_formalities[en] = set()
        en_formalities[en].add(formality)
    fully_covered = {en for en, fmts in en_formalities.items() if len(fmts) >= 3}

    candidate_pool = [
        en for en in {en for en, _, _ in pairs if len(en.split()) <= 10}
        if en not in seen_en and en not in fully_covered
    ]
    random.shuffle(candidate_pool)
    logger.info(
        f"Triplet candidate pool: {len(candidate_pool):,} unique short English sentences "
        f"({len(fully_covered):,} already in all 3 formalities excluded)"
    )

    system_prompt = [
        {
            "type": "text",
            "text": (
                "You are a Korean linguistics expert. For each English sentence you receive, "
                "produce all three Korean formality-level translations.\n\n"
                "OUTPUT FORMAT — respond with ONLY a numbered list, one row per sentence:\n"
                "1. English | formal Korean | polite Korean | casual Korean\n"
                "2. English | formal Korean | polite Korean | casual Korean\n"
                "...\n\n"
                "SPEECH LEVEL RULES:\n"
                "- Formal (하십시오체): must end with -습니다, -ㅂ니다, or -습니까\n"
                "- Polite (해요체): must end with -아요, -어요, or -여요\n"
                "- Casual (해체): must end with -아, -어, -냐, or -지\n\n"
                "RULES:\n"
                "- Use the pipe character | to separate the four fields.\n"
                "- No romanization, no parenthetical notes, no extra commentary.\n"
                "- The three Korean translations must differ only in speech level, not meaning.\n"
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    client = anthropic.Anthropic(api_key=api_key)
    candidate_idx = 0
    last_checkpoint_at = collected_triplet_count

    try:
        while collected_triplet_count < target_triplets:
            # Pull the next batch of candidate English sentences
            batch_en = candidate_pool[candidate_idx: candidate_idx + batch_size]
            if not batch_en:
                logger.warning("Candidate pool exhausted before reaching target — stopping")
                break
            candidate_idx += batch_size

            numbered = "\n".join(f"{i + 1}. {en}" for i, en in enumerate(batch_en))
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=min(batch_size * 120, 4096),
                system=system_prompt,  # type: ignore[arg-type]
                messages=[{
                    "role": "user",
                    "content": (
                        f"Translate each of the following {len(batch_en)} English sentences "
                        "into all three Korean formality levels:\n\n"
                        + numbered
                    ),
                }],
                timeout=60.0,
            )

            text = next((b.text for b in response.content if b.type == "text"), "")
            new_triplets = _parse_and_verify_triplets(text, label_sentence)
            assert len(new_triplets) % 3 == 0, f"triplet verifier returned {len(new_triplets)} rows — not a multiple of 3"

            collected.extend(new_triplets)
            collected_triplet_count += len(new_triplets) // 3
            logger.info(
                f"[triplets] {collected_triplet_count:,}/{target_triplets:,} complete triplets"
            )

            if collected_triplet_count - last_checkpoint_at >= checkpoint_every:
                with open(ckpt_file, "w", encoding="utf-8") as f:
                    json.dump(collected, f, ensure_ascii=False)
                last_checkpoint_at = collected_triplet_count
                logger.info(f"[triplets] checkpoint saved ({collected_triplet_count:,} triplets)")

    except KeyboardInterrupt:
        logger.warning("[triplets] interrupted — saving checkpoint and exiting")
    finally:
        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False)
        logger.info(f"[triplets] checkpoint saved: {collected_triplet_count:,} triplets in {ckpt_file}")

    result = collected[: target_triplets * 3]
    logger.info(f"Triplet augmentation complete: {len(result) // 3:,} triplets ({len(result):,} rows)")
    return result


def _parse_and_verify_triplets(
    text: str,
    label_fn,
) -> list[tuple[str, str, str]]:
    """
    Parse pipe-delimited triplet output: "English | formal KO | polite KO | casual KO".
    Only accepts complete triplets where all three Korean sentences pass morphological
    verification AND class-aware length bounds. Returns a flat list of (en, ko, formality)
    tuples (3 per triplet).
    """
    from .filter import is_valid_pair

    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _LIST_PREFIX.sub("", line)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        en, formal_ko, polite_ko, casual_ko = parts
        if not all([en, formal_ko, polite_ko, casual_ko]):
            continue
        if len(en.split()) < _MIN_EN_WORDS:
            continue
        if (
            label_fn(formal_ko) == "formal"
            and label_fn(polite_ko) == "polite"
            and label_fn(casual_ko) == "casual"
            and is_valid_pair(en, formal_ko, "formal")
            and is_valid_pair(en, polite_ko, "polite")
            and is_valid_pair(en, casual_ko, "casual")
        ):
            result.extend([
                (en, formal_ko, "formal"),
                (en, polite_ko, "polite"),
                (en, casual_ko, "casual"),
            ])
    return result


def _parse_and_verify(
    text: str,
    expected_label: str,
    label_fn,
) -> list[tuple[str, str, str]]:
    """
    Parse numbered list output from the LLM and verify each Korean sentence
    with the morphological labeler and the same per-class length bounds applied
    to corpus pairs. Returns only verified pairs.
    """
    from .filter import is_valid_pair

    verified = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip leading list marker anchored at start: "1. ", "2) ", etc.
        line = _LIST_PREFIX.sub("", line)

        if "|" not in line:
            continue

        parts = line.split("|", 1)
        if len(parts) != 2:
            continue

        en, ko = parts[0].strip(), parts[1].strip()
        if not en or not ko:
            continue

        # Reject pairs that are too short to be useful training examples
        if len(en.split()) < _MIN_EN_WORDS:
            continue

        label = label_fn(ko)
        if label == expected_label and is_valid_pair(en, ko, label):
            verified.append((en, ko, label))

    return verified
