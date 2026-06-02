"""
Diagnostic: spot-check naturalness of suffix-substitution augmented pairs.

Re-runs augment_by_substitution on a sample of polite training pairs, then
asks Claude Haiku to rate whether each synthetic sentence sounds natural.
Surfaces systematic failure patterns (e.g. compound verb constructions that
always produce awkward output) so they can be fixed in augment.py.

Usage:
    python -m backend.data_pipeline.check_naturalness --data data/train.tsv --sample 100
"""

import argparse
import csv
import json
import logging
import os
import random
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_TOOL = {
    "name": "rate_naturalness",
    "description": "Rate whether a Korean sentence sounds natural to a native speaker.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ratings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "natural": {"type": "boolean"},
                        "issue": {
                            "type": ["string", "null"],
                            "description": "Brief description of the naturalness issue, or null if natural.",
                        },
                    },
                    "required": ["index", "natural", "issue"],
                },
            }
        },
        "required": ["ratings"],
    },
}


def load_polite_pairs(data_path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(data_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3 and row[2] == "polite":
                pairs.append((row[0], row[1]))
    return pairs


def check_naturalness(data_path: str, sample_per_class: int) -> None:
    from dotenv import load_dotenv
    import anthropic
    from .augment import augment_by_substitution

    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    polite_pairs = load_polite_pairs(data_path)
    random.shuffle(polite_pairs)
    sample_polite = polite_pairs[: sample_per_class * 5]  # oversample to get enough augmented output

    labeled_polite = [(en, ko, "polite") for en, ko in sample_polite]
    augmented = augment_by_substitution(labeled_polite)

    formal_aug = [(en, ko) for en, ko, label in augmented if label == "formal"]
    casual_aug = [(en, ko) for en, ko, label in augmented if label == "casual"]

    random.shuffle(formal_aug)
    random.shuffle(casual_aug)
    formal_sample = formal_aug[:sample_per_class]
    casual_sample = casual_aug[:sample_per_class]

    logger.info(f"Checking {len(formal_sample)} formal and {len(casual_sample)} casual augmented pairs")

    client = anthropic.Anthropic(api_key=api_key)

    def rate_batch(pairs: list[tuple[str, str]], register: str) -> list[dict]:
        batch_size = 20
        all_ratings = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            numbered = "\n".join(
                f"{j + 1}. EN: {en}\n   KO: {ko}" for j, (en, ko) in enumerate(batch)
            )
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1024,
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": "rate_naturalness"},
                messages=[{
                    "role": "user",
                    "content": (
                        f"You are a native Korean speaker evaluating {register} register translations. "
                        "For each English→Korean pair below, decide if the Korean sentence sounds natural "
                        "to a native speaker. Focus on whether the sentence-final ending and overall phrasing "
                        "feel idiomatic — not just grammatically correct.\n\n"
                        + numbered
                    ),
                }],
                timeout=60.0,
            )
            tool_block = next((b for b in response.content if b.type == "tool_use"), None)
            if tool_block:
                ratings = tool_block.input.get("ratings", [])
                for r in ratings:
                    idx = r["index"] - 1
                    if 0 <= idx < len(batch):
                        en, ko = batch[idx]
                        all_ratings.append({
                            "register": register,
                            "en": en,
                            "ko": ko,
                            "natural": r["natural"],
                            "issue": r["issue"],
                        })
        return all_ratings

    formal_ratings = rate_batch(formal_sample, "formal")
    casual_ratings = rate_batch(casual_sample, "casual")
    all_ratings = formal_ratings + casual_ratings

    def print_report(ratings: list[dict], register: str) -> None:
        total = len(ratings)
        if total == 0:
            print(f"\n{register.upper()}: no ratings returned")
            return
        natural_count = sum(1 for r in ratings if r["natural"])
        flagged = [r for r in ratings if not r["natural"]]
        print(f"\n{'=' * 60}")
        print(f"{register.upper()} — {natural_count}/{total} natural ({natural_count / total * 100:.1f}%)")
        if flagged:
            issues = [r["issue"] for r in flagged if r["issue"]]
            issue_counts = Counter(issues)
            print(f"\nTop issues:")
            for issue, count in issue_counts.most_common(5):
                print(f"  [{count}x] {issue}")
            print(f"\nFlagged examples (up to 10):")
            for r in flagged[:10]:
                print(f"  EN: {r['en']}")
                print(f"  KO: {r['ko']}")
                print(f"  Issue: {r['issue']}")
                print()

    print_report(formal_ratings, "formal")
    print_report(casual_ratings, "casual")

    output_path = Path("data") / "check_naturalness_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_ratings, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.tsv", help="Path to train.tsv")
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        metavar="N",
        help="Number of augmented pairs to check per formality class (default: 100)",
    )
    args = parser.parse_args()
    check_naturalness(args.data, args.sample)
