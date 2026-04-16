"""
End-to-end data pipeline orchestration.

Runs all stages and writes the final TSV dataset:
  en  <TAB>  ko  <TAB>  formality

Usage:
    # Basic run (substitution augmentation only):
    python -m backend.data_pipeline.pipeline --data-dir data/ --output data/train.tsv

    # With LLM augmentation — bring each class up to 150,000 pairs:
    python -m backend.data_pipeline.pipeline --data-dir data/ --output data/train.tsv --augment-to 150000
"""

import argparse
import csv
import logging
from pathlib import Path

from .extract import load_corpus
from .label import label_sentence
from .filter import filter_pairs
from .augment import augment_by_substitution, augment_by_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run(data_dir: str, output_path: str, augment_to: int | None = None) -> None:
    print("Stage 1: Extracting corpora...")
    raw_pairs = load_corpus(data_dir)
    print(f"  {len(raw_pairs):,} raw pairs loaded")

    print("Stage 2–3: Morphological analysis and formality labeling...")
    labeled = [(en, ko, label_sentence(ko)) for en, ko in raw_pairs]

    print("Stage 4: Confidence filtering...")
    filtered = filter_pairs(labeled)
    print(f"  {len(filtered):,} pairs retained ({len(labeled) - len(filtered):,} removed)")

    print("Stage 5: Augmenting underrepresented registers...")
    augmented = augment_by_substitution(filtered)
    all_pairs = filtered + augmented
    print(f"  Substitution: {len(augmented):,} synthetic pairs added → {len(all_pairs):,} total")

    if augment_to is not None:
        counts: dict[str, int] = {}
        for _, _, label in all_pairs:
            if label:
                counts[label] = counts.get(label, 0) + 1

        for label in ("formal", "polite", "casual"):
            current = counts.get(label, 0)
            if current < augment_to:
                needed = augment_to - current
                print(f"  LLM augmentation [{label}]: generating {needed:,} pairs (current: {current:,})")
                llm_pairs = augment_by_llm(label, needed)  # type: ignore[arg-type]
                all_pairs = all_pairs + llm_pairs
                print(f"  LLM augmentation [{label}]: {len(llm_pairs):,} verified pairs added")
            else:
                print(f"  LLM augmentation [{label}]: already at {current:,} ≥ {augment_to:,}, skipping")

    print("Stage 6: Writing dataset...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["en", "ko", "formality"])
        writer.writerows(all_pairs)
    print(f"  Dataset written to {output}")

    final_counts: dict[str, int] = {}
    for _, _, label in all_pairs:
        if label:
            final_counts[label] = final_counts.get(label, 0) + 1
    print(f"  Class distribution: {final_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output", default="data/train.tsv")
    parser.add_argument(
        "--augment-to",
        type=int,
        default=None,
        metavar="N",
        help="Use LLM augmentation to bring each class up to N pairs (e.g. 150000)",
    )
    args = parser.parse_args()
    run(args.data_dir, args.output, augment_to=args.augment_to)
