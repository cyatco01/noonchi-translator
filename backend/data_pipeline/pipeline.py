"""
End-to-end data pipeline orchestration.

Runs all six stages and writes the final TSV dataset:
  en  <TAB>  ko  <TAB>  formality

Usage:
    python -m backend.data_pipeline.pipeline --data-dir data/ --output data/train.tsv
"""

import argparse
import csv
from pathlib import Path

from .extract import load_corpus
from .label import label_sentence
from .filter import filter_pairs
from .augment import augment_by_substitution


def run(data_dir: str, output_path: str) -> None:
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
    print(f"  {len(augmented):,} synthetic pairs added → {len(all_pairs):,} total")

    print("Stage 6: Writing dataset...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["en", "ko", "formality"])
        writer.writerows(all_pairs)
    print(f"  Dataset written to {output}")

    counts = {}
    for _, _, label in all_pairs:
        counts[label] = counts.get(label, 0) + 1
    print(f"  Class distribution: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output", default="data/train.tsv")
    args = parser.parse_args()
    run(args.data_dir, args.output)
