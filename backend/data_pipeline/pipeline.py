"""
End-to-end data pipeline orchestration.

Runs all stages and writes stratified train/val/test TSV files.
Augmentation is applied to the training split only — val and test contain
only original-corpus sentences, preventing synthetic data from leaking into
evaluation.

Usage:
    # Basic run (substitution augmentation only):
    python -m backend.data_pipeline.pipeline --data-dir data/ --output-dir data/

    # With LLM augmentation — bring training classes up to 150,000 pairs each:
    python -m backend.data_pipeline.pipeline --data-dir data/ --output-dir data/ --augment-to 150000

    # With contrastive triplets:
    python -m backend.data_pipeline.pipeline --data-dir data/ --output-dir data/ --triplets 5000

Output:
    data/train.tsv  (filtered + augmented, ~80% of original corpus)
    data/val.tsv    (filtered only, ~10%)
    data/test.tsv   (filtered only, ~10%)
"""

import argparse
import csv
import logging
from pathlib import Path

from .extract import load_corpus
from .label import label_sentence
from .filter import filter_pairs
from .augment import augment_by_substitution, augment_by_llm, augment_by_triplets
from .split import split_rows

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _write_tsv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["en", "ko", "formality"])
        writer.writerows(rows)


def run(
    data_dir: str,
    output_dir: str,
    augment_to: int | None = None,
    triplets: int | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    split_seed: int = 42,
) -> None:
    print("Stage 1: Extracting corpora...")
    raw_pairs = load_corpus(data_dir)
    print(f"  {len(raw_pairs):,} raw pairs loaded")

    print("Stage 2–3: Morphological analysis and formality labeling...")
    labeled = [(en, ko, label_sentence(ko)) for en, ko in raw_pairs]

    print("Stage 4: Confidence filtering...")
    filtered = filter_pairs(labeled)
    print(f"  {len(filtered):,} pairs retained ({len(labeled) - len(filtered):,} removed)")

    # Split BEFORE augmentation so val and test contain only original-corpus
    # sentences. Augmented rows (substitution, LLM, triplets) go to training only.
    print(f"Stage 5: Stratified split ({train_frac:.0%} train / {val_frac:.0%} val / "
          f"{1 - train_frac - val_frac:.0%} test, seed={split_seed})...")
    train_base, val, test = split_rows(filtered, train_frac, val_frac, split_seed)
    print(f"  {len(train_base):,} train / {len(val):,} val / {len(test):,} test")

    print("Stage 6: Augmenting training split...")
    augmented = augment_by_substitution(train_base)
    train = train_base + augmented
    print(f"  Substitution: {len(augmented):,} synthetic pairs added → {len(train):,} training rows")

    if augment_to is not None:
        counts: dict[str, int] = {}
        for _, _, label in train:
            if label:
                counts[label] = counts.get(label, 0) + 1

        # Build KO dedup set from all original corpus sentences (train + val + test)
        # so LLM-generated Korean doesn't duplicate existing corpus entries.
        corpus_ko = {ko for _, ko, _ in filtered}

        for label in ("formal", "polite", "casual"):
            current = counts.get(label, 0)
            if current < augment_to:
                needed = augment_to - current
                print(f"  LLM augmentation [{label}]: generating {needed:,} pairs (current: {current:,})")
                llm_pairs = augment_by_llm(label, needed, corpus_ko=corpus_ko)  # type: ignore[arg-type]
                train = train + llm_pairs
                print(f"  LLM augmentation [{label}]: {len(llm_pairs):,} verified pairs added")
            else:
                print(f"  LLM augmentation [{label}]: already at {current:,} ≥ {augment_to:,}, skipping")

    if triplets is not None:
        print(f"Stage 6b: Generating {triplets:,} contrastive triplets (training only)...")
        triplet_pairs = augment_by_triplets(train, triplets)
        train = train + triplet_pairs
        print(f"  Triplets: {len(triplet_pairs):,} rows added ({len(triplet_pairs) // 3:,} complete triplets) → {len(train):,} training rows")

    print("Stage 6c: Deduplicating splits...")
    def _dedup(rows: list) -> list:
        seen: set[tuple[str, str]] = set()
        out = []
        for row in rows:
            key = (row[0], row[1])
            if key not in seen:
                seen.add(key)
                out.append(row)
        return out

    before_train, before_val, before_test = len(train), len(val), len(test)
    train, val, test = _dedup(train), _dedup(val), _dedup(test)
    print(f"  Removed {before_train - len(train):,} train / "
          f"{before_val - len(val):,} val / "
          f"{before_test - len(test):,} test duplicates")

    print("Stage 7: Writing splits...")
    out = Path(output_dir)
    _write_tsv(train, out / "train.tsv")
    _write_tsv(val,   out / "val.tsv")
    _write_tsv(test,  out / "test.tsv")
    print(f"  train.tsv: {len(train):,} rows  |  val.tsv: {len(val):,} rows  |  test.tsv: {len(test):,} rows")

    train_counts: dict[str, int] = {}
    for _, _, label in train:
        if label:
            train_counts[label] = train_counts.get(label, 0) + 1
    print(f"  Training class distribution: {train_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument(
        "--augment-to",
        type=int,
        default=None,
        metavar="N",
        help="Use LLM augmentation to bring training classes up to N pairs each (e.g. 150000)",
    )
    parser.add_argument(
        "--triplets",
        type=int,
        default=None,
        metavar="N",
        help="Generate N contrastive triplets for training (same EN → formal + polite + casual KO). Requires ANTHROPIC_API_KEY.",
    )
    parser.add_argument("--train", type=float, default=0.8, dest="train_frac", metavar="FRAC")
    parser.add_argument("--val",   type=float, default=0.1, dest="val_frac",   metavar="FRAC")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.output_dir,
        augment_to=args.augment_to,
        triplets=args.triplets,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
    )
