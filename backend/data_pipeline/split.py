"""
Split a labeled TSV dataset into stratified train/val/test splits.

Reads the full pipeline output (en, ko, formality) and writes three files
with no overlap, preserving class distribution across splits.

Usage:
    python -m backend.data_pipeline.split \
        --input data/train.tsv \
        --output-dir data/ \
        --train 0.8 --val 0.1 --test 0.1

Output:
    data/train.tsv  (~80%)
    data/val.tsv    (~10%)
    data/test.tsv   (~10%)
"""

import argparse
import csv
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def split_dataset(
    input_path: str,
    output_dir: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> None:
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train + val fractions must be less than 1.0"

    rows: list[tuple[str, str, str]] = []
    with open(input_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) == 3 and row[2] in ("formal", "polite", "casual"):
                rows.append((row[0], row[1], row[2]))

    logger.info(f"Loaded {len(rows):,} rows from {input_path}")

    # Stratified split: split each class independently then combine.
    by_class: dict[str, list] = {}
    for row in rows:
        by_class.setdefault(row[2], []).append(row)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for label, class_rows in by_class.items():
        rng.shuffle(class_rows)
        n = len(class_rows)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train.extend(class_rows[:n_train])
        val.extend(class_rows[n_train:n_train + n_val])
        test.extend(class_rows[n_train + n_val:])

        logger.info(f"  {label}: {n_train:,} train / {n_val:,} val / {n - n_train - n_val:,} test")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{split_name}.tsv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["en", "ko", "formality"])
            writer.writerows(split_rows)
        logger.info(f"Wrote {len(split_rows):,} rows to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Stratified train/val/test split.")
    parser.add_argument("--input", required=True, help="Full labeled TSV (pipeline output)")
    parser.add_argument("--output-dir", default="data/", help="Directory for split files")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.input, args.output_dir, args.train, args.val, args.seed)
