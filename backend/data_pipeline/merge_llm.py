"""
Merge LLM-generated formal pairs from checkpoint into train.tsv.

Reads data/checkpoints/llm_augment_formal.json, deduplicates against
existing train.tsv Korean sentences, and appends enough pairs to bring
the formal class to TARGET_FORMAL.

Run from project root:
    python -m backend.data_pipeline.merge_llm
"""

import csv
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_FORMAL = 150_000

ROOT = Path(__file__).parent.parent.parent
CHECKPOINT = ROOT / "data" / "checkpoints" / "llm_augment_formal.json"
TRAIN_TSV = ROOT / "data" / "train.tsv"


def main() -> None:
    with open(CHECKPOINT, encoding="utf-8") as f:
        checkpoint: list[list[str]] = json.load(f)
    logger.info(f"Checkpoint loaded: {len(checkpoint):,} formal pairs")

    existing_ko: set[str] = set()
    formal_count = 0
    with open(TRAIN_TSV, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for en, ko, label in reader:
            existing_ko.add(ko)
            if label == "formal":
                formal_count += 1

    logger.info(f"train.tsv: {formal_count:,} formal pairs already present")

    needed = TARGET_FORMAL - formal_count
    if needed <= 0:
        logger.info(f"Formal class already at {formal_count:,} — nothing to do.")
        return

    logger.info(f"Need {needed:,} more formal pairs to reach {TARGET_FORMAL:,}")

    to_add: list[tuple[str, str, str]] = []
    skipped_dupes = 0
    for row in checkpoint:
        if len(to_add) >= needed:
            break
        en, ko, label = row
        if ko in existing_ko:
            skipped_dupes += 1
            continue
        existing_ko.add(ko)
        to_add.append((en, ko, label))

    if skipped_dupes:
        logger.info(f"Skipped {skipped_dupes:,} duplicate Korean sentences")

    if len(to_add) < needed:
        logger.warning(
            f"Only {len(to_add):,} unique pairs available — "
            f"formal class will reach {formal_count + len(to_add):,}, not {TARGET_FORMAL:,}"
        )

    # Write to a temp file on the same filesystem, then atomically replace train.tsv.
    # A crash before os.replace leaves train.tsv untouched.
    tmp_path = TRAIN_TSV.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="\t")
            writer.writerow(["en", "ko", "formality"])
            with open(TRAIN_TSV, encoding="utf-8", newline="") as src:
                reader = csv.reader(src, delimiter="\t")
                next(reader)  # skip original header
                writer.writerows(reader)
            writer.writerows(to_add)
        os.replace(tmp_path, TRAIN_TSV)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    logger.info(
        f"Merged {len(to_add):,} pairs → formal class now "
        f"{formal_count + len(to_add):,} in train.tsv"
    )


if __name__ == "__main__":
    main()
