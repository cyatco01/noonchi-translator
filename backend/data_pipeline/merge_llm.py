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

    with open(TRAIN_TSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(to_add)

    logger.info(
        f"Appended {len(to_add):,} pairs → formal class now "
        f"{formal_count + len(to_add):,} in train.tsv"
    )


if __name__ == "__main__":
    main()
