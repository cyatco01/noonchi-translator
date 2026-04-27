"""
PyTorch Dataset for formality-conditioned EN→KR translation.

Each item prepends the formality token to the English source:
  "<formal> Can you help me with this?" → tokenized input
  "도와주시겠습니까?" → tokenized labels

The tokenizer's src_lang and tgt_lang are set once at Dataset construction,
not inside __getitem__, to avoid thread-safety issues with DataLoader workers.

Padding is intentionally omitted here — DataCollatorForSeq2Seq handles dynamic
per-batch padding, which is far more efficient than fixed max_length padding
when most sentences are short (p95 English is 17 words in this corpus).
"""

import csv
import random
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


class NoonchiDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[str, str, str]],
        tokenizer,
        max_length: int = 128,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set language codes once — not safe to mutate inside __getitem__
        # when DataLoader uses multiple workers.
        self.tokenizer.src_lang = "en_XX"
        self.tokenizer.tgt_lang = "ko_KR"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        en, ko, formality = self.rows[idx]
        src = f"<{formality}> {en}"

        # Single tokenizer call encodes source (with en_XX lang token) and
        # target (with ko_KR lang token) together. Replaces the deprecated
        # as_target_tokenizer() context manager removed in transformers 4.41.
        model_inputs = self.tokenizer(
            src,
            text_target=ko,
            max_length=self.max_length,
            truncation=True,
        )

        # Mask padding positions in labels so loss ignores them.
        model_inputs["labels"] = [
            -100 if t == self.tokenizer.pad_token_id else t
            for t in model_inputs["labels"]
        ]
        return model_inputs


def load_split(
    tsv_path: str,
    tokenizer,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> NoonchiDataset:
    """
    Load a pre-split TSV file (train.tsv, val.tsv, or test.tsv) as a Dataset.

    If max_rows is set, returns a stratified sample of that size — used to
    fit training within a Colab free T4 session (~50K rows, ~6h).
    """
    rows: list[tuple[str, str, str]] = []
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) == 3 and row[2] in ("formal", "polite", "casual"):
                rows.append((row[0], row[1], row[2]))

    if max_rows is not None and max_rows < len(rows):
        rows = _stratified_sample(rows, max_rows, seed)

    return NoonchiDataset(rows, tokenizer)


def _stratified_sample(
    rows: list[tuple[str, str, str]], n: int, seed: int
) -> list[tuple[str, str, str]]:
    by_class: dict[str, list] = {}
    for row in rows:
        by_class.setdefault(row[2], []).append(row)

    rng = random.Random(seed)
    per_class = n // len(by_class)
    sampled: list[tuple[str, str, str]] = []
    for class_rows in by_class.values():
        rng.shuffle(class_rows)
        sampled.extend(class_rows[:per_class])

    rng.shuffle(sampled)
    return sampled
