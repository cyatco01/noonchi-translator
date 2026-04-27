"""
mBART-50 fine-tuning for formality-conditioned English-to-Korean translation.

Model: facebook/mbart-large-50-many-to-many-mmt
New vocab tokens: <formal>, <polite>, <casual>

The resolved formality token is prepended to the English source sentence
before encoding, conditioning the decoder on the target speech level:
  "<formal> Can you help me with this?" → "도와주시겠습니까?"

Training configuration:
  optimizer:                 AdamW
  learning rate:             5e-5 with linear warmup (500 steps)
  effective batch size:      32 (batch 4 × grad_accum 8)
  epochs:                    3 with early stopping on eval_chrf
  src lang:                  en_XX
  tgt lang:                  ko_KR
  fp16:                      True (T4-compatible)
  gradient_checkpointing:    True (required to fit mBART-large on T4 16GB)

Usage (Colab — 50K stratified sample for free T4):
    python -m backend.model.train \\
        --data data/train.tsv \\
        --output /content/drive/MyDrive/noonchi/checkpoints/noonchi-mbart \\
        --max-rows 50000

Usage (full dataset — A100 or university cluster):
    python -m backend.model.train --data data/train.tsv --output models/noonchi-mbart
"""

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
from transformers import (
    DataCollatorForSeq2Seq,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from backend.evaluation.metrics import compute_chrf
from backend.model.dataset import NoonchiDataset, load_split

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
FORMALITY_TOKENS = ["<formal>", "<polite>", "<casual>"]
SRC_LANG = "en_XX"
TGT_LANG = "ko_KR"

# All training hyperparameters in one place for easy tuning.
# gradient_checkpointing trades ~25% speed for ~40% VRAM — required on T4.
# save_strategy must match eval_strategy for load_best_model_at_end to work.
TRAINING_ARGS = {
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "metric_for_best_model": "eval_chrf",
    "greater_is_better": True,
    "load_best_model_at_end": True,
    "fp16": True,
    "gradient_checkpointing": True,
    "predict_with_generate": True,
    "generation_max_length": 128,
    "dataloader_num_workers": 0,
}


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load mBART-50 and expand the tokenizer with formality conditioning tokens.

    When fine-tuning from scratch, pass MODEL_NAME (downloads from HuggingFace).
    When loading a trained checkpoint, pass the local directory path instead.
    """
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    tokenizer.add_special_tokens({"additional_special_tokens": FORMALITY_TOKENS})
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    model.resize_token_embeddings(len(tokenizer))

    # Align decoder start with inference: both must use ko_KR as the first
    # decoder token, otherwise training and generation are misaligned.
    ko_id = tokenizer.lang_code_to_id[TGT_LANG]
    model.config.decoder_start_token_id = ko_id
    model.config.forced_bos_token_id = ko_id

    return model, tokenizer


def _sample_stratified(rows: list[tuple], max_rows: int, seed: int = 42) -> list[tuple]:
    """Return a stratified sample of `max_rows` from rows, balanced across formality classes."""
    by_class: dict[str, list] = {}
    for row in rows:
        label = row[2]
        by_class.setdefault(label, []).append(row)

    rng = random.Random(seed)
    per_class = max_rows // len(by_class)
    sampled = []
    for label_rows in by_class.values():
        rng.shuffle(label_rows)
        sampled.extend(label_rows[:per_class])

    rng.shuffle(sampled)
    return sampled


def train(data_path: str, output_dir: str, max_rows: int | None = None) -> None:
    model, tokenizer = load_model_and_tokenizer()

    train_ds = load_split(data_path, tokenizer, max_rows=max_rows)
    val_path = str(Path(data_path).parent / "val.tsv")
    val_ds = load_split(val_path, tokenizer)

    logger.info(
        f"Dataset: {len(train_ds):,} train rows, {len(val_ds):,} val rows"
    )

    def compute_metrics(eval_preds):
        preds, label_ids = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        return {"chrf": compute_chrf(decoded_preds, decoded_labels)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **TRAINING_ARGS,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Fine-tune mBART-50 for formality-conditioned EN→KR translation."
    )
    parser.add_argument("--data", required=True, help="Path to train.tsv")
    parser.add_argument("--output", required=True, help="Output directory for trained model")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stratified sample size (default: use all rows). Use 50000 for Colab free T4.",
    )
    args = parser.parse_args()
    train(args.data, args.output, max_rows=args.max_rows)
