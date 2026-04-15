"""
mBART-50 fine-tuning for formality-conditioned English-to-Korean translation.

Model: facebook/mbart-large-50-many-to-many-mmt
New vocab tokens: <formal>, <polite>, <casual>

The resolved formality token is prepended to the English source sentence
before encoding, conditioning the decoder on the target speech level:
  "<formal> Can you help me with this?" → "도와주시겠습니까?"

Training configuration (from hyperparameter search):
  optimizer:     AdamW
  learning rate: 5e-5 with linear warmup
  batch size:    16–32 (gradient accumulation as needed)
  epochs:        3–5 (early stopping on validation chrF)
  src lang:      en_XX
  tgt lang:      ko_KR

Usage:
    python -m backend.model.train --data data/train.tsv --output models/noonchi-mbart
"""

import argparse

from transformers import MBartForConditionalGeneration, MBart50Tokenizer

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
FORMALITY_TOKENS = ["<formal>", "<polite>", "<casual>"]
SRC_LANG = "en_XX"
TGT_LANG = "ko_KR"

TRAINING_ARGS = {
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "per_device_train_batch_size": 16,
    "num_train_epochs": 5,
    "evaluation_strategy": "epoch",
    "metric_for_best_model": "chrf",
    "load_best_model_at_end": True,
}


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load mBART-50 and expand the tokenizer with formality conditioning tokens."""
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    tokenizer.add_special_tokens({"additional_special_tokens": FORMALITY_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def train(data_path: str, output_dir: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled TSV dataset")
    parser.add_argument("--output", required=True, help="Output directory for trained model")
    args = parser.parse_args()
    train(args.data, args.output)
