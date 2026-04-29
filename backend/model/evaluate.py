"""
Evaluation for the fine-tuned mBART formality-conditioned model.

Runs batched inference over test.tsv and reports:
  - chrF  (primary translation quality metric)
  - BLEU  (secondary, for comparability)
  - Formality Accuracy — % of outputs whose sentence-final morpheme matches
    the requested formality token (requires Mecab/KoNLPy)

Target bars (50K-row model): chrF > 20, formality_accuracy > 0.60
Target bars (full dataset):   chrF > 30, formality_accuracy > 0.80

Usage:
    python -m backend.model.evaluate \\
        --model /content/drive/MyDrive/noonchi/checkpoints/noonchi-mbart \\
        --test data/test.tsv \\
        --batch-size 8 \\
        --num-beams 4
"""

import argparse
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from backend.evaluation.metrics import evaluate
from backend.model.dataset import load_split
from backend.model.train import TGT_LANG, load_model_and_tokenizer

logger = logging.getLogger(__name__)


def evaluate_model(
    model_dir: str,
    test_tsv: str,
    batch_size: int = 8,
    num_beams: int = 4,
) -> dict[str, float]:
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the pre-split test file directly — no re-splitting.
    test_ds = load_split(test_tsv, tokenizer)
    logger.info(f"Evaluating on {len(test_ds):,} test rows from {test_tsv}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collator)

    ko_id = tokenizer.lang_code_to_id[TGT_LANG]
    hypotheses: list[str] = []
    references: list[str] = []
    labels: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Translating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=ko_id,
                num_beams=num_beams,
                max_length=128,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            hypotheses.extend(decoded)

    # Recover references and formality labels from the dataset rows.
    for _, ko_ref, formality in test_ds.rows:
        references.append(ko_ref)
        labels.append(formality)

    logger.info("Computing metrics...")
    results = evaluate(hypotheses, references, labels)
    logger.info("Evaluation results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned mBART model on a test split."
    )
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--test", required=True, help="Path to test.tsv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4, help="Beam search width (use 1 for fast eval)")
    args = parser.parse_args()

    evaluate_model(args.model, args.test, batch_size=args.batch_size, num_beams=args.num_beams)
