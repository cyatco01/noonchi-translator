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

    # Skip translation and re-run metrics only (after a prior run cached hypotheses):
    python -m backend.model.evaluate \\
        --model /content/drive/MyDrive/noonchi/checkpoints/noonchi-mbart \\
        --test data/test.tsv \\
        --hypotheses-cache hypotheses_cache.txt
"""

import argparse
import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from backend.evaluation.metrics import evaluate
from backend.model.dataset import load_split
from backend.model.train import TGT_LANG, load_model_and_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_CACHE = "hypotheses_cache.txt"


def _load_rows(test_tsv: str) -> list[tuple[str, str, str]]:
    rows = []
    with open(test_tsv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) == 3 and row[2] in ("formal", "polite", "casual"):
                rows.append((row[0], row[1], row[2]))
    return rows


def evaluate_model(
    model_dir: str,
    test_tsv: str,
    batch_size: int = 8,
    num_beams: int = 4,
    hypotheses_cache: str = DEFAULT_CACHE,
) -> dict[str, float]:
    cache_path = Path(hypotheses_cache)

    if cache_path.exists():
        logger.info(f"Loading cached hypotheses from {cache_path} — skipping translation")
        hypotheses = cache_path.read_text(encoding="utf-8").splitlines()
    else:
        model, tokenizer = load_model_and_tokenizer(model_dir)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_ds = load_split(test_tsv, tokenizer)
        logger.info(f"Evaluating on {len(test_ds):,} test rows from {test_tsv}")

        collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
        loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collator)
        ko_id = tokenizer.lang_code_to_id[TGT_LANG]
        hypotheses: list[str] = []

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
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                )
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                hypotheses.extend(decoded)

        cache_path.write_text("\n".join(hypotheses), encoding="utf-8")
        logger.info(f"Hypotheses cached to {cache_path} — metrics can be re-run without re-translating")

    rows = _load_rows(test_tsv)
    references = [ko for _, ko, _ in rows]
    labels = [formality for _, _, formality in rows]

    if len(hypotheses) != len(references):
        cache_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Cache has {len(hypotheses)} hypotheses but test file has {len(references)} rows. "
            "Stale cache deleted — re-run Cell 8 to re-translate."
        )

    logger.info("Computing metrics...")
    results = evaluate(hypotheses, references, labels)

    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print("==========================\n")
    logger.info("Evaluation complete.")

    return results


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned mBART model on a test split."
    )
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--test", required=True, help="Path to test.tsv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4, help="Beam search width (use 1 for fast eval)")
    parser.add_argument("--hypotheses-cache", default=DEFAULT_CACHE, help="File to cache/load translated outputs")
    parser.add_argument("--output", default=None, help="Path to save evaluation results as JSON")
    args = parser.parse_args()

    results = evaluate_model(
        args.model, args.test,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        hypotheses_cache=args.hypotheses_cache,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to {out_path}")
