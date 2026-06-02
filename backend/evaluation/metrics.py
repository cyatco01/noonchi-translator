"""
Evaluation metrics for formality-conditioned translation.

Metrics:
  - chrF  (primary)  — character n-gram F-score; more sensitive to Korean
                       morphological variation than word-level BLEU
  - BLEU  (secondary) — word n-gram precision; reported for comparability
  - Formality Accuracy (FA) — custom metric; checks that sentence-final
                              endings match the requested formality token
"""

from sacrebleu.metrics import BLEU, CHRF


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    """Compute chrF score (primary translation quality metric)."""
    metric = CHRF()
    result = metric.corpus_score(hypotheses, [references])
    return result.score


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute BLEU score."""
    metric = BLEU(effective_order=True)
    result = metric.corpus_score(hypotheses, [references])
    return result.score


def formality_accuracy(
    predictions: list[str], requested_labels: list[str]
) -> dict:
    """
    Formality Accuracy (FA): percentage of model outputs whose sentence-final
    morpheme matches the requested formality token.

    Returns a dict with:
      accuracy     — correct / classifiable (excludes None predictions)
      none_count   — predictions where Mecab couldn't extract an EF morpheme
      classifiable — predictions where Mecab returned a label
    """
    from tqdm import tqdm
    from backend.data_pipeline.label import label_sentence  # lazy: needs Mecab, only used in Cell 8
    correct = 0
    none_count = 0
    total = len(predictions)
    for pred, label in tqdm(zip(predictions, requested_labels), total=total, desc="Formality accuracy", unit="sent"):
        predicted = label_sentence(pred)
        if predicted is None:
            none_count += 1
            continue
        if predicted == label:
            correct += 1
    classifiable = total - none_count
    return {
        "accuracy": correct / classifiable if classifiable else 0.0,
        "none_count": none_count,
        "classifiable": classifiable,
    }


def evaluate(
    hypotheses: list[str],
    references: list[str],
    requested_labels: list[str],
) -> dict:
    """Run all three metrics and return results as a dict."""
    fa = formality_accuracy(hypotheses, requested_labels)
    return {
        "chrF": compute_chrf(hypotheses, references),
        "BLEU": compute_bleu(hypotheses, references),
        "formality_accuracy": fa["accuracy"],
        "fa_none_count": fa["none_count"],
    }


def evaluate_by_class(
    hypotheses: list[str],
    references: list[str],
    requested_labels: list[str],
) -> dict[str, dict]:
    """Compute chrF and FA broken down by requested formality class."""
    by_class: dict[str, tuple[list, list]] = {}
    for hyp, ref, label in zip(hypotheses, references, requested_labels):
        hyps, refs = by_class.setdefault(label, ([], []))
        hyps.append(hyp)
        refs.append(ref)
    results = {}
    for label, (hyps, refs) in by_class.items():
        fa = formality_accuracy(hyps, [label] * len(hyps))
        results[label] = {
            "chrF": compute_chrf(hyps, refs),
            "formality_accuracy": fa["accuracy"],
            "none_count": fa["none_count"],
            "n": len(hyps),
        }
    return results
