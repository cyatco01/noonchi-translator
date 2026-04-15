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

from backend.data_pipeline.label import label_sentence


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
) -> float:
    """
    Formality Accuracy (FA): the percentage of model outputs whose
    sentence-final morpheme matches the requested formality token.

    Applies the same morphological labeling pipeline used in data construction.
    """
    correct = 0
    for pred, label in zip(predictions, requested_labels):
        predicted_label = label_sentence(pred)
        if predicted_label == label:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def evaluate(
    hypotheses: list[str],
    references: list[str],
    requested_labels: list[str],
) -> dict[str, float]:
    """Run all three metrics and return results as a dict."""
    return {
        "chrF": compute_chrf(hypotheses, references),
        "BLEU": compute_bleu(hypotheses, references),
        "formality_accuracy": formality_accuracy(hypotheses, requested_labels),
    }
