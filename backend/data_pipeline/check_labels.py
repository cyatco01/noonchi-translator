"""
Diagnostic: spot-check Mecab formality labels against Claude's independent judgment.

Samples rows from train.tsv (stratified by class), asks Claude Haiku to label
each Korean sentence's formality independently, then reports mismatch rate per class.
Systematic mismatches indicate labeler bugs to fix in label.py.

Note: train.tsv contains a mix of original corpus rows (Mecab-labeled) and synthetic rows
from augmentation (substitution rules or LLM + morphological verification). Mismatches on
synthetic rows reflect augmentation artifacts, not Mecab errors. For a pure Mecab accuracy
check, run this against the pre-pipeline corpus instead of train.tsv.

Usage:
    python -m backend.data_pipeline.check_labels --data data/train.tsv --sample 500
"""

import argparse
import csv
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_TOOL = {
    "name": "label_formality",
    "description": "Label the formality level of each Korean sentence based on its sentence-final verb ending.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "formality": {
                            "type": "string",
                            "enum": ["formal", "polite", "casual"],
                        },
                        "ending": {
                            "type": "string",
                            "description": "The sentence-final morpheme you identified (e.g. 습니다, 아요, 어).",
                        },
                    },
                    "required": ["index", "formality", "ending"],
                },
            }
        },
        "required": ["labels"],
    },
}


def stratified_sample(data_path: str, per_class: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    with open(data_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3 and row[2] in ("formal", "polite", "casual"):
                buckets[row[2]].append({"en": row[0], "ko": row[1], "pipeline_label": row[2]})
    sample = []
    for label, rows in buckets.items():
        random.shuffle(rows)
        sample.extend(rows[:per_class])
    random.shuffle(sample)
    return sample


def check_labels(data_path: str, total_sample: int) -> None:
    from dotenv import load_dotenv
    import anthropic

    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    per_class = total_sample // 3
    sample = stratified_sample(data_path, per_class)
    logger.info(f"Loaded {len(sample)} rows for label spot-check ({per_class} per class)")

    client = anthropic.Anthropic(api_key=api_key)
    results = []
    batch_size = 20

    for i in range(0, len(sample), batch_size):
        batch = sample[i: i + batch_size]
        numbered = "\n".join(f"{j + 1}. {row['ko']}" for j, row in enumerate(batch))
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "label_formality"},
            messages=[{
                "role": "user",
                "content": (
                    "You are a Korean linguistics expert. Classify each Korean sentence below "
                    "as formal, polite, or casual based solely on the sentence-final verb ending.\n\n"
                    "Speech level guide:\n"
                    "- formal (하십시오체): ends with -습니다, -ㅂ니다, -습니까, -십시오\n"
                    "- polite (해요체): ends with -아요, -어요, -여요, -죠\n"
                    "- casual (해체): ends with -아, -어, -냐, -지, -야\n\n"
                    + numbered
                ),
            }],
            timeout=60.0,
        )

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_block:
            logger.warning(f"Batch {i // batch_size + 1}: no tool call returned")
            continue

        for r in tool_block.input.get("labels", []):
            idx = r["index"] - 1
            if 0 <= idx < len(batch):
                row = batch[idx]
                results.append({
                    "ko": row["ko"],
                    "en": row["en"],
                    "pipeline_label": row["pipeline_label"],
                    "claude_label": r["formality"],
                    "ending": r["ending"],
                    "match": row["pipeline_label"] == r["formality"],
                })

        logger.info(f"Processed {min(i + batch_size, len(sample))}/{len(sample)} rows")

    # --- Report ---
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_class[r["pipeline_label"]].append(r)

    print(f"\n{'=' * 60}")
    print(f"LABEL SPOT-CHECK  —  {len(results)} rows evaluated")
    print(f"{'=' * 60}")

    mismatches = []
    for cls in ("formal", "polite", "casual"):
        rows = by_class[cls]
        if not rows:
            continue
        n = len(rows)
        matches = sum(1 for r in rows if r["match"])
        pct = matches / n * 100
        print(f"\n{cls.upper():8} — {matches}/{n} match ({pct:.1f}% agreement)")
        cls_mismatches = [r for r in rows if not r["match"]]
        mismatches.extend(cls_mismatches)
        if cls_mismatches:
            endings = Counter(r["ending"] for r in cls_mismatches)
            print(f"  Mismatched endings: {dict(endings.most_common(5))}")

    if mismatches:
        print(f"\nMISMATCH EXAMPLES (up to 15):")
        for r in mismatches[:15]:
            print(f"  KO: {r['ko']}")
            print(f"  Pipeline: {r['pipeline_label']}  |  Claude: {r['claude_label']}  |  Ending: {r['ending']}")
            print()
    else:
        print("\nNo mismatches found.")

    output_path = Path("data") / "check_labels_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.tsv", help="Path to train.tsv")
    parser.add_argument(
        "--sample",
        type=int,
        default=500,
        metavar="N",
        help="Total rows to sample, stratified across 3 classes (default: 500)",
    )
    args = parser.parse_args()
    check_labels(args.data, args.sample)
