"""Measure answerable/unanswerable classification from generated answers."""

import argparse
import json
import re
from pathlib import Path

from ctx_to_lora.data.definitions import VIQUAD2_ABSTENTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prediction-field", default="d2l_answer")
    return parser.parse_args()


def normalize(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    return normalized.strip(" .!")


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    canonical_abstention = normalize(VIQUAD2_ABSTENTION)
    tp = fp = fn = tn = 0

    for row in rows:
        gold_unanswerable = as_bool(
            row.get("is_impossible", row.get("question_type") == "unanswerable")
        )
        predicted_unanswerable = (
            normalize(row.get(args.prediction_field, "")) == canonical_abstention
        )
        if gold_unanswerable and predicted_unanswerable:
            tp += 1
        elif not gold_unanswerable and predicted_unanswerable:
            fp += 1
        elif gold_unanswerable:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    refusal_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    answerable_accuracy = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (answerable_accuracy + recall) / 2

    report = {
        "count": len(rows),
        "answerable_correct": tn,
        "answerable_false_refusal": fp,
        "unanswerable_correct_refusal": tp,
        "unanswerable_incorrect_answer": fn,
        "answerable_accuracy": answerable_accuracy,
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": refusal_f1,
        "balanced_accuracy": balanced_accuracy,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
