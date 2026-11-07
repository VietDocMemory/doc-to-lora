"""Prepare deterministic, class-balanced ViQuAD2 parquet files."""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="taidng/UIT-ViQuAD2.0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_datasets/viquad2_balanced"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_impossible(row: dict) -> bool:
    return bool(row.get("is_impossible", False))


def oversample(dataset: Dataset, target_size: int, rng: random.Random) -> Dataset:
    if len(dataset) >= target_size:
        return dataset
    extra_indices = [
        rng.randrange(len(dataset)) for _ in range(target_size - len(dataset))
    ]
    return concatenate_datasets([dataset, dataset.select(extra_indices)])


def write_split(dataset: Dataset, output_dir: Path, split: str) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(split_dir / "ds.parquet")

    answerable = sum(not is_impossible(row) for row in dataset)
    unanswerable = len(dataset) - answerable
    manifest = {
        "split": split,
        "count": len(dataset),
        "answerable": answerable,
        "unanswerable": unanswerable,
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    source = load_dataset(args.dataset)
    train = source["train"]

    answerable = train.filter(lambda row: not is_impossible(row))
    unanswerable = train.filter(is_impossible)
    target_size = max(len(answerable), len(unanswerable))
    rng = random.Random(args.seed)

    balanced_train = concatenate_datasets(
        [
            oversample(answerable, target_size, rng),
            oversample(unanswerable, target_size, rng),
        ]
    ).shuffle(seed=args.seed)

    write_split(balanced_train, args.output_dir, "train")
    write_split(source["validation"], args.output_dir, "validation")
    print("Test split was not written and must not be used for training.")


if __name__ == "__main__":
    main()
