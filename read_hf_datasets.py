#!/usr/bin/env python3
"""Read EditEval datasets from HuggingFace Hub with an optional --limit flag.

Each sub-dataset is stored in its own subdirectory and can be loaded
independently via:

    load_dataset("bzz2/EditEval", data_dir="jfleg")

Available datasets:
    jfleg, asset, turk, iterater, iterater_fluency, iterater_clarity,
    iterater_coherence, stsb_multi_mt, wnc, wafer_insert, fruit
"""

import argparse

ALL_DATASETS = [
    "jfleg",
    "asset",
    "turk",
    "iterater",
    "iterater_fluency",
    "iterater_clarity",
    "iterater_coherence",
    "stsb_multi_mt",
    "wnc",
    "wafer_insert",
    "fruit",
]


def main():
    parser = argparse.ArgumentParser(description="Read EditEval datasets from HuggingFace Hub")
    parser.add_argument(
        "--limit", type=int, default=3, help="Number of samples to read from each dataset (default: 3)"
    )
    parser.add_argument(
        "--repo_id", type=str, default="bzz2/EditEval", help="HuggingFace dataset repo id"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Specific dataset names to read (e.g., jfleg asset). Default: all.",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    dataset_names = args.datasets if args.datasets else ALL_DATASETS
    print(f"Loading {len(dataset_names)} datasets from {args.repo_id}:\n")

    for name in dataset_names:
        ds = load_dataset(args.repo_id, data_dir=name, split="train")
        total = len(ds)

        print(f"{'=' * 60}")
        print(f"Dataset: {name}  ({total} examples)")
        print(f"{'=' * 60}")

        for i in range(min(args.limit, total)):
            sample = ds[i]
            print(f"\n--- Sample {i + 1} ---")
            print(f"  id:        {sample['id']}")
            print(f"  task_type: {sample['task_type']}")
            input_text = sample["input"]
            if len(input_text) > 200:
                input_text = input_text[:200] + "..."
            print(f"  input:     {input_text}")

        print(f"\n  (showing {min(args.limit, total)} of {total} total examples)\n")


if __name__ == "__main__":
    main()
