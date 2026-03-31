#!/usr/bin/env python3
"""Read EditEval datasets from HuggingFace Hub with an optional --limit flag."""

import argparse
import json


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

    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    jsonl_files = sorted([f for f in files if f.endswith(".jsonl")])

    if args.datasets:
        jsonl_files = [f for f in jsonl_files if any(d in f for d in args.datasets)]

    print(f"Found {len(jsonl_files)} dataset files in {args.repo_id}:\n")

    for jsonl_file in jsonl_files:
        dataset_name = jsonl_file.replace("_input.jsonl", "")
        print(f"{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        # Stream the file and read only --limit lines
        path = api.hf_hub_download(repo_id=args.repo_id, filename=jsonl_file, repo_type="dataset")
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                record = json.loads(line)
                print(f"\n--- Sample {i + 1} ---")
                print(f"  id:        {record.get('id', 'N/A')}")
                print(f"  task_type: {record.get('task_type', 'N/A')}")
                input_text = record.get("input", "")
                if len(input_text) > 200:
                    input_text = input_text[:200] + "..."
                print(f"  input:     {input_text}")

        # Count total lines
        with open(path, "r") as f:
            total = sum(1 for _ in f)
        print(f"\n  (showing {min(args.limit, total)} of {total} total examples)\n")


if __name__ == "__main__":
    main()
