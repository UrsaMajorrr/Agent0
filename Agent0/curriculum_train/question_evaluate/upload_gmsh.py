#!/usr/bin/env python
"""
Upload/Filter GMSH Curriculum Tasks

Combines task files from multiple GPUs, filters by score, and saves for executor training.
"""

import argparse
import json
import os
import sys

from datasets import Dataset

try:
    STORAGE_PATH = os.environ["STORAGE_PATH"]
    print(f"STORAGE_PATH is: {STORAGE_PATH}", file=sys.stderr)
except KeyError:
    print("Error: STORAGE_PATH environment variable not set.", file=sys.stderr)
    sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--experiment_name", type=str, default="gmsh_tasks")
args = parser.parse_args()

# Load data from all GPU outputs
datas = []
for i in range(8):
    file_path = f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            datas.extend(data)
            print(f"Loaded {len(data)} tasks from {file_path}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found, skipping.", file=sys.stderr)
        continue

print(f"Total tasks loaded: {len(datas)}", file=sys.stderr)

# Filter by score (completeness * diversity adjustment)
# Note: evaluate_gmsh.py saves task as 'question', so check both keys
filtered_datas = [
    {
        'task': data.get('task') or data.get('question', ''),
        'score': data.get('score', 0),
        'completeness': data.get('completeness', 0),
        'answer': data.get('answer', ''),  # Include the best script
    }
    for data in datas
    if (data.get('task') or data.get('question')) and args.min_score <= data.get('score', 0) <= args.max_score
]

print(f"Filtered to {len(filtered_datas)} tasks (score: {args.min_score}-{args.max_score})", file=sys.stderr)

if filtered_datas:
    # Save as parquet for executor training
    train_dataset = Dataset.from_list(filtered_datas)

    save_dir = f"{STORAGE_PATH}/generated_question/{args.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/train.parquet"
    train_dataset.to_parquet(save_path)

    # Print path to stdout for capture by shell script
    print(save_path)

    # Also save as JSON for convert_curriculum_to_executor.py
    json_path = f"{save_dir}/filtered_tasks.json"
    with open(json_path, 'w') as f:
        json.dump(filtered_datas, f, indent=2)
    print(f"Also saved to: {json_path}", file=sys.stderr)
else:
    print("Warning: No data to save after filtering.", file=sys.stderr)
    sys.exit(1)
