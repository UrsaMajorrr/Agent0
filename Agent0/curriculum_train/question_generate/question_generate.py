#!/usr/bin/env python
"""
Task Generator for GMSH Curriculum

Uses trained curriculum agent to generate meshing tasks.
The model picks simple primitives (box, cylinder, etc.) and creates meshing tasks.

Tasks are scored for completeness and diversity before saving.
"""

import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import json
import regex as re
import os
import sys
import random

# Add parent dir for reward function imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from examples.reward_function.gmsh_curriculum_reward import check_task_completeness, cluster_share_per_task

STORAGE_PATH = os.getenv("STORAGE_PATH", "")


# Load geometry metadata for curriculum training (from dataset.py)
def load_geometry_metadata(metadata_path: str = None) -> list:
    """Load geometry metadata from JSON file."""
    if metadata_path is None:
        # Default path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metadata_path = os.path.join(base_dir, "utils", "metadata_all.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            data = json.load(f)
            return data.get("metadata", [])
    return []


def format_geometry_for_prompt(geom_meta: dict) -> str:
    """Format geometry metadata into a prompt string for the curriculum agent."""
    filepath = geom_meta["file"]
    filename = os.path.basename(filepath)

    # Summarize surface types
    entity_types = geom_meta.get("entity_types", {})
    type_counts = {}
    for surf_id, surf_type in entity_types.items():
        type_counts[surf_type] = type_counts.get(surf_type, 0) + 1
    type_summary = ", ".join([f"{count} {t}" for t, count in sorted(type_counts.items())])

    return (
        f"**GEOMETRY FILE:** {filename}\n"
        f"**GEOMETRY PATH:** {filepath}\n"
        f"**TOPOLOGY:** {geom_meta['surfaces']} surfaces, {geom_meta['volumes']} volume(s), "
        f"{geom_meta['points']} points, {geom_meta['curves']} curves\n"
        f"**SURFACE TYPES:** {type_summary}\n\n"
        f"To load this geometry in Gmsh:\n"
        f"```python\n"
        f"gmsh.model.occ.importShapes(\"{filepath}\")\n"
        f"gmsh.model.occ.synchronize()\n"
        f"```"
    )

# Same prompt used during training (gmsh_format in dataset.py)
SYSTEM_PROMPT = (
    "You are a curriculum designer for Gmsh meshing training.\n\n"
    "You will be given a STEP geometry file. Your job is to create a challenging meshing task for this geometry.\n\n"
    "Your task MUST specify:\n"
    "1. The structural FEA analysis type (e.g. static, dynamic, thermal, modal, buckling, fatigue)\n"
    "2. Physical groups with meaningful names based on the geometry's surfaces\n"
    "   - Identify surfaces by their type (Plane, Cylinder, Torus, etc.) and assign appropriate boundary conditions\n"
    "   - Examples: 'inlet', 'outlet', 'wall', 'fixed_support', 'heat_source', 'symmetry_plane'\n"
    "3. Mesh size requirements:\n"
    "   - Global mesh size\n"
    "   - Local refinement near curved surfaces, small features, or boundary layers\n"
    "   - Use mesh size fields (Distance, Threshold, Box) for advanced refinement\n"
    "4. Mesh quality requirements if applicable (element order, optimization)\n\n"
    "DIFFICULTY LEVELS to vary:\n"
    "- BASIC: Simple mesh with uniform size, 2-3 physical groups\n"
    "- INTERMEDIATE: Multiple physical groups, local refinement near one region\n"
    "- ADVANCED: Boundary layers, multiple refinement zones, transfinite meshing\n"
    "- EXPERT: Anisotropic mesh, curvature-based sizing, structured regions\n\n"
    "Output exactly:\n"
    "<task>[complete meshing task description including the geometry path]</task>\n\n"
    "Example:\n"
    "<task>Load the geometry from [path]. Mesh it for thermal analysis. Create physical groups: 'heat_source' for the cylindrical surfaces (apply heat flux), 'convection_surfaces' for planar faces (convective cooling), 'volume' for the solid domain. Use global element size 2.0 with refinement to 0.5 near cylindrical surfaces using a Distance field.</task>"
)

USER_PROMPT_TEMPLATE = "Create a meshing task for this geometry:\n\n{geometry_context}"


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        seed=int(args.suffix),
    )

    sample_params = vllm.SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Load geometry metadata
    geometry_metadata = load_geometry_metadata()
    if geometry_metadata:
        print(f"Loaded {len(geometry_metadata)} geometries for task generation")
    else:
        print("WARNING: No geometry metadata found. Using fallback prompt.")

    # Generate prompts with different geometries for each sample
    prompts = []
    geometry_info = []  # Track which geometry was used for each prompt
    for _ in range(args.num_samples):
        if geometry_metadata:
            geom_meta = random.choice(geometry_metadata)
            geometry_context = format_geometry_for_prompt(geom_meta)
            geometry_info.append(geom_meta.get("file", "unknown"))
        else:
            geometry_context = "No geometry available. Create your own simple geometry."
            geometry_info.append(None)

        user_prompt = USER_PROMPT_TEMPLATE.format(geometry_context=geometry_context)
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True
            )
        else:
            prompt = "system: " + SYSTEM_PROMPT + '\n' + "user: " + user_prompt

        prompts.append(prompt)

    print(f"Generating {args.num_samples} tasks using {args.model}...")
    completions: List[RequestOutput] = model.generate(prompts, sampling_params=sample_params)

    # Extract tasks
    results = []
    for i, completion in enumerate(completions):
        response = completion.outputs[0].text
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)

        if task_matches:
            task = task_matches[-1].strip()
            results.append({
                "task": task,
                "raw_response": response,
                "geometry_file": geometry_info[i],
                "score": 0  # Will be updated below
            })
        else:
            results.append({
                "task": "",
                "raw_response": response,
                "geometry_file": geometry_info[i],
                "score": -1  # Invalid format
            })

    # Score tasks for completeness
    print("Scoring tasks for completeness...")
    for result in results:
        if result["task"]:
            completeness = check_task_completeness(result["task"])
            result["completeness"] = completeness
            result["score"] = completeness  # Use completeness as score
        else:
            result["completeness"] = 0.0

    # Calculate diversity penalties
    print("Calculating diversity...")
    valid_tasks = [r["task"] for r in results if r["task"]]
    if len(valid_tasks) > 1:
        penalties = cluster_share_per_task(valid_tasks, distance_threshold=0.5)
        penalty_idx = 0
        for result in results:
            if result["task"]:
                result["diversity_penalty"] = penalties[penalty_idx]
                # Adjust score: reduce for low diversity
                result["score"] = result["score"] * (1 - 0.3 * penalties[penalty_idx])
                penalty_idx += 1
            else:
                result["diversity_penalty"] = 1.0

    # Create output directory
    output_dir = f"{STORAGE_PATH}/generated_question"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{args.save_name}_{args.suffix}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    # Print summary
    valid_count = sum(1 for r in results if r["task"])
    avg_score = sum(r["score"] for r in results if r["task"]) / max(valid_count, 1)
    avg_completeness = sum(r["completeness"] for r in results if r["task"]) / max(valid_count, 1)

    print(f"\nSaved {len(results)} tasks to {output_file}")
    print(f"Valid tasks: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
    print(f"Average score: {avg_score:.3f}")
    print(f"Average completeness: {avg_completeness:.3f}")

    # Print sample
    if results and results[0]["task"]:
        print(f"\nSample task (score={results[0]['score']:.2f}):")
        print(f"  {results[0]['task'][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of tasks to generate")
    parser.add_argument("--suffix", type=str, default="0", help="Suffix for output file")
    parser.add_argument("--save_name", type=str, default="gmsh_tasks", help="Base name for output file")
    args = parser.parse_args()

    main(args)
