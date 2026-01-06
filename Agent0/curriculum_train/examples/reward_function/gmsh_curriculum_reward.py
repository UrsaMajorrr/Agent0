#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gmsh Curriculum Reward Function

Evaluates curriculum agent's generated meshing tasks by:
1. Parsing task format (has <task>...</task> tags)
2. Calling vLLM servers to generate Gmsh scripts and validate them
3. Penalizing duplicate/similar tasks using BLEU clustering

Scoring:
- Format: Has proper task tags (10%)
- Executability: vLLM executor can generate working scripts (70%)
- Diversity: Tasks are varied using BLEU clustering (20%)
"""

import regex as re
from typing import Dict, List
import json
import os
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import numpy as np

# BLEU-based clustering for diversity
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sklearn.cluster import AgglomerativeClustering
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    print("[Warning] NLTK/sklearn not available, using simple diversity metric")

STORAGE_PATH = os.getenv("STORAGE_PATH", "")

# ============================================================================
# Task Clustering (for diversity reward)
# ============================================================================

def _bleu_distance_matrix(sentences):
    """Calculate BLEU distance matrix for task clustering"""
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in tqdm(range(n), desc="  - Calculating BLEU distance matrix", leave=False):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist


def cluster_share_per_task(
        tasks,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    """
    Cluster tasks by similarity and return diversity penalty per task.
    Tasks in large clusters get higher penalty (less diverse).
    """
    if not tasks:
        return []

    if not HAS_CLUSTERING or len(tasks) < 3:
        # Fallback: simple hash-based deduplication
        seen = {}
        penalties = []
        for task in tasks:
            key = hash(task[:100])  # Hash first 100 chars
            if key in seen:
                seen[key] += 1
            else:
                seen[key] = 1
            penalties.append(seen[key] / len(tasks))
        return penalties

    print('  - Starting task clustering...')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(tasks)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'  - Clustering done, time: {time.time() - start_time:.2f}s')

    total = len(tasks)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    # Penalty proportional to cluster size (discourage duplicate tasks)
    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


# ============================================================================
# vLLM Server Communication
# ============================================================================

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"


def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"


def fetch(index, filename):
    """Send task file to vLLM server for processing"""
    response = requests.get(f"http://0.0.0.0:{5000+index}/hello?name={filename}")
    return True


def generate_results(data):
    """
    Send tasks to 4 vLLM servers in parallel for script generation and evaluation.

    Args:
        data: List of dicts with 'task' and 'geometry_file'

    Returns:
        List of dicts with 'task', 'best_score', etc.
    """
    datas = split_list(data, 4)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(4)]

    # Write task files
    for i in range(4):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    # Send to servers in parallel
    final_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i, random_names[i]) for i in range(4)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="  - Servers processing"):
            future.result()

    # Read result files
    for i in tqdm(range(4), desc="  - Reading result files", leave=False):
        result_file = random_names[i].replace('.json', '_results.json')
        try:
            with open(result_file, 'r') as f:
                final_results.extend(json.load(f))
            os.remove(result_file)
        except FileNotFoundError:
            print(f"  - WARNING: Result file not found: {result_file}")

    return final_results


# ============================================================================
# Task Completeness (fallback when vLLM not available)
# ============================================================================

# Geometry types the agent should pick from
GEOMETRY_TYPES = ['box', 'cylinder', 'sphere', 'cone', 'wedge', 'torus', 'cube', 'block']

# Analysis types
ANALYSIS_TYPES = ['thermal', 'structural', 'modal', 'buckling', 'fatigue', 'dynamic',
                  'stress', 'heat', 'vibration', 'frequency']

# Keywords indicating physical groups
PHYSICAL_GROUP_KEYWORDS = ['physical group', 'physical_group', 'physicalgroup',
                           'boundary', 'face', 'surface', 'volume', 'top', 'bottom',
                           'inlet', 'outlet', 'wall', 'domain']


def check_task_completeness(task: str) -> float:
    """
    Check if task mentions all required elements.
    Returns score 0.0-1.0 based on completeness.
    """
    task_lower = task.lower()
    score = 0.0

    # Check for geometry type (required)
    for geom in GEOMETRY_TYPES:
        if geom in task_lower:
            score += 0.25
            break

    # Check for dimensions (numbers like 10x10x10, radius 5, etc.)
    if re.search(r'\d+\s*[x×]\s*\d+', task_lower):  # 10x10 format
        score += 0.25
    elif re.search(r'(radius|height|width|length|depth|size)\s*[=:]?\s*\d+', task_lower):
        score += 0.25
    elif re.search(r'\d+\s*(mm|cm|m|unit)', task_lower):
        score += 0.25

    # Check for analysis type
    for analysis in ANALYSIS_TYPES:
        if analysis in task_lower:
            score += 0.25
            break

    # Check for physical groups
    for keyword in PHYSICAL_GROUP_KEYWORDS:
        if keyword in task_lower:
            score += 0.25
            break

    return min(score, 1.0)


# ============================================================================
# Main Reward Computation
# ============================================================================

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.1,
    file_path: str = ""
) -> List[Dict[str, float]]:
    """
    Main reward function for Gmsh curriculum training.

    Evaluates tasks based on:
    - Format: Has <task>...</task> tags (10%)
    - Executability: vLLM executor can generate working scripts (70%)
    - Diversity: Tasks are varied using BLEU clustering (20%)
    """
    print("\n[Gmsh Curriculum Reward]")
    print(f" - Processing {len(predicts)} predictions...")

    # Save predictions for debugging
    with open('test.json', 'w') as f:
        json.dump(predicts, f, indent=4)

    # Step 1: Parse predictions to extract tasks
    tasks = []
    for i in tqdm(range(len(predicts)), desc=" - Parsing predictions"):
        task_matches = re.findall(r"<task>(.*?)</task>", predicts[i], re.DOTALL)

        if task_matches:
            task = task_matches[-1].strip()
            tasks.append({"task": task})
        else:
            tasks.append({"task": ""})

    # Step 2: Try to call vLLM servers for validation
    valid_tasks_data = [t for t in tasks if t['task']]

    try:
        if valid_tasks_data:
            print(" - Sending tasks to vLLM servers for validation...")
            final_results = generate_results(valid_tasks_data)
        else:
            print(" - No valid tasks to process")
            final_results = []
    except Exception as e:
        print(f" - WARNING: vLLM servers not available ({e}), using static analysis")
        # Fallback to static completeness check
        final_results = []
        for t in valid_tasks_data:
            completeness = check_task_completeness(t['task'])
            final_results.append({
                'task': t['task'],
                'best_score': completeness,
                'num_successful': 1 if completeness > 0.5 else 0
            })

    # Map results back to original indices
    final_results_map = {r['task']: r for r in final_results}

    # Step 3: Calculate diversity penalty
    valid_task_strings = [t['task'] for t in tasks if t['task']]
    if valid_task_strings:
        penalty = cluster_share_per_task(valid_task_strings, distance_threshold=0.5)
    else:
        penalty = []

    # Map penalties back
    penalty_map = {}
    for i, task_str in enumerate(valid_task_strings):
        penalty_map[task_str] = penalty[i] if i < len(penalty) else 0.0

    # Step 4: Compute final scores
    print(" - Computing final scores...")
    scores = []

    for i in tqdm(range(len(tasks)), desc=" - Calculating final scores"):
        task = tasks[i]['task']

        if not task:
            # No valid task extracted
            scores.append({
                "overall": -1.0,
                "format": 0.0,
                "executability": 0.0,
                "diversity": 0.0
            })
            continue

        # Get executor score
        result = final_results_map.get(task, {})
        executor_score = result.get('best_score', 0.0)

        # Get diversity penalty
        div_penalty = penalty_map.get(task, 0.0)

        # Calculate weighted score
        # Format: 10%, Executability: 70%, Diversity: 20%
        format_score = 1.0  # Task was successfully extracted
        diversity_score = 1.0 - div_penalty

        final_score = (
            0.10 * format_score +
            0.70 * executor_score +
            0.20 * diversity_score
        )

        scores.append({
            "overall": final_score,
            "format": format_score,
            "executability": executor_score,
            "diversity": diversity_score
        })

    # Print summary
    valid_scores = [s for s in scores if s['overall'] >= 0]
    if valid_scores:
        print(f" - Average score: {np.mean([s['overall'] for s in valid_scores]):.3f}")
        print(f" - Format rate: {np.mean([s['format'] for s in valid_scores]):.3f}")
        print(f" - Executability: {np.mean([s['executability'] for s in valid_scores]):.3f}")
        print(f" - Diversity: {np.mean([s['diversity'] for s in valid_scores]):.3f}")
    else:
        print(" - No valid scores to report")

    return scores
