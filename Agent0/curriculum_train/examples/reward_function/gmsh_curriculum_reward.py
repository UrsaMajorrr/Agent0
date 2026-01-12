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
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="curriculum_train_logger.log", level=logging.DEBUG)

geometry_directory = Path("geometries")
geometry_files = [str(item) for item in geometry_directory.iterdir() if item.is_file()]

# BLEU-based clustering for diversity
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sklearn.cluster import AgglomerativeClustering
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    logger.error("[Warning] NLTK/sklearn not available, using simple diversity metric")

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

    logger.info('  - Starting task clustering...')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(tasks)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    logger.info(f'  - Clustering done, time: {time.time() - start_time:.2f}s')

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
            logger.exception(f"  - WARNING: Result file not found: {result_file}")

    return final_results


# ============================================================================
# Task Completeness (fallback when vLLM not available)
# ============================================================================

# Analysis types for FEA meshing tasks
ANALYSIS_TYPES = [
    'thermal', 'structural', 'modal', 'buckling', 'fatigue', 'dynamic',
    'static', 'stress', 'heat', 'vibration', 'frequency',
    'transient', 'steady-state'
]

# Keywords indicating physical groups with meaningful assignments
PHYSICAL_GROUP_KEYWORDS = [
    'physical group', 'physical_group', 'physicalgroup',
    'boundary', 'boundary condition',
]

# Mesh refinement keywords
MESH_REFINEMENT_KEYWORDS = [
    'mesh size', 'element size', 'mesh_size', 'element_size',
    'refinement', 'refine', 'local refinement',
    'size field', 'distance field', 'threshold field', 'box field',
    'boundary layer', 'boundary_layer', 'boundarylayer',
    'transfinite', 'structured', 'unstructured',
    'curvature', 'curvature-based', 'adaptive',
    'global size', 'local size', 'min size', 'max size',
    'setsize', 'lc', 'characteristic length'
]

# Surface type keywords (from geometry metadata)
SURFACE_TYPE_KEYWORDS = [
    'plane', 'planar', 'cylinder', 'cylindrical', 'torus', 'toroidal',
    'sphere', 'spherical', 'cone', 'conical', 'curved', 'fillet'
]


def check_geometry_file_reference(task: str) -> float:
    """
    Check if task properly references a geometry file.
    Returns 1.0 if valid file found, 0.0 otherwise.
    """
    match = re.search(r'geometries/[a-zA-Z0-9_\-\.]+\.step', task, re.IGNORECASE)

    if not match:
        return 0.0

    geom_file_string = match.group(0)

    if geom_file_string in geometry_files:
        return 1.0

    return 0.0


def check_analysis_type(task: str) -> float:
    """Check if task specifies an analysis type."""
    task_lower = task.lower()
    for analysis in ANALYSIS_TYPES:
        if analysis in task_lower:
            # Bonus for context (e.g., "thermal analysis", "structural simulation")
            if re.search(rf'{analysis}\s+(analysis|simulation|problem|study)', task_lower):
                return 1.0
            return 0.8
    return 0.0


def check_physical_groups(task: str) -> float:
    """
    Check if task defines physical groups.
    Returns score 0.0-1.0 based on quality of physical group definitions.
    """
    task_lower = task.lower()
    score = 0.0

    # Check for physical group keyword
    has_pg_keyword = any(kw in task_lower for kw in PHYSICAL_GROUP_KEYWORDS)
    if has_pg_keyword:
        score += 0.4

    # Check for quoted group names (e.g., 'inlet', "wall", etc.)
    quoted_names = re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]", task)
    if len(quoted_names) >= 3:
        score += 0.4
    elif len(quoted_names) >= 2:
        score += 0.3
    elif len(quoted_names) >= 1:
        score += 0.2

    # Check for surface assignment patterns (e.g., "for the cylindrical surfaces")
    surface_assignment = re.search(
        r'(for|assign|on|at)\s+(the\s+)?(plane|planar|cylinder|cylindrical|curved|torus|spherical|conical)\s+(surface|face)',
        task_lower
    )
    if surface_assignment:
        score += 0.2

    return min(score, 1.0)


def check_mesh_refinement(task: str) -> float:
    """
    Check if task specifies mesh refinement requirements.
    Returns score 0.0-1.0 based on quality of mesh specifications.
    """
    task_lower = task.lower()
    score = 0.0

    # Count mesh refinement keywords found
    refinement_keywords_found = sum(1 for kw in MESH_REFINEMENT_KEYWORDS if kw in task_lower)
    if refinement_keywords_found:
        score += 0.5

    # Check for numeric mesh size values
    if re.search(r'(size|lc|length)\s*[=:]?\s*\d+\.?\d*', task_lower):
        score += 0.25

    # Check for size field specification (advanced)
    if re.search(r'(distance|threshold|box|min|max)\s*field', task_lower):
        score += 0.25

    return min(score, 1.0)


def check_task_completeness(task: str) -> float:
    """
    Check if task mentions all required elements for geometry-based meshing.

    Tasks should include:
    1. Reference to a .step geometry file (25%)
    2. Analysis type (20%)
    3. Physical groups with meaningful names (30%)
    4. Mesh refinement specifications (25%)

    Returns score 0.0-1.0 based on completeness.
    """
    if not task:
        return 0.0

    # Score each component
    geometry_score = check_geometry_file_reference(task)
    analysis_score = check_analysis_type(task)
    physical_group_score = check_physical_groups(task)
    mesh_refinement_score = check_mesh_refinement(task)

    # Weighted combination
    final_score = (
        0.25 * geometry_score +
        0.25 * analysis_score +
        0.25 * physical_group_score +
        0.25 * mesh_refinement_score
    )

    return final_score


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
    logger.info("\n[Gmsh Curriculum Reward]")
    logger.info(f" - Processing {len(predicts)} predictions...")

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

    if valid_tasks_data:
        logger.info(" - Sending tasks to vLLM servers for validation...")
        final_results = generate_results(valid_tasks_data)
    else:
        logger.warning(" - No valid tasks to process")
        final_results = []

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
    logger.info(" - Computing final scores...")
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
        logger.info(f" - Average score: {np.mean([s['overall'] for s in valid_scores]):.3f}")
        logger.info(f" - Format rate: {np.mean([s['format'] for s in valid_scores]):.3f}")
        logger.info(f" - Executability: {np.mean([s['executability'] for s in valid_scores]):.3f}")
        logger.info(f" - Diversity: {np.mean([s['diversity'] for s in valid_scores]):.3f}")
    else:
        logger.warning(" - No valid scores to report")

    return scores
