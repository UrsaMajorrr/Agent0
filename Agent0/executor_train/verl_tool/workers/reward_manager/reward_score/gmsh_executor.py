#!/usr/bin/env python
"""
GMSH Executor Reward - Task Completion Matching

Evaluates whether generated mesh addresses task requirements.
Uses geometry-aware scoring that accounts for geometry size to avoid
rewarding over-refinement or under-refinement.
"""

import re
from typing import Dict, Optional

def parse_task_requirements(task: str) -> Dict:
    """
    Extract requirements from task description

    Args:
        task: Natural language task description

    Returns:
        Dict of parsed requirements:
        - analysis_type: str (thermal, structural, modal, etc.)
        - min_physical_groups: int (always >= 1, physical groups REQUIRED)
        - requires_refinement: bool
        - element_size_range: (min, max) or None
        - quality_target: float or None
    """
    task_lower = task.lower()

    requirements = {
        'analysis_type': 'unknown',
        'min_physical_groups': 1,  # Always required, default minimum
        'requires_refinement': False,
        'element_size_range': None,
        'quality_target': None
    }

    # Detect analysis type
    analysis_types = {
        'thermal': ['thermal', 'heat', 'temperature'],
        'structural': ['structural', 'stress', 'strain', 'static'],
        'modal': ['modal', 'vibration', 'frequency'],
        'buckling': ['buckling', 'stability'],
        'fatigue': ['fatigue', 'cyclic'],
        'contact': ['contact', 'interface']
    }

    for atype, keywords in analysis_types.items():
        if any(kw in task_lower for kw in keywords):
            requirements['analysis_type'] = atype
            break

    # Physical groups - ALWAYS REQUIRED
    # Check if specific count mentioned
    pg_patterns = [
        r'(\d+)\s+physical\s+group',
        r'physical\s+group.*?(\d+)',
        r'create\s+(\d+)\s+(?:physical|group)'
    ]

    for pattern in pg_patterns:
        pg_match = re.search(pattern, task_lower)
        if pg_match:
            requirements['min_physical_groups'] = int(pg_match.group(1))
            break

    # If task mentions physical groups but no specific count, require at least 1
    if 'physical' in task_lower and 'group' in task_lower:
        if requirements['min_physical_groups'] < 1:
            requirements['min_physical_groups'] = 1

    # Refinement requirements
    refinement_keywords = ['refine', 'refinement', 'adaptive', 'local mesh', 'mesh size control']
    if any(kw in task_lower for kw in refinement_keywords):
        requirements['requires_refinement'] = True

    # Element size constraints
    size_patterns = [
        r'element size[:\s]+(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',
        r'size[:\s]+(\d+\.?\d*)\s*to\s+(\d+\.?\d*)',
        r'mesh size[:\s]+(\d+\.?\d*)\s*-\s*(\d+\.?\d*)'
    ]

    for pattern in size_patterns:
        size_match = re.search(pattern, task_lower)
        if size_match:
            requirements['element_size_range'] = (
                float(size_match.group(1)),
                float(size_match.group(2))
            )
            break

    # Quality target
    quality_patterns = [
        r'quality[:\s]+(\d+\.?\d*)',
        r'minimum quality[:\s]+(\d+\.?\d*)'
    ]

    for pattern in quality_patterns:
        quality_match = re.search(pattern, task_lower)
        if quality_match:
            requirements['quality_target'] = float(quality_match.group(1))
            break

    return requirements

def score_mesh_density(
    num_elements: int,
    geometry_metadata: Dict,
    element_size_hint: Optional[float] = None
) -> float:
    """
    Score mesh density relative to geometry size

    Penalizes both over-refinement (too many elements) and under-refinement (too few).

    Args:
        num_elements: Number of elements in mesh
        geometry_metadata: Dict with bbox, characteristic_length, volume
        element_size_hint: Optional target element size from task

    Returns:
        Score in [0, 1]
    """
    if num_elements == 0:
        return 0.0

    # Get geometry info
    volume = geometry_metadata.get('volume', 1.0)
    char_length = geometry_metadata.get('characteristic_length', 5.0)

    # Calculate mesh density (elements per unit volume)
    density = num_elements / max(volume, 0.01)  # Avoid division by zero

    # Estimate expected density
    # Use hint if provided, otherwise use characteristic_length
    if element_size_hint:
        target_element_size = element_size_hint
    else:
        # Default: aim for ~10-20 elements along characteristic dimension
        target_element_size = char_length / 15.0

    # Expected number of elements for this geometry
    # Rough estimate: volume / (element_size^3)
    expected_elements = volume / (target_element_size ** 3)
    expected_density = expected_elements / volume

    # Calculate ratio of actual to expected
    if expected_density > 0:
        density_ratio = density / expected_density
    else:
        density_ratio = 1.0

    # Score based on ratio
    # Ideal range: 0.5x to 2x expected density
    if 0.5 <= density_ratio <= 2.0:
        score = 1.0
    elif 0.25 <= density_ratio < 0.5:
        # Under-refined but acceptable
        score = 0.7
    elif 2.0 < density_ratio <= 4.0:
        # Over-refined but acceptable
        score = 0.7
    elif 0.1 <= density_ratio < 0.25:
        # Significantly under-refined
        score = 0.3
    elif 4.0 < density_ratio <= 8.0:
        # Significantly over-refined
        score = 0.3
    else:
        # Extremely under or over-refined
        score = 0.1

    return score

def analyze_script_for_physical_groups(script: str) -> int:
    """
    Analyze the generated script to count physical group definitions.

    Returns estimated number of physical groups defined in the script.
    """
    if not script:
        return 0

    # Look for physical group definitions in gmsh API calls
    patterns = [
        r'addPhysicalGroup',
        r'physical\s*group',
        r'PhysicalGroup',
        r'gmsh\.model\.addPhysicalGroup',
        r'setPhysicalName',
    ]

    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, script, re.IGNORECASE)
        count += len(matches)

    return count


def analyze_script_for_refinement(script: str) -> bool:
    """
    Analyze the generated script to detect mesh refinement attempts.

    Returns True if script contains refinement-related code.
    """
    if not script:
        return False

    refinement_patterns = [
        r'setSize',
        r'MeshSize',
        r'mesh\.size',
        r'field\.add',
        r'Field\.add',
        r'refine',
        r'characteristic\s*length',
        r'lc\s*=',
        r'mesh_size',
        r'setMeshSize',
    ]

    for pattern in refinement_patterns:
        if re.search(pattern, script, re.IGNORECASE):
            return True

    return False


def analyze_script_for_mesh_generation(script: str) -> bool:
    """
    Analyze the generated script to detect mesh generation attempts.

    Returns True if script contains mesh generation code.
    """
    if not script:
        return False

    mesh_patterns = [
        r'gmsh\.model\.mesh\.generate',
        r'\.generate\(',
        r'mesh\.generate',
    ]

    for pattern in mesh_patterns:
        if re.search(pattern, script, re.IGNORECASE):
            return True

    return False


def compute_score(
    solution_str: str,
    ground_truth: str,
    mesh_stats: Optional[Dict] = None,
    extra_info: Optional[Dict] = None
) -> float:
    """
    Compute reward based on task completion.

    Scores what we can from both mesh_stats (if available) and script analysis.
    No early returns - always compute a score based on available information.

    Args:
        solution_str: Model's generated code
        ground_truth: Task description
        mesh_stats: Mesh analysis results from GmshTool (may be None)
        extra_info: Additional context including geometry metadata

    Returns:
        Score in range [-1.0, 1.0]
    """
    # Scoring components - initialize all to 0
    scores = {
        'mesh_generated': 0.0,
        'physical_groups': 0.0,
        'mesh_density': 0.0,
        'mesh_quality': 0.0
    }

    # Penalties for issues (subtracted from final score)
    penalties = 0.0

    # Extract geometry metadata
    geometry_metadata = {}
    if extra_info:
        geometry_metadata = extra_info.get('geometry_metadata', {})

    # Parse task requirements
    requirements = parse_task_requirements(ground_truth)

    # Check if we have valid mesh_stats
    has_mesh_stats = mesh_stats and 'error' not in mesh_stats

    # DEBUG: Log mesh_stats content
    print(f"[DEBUG SCORE] mesh_stats={mesh_stats}", flush=True)
    print(f"[DEBUG SCORE] has_mesh_stats={has_mesh_stats}, 'error' in mesh_stats={'error' in mesh_stats if mesh_stats else 'N/A'}", flush=True)

    if not mesh_stats:
        # No mesh_stats at all - apply penalty but continue scoring from script
        penalties += 0.3
    elif 'error' in mesh_stats:
        # Tool ran but errored - smaller penalty
        penalties += 0.2

    # 1. Mesh generation (30%)
    if has_mesh_stats:
        num_elements = mesh_stats.get('num_elements', 0)
        if num_elements > 0:
            scores['mesh_generated'] = 1.0
        else:
            # Mesh stats exist but no elements - check if script tried to generate
            if analyze_script_for_mesh_generation(solution_str):
                scores['mesh_generated'] = 0.3  # Tried but failed
    else:
        # No mesh stats - score based on script analysis
        if analyze_script_for_mesh_generation(solution_str):
            scores['mesh_generated'] = 0.4  # Has generate call, can't verify result

    # 2. Physical groups (30%)
    if has_mesh_stats:
        num_pg = mesh_stats.get('num_physical_groups', 0)
        min_pg = requirements['min_physical_groups']

        if num_pg >= min_pg:
            scores['physical_groups'] = 1.0
        elif num_pg > 0:
            scores['physical_groups'] = num_pg / max(min_pg, 1)
        else:
            # No physical groups in mesh - check script
            script_pg = analyze_script_for_physical_groups(solution_str)
            if script_pg > 0:
                scores['physical_groups'] = 0.3  # Script has PG code but mesh doesn't
    else:
        # No mesh stats - score based on script analysis
        script_pg = analyze_script_for_physical_groups(solution_str)
        min_pg = requirements['min_physical_groups']
        if script_pg >= min_pg:
            scores['physical_groups'] = 0.5  # Has enough PG calls, can't verify
        elif script_pg > 0:
            scores['physical_groups'] = 0.3 * (script_pg / max(min_pg, 1))

    # 3. Mesh density (20%)
    if has_mesh_stats:
        num_elements = mesh_stats.get('num_elements', 0)
        if num_elements > 0:
            element_size_hint = None
            if requirements['element_size_range']:
                min_size, max_size = requirements['element_size_range']
                element_size_hint = (min_size + max_size) / 2.0

            # Use geometry info from mesh_stats (computed from actual mesh bbox)
            # Fall back to geometry_metadata from extra_info if available
            mesh_geometry = {
                'volume': mesh_stats.get('volume', 0),
                'characteristic_length': mesh_stats.get('characteristic_length', 0)
            }
            # Use mesh-derived values if available, else try extra_info
            if not mesh_geometry['volume'] and geometry_metadata:
                mesh_geometry = geometry_metadata

            if mesh_geometry.get('volume'):
                scores['mesh_density'] = score_mesh_density(
                    num_elements,
                    mesh_geometry,
                    element_size_hint
                )
            else:
                # Can't evaluate density without geometry info - give neutral score
                scores['mesh_density'] = 0.7
    else:
        # No mesh stats - check if script has size control
        if analyze_script_for_refinement(solution_str):
            scores['mesh_density'] = 0.3  # Has size control code

    # 4. Mesh quality (20%)
    if has_mesh_stats:
        quality_min = mesh_stats.get('quality_min', 0) or 0
        quality_target = requirements.get('quality_target') or 0.3  # Default if None

        if quality_min > quality_target:
            scores['mesh_quality'] = 1.0
        elif quality_min > quality_target * 0.5:
            scores['mesh_quality'] = 0.5
        elif quality_min > 0:
            scores['mesh_quality'] = 0.2
    # If no mesh_stats, quality stays at 0 (can't analyze from script)

    # Weighted final score
    final_score = (
        0.30 * scores['mesh_generated'] +
        0.30 * scores['physical_groups'] +
        0.20 * scores['mesh_density'] +
        0.20 * scores['mesh_quality']
    )

    # Apply penalties
    final_score = max(0.0, final_score - penalties)

    # Map [0, 1] to [-1, 1] for compatibility with ADPO
    mapped_score = 2.0 * final_score - 1.0
    print(f"[DEBUG SCORE] scores={scores}, penalties={penalties}, final={final_score}, mapped={mapped_score}", flush=True)
    return mapped_score
