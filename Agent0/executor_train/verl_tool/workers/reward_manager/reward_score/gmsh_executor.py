#!/usr/bin/env python
"""
GMSH Executor Reward - Task Completion Matching

Evaluates whether generated mesh addresses task requirements.
Uses geometry-aware scoring that accounts for geometry size to avoid
rewarding over-refinement or under-refinement.

Includes comprehensive refinement scoring for:
- Distance fields
- Threshold fields
- Boundary layer meshing
- Transfinite meshing
- Curvature-based sizing
- Box/Cylinder fields
- Anisotropic meshing
- Element order (quadratic elements)
"""

import re
from typing import Dict, List, Optional, Tuple

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


def parse_refinement_requirements(task: str) -> Dict:
    """
    Extract detailed refinement requirements from task description.

    Returns:
        Dict with refinement types and their parameters:
        - distance_field: List of (target_size, region_description) tuples
        - threshold_field: List of (target_size, region_description) tuples
        - boundary_layer: Dict with layers, thickness, growth_rate
        - transfinite: Dict with aspect_ratio, surfaces
        - curvature_based: bool
        - box_field: List of (size, region) tuples
        - anisotropic: bool
        - quadratic_elements: bool
        - global_size: float or None
        - local_sizes: List of (size, region) tuples
    """
    task_lower = task.lower()

    refinement = {
        'distance_field': [],
        'threshold_field': [],
        'boundary_layer': None,
        'transfinite': None,
        'curvature_based': False,
        'box_field': [],
        'anisotropic': False,
        'quadratic_elements': False,
        'global_size': None,
        'local_sizes': [],
    }

    # Distance field patterns
    # "refine to 0.5 near X using a Distance field"
    # "using a Distance field with a threshold of 0.1"
    distance_patterns = [
        r'refine[d]?\s+(?:the\s+mesh\s+)?to\s+(\d+\.?\d*)\s+near\s+([^.]+?)(?:\s+using\s+a\s+distance\s+field)',
        r'(?:local\s+)?refinement\s+(?:to\s+)?(\d+\.?\d*)\s+(?:near|around)\s+([^.]+?)(?:\s+using\s+a\s+distance\s+field)',
        r'distance\s+field\s+(?:with\s+)?(?:a\s+)?(?:value\s+of\s+|threshold\s+of\s+)?(\d+\.?\d*)\s+near\s+([^.]+)',
        r'using\s+a\s+distance\s+field.*?(\d+\.?\d*)',
    ]

    for pattern in distance_patterns:
        matches = re.findall(pattern, task_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                size = float(match[0])
                region = match[1].strip() if len(match) > 1 else "unspecified"
                refinement['distance_field'].append((size, region))
            elif isinstance(match, str):
                refinement['distance_field'].append((float(match), "unspecified"))

    # Also check for simple "Distance field" mention without parsed values
    if not refinement['distance_field'] and 'distance field' in task_lower:
        refinement['distance_field'].append((None, "unspecified"))

    # Threshold field patterns
    threshold_patterns = [
        r'(?:local\s+)?refinement\s+(?:to\s+)?(\d+\.?\d*)\s+(?:near|around)\s+([^.]+?)(?:\s+using\s+a\s+threshold\s+field)',
        r'threshold\s+field\s+(?:with\s+)?(?:a\s+)?(?:size\s+of\s+)?(\d+\.?\d*)',
        r'threshold\s+refinement\s+of\s+(\d+\.?\d*)',
    ]

    for pattern in threshold_patterns:
        matches = re.findall(pattern, task_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 1:
                size = float(match[0])
                region = match[1].strip() if len(match) > 1 else "unspecified"
                refinement['threshold_field'].append((size, region))
            elif isinstance(match, str):
                refinement['threshold_field'].append((float(match), "unspecified"))

    if not refinement['threshold_field'] and 'threshold field' in task_lower:
        refinement['threshold_field'].append((None, "unspecified"))

    # Boundary layer patterns
    # "boundary layer mesh with 6 layers and a thickness of 0.15"
    # "boundary layer with 5 layers and a growth rate of 1.2"
    bl_patterns = [
        r'boundary\s+layer.*?(\d+)\s+layers?.*?(?:thickness|height)\s+(?:of\s+)?(\d+\.?\d*)',
        r'boundary\s+layer.*?(\d+)\s+layers?.*?growth\s+rate\s+(?:of\s+)?(\d+\.?\d*)',
        r'boundary\s+layer.*?(\d+)\s+layers?',
    ]

    for pattern in bl_patterns:
        match = re.search(pattern, task_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            bl_info = {'layers': int(groups[0])}
            if len(groups) > 1:
                # Could be thickness or growth_rate depending on pattern
                if 'thickness' in pattern or 'height' in pattern:
                    bl_info['thickness'] = float(groups[1])
                else:
                    bl_info['growth_rate'] = float(groups[1])
            refinement['boundary_layer'] = bl_info
            break

    # Also check for simple boundary layer mention
    if not refinement['boundary_layer'] and 'boundary layer' in task_lower:
        refinement['boundary_layer'] = {'layers': None}

    # Transfinite patterns
    # "transfinite mesh with an 8:1 aspect ratio"
    transfinite_patterns = [
        r'transfinite\s+mesh.*?(\d+):(\d+)\s+aspect\s+ratio',
        r'transfinite\s+mesh',
        r'transfinite\s+meshing',
    ]

    for pattern in transfinite_patterns:
        match = re.search(pattern, task_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            trans_info = {}
            if groups and len(groups) >= 2:
                trans_info['aspect_ratio'] = f"{groups[0]}:{groups[1]}"
            refinement['transfinite'] = trans_info
            break

    # Curvature-based sizing
    if 'curvature-based' in task_lower or 'curvature based' in task_lower:
        refinement['curvature_based'] = True

    # Box field patterns
    box_patterns = [
        r'box\s+(?:field\s+)?refinement\s+(?:of\s+)?(\d+\.?\d*)',
        r'box\s+field.*?(\d+\.?\d*)',
    ]

    for pattern in box_patterns:
        matches = re.findall(pattern, task_lower, re.IGNORECASE)
        for match in matches:
            refinement['box_field'].append((float(match), "unspecified"))

    # Anisotropic meshing
    if 'anisotropic' in task_lower:
        refinement['anisotropic'] = True

    # Quadratic elements / element order
    if any(kw in task_lower for kw in ['quadratic element', 'element order', 'second order', '2nd order']):
        refinement['quadratic_elements'] = True

    # Global element size
    global_size_patterns = [
        r'global\s+(?:element\s+)?size\s+(?:of\s+)?(\d+\.?\d*)',
        r'global\s+mesh\s+size\s+(?:of\s+)?(\d+\.?\d*)',
        r'use\s+(?:a\s+)?global\s+(?:element\s+)?size\s+(?:of\s+)?(\d+\.?\d*)',
    ]

    for pattern in global_size_patterns:
        match = re.search(pattern, task_lower, re.IGNORECASE)
        if match:
            refinement['global_size'] = float(match.group(1))
            break

    # Local sizes (generic "refine to X near Y")
    local_size_patterns = [
        r'refine[d]?\s+(?:the\s+mesh\s+)?(?:to\s+)?(\d+\.?\d*)\s+near\s+([^,.]+)',
        r'(?:local\s+)?refinement\s+(?:to\s+)?(\d+\.?\d*)\s+(?:near|around)\s+([^,.]+)',
        r'mesh\s+size\s+(?:of\s+)?(\d+\.?\d*)\s+(?:near|for)\s+([^,.]+)',
    ]

    for pattern in local_size_patterns:
        matches = re.findall(pattern, task_lower, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                size = float(match[0])
                region = match[1].strip()
                refinement['local_sizes'].append((size, region))

    return refinement


def analyze_script_refinement_details(script: str) -> Dict:
    """
    Analyze the script to detect specific refinement implementations.

    Returns:
        Dict with detected refinement types:
        - distance_field: bool (and count)
        - threshold_field: bool (and count)
        - boundary_layer: bool
        - transfinite: bool
        - curvature_based: bool
        - box_field: bool
        - anisotropic: bool
        - quadratic_elements: bool
        - has_mesh_fields: bool
        - has_size_control: bool
    """
    if not script:
        return {
            'distance_field': False,
            'threshold_field': False,
            'boundary_layer': False,
            'transfinite': False,
            'curvature_based': False,
            'box_field': False,
            'anisotropic': False,
            'quadratic_elements': False,
            'has_mesh_fields': False,
            'has_size_control': False,
        }

    result = {}

    # Distance field detection
    distance_patterns = [
        r'field\.add\s*\(\s*["\']Distance["\']',
        r'addField\s*\(\s*["\']Distance["\']',
        r'gmsh\.model\.mesh\.field\.add\s*\(\s*["\']Distance["\']',
        r'["\']Distance["\']\s*\)',
    ]
    result['distance_field'] = any(re.search(p, script, re.IGNORECASE) for p in distance_patterns)

    # Threshold field detection
    threshold_patterns = [
        r'field\.add\s*\(\s*["\']Threshold["\']',
        r'addField\s*\(\s*["\']Threshold["\']',
        r'gmsh\.model\.mesh\.field\.add\s*\(\s*["\']Threshold["\']',
        r'["\']Threshold["\']\s*\)',
    ]
    result['threshold_field'] = any(re.search(p, script, re.IGNORECASE) for p in threshold_patterns)

    # Boundary layer detection
    bl_patterns = [
        r'field\.add\s*\(\s*["\']BoundaryLayer["\']',
        r'addField\s*\(\s*["\']BoundaryLayer["\']',
        r'BoundaryLayer',
        r'boundary.*layer',
        r'setNumber.*NbLayers',
        r'setNumber.*hwall',
        r'setNumber.*ratio',
    ]
    result['boundary_layer'] = any(re.search(p, script, re.IGNORECASE) for p in bl_patterns)

    # Transfinite detection
    transfinite_patterns = [
        r'setTransfiniteCurve',
        r'setTransfiniteSurface',
        r'setTransfiniteVolume',
        r'transfinite',
        r'Transfinite',
    ]
    result['transfinite'] = any(re.search(p, script, re.IGNORECASE) for p in transfinite_patterns)

    # Curvature-based detection
    curvature_patterns = [
        r'MeshSizeFromCurvature',
        r'CharacteristicLengthFromCurvature',
        r'Curvature',
        r'curvature',
        r'field\.add\s*\(\s*["\']Curvature["\']',
    ]
    result['curvature_based'] = any(re.search(p, script, re.IGNORECASE) for p in curvature_patterns)

    # Box field detection
    box_patterns = [
        r'field\.add\s*\(\s*["\']Box["\']',
        r'addField\s*\(\s*["\']Box["\']',
        r'["\']Box["\']\s*\)',
    ]
    result['box_field'] = any(re.search(p, script, re.IGNORECASE) for p in box_patterns)

    # Anisotropic detection
    aniso_patterns = [
        r'anisotropic',
        r'Anisotropic',
        r'setAniso',
    ]
    result['anisotropic'] = any(re.search(p, script, re.IGNORECASE) for p in aniso_patterns)

    # Quadratic elements / element order detection
    quad_patterns = [
        r'setOrder\s*\(\s*2',
        r'ElementOrder.*2',
        r'Mesh\.ElementOrder.*2',
        r'quadratic',
        r'second.*order',
    ]
    result['quadratic_elements'] = any(re.search(p, script, re.IGNORECASE) for p in quad_patterns)

    # General mesh field usage
    field_patterns = [
        r'field\.add',
        r'addField',
        r'gmsh\.model\.mesh\.field',
        r'setAsBackgroundMesh',
        r'setBackgroundField',
    ]
    result['has_mesh_fields'] = any(re.search(p, script, re.IGNORECASE) for p in field_patterns)

    # General size control
    size_patterns = [
        r'setSize',
        r'MeshSize',
        r'CharacteristicLength',
        r'lc\s*=',
        r'mesh_size',
    ]
    result['has_size_control'] = any(re.search(p, script, re.IGNORECASE) for p in size_patterns)

    return result


def score_refinement(task_refinement: Dict, script_refinement: Dict) -> Tuple[float, Dict]:
    """
    Score how well the script's refinement matches task requirements.

    Args:
        task_refinement: Output from parse_refinement_requirements()
        script_refinement: Output from analyze_script_refinement_details()

    Returns:
        Tuple of (score, details_dict)
        - score: float in [0, 1]
        - details_dict: breakdown of scoring
    """
    details = {}
    scores = []
    weights = []

    # Count how many refinement types are required
    required_types = []

    # Distance field scoring
    if task_refinement['distance_field']:
        required_types.append('distance_field')
        if script_refinement['distance_field']:
            details['distance_field'] = 1.0
        elif script_refinement['has_mesh_fields']:
            details['distance_field'] = 0.3  # Has fields but not the right type
        else:
            details['distance_field'] = 0.0
        scores.append(details['distance_field'])
        weights.append(1.5)  # Higher weight for Distance field (most common)

    # Threshold field scoring
    if task_refinement['threshold_field']:
        required_types.append('threshold_field')
        if script_refinement['threshold_field']:
            details['threshold_field'] = 1.0
        elif script_refinement['has_mesh_fields']:
            details['threshold_field'] = 0.3
        else:
            details['threshold_field'] = 0.0
        scores.append(details['threshold_field'])
        weights.append(1.0)

    # Boundary layer scoring
    if task_refinement['boundary_layer']:
        required_types.append('boundary_layer')
        if script_refinement['boundary_layer']:
            details['boundary_layer'] = 1.0
        else:
            details['boundary_layer'] = 0.0
        scores.append(details['boundary_layer'])
        weights.append(1.2)

    # Transfinite scoring
    if task_refinement['transfinite']:
        required_types.append('transfinite')
        if script_refinement['transfinite']:
            details['transfinite'] = 1.0
        else:
            details['transfinite'] = 0.0
        scores.append(details['transfinite'])
        weights.append(1.0)

    # Curvature-based scoring
    if task_refinement['curvature_based']:
        required_types.append('curvature_based')
        if script_refinement['curvature_based']:
            details['curvature_based'] = 1.0
        else:
            details['curvature_based'] = 0.0
        scores.append(details['curvature_based'])
        weights.append(0.8)

    # Box field scoring
    if task_refinement['box_field']:
        required_types.append('box_field')
        if script_refinement['box_field']:
            details['box_field'] = 1.0
        elif script_refinement['has_mesh_fields']:
            details['box_field'] = 0.3
        else:
            details['box_field'] = 0.0
        scores.append(details['box_field'])
        weights.append(0.8)

    # Anisotropic scoring
    if task_refinement['anisotropic']:
        required_types.append('anisotropic')
        if script_refinement['anisotropic']:
            details['anisotropic'] = 1.0
        else:
            details['anisotropic'] = 0.0
        scores.append(details['anisotropic'])
        weights.append(0.6)

    # Quadratic elements scoring
    if task_refinement['quadratic_elements']:
        required_types.append('quadratic_elements')
        if script_refinement['quadratic_elements']:
            details['quadratic_elements'] = 1.0
        else:
            details['quadratic_elements'] = 0.0
        scores.append(details['quadratic_elements'])
        weights.append(0.8)

    # Calculate weighted score
    if scores:
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:
        # No refinement requirements - give full score if has any size control
        if script_refinement['has_size_control'] or script_refinement['has_mesh_fields']:
            weighted_score = 1.0
        else:
            weighted_score = 0.5  # Neutral if no requirements and no control

    details['required_types'] = required_types
    details['num_required'] = len(required_types)
    details['num_matched'] = sum(1 for t in required_types if details.get(t, 0) >= 0.5)

    return weighted_score, details


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
        'mesh_quality': 0.0,
        'refinement': 0.0
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

    # 1. Mesh generation (25%)
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

    # 2. Physical groups (25%)
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

    # 3. Mesh density (15%)
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

    # 4. Mesh quality (15%)
    if has_mesh_stats:
        quality_min = mesh_stats.get('quality_min', 0) or 0
        quality_target = requirements.get('quality_target') or 0.3  # Default if None

        if quality_min > quality_target:
            scores['mesh_quality'] = 1.0
        elif quality_min > quality_target * 0.5:
            scores['mesh_quality'] = 0.5
        elif quality_min > 0:
            scores['mesh_quality'] = 0.2

    # 5. Refinement scoring (20%)
    # Parse detailed refinement requirements from task
    refinement_requirements = parse_refinement_requirements(ground_truth)

    # Analyze script for refinement implementation
    script_refinement = analyze_script_refinement_details(solution_str)

    # Score refinement match
    refinement_score, refinement_details = score_refinement(
        refinement_requirements, script_refinement
    )
    scores['refinement'] = refinement_score

    # Weighted final score
    # Adjusted weights: mesh_generated=25%, physical_groups=25%, mesh_density=15%, mesh_quality=15%, refinement=20%
    final_score = (
        0.25 * scores['mesh_generated'] +
        0.25 * scores['physical_groups'] +
        0.15 * scores['mesh_density'] +
        0.15 * scores['mesh_quality'] +
        0.20 * scores['refinement']
    )

    # Apply penalties
    final_score = max(0.0, final_score - penalties)

    # Map [0, 1] to [-1, 1] for compatibility with ADPO
    mapped_score = 2.0 * final_score - 1.0
    print(f"[DEBUG SCORE] scores={scores}, penalties={penalties}, final={final_score}, mapped={mapped_score}", flush=True)
    return mapped_score
