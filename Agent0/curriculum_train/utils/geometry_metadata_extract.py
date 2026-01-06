import gmsh
import numpy as np

# Geometry metadata extraction
def extract_geometry_metadata(step_file_path: str) -> dict:
    """Extract geometry features that LLM can use for meshing"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # Suppress output
    gmsh.merge(step_file_path)
    gmsh.model.occ.synchronize()

    # Get entities
    points = gmsh.model.get_entities(0)
    curves = gmsh.model.get_entities(1)
    surfaces = gmsh.model.get_entities(2)
    volumes = gmsh.model.get_entities(3)

    metadata = {
        "file": step_file_path,
        "points": len(points),
        "curves": len(curves),
        "surfaces": len(surfaces),
        "volumes": len(volumes),
        "entity_types": {},  # {entity_id: type (cylinder, plane, etc.)}
        "bounding_boxes": {},  # {entity_id: (xmin, ymin, zmin, xmax, ymax, zmax)}
    }

    # Get overall bounding box
    # For overall bbox, use gmsh.model.getBoundingBox (no .occ) with dim=-1, tag=-1
    overall_bbox = gmsh.model.getBoundingBox(-1, -1)
    metadata["overall_bbox"] = {
        "xmin": overall_bbox[0], "ymin": overall_bbox[1], "zmin": overall_bbox[2],
        "xmax": overall_bbox[3], "ymax": overall_bbox[4], "zmax": overall_bbox[5],
        "width": overall_bbox[3] - overall_bbox[0],
        "height": overall_bbox[4] - overall_bbox[1],
        "depth": overall_bbox[5] - overall_bbox[2]
    }

    # Characteristic length (average of bbox dimensions)
    char_length = (metadata["overall_bbox"]["width"] +
                   metadata["overall_bbox"]["height"] +
                   metadata["overall_bbox"]["depth"]) / 3.0
    metadata["characteristic_length"] = char_length

    # Classify surfaces and collect sizes
    surface_sizes = []
    surface_type_counts = {}

    for surf in surfaces:
        # Get entity type using correct Gmsh API
        entity_type = gmsh.model.getType(surf[0], surf[1])
        bbox = gmsh.model.occ.get_bounding_box(surf[0], surf[1])
        metadata["entity_types"][surf[1]] = entity_type
        metadata["bounding_boxes"][surf[1]] = bbox

        # Calculate characteristic size of this surface
        width = bbox[3] - bbox[0]
        height = bbox[4] - bbox[1]
        depth = bbox[5] - bbox[2]
        char_size = max(width, height, depth)  # Largest dimension
        surface_sizes.append(char_size)

        # Count entity types
        surface_type_counts[entity_type] = surface_type_counts.get(entity_type, 0) + 1

    # Classify curves and collect sizes
    curve_sizes = []
    curve_type_counts = {}

    for curve in curves:
        # Get entity type using correct Gmsh API
        entity_type = gmsh.model.getType(curve[0], curve[1])
        bbox = gmsh.model.occ.get_bounding_box(curve[0], curve[1])

        # Calculate length estimate
        width = bbox[3] - bbox[0]
        height = bbox[4] - bbox[1]
        depth = bbox[5] - bbox[2]
        length = max(width, height, depth)
        curve_sizes.append(length)

        curve_type_counts[entity_type] = curve_type_counts.get(entity_type, 0) + 1

    # Compute size statistics
    if surface_sizes:
        metadata["surface_size_stats"] = {
            "min": float(np.min(surface_sizes)),
            "max": float(np.max(surface_sizes)),
            "mean": float(np.mean(surface_sizes)),
            "std": float(np.std(surface_sizes)),
            "median": float(np.median(surface_sizes))
        }
    else:
        metadata["surface_size_stats"] = {}

    if curve_sizes:
        metadata["curve_size_stats"] = {
            "min": float(np.min(curve_sizes)),
            "max": float(np.max(curve_sizes)),
            "mean": float(np.mean(curve_sizes)),
            "std": float(np.std(curve_sizes)),
            "median": float(np.median(curve_sizes))
        }
    else:
        metadata["curve_size_stats"] = {}

    # Entity type distributions
    metadata["surface_type_distribution"] = surface_type_counts
    metadata["curve_type_distribution"] = curve_type_counts

    # Multi-scale ratio (indicates if geometry has features at very different scales)
    if surface_sizes and len(surface_sizes) > 1:
        metadata["multi_scale_ratio"] = float(np.max(surface_sizes) / np.min(surface_sizes))
    else:
        metadata["multi_scale_ratio"] = 1.0

    # Complexity score (heuristic based on entity counts and type diversity)
    num_surface_types = len(surface_type_counts)
    num_curve_types = len(curve_type_counts)
    total_entities = len(surfaces) + len(curves)

    complexity_score = (total_entities / 10.0) * (1 + 0.1 * (num_surface_types + num_curve_types))
    metadata["complexity_score"] = min(complexity_score, 100.0)  # Cap at 100

    # Suggested mesh sizes based on geometry scale
    if surface_sizes:
        # Suggested fine mesh size (for small features)
        metadata["suggested_mesh_size_min"] = float(np.min(surface_sizes) / 10.0)
        # Suggested coarse mesh size (for large features)
        metadata["suggested_mesh_size_max"] = float(char_length / 5.0)
    else:
        metadata["suggested_mesh_size_min"] = char_length / 100.0
        metadata["suggested_mesh_size_max"] = char_length / 5.0

    gmsh.finalize()
    return metadata
