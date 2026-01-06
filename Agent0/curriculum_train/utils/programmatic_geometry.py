#!/usr/bin/env python
"""
Programmatic Geometry Generator for Curriculum Training

Generates simple Gmsh geometry code (primitives) instead of loading complex STEP files.
This allows the model to learn on achievable tasks before tackling complex geometries.
"""

import random
from typing import Dict, List, Tuple


# Analysis types the curriculum agent can choose from
ANALYSIS_TYPES = [
    "static structural",
    "thermal",
    "modal",
    "buckling",
    "fatigue",
    "dynamic"
]


class ProgrammaticGeometryGenerator:
    """
    Generates simple programmatic geometry code for Gmsh.

    Provides geometry code snippets and metadata that the curriculum agent
    can use to create meshing tasks.
    """

    # Primitive geometry templates
    # Each template has:
    #   - name: Human-readable name
    #   - code_template: Gmsh Python code with placeholders
    #   - param_ranges: Valid ranges for each parameter
    #   - expected_surfaces: Number of surfaces in the geometry
    #   - expected_volumes: Number of volumes
    #   - description: What this geometry represents

    TEMPLATES = [
        {
            "name": "box",
            "code_template": """# Create a simple box
box = gmsh.model.occ.addBox({x}, {y}, {z}, {dx}, {dy}, {dz})
gmsh.model.occ.synchronize()""",
            "param_generator": lambda: {
                "x": 0, "y": 0, "z": 0,
                "dx": round(random.uniform(5, 30), 1),
                "dy": round(random.uniform(5, 30), 1),
                "dz": round(random.uniform(5, 30), 1)
            },
            "expected_surfaces": 6,
            "expected_volumes": 1,
            "description": "rectangular block"
        },
        {
            "name": "cylinder",
            "code_template": """# Create a cylinder
cylinder = gmsh.model.occ.addCylinder({x}, {y}, {z}, {dx}, {dy}, {dz}, {r})
gmsh.model.occ.synchronize()""",
            "param_generator": lambda: {
                "x": 0, "y": 0, "z": 0,
                "dx": 0, "dy": 0,
                "dz": round(random.uniform(10, 40), 1),  # Height along Z
                "r": round(random.uniform(3, 15), 1)
            },
            "expected_surfaces": 3,  # Top, bottom, lateral
            "expected_volumes": 1,
            "description": "cylindrical shaft"
        },
        {
            "name": "sphere",
            "code_template": """# Create a sphere
sphere = gmsh.model.occ.addSphere({x}, {y}, {z}, {r})
gmsh.model.occ.synchronize()""",
            "param_generator": lambda: {
                "x": 0, "y": 0, "z": 0,
                "r": round(random.uniform(5, 20), 1)
            },
            "expected_surfaces": 1,
            "expected_volumes": 1,
            "description": "spherical body"
        },
        {
            "name": "cone",
            "code_template": """# Create a cone
cone = gmsh.model.occ.addCone({x}, {y}, {z}, {dx}, {dy}, {dz}, {r1}, {r2})
gmsh.model.occ.synchronize()""",
            "param_generator": lambda: {
                "x": 0, "y": 0, "z": 0,
                "dx": 0, "dy": 0,
                "dz": round(random.uniform(10, 30), 1),  # Height
                "r1": round(random.uniform(5, 15), 1),   # Bottom radius
                "r2": round(random.uniform(1, 5), 1)     # Top radius (smaller)
            },
            "expected_surfaces": 3,  # Base, top, lateral
            "expected_volumes": 1,
            "description": "conical shape"
        },
        {
            "name": "wedge",
            "code_template": """# Create a wedge
wedge = gmsh.model.occ.addWedge({x}, {y}, {z}, {dx}, {dy}, {dz})
gmsh.model.occ.synchronize()""",
            "param_generator": lambda: {
                "x": 0, "y": 0, "z": 0,
                "dx": round(random.uniform(10, 30), 1),
                "dy": round(random.uniform(10, 30), 1),
                "dz": round(random.uniform(5, 20), 1)
            },
            "expected_surfaces": 5,
            "expected_volumes": 1,
            "description": "wedge-shaped solid"
        },
        {
            "name": "torus",
            "code_template": """# Create a torus (donut shape)
torus = gmsh.model.occ.addTorus({x}, {y}, {z}, {R}, {r})
gmsh.model.occ.synchronize()""",
            "param_generator": lambda: {
                "x": 0, "y": 0, "z": 0,
                "R": round(random.uniform(10, 25), 1),  # Major radius
                "r": round(random.uniform(2, 8), 1)     # Minor radius (tube)
            },
            "expected_surfaces": 1,
            "expected_volumes": 1,
            "description": "toroidal ring"
        },
    ]

    def __init__(self, seed: int = None):
        """
        Initialize generator with optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def generate(self) -> Dict:
        """
        Generate a random geometry with code and metadata.

        Returns:
            Dict with:
                - geometry_code: Gmsh Python code to create the geometry
                - geometry_name: Name of the geometry type
                - geometry_description: Human-readable description
                - expected_surfaces: Number of surfaces
                - expected_volumes: Number of volumes
                - parameters: Dict of parameter values used
                - characteristic_length: Suggested mesh size
                - analysis_types: List of suitable analysis types
        """
        template = random.choice(self.TEMPLATES)
        params = template["param_generator"]()

        # Generate the code
        geometry_code = template["code_template"].format(**params)

        # Calculate characteristic length based on geometry size
        if template["name"] == "box":
            char_length = min(params["dx"], params["dy"], params["dz"]) / 5
        elif template["name"] == "cylinder":
            char_length = min(params["r"], params["dz"]) / 5
        elif template["name"] == "sphere":
            char_length = params["r"] / 5
        elif template["name"] == "cone":
            char_length = min(params["r1"], params["dz"]) / 5
        elif template["name"] == "wedge":
            char_length = min(params["dx"], params["dy"], params["dz"]) / 5
        elif template["name"] == "torus":
            char_length = params["r"] / 3
        else:
            char_length = 2.0

        return {
            "geometry_code": geometry_code,
            "geometry_name": template["name"],
            "geometry_description": template["description"],
            "expected_surfaces": template["expected_surfaces"],
            "expected_volumes": template["expected_volumes"],
            "parameters": params,
            "characteristic_length": round(char_length, 2),
            "analysis_types": ANALYSIS_TYPES
        }

    def generate_batch(self, count: int) -> List[Dict]:
        """
        Generate multiple geometries.

        Args:
            count: Number of geometries to generate

        Returns:
            List of geometry dicts
        """
        return [self.generate() for _ in range(count)]


def format_geometry_context(geometry: Dict) -> str:
    """
    Format geometry metadata into context string for the curriculum agent.

    Args:
        geometry: Dict from ProgrammaticGeometryGenerator.generate()

    Returns:
        Formatted context string
    """
    params_str = ", ".join([f"{k}={v}" for k, v in geometry["parameters"].items()])

    context = (
        f"**GEOMETRY INFORMATION:**\n"
        f"Type: {geometry['geometry_name']} ({geometry['geometry_description']})\n"
        f"Parameters: {params_str}\n"
        f"Surfaces: {geometry['expected_surfaces']}, Volumes: {geometry['expected_volumes']}\n"
        f"Characteristic length: {geometry['characteristic_length']}\n"
        f"\n"
        f"**GEOMETRY CREATION CODE:**\n"
        f"```python\n"
        f"{geometry['geometry_code']}\n"
        f"```"
    )

    return context


if __name__ == "__main__":
    # Test the generator
    gen = ProgrammaticGeometryGenerator(seed=42)

    print("Testing ProgrammaticGeometryGenerator...")
    print("=" * 60)

    for i in range(3):
        geom = gen.generate()
        print(f"\nGeometry {i+1}: {geom['geometry_name']}")
        print(f"Description: {geom['geometry_description']}")
        print(f"Surfaces: {geom['expected_surfaces']}, Volumes: {geom['expected_volumes']}")
        print(f"Characteristic length: {geom['characteristic_length']}")
        print(f"Code:\n{geom['geometry_code']}")
        print("-" * 40)
        print("Formatted context:")
        print(format_geometry_context(geom))
        print("=" * 60)
