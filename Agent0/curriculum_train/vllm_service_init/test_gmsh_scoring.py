#!/usr/bin/env python
"""
Standalone test for GMSH mesh analysis and scoring.

Run this to test the mesh analysis pipeline without the full training:
    python test_gmsh_scoring.py

Or test with a specific script file:
    python test_gmsh_scoring.py --script /path/to/script.py
"""

import argparse
import tempfile
import os
import subprocess
import time
import re
import numpy as np

# Copy the functions from start_vllm_server_gmsh.py for testing
TIMEOUT = 120

def analyze_mesh(mesh_file: str):
    """Analyze generated mesh and extract quality metrics."""
    print(f"[analyze_mesh] Opening mesh file: {mesh_file}")
    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mesh_file)

        stats = {
            "num_nodes": 0,
            "num_elements": 0,
            "num_physical_groups": 0,
            "element_types": {},
            "quality_min": 0.0,
            "quality_mean": 0.0
        }

        # Count nodes
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats["num_nodes"] = len(node_tags)

        # Count elements
        elem_types = gmsh.model.mesh.getElementTypes()
        total_elements = 0
        for elem_type in elem_types:
            elem_tags, _ = gmsh.model.mesh.getElementsByType(elem_type)
            count = len(elem_tags)
            total_elements += count
            stats["element_types"][str(elem_type)] = count

        stats["num_elements"] = total_elements

        # Count physical groups
        for dim in range(4):
            groups = gmsh.model.getPhysicalGroups(dim)
            stats["num_physical_groups"] += len(groups)

        # Quality metrics (for tetrahedra, type 4)
        if 4 in elem_types:
            try:
                # Get element tags for tetrahedra first
                tet_tags, _ = gmsh.model.mesh.getElementsByType(4)
                if len(tet_tags) > 0:
                    # Get scaled Jacobian quality
                    qualities = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
                    if len(qualities) > 0:
                        stats["quality_min"] = float(np.min(qualities))
                        stats["quality_mean"] = float(np.mean(qualities))
            except Exception as e:
                print(f"[analyze_mesh] Quality extraction failed: {e}")
                pass

        gmsh.finalize()
        print(f"[analyze_mesh] Mesh stats: {stats}")
        return stats

    except Exception as e:
        print(f"[analyze_mesh] ERROR: {e}")
        return {
            "num_nodes": 0,
            "num_elements": 0,
            "num_physical_groups": 0,
            "error": str(e)
        }


def execute_gmsh_script(code: str, timeout: int = TIMEOUT):
    """Execute Gmsh script in isolated environment and analyze results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            script_path = os.path.join(tmpdir, "mesh_script.py")
            output_mesh = os.path.join(tmpdir, "output.msh")

            # Ensure script saves to our output path
            if "gmsh.write(" not in code:
                code = code.replace("gmsh.finalize()", f'gmsh.write("{output_mesh}")\ngmsh.finalize()')
            else:
                code = re.sub(r'gmsh\.write\(["\'].*?["\']\)', f'gmsh.write("{output_mesh}")', code)

            with open(script_path, 'w') as f:
                f.write(code)

            print(f"[execute_gmsh_script] Running script...")
            start_time = time.time()
            result = subprocess.run(
                ['python', script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            print(f"[execute_gmsh_script] Execution time: {execution_time:.2f}s")
            print(f"[execute_gmsh_script] Return code: {result.returncode}")

            if result.stdout:
                print(f"[execute_gmsh_script] STDOUT:\n{result.stdout[:500]}")
            if result.stderr:
                print(f"[execute_gmsh_script] STDERR:\n{result.stderr[:500]}")

            # Check if mesh was generated
            if os.path.exists(output_mesh):
                print(f"[execute_gmsh_script] Mesh file exists: {output_mesh}")
                print(f"[execute_gmsh_script] Mesh file size: {os.path.getsize(output_mesh)} bytes")
                mesh_stats = analyze_mesh(output_mesh)
                return {
                    "status": "success",
                    "mesh_generated": True,
                    "execution_time": execution_time,
                    "mesh_stats": mesh_stats,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"[execute_gmsh_script] No mesh file generated!")
                return {
                    "status": "error",
                    "mesh_generated": False,
                    "execution_time": execution_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error": "No mesh file generated"
                }

        except subprocess.TimeoutExpired:
            print(f"[execute_gmsh_script] Timeout!")
            return {
                "status": "error",
                "mesh_generated": False,
                "error": "Execution timeout"
            }
        except Exception as e:
            print(f"[execute_gmsh_script] Exception: {e}")
            return {
                "status": "error",
                "mesh_generated": False,
                "error": str(e)
            }


def calculate_script_score(eval_result: dict) -> float:
    """Calculate quality score for a single script execution."""
    print(f"\n[calculate_script_score] eval_result keys: {list(eval_result.keys())}")
    print(f"[calculate_script_score] mesh_generated: {eval_result.get('mesh_generated', False)}")

    if not eval_result.get("mesh_generated", False):
        print(f"[calculate_script_score] Returning 0.0 - no mesh generated")
        return 0.0

    mesh_stats = eval_result.get("mesh_stats", {})
    print(f"[calculate_script_score] mesh_stats: {mesh_stats}")
    score = 0.0

    # Execution success (mesh generated)
    score += 0.4
    print(f"  +0.4 for mesh generated (total: {score})")

    # Physical groups created
    num_pg = mesh_stats.get("num_physical_groups", 0)
    if num_pg > 0:
        pg_score = 0.2 * min(num_pg / 3.0, 1.0)
        score += pg_score
        print(f"  +{pg_score:.2f} for {num_pg} physical groups (total: {score})")
    else:
        print(f"  +0.0 for physical groups (none found)")

    # Reasonable mesh size
    num_elements = mesh_stats.get("num_elements", 0)
    if 100 <= num_elements <= 500000:
        score += 0.2
        print(f"  +0.2 for {num_elements} elements in range [100, 500000] (total: {score})")
    elif num_elements > 0:
        score += 0.1
        print(f"  +0.1 for {num_elements} elements (outside ideal range) (total: {score})")
    else:
        print(f"  +0.0 for elements (count: {num_elements})")

    # Element quality
    quality_min = mesh_stats.get("quality_min", 0)
    if quality_min > 0.3:
        score += 0.2
        print(f"  +0.2 for quality_min={quality_min:.3f} > 0.3 (total: {score})")
    elif quality_min > 0.1:
        score += 0.1
        print(f"  +0.1 for quality_min={quality_min:.3f} > 0.1 (total: {score})")
    else:
        print(f"  +0.0 for quality (quality_min={quality_min:.3f})")

    print(f"\n[calculate_script_score] FINAL SCORE: {score}")
    return score


# Sample test scripts
SAMPLE_SCRIPTS = {
    "box_with_groups": '''
import gmsh

gmsh.initialize()
gmsh.model.add("box_mesh")

# Create a 10x10x10 box
box = gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
gmsh.model.occ.synchronize()

# Add physical groups
gmsh.model.addPhysicalGroup(2, [1], name="bottom_face")
gmsh.model.addPhysicalGroup(2, [2], name="top_face")
gmsh.model.addPhysicalGroup(3, [1], name="volume")

# Set mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

gmsh.write("output.msh")
gmsh.finalize()
''',

    "box_no_groups": '''
import gmsh

gmsh.initialize()
gmsh.model.add("box_mesh")

# Create a 10x10x10 box
box = gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
gmsh.model.occ.synchronize()

# No physical groups!

# Set mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

gmsh.write("output.msh")
gmsh.finalize()
''',

    "sphere_with_groups": '''
import gmsh

gmsh.initialize()
gmsh.model.add("sphere_mesh")

# Create a sphere
sphere = gmsh.model.occ.addSphere(0, 0, 0, 5)
gmsh.model.occ.synchronize()

# Add physical groups
gmsh.model.addPhysicalGroup(2, [1], name="surface")
gmsh.model.addPhysicalGroup(3, [1], name="volume")

# Set mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)

# Generate 3D mesh
gmsh.model.mesh.generate(3)

gmsh.write("output.msh")
gmsh.finalize()
''',

    "broken_script": '''
import gmsh
gmsh.initialize()
# This will fail - no geometry created
gmsh.model.mesh.generate(3)
gmsh.finalize()
'''
}


def main():
    parser = argparse.ArgumentParser(description="Test GMSH mesh analysis and scoring")
    parser.add_argument("--script", type=str, help="Path to a script file to test")
    parser.add_argument("--sample", type=str, choices=list(SAMPLE_SCRIPTS.keys()),
                        help="Use a built-in sample script")
    parser.add_argument("--all", action="store_true", help="Run all sample scripts")
    args = parser.parse_args()

    if args.script:
        print(f"\n{'='*60}")
        print(f"Testing script from file: {args.script}")
        print('='*60)
        with open(args.script, 'r') as f:
            code = f.read()
        print(f"Script:\n{code[:500]}...")
        result = execute_gmsh_script(code)
        score = calculate_script_score(result)
        print(f"\n>>> FINAL SCORE: {score}")

    elif args.sample:
        print(f"\n{'='*60}")
        print(f"Testing sample: {args.sample}")
        print('='*60)
        code = SAMPLE_SCRIPTS[args.sample]
        print(f"Script:\n{code}")
        result = execute_gmsh_script(code)
        score = calculate_script_score(result)
        print(f"\n>>> FINAL SCORE: {score}")

    elif args.all:
        results = {}
        for name, code in SAMPLE_SCRIPTS.items():
            print(f"\n{'='*60}")
            print(f"Testing sample: {name}")
            print('='*60)
            result = execute_gmsh_script(code)
            score = calculate_script_score(result)
            results[name] = score
            print(f"\n>>> SCORE for {name}: {score}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        for name, score in results.items():
            print(f"  {name}: {score}")

    else:
        # Default: run box_with_groups
        print("No arguments provided. Running 'box_with_groups' sample.")
        print("Use --help for options, --all to run all samples.\n")

        print(f"{'='*60}")
        print("Testing sample: box_with_groups")
        print('='*60)
        code = SAMPLE_SCRIPTS["box_with_groups"]
        print(f"Script:\n{code}")
        result = execute_gmsh_script(code)
        score = calculate_script_score(result)
        print(f"\n>>> FINAL SCORE: {score}")


if __name__ == "__main__":
    main()
