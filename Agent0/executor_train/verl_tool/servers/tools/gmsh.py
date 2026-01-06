#!/usr/bin/env python
"""
GMSH Tool - Executes Gmsh meshing scripts in isolated sandbox

This tool executes Gmsh Python scripts, analyzes the generated mesh,
and returns mesh quality statistics for reward computation.
"""

import ray
from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import uuid
import shutil
import resource
import tempfile
import json
from typing import Tuple, Dict, Any, Optional, Union, List

# Timeout for GMSH execution in seconds
TIMEOUT = 120

def set_limits():
    """Resource limits for GMSH execution"""
    # Memory limit (8GB)
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, resource.RLIM_INFINITY))
    # CPU time
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT, resource.RLIM_INFINITY))
    # File size limit (500MB)
    resource.setrlimit(resource.RLIMIT_FSIZE, (500*1024*1024, 500*1024*1024))

def analyze_mesh(mesh_file: str) -> Dict:
    """
    Analyze mesh quality metrics using Gmsh API in subprocess.

    Runs in subprocess to avoid signal/thread issues with gmsh.

    Args:
        mesh_file: Path to .msh file

    Returns:
        Dict with mesh statistics
    """
    # Run analysis in subprocess to avoid thread/signal conflicts
    stats_script = f'''
import gmsh
import json
try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open("{mesh_file}")

    stats = {{
        "num_nodes": 0,
        "num_elements": 0,
        "num_physical_groups": 0,
        "element_types": {{}},
        "quality_min": 0.0,
        "quality_mean": 0.0,
        "bbox": None,
        "volume": 0.0,
        "characteristic_length": 0.0
    }}

    # Get bounding box from all entities
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        stats["bbox"] = [xmin, ymin, zmin, xmax, ymax, zmax]
        # Estimate volume from bbox
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        stats["volume"] = dx * dy * dz
        stats["characteristic_length"] = max(dx, dy, dz)
    except:
        pass

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    stats["num_nodes"] = len(node_tags)

    elem_types = gmsh.model.mesh.getElementTypes()
    total_elements = 0
    for elem_type in elem_types:
        elem_tags, _ = gmsh.model.mesh.getElementsByType(elem_type)
        count = len(elem_tags)
        total_elements += count
        stats["element_types"][str(elem_type)] = count

    stats["num_elements"] = total_elements

    for dim in range(4):
        groups = gmsh.model.getPhysicalGroups(dim)
        stats["num_physical_groups"] += len(groups)

    if 4 in elem_types:
        try:
            tet_tags, _ = gmsh.model.mesh.getElementsByType(4)
            if len(tet_tags) > 0:
                qualities = gmsh.model.mesh.getElementQualities(tet_tags, "minSJ")
                if len(qualities) > 0:
                    stats["quality_min"] = float(min(qualities))
                    stats["quality_mean"] = float(sum(qualities) / len(qualities))
        except:
            pass

    gmsh.finalize()
    print(json.dumps(stats))
except Exception as e:
    try:
        gmsh.finalize()
    except:
        pass
    print(json.dumps({{"error": str(e)}}))
'''

    try:
        result = subprocess.run(
            ['python', '-c', stats_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
        return {"error": result.stderr[:200] if result.stderr else "No output"}
    except Exception as e:
        return {"error": str(e)}

def execute_gmsh_script(
    code: str,
    timeout: int = TIMEOUT
) -> Tuple[str, str, bool, Dict]:
    """
    Execute self-contained GMSH script in sandbox and analyze results

    Scripts create geometry programmatically (no external files needed).

    Args:
        code: Gmsh Python script (self-contained with programmatic geometry)
        timeout: Execution timeout in seconds

    Returns:
        (stdout, stderr, has_error, mesh_stats)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            script_path = os.path.join(tmpdir, "mesh_script.py")
            output_mesh = os.path.join(tmpdir, "output.msh")

            # Ensure script saves mesh to our output path
            if "gmsh.write(" not in code:
                if "gmsh.finalize()" in code:
                    code = code.replace("gmsh.finalize()",
                        f'gmsh.write("{output_mesh}")\ngmsh.finalize()')
                else:
                    code = code + f'\ngmsh.write("{output_mesh}")\ngmsh.finalize()'
            else:
                # Replace existing gmsh.write() with our output path
                code = re.sub(r'gmsh\.write\(["\'].*?["\']\)',
                    f'gmsh.write("{output_mesh}")', code)

            # Write script to file
            with open(script_path, 'w') as f:
                f.write(code)

            # Execute with timeout and resource limits
            result = subprocess.run(
                ['python', script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=set_limits
            )

            stdout = result.stdout
            stderr = result.stderr
            has_error = bool(stderr) or result.returncode != 0

            # Analyze mesh if generated
            mesh_stats = {}
            if os.path.exists(output_mesh):
                mesh_stats = analyze_mesh(output_mesh)
            else:
                has_error = True
                if not stderr:
                    stderr = "No mesh file generated"

            return stdout, stderr, has_error, mesh_stats

        except subprocess.TimeoutExpired:
            return ("", f"Execution timed out after {timeout}s", True, {})
        except Exception as e:
            return ("", str(e), True, {})

@register_tool
class GmshTool(BaseTool):
    """
    Tool for executing Gmsh meshing scripts

    Parses Python code from model responses, executes it in a sandboxed
    environment with access to geometry files, and returns mesh statistics.
    """
    tool_type = "gmsh"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]

    def get_usage_inst(self):
        return "Execute Gmsh meshing scripts. Wrap code in ```python...``` blocks."

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Extract Python code from response

        Supports formats:
        1. ```python ... ```
        2. <python>...</python>
        3. Malformed: ```python ```import gmsh... (extra backticks)

        Args:
            action: Model's response string

        Returns:
            (parsed_code, is_valid)
        """
        # Try ```python...``` format first
        code_blocks = re.findall(r"```python(.*?)```", action, re.DOTALL)

        if not code_blocks:
            # Try <python>...</python> format
            code_blocks = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)

        # Filter out empty/whitespace-only blocks
        code_blocks = [block.strip() for block in code_blocks if block.strip()]

        if not code_blocks:
            # Fallback: look for gmsh code anywhere in the response
            # Handles malformed blocks like "```python ```import gmsh..."
            if "import gmsh" in action:
                # Find code from "import gmsh" to last gmsh.finalize() or end
                match = re.search(r'(import gmsh.*?gmsh\.finalize\(\))', action, re.DOTALL)
                if match:
                    return match.group(1).strip(), True
                # Try without finalize
                match = re.search(r'(import gmsh.*?)(?:```|$)', action, re.DOTALL)
                if match:
                    code = match.group(1).strip()
                    code = re.sub(r'```.*$', '', code).strip()
                    if len(code) > 50:  # Sanity check - code should be substantial
                        return code, True
            return "", False

        # Combine all code blocks
        parsed_code = "\n".join(code_blocks)
        return parsed_code, True

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute GMSH script and return observation

        Scripts are self-contained with programmatic geometry creation.

        Args:
            trajectory_id: Unique ID for this trajectory
            action: Model's response containing code
            extra_field: Dict with task context (geometry_code, etc.)

        Returns:
            (observation, done, valid)
        """
        # DEBUG: Log what we receive
        print(f"[DEBUG GMSH] trajectory_id={trajectory_id}", flush=True)
        print(f"[DEBUG GMSH] extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}", flush=True)
        print(f"[DEBUG GMSH] action length: {len(action) if action else 0}", flush=True)

        # Parse code from response
        parsed_action, is_valid = self.parse_action(action)
        print(f"[DEBUG GMSH] is_valid={is_valid}, parsed_action length: {len(parsed_action) if parsed_action else 0}", flush=True)

        if not is_valid:
            print(f"[DEBUG GMSH] EARLY RETURN: No valid Python code", flush=True)
            observation = {
                "obs": "",
                "invalid_reason": "No Python code found. Use ```python...``` blocks."
            }
            return observation, False, False

        # Execute self-contained script (no geometry file needed)
        print(f"[DEBUG GMSH] Executing self-contained script", flush=True)
        stdout, stderr, has_error, mesh_stats = execute_gmsh_script(
            parsed_action, self.timeout
        )
        print(f"[DEBUG GMSH] Execution done. has_error={has_error}, mesh_stats keys={list(mesh_stats.keys()) if mesh_stats else 'None'}", flush=True)

        # Format observation
        execution_result = stdout
        if stderr:
            execution_result += f"\nERROR:\n{stderr}"

        # Add mesh stats if available
        if mesh_stats and 'error' not in mesh_stats:
            stats_str = f"\n\nMesh Statistics:\n"
            stats_str += f"  Nodes: {mesh_stats.get('num_nodes', 0)}\n"
            stats_str += f"  Elements: {mesh_stats.get('num_elements', 0)}\n"
            stats_str += f"  Physical Groups: {mesh_stats.get('num_physical_groups', 0)}\n"
            if mesh_stats.get('quality_min'):
                stats_str += f"  Quality (min): {mesh_stats['quality_min']:.3f}\n"
                stats_str += f"  Quality (mean): {mesh_stats['quality_mean']:.3f}\n"
            execution_result += stats_str

        observation = {
            "obs": execution_result,
            "mesh_stats": mesh_stats,
            "has_error": has_error
        }

        # Format with output tags
        formatted_obs = self.postprocess_observation(action, observation)

        # Store mesh stats for reward computation
        # Load existing environment state
        env = self.load_env(trajectory_id)

        # Update environment with action and observation
        self.update_env(
            trajectory_id, env, parsed_action, is_valid,
            extra_field, formatted_obs,
            mesh_stats=mesh_stats  # Store mesh stats
        )

        # Save environment state
        self.save_env(trajectory_id, env)

        # Check if we should continue for multi-turn
        # extra_field contains 'turns_left' and 'max_turns' from manager
        turns_left = extra_field.get('turns_left', 0) if extra_field else 0
        done = (turns_left <= 0) or has_error

        print(f"[DEBUG GMSH] Returning: done={done}, is_valid={is_valid}, obs_type={type(formatted_obs)}, has_mesh_stats={'mesh_stats' in formatted_obs if isinstance(formatted_obs, dict) else False}", flush=True)
        return formatted_obs, done, is_valid

    def postprocess_observation(self, action, observation):
        """
        Format observation with appropriate tags

        Matches the format used in the action (```python -> ```output or <python> -> <output>)

        Args:
            action: Original action string
            observation: Observation dict or string

        Returns:
            Formatted observation string or dict
        """
        if isinstance(observation, dict):
            raw_obs = observation.get("obs", "")
        else:
            raw_obs = observation

        # Match format to action
        if "```python" in action:
            formatted = f"\n```output\n{raw_obs}\n```\n"
        else:
            formatted = f"\n<output>\n{raw_obs}\n</output>\n"

        if isinstance(observation, dict):
            result = observation.copy()
            result['obs'] = formatted
            return result

        return formatted
