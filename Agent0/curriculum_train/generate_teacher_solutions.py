#!/usr/bin/env python
"""
Generate Teacher Solutions using GPT-5.2 for GMSH Curriculum Tasks

This script implements "pre-generation distillation":
1. Loads curriculum tasks from parquet/json
2. Calls GPT-5.2 API for each task with GMSH system prompt
3. Validates generated scripts by executing them
4. Saves (task, teacher_solution) pairs for training

The teacher solutions serve as high-quality examples that Qwen can learn from
during RL training via in-context learning.

Usage:
    export OPENAI_API_KEY="your-key"
    python generate_teacher_solutions.py \
        --input_path /path/to/tasks.parquet \
        --output_path /path/to/teacher_solutions.parquet \
        --model gpt-4o \
        --num_samples 3
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, List, Any

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("Please install pyarrow: pip install pyarrow")
    sys.exit(1)

from tqdm import tqdm

# System prompt for GPT - teaches correct GMSH patterns
GMSH_SYSTEM_PROMPT = """You are an expert in computational meshing using Gmsh.
Write a complete, self-contained Python script using the Gmsh library.

Requirements:
1. Start with `import gmsh` and `gmsh.initialize()`
2. Create geometry using `gmsh.model.occ` API (addSphere, addCylinder, addBox, etc.)
3. Call `gmsh.model.occ.synchronize()` after geometry creation
4. Create physical groups with `gmsh.model.addPhysicalGroup()` and `gmsh.model.setPhysicalName()`
5. Set mesh size with `gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)` or any other mesh refinement strategy like Distance and Threshold fields
6. Generate mesh with `gmsh.model.mesh.generate(3)` for 3D or gmsh.model.mesh.generate(2) for 2D
7. Save with `gmsh.write("output.msh")`
8. End with `gmsh.finalize()`

Important API notes:
- addSphere(x, y, z, radius) - creates sphere at (x,y,z) with given radius
- addCylinder(x, y, z, dx, dy, dz, radius) - cylinder from (x,y,z) along direction (dx,dy,dz)
- addBox(x, y, z, dx, dy, dz) - box from corner (x,y,z) with dimensions (dx,dy,dz)
- For boolean operations: cut([(3,obj1)], [(3,obj2)]), fuse(), intersect()
- Get entities: gmsh.model.getEntities(dim=2) for surfaces, dim=3 for volumes

Output ONLY the Python code wrapped in ```python ... ``` blocks. No explanations."""


def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from markdown code blocks."""
    # Try ```python ... ```
    matches = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try ``` ... ```
    matches = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if 'import gmsh' in code:
            return code

    # Fallback: find code starting with import gmsh
    match = re.search(r'(import gmsh.*?gmsh\.finalize\(\))', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def validate_gmsh_script(code: str, timeout: int = 60) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute GMSH script and validate it produces a mesh.

    Returns:
        (success, stats_dict)
    """
    if not code or 'import gmsh' not in code:
        return False, {"error": "No valid gmsh code"}

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "test_script.py")
        output_mesh = os.path.join(tmpdir, "output.msh")

        # Ensure script writes to our output path
        if "gmsh.write(" in code:
            code = re.sub(r'gmsh\.write\(["\'].*?["\']\)',
                         f'gmsh.write("{output_mesh}")', code)
        else:
            if "gmsh.finalize()" in code:
                code = code.replace("gmsh.finalize()",
                    f'gmsh.write("{output_mesh}")\ngmsh.finalize()')
            else:
                code += f'\ngmsh.write("{output_mesh}")\ngmsh.finalize()'

        with open(script_path, 'w') as f:
            f.write(code)

        try:
            result = subprocess.run(
                ['python', script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if os.path.exists(output_mesh) and os.path.getsize(output_mesh) > 100:
                # Try to get mesh stats
                stats = get_mesh_stats(output_mesh)
                if stats.get("num_elements", 0) > 0:
                    return True, stats

            return False, {
                "error": result.stderr[:500] if result.stderr else "No mesh generated",
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return False, {"error": "Timeout"}
        except Exception as e:
            return False, {"error": str(e)}


def get_mesh_stats(mesh_file: str) -> Dict[str, Any]:
    """Get mesh statistics by running gmsh in a subprocess to avoid state pollution."""
    import json

    # Run stats extraction in subprocess to avoid gmsh state issues
    stats_script = f'''
import gmsh
import json
try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open("{mesh_file}")

    stats = {{"num_nodes": 0, "num_elements": 0, "num_physical_groups": 0}}

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    stats["num_nodes"] = len(node_tags)

    elem_types = gmsh.model.mesh.getElementTypes()
    total = 0
    for et in elem_types:
        tags, _ = gmsh.model.mesh.getElementsByType(et)
        total += len(tags)
    stats["num_elements"] = total

    for dim in range(4):
        groups = gmsh.model.getPhysicalGroups(dim)
        stats["num_physical_groups"] += len(groups)

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


def generate_solution(
    client: OpenAI,
    task: str,
    model: str,
    max_retries: int = 3,
    num_samples: int = 1,
    verbose: bool = True
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Generate and validate a GMSH solution using GPT.

    Args:
        client: OpenAI client
        task: Task description
        model: Model name (e.g., "gpt-4o", "gpt-4-turbo")
        max_retries: Number of retry attempts per sample
        num_samples: Number of samples to generate (picks first working one)
        verbose: Print debug info

    Returns:
        (working_code, mesh_stats) or (None, None) if all fail
    """
    last_error = None
    last_validation_error = None

    for sample_idx in range(num_samples):
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": GMSH_SYSTEM_PROMPT},
                        {"role": "user", "content": task}
                    ],
                    temperature=0.7 if sample_idx > 0 else 0.3,  # Lower temp for first try
                )

                content = response.choices[0].message.content
                code = extract_python_code(content)

                if code:
                    success, stats = validate_gmsh_script(code)
                    if success:
                        return code, stats
                    else:
                        last_validation_error = stats.get("error", "Unknown validation error")
                        if verbose:
                            print(f"  [Validation failed] {last_validation_error[:100]}", flush=True)
                            # Show first few lines of generated code for debugging
                            code_preview = '\n'.join(code.split('\n')[:5])
                            print(f"  [Code preview] {code_preview[:200]}...", flush=True)
                else:
                    if verbose:
                        print(f"  [No code extracted] Response length: {len(content) if content else 0}", flush=True)
                        if content:
                            print(f"  [Response preview] {content[:200]}...", flush=True)

            except Exception as e:
                last_error = str(e)
                if verbose:
                    print(f"  [API Error] {last_error[:100]}", flush=True)
                if "rate_limit" in str(e).lower():
                    time.sleep(5 * (retry + 1))  # Exponential backoff
                continue

    if verbose and (last_error or last_validation_error):
        print(f"  [Final failure] API: {last_error}, Validation: {last_validation_error}", flush=True)

    return None, None


def process_task(
    args_tuple: Tuple[int, str, OpenAI, str, int, int]
) -> Optional[Dict[str, Any]]:
    """Process a single task (for parallel execution)."""
    idx, task, client, model, max_retries, num_samples = args_tuple

    try:
        code, stats = generate_solution(client, task, model, max_retries, num_samples)

        if code:
            return {
                "task": task,
                "teacher_solution": code,
                "mesh_stats": stats,
                "success": True
            }
        print(f"[Task {idx}] No valid code generated", flush=True)
        return None
    except Exception as e:
        print(f"[Task {idx}] Exception: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate teacher solutions using GPT")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input tasks (parquet or json)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save output parquet")
    parser.add_argument("--model", type=str, default="gpt-5.2",
                       help="OpenAI model to use")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to try per task")
    parser.add_argument("--max_retries", type=int, default=2,
                       help="Max retries per sample")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--task_key", type=str, default="task",
                       help="Key for task field in input data")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks to process")
    args = parser.parse_args()

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load input tasks
    print(f"Loading tasks from: {args.input_path}")
    if args.input_path.endswith('.parquet'):
        table = pq.read_table(args.input_path)
        df = table.to_pandas()
        tasks = df[args.task_key].tolist()
    elif args.input_path.endswith('.json'):
        with open(args.input_path) as f:
            data = json.load(f)
        tasks = [item.get(args.task_key) or item.get('question', '') for item in data]
    else:
        print(f"Unsupported file format: {args.input_path}")
        sys.exit(1)

    # Filter empty tasks
    tasks = [t for t in tasks if t and len(t.strip()) > 10]

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Processing {len(tasks)} tasks with model {args.model}")

    # Process tasks
    results = []
    failed_count = 0

    # Prepare arguments for parallel processing
    task_args = [
        (i, task, client, args.model, args.max_retries, args.num_samples)
        for i, task in enumerate(tasks)
    ]

    # Process with progress bar
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_task, arg): arg[0] for arg in task_args}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_count += 1
    else:
        # Sequential processing
        for arg in tqdm(task_args, desc="Generating"):
            result = process_task(arg)
            if result:
                results.append(result)
            else:
                failed_count += 1

    print(f"\nSuccess: {len(results)}/{len(tasks)} ({100*len(results)/len(tasks):.1f}%)")
    print(f"Failed: {failed_count}")

    if not results:
        print("No successful results to save")
        sys.exit(1)

    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert to parquet
    output_data = {
        "task": [r["task"] for r in results],
        "teacher_solution": [r["teacher_solution"] for r in results],
        "num_elements": [r["mesh_stats"].get("num_elements", 0) for r in results],
        "num_nodes": [r["mesh_stats"].get("num_nodes", 0) for r in results],
        "num_physical_groups": [r["mesh_stats"].get("num_physical_groups", 0) for r in results],
    }

    table = pa.table(output_data)
    pq.write_table(table, args.output_path)
    print(f"Saved {len(results)} teacher solutions to: {args.output_path}")

    # Also save as JSON for inspection
    json_path = args.output_path.replace('.parquet', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Also saved JSON to: {json_path}")


if __name__ == "__main__":
    main()
