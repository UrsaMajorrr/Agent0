#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
vLLM Server for Gmsh Curriculum Training

This server receives meshing tasks from the curriculum agent and uses student models
to generate Gmsh Python scripts for primitive geometries (box, cylinder, sphere, etc.).
Scripts are executed and evaluated for quality.

Setup Instructions:
    pip install flask vllm transformers torch gmsh

Usage:
    python start_vllm_server_gmsh.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
import re
from tqdm import tqdm
import tempfile
import subprocess
import numpy as np

# ---------------------------- Configuration --------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
args = parser.parse_args()

# ---------------------------- Model Setup --------------------------------- #

print('[init] Loading model...')
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
)

sampling_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
    n=1,
    stop_token_ids=[tokenizer.eos_token_id]
)

# System prompt for Gmsh meshing with primitives
SYSTEM_PROMPT = """You are an expert in computational meshing using Gmsh Python API.

Generate a complete, executable Python script that:
1. Imports gmsh and initializes it
2. Creates the geometry using Gmsh OCC primitives (addBox, addCylinder, addSphere, etc.)
3. Synchronizes with gmsh.model.occ.synchronize()
4. Creates appropriate physical groups as specified
5. Sets mesh size
6. Generates the 3D mesh with gmsh.model.mesh.generate(3)
7. Saves to output.msh and finalizes gmsh

Available OCC primitives:
- gmsh.model.occ.addBox(x, y, z, dx, dy, dz) - box
- gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r) - cylinder along axis
- gmsh.model.occ.addSphere(xc, yc, zc, r) - sphere
- gmsh.model.occ.addCone(x, y, z, dx, dy, dz, r1, r2) - cone/frustum
- gmsh.model.occ.addWedge(x, y, z, dx, dy, dz) - wedge
- gmsh.model.occ.addTorus(x, y, z, r1, r2) - torus

Wrap your complete script in ```python ... ```"""

# ---------------------------- GPU Idle Worker ------------------- #
stop_event = threading.Event()
pause_event = threading.Event()

def gpu_idle_worker():
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                running = False
            time.sleep(0.1)
            continue
        else:
            if not running:
                running = True
        try:
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError:
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# ---------------------------- Gmsh Execution ----------------------- #

def execute_gmsh_script(code: str, timeout: int = 120):
    """
    Execute Gmsh script in isolated environment and analyze results.

    Returns:
        {
            "status": "success" | "error",
            "mesh_generated": bool,
            "execution_time": float,
            "mesh_stats": {...},
            "stdout": str,
            "stderr": str
        }
    """
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

            # Execute with timeout
            start_time = time.time()
            result = subprocess.run(
                ['python', script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time

            # Check if mesh was generated
            if os.path.exists(output_mesh):
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
                return {
                    "status": "error",
                    "mesh_generated": False,
                    "execution_time": execution_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error": "No mesh file generated"
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "mesh_generated": False,
                "error": "Execution timeout"
            }
        except Exception as e:
            return {
                "status": "error",
                "mesh_generated": False,
                "error": str(e)
            }


def analyze_mesh(mesh_file: str):
    """
    Analyze generated mesh and extract quality metrics.

    Runs in a subprocess to avoid signal issues with gmsh in threaded Flask.
    """
    print(f"[analyze_mesh] Opening mesh file: {mesh_file}", flush=True)

    # Create a small script to analyze the mesh in a subprocess
    analysis_script = f'''
import json
import sys
import numpy as np

try:
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open("{mesh_file}")

    stats = {{
        "num_nodes": 0,
        "num_elements": 0,
        "num_physical_groups": 0,
        "element_types": {{}},
        "quality_min": 0.0,
        "quality_mean": 0.0
    }}

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
            tet_tags, _ = gmsh.model.mesh.getElementsByType(4)
            if len(tet_tags) > 0:
                qualities = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
                if len(qualities) > 0:
                    stats["quality_min"] = float(np.min(qualities))
                    stats["quality_mean"] = float(np.mean(qualities))
        except:
            pass

    gmsh.finalize()
    print(json.dumps(stats))
except Exception as e:
    print(json.dumps({{"num_nodes": 0, "num_elements": 0, "num_physical_groups": 0, "error": str(e)}}))
'''

    try:
        # Run analysis in subprocess to avoid signal issues in Flask threads
        result = subprocess.run(
            ['python', '-c', analysis_script],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Parse the JSON output
        if result.stdout.strip():
            stats = json.loads(result.stdout.strip().split('\n')[-1])
            print(f"[analyze_mesh] Mesh stats: {stats}", flush=True)
            return stats
        else:
            error_msg = result.stderr[:200] if result.stderr else "No output from analysis"
            print(f"[analyze_mesh] ERROR: {error_msg}", flush=True)
            return {
                "num_nodes": 0,
                "num_elements": 0,
                "num_physical_groups": 0,
                "error": error_msg
            }

    except subprocess.TimeoutExpired:
        print(f"[analyze_mesh] ERROR: Analysis timeout", flush=True)
        return {
            "num_nodes": 0,
            "num_elements": 0,
            "num_physical_groups": 0,
            "error": "Analysis timeout"
        }
    except Exception as e:
        print(f"[analyze_mesh] ERROR: {e}", flush=True)
        return {
            "num_nodes": 0,
            "num_elements": 0,
            "num_physical_groups": 0,
            "error": str(e)
        }


# ---------------------------- Core Logic ----------------------- #

def extract_script_from_response(response: str) -> str:
    """Extract Python script from model response."""
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try without newline after python
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return ""


def generate_scripts(tasks: list, num_candidates: int = 10):
    """
    Generate Gmsh scripts for given tasks using student model.

    Args:
        tasks: List of task description strings
        num_candidates: Number of script candidates to generate per task

    Returns:
        List of dicts with generated scripts and evaluation results
    """
    results = []

    for task in tqdm(tasks, desc="Processing tasks"):
        if not task:
            results.append({
                'task': task,
                'scripts': [],
                'evaluations': [],
                'best_score': 0.0
            })
            continue

        # Format prompt
        prompt = f"Task:\n{task}\n\nGenerate the complete Gmsh Python script:"

        # Create conversation
        conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ]

        # Format for model
        formatted_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate multiple candidates
        responses = model.generate(
            [formatted_prompt] * num_candidates,
            sampling_params,
            use_tqdm=False
        )

        # Extract scripts and evaluate
        scripts = []
        evaluations = []

        for response in responses:
            output = response.outputs[0].text.strip()
            script = extract_script_from_response(output)

            if script:
                eval_result = execute_gmsh_script(script)
                scripts.append(script)
                evaluations.append(eval_result)
            else:
                scripts.append("")
                evaluations.append({
                    "status": "error",
                    "mesh_generated": False,
                    "error": "No script extracted"
                })

        # Calculate best score
        best_score = 0.0
        for eval_result in evaluations:
            if eval_result.get("mesh_generated", False):
                score = calculate_script_score(eval_result)
                best_score = max(best_score, score)

        results.append({
            'task': task,
            'scripts': scripts,
            'evaluations': evaluations,
            'best_score': best_score
        })

    return results


def calculate_script_score(eval_result: dict) -> float:
    """Calculate quality score for a single script execution."""
    print(f"[calculate_script_score] eval_result keys: {list(eval_result.keys())}", flush=True)
    print(f"[calculate_script_score] mesh_generated: {eval_result.get('mesh_generated', False)}", flush=True)

    if not eval_result.get("mesh_generated", False):
        print(f"[calculate_script_score] Returning 0.0 - no mesh generated", flush=True)
        return 0.0

    mesh_stats = eval_result.get("mesh_stats", {})
    print(f"[calculate_script_score] mesh_stats: {mesh_stats}", flush=True)
    score = 0.0

    # Execution success (mesh generated)
    score += 0.4

    # Physical groups created
    if mesh_stats.get("num_physical_groups", 0) > 0:
        score += 0.2 * min(mesh_stats["num_physical_groups"] / 3.0, 1.0)

    # Reasonable mesh size
    num_elements = mesh_stats.get("num_elements", 0)
    if 100 <= num_elements <= 500000:
        score += 0.2
    elif num_elements > 0:
        score += 0.1

    # Element quality
    if mesh_stats.get("quality_min", 0) > 0.3:
        score += 0.2
    elif mesh_stats.get("quality_min", 0) > 0.1:
        score += 0.1

    print(f"[calculate_script_score] Final score: {score}", flush=True)
    return score


# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')

    # Load task data
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    # Extract tasks
    tasks = [item.get('task', '') for item in data]

    # Generate and evaluate scripts
    results = generate_scripts(tasks, num_candidates=10)

    # Format results
    results_all = []
    for result in results:
        results_all.append({
            'task': result['task'],
            'best_score': result['best_score'],
            'num_successful': sum(1 for e in result['evaluations'] if e.get('mesh_generated', False))
        })

    # Save results
    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})


# ------------------------- Main Application Entrypoint --------------------------- #
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        stop_event.set()
        if idle_thread.is_alive():
            idle_thread.join()
        print('[main] Application shutdown complete.')
