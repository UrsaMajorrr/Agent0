import io
import base64
import numpy as np
import regex as re
import time
import datasets
from verl.utils.dataset.rl_dataset import RLHFDataset
from pathlib import Path
from typing import List
from copy import deepcopy
from collections import defaultdict

def encode_image(img_path: str) -> str:
    with open(img_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_str = encoded_bytes.decode("utf-8")
        return encoded_str
    
def nested_copy(obj):
    """
    Recursively copy nested objects (lists, dicts, etc.) to avoid reference issues.
    """
    if isinstance(obj, dict):
        return {k: nested_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nested_copy(item) for item in obj]
    elif hasattr(obj, 'copy'):
        return obj.copy()
    else:
        return obj
    
class RolloutMessagesMixin:
    """Mixin class to handle rollout messages in reinforcement learning datasets.

    This mixin provides methods to update and manage rollout messages, which are used
    to store the conversation history and interactions during the reinforcement learning process.
    """
    def __init__(self, messages: List[dict]):
        self.messages = messages if messages is not None else []
    
    def update_rollout_messages(self, new_message: dict) -> List[dict]:
        """Update the rollout messages with new messages."""
        messages = self.messages
        role = new_message['role']
        content_list = new_message['content']
        if isinstance(content_list, str):
            content_list = [{"type": "text", "text": content_list}]
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        assert isinstance(content_list, list), f"content_list should be a list, but got {type(content_list)}"
        
        if messages[-1]['role'] != role:
            messages.append({'role': role, 'content': content_list})
        else:
            for content in content_list:
                if isinstance(content, dict) and content.get('type') == 'text' and messages[-1]['content'][-1].get('type') == 'text':
                    messages[-1]['content'][-1]['text'] += content['text']
                else:
                    messages[-1]['content'].append(content)
        return messages

    def tolist(self):
        """Convert the messages to a list format."""
        if isinstance(self.messages, list):
            return self.messages.copy()
        elif isinstance(self.messages, str):
            # Fallback for legacy data - wrap string in user message
            return [{"role": "user", "content": self.messages}]
        elif isinstance(self.messages, np.ndarray):
            return self.messages.tolist()
        else:
            # Final fallback
            return list(self.messages) if hasattr(self.messages, '__iter__') else [self.messages]
    
    def __copy__(self):
        """Create a shallow copy of the RolloutMessagesMixin instance."""
        return RolloutMessagesMixin(nested_copy(self.messages))
        
# System prompt for GMSH meshing tasks
GMSH_SYSTEM_PROMPT = """You are an expert in computational meshing using Gmsh.
Write a complete Python script using the Gmsh library to complete the meshing task.

The task will specify a STEP geometry file to load. Your script must:
1. Load the specified STEP file using gmsh.model.occ.importShapes()
2. Create the physical groups as specified in the task
3. Apply the mesh refinement settings as specified
4. Generate a quality 3D mesh

## GMSH API Reference

### Initialization and Loading STEP Files
```python
import gmsh
gmsh.initialize()
gmsh.model.add("model_name")

# Import CAD geometry from STEP file
entities = gmsh.model.occ.importShapes("path/to/geometry.step")
gmsh.model.occ.synchronize()

# Get all volumes and surfaces after import
volumes = gmsh.model.getEntities(dim=3)  # [(3, vol_tag), ...]
surfaces = gmsh.model.getEntities(dim=2)  # [(2, surf_tag), ...]
```

### Physical Groups (required for FEA)
```python
# Get entities by dimension: 0=points, 1=curves, 2=surfaces, 3=volumes
volumes = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)

# Create physical groups with meaningful names
vol_group = gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes])
gmsh.model.setPhysicalName(3, vol_group, "domain")

# Group specific surfaces by their tags
surf_group = gmsh.model.addPhysicalGroup(2, [surf_tag])
gmsh.model.setPhysicalName(2, surf_group, "boundary_name")
```

### Mesh Size Fields (Local Refinement)
```python
# Distance field - refine near surfaces
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [surf_tag1, surf_tag2])

# Threshold field - size based on distance
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.1)   # size at surface
gmsh.model.mesh.field.setNumber(2, "SizeMax", 1.0)   # size far away
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)   # distance for SizeMin
gmsh.model.mesh.field.setNumber(2, "DistMax", 5.0)   # distance for SizeMax

# Box field - refine in region
gmsh.model.mesh.field.add("Box", 3)
gmsh.model.mesh.field.setNumber(3, "VIn", 0.2)   # size inside box
gmsh.model.mesh.field.setNumber(3, "VOut", 1.0)  # size outside
gmsh.model.mesh.field.setNumber(3, "XMin", -1)
gmsh.model.mesh.field.setNumber(3, "XMax", 1)
gmsh.model.mesh.field.setNumber(3, "YMin", -1)
gmsh.model.mesh.field.setNumber(3, "YMax", 1)
gmsh.model.mesh.field.setNumber(3, "ZMin", -1)
gmsh.model.mesh.field.setNumber(3, "ZMax", 1)

# Combine fields with Min
gmsh.model.mesh.field.add("Min", 4)
gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
gmsh.model.mesh.field.setAsBackgroundMesh(4)
```

### Transfinite Meshing (Structured Meshes)
```python
# Set number of nodes on curves
gmsh.model.mesh.setTransfiniteCurve(curve_tag, numNodes, meshType="Progression", coef=1.0)

# Set transfinite surface (requires 3 or 4 corner points)
gmsh.model.mesh.setTransfiniteSurface(surf_tag, arrangement="Left", cornerTags=[p1, p2, p3, p4])

# Recombine triangles into quads
gmsh.model.mesh.setRecombine(2, surf_tag)
```

### Boundary Layer Meshing
```python
gmsh.model.mesh.field.add("BoundaryLayer", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [surf1, surf2])
gmsh.model.mesh.field.setNumber(1, "Size", 0.01)       # first layer thickness
gmsh.model.mesh.field.setNumber(1, "Ratio", 1.2)       # growth ratio
gmsh.model.mesh.field.setNumber(1, "NbLayers", 10)     # number of layers
gmsh.model.mesh.field.setNumber(1, "Quads", 1)         # use quads
gmsh.model.mesh.field.setAsBoundaryLayer(1)
```

### Mesh Options
```python
# Global mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)

# Mesh from curvature (elements per 2*pi radians)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

# Algorithm selection
gmsh.option.setNumber("Mesh.Algorithm", 6)      # 2D: Frontal-Delaunay
gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # 3D: HXT (fast)

# Quality optimization
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.option.setNumber("Mesh.Smoothing", 10)

# Element order
gmsh.option.setNumber("Mesh.ElementOrder", 1)   # 1=linear, 2=quadratic
```

### Finalization
```python
gmsh.model.mesh.generate(3)  # Generate 3D mesh
gmsh.write("output.msh")
gmsh.finalize()
```

## Example: Loading STEP file with refinement near cylindrical surfaces
```python
import gmsh
gmsh.initialize()
gmsh.model.add("thermal_mesh")

# Load the STEP geometry
gmsh.model.occ.importShapes("geometries/example_part.step")
gmsh.model.occ.synchronize()

# Get all entities
volumes = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)

# Create physical groups
vol_group = gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes])
gmsh.model.setPhysicalName(3, vol_group, "domain")

# Group all surfaces as boundary (or split by surface type if needed)
all_surf_tags = [s[1] for s in surfaces]
surf_group = gmsh.model.addPhysicalGroup(2, all_surf_tags)
gmsh.model.setPhysicalName(2, surf_group, "boundary")

# Distance field for refinement near specific surfaces
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", all_surf_tags[:3])  # refine near first 3 surfaces

# Threshold field
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.5)
gmsh.model.mesh.field.setNumber(2, "SizeMax", 2.0)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)
gmsh.model.mesh.field.setNumber(2, "DistMax", 5.0)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# Global mesh settings
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

# Generate mesh
gmsh.model.mesh.generate(3)
gmsh.write("output.msh")
gmsh.finalize()
```

Wrap your code in ```python ... ``` blocks."""

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

# System prompt for distillation mode (with teacher solutions)
GMSH_DISTILLATION_PROMPT = """You are an expert in computational meshing using Gmsh.
You will be shown a reference solution, then asked to complete a similar task.
Study the reference solution carefully - it demonstrates correct Gmsh API usage.

Your script must:
1. Initialize Gmsh with gmsh.initialize()
2. Load or create geometry:
   - For STEP files: gmsh.model.occ.importShapes("path/to/file.step")
   - For primitives: addSphere, addCylinder, addBox, etc.
3. Synchronize with gmsh.model.occ.synchronize()
4. Create physical groups with gmsh.model.addPhysicalGroup() and gmsh.model.setPhysicalName()
5. Set mesh sizes with gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
6. Generate the mesh with gmsh.model.mesh.generate(3)
7. Save with gmsh.write("output.msh") and finalize with gmsh.finalize()

Important API patterns:
- importShapes("file.step") - load STEP/IGES CAD geometry
- addSphere(x, y, z, radius) - sphere at (x,y,z)
- addCylinder(x, y, z, dx, dy, dz, radius) - cylinder from (x,y,z) along direction
- addBox(x, y, z, dx, dy, dz) - box from corner with dimensions
- Boolean ops: cut([(3,obj1)], [(3,obj2)]), fuse(), intersect()
- Get entities: gmsh.model.getEntities(dim=2) for surfaces, dim=3 for volumes

Wrap your code in ```python ... ``` blocks."""


def format_task_with_teacher(task: str, teacher_solution: str) -> str:
    """Format task prompt with teacher solution as reference."""
    return f"""Here is a reference solution demonstrating correct Gmsh usage:

```python
{teacher_solution}
```

Now complete this task using similar patterns:

{task}"""


class VerlToolRLHFDataset(RLHFDataset):
    """A dataset class for reinforcement learning tasks in verl-tool.

    This class extends the base RLHFDataset class to provide additional functionality
    specific to verl-tool, such as custom data loading and processing methods.

    Supports distillation mode: when data contains 'teacher_solution' field,
    the prompt includes the teacher's solution as a reference example.
    """

    def __init__(self, data_files, tokenizer, processor, config):
        # Get system_prompt - use GMSH prompt by default, or from config
        if config is None:
            config = {}

        # Check if distillation mode is enabled
        self.use_distillation = config.get("use_distillation", True)  # Auto-detect from data
        self.teacher_solution_key = config.get("teacher_solution_key", "teacher_solution")

        # Use distillation prompt if enabled, otherwise regular prompt
        default_prompt = GMSH_DISTILLATION_PROMPT if self.use_distillation else GMSH_SYSTEM_PROMPT
        self.system_prompt = config.get("system_prompt", default_prompt)

        # Parent class has different arg order: (data_files, tokenizer, config, processor)
        super().__init__(data_files, tokenizer, config, processor)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """

        row_dict: dict = self.dataframe[item]
        start = time.time()
        rollout_messages = self._build_rollout_messages(row_dict)
        # print(f'finish getting {item}-th item rollout messages in {time.time() - start} seconds')
        start = time.time()
        result = super().__getitem__(item)
        result['rollout_messages'] = rollout_messages
        # print(f'finish getting {item}-th item in {time.time() - start} seconds')

        extra_info = row_dict.get('extra_info')

        if isinstance(extra_info, dict) and 'score' in extra_info:
            result['score'] = float(extra_info['score'])
        else:
            result['score'] = float(0.6)

        # Add reward_model field if not present
        if 'reward_model' not in result:
            # Use 'answer' field as ground_truth if available
            ground_truth = row_dict.get('answer', row_dict.get('solution', ''))
            result['reward_model'] = {
                'ground_truth': ground_truth,
                'style': 'rule'
            }

        # Add data_source field if not present
        if 'data_source' not in result:
            # Use a default value or extract from extra_info if available
            # Default to 'gmsh_mesh' for GMSH executor training
            result['data_source'] = row_dict.get('data_source',
                                                 row_dict.get('extra_info', {}).get('data_source', 'gmsh_mesh'))

        return result
    
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]] if image_key in doc else None # changed to get images from doc
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]] if video_key in doc else None # changed to get videos from doc
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe
    
    def _build_messages(self, example: dict):
        """Override base class to handle both string and list prompts.

        When teacher_solution is available in the data, formats the prompt
        to include it as a reference example for the model to learn from.
        """
        prompt_data = example.pop(self.prompt_key)

        # Check for teacher solution (distillation mode)
        teacher_solution = example.get(self.teacher_solution_key, None)

        # Handle both raw strings and message lists
        if isinstance(prompt_data, str):
            # If we have a teacher solution, format the task to include it
            if self.use_distillation and teacher_solution:
                formatted_content = format_task_with_teacher(prompt_data, teacher_solution)
            else:
                formatted_content = prompt_data
            # Convert raw string to message format
            messages = [{"role": "user", "content": formatted_content}]
        elif isinstance(prompt_data, list):
            # Already in message format - inject teacher solution into last user message
            messages = deepcopy(prompt_data)
            if self.use_distillation and teacher_solution:
                # Find the last user message and augment it
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        original_content = messages[i]["content"]
                        if isinstance(original_content, str):
                            messages[i]["content"] = format_task_with_teacher(original_content, teacher_solution)
                        break
        else:
            raise ValueError(f"Unexpected prompt_key type: {type(prompt_data)}. Expected str or list.")

        # Prepend system prompt if configured and not already present
        if self.system_prompt and (not messages or messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Continue with multimodal processing from base class
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                try:
                    segments = re.split("(<image>|<video>)", content)
                except Exception as e:
                    raise ValueError(f"Error splitting content: {content}") from e
                segments = [item for item in segments if item != ""]
                segment_idx = defaultdict(int)
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                        segment_idx[segment] += 1
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                        segment_idx[segment] += 1
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def _build_rollout_messages(self, example: dict):
        """Build rollout messages with optional teacher solution.

        When teacher_solution is available, includes it as a reference example.
        """
        prompt_data = example[self.prompt_key]

        # Check for teacher solution (distillation mode)
        teacher_solution = example.get(self.teacher_solution_key, None)

        # Handle both raw strings and message lists
        if isinstance(prompt_data, str):
            # If we have a teacher solution, format the task to include it
            if self.use_distillation and teacher_solution:
                formatted_content = format_task_with_teacher(prompt_data, teacher_solution)
            else:
                formatted_content = prompt_data
            # Convert raw string to message format
            messages = [{"role": "user", "content": formatted_content}]
        elif isinstance(prompt_data, list):
            # Already in message format - inject teacher solution into last user message
            messages = deepcopy(prompt_data)
            if self.use_distillation and teacher_solution:
                # Find the last user message and augment it
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        original_content = messages[i]["content"]
                        if isinstance(original_content, str):
                            messages[i]["content"] = format_task_with_teacher(original_content, teacher_solution)
                        break
        else:
            raise ValueError(f"Unexpected prompt_key type: {type(prompt_data)}. Expected str or list.")

        # Prepend system prompt if configured and not already present
        if self.system_prompt and (not messages or messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                try:
                    segments = re.split("(<image>|<video>)", content)
                except Exception as e:
                    raise ValueError(f"Error splitting content: {content}") from e
                segments = [item for item in segments if item != ""]
                segment_idx = defaultdict(int)
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image", "image": example[self.image_key][segment_idx[segment]]["image"]})
                        segment_idx[segment] += 1
                    elif segment == "<video>":
                        content_list.append({"type": "video", "video": example[self.video_key][segment_idx[segment]]["video"]})
                        segment_idx[segment] += 1
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        if self.processor is not None:
            # multi-modal inputs
            from verl_tool.llm_agent.vision_utils import encode_image_url, encode_video_url
            for i, message in enumerate(messages):
                if isinstance(message['content'], list):
                    for j in range(len(message['content'])):
                        content = message['content'][j]
                        if content['type'] == 'image':
                            message['content'][j] = {
                                "type": "image_url",
                                "image_url": {
                                    "url": encode_image_url(content['image']),
                                }
                            }
                            assert Path(content['image']).exists(), f"Image file {content['image']} does not exist."
                        elif content['type'] == 'video':
                            message['content'][j] = {
                                "type": "video_url",
                                "video_url": {
                                    "url": encode_video_url(content['video']),
                                }
                            }
                            assert Path(content['video']).exists(), f"Video file {content['video']} does not exist."
                        elif content['type'] == 'text':
                            message['content'][j] = {
                                "type": "text",
                                "text": content['text']
                            }
                        else:
                            raise ValueError(f"Unknown content element type: {content['type']}")
                elif isinstance(message['content'], str):
                    message['content'] = [{"type": "text", "text": message['content']}]
                else:
                    raise ValueError(f"Unknown content type: {type(message['content'])}")
                    
        return RolloutMessagesMixin(messages)