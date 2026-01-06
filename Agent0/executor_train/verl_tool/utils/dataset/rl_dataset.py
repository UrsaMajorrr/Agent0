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

## GMSH API Reference

### Initialization
```
import gmsh
gmsh.initialize()
gmsh.model.add("model_name")
```

### Loading STEP/IGES Files
```
# Import CAD geometry from STEP or IGES file
# Returns list of (dim, tag) tuples for imported entities
entities = gmsh.model.occ.importShapes("path/to/geometry.step")
gmsh.model.occ.synchronize()

# Get all volumes and surfaces after import
volumes = gmsh.model.getEntities(dim=3)  # [(3, vol_tag), ...]
surfaces = gmsh.model.getEntities(dim=2)  # [(2, surf_tag), ...]
```

### Geometry Creation (OCC kernel)
```
# Primitives - return volume tag
sphere_tag = gmsh.model.occ.addSphere(x, y, z, radius)
box_tag = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
cylinder_tag = gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, radius)
cone_tag = gmsh.model.occ.addCone(x, y, z, dx, dy, dz, r1, r2)
torus_tag = gmsh.model.occ.addTorus(x, y, z, r1, r2)
wedge_tag = gmsh.model.occ.addWedge(x, y, z, dx, dy, dz)  # triangular prism

# Boolean operations - dimTags format: [(dim, tag), ...]
gmsh.model.occ.fuse([(3, tag1)], [(3, tag2)])  # union
gmsh.model.occ.cut([(3, tag1)], [(3, tag2)])   # subtraction
gmsh.model.occ.intersect([(3, tag1)], [(3, tag2)])

# MUST synchronize after geometry operations
gmsh.model.occ.synchronize()
```

### Extrusion Operations
```
# Extrude surfaces to create volumes
# Returns list of (dim, tag) for created entities
new_entities = gmsh.model.occ.extrude([(2, surf_tag)], dx, dy, dz)
# new_entities[0] = extruded surface, new_entities[1] = created volume

# Revolve surface around axis (x,y,z) with direction (ax,ay,az)
new_entities = gmsh.model.occ.revolve([(2, surf_tag)], x, y, z, ax, ay, az, angle)

# Create pipe along wire path
gmsh.model.occ.addPipe([(2, profile_tag)], wire_tag)

# Loft between multiple wire profiles
gmsh.model.occ.addThruSections([wire1, wire2, wire3])
```

### Physical Groups (required for FEA)
```
# Get entities by dimension: 0=points, 1=curves, 2=surfaces, 3=volumes
volumes = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)

# Create physical groups
vol_group = gmsh.model.addPhysicalGroup(3, [tag1, tag2])
gmsh.model.setPhysicalName(3, vol_group, "material_domain")

surf_group = gmsh.model.addPhysicalGroup(2, [surf_tag])
gmsh.model.setPhysicalName(2, surf_group, "boundary_condition")
```

### Mesh Size Fields (Local Refinement)
```
# Distance field - refine near entity
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [surf_tag])

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
```
# Set number of nodes on curves
gmsh.model.mesh.setTransfiniteCurve(curve_tag, numNodes, meshType="Progression", coef=1.0)
# meshType: "Progression", "Bump" (refine at ends), "Beta" (refine at one end)

# Set transfinite surface (requires 3 or 4 corner points)
gmsh.model.mesh.setTransfiniteSurface(surf_tag, arrangement="Left", cornerTags=[p1, p2, p3, p4])
# arrangement: "Left", "Right", "Alternate"

# Set transfinite volume (requires 6 or 8 corner points for hex)
gmsh.model.mesh.setTransfiniteVolume(vol_tag, cornerTags=[p1, p2, p3, p4, p5, p6, p7, p8])

# Recombine triangles into quads (required for hex meshing)
gmsh.model.mesh.setRecombine(2, surf_tag)  # 2D recombine
gmsh.model.mesh.setRecombine(3, vol_tag)   # 3D recombine for hex elements
```

### Mesh Algorithms
```
# 2D algorithms: 1=MeshAdapt, 2=Auto, 3=Initial mesh only, 5=Delaunay, 6=Frontal-Delaunay
#                7=BAMG, 8=Frontal-Delaunay for Quads, 9=Packing of Parallelograms, 11=Quasi-Structured Quad
gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (good quality)

# 3D algorithms: 1=Delaunay, 3=Initial mesh only, 4=Frontal, 7=MMG3D, 10=HXT
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (fast, parallel)

# Optimization algorithms
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Netgen optimizer
gmsh.option.setNumber("Mesh.Optimize", 1)  # Standard optimization
```

### Boundary Layer Meshing
```
# Create boundary layer field
gmsh.model.mesh.field.add("BoundaryLayer", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [surf1, surf2])  # surfaces to extrude from
gmsh.model.mesh.field.setNumber(1, "Size", 0.01)       # first layer thickness
gmsh.model.mesh.field.setNumber(1, "Ratio", 1.2)       # growth ratio
gmsh.model.mesh.field.setNumber(1, "NbLayers", 10)     # number of layers
gmsh.model.mesh.field.setNumber(1, "Quads", 1)         # use quads (1) or triangles (0)
gmsh.model.mesh.field.setAsBoundaryLayer(1)

# Alternative: ExtrudeBoundaryLayer for explicit control
gmsh.model.geo.extrudeBoundaryLayer([(2, surf_tag)], [1, 2, 3], [0.01, 0.02, 0.05], True)
```

### Mesh Quality Options
```
# Element quality thresholds
gmsh.option.setNumber("Mesh.QualityType", 2)  # 0=SICN, 1=SIGE, 2=Gamma (default)
gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)  # optimize elements below this quality

# Mesh smoothing iterations
gmsh.option.setNumber("Mesh.Smoothing", 10)

# High-order mesh options
gmsh.option.setNumber("Mesh.ElementOrder", 2)  # quadratic elements
gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)  # optimize high-order mesh

# Mesh size from curvature
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)  # elements per 2*pi radians
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # extend size from boundary

# Anisotropic mesh options
gmsh.option.setNumber("Mesh.AnisoMax", 10)  # max anisotropy ratio
gmsh.option.setNumber("Mesh.SmoothRatio", 1.8)  # smoothing ratio
```

### Basic Mesh Control
```
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
gmsh.model.mesh.generate(3)  # 3D mesh
```

### Finalization
```
gmsh.write("output.msh")
gmsh.finalize()
```

## Example: Cube with refined region near hole
```
import gmsh
gmsh.initialize()
gmsh.model.add("refined_cube")

# Create cube with cylindrical hole
box = gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
cyl = gmsh.model.occ.addCylinder(5, 5, 0, 0, 0, 10, 2)
gmsh.model.occ.cut([(3, box)], [(3, cyl)])
gmsh.model.occ.synchronize()

# Get hole surface for refinement
surfs = gmsh.model.getEntities(dim=2)
hole_surfs = [s[1] for s in surfs if gmsh.model.isInside(2, s[1], [5, 5, 5])]

# Distance field from hole
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", hole_surfs if hole_surfs else [surfs[0][1]])

# Threshold: fine near hole, coarse away
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.3)
gmsh.model.mesh.field.setNumber(2, "SizeMax", 2.0)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)
gmsh.model.mesh.field.setNumber(2, "DistMax", 3.0)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# Physical groups
vols = gmsh.model.getEntities(dim=3)
surfs = gmsh.model.getEntities(dim=2)
gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], name="domain")
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], name="boundary")

# Quality meshing
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.model.mesh.generate(3)
gmsh.write("output.msh")
gmsh.finalize()
```

Wrap your code in ```python ... ``` blocks."""

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