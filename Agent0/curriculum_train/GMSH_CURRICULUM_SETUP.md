# Gmsh Curriculum Training System - Complete Setup

## Overview

This system trains a **curriculum agent** to generate meshing tasks, and a **student agent** to solve them by writing Gmsh Python scripts.

## Architecture

```
Curriculum Agent → Generates Tasks → Student Models → Generate Scripts
        ↑                                                      ↓
        |                                             Execute & Evaluate
        |                                                      ↓
        └──────────────── Reward Signal ←──────────────────────┘
```

## Components Created ✅

### 1. Geometry Metadata System
- **`utils/geometry_metadata_extract.py`** - Extract features from STEP files
- **`utils/batch_extract_metadata.py`** - Process all 55 geometries
- **`utils/metadata_all.json`** - Metadata database (generated)

**Key metadata fields:**
- `complexity_score` - Task difficulty indicator
- `multi_scale_ratio` - Determines refinement needs
- `suggested_mesh_size_min/max` - Auto-computed sizing
- `surface_type_distribution` - Entity type counts
- `characteristic_length` - Geometry scale

### 2. Reward Function
- **`examples/reward_function/gmsh_curriculum_reward.py`**

**Rewards curriculum agent for:**
- Valid task format (30%)
- Appropriate difficulty - not too easy/hard (50%)
- Task diversity - penalizes duplicates (20%)

### 3. vLLM Student Server
- **`vllm_service_init/start_vllm_server_gmsh.py`**

**What it does:**
1. Receives tasks from curriculum agent
2. Student models generate Gmsh scripts (10 candidates)
3. Executes scripts in isolated sandbox
4. Evaluates mesh quality
5. Returns best score to reward function

### 4. Prompt Template
- **`examples/format_prompt/gmsh_meshing.jinja`**

## Complete Workflow

### Step 1: Metadata Extraction (DONE ✅)
```bash
cd Agent0/curriculum_train/utils
python batch_extract_metadata.py \
    --geometry_dir ../geometries \
    --output_file metadata_all.json
```

You have: `metadata_all.json` with 55 geometries

### Step 2: Start vLLM Student Servers

Start 4 servers on different ports (for parallel processing):

```bash
# Terminal 1
python vllm_service_init/start_vllm_server_gmsh.py \
    --port 5000 \
    --model_path Qwen/Qwen3-4B-Base \
    --geometry_dir geometries \
    --metadata_file utils/metadata_all.json

# Terminal 2
python vllm_service_init/start_vllm_server_gmsh.py \
    --port 5001 \
    --model_path Qwen/Qwen3-4B-Base \
    --geometry_dir geometries \
    --metadata_file utils/metadata_all.json

# Terminal 3
python vllm_service_init/start_vllm_server_gmsh.py \
    --port 5002 \
    --model_path Qwen/Qwen3-4B-Base \
    --geometry_dir geometries \
    --metadata_file utils/metadata_all.json

# Terminal 4
python vllm_service_init/start_vllm_server_gmsh.py \
    --port 5003 \
    --model_path Qwen/Qwen3-4B-Base \
    --geometry_dir geometries \
    --metadata_file utils/metadata_all.json
```

### Step 3: Configure Curriculum Training

Create/modify `examples/config_gmsh.yaml`:

```yaml
data:
  train_files:
    - "path/to/geometry_list_train.parquet"  # List of geometry files
  val_files:
    - "path/to/geometry_list_val.parquet"

  # Fields in dataset
  prompt_key: "geometry_file"  # Field containing geometry file path
  answer_key: "geometry_file"  # Same for curriculum (no ground truth)

  # Prompt formatting
  format_prompt: "examples/format_prompt/gmsh_meshing.jinja"
  max_prompt_length: 2048
  max_response_length: 4096  # Longer for code generation

algorithm:
  adv_estimator: "GRPO"
  gamma: 1.0
  lam: 0.95
  kl_coef: 0.02

trainer:
  total_epochs: 10
  n_gpus_per_node: 8
  val_freq: 100
  save_freq: 500

actor:
  lr: 1e-6
  model_path: "Qwen/Qwen3-4B-Base"

critic:
  lr: 5e-6

ref:
  log_prob_micro_batch_size: 16

reward:
  reward_fn_path: "examples.reward_function.gmsh_curriculum_reward"
  reward_fn_name: "compute_score"

rollout:
  name: "vllm"
  gpu_memory_utilization: 0.85
  temperature: 0.7
  top_p: 0.9
  max_new_tokens: 4096
  n: 4  # Generate 4 task candidates per geometry
```

### Step 4: Create Training Dataset

The dataset should be a parquet/jsonl file with one geometry per row:

```python
import pandas as pd
import json

# Load metadata
with open('utils/metadata_all.json') as f:
    metadata = json.load(f)

# Create dataset - each geometry is one training example
data = []
for meta in metadata['metadata']:
    data.append({
        'geometry_file': meta['file'],  # Path to STEP file
        'metadata': meta  # Full metadata (optional)
    })

# Split train/val (90/10)
train_data = data[:50]
val_data = data[50:]

# Save as parquet
pd.DataFrame(train_data).to_parquet('geometry_list_train.parquet')
pd.DataFrame(val_data).to_parquet('geometry_list_val.parquet')
```

### Step 5: Run Curriculum Training

```bash
export STORAGE_PATH=/path/to/storage
export GEOMETRY_DIR=/home/kade/Agent0_backup/Agent0/curriculum_train/geometries

# Make sure temp_results directory exists
mkdir -p $STORAGE_PATH/temp_results

# Run training
python verl/trainer/main.py \
    --config examples/config_gmsh.yaml
```

## How It Works During Training

1. **Curriculum agent receives geometry file**:
   - Input: `geometries/bracket.step`
   - Metadata loaded automatically

2. **Curriculum agent generates task**:
   ```
   <task>Prepare mesh for thermal analysis of bracket.step.
   Refine at cylindrical holes. Create physical groups for
   thermal boundary conditions. Target element size: 0.5-5.0 units.</task>
   ```

3. **Task sent to vLLM servers**:
   - 4 servers process in parallel
   - Each generates 10 script candidates (40 total)

4. **Student models generate scripts**:
   ```python
   import gmsh
   gmsh.initialize()
   gmsh.merge("bracket.step")
   gmsh.model.occ.synchronize()

   # Create physical groups
   surfaces = gmsh.model.get_entities(2)
   cylinders = [s for s in surfaces if ...]
   gmsh.model.add_physical_group(2, cylinders, 1)

   # Mesh with refinement
   ...
   gmsh.model.mesh.generate(3)
   gmsh.write("output.msh")
   gmsh.finalize()
   ```

5. **Scripts executed in sandbox**:
   - Isolated temp directory
   - 120s timeout
   - Mesh analyzed for quality

6. **Results aggregated**:
   - Best score from 40 candidates: 0.75
   - Mesh generated: ✓
   - Physical groups: 3
   - Elements: 15,234

7. **Reward computed**:
   - Task format: ✓ (1.0)
   - Difficulty score: 0.75 (good - not too easy/hard)
   - Diversity: Compared to other tasks (cluster analysis)
   - **Final reward**: 0.68

8. **Curriculum agent updated**:
   - Learns to generate challenging, solvable tasks
   - Encouraged to create diverse tasks

## Key Files Summary

| File | Purpose |
|------|---------|
| `geometry_metadata_extract.py` | Extract geometry features |
| `metadata_all.json` | Metadata for all 55 geometries |
| `gmsh_curriculum_reward.py` | Reward function for curriculum agent |
| `start_vllm_server_gmsh.py` | Student model server |
| `gmsh_meshing.jinja` | Task prompt template |
| `config_gmsh.yaml` | Training configuration (create this) |

## Testing Before Full Training

Test with a single geometry:

```python
# Test reward function
from examples.reward_function.gmsh_curriculum_reward import compute_score

predicts = [
    "<task>Mesh bracket.step for thermal analysis. Refine at holes.</task>"
]
ground_truths = [
    "geometries/bracket.step"
]

scores = compute_score(predicts, ground_truths)
print(scores)
```

## Next Steps

1. ✅ Metadata extraction complete
2. ✅ Reward function created
3. ✅ vLLM server created
4. ⏳ Create training dataset (parquet files)
5. ⏳ Create config_gmsh.yaml
6. ⏳ Start vLLM servers
7. ⏳ Run training

## Notes

- Start with small model (Qwen3-4B) for testing
- Monitor task diversity in wandb
- Adjust reward weights if needed
- GPU memory: ~40GB per vLLM server
