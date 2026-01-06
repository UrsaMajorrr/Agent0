# GMSH Executor Training

Train Qwen3-4B to generate GMSH meshing scripts using task completion matching rewards.

## Prerequisites

### 1. Environment Setup

```bash
# Install GMSH Python package
pip install gmsh

# Or using conda
conda install -c conda-forge gmsh

# Set STORAGE_PATH environment variable
export STORAGE_PATH=/path/to/storage
```

### 2. Data Preparation

Ensure you have converted curriculum training data:

```bash
cd $STORAGE_PATH/executor_train/data/gmsh_mesh

# Should have:
# - train.parquet
# - val.parquet (optional)

# Verify data
python -c "import pandas as pd; df = pd.read_parquet('train.parquet'); print(f'Tasks: {len(df)}'); print(f'Columns: {df.columns.tolist()}')"
```

## Training

### Quick Start

```bash
cd executor_train/examples/train/gmsh_mesh
bash train_qwen3_4b_adpo.sh
```

### Configuration

Edit `train_qwen3_4b_adpo.sh` to customize:

**Key Parameters**:
- `n_gpus_per_node=8`: Number of GPUs (adjust for your setup)
- `batch_size=128`: Training batch size
- `max_turns=1`: Single-turn only (no iteration)
- `lr=1e-6`: Learning rate
- `trainer.total_training_steps=500`: Total steps
- `trainer.total_epochs=10`: Total epochs

**Memory Management**:
- `gpu_memory_utilization=0.7`: VLLM GPU memory usage
- `do_offload=False`: FSDP offloading (enable if OOM)

### Small-Scale Test

Test the setup with minimal resources:

```bash
# Modify train_qwen3_4b_adpo.sh:
# trainer.total_training_steps=2
# batch_size=4

bash train_qwen3_4b_adpo.sh
```

Expected output:
- Tool server starts on random port
- Training begins with step 0
- Wandb logging to project "gmsh_executor"
- No crashes or OOM errors

## Monitoring

### Weights & Biases

Training logs to wandb project `gmsh_executor`:

**Key Metrics**:
- `reward/mean`: Average reward (should increase)
- `reward/accuracy`: % of positive rewards
- `actor/learning_rate`: Learning rate schedule
- `rollout/num_tokens`: Token statistics

### Local Logs

Logs saved to:
- `verl_step_records/{run_name}/`: Step-by-step records
- Console output: Real-time training progress

## Architecture

### Data Flow

```
Parquet Dataset
    ↓
VerlToolRLHFDataset (loads tasks + geometry metadata)
    ↓
PPO Trainer (generates rollouts)
    ↓
VLLM Rollout (model generates GMSH code)
    ↓
GMSH Tool Server (executes code, analyzes mesh)
    ↓
Task Completion Reward (evaluates mesh vs requirements)
    ↓
Policy Update (PPO/ADPO)
```

### Components

1. **GMSH Tool** (`executor_train/verl_tool/servers/tools/gmsh_tool.py`)
   - Parses Python code from model responses
   - Executes in sandbox with geometry file
   - Analyzes mesh quality
   - Returns observations + mesh statistics

2. **Reward Function** (`executor_train/verl_tool/workers/reward_manager/reward_score/gmsh_executor.py`)
   - Parses task requirements (physical groups, refinement, etc.)
   - Scores mesh based on task completion
   - Geometry-aware density scoring
   - Physical groups REQUIRED (no physical groups → -1.0)

3. **Reward Manager** (`executor_train/verl_tool/workers/reward_manager/torl.py`)
   - Dispatches to GMSH reward for `data_source='gmsh_mesh'`
   - Extracts mesh stats from tool interaction
   - Returns rewards in [-1, 1] range

## Reward Scoring

### Components (Weighted)

1. **Mesh Generated (30%)**: Basic success (num_elements > 0)
2. **Physical Groups (30%)**: ALWAYS REQUIRED
   - 0 groups → -1.0 (immediate failure)
   - ≥ required → 1.0
   - Partial credit if some but not all
3. **Mesh Density (20%)**: Geometry-aware
   - Penalizes over-refinement (too many elements)
   - Penalizes under-refinement (too few elements)
   - Uses bbox volume and characteristic_length
4. **Mesh Quality (20%)**: Element quality metrics
   - quality_min > 0.3 → 1.0
   - quality_min > 0.1 → 0.5

Final score mapped to [-1, 1] for ADPO compatibility.

## Checkpoints

Checkpoints saved to:
```
$STORAGE_PATH/models/{run_name}/global_step_{N}/
    actor/
        pytorch_model.bin
        config.json
```

**Frequency**:
- `trainer.save_freq=20`: Save every 20 steps
- Keeps 5 most recent checkpoints

## Validation

Validation runs every `trainer.test_freq=10` steps.

**Metrics**:
- Success rate (% with valid mesh)
- Physical group rate (% with physical groups)
- Average reward
- Mesh density distribution

## Troubleshooting

### Issue: Tool server fails to start

**Symptom**: "Connection refused" or "Tool server not responding"

**Solutions**:
1. Check GMSH installation: `python -c "import gmsh; print('OK')"`
2. Check port availability: `netstat -ln | grep {port}`
3. Increase sleep time after server start (line 77)

### Issue: All rewards are -1.0

**Symptom**: No positive rewards after many steps

**Causes**:
1. **No mesh generated**: Check tool execution logs
2. **No physical groups**: Model not learning to create them
3. **Geometry files missing**: Check paths in parquet

**Solutions**:
- Log mesh_stats in reward function for debugging
- Reduce task difficulty (simpler geometries)
- Check geometry file paths exist

### Issue: GPU OOM

**Symptom**: CUDA out of memory errors

**Solutions**:
1. Reduce `gpu_memory_utilization=0.7` → `0.5`
2. Enable offloading: `do_offload=True`
3. Reduce `batch_size=128` → `64`
4. Reduce `n=16` (rollouts per prompt) → `8`

### Issue: Training very slow

**Symptom**: <5 steps/minute

**Causes**:
- GMSH execution timeout (120s per script)
- Too many workers (`workers_per_tool=8`)

**Solutions**:
- Reduce timeout in gmsh_tool.py if acceptable
- Reduce `workers_per_tool=8` → `4`
- Use faster geometries (fewer elements)

## Expected Performance

### Short-term (100 steps)
- Success rate: 50-60%
- Some positive rewards
- Physical group rate: 20-30%

### Medium-term (500 steps)
- Success rate: 70-80%
- Average reward: >0.0
- Physical group rate: 40-50%
- Model learns basic GMSH patterns

### Long-term (Full training)
- Success rate: 85%+
- Average reward: >0.3
- Task completion: 60%+
- Generalizes to validation geometries

## Advanced Configuration

### Warm Start from Curriculum Student

Use curriculum student model as initialization:

```bash
# Modify train_qwen3_4b_adpo.sh
model_name=/path/to/curriculum/student/checkpoint
```

### Multi-Node Training

```bash
# Modify train_qwen3_4b_adpo.sh
n_nodes=2
n_gpus_per_node=8

# Launch with torchrun or slurm
```

### Custom Reward Weights

Edit `gmsh_executor.py` line 256:

```python
final_score = (
    0.30 * scores['mesh_generated'] +
    0.30 * scores['physical_groups'] +
    0.20 * scores['mesh_density'] +
    0.20 * scores['mesh_quality']
)
```

## Next Steps

1. Run small-scale test to verify setup
2. Monitor wandb for first few steps
3. Check validation metrics
4. Adjust hyperparameters if needed
5. Launch full training run
6. Evaluate checkpoints on validation set
7. Deploy best checkpoint for inference
