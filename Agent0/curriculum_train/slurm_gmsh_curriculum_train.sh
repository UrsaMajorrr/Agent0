#!/bin/bash
#SBATCH --job-name=curriculum_agent
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=512G
#SBATCH --time=96:00:00

# Adjust these for your cluster:
# - Change partition name if needed
# - Adjust time limit (12 hours should be enough)
# - Adjust memory if needed
# - May need to add --account or --qos flags

# Load required modules (adjust for your cluster)
# module load cuda/12.1
# module load anaconda3

# Set environment variables
export STORAGE_PATH="/home/kade/Agent0_backup/Agent0/fea_experiments"  # TODO: Fill this in

executor=/home/kade/Agent0_backup/Agent0/executor_train/checkpoints/gmsh_distillation/gmsh_rl_20260108_164601/global_step_440/actor/huggingface

# Ensure required directories exist
mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results"

# Activate conda properly in SLURM
source /home/kade/miniconda3/etc/profile.d/conda.sh
conda activate curriculum

# Verify activation
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Navigate to working directory
cd /home/kade/Agent0_backup/Agent0/curriculum_train

bash scripts/gmsh_curriculum_train.sh $STORAGE_PATH/models/qwen3_4b_curriculum_iter3/global_step_1006/actor/huggingface qwen3_4b_curriculum_iter4 $executor

echo "Training completed at $(date)"
