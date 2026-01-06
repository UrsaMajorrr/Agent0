#!/bin/bash
#SBATCH --job-name=curriculum_agent
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=512G
#SBATCH --time=48:00:00

# Adjust these for your cluster:
# - Change partition name if needed
# - Adjust time limit (12 hours should be enough)
# - Adjust memory if needed
# - May need to add --account or --qos flags

# Load required modules (adjust for your cluster)
# module load cuda/12.1
# module load anaconda3

# Set environment variables
export STORAGE_PATH="/home/kade/Agent0_backup/Agent0"  # TODO: Fill this in

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

bash scripts/curriculum_train.sh Qwen/Qwen3-4B-Base Qwen/Qwen3-4B-Base qwen3_4b_curriculum_v1
bash scripts/gmsh_curriculum_train.sh Qwen/Qwen3-4B-Base qwen3_4b_curriculum_fea

echo "Training completed at $(date)"
