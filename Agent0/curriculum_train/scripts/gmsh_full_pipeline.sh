#!/bin/bash
#
# GMSH Full Curriculum + Executor Pipeline
#
# Usage:
#   cd curriculum_train/
#   export STORAGE_PATH="/path/to/storage"
#   export WANDB_API_KEY="your_key"
#   bash scripts/gmsh_full_pipeline.sh
#

set -e

# ============================================
# Configuration
# ============================================
curriculum_agent_path=Qwen/Qwen3-4B-Base
experiment_name=qwen3_4b_gmsh_v1
num_samples_per_gpu=125  # 125 * 8 = 1000 total tasks

# ============================================
# Setup
# ============================================
mkdir -p \
    "${STORAGE_PATH}/evaluation" \
    "${STORAGE_PATH}/models" \
    "${STORAGE_PATH}/generated_question" \
    "${STORAGE_PATH}/temp_results"

echo "========================================="
echo "GMSH Full Pipeline"
echo "========================================="
echo "Curriculum model: $curriculum_agent_path"
echo "Experiment: $experiment_name"
echo "Storage: ${STORAGE_PATH}"
echo "========================================="

# ============================================
# Step 1: Curriculum Training
# ============================================
echo ""
echo "Step 1: Training curriculum agent..."
echo "========================================="

bash scripts/gmsh_curriculum_train.sh $curriculum_agent_path $experiment_name

# Get trained model path
trained_model=${STORAGE_PATH}/models/${experiment_name}/global_step_*/actor/huggingface
trained_model=$(ls -td $trained_model 2>/dev/null | head -1)

if [ -z "$trained_model" ]; then
    echo "ERROR: No trained model found"
    exit 1
fi
echo "Trained model: $trained_model"

# ============================================
# Step 2: Generate Tasks
# ============================================
echo ""
echo "Step 2: Generating tasks..."
echo "========================================="

export VLLM_DISABLE_COMPILE_CACHE=1

# Generate on 8 GPUs in parallel
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python -m question_generate.question_generate \
        --model $trained_model \
        --suffix $i \
        --num_samples $num_samples_per_gpu \
        --save_name $experiment_name &
done
wait

echo "Task generation complete."

# ============================================
# Step 3: Filter and Upload Tasks
# ============================================
echo ""
echo "Step 3: Filtering tasks by score..."
echo "========================================="

# Filter tasks with score >= 0.5
LOCAL_DATA_PATH=$(python question_evaluate/upload_gmsh.py \
    --min_score 0.5 \
    --max_score 1.0 \
    --experiment_name $experiment_name)

echo "Filtered data saved to: ${LOCAL_DATA_PATH}"

# ============================================
# Step 4: Convert to Executor Format
# ============================================
echo ""
echo "Step 4: Converting to executor format..."
echo "========================================="

# Get the filtered JSON
filtered_json="${STORAGE_PATH}/generated_question/${experiment_name}/filtered_tasks.json"
executor_parquet="${STORAGE_PATH}/generated_question/${experiment_name}/executor_train.parquet"

python ../executor_train/data/gmsh_mesh/convert_curriculum_to_executor.py \
    --input "$filtered_json" \
    --output "$executor_parquet" \
    --score-min 0.5 \
    --score-max 1.0

# ============================================
# Done
# ============================================
echo ""
echo "========================================="
echo "GMSH Pipeline Complete!"
echo "========================================="
echo ""
echo "Outputs:"
echo "  Trained curriculum model: $trained_model"
echo "  Filtered tasks: ${LOCAL_DATA_PATH}"
echo "  Executor training data: ${executor_parquet}"
echo ""
echo "Next: Run executor training with the generated data"
