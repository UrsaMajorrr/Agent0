#!/bin/bash
#
# GMSH Curriculum Training Script
#
# Usage:
#   cd curriculum_train/
#   export STORAGE_PATH="/path/to/storage"
#   export WANDB_API_KEY="your_key"
#   bash scripts/gmsh_curriculum_train.sh Qwen/Qwen3-4B-Base qwen3_4b_gmsh_curriculum_v1
#

set -e

project_name=gmsh_curriculum

curriculum_agent_path=$1
save_path=$2
executor_agent_path=$3

if [ -z "$curriculum_agent_path" ] || [ -z "$save_path" ]; then
    echo "Usage: bash scripts/gmsh_curriculum_train.sh <model_path> <save_name>"
    echo "Example: bash scripts/gmsh_curriculum_train.sh Qwen/Qwen3-4B-Base qwen3_4b_gmsh_v1"
    exit 1
fi

# Create required directories
mkdir -p \
    "${STORAGE_PATH}/evaluation" \
    "${STORAGE_PATH}/models" \
    "${STORAGE_PATH}/generated_question" \
    "${STORAGE_PATH}/temp_results"

echo "========================================="
echo "GMSH Curriculum Training"
echo "========================================="
echo "Model: $curriculum_agent_path"
echo "Save path: $save_path"
echo "Storage: ${STORAGE_PATH}"
echo "========================================="

# Generate unique run ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Start vLLM service for executor validation (GPUs 4-7)
echo "Starting vLLM services..."
bash vllm_service_init/gmsh_start.sh $executor_agent_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# Wait for servers to initialize
sleep 10

# Training (GPUs 0-3)
echo "Starting curriculum agent training..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config_gmsh.yaml \
    data.max_response_length=1024 \
    worker.actor.model.model_path=$curriculum_agent_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    worker.reward.reward_function=./examples/reward_function/gmsh_curriculum_reward.py:compute_score \
    trainer.val_freq=10 \
    trainer.n_gpus_per_node=4 \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=128 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.max_steps=6 \
    trainer.save_freq=50 \
    data.format_prompt=./examples/format_prompt/gmsh_meshing.jinja

echo "========================================="
echo "Training complete. Merging model..."
echo "========================================="

sleep 5

# Find and merge the latest checkpoint
latest_checkpoint=$(ls -td ${STORAGE_PATH}/models/$save_path/global_step_* 2>/dev/null | head -1)

if [ -n "$latest_checkpoint" ]; then
    echo "Merging model from: $latest_checkpoint/actor"
    python scripts/model_merger.py --local_dir $latest_checkpoint/actor
else
    echo "No checkpoint found to merge"
fi

# Kill vLLM servers
echo "Stopping vLLM services..."
pkill -f "start_vllm_server_gmsh" || true
sleep 5

echo "========================================="
echo "GMSH Curriculum Training Complete!"
echo "========================================="
echo "Model saved to: ${STORAGE_PATH}/models/$save_path"
echo ""
echo "Next steps:"
echo "  1. Generate tasks: python question_generate/question_generate.py --model ${STORAGE_PATH}/models/$save_path --save_name gmsh_tasks"
echo "  2. Convert to executor format: python ../executor_train/data/gmsh_mesh/convert_curriculum_to_executor.py --input \${STORAGE_PATH}/generated_question/gmsh_tasks_0.json --output executor_data.parquet"
