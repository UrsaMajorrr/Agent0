#!/bin/bash
#SBATCH --job-name=curriculum_agent_question_generate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=512G
#SBATCH --time=48:00:00

export STORAGE_PATH="/home/kade/Agent0_backup/Agent0/fea_experiments"
curriculum_agent_path="/home/kade/Agent0_backup/Agent0/fea_experiments/models/qwen3_4b_curriculum_iter3/global_step_1006/actor/huggingface"
executor_agent_path="/home/kade/Agent0_backup/Agent0/executor_train/checkpoints/gmsh_distillation/gmsh_rl_20260101_181743/global_step_400/actor/huggingface"
experiment_name="qwen3_4b_executor_iter3"

source /home/kade/miniconda3/etc/profile.d/conda.sh
conda activate curriculum

cd /home/kade/Agent0_backup/Agent0/curriculum_train

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $curriculum_agent_path 500 $experiment_name

echo 'start evaluate generated question'
bash question_evaluate/evaluate_gmsh.sh $executor_agent_path $experiment_name

echo 'start upload'
LOCAL_DATA_PATH=$(python question_evaluate/upload_gmsh.py --max_score 0.8 --min_score 0.1 --experiment_name ${experiment_name})
echo "training data saved to: ${LOCAL_DATA_PATH}"