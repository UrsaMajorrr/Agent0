#!/bin/bash
set -x

# GMSH Executor Training with Teacher Distillation
#
# This script:
# 1. Generates teacher solutions using GPT-5.2 for curriculum tasks
# 2. Trains Qwen3-4B using the teacher solutions as in-context examples
#
# Usage:
#   export OPENAI_API_KEY="your-key"
#   bash train_with_distillation.sh [curriculum_tasks.parquet]


# Input curriculum tasks (from curriculum training)
curriculum_tasks="${1:-$STORAGE_PATH/generated_question/qwen3_4b_executor_iter3/train.parquet}"

# Step 2: Train executor with distillation
# Switch to executor env for training
source ~/miniconda3/etc/profile.d/conda.sh
conda activate executor

echo "=== Step 2: Training Qwen3-4B with teacher distillation ==="

# Model configuration
model_name=Qwen/Qwen3-4B-Instruct-2507
rl_alg=adpo

# Hardware configuration
n_gpus_per_node=8
n_nodes=1

# Sampling configuration
n=16
batch_size=16
ppo_mini_batch_size=16
max_prompt_length=4096      # Increased for teacher solution in prompt
max_response_length=4096
max_obs_length=20000
temperature=1.0
top_p=1.0

# Agent configuration
enable_agent=True
strategy="fsdp"
action_stop_tokens='```output'
max_turns=2
min_turns=1  # Force first turn to always execute (don't require stop token)

# Optimization
kl_loss_coef=1e-2
kl_coef=1e-2
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6

# Reward
reward_manager=torl

# Memory configuration
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8
tensor_model_parallel_size=1
gpu_memory_utilization=0.7
do_offload=False
use_dynamic_bsz=False
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True
max_action_length=2048

# Run configuration
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="gmsh_rl_$(date +%Y%m%d_%H%M%S)"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

# Temp file for action tokens
action_stop_tokens_file="/tmp/gmsh_action_stop_tokens_$$"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file

# Start GMSH Tool Server
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

echo "Starting GMSH Tool Server..."
python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "gmsh" \
    --workers_per_tool 8 \
    --max_concurrent_requests 64 &
server_pid=$!

echo "Server (pid=$server_pid) started at $tool_server_url"
sleep 5

# Main training with distillation enabled
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    +actor_rollout_ref.actor.policy_loss_fn=$rl_alg \
    +algorithm.min_score_for_scaling=0.3 \
    +algorithm.max_score_for_scaling=0.8 \
    +algorithm.min_advantage_scale=0.6 \
    +actor_rollout_ref.actor.max_epsilon_bonus=0.1 \
    data.train_files=$curriculum_tasks \
    data.val_files=null \
    data.train_batch_size=$batch_size \
    data.seed=42 \
    data.prompt_key=task \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    +data.use_distillation=False \
    +data.teacher_solution_key=teacher_solution \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.min_turns=$min_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=gmsh_distillation \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=False \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=9999 \
    trainer.total_training_steps=500

# Cleanup
pkill -P $server_pid
kill -9 $server_pid
rm -f $action_stop_tokens_file

echo "Distillation training complete!"
