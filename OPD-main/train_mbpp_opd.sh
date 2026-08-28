#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VERL_DIR="${SCRIPT_DIR}/verl"

OPD_PYTHON=${OPD_PYTHON:-python3}
STUDENT_MODEL=${STUDENT_MODEL:-"${PROJECT_ROOT}/student_model"}
TEACHER_MODEL=${TEACHER_MODEL:-"${PROJECT_ROOT}/teacher_model"}
MBPP_SOURCE_DIR=${MBPP_SOURCE_DIR:-"${PROJECT_ROOT}/dataset/full"}
MBPP_OPD_DIR=${MBPP_OPD_DIR:-"${SCRIPT_DIR}/datasets/mbpp_opd"}
TRAIN_DATA=${TRAIN_DATA:-"${MBPP_OPD_DIR}/train.parquet"}
VAL_DATA=${VAL_DATA:-"${MBPP_OPD_DIR}/validation.parquet"}

TOTAL_STEPS=${TOTAL_STEPS:-200}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
N_RESPONSES=1

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_MODEL_LEN * PER_DEVICE_BATCH_SIZE))

DISTILLATION_TOPK=${DISTILLATION_TOPK:-16}
TOP_K_STRATEGY=${TOP_K_STRATEGY:-union}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-teacher_p}
ACTOR_LR=${ACTOR_LR:-1e-6}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-${GLOBAL_BATCH_SIZE}}
TEACHER_MICRO_BATCH_SIZE=${TEACHER_MICRO_BATCH_SIZE:-1}
SAVE_FREQ=${SAVE_FREQ:-20}
ALLOW_LOW_MEMORY=${ALLOW_LOW_MEMORY:-0}
RAY_CPUS=${RAY_CPUS:-8}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
ENABLE_THINKING=${ENABLE_THINKING:-true}
PATCH_VLLM_NUMPY_INDEX=${PATCH_VLLM_NUMPY_INDEX:-1}

case "${ENABLE_THINKING,,}" in
    1|true|yes|on)
        ENABLE_THINKING_BOOL=True
        ;;
    0|false|no|off)
        ENABLE_THINKING_BOOL=False
        ;;
    *)
        printf 'ENABLE_THINKING must be true/false (received: %s)\n' "${ENABLE_THINKING}" >&2
        exit 2
        ;;
esac

PROJECT_NAME=${PROJECT_NAME:-mbpp_qwen3_opd}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1p7b_from_qwen3_4b_steps${TOTAL_STEPS}_bs${GLOBAL_BATCH_SIZE}}
CKPT_DIR=${CKPT_DIR:-"${SCRIPT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

MODE=${1:-train}
if [[ "${MODE}" != "train" && "${MODE}" != "prepare" && "${MODE}" != "config" && "${MODE}" != "preflight" ]]; then
    printf 'Usage: %s [prepare|config|preflight|train]\n' "$0" >&2
    exit 2
fi

if (( GLOBAL_BATCH_SIZE != PER_DEVICE_BATCH_SIZE * GRAD_ACC_STEPS * NGPUS_PER_NODE )); then
    printf 'Batch mismatch: %s != %s * %s * %s\n' \
        "${GLOBAL_BATCH_SIZE}" "${PER_DEVICE_BATCH_SIZE}" "${GRAD_ACC_STEPS}" "${NGPUS_PER_NODE}" >&2
    exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${VERL_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}

prepare_data() {
    "${OPD_PYTHON}" -m code_rewrite_feedback_expander.mbpp_to_opd_parquet \
        --dataset-dir "${MBPP_SOURCE_DIR}" \
        --output-dir "${MBPP_OPD_DIR}" \
        --splits train validation test
}

if [[ "${MODE}" == "prepare" ]]; then
    prepare_data
    exit 0
fi

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
    prepare_data
fi

case "${PATCH_VLLM_NUMPY_INDEX,,}" in
    1|true|yes|on)
        "${OPD_PYTHON}" "${SCRIPT_DIR}/patch_vllm_numpy_index.py"
        ;;
    0|false|no|off)
        ;;
    *)
        printf 'PATCH_VLLM_NUMPY_INDEX must be true/false (received: %s)\n' \
            "${PATCH_VLLM_NUMPY_INDEX}" >&2
        exit 2
        ;;
esac

TRAIN_OVERRIDES=(
    "algorithm.adv_estimator=token_reward_direct"
    "algorithm.use_kl_in_reward=False"
    "data.train_files=${TRAIN_DATA}"
    "data.val_files=${VAL_DATA}"
    "data.train_batch_size=${GLOBAL_BATCH_SIZE}"
    "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.shuffle=True"
    "data.return_raw_chat=True"
    "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING_BOOL}"
    "actor_rollout_ref.model.path=${STUDENT_MODEL}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.enable_activation_offload=True"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${GLOBAL_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PER_DEVICE_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    "actor_rollout_ref.actor.use_dynamic_bsz=False"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch=False"
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=${N_RESPONSES}"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.calculate_log_probs=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "+actor_rollout_ref.rollout.log_prob_top_k=${DISTILLATION_TOPK}"
    "+actor_rollout_ref.rollout.top_k_strategy=${TOP_K_STRATEGY}"
    "+actor_rollout_ref.rollout.reward_weight_mode=${REWARD_WEIGHT_MODE}"
    "+actor_rollout_ref.rollout.teacher_temperature=1.0"
    "reward_model.enable=True"
    "reward_model.model.path=${TEACHER_MODEL}"
    "reward_model.model.input_tokenizer=null"
    "reward_model.model.use_remove_padding=True"
    "reward_model.model.fsdp_config.param_offload=True"
    "+reward_model.model.dtype=bf16"
    "reward_model.micro_batch_size_per_gpu=${TEACHER_MICRO_BATCH_SIZE}"
    "custom_reward_function.path=${PROJECT_ROOT}/code_rewrite_feedback_expander/mbpp_reward.py"
    "custom_reward_function.name=reward_func"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.total_epochs=${TOTAL_STEPS}"
    "trainer.total_training_steps=${TOTAL_STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=-1"
    "trainer.val_before_train=False"
    "trainer.balance_batch=False"
    'trainer.logger=["console"]'
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.default_local_dir=${CKPT_DIR}"
)

validate_hydra_config() {
    "${OPD_PYTHON}" "${SCRIPT_DIR}/validate_mbpp_opd_config.py" \
        --config-dir "${VERL_DIR}/verl/trainer/config" \
        --gradient-accumulation-steps "${GRAD_ACC_STEPS}" \
        --n-gpus "${NGPUS_PER_NODE}" \
        "${TRAIN_OVERRIDES[@]}"
}

validate_main_entrypoint() {
    "${OPD_PYTHON}" -m verl.trainer.main_ppo --cfg job --resolve "${TRAIN_OVERRIDES[@]}" >/dev/null
    printf 'verl main_ppo entrypoint config passed.\n'
}

if [[ "${MODE}" == "config" ]]; then
    validate_hydra_config
    validate_main_entrypoint
    exit 0
fi

PREFLIGHT_ARGS=(
    --student-model "${STUDENT_MODEL}"
    --teacher-model "${TEACHER_MODEL}"
    --train-data "${TRAIN_DATA}"
    --val-data "${VAL_DATA}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACC_STEPS}"
    --n-gpus "${NGPUS_PER_NODE}"
)
if [[ "${ALLOW_LOW_MEMORY}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--allow-low-memory)
fi
"${OPD_PYTHON}" "${SCRIPT_DIR}/preflight_mbpp_opd.py" "${PREFLIGHT_ARGS[@]}"
validate_hydra_config
validate_main_entrypoint

if [[ "${MODE}" == "preflight" ]]; then
    exit 0
fi

mkdir -p "${CKPT_DIR}"
cd "${SCRIPT_DIR}"

OPD_PYTHON_BIN=$(command -v -- "${OPD_PYTHON}")
RAY_BIN=${RAY_BIN:-"$(dirname -- "${OPD_PYTHON_BIN}")/ray"}
if [[ ! -x "${RAY_BIN}" ]]; then
    printf 'Ray CLI not found next to OPD_PYTHON: %s\n' "${RAY_BIN}" >&2
    exit 2
fi

"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
"${RAY_BIN}" start --head --num-cpus="${RAY_CPUS}"
trap '"${RAY_BIN}" stop --force >/dev/null 2>&1 || true' EXIT

"${OPD_PYTHON}" -m verl.trainer.main_ppo "${TRAIN_OVERRIDES[@]}"
