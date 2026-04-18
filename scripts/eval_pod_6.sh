#!/usr/bin/env bash
# Pod 6 — sycophancy/42 medical second half (10 small steps). ~12 min.
# Replace REDACTED_* values with real keys via vim before running.

set -u

BASE_DIR=/Obfuscation_Generalization
VENV_PATH=./venv/bin/activate
LOG_DIR="${BASE_DIR}/eval_logs"
SESSION_NAME="eval_pod_6"
CUDA_DEVICE=0
MASTER_PORT=25006

mkdir -p "${LOG_DIR}"

JOBS=(
  "sum|leave_out_sycophancy_refined2|42|refined2/eval_sycophancy_medical_with_summary|2000,2200,2400,2600,2800,3000,3200,3400,3600,3800"
)

JOB_CMDS=""
idx=0
TOTAL=${#JOBS[@]}
for entry in "${JOBS[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r TYPE DATA SEED EXPERIMENT STEPS <<<"${entry}"
  case "${TYPE}" in
    ref) CONFIG_NAME="eval";     TRAIN_RUN_NAME="run_ref_lr_7.5e-05_ovs_refined_pen_-0.1_data_${DATA}_ts_${SEED}";;
    sum) CONFIG_NAME="eval_sum"; TRAIN_RUN_NAME="run_sum_lr_7.5e-05_ovs_refined_summary_data_${DATA}_ts_${SEED}";;
    *)   echo "Unknown TYPE '${TYPE}'" >&2; exit 1;;
  esac
  EXP_TAG="${EXPERIMENT//\//__}"
  LOG_FILE_EXPR="${LOG_DIR}/pod6_${TYPE}_${DATA}_ts${SEED}_${EXP_TAG}_\$(date +%Y%m%d_%H%M%S).log"

  JOB_CMDS+="
    echo &&
    echo '==========================================================' &&
    echo '[${idx}/${TOTAL}] ${TYPE} ${DATA} seed=${SEED} ${EXPERIMENT}' &&
    echo '  steps: ${STEPS}' &&
    echo '==========================================================' &&
    LOG_FILE=\"${LOG_FILE_EXPR}\" &&
    (python -m src.eval --multirun 'hydra.sweep.subdir=\${hydra.job.num}' \\
       experiment=${EXPERIMENT} \\
       training_group=${DATA}_seed_${SEED} \\
       model=qwen3_8b \\
       config_name=${CONFIG_NAME} \\
       training_run_name=${TRAIN_RUN_NAME} \\
       artifact_step=${STEPS} \\
       ++wandb.entity=nathanielmitrani-cfis-upc \\
       +train.seed=${SEED} 2>&1 | tee \"\$LOG_FILE\") || echo '[${idx}/${TOTAL}] FAILED -- continuing' &&
"
done

tmux new-session -d -s "${SESSION_NAME}" "
  cd ${BASE_DIR} &&
  source ${VENV_PATH} &&
  export TOGETHER_API_KEY=REDACTED_TOGETHER_API_KEY &&
  export HF_TOKEN=REDACTED_HF_TOKEN &&
  export VLLM_TORCH_COMPILE_LEVEL=0 &&
  export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} &&
  export MASTER_PORT=${MASTER_PORT} &&
  wandb login REDACTED_WANDB_API_KEY &&
  ${JOB_CMDS}
  echo 'Done.' &&
  exec bash
"

echo "Launched tmux session '${SESSION_NAME}'. Attach: tmux attach -t ${SESSION_NAME}"
