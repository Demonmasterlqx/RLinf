#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s MICRO_BATCH [probe|validate]\n' "$0" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

MICRO_BATCH="${1:-}"
MODE="${2:-probe}"
MAX_STEPS="${TABERO_FIRM_CAPACITY_MAX_STEPS:-1}"
[[ $# -le 2 ]] || { usage; die "too many arguments"; }
[[ "${MICRO_BATCH}" =~ ^(1|2|4|8|16)$ ]] || {
  usage
  die "MICRO_BATCH must be one of 1,2,4,8,16"
}
[[ "${MODE}" == "probe" || "${MODE}" == "validate" ]] || {
  usage
  die "mode must be probe or validate"
}
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || \
  die "TABERO_FIRM_CAPACITY_MAX_STEPS must be a positive integer"

if [[ "${MODE}" == "probe" ]]; then
  GLOBAL_BATCH=$((MICRO_BATCH * 2))
else
  GLOBAL_BATCH=32
fi
((GLOBAL_BATCH % (MICRO_BATCH * 2) == 0)) || \
  die "global batch is not divisible by micro batch x two ranks"
GRADIENT_ACCUMULATION=$((GLOBAL_BATCH / MICRO_BATCH / 2))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
CONFIG_NAME="tabero_firm_sft_2gpu_selective_siglip_capacity"
CONFIG_PATH="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"
MODEL_PATH="${WORKSPACE_ROOT}/models/pi0_base/model.safetensors"
DATASET_PATH="${WORKSPACE_ROOT}/datas/tabero_firm"
RUN_STAMP="${TABERO_FIRM_CAPACITY_RUN_STAMP:-$(date +'%Y%m%d_%H%M%S')}"
RUN_NAME="tabero_firm_rlinf_2gpu_siglip_capacity_mb${MICRO_BATCH}_${MODE}_${RUN_STAMP}"
RUN_DIR="${WORKSPACE_ROOT}/results/${RUN_NAME}"

[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || die "RLinf Python is missing"
[[ -f "${CONFIG_PATH}" ]] || die "capacity config is missing"
[[ -f "${MODEL_PATH}" ]] || die "base model is missing"
[[ -f "${DATASET_PATH}/meta/info.json" ]] || die "Firm dataset is missing"
[[ ! -e "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"

for gpu_id in 0 1; do
  compute_pids="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid \
    --format=csv,noheader,nounits)"
  [[ -z "${compute_pids//[[:space:]]/}" ]] || \
    die "GPU ${gpu_id} has compute process ${compute_pids}"
  used_memory="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]')"
  ((used_memory < 1024)) || die "GPU ${gpu_id} already uses ${used_memory} MiB"
done

export TABERO_FIRM_2GPU_CAPACITY_RUN_DIR="${RUN_DIR}"
export TABERO_FIRM_2GPU_CAPACITY_RUN_NAME="${RUN_NAME}"
export TABERO_FIRM_2GPU_CAPACITY_CHECKPOINT_ROOT="${RUN_DIR}/checkpoints"
export TABERO_FIRM_NORM_STATS="${RUN_DIR}/norm_stats/norm_stats.json"
export EMBODIED_PATH="${SCRIPT_DIR}"
export REPO_PATH="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="0,1"
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

mkdir -p "${RUN_DIR}/norm_stats" "${RUN_DIR}/checkpoints"
if [[ -n "${TABERO_FIRM_EXISTING_NORM_STATS:-}" ]]; then
  NORM_SOURCE="${TABERO_FIRM_EXISTING_NORM_STATS}"
  [[ -f "${NORM_SOURCE}" ]] || die "Firm norm stats are missing"
  cp -- "${NORM_SOURCE}" "${TABERO_FIRM_NORM_STATS}"
  printf 'Reused tactile-aware Firm norm stats from %s\n' "${NORM_SOURCE}" | \
    tee "${RUN_DIR}/norm_stats.log"
else
  "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/toolkits/lerobot/calculate_norm_stats.py" \
    --config-name pi0_lora_tacfield_tabero \
    --repo-id "${DATASET_PATH}" \
    --output-dir "${RUN_DIR}/norm_stats" \
    2>&1 | tee "${RUN_DIR}/norm_stats.log"
fi
(cd "${RUN_DIR}/norm_stats" && sha256sum norm_stats.json >norm_stats.sha256)
"${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/toolkits/lerobot/audit_tabero_firm_sft_batch.py" \
  --dataset-path "${DATASET_PATH}" \
  --norm-stats-path "${TABERO_FIRM_NORM_STATS}" \
  --output "${RUN_DIR}/batch_audit.json" \
  2>&1 | tee "${RUN_DIR}/batch_audit.log"

{
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'physical_gpus=0,1\n'
  printf 'micro_batch_size=%s\n' "${MICRO_BATCH}"
  printf 'global_batch_size=%s\n' "${GLOBAL_BATCH}"
  printf 'gradient_accumulation=%s\n' "${GRADIENT_ACCUMULATION}"
  printf 'max_steps=%s\n' "${MAX_STEPS}"
  printf 'mode=%s\n' "${MODE}"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} | tee "${RUN_DIR}/launch_meta.txt"

"${REPO_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/train_vla_sft.py" \
  --config-path "${SCRIPT_DIR}/config" \
  --config-name "${CONFIG_NAME}" \
  actor.micro_batch_size="${MICRO_BATCH}" \
  actor.global_batch_size="${GLOBAL_BATCH}" \
  runner.max_steps="${MAX_STEPS}" \
  actor.optim.total_training_steps="${MAX_STEPS}" \
  actor.optim.lr_decay_steps="${MAX_STEPS}" \
  2>&1 | tee "${RUN_DIR}/train.log" &
TRAIN_PID=$!

(
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    timestamp="$(date --iso-8601=seconds)"
    nvidia-smi --id=0,1 --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | \
      while IFS= read -r sample; do
        printf '%s,%s\n' "${timestamp}" "${sample}"
      done
    sleep 1
  done
) >"${RUN_DIR}/gpu_samples.csv" &
MONITOR_PID=$!

set +e
wait "${TRAIN_PID}"
EXIT_CODE=$?
set -e
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

printf '%s\n' "${EXIT_CODE}" >"${RUN_DIR}/exit_code"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" | \
  tee -a "${RUN_DIR}/launch_meta.txt"
for gpu_id in 0 1; do
  peak_memory="$(awk -F, -v gpu="${gpu_id}" \
    '$2 + 0 == gpu {memory=$3 + 0; if (memory > max) max=memory} END {print max + 0}' \
    "${RUN_DIR}/gpu_samples.csv")"
  printf 'gpu%s_peak_memory_mib=%s\n' "${gpu_id}" "${peak_memory}" | \
    tee -a "${RUN_DIR}/launch_meta.txt"
done
printf '%s\n' "${RUN_DIR}"
exit "${EXIT_CODE}"
