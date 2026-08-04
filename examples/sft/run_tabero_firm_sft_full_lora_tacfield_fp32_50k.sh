#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s [formal|preflight|resume]\n' "$0" >&2
  printf 'resume requires TABERO_FIRM_RESUME_DIR=.../checkpoints/global_step_N\n' >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

MODE="${1:-formal}"
[[ $# -le 1 ]] || { usage; die "too many arguments"; }
case "${MODE}" in
  formal|preflight|resume) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    die "mode must be formal, preflight, or resume"
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
CONFIG_NAME="tabero_firm_sft_full_lora_tacfield_fp32_50k"
CONFIG_PATH="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"
DATASET_PATH="${WORKSPACE_ROOT}/datas/tabero_firm"
MODEL_PATH="${WORKSPACE_ROOT}/models/pi0_base"
REFERENCE_MODEL_PATH="${WORKSPACE_ROOT}/models/pi0_lora_tacfield_tabero_safetensors"
MIN_FREE_DISK_KIB=314572800
PHYSICAL_GPUS=(0 1 3 4 5 6 7)
PHYSICAL_GPU_CSV="0,1,3,4,5,6,7"

[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || die "RLinf Python is missing"
[[ -f "${CONFIG_PATH}" ]] || die "training config is missing: ${CONFIG_PATH}"
[[ -f "${MODEL_PATH}/model.safetensors" ]] || die "base model is missing"
[[ -f "${REFERENCE_MODEL_PATH}/model.safetensors" ]] || \
  die "reference model is missing"
[[ -f "${DATASET_PATH}/meta/info.json" ]] || die "Firm dataset is missing"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

if [[ "${MODE}" == "resume" ]]; then
  RESUME_DIR="${TABERO_FIRM_RESUME_DIR:-}"
  [[ -n "${RESUME_DIR}" ]] || die "TABERO_FIRM_RESUME_DIR is required"
  [[ "${RESUME_DIR}" == /* ]] || die "TABERO_FIRM_RESUME_DIR must be absolute"
  [[ -d "${RESUME_DIR}/actor/dcp_checkpoint" ]] || \
    die "resume DCP checkpoint is missing: ${RESUME_DIR}"
  [[ "$(basename "${RESUME_DIR}")" =~ ^global_step_[0-9]+$ ]] || \
    die "resume directory must end in global_step_N"
  RUN_DIR="$(cd "$(dirname "$(dirname "${RESUME_DIR}")")" && pwd -P)"
  RUN_NAME="$(basename "${RUN_DIR}")"
else
  RUN_STAMP="$(date +'%Y%m%d_%H%M%S')"
  if [[ "${MODE}" == "preflight" ]]; then
    RUN_NAME="tabero_firm_pi0_base_full_lora_fp32_50k_preflight_${RUN_STAMP}"
  else
    RUN_NAME="tabero_firm_pi0_base_full_lora_fp32_50k_${RUN_STAMP}"
  fi
  RUN_DIR="${WORKSPACE_ROOT}/results/${RUN_NAME}"
  RESUME_DIR=""
  mkdir -p "${RUN_DIR}"
fi

export TABERO_FIRM_50K_RUN_NAME="${RUN_NAME}"
export TABERO_FIRM_50K_RUN_DIR="${RUN_DIR}"
export TABERO_FIRM_50K_CHECKPOINT_ROOT="${RUN_DIR}/checkpoints"
NORM_DIR="${RUN_DIR}/norm_stats"
export TABERO_FIRM_NORM_STATS="${NORM_DIR}/norm_stats.json"
export EMBODIED_PATH="${SCRIPT_DIR}"
export REPO_PATH="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_CSV}"
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1

audit_gpus() {
  local gpu_id compute_pids used_memory
  for gpu_id in "${PHYSICAL_GPUS[@]}"; do
    nvidia-smi --id="${gpu_id}" --query-gpu=index \
      --format=csv,noheader,nounits >/dev/null 2>&1 || \
      die "required GPU ${gpu_id} is not installed"
    compute_pids="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid \
      --format=csv,noheader,nounits)"
    if [[ -n "${compute_pids//[[:space:]]/}" ]]; then
      printf 'GPU %s compute processes:\n' "${gpu_id}" >&2
      while IFS= read -r compute_pid; do
        compute_pid="${compute_pid//[[:space:]]/}"
        [[ -n "${compute_pid}" ]] || continue
        ps -o user,pid,ppid,etime,cmd -p "${compute_pid}" >&2 || true
      done <<<"${compute_pids}"
      die "GPU ${gpu_id} is in use"
    fi
    used_memory="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${used_memory}" =~ ^[0-9]+$ ]] || \
      die "could not read GPU ${gpu_id} memory"
    ((used_memory < 1024)) || \
      die "GPU ${gpu_id} has ${used_memory} MiB allocated without a visible compute PID"
  done
}

audit_gpus

FREE_DISK_KIB="$(df -Pk "${WORKSPACE_ROOT}/results" | awk 'NR == 2 {print $4}')"
[[ "${FREE_DISK_KIB}" =~ ^[0-9]+$ ]] || die "could not read free disk space"
((FREE_DISK_KIB >= MIN_FREE_DISK_KIB)) || \
  die "at least ${MIN_FREE_DISK_KIB} KiB free disk is required"

if [[ "${MODE}" == "preflight" ]]; then
  export WANDB_MODE=offline
else
  export WANDB_MODE=online
  "${REPO_ROOT}/.venv/bin/python" - <<'PY' || \
    die "W&B online authentication or connectivity check failed"
import wandb

if not wandb.login(verify=True, timeout=20):
    raise SystemExit(1)
PY
fi

mkdir -p "${NORM_DIR}" "${TABERO_FIRM_50K_CHECKPOINT_ROOT}"
if [[ "${MODE}" == "resume" ]]; then
  [[ -f "${TABERO_FIRM_NORM_STATS}" ]] || \
    die "resume norm stats are missing: ${TABERO_FIRM_NORM_STATS}"
  [[ -f "${NORM_DIR}/norm_stats.sha256" ]] || \
    die "resume norm stats checksum is missing"
  (cd "${NORM_DIR}" && sha256sum --check norm_stats.sha256) || \
    die "resume norm stats checksum mismatch"
else
  if [[ -n "${TABERO_FIRM_EXISTING_NORM_STATS:-}" ]]; then
    [[ -f "${TABERO_FIRM_EXISTING_NORM_STATS}" ]] || \
      die "TABERO_FIRM_EXISTING_NORM_STATS does not exist"
    cp -- "${TABERO_FIRM_EXISTING_NORM_STATS}" "${TABERO_FIRM_NORM_STATS}"
    printf 'Reused Firm-only norm stats from %s\n' \
      "${TABERO_FIRM_EXISTING_NORM_STATS}" | tee "${RUN_DIR}/norm_stats.log"
  else
    "${REPO_ROOT}/.venv/bin/python" \
      "${REPO_ROOT}/toolkits/lerobot/calculate_norm_stats.py" \
      --config-name pi0_lora_tacfield_tabero \
      --repo-id "${DATASET_PATH}" \
      --output-dir "${NORM_DIR}" \
      2>&1 | tee "${RUN_DIR}/norm_stats.log"
  fi
  NORM_SHA256="$(sha256sum "${TABERO_FIRM_NORM_STATS}" | awk '{print $1}')"
  printf '%s  norm_stats.json\n' "${NORM_SHA256}" >"${NORM_DIR}/norm_stats.sha256"
  "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/toolkits/lerobot/audit_tabero_firm_sft_batch.py" \
    --dataset-path "${DATASET_PATH}" \
    --norm-stats-path "${TABERO_FIRM_NORM_STATS}" \
    --output "${RUN_DIR}/batch_audit.json" \
    2>&1 | tee "${RUN_DIR}/batch_audit.log"
fi

{
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'mode=%s\n' "${MODE}"
  printf 'resume_dir=%s\n' "${RESUME_DIR}"
  printf 'norm_stats=%s\n' "${TABERO_FIRM_NORM_STATS}"
  printf 'norm_stats_sha256=%s\n' \
    "$(sha256sum "${TABERO_FIRM_NORM_STATS}" | awk '{print $1}')"
  printf 'free_disk_kib=%s\n' "${FREE_DISK_KIB}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'gpu_mapping=actor_rank0-6_to_physical0,1,3,4,5,6,7\n'
  printf 'NVIDIA_TF32_OVERRIDE=%s\n' "${NVIDIA_TF32_OVERRIDE}"
  printf 'WANDB_MODE=%s\n' "${WANDB_MODE}"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} | tee -a "${RUN_DIR}/launch_meta.txt"
for gpu_id in "${PHYSICAL_GPUS[@]}"; do
  nvidia-smi --id="${gpu_id}" \
    --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
done | tee -a "${RUN_DIR}/gpu_preflight.log"

run_training() {
  local log_name="$1"
  shift
  "${REPO_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/train_vla_sft.py" \
    --config-path "${SCRIPT_DIR}/config" \
    --config-name "${CONFIG_NAME}" \
    "$@" 2>&1 | tee -a "${RUN_DIR}/${log_name}"
}

if [[ "${MODE}" == "preflight" ]]; then
  run_training train_stage1.log \
    runner.max_steps=2 \
    runner.save_interval=2 \
    actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=4
  STEP2_DIR="${TABERO_FIRM_50K_CHECKPOINT_ROOT}/global_step_2"
  [[ -d "${STEP2_DIR}/actor/dcp_checkpoint" ]] || die "step-2 DCP was not saved"
  [[ -f "${STEP2_DIR}/actor/data_state.json" ]] || die "step-2 data state was not saved"
  run_training train_stage2_resume.log \
    runner.resume_dir="${STEP2_DIR}" \
    runner.max_steps=4 \
    runner.save_interval=2 \
    actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=4
  STEP4_DIR="${TABERO_FIRM_50K_CHECKPOINT_ROOT}/global_step_4"
  [[ -d "${STEP4_DIR}/actor/dcp_checkpoint" ]] || die "step-4 DCP was not saved"
  [[ -f "${STEP4_DIR}/actor/data_state.json" ]] || die "step-4 data state was not saved"
  "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/toolkits/checkpoint/audit_tabero_firm_sft_smoke.py" \
    --step1 "${STEP2_DIR}/actor/model_state_dict/trainable_weights.pt" \
    --step2 "${STEP4_DIR}/actor/model_state_dict/trainable_weights.pt" \
    --expected-steps 2 4 \
    --output "${RUN_DIR}/checkpoint_audit.json"
else
  training_overrides=()
  if [[ "${MODE}" == "resume" ]]; then
    training_overrides+=("runner.resume_dir=${RESUME_DIR}")
  fi
  run_training train.log "${training_overrides[@]}"

  FINAL_STEP_DIR="${TABERO_FIRM_50K_CHECKPOINT_ROOT}/global_step_50000"
  FINAL_SIDECAR="${FINAL_STEP_DIR}/actor/model_state_dict/trainable_weights.pt"
  [[ -f "${FINAL_SIDECAR}" ]] || die "final trainable checkpoint is missing"
  EXPORT_DIR="${RUN_DIR}/exports/step_50000"
  "${REPO_ROOT}/.venv/bin/python" -m \
    rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla \
    --train_config_path "${CONFIG_PATH}" \
    --ckpt_path "${FINAL_SIDECAR}" \
    --output_dir "${EXPORT_DIR}" \
    2>&1 | tee "${RUN_DIR}/export.log"
fi

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" | \
  tee -a "${RUN_DIR}/launch_meta.txt"
printf '%s\n' "${RUN_DIR}"
