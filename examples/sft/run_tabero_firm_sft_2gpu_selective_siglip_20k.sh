#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s [preflight|formal|resume]\n' "$0" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

MODE="${1:-formal}"
[[ $# -le 1 ]] || { usage; die "too many arguments"; }
case "${MODE}" in
  preflight|formal|resume) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    die "mode must be preflight, formal, or resume"
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
CONFIG_NAME="tabero_firm_sft_2gpu_selective_siglip_20k"
CONFIG_PATH="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"
MODEL_PATH="${WORKSPACE_ROOT}/models/pi0_base/model.safetensors"
REFERENCE_PATH="${WORKSPACE_ROOT}/models/pi0_lora_tacfield_tabero_safetensors/model.safetensors"
DATASET_PATH="${WORKSPACE_ROOT}/datas/tabero_firm"
MIN_FREE_DISK_KIB=524288000
PHYSICAL_GPUS=(0 1)
PHYSICAL_GPU_CSV="0,1"

[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || die "RLinf Python is missing"
[[ -f "${CONFIG_PATH}" ]] || die "training config is missing"
[[ -f "${MODEL_PATH}" ]] || die "base model is missing"
[[ -f "${REFERENCE_PATH}" ]] || die "export reference model is missing"
[[ -f "${DATASET_PATH}/meta/info.json" ]] || die "Firm dataset is missing"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

RUN_STAMP="${TABERO_FIRM_2GPU_RUN_STAMP:-$(date +'%Y%m%d_%H%M%S')}"
if [[ "${MODE}" == "resume" ]]; then
  RESUME_DIR="${TABERO_FIRM_2GPU_RESUME_DIR:-}"
  [[ -n "${RESUME_DIR}" && "${RESUME_DIR}" == /* ]] || \
    die "TABERO_FIRM_2GPU_RESUME_DIR must be an absolute checkpoint path"
  [[ "$(basename "${RESUME_DIR}")" =~ ^global_step_[0-9]+$ ]] || \
    die "resume directory must end in global_step_N"
  RUN_DIR="$(cd "$(dirname "$(dirname "${RESUME_DIR}")")" && pwd -P)"
  RUN_NAME="$(basename "${RUN_DIR}")"
else
  RESUME_DIR=""
  if [[ "${MODE}" == "preflight" ]]; then
    RUN_NAME="tabero_firm_rlinf_2gpu_mb16_20k_preflight_${RUN_STAMP}"
  else
    RUN_NAME="tabero_firm_rlinf_2gpu_mb16_20k_${RUN_STAMP}"
  fi
  RUN_DIR="${WORKSPACE_ROOT}/results/${RUN_NAME}"
  [[ ! -e "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"
  mkdir -p "${RUN_DIR}"
fi

export TABERO_FIRM_2GPU_20K_RUN_DIR="${RUN_DIR}"
export TABERO_FIRM_2GPU_20K_RUN_NAME="${RUN_NAME}"
export TABERO_FIRM_2GPU_20K_CHECKPOINT_ROOT="${RUN_DIR}/checkpoints"
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
    [[ "${used_memory}" =~ ^[0-9]+$ ]] || die "could not read GPU ${gpu_id} memory"
    ((used_memory < 1024)) || die "GPU ${gpu_id} already uses ${used_memory} MiB"
  done
}

audit_checkpoint_layout() {
  local step_dir="$1"
  [[ -f "${step_dir}/actor/dcp_checkpoint/.metadata" ]] || \
    die "DCP metadata is missing: ${step_dir}"
  [[ -n "$(find "${step_dir}/actor/dcp_checkpoint" -maxdepth 1 \
    -name '*.distcp' -type f -print -quit)" ]] || die "DCP shards are missing: ${step_dir}"
  [[ -f "${step_dir}/actor/data_state.json" ]] || \
    die "OpenPI data state is missing: ${step_dir}"
  [[ -f "${step_dir}/actor/model_state_dict/trainable_weights.pt" ]] || \
    die "trainable sidecar is missing: ${step_dir}"
}

audit_gpus

FREE_DISK_KIB="$(df -Pk "${WORKSPACE_ROOT}/results" | awk 'NR == 2 {print $4}')"
[[ "${FREE_DISK_KIB}" =~ ^[0-9]+$ ]] || die "could not read free disk space"
((FREE_DISK_KIB >= MIN_FREE_DISK_KIB)) || \
  die "at least ${MIN_FREE_DISK_KIB} KiB free disk is required"

mkdir -p "${NORM_DIR}" "${TABERO_FIRM_2GPU_20K_CHECKPOINT_ROOT}"
if [[ "${MODE}" == "resume" ]]; then
  audit_checkpoint_layout "${RESUME_DIR}"
  [[ -f "${NORM_DIR}/norm_stats.sha256" ]] || die "norm stats checksum is missing"
  (cd "${NORM_DIR}" && sha256sum --check norm_stats.sha256) || \
    die "norm stats checksum mismatch"
else
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
      --output-dir "${NORM_DIR}" \
      2>&1 | tee "${RUN_DIR}/norm_stats.log"
  fi
  (cd "${NORM_DIR}" && sha256sum norm_stats.json >norm_stats.sha256)
fi

"${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/toolkits/lerobot/audit_tabero_firm_sft_batch.py" \
  --dataset-path "${DATASET_PATH}" \
  --norm-stats-path "${TABERO_FIRM_NORM_STATS}" \
  --output "${RUN_DIR}/batch_audit.json" \
  2>&1 | tee "${RUN_DIR}/batch_audit.log"

if [[ "${MODE}" == "preflight" ]]; then
  export WANDB_MODE=offline
else
  REQUESTED_WANDB_MODE="${TABERO_FIRM_WANDB_MODE:-online}"
  if [[ "${REQUESTED_WANDB_MODE}" == "online" ]]; then
    set +e
    "${REPO_ROOT}/.venv/bin/python" - <<'PY' >"${RUN_DIR}/wandb_auth.log" 2>&1
import wandb

raise SystemExit(0 if wandb.login(verify=True, timeout=20) else 1)
PY
    WANDB_AUTH_EXIT=$?
    set -e
    if ((WANDB_AUTH_EXIT == 0)); then
      export WANDB_MODE=online
    else
      export WANDB_MODE=offline
      printf 'W&B online authentication failed; using offline mode.\n' | \
        tee -a "${RUN_DIR}/wandb_auth.log"
    fi
  else
    export WANDB_MODE="${REQUESTED_WANDB_MODE}"
  fi
fi

if [[ -f "${RUN_DIR}/wandb_run_id" ]]; then
  export WANDB_RUN_ID="$(tr -d '[:space:]' <"${RUN_DIR}/wandb_run_id")"
  export WANDB_RESUME=allow
else
  export WANDB_RUN_ID="$("${REPO_ROOT}/.venv/bin/python" -c \
    'import uuid; print(uuid.uuid4().hex[:8])')"
  printf '%s\n' "${WANDB_RUN_ID}" >"${RUN_DIR}/wandb_run_id"
fi

{
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'mode=%s\n' "${MODE}"
  printf 'resume_dir=%s\n' "${RESUME_DIR}"
  printf 'config=%s\n' "${CONFIG_NAME}"
  printf 'physical_gpus=0,1\n'
  printf 'global_batch_size=32\n'
  printf 'micro_batch_size=16\n'
  printf 'gradient_accumulation=1\n'
  printf 'checkpoint_interval=2000\n'
  printf 'frozen_parameter_precision=bf16\n'
  printf 'trainable_parameter_precision=fp32\n'
  printf 'compute_precision=bf16_amp\n'
  printf 'WANDB_MODE=%s\n' "${WANDB_MODE}"
  printf 'WANDB_RUN_ID=%s\n' "${WANDB_RUN_ID}"
  printf 'code_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  printf 'code_dirty=%s\n' "$(git -C "${REPO_ROOT}" status --porcelain | wc -l)"
  printf 'free_disk_kib=%s\n' "${FREE_DISK_KIB}"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} | tee -a "${RUN_DIR}/launch_meta.txt"

nvidia-smi --id=0,1 \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader | tee -a "${RUN_DIR}/gpu_preflight.log"

run_training() {
  local log_name="$1"
  shift
  audit_gpus
  "${REPO_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/train_vla_sft.py" \
    --config-path "${SCRIPT_DIR}/config" \
    --config-name "${CONFIG_NAME}" \
    "$@" 2>&1 | tee -a "${RUN_DIR}/${log_name}"
}

if [[ "${MODE}" == "preflight" ]]; then
  run_training train_stage1.log \
    runner.max_steps=2 \
    runner.save_interval=2 \
    actor.optim.total_training_steps=4 \
    actor.optim.lr_warmup_steps=0 \
    actor.optim.lr_decay_steps=4 \
    actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=4
  STEP2_DIR="${TABERO_FIRM_2GPU_20K_CHECKPOINT_ROOT}/global_step_2"
  audit_checkpoint_layout "${STEP2_DIR}"
  run_training train_stage2_resume.log \
    runner.resume_dir="${STEP2_DIR}" \
    runner.max_steps=4 \
    runner.save_interval=2 \
    actor.optim.total_training_steps=4 \
    actor.optim.lr_warmup_steps=0 \
    actor.optim.lr_decay_steps=4 \
    actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=4
  STEP4_DIR="${TABERO_FIRM_2GPU_20K_CHECKPOINT_ROOT}/global_step_4"
  audit_checkpoint_layout "${STEP4_DIR}"
  "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/toolkits/checkpoint/audit_tabero_firm_sft_smoke.py" \
    --step1 "${STEP2_DIR}/actor/model_state_dict/trainable_weights.pt" \
    --step2 "${STEP4_DIR}/actor/model_state_dict/trainable_weights.pt" \
    --expected-steps 2 4 \
    --expect-outer-trainable \
    --output "${RUN_DIR}/checkpoint_audit.json"
  EXPORT_DIR="${RUN_DIR}/exports/step_4"
  "${REPO_ROOT}/.venv/bin/python" -m \
    rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla \
    --train_config_path "${CONFIG_PATH}" \
    --ckpt_path "${STEP4_DIR}/actor/model_state_dict/trainable_weights.pt" \
    --output_dir "${EXPORT_DIR}" 2>&1 | tee "${RUN_DIR}/export.log"
  [[ -f "${EXPORT_DIR}/export_meta.json" ]] || die "preflight export metadata is missing"
  sha256sum "${EXPORT_DIR}/model.safetensors" >"${EXPORT_DIR}/model.safetensors.sha256"
else
  TRAIN_LOG_NAME="${TABERO_FIRM_TRAIN_LOG_NAME:-train.log}"
  TRAIN_OVERRIDES=()
  if [[ "${MODE}" == "resume" ]]; then
    TRAIN_OVERRIDES+=("runner.resume_dir=${RESUME_DIR}")
  fi
  run_training "${TRAIN_LOG_NAME}" "${TRAIN_OVERRIDES[@]}"

  FINAL_STEP_DIR="${TABERO_FIRM_2GPU_20K_CHECKPOINT_ROOT}/global_step_20000"
  audit_checkpoint_layout "${FINAL_STEP_DIR}"
  FINAL_SIDECAR="${FINAL_STEP_DIR}/actor/model_state_dict/trainable_weights.pt"
  "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/toolkits/checkpoint/audit_tabero_firm_sft_checkpoint.py" \
    --checkpoint "${FINAL_SIDECAR}" \
    --expected-step 20000 \
    --expect-final \
    --output "${RUN_DIR}/final_checkpoint_audit.json"
  EXPORT_DIR="${RUN_DIR}/exports/step_20000"
  [[ ! -e "${EXPORT_DIR}" ]] || die "final export directory already exists"
  "${REPO_ROOT}/.venv/bin/python" -m \
    rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla \
    --train_config_path "${CONFIG_PATH}" \
    --ckpt_path "${FINAL_SIDECAR}" \
    --output_dir "${EXPORT_DIR}" 2>&1 | tee "${RUN_DIR}/export.log"
  [[ -f "${EXPORT_DIR}/export_meta.json" ]] || die "final export metadata is missing"
  sha256sum "${EXPORT_DIR}/model.safetensors" >"${EXPORT_DIR}/model.safetensors.sha256"
fi

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" | \
  tee -a "${RUN_DIR}/launch_meta.txt"
printf '%s\n' "${RUN_DIR}"
