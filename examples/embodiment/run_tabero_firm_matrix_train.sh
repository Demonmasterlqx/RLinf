#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <rlt|pirl|dsrl> <0|5> <smoke|formal> [--resume-dir PATH] [--restore-only] [--dry-run]" >&2
}

die() {
  echo "error: $*" >&2
  exit 1
}

METHOD="${1:-}"
TASK_ID="${2:-}"
MODE="${3:-}"
[[ "${METHOD}" == "rlt" || "${METHOD}" == "pirl" || "${METHOD}" == "dsrl" ]] || {
  usage
  exit 2
}
[[ "${TASK_ID}" == "0" || "${TASK_ID}" == "5" ]] || {
  usage
  exit 2
}
[[ "${MODE}" == "smoke" || "${MODE}" == "formal" ]] || {
  usage
  exit 2
}
shift 3

RESUME_DIR=""
RESTORE_ONLY=false
DRY_RUN=false
while (($#)); do
  case "$1" in
    --resume-dir)
      (($# >= 2)) || die "--resume-dir requires a path"
      RESUME_DIR="$2"
      shift 2
      ;;
    --restore-only)
      RESTORE_ONLY=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done
if [[ "${RESTORE_ONLY}" == true && -z "${RESUME_DIR}" ]]; then
  die "--restore-only requires --resume-dir"
fi

case "${METHOD}" in
  rlt)
    CONFIG_NAME="tabero_rlt_stage2_ac_task${TASK_ID}_firm"
    CONFIG_DIR_REL="examples/tabero"
    ;;
  pirl)
    CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task${TASK_ID}_firm_8gpu_50step"
    CONFIG_DIR_REL="examples/embodiment/config"
    ;;
  dsrl)
    CONFIG_NAME="isaaclab_pi0_dsrl_tacfield_tabero_task${TASK_ID}_firm_8gpu_50step"
    CONFIG_DIR_REL="examples/embodiment/config"
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_embodied_agent.py"
PROVENANCE_SCRIPT="${SCRIPT_DIR}/tabero_formal_provenance.py"
CONFIG_DIR="${REPO_ROOT}/${CONFIG_DIR_REL}"
CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
ISAAC_SETUP="${REPO_ROOT}/isaac_sim/setup_conda_env.sh"
RESULTS_ROOT="${TABERO_RESULTS_ROOT:-/data/home/sim6g/code/tabero/results}"
MODEL_PATH="/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_safetensors"
MODEL_WEIGHTS_PATH="${MODEL_PATH}/model.safetensors"
GPU_LOCK_DIR="/tmp/tabero-formal-gpu-locks-${UID}"
if [[ -n "${TABERO_GPU_LOCK_DIR:-}" ]]; then
  [[ "${DRY_RUN}" == true ]] || die "TABERO_GPU_LOCK_DIR is only allowed with --dry-run"
  GPU_LOCK_DIR="${TABERO_GPU_LOCK_DIR}"
fi
MIN_FREE_KIB=1073741824
if [[ -n "${TABERO_MIN_FREE_KIB:-}" ]]; then
  [[ "${DRY_RUN}" == true ]] || die "TABERO_MIN_FREE_KIB is only allowed with --dry-run"
  [[ "${TABERO_MIN_FREE_KIB}" =~ ^[1-9][0-9]*$ ]] || die "TABERO_MIN_FREE_KIB must be a positive integer"
  MIN_FREE_KIB="${TABERO_MIN_FREE_KIB}"
fi

[[ -x "${PYTHON_BIN}" ]] || die "RLinf Python not found: ${PYTHON_BIN}"
[[ -f "${TRAIN_SCRIPT}" ]] || die "training entrypoint not found: ${TRAIN_SCRIPT}"
[[ -f "${PROVENANCE_SCRIPT}" ]] || die "provenance helper not found: ${PROVENANCE_SCRIPT}"
[[ -f "${CONFIG_PATH}" ]] || die "training config not found: ${CONFIG_PATH}"
[[ -f "${ISAAC_SETUP}" ]] || die "Isaac environment setup not found: ${ISAAC_SETUP}"
[[ -e "${MODEL_PATH}" ]] || die "base model not found: ${MODEL_PATH}"
[[ -f "${MODEL_WEIGHTS_PATH}" ]] || die "base model weights not found: ${MODEL_WEIGHTS_PATH}"

set +u
source "${ISAAC_SETUP}"
set -u

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
declare -A installed_gpus=()
while IFS= read -r gpu_id; do
  gpu_id="${gpu_id// /}"
  gpu_id="${gpu_id//$'\r'/}"
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "nvidia-smi returned invalid GPU index: ${gpu_id}"
  installed_gpus["${gpu_id}"]=1
done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
[[ "${#installed_gpus[@]}" -gt 0 ]] || die "nvidia-smi reported no installed GPUs"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT="${#visible_gpus[@]}"
  declare -A seen_gpus=()
  for gpu_id in "${visible_gpus[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must contain physical numeric GPU IDs"
    [[ -z "${seen_gpus[${gpu_id}]:-}" ]] || die "CUDA_VISIBLE_DEVICES must contain 8 unique physical GPU IDs"
    [[ -n "${installed_gpus[${gpu_id}]:-}" ]] || die "CUDA_VISIBLE_DEVICES GPU ${gpu_id} is not installed"
    seen_gpus["${gpu_id}"]=1
  done
else
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT="${#visible_gpus[@]}"
fi
[[ "${GPU_COUNT}" -eq 8 ]] || die "exactly 8 visible GPUs are required; found ${GPU_COUNT}"
command -v flock >/dev/null 2>&1 || die "flock is required for exclusive formal runs"
mkdir -p "${GPU_LOCK_DIR}"
mapfile -t sorted_visible_gpus < <(printf '%s\n' "${visible_gpus[@]}" | sort -n)
GPU_LOCK_FDS=()
for gpu_id in "${sorted_visible_gpus[@]}"; do
  exec {gpu_lock_fd}>"${GPU_LOCK_DIR}/gpu_${gpu_id}.lock"
  flock -n "${gpu_lock_fd}" || die "GPU ${gpu_id} lease is already held by another formal launcher"
  GPU_LOCK_FDS+=("${gpu_lock_fd}")
done
for gpu_id in "${sorted_visible_gpus[@]}"; do
  compute_pids="$(
    nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits
  )" || die "failed to query compute processes for GPU ${gpu_id}"
  while IFS= read -r compute_pid; do
    compute_pid="${compute_pid// /}"
    compute_pid="${compute_pid//$'\r'/}"
    [[ -z "${compute_pid}" ]] && continue
    [[ "${compute_pid}" =~ ^[0-9]+$ ]] || die "nvidia-smi returned invalid compute PID for GPU ${gpu_id}: ${compute_pid}"
    die "GPU ${gpu_id} is busy with compute PID ${compute_pid}; formal training requires exclusive GPUs"
  done <<<"${compute_pids}"
done

load_run_env() {
  local env_file="$1" key value field_count=0
  declare -A seen=()
  while IFS='=' read -r key value; do
    [[ -n "${key}" && -n "${value}" ]] || die "invalid run.env: empty key or value"
    [[ "${value}" =~ ^[A-Za-z0-9_./:+-]+$ ]] || die "invalid run.env value for ${key}"
    [[ -z "${seen[${key}]:-}" ]] || die "invalid run.env: duplicate ${key}"
    seen["${key}"]=1
    case "${key}" in
      TABERO_MATRIX_RUN_ID) STORED_RUN_ID="${value}" ;;
      WANDB_RUN_ID) STORED_WANDB_RUN_ID="${value}" ;;
      WANDB_RESUME) STORED_WANDB_RESUME="${value}" ;;
      TABERO_METHOD) STORED_METHOD="${value}" ;;
      TABERO_TASK_ID) STORED_TASK_ID="${value}" ;;
      TABERO_RUN_MODE) STORED_MODE="${value}" ;;
      TABERO_OUTPUT_DIR) STORED_OUTPUT_DIR="${value}" ;;
      TABERO_CONFIG_NAME) STORED_CONFIG_NAME="${value}" ;;
      TABERO_CONFIG_DIR) STORED_CONFIG_DIR="${value}" ;;
      TABERO_EXPERIMENT_NAME) STORED_EXPERIMENT_NAME="${value}" ;;
      TABERO_START_TIME_UTC) STORED_START_UTC="${value}" ;;
      TABERO_START_TIME_LOCAL) STORED_START_LOCAL="${value}" ;;
      TABERO_LAUNCH_KIND) STORED_LAUNCH_KIND="${value}" ;;
      *) die "invalid run.env key: ${key}" ;;
    esac
    ((field_count += 1))
  done <"${env_file}"
  local required_key
  for required_key in \
    TABERO_MATRIX_RUN_ID WANDB_RUN_ID WANDB_RESUME TABERO_METHOD TABERO_TASK_ID \
    TABERO_RUN_MODE TABERO_OUTPUT_DIR TABERO_CONFIG_NAME TABERO_CONFIG_DIR \
    TABERO_EXPERIMENT_NAME TABERO_START_TIME_UTC TABERO_START_TIME_LOCAL; do
    [[ -n "${seen[${required_key}]:-}" ]] || die "invalid run.env: missing ${required_key}"
  done
  [[ "${field_count}" -eq 12 || "${field_count}" -eq 13 ]] || \
    die "invalid run.env: expected 12 legacy or 13 current fields"
  if [[ "${field_count}" -eq 12 ]]; then
    [[ -z "${seen[TABERO_LAUNCH_KIND]:-}" ]] || die "invalid legacy run.env keyspace"
    RUN_ENV_FORMAT="legacy"
  else
    [[ -n "${seen[TABERO_LAUNCH_KIND]:-}" ]] || die "invalid current run.env keyspace"
    RUN_ENV_FORMAT="current"
  fi
}

START_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_TIME_LOCAL="$(date +%Y-%m-%dT%H:%M:%S%z)"
START_EPOCH="$(date +%s)"
LAUNCH_KIND="new"

if [[ -n "${RESUME_DIR}" ]]; then
  [[ -d "${RESUME_DIR}" ]] || die "resume directory not found: ${RESUME_DIR}"
  RESUME_DIR="$(cd "${RESUME_DIR}" && pwd)"
  checkpoint_name="$(basename "${RESUME_DIR}")"
  [[ "${checkpoint_name}" =~ ^global_step_([1-9][0-9]*)$ ]] ||     die "resume directory basename must match global_step_<positive integer>"
  CHECKPOINT_STEP="${BASH_REMATCH[1]}"
  [[ -d "${RESUME_DIR}/actor" ]] || die "resume checkpoint has no actor directory"

  search_dir="${RESUME_DIR}"
  RUN_ENV=""
  while [[ "${search_dir}" != "/" ]]; do
    if [[ -f "${search_dir}/run.env" ]]; then
      RUN_ENV="${search_dir}/run.env"
      break
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  [[ -n "${RUN_ENV}" ]] || die "no run.env associated with resume directory"
  load_run_env "${RUN_ENV}"
  [[ "${STORED_METHOD}" == "${METHOD}" ]] || die "resume method does not match run.env"
  [[ "${STORED_TASK_ID}" == "${TASK_ID}" ]] || die "resume task does not match run.env"
  [[ "${STORED_MODE}" == "${MODE}" ]] || die "resume mode does not match run.env"
  [[ "${STORED_CONFIG_NAME}" == "${CONFIG_NAME}" ]] || die "resume config does not match run.env"
  [[ "${STORED_CONFIG_DIR}" == "${CONFIG_DIR_REL}" ]] || die "resume config directory does not match run.env"

  TABERO_MATRIX_RUN_ID="${STORED_RUN_ID}"
  WANDB_RUN_ID="${STORED_WANDB_RUN_ID}"
  WANDB_RESUME="must"
  OUTPUT_DIR="${STORED_OUTPUT_DIR}"
  EXPERIMENT_NAME="${STORED_EXPERIMENT_NAME}"
  [[ "${RUN_ENV}" == "${OUTPUT_DIR}/run.env" ]] || die "run.env output directory mismatch"
  expected_resume="${OUTPUT_DIR}/${EXPERIMENT_NAME}/checkpoints/${checkpoint_name}"
  [[ "${RESUME_DIR}" == "${expected_resume}" ]] || die "resume directory must be ${expected_resume}"
  LAUNCH_KIND="$([[ "${RESTORE_ONLY}" == true ]] && printf restore_only || printf resume)"
else
  TABERO_MATRIX_RUN_ID="${TABERO_MATRIX_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${MODE}}"
  [[ "${TABERO_MATRIX_RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}_${MODE}$ ]] ||     die "TABERO_MATRIX_RUN_ID must match YYYYMMDD_HHMMSS_${MODE}"
  EXPERIMENT_NAME="tabero_firm_matrix_${METHOD}_task${TASK_ID}_${MODE}_${TABERO_MATRIX_RUN_ID}"
  OUTPUT_DIR="${RESULTS_ROOT}/${EXPERIMENT_NAME}"
  mkdir -p "${RESULTS_ROOT}"
  mkdir "${OUTPUT_DIR}" 2>/dev/null || die "output directory already exists: ${OUTPUT_DIR}"
  RUN_ENV="${OUTPUT_DIR}/run.env"
  WANDB_RUN_ID="${WANDB_RUN_ID:-$("${PYTHON_BIN}" -c 'import wandb; print(wandb.util.generate_id())')}"
  [[ "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid W&B run ID"
  WANDB_RESUME="allow"
  COMMAND_FILE="${OUTPUT_DIR}/command.txt"
  LOG_FILE="${OUTPUT_DIR}/train.log"
  STATUS_FILE="${OUTPUT_DIR}/run_status.env"

  metadata_tmp="$(mktemp "${OUTPUT_DIR}/.run.env.XXXXXX")"
  {
    printf 'TABERO_MATRIX_RUN_ID=%s\n' "${TABERO_MATRIX_RUN_ID}"
    printf 'WANDB_RUN_ID=%s\n' "${WANDB_RUN_ID}"
    printf 'WANDB_RESUME=%s\n' "${WANDB_RESUME}"
    printf 'TABERO_METHOD=%s\n' "${METHOD}"
    printf 'TABERO_TASK_ID=%s\n' "${TASK_ID}"
    printf 'TABERO_RUN_MODE=%s\n' "${MODE}"
    printf 'TABERO_OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
    printf 'TABERO_CONFIG_NAME=%s\n' "${CONFIG_NAME}"
    printf 'TABERO_CONFIG_DIR=%s\n' "${CONFIG_DIR_REL}"
    printf 'TABERO_EXPERIMENT_NAME=%s\n' "${EXPERIMENT_NAME}"
    printf 'TABERO_START_TIME_UTC=%s\n' "${START_TIME_UTC}"
    printf 'TABERO_START_TIME_LOCAL=%s\n' "${START_TIME_LOCAL}"
    printf 'TABERO_LAUNCH_KIND=new\n'
  } >"${metadata_tmp}"
  mv "${metadata_tmp}" "${RUN_ENV}"
  RUN_ENV_FORMAT="current"
fi

exec {OUTPUT_LOCK_FD}>"${OUTPUT_DIR}/.formal_run.lock"
flock -n "${OUTPUT_LOCK_FD}" || die "formal output lease is already held: ${OUTPUT_DIR}"

FREE_DISK_KIB="$(df -Pk "${OUTPUT_DIR}" | awk 'NR == 2 {print $4}')"
[[ "${FREE_DISK_KIB}" =~ ^[0-9]+$ ]] || die "failed to determine free disk space on ${OUTPUT_DIR}"
[[ "${FREE_DISK_KIB}" -ge "${MIN_FREE_KIB}" ]] || \
  die "insufficient disk space on ${OUTPUT_DIR}: ${FREE_DISK_KIB} KiB free, require ${MIN_FREE_KIB} KiB"

if [[ -n "${TABERO_TEST_MODEL_SHA256:-}" ]]; then
  [[ "${DRY_RUN}" == true ]] || die "TABERO_TEST_MODEL_SHA256 is only allowed with --dry-run"
  BASE_MODEL_SHA256="${TABERO_TEST_MODEL_SHA256}"
else
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
  BASE_MODEL_SHA256="$(sha256sum "${MODEL_WEIGHTS_PATH}" | awk '{print $1}')"
fi
[[ "${BASE_MODEL_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid base model SHA-256"

provenance_args=(
  --output-dir "${OUTPUT_DIR}"
  --config-path "${CONFIG_PATH}"
  --model-path "${MODEL_PATH}"
  --base-model-sha256 "${BASE_MODEL_SHA256}"
  --repo-root "${REPO_ROOT}"
)
if [[ "${DRY_RUN}" == true ]]; then
  provenance_args+=(--allow-dirty)
fi
if [[ -n "${RESUME_DIR}" ]]; then
  if [[ -f "${OUTPUT_DIR}/provenance.env" || -f "${OUTPUT_DIR}/config_snapshot.yaml" ]]; then
    "${PYTHON_BIN}" "${PROVENANCE_SCRIPT}" verify "${provenance_args[@]}"
  elif [[ "${RUN_ENV_FORMAT}" == "legacy" ]]; then
    "${PYTHON_BIN}" "${PROVENANCE_SCRIPT}" capture "${provenance_args[@]}" \
      --legacy-source-config "${OUTPUT_DIR}/tensorboard/config.yaml"
  else
    die "current run.env requires provenance.env and config_snapshot.yaml"
  fi

  artifact_base="$(date +%Y%m%d_%H%M%S)_${BASHPID}_${LAUNCH_KIND}"
  artifact_id="${artifact_base}"
  artifact_collision=0
  while [[ -e "${OUTPUT_DIR}/${LAUNCH_KIND}_command_${artifact_id}.txt" || \
    -e "${OUTPUT_DIR}/${LAUNCH_KIND}_${artifact_id}.log" || \
    -e "${OUTPUT_DIR}/${LAUNCH_KIND}_status_${artifact_id}.env" ]]; do
    ((artifact_collision += 1))
    artifact_id="${artifact_base}_${artifact_collision}"
  done
  COMMAND_FILE="${OUTPUT_DIR}/${LAUNCH_KIND}_command_${artifact_id}.txt"
  LOG_FILE="${OUTPUT_DIR}/${LAUNCH_KIND}_${artifact_id}.log"
  STATUS_FILE="${OUTPUT_DIR}/${LAUNCH_KIND}_status_${artifact_id}.env"
else
  "${PYTHON_BIN}" "${PROVENANCE_SCRIPT}" capture "${provenance_args[@]}"
fi

export TABERO_MATRIX_RUN_ID WANDB_RUN_ID WANDB_RESUME CUDA_VISIBLE_DEVICES
export EMBODIED_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

command=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  --config-path "${CONFIG_DIR}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${OUTPUT_DIR}"
  "runner.logger.experiment_name=${EXPERIMENT_NAME}"
)
if [[ "${MODE}" == "smoke" ]]; then
  command+=("runner.max_epochs=1" "runner.save_interval=1")
fi
if [[ -n "${RESUME_DIR}" ]]; then
  command+=("runner.resume_dir=${RESUME_DIR}")
fi
if [[ "${RESTORE_ONLY}" == true ]]; then
  command+=("runner.max_epochs=${CHECKPOINT_STEP}" "runner.save_interval=-1")
fi

{
  printf 'TABERO_MATRIX_RUN_ID=%s\n' "${TABERO_MATRIX_RUN_ID}"
  printf 'WANDB_RUN_ID=%s\n' "${WANDB_RUN_ID}"
  printf 'WANDB_RESUME=%s\n' "${WANDB_RESUME}"
  printf 'TABERO_LAUNCH_KIND=%s\n' "${LAUNCH_KIND}"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
} >"${COMMAND_FILE}"

printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Log file: %s\n' "${LOG_FILE}"
cat "${COMMAND_FILE}"

if [[ "${DRY_RUN}" == true ]]; then
  exit 0
fi

{
  printf 'TABERO_RUN_STATUS=running\n'
  printf 'TABERO_EXIT_STATUS=\n'
  printf 'TABERO_START_TIME_UTC=%s\n' "${START_TIME_UTC}"
  printf 'TABERO_START_TIME_LOCAL=%s\n' "${START_TIME_LOCAL}"
  printf 'TABERO_LAUNCH_KIND=%s\n' "${LAUNCH_KIND}"
} >"${STATUS_FILE}"

GPU_SAMPLER_PID=""
stop_gpu_sampler() {
  if [[ -n "${GPU_SAMPLER_PID}" ]] && kill -0 "${GPU_SAMPLER_PID}" 2>/dev/null; then
    kill "${GPU_SAMPLER_PID}" 2>/dev/null || true
    wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
  fi
  GPU_SAMPLER_PID=""
}
start_gpu_sampler() {
  local gpu_file="${OUTPUT_DIR}/gpu_samples.csv"
  local process_file="${OUTPUT_DIR}/gpu_process_samples.csv"
  [[ -s "${gpu_file}" ]] || printf '%s\n'     'timestamp_utc,index,name,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_w,pstate' >"${gpu_file}"
  [[ -s "${process_file}" ]] || printf '%s\n'     'timestamp_utc,gpu_uuid,pid,process_name,used_gpu_memory_mib' >"${process_file}"
  (
    while true; do
      sample_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      while IFS= read -r row; do printf '%s,%s\n' "${sample_time}" "${row}"; done < <(
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,pstate --format=csv,noheader,nounits
      ) >>"${gpu_file}"
      while IFS= read -r row; do
        [[ -n "${row}" ]] && printf '%s,%s\n' "${sample_time}" "${row}"
      done < <(
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
      ) >>"${process_file}"
      sleep 5
    done
  ) &
  GPU_SAMPLER_PID=$!
}

start_gpu_sampler
trap stop_gpu_sampler EXIT
set +e
"${command[@]}" 2>&1 | tee "${LOG_FILE}"
TRAIN_STATUS="${PIPESTATUS[0]}"
set -e
stop_gpu_sampler
trap - EXIT
END_EPOCH="$(date +%s)"
if [[ "${TRAIN_STATUS}" -eq 0 ]]; then
  RUN_STATUS="completed"
else
  RUN_STATUS="failed"
fi
{
  printf 'TABERO_RUN_STATUS=%s\n' "${RUN_STATUS}"
  printf 'TABERO_EXIT_STATUS=%s\n' "${TRAIN_STATUS}"
  printf 'TABERO_END_TIME_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'TABERO_END_TIME_LOCAL=%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)"
  printf 'TABERO_DURATION_SECONDS=%s\n' "$((END_EPOCH - START_EPOCH))"
  printf 'TABERO_LAUNCH_KIND=%s\n' "${LAUNCH_KIND}"
} >"${STATUS_FILE}"
exit "${TRAIN_STATUS}"
