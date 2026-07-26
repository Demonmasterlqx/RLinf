#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_NAME="isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_smoke"
readonly MODEL_PATH="/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_safetensors"
readonly HDF5_PATH="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5/libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"

usage() {
  echo "Usage: $0 [--resume-dir PATH] [--dry-run]" >&2
}

die() {
  echo "error: $*" >&2
  exit 1
}

GPU_SAMPLER_PID=""

start_gpu_sampler() {
  GPU_SAMPLES_FILE="${OUTPUT_DIR}/gpu_samples.csv"
  GPU_PROCESS_SAMPLES_FILE="${OUTPUT_DIR}/gpu_process_samples.csv"
  if [[ ! -s "${GPU_SAMPLES_FILE}" ]]; then
    printf '%s\n' \
      'timestamp_utc,index,name,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_w,pstate' \
      >"${GPU_SAMPLES_FILE}"
  fi
  if [[ ! -s "${GPU_PROCESS_SAMPLES_FILE}" ]]; then
    printf '%s\n' \
      'timestamp_utc,gpu_uuid,pid,process_name,used_gpu_memory_mib' \
      >"${GPU_PROCESS_SAMPLES_FILE}"
  fi

  (
    while true; do
      sample_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      while IFS= read -r gpu_row; do
        printf '%s,%s\n' "${sample_time}" "${gpu_row}"
      done < <(
        nvidia-smi \
          --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,pstate \
          --format=csv,noheader,nounits
      ) >>"${GPU_SAMPLES_FILE}"
      while IFS= read -r process_row; do
        [[ -n "${process_row}" ]] || continue
        printf '%s,%s\n' "${sample_time}" "${process_row}"
      done < <(
        nvidia-smi \
          --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
          --format=csv,noheader,nounits 2>/dev/null || true
      ) >>"${GPU_PROCESS_SAMPLES_FILE}"
      sleep 5
    done
  ) &
  GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
  if [[ -n "${GPU_SAMPLER_PID}" ]] && kill -0 "${GPU_SAMPLER_PID}" 2>/dev/null; then
    kill "${GPU_SAMPLER_PID}" 2>/dev/null || true
    wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
  fi
  GPU_SAMPLER_PID=""
}

load_run_env() {
  local env_file="$1" key value
  local field_count=0
  declare -A seen=()

  while IFS='=' read -r key value; do
    [[ -n "${key}" && -n "${value}" ]] || die "invalid run.env: empty key or value"
    [[ "${value}" =~ ^[A-Za-z0-9_./:+-]+$ ]] || die "invalid run.env value for ${key}"
    [[ -z "${seen[${key}]:-}" ]] || die "invalid run.env: duplicate ${key}"
    seen["${key}"]=1
    case "${key}" in
      TABERO_TASK0_DSRL_RUN_ID) TABERO_TASK0_DSRL_RUN_ID="${value}" ;;
      WANDB_RUN_ID) WANDB_RUN_ID="${value}" ;;
      WANDB_RESUME) WANDB_RESUME="${value}" ;;
      TABERO_RUN_KIND) TABERO_RUN_KIND="${value}" ;;
      TABERO_OUTPUT_DIR) TABERO_OUTPUT_DIR="${value}" ;;
      TABERO_CONFIG_NAME) TABERO_CONFIG_NAME="${value}" ;;
      TABERO_START_TIME_UTC) TABERO_START_TIME_UTC="${value}" ;;
      TABERO_START_TIME_LOCAL) TABERO_START_TIME_LOCAL="${value}" ;;
      *) die "invalid run.env key: ${key}" ;;
    esac
    ((field_count += 1))
  done <"${env_file}"

  [[ "${field_count}" -eq 8 ]] || die "invalid run.env: expected 8 fields"
  [[ "${TABERO_TASK0_DSRL_RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}_smoke$ ]] || die "invalid run.env run ID"
  [[ "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid run.env W&B ID"
  [[ "${WANDB_RESUME}" == "allow" || "${WANDB_RESUME}" == "must" ]] || die "invalid run.env W&B resume mode"
  [[ "${TABERO_RUN_KIND}" == "dsrl_smoke" ]] || die "invalid run.env run kind"
  [[ "${TABERO_OUTPUT_DIR}" == /* ]] || die "invalid run.env output directory"
  [[ "${TABERO_CONFIG_NAME}" == "${CONFIG_NAME}" ]] || die "invalid run.env config name"
  [[ "${TABERO_START_TIME_UTC}" =~ ^[0-9T:Z+-]+$ ]] || die "invalid run.env UTC start time"
  [[ "${TABERO_START_TIME_LOCAL}" =~ ^[0-9T:+-]+$ ]] || die "invalid run.env local start time"
}

RESUME_DIR=""
DRY_RUN=false
while (($#)); do
  case "$1" in
    --resume-dir)
      (($# >= 2)) || die "--resume-dir requires a path"
      RESUME_DIR="$2"
      shift 2
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LAUNCHER_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${LAUNCHER_DIR}/../.." && pwd)"
ISAAC_SETUP="${REPO_ROOT}/isaac_sim/setup_conda_env.sh"
[[ -f "${ISAAC_SETUP}" ]] || die "Isaac environment setup not found: ${ISAAC_SETUP}"
set +u
source "${ISAAC_SETUP}"
set -u
SCRIPT_DIR="${LAUNCHER_DIR}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_embodied_agent.py"
CONFIG_DIR="${SCRIPT_DIR}/config"
CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
RESULTS_ROOT="${TABERO_RESULTS_ROOT:-/data/home/sim6g/code/tabero/results}"

[[ -x "${PYTHON_BIN}" ]] || die "RLinf Python not found: ${PYTHON_BIN}"
[[ -f "${TRAIN_SCRIPT}" ]] || die "training entrypoint not found: ${TRAIN_SCRIPT}"
[[ -f "${CONFIG_PATH}" ]] || die "training config not found: ${CONFIG_PATH}"
[[ -e "${MODEL_PATH}" ]] || die "model not found: ${MODEL_PATH}"
[[ -e "${HDF5_PATH}" ]] || die "HDF5 initial states not found: ${HDF5_PATH}"

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
  GPU_COUNT="${#installed_gpus[@]}"
fi
[[ "${GPU_COUNT}" -eq 8 ]] || die "exactly 8 visible GPUs are required; found ${GPU_COUNT}"

START_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_TIME_LOCAL="$(date +%Y-%m-%dT%H:%M:%S%z)"

if [[ -n "${RESUME_DIR}" ]]; then
  [[ -d "${RESUME_DIR}" ]] || die "resume directory not found: ${RESUME_DIR}"
  RESUME_DIR="$(cd "${RESUME_DIR}" && pwd)"
  resume_basename="$(basename "${RESUME_DIR}")"
  [[ "${resume_basename}" =~ ^global_step_(0|[1-9][0-9]*)$ ]] || \
    die "resume directory basename must match global_step_<nonnegative integer>"
  [[ -d "${RESUME_DIR}/actor" ]] || die "resume checkpoint has no actor directory: ${RESUME_DIR}/actor"
  search_dir="${RESUME_DIR}"
  RUN_ENV=""
  while [[ "${search_dir}" != "/" ]]; do
    if [[ -f "${search_dir}/run.env" ]]; then
      RUN_ENV="${search_dir}/run.env"
      break
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  [[ -n "${RUN_ENV}" ]] || die "no run.env associated with resume directory: ${RESUME_DIR}"

  load_run_env "${RUN_ENV}"
  OUTPUT_DIR="${TABERO_OUTPUT_DIR}"
  [[ "${RUN_ENV}" == "${OUTPUT_DIR}/run.env" ]] || die "run.env output directory mismatch"
  expected_resume_dir="${OUTPUT_DIR}/$(basename "${OUTPUT_DIR}")/checkpoints/${resume_basename}"
  [[ "${RESUME_DIR}" == "${expected_resume_dir}" ]] || \
    die "resume directory must use nested RLinf checkpoint layout: ${expected_resume_dir}"
  WANDB_RESUME="must"
  command_stamp="$(date +%Y%m%d_%H%M%S)"
  COMMAND_FILE="$(mktemp "${OUTPUT_DIR}/resume_command_${command_stamp}_XXXXXX.txt")"
  resume_artifact_id="$(basename "${COMMAND_FILE}" .txt)"
  resume_artifact_id="${resume_artifact_id#resume_command_}"
  LOG_FILE="${OUTPUT_DIR}/resume_${resume_artifact_id}.log"
  STATUS_FILE="${OUTPUT_DIR}/resume_status_${resume_artifact_id}.env"
else
  TABERO_TASK0_DSRL_RUN_ID="${TABERO_TASK0_DSRL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_smoke}"
  [[ "${TABERO_TASK0_DSRL_RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}_smoke$ ]] || \
    die "TABERO_TASK0_DSRL_RUN_ID must match YYYYMMDD_HHMMSS_smoke"
  OUTPUT_DIR="${RESULTS_ROOT}/tabero_task0_firm_tactile_dsrl_8gpu_smoke_${TABERO_TASK0_DSRL_RUN_ID}"
  mkdir -p "${RESULTS_ROOT}"
  mkdir "${OUTPUT_DIR}" 2>/dev/null || die "output directory already exists: ${OUTPUT_DIR}"
  RUN_ENV="${OUTPUT_DIR}/run.env"
  WANDB_RUN_ID="${WANDB_RUN_ID:-$("${PYTHON_BIN}" -c 'import wandb; print(wandb.util.generate_id())')}"
  [[ -n "${WANDB_RUN_ID}" ]] || die "failed to generate WANDB_RUN_ID"
  WANDB_RESUME="allow"
  TABERO_RUN_KIND="dsrl_smoke"
  COMMAND_FILE="${OUTPUT_DIR}/command.txt"
  LOG_FILE="${OUTPUT_DIR}/train.log"
  STATUS_FILE="${OUTPUT_DIR}/run_status.env"
fi

export TABERO_TASK0_DSRL_RUN_ID WANDB_RUN_ID WANDB_RESUME
export EMBODIED_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${RESUME_DIR}" ]]; then
  metadata_tmp="$(mktemp "${OUTPUT_DIR}/.run.env.XXXXXX")"
  cleanup_metadata() { rm -f "${metadata_tmp}"; }
  trap cleanup_metadata EXIT
  {
    printf 'TABERO_TASK0_DSRL_RUN_ID=%q\n' "${TABERO_TASK0_DSRL_RUN_ID}"
    printf 'WANDB_RUN_ID=%q\n' "${WANDB_RUN_ID}"
    printf 'WANDB_RESUME=%q\n' "${WANDB_RESUME}"
    printf 'TABERO_RUN_KIND=%q\n' "${TABERO_RUN_KIND}"
    printf 'TABERO_OUTPUT_DIR=%q\n' "${OUTPUT_DIR}"
    printf 'TABERO_CONFIG_NAME=%q\n' "${CONFIG_NAME}"
    printf 'TABERO_START_TIME_UTC=%q\n' "${START_TIME_UTC}"
    printf 'TABERO_START_TIME_LOCAL=%q\n' "${START_TIME_LOCAL}"
  } >"${metadata_tmp}"
  mv "${metadata_tmp}" "${RUN_ENV}"
  trap - EXIT
fi

command=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  --config-path "${CONFIG_DIR}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${OUTPUT_DIR}"
  "runner.logger.experiment_name=$(basename "${OUTPUT_DIR}")"
)
if [[ -n "${RESUME_DIR}" ]]; then
  command+=("runner.resume_dir=${RESUME_DIR}")
fi

{
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

start_epoch="$(date +%s)"
start_gpu_sampler
trap stop_gpu_sampler EXIT
cat "${COMMAND_FILE}" >"${LOG_FILE}"
set +e
"${command[@]}" 2>&1 | tee -a "${LOG_FILE}"
train_status="${PIPESTATUS[0]}"
set -e
stop_gpu_sampler
trap - EXIT
end_epoch="$(date +%s)"
{
  printf 'TABERO_EXIT_STATUS=%q\n' "${train_status}"
  printf 'TABERO_END_TIME_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'TABERO_END_TIME_LOCAL=%q\n' "$(date +%Y-%m-%dT%H:%M:%S%z)"
  printf 'TABERO_DURATION_SECONDS=%q\n' "$((end_epoch - start_epoch))"
} >"${STATUS_FILE}"
exit "${train_status}"
