#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task1_firm_8gpu_50step"
readonly MODEL_PATH="/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_safetensors"
readonly HDF5_PATH="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5/libero_object_task1_pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5"

usage() {
  echo "Usage: $0 <smoke|formal> [--resume-dir PATH] [--dry-run]" >&2
}

die() {
  echo "error: $*" >&2
  exit 1
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
      TABERO_TASK1_RUN_ID) TABERO_TASK1_RUN_ID="${value}" ;;
      WANDB_RUN_ID) WANDB_RUN_ID="${value}" ;;
      WANDB_RESUME) WANDB_RESUME="${value}" ;;
      TABERO_RUN_MODE) TABERO_RUN_MODE="${value}" ;;
      TABERO_OUTPUT_DIR) TABERO_OUTPUT_DIR="${value}" ;;
      TABERO_CONFIG_NAME) TABERO_CONFIG_NAME="${value}" ;;
      TABERO_START_TIME_UTC) TABERO_START_TIME_UTC="${value}" ;;
      TABERO_START_TIME_LOCAL) TABERO_START_TIME_LOCAL="${value}" ;;
      *) die "invalid run.env key: ${key}" ;;
    esac
    ((field_count += 1))
  done <"${env_file}"

  [[ "${field_count}" -eq 8 ]] || die "invalid run.env: expected 8 fields"
  [[ "${TABERO_TASK1_RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}_(smoke|formal)$ ]] || die "invalid run.env run ID"
  [[ "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid run.env W&B ID"
  [[ "${WANDB_RESUME}" == "allow" || "${WANDB_RESUME}" == "must" ]] || die "invalid run.env W&B resume mode"
  [[ "${TABERO_RUN_MODE}" == "smoke" || "${TABERO_RUN_MODE}" == "formal" ]] || die "invalid run.env mode"
  [[ "${TABERO_OUTPUT_DIR}" == /* ]] || die "invalid run.env output directory"
  [[ "${TABERO_CONFIG_NAME}" == "${CONFIG_NAME}" ]] || die "invalid run.env config name"
  [[ "${TABERO_START_TIME_UTC}" =~ ^[0-9T:Z+-]+$ ]] || die "invalid run.env UTC start time"
  [[ "${TABERO_START_TIME_LOCAL}" =~ ^[0-9T:+-]+$ ]] || die "invalid run.env local start time"
}

MODE="${1:-}"
[[ "${MODE}" == "smoke" || "${MODE}" == "formal" ]] || {
  usage
  exit 2
}
shift

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
# NVIDIA's setup script probes optional variables without nounset guards.
set +u
source "${ISAAC_SETUP}"
set -u
SCRIPT_DIR="${LAUNCHER_DIR}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_embodied_agent.py"
CONFIG_DIR="${SCRIPT_DIR}/config"
RESULTS_ROOT="${TABERO_RESULTS_ROOT:-/data/home/sim6g/code/tabero/results}"

[[ -x "${PYTHON_BIN}" ]] || die "RLinf Python not found: ${PYTHON_BIN}"
[[ -f "${TRAIN_SCRIPT}" ]] || die "training entrypoint not found: ${TRAIN_SCRIPT}"
[[ -e "${MODEL_PATH}" ]] || die "model not found: ${MODEL_PATH}"
[[ -e "${HDF5_PATH}" ]] || die "HDF5 initial states not found: ${HDF5_PATH}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT="${#visible_gpus[@]}"
  declare -A seen_gpus=()
  for gpu_id in "${visible_gpus[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must contain physical numeric GPU IDs"
    [[ -z "${seen_gpus[${gpu_id}]:-}" ]] || die "CUDA_VISIBLE_DEVICES must contain 8 unique physical GPU IDs"
    seen_gpus["${gpu_id}"]=1
  done
else
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
  GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
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
  [[ "${TABERO_RUN_MODE}" == "${MODE}" ]] || die "resume mode does not match run.env"
  OUTPUT_DIR="${TABERO_OUTPUT_DIR}"
  [[ "${RUN_ENV}" == "${OUTPUT_DIR}/run.env" ]] || die "run.env output directory mismatch"
  expected_resume_dir="${OUTPUT_DIR}/$(basename "${OUTPUT_DIR}")/checkpoints/${resume_basename}"
  [[ "${RESUME_DIR}" == "${expected_resume_dir}" ]] || \
    die "resume directory must use nested RLinf checkpoint layout: ${expected_resume_dir}"
  [[ -n "${WANDB_RUN_ID}" ]] || die "run.env has no WANDB_RUN_ID"
  WANDB_RESUME="must"
  LOG_FILE="$(mktemp "${OUTPUT_DIR}/resume_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
else
  TABERO_TASK1_RUN_ID="${TABERO_TASK1_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${MODE}}"
  [[ "${TABERO_TASK1_RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}_${MODE}$ ]] || \
    die "TABERO_TASK1_RUN_ID must match YYYYMMDD_HHMMSS_${MODE}"

  if [[ "${MODE}" == "smoke" ]]; then
    output_prefix="tabero_task1_firm_action_expert_lora_8gpu_capacity_smoke"
  else
    output_prefix="tabero_task1_firm_action_expert_lora_8gpu_50step"
  fi
  OUTPUT_DIR="${RESULTS_ROOT}/${output_prefix}_${TABERO_TASK1_RUN_ID}"
  mkdir -p "${RESULTS_ROOT}"
  mkdir "${OUTPUT_DIR}" 2>/dev/null || die "output directory already exists: ${OUTPUT_DIR}"
  RUN_ENV="${OUTPUT_DIR}/run.env"
  WANDB_RUN_ID="${WANDB_RUN_ID:-$("${PYTHON_BIN}" -c 'import wandb; print(wandb.util.generate_id())')}"
  [[ -n "${WANDB_RUN_ID}" ]] || die "failed to generate WANDB_RUN_ID"
  WANDB_RESUME="allow"
  LOG_FILE="${OUTPUT_DIR}/train.log"
fi

export TABERO_TASK1_RUN_ID WANDB_RUN_ID WANDB_RESUME
export EMBODIED_PATH="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

metadata_tmp="$(mktemp "${OUTPUT_DIR}/.run.env.XXXXXX")"
cleanup() { rm -f "${metadata_tmp}"; }
trap cleanup EXIT
{
  printf 'TABERO_TASK1_RUN_ID=%q\n' "${TABERO_TASK1_RUN_ID}"
  printf 'WANDB_RUN_ID=%q\n' "${WANDB_RUN_ID}"
  printf 'WANDB_RESUME=%q\n' "${WANDB_RESUME}"
  printf 'TABERO_RUN_MODE=%q\n' "${MODE}"
  printf 'TABERO_OUTPUT_DIR=%q\n' "${OUTPUT_DIR}"
  printf 'TABERO_CONFIG_NAME=%q\n' "${CONFIG_NAME}"
  printf 'TABERO_START_TIME_UTC=%q\n' "${START_TIME_UTC}"
  printf 'TABERO_START_TIME_LOCAL=%q\n' "${START_TIME_LOCAL}"
} >"${metadata_tmp}"
mv "${metadata_tmp}" "${RUN_ENV}"
trap - EXIT

command=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  --config-path "${CONFIG_DIR}"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${OUTPUT_DIR}"
  "runner.logger.experiment_name=$(basename "${OUTPUT_DIR}")"
)
if [[ "${MODE}" == "smoke" ]]; then
  command+=("runner.max_epochs=1" "runner.save_interval=1")
fi
if [[ -n "${RESUME_DIR}" ]]; then
  command+=("runner.resume_dir=${RESUME_DIR}")
fi

printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Log file: %s\n' "${LOG_FILE}"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN}" == true ]]; then
  exit 0
fi

printf 'Command:' >"${LOG_FILE}"
printf ' %q' "${command[@]}" >>"${LOG_FILE}"
printf '\n' >>"${LOG_FILE}"
"${command[@]}" 2>&1 | tee -a "${LOG_FILE}"
