#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_mass_friction_2gpu_100step"
readonly MODEL_PATH="/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_all_firm_safetensors"
readonly PROFILE_DIR="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k"
readonly HDF5_PATH="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5/libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
readonly PPO_BOUNDARY_SEMANTICS="terminal_observation_first_done_prefix_logprob_hdf5_reset_v1"

usage() {
  echo "Usage: $0 <smoke|formal> [--resume-dir PATH] [--dry-run]" >&2
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
    [[ "${value}" =~ ^[A-Za-z0-9_./:+,-]+$ ]] || die "invalid run.env value for ${key}"
    [[ -z "${seen[${key}]:-}" ]] || die "invalid run.env: duplicate ${key}"
    seen["${key}"]=1
    case "${key}" in
      TABERO_TASK0_NO_ADVERB_RUN_ID) TABERO_TASK0_NO_ADVERB_RUN_ID="${value}" ;;
      WANDB_RUN_ID) WANDB_RUN_ID="${value}" ;;
      WANDB_RESUME) WANDB_RESUME="${value}" ;;
      TABERO_RUN_MODE) TABERO_RUN_MODE="${value}" ;;
      TABERO_OUTPUT_DIR) TABERO_OUTPUT_DIR="${value}" ;;
      TABERO_CONFIG_NAME) TABERO_CONFIG_NAME="${value}" ;;
      TABERO_VISIBLE_GPUS) TABERO_VISIBLE_GPUS="${value}" ;;
      TABERO_START_TIME_UTC) TABERO_START_TIME_UTC="${value}" ;;
      TABERO_START_TIME_LOCAL) TABERO_START_TIME_LOCAL="${value}" ;;
      TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS) TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS="${value}" ;;
      *) die "invalid run.env key: ${key}" ;;
    esac
    ((field_count += 1))
  done <"${env_file}"

  [[ "${field_count}" -eq 10 ]] || die "invalid run.env: expected 10 fields"
  [[ "${TABERO_TASK0_NO_ADVERB_RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}_(smoke|formal)$ ]] || die "invalid run.env run ID"
  [[ "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid run.env W&B ID"
  [[ "${WANDB_RESUME}" == "allow" || "${WANDB_RESUME}" == "must" ]] || die "invalid run.env W&B resume mode"
  [[ "${TABERO_RUN_MODE}" == "smoke" || "${TABERO_RUN_MODE}" == "formal" ]] || die "invalid run.env mode"
  [[ "${TABERO_OUTPUT_DIR}" == /* ]] || die "invalid run.env output directory"
  [[ "${TABERO_CONFIG_NAME}" == "${CONFIG_NAME}" ]] || die "invalid run.env config name"
  [[ "${TABERO_VISIBLE_GPUS}" =~ ^[0-9]+,[0-9]+$ ]] || die "invalid run.env visible GPU mapping"
  [[ "${TABERO_START_TIME_UTC}" =~ ^[0-9T:Z+-]+$ ]] || die "invalid run.env UTC start time"
  [[ "${TABERO_START_TIME_LOCAL}" =~ ^[0-9T:+-]+$ ]] || die "invalid run.env local start time"
  [[ "${TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS}" == "${PPO_BOUNDARY_SEMANTICS}" ]] || die "invalid run.env PPO boundary semantics"
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

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAUNCHER_DIR}/../.." && pwd)"
ISAAC_SETUP="${REPO_ROOT}/isaac_sim/setup_conda_env.sh"
[[ -f "${ISAAC_SETUP}" ]] || die "Isaac environment setup not found: ${ISAAC_SETUP}"
# NVIDIA's setup script probes optional variables without nounset guards.
set +u
source "${ISAAC_SETUP}"
set -u
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${LAUNCHER_DIR}/train_embodied_agent.py"
CONFIG_DIR="${LAUNCHER_DIR}/config"
RESULTS_ROOT="${TABERO_RESULTS_ROOT:-/data/home/sim6g/code/tabero/results}"

[[ -x "${PYTHON_BIN}" ]] || die "RLinf Python not found: ${PYTHON_BIN}"
[[ -f "${TRAIN_SCRIPT}" ]] || die "training entrypoint not found: ${TRAIN_SCRIPT}"
[[ -f "${MODEL_PATH}/model.safetensors" ]] || die "model not found: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/config.json" ]] || die "model config not found: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/export_meta.json" ]] || die "model metadata not found: ${MODEL_PATH}"
[[ -f "${PROFILE_DIR}/libero_object.json" ]] || die "profile not found: ${PROFILE_DIR}"
[[ -f "${HDF5_PATH}" ]] || die "HDF5 initial states not found: ${HDF5_PATH}"

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a visible_gpu_ids <<<"${VISIBLE_GPUS}"
[[ "${#visible_gpu_ids[@]}" -eq 2 ]] || die "exactly 2 visible GPUs are required"
declare -A seen_gpus=()
for gpu_id in "${visible_gpu_ids[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must contain physical numeric GPU IDs"
  [[ -z "${seen_gpus[${gpu_id}]:-}" ]] || die "CUDA_VISIBLE_DEVICES must contain 2 unique physical GPU IDs"
  seen_gpus["${gpu_id}"]=1
done
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

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

  requested_visible_gpus="${VISIBLE_GPUS}"
  load_run_env "${RUN_ENV}"
  [[ "${TABERO_RUN_MODE}" == "${MODE}" ]] || die "resume mode does not match run.env"
  [[ "${TABERO_VISIBLE_GPUS}" == "${requested_visible_gpus}" ]] || die "resume GPU mapping does not match run.env"
  [[ "${TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS}" == "${PPO_BOUNDARY_SEMANTICS}" ]] || \
    die "resume PPO boundary semantics does not match run.env"
  OUTPUT_DIR="${TABERO_OUTPUT_DIR}"
  [[ "${RUN_ENV}" == "${OUTPUT_DIR}/run.env" ]] || die "run.env output directory mismatch"
  expected_resume_dir="${OUTPUT_DIR}/$(basename "${OUTPUT_DIR}")/checkpoints/${resume_basename}"
  [[ "${RESUME_DIR}" == "${expected_resume_dir}" ]] || \
    die "resume directory must use nested RLinf checkpoint layout: ${expected_resume_dir}"
  CHECKPOINT_SIDECAR="${RESUME_DIR}/actor/model_state_dict/trainable_weights.pt"
  [[ -f "${CHECKPOINT_SIDECAR}" ]] || \
    die "boundary-safe resume requires checkpoint sidecar: ${CHECKPOINT_SIDECAR}"
  "${PYTHON_BIN}" - "${CHECKPOINT_SIDECAR}" "${PPO_BOUNDARY_SEMANTICS}" <<'PY' || \
    die "legacy or mismatched checkpoint cannot resume boundary-safe PPO"
import sys

import torch

sidecar_path, expected = sys.argv[1:]
payload = torch.load(sidecar_path, map_location="cpu", weights_only=True)
metadata = payload.get("metadata") if isinstance(payload, dict) else None
if not isinstance(metadata, dict):
    raise SystemExit("checkpoint sidecar has no metadata mapping")
actual = metadata.get("tabero_ppo_transition_boundary_semantics")
if actual != expected:
    raise SystemExit(
        f"checkpoint boundary semantics mismatch: expected {expected!r}, got {actual!r}"
    )
PY
  WANDB_RESUME="must"
  LOG_FILE="$(mktemp "${OUTPUT_DIR}/resume_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
else
  TABERO_TASK0_NO_ADVERB_RUN_ID="${TABERO_TASK0_NO_ADVERB_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${MODE}}"
  [[ "${TABERO_TASK0_NO_ADVERB_RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}_${MODE}$ ]] || \
    die "TABERO_TASK0_NO_ADVERB_RUN_ID must match YYYYMMDD_HHMMSS_${MODE}"

  if [[ "${MODE}" == "smoke" ]]; then
    output_prefix="tabero_task0_no_adverb_mass_friction_action_expert_lora_2gpu_capacity_smoke"
  else
    output_prefix="tabero_task0_no_adverb_mass_friction_action_expert_lora_2gpu_100step"
  fi
  OUTPUT_DIR="${RESULTS_ROOT}/${output_prefix}_${TABERO_TASK0_NO_ADVERB_RUN_ID}"
  mkdir -p "${RESULTS_ROOT}"
  mkdir "${OUTPUT_DIR}" 2>/dev/null || die "output directory already exists: ${OUTPUT_DIR}"
  RUN_ENV="${OUTPUT_DIR}/run.env"
  WANDB_RUN_ID="${WANDB_RUN_ID:-$("${PYTHON_BIN}" -c 'import wandb; print(wandb.util.generate_id())')}"
  [[ -n "${WANDB_RUN_ID}" ]] || die "failed to generate WANDB_RUN_ID"
  WANDB_RESUME="allow"
  LOG_FILE="${OUTPUT_DIR}/train.log"
fi

TABERO_VISIBLE_GPUS="${VISIBLE_GPUS}"
TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS="${PPO_BOUNDARY_SEMANTICS}"
export TABERO_TASK0_NO_ADVERB_RUN_ID WANDB_RUN_ID WANDB_RESUME
export EMBODIED_PATH="${LAUNCHER_DIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

metadata_tmp="$(mktemp "${OUTPUT_DIR}/.run.env.XXXXXX")"
cleanup() { rm -f "${metadata_tmp}"; }
trap cleanup EXIT
{
  printf 'TABERO_TASK0_NO_ADVERB_RUN_ID=%q\n' "${TABERO_TASK0_NO_ADVERB_RUN_ID}"
  printf 'WANDB_RUN_ID=%q\n' "${WANDB_RUN_ID}"
  printf 'WANDB_RESUME=%q\n' "${WANDB_RESUME}"
  printf 'TABERO_RUN_MODE=%q\n' "${MODE}"
  printf 'TABERO_OUTPUT_DIR=%q\n' "${OUTPUT_DIR}"
  printf 'TABERO_CONFIG_NAME=%q\n' "${CONFIG_NAME}"
  printf 'TABERO_VISIBLE_GPUS=%s\n' "${TABERO_VISIBLE_GPUS}"
  printf 'TABERO_START_TIME_UTC=%q\n' "${START_TIME_UTC}"
  printf 'TABERO_START_TIME_LOCAL=%q\n' "${START_TIME_LOCAL}"
  printf 'TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS=%s\n' "${TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS}"
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
  command+=(
    "runner.max_epochs=1"
    "runner.save_interval=1"
    "actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=1"
  )
fi
if [[ -n "${RESUME_DIR}" ]]; then
  command+=("runner.resume_dir=${RESUME_DIR}")
fi

printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Log file: %s\n' "${LOG_FILE}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN}" == true ]]; then
  exit 0
fi

start_gpu_sampler
trap stop_gpu_sampler EXIT
printf 'Command:' >"${LOG_FILE}"
printf ' %q' "${command[@]}" >>"${LOG_FILE}"
printf '\n' >>"${LOG_FILE}"
"${command[@]}" 2>&1 | tee -a "${LOG_FILE}"
