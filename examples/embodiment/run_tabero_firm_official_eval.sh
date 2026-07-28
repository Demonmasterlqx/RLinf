#!/usr/bin/env bash
set -euo pipefail

readonly FIXED_PROJECT_ROOT="/data/home/sim6g/code/tabero"
readonly FIXED_BASE_MODEL="${FIXED_PROJECT_ROOT}/models/pi0_lora_tacfield_tabero_safetensors"
readonly DEFAULT_RESULTS_ROOT="${FIXED_PROJECT_ROOT}/results"
readonly DEFAULT_GPU_LOCK_DIR="/tmp/tabero-formal-gpu-locks"
readonly MIN_FREE_DISK_KIB=52428800

usage() {
  echo "Usage: $0 dsrl <0|5> formal --dsrl-bundle ABS_PATH [--dry-run]" >&2
}

die() {
  echo "error: $*" >&2
  exit 1
}

[[ $# -ge 5 ]] || { usage; die "missing required arguments"; }
METHOD="$1"
TASK_ID="$2"
RUN_MODE="$3"
shift 3
[[ "${METHOD}" == "dsrl" ]] || die "method must be exactly dsrl"
[[ "${TASK_ID}" == "0" || "${TASK_ID}" == "5" ]] || die "task must be exactly 0 or 5"
[[ "${RUN_MODE}" == "formal" ]] || die "run mode must be exactly formal"

DSRL_BUNDLE=""
DRY_RUN=false
while (($#)); do
  case "$1" in
    --dsrl-bundle)
      (($# >= 2)) || die "--dsrl-bundle requires an absolute path"
      [[ -z "${DSRL_BUNDLE}" ]] || die "--dsrl-bundle may be specified only once"
      DSRL_BUNDLE="$2"
      shift 2
      ;;
    --dry-run)
      [[ "${DRY_RUN}" == false ]] || die "--dry-run may be specified only once"
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "${DSRL_BUNDLE}" ]] || die "--dsrl-bundle is required"
[[ "${DSRL_BUNDLE}" == /* ]] || die "--dsrl-bundle must be an absolute path"
[[ -d "${DSRL_BUNDLE}" ]] || die "DSRL bundle must be a directory: ${DSRL_BUNDLE}"
DSRL_BUNDLE="$(cd "${DSRL_BUNDLE}" && pwd -P)"

for variable_name in $(compgen -e); do
  [[ "${variable_name}" == TABERO_TEST_* ]] || continue
  if [[ "${DRY_RUN}" != true ]]; then
    die "${variable_name} is only allowed with --dry-run"
  fi
  case "${variable_name}" in
    TABERO_TEST_PROJECT_ROOT|TABERO_TEST_BASE_MODEL|TABERO_TEST_RESULTS_ROOT|TABERO_TEST_GPU_LOCK_DIR|TABERO_TEST_TIMESTAMP|TABERO_TEST_PORT|TABERO_TEST_ALLOW_DIRTY|TABERO_TEST_FREE_DISK_KIB) ;;
    *) die "unknown dry-run test override: ${variable_name}" ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RLINF_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_ROOT="${FIXED_PROJECT_ROOT}"
BASE_MODEL="${FIXED_BASE_MODEL}"
RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}"
GPU_LOCK_DIR="${DEFAULT_GPU_LOCK_DIR}"
if [[ "${DRY_RUN}" == true ]]; then
  PROJECT_ROOT="${TABERO_TEST_PROJECT_ROOT:-${PROJECT_ROOT}}"
  BASE_MODEL="${TABERO_TEST_BASE_MODEL:-${BASE_MODEL}}"
  RESULTS_ROOT="${TABERO_TEST_RESULTS_ROOT:-${RESULTS_ROOT}}"
  GPU_LOCK_DIR="${TABERO_TEST_GPU_LOCK_DIR:-${GPU_LOCK_DIR}}"
fi
T2_ROOT="${PROJECT_ROOT}/T2-VLA"
TABERO_ROOT="${PROJECT_ROOT}/Tabero"
T2_PYTHON="${T2_ROOT}/.venv/bin/python"
SERVER_SCRIPT="${T2_ROOT}/scripts/serve_policy.py"
CLIENT_SCRIPT="${TABERO_ROOT}/scripts/tools/run_task_evaluations.py"
HDF5_FOLDER="${TABERO_ROOT}/benchmarks/datasets/libero/assembled_hdf5"
RLINF_PYTHON="${RLINF_ROOT}/.venv/bin/python"
HELPER="${SCRIPT_DIR}/tabero_dsrl_official_eval.py"

[[ -x "${RLINF_PYTHON}" ]] || die "RLinf Python is missing: ${RLINF_PYTHON}"
[[ -x "${T2_PYTHON}" ]] || die "T2 Python is missing: ${T2_PYTHON}"
[[ -f "${SERVER_SCRIPT}" ]] || die "T2 server script is missing: ${SERVER_SCRIPT}"
[[ -f "${CLIENT_SCRIPT}" ]] || die "Tabero client script is missing: ${CLIENT_SCRIPT}"
[[ -d "${HDF5_FOLDER}" ]] || die "assembled HDF5 directory is missing: ${HDF5_FOLDER}"
[[ -f "${HELPER}" ]] || die "official-eval helper is missing: ${HELPER}"
[[ -f "${BASE_MODEL}/model.safetensors" ]] || die "fixed base weights are missing: ${BASE_MODEL}/model.safetensors"
command -v conda >/dev/null 2>&1 || die "conda is required"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"

# The client pins physical GPU 1 by index; inherited remapping would violate it.
unset CUDA_VISIBLE_DEVICES

declare -A INSTALLED_GPUS=()
while IFS= read -r gpu_id; do
  gpu_id="${gpu_id// /}"
  gpu_id="${gpu_id//$'\r'/}"
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "nvidia-smi returned invalid GPU index: ${gpu_id}"
  [[ -z "${INSTALLED_GPUS[${gpu_id}]:-}" ]] || die "nvidia-smi returned duplicate GPU index: ${gpu_id}"
  INSTALLED_GPUS["${gpu_id}"]=1
done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
[[ -n "${INSTALLED_GPUS[0]:-}" ]] || die "required physical GPU 0 is not installed"
[[ -n "${INSTALLED_GPUS[1]:-}" ]] || die "required physical GPU 1 is not installed"

mkdir -p "${GPU_LOCK_DIR}"
GPU_LOCK_FDS=()
for gpu_id in 0 1; do
  exec {gpu_lock_fd}>"${GPU_LOCK_DIR}/gpu_${gpu_id}.lock"
  flock -n "${gpu_lock_fd}" || die "GPU ${gpu_id} lease is already held by another formal launcher"
  GPU_LOCK_FDS+=("${gpu_lock_fd}")
done
for gpu_id in 0 1; do
  compute_pids="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits)" || \
    die "failed to query compute processes for GPU ${gpu_id}"
  while IFS= read -r compute_pid; do
    compute_pid="${compute_pid// /}"
    compute_pid="${compute_pid//$'\r'/}"
    [[ -z "${compute_pid}" ]] && continue
    [[ "${compute_pid}" =~ ^[0-9]+$ ]] || die "nvidia-smi returned invalid compute PID for GPU ${gpu_id}: ${compute_pid}"
    die "GPU ${gpu_id} is busy with compute PID ${compute_pid}"
  done <<<"${compute_pids}"
done

mkdir -p "${RESULTS_ROOT}"
FREE_DISK_KIB="$(df -Pk "${RESULTS_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ "${DRY_RUN}" == true && -n "${TABERO_TEST_FREE_DISK_KIB:-}" ]]; then
  FREE_DISK_KIB="${TABERO_TEST_FREE_DISK_KIB}"
fi
[[ "${FREE_DISK_KIB}" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
((FREE_DISK_KIB >= MIN_FREE_DISK_KIB)) || \
  die "disk gate failed: ${FREE_DISK_KIB} KiB free, ${MIN_FREE_DISK_KIB} KiB required"

PREFLIGHT_DIR="$(mktemp -d)"
cleanup_preflight() {
  rm -f -- "${PREFLIGHT_DIR}/bundle_and_repos.env"
  rmdir -- "${PREFLIGHT_DIR}" 2>/dev/null || true
}
trap cleanup_preflight EXIT
preflight_command=(
  "${RLINF_PYTHON}" "${HELPER}" preflight
  --bundle "${DSRL_BUNDLE}"
  --task-id "${TASK_ID}"
  --base-model "${BASE_MODEL}"
  --rlinf-repo "${RLINF_ROOT}"
  --t2-repo "${T2_ROOT}"
  --tabero-repo "${TABERO_ROOT}"
  --metadata-out "${PREFLIGHT_DIR}/bundle_and_repos.env"
)
if [[ "${DRY_RUN}" == true && "${TABERO_TEST_ALLOW_DIRTY:-}" == "1" ]]; then
  preflight_command+=(--allow-dirty)
fi
PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${preflight_command[@]}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)_formal"
if [[ "${DRY_RUN}" == true && -n "${TABERO_TEST_TIMESTAMP:-}" ]]; then
  RUN_STAMP="${TABERO_TEST_TIMESTAMP}"
fi
[[ "${RUN_STAMP}" =~ ^[0-9]{8}_[0-9]{6}_formal$ ]] || die "run timestamp must match YYYYMMDD_HHMMSS_formal"
OUTPUT_DIR="${RESULTS_ROOT}/tabero_task${TASK_ID}_firm_dsrl_official_eval_formal_${RUN_STAMP}"
mkdir "${OUTPUT_DIR}" 2>/dev/null || die "output directory already exists: ${OUTPUT_DIR}"
START_EPOCH="$(date +%s)"
START_TIME_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STATUS_FILE="${OUTPUT_DIR}/run_status.env"
SERVER_PID=""
CLIENT_PID=""
GPU_SAMPLER_PID=""
STATUS_FINALIZED=false

write_status() {
  local run_status="$1" exit_status="$2" end_epoch temporary
  end_epoch="$(date +%s)"
  temporary="$(mktemp "${OUTPUT_DIR}/.run_status.env.XXXXXX")"
  {
    printf 'TABERO_RUN_STATUS=%s\n' "${run_status}"
    printf 'TABERO_EXIT_STATUS=%s\n' "${exit_status}"
    printf 'TABERO_END_TIME_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'TABERO_DURATION_SECONDS=%s\n' "$((end_epoch - START_EPOCH))"
  } >"${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

terminate_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  terminate_group "${CLIENT_PID}"
  terminate_group "${SERVER_PID}"
  terminate_group "${GPU_SAMPLER_PID}"
  cleanup_preflight
  if [[ "${STATUS_FINALIZED}" != true ]]; then
    write_status failed "${exit_status}"
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/output"
RUN_ID_SUFFIX="${RUN_STAMP/_/-}"
RUN_ID_SUFFIX="${RUN_ID_SUFFIX/_/-}"
WANDB_RUN_ID="tabero-official-dsrl-task${TASK_ID}-${RUN_ID_SUFFIX//_/-}"

PORT=""
if [[ "${DRY_RUN}" == true && -n "${TABERO_TEST_PORT:-}" ]]; then
  PORT="${TABERO_TEST_PORT}"
else
  PORT="$("${RLINF_PYTHON}" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi
[[ "${PORT}" =~ ^[0-9]+$ ]] && ((PORT >= 1024 && PORT <= 65535)) || die "invalid local port: ${PORT}"

RUN_ENV="${OUTPUT_DIR}/run.env"
START_METADATA="${OUTPUT_DIR}/start_metadata.env"
{
  printf 'TABERO_OFFICIAL_EVAL_RUN_ID=%s\n' "${RUN_STAMP}"
  printf 'WANDB_RUN_ID=%s\n' "${WANDB_RUN_ID}"
  printf 'TABERO_METHOD=dsrl\n'
  printf 'TABERO_TASK_ID=%s\n' "${TASK_ID}"
  printf 'TABERO_RUN_MODE=formal\n'
  printf 'TABERO_EXPECTED_EPISODES=50\n'
  printf 'TABERO_DSRL_BUNDLE=%s\n' "${DSRL_BUNDLE}"
  printf 'TABERO_OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'TABERO_START_TIME_UTC=%s\n' "${START_TIME_UTC}"
} >"${RUN_ENV}"
{
  cat "${PREFLIGHT_DIR}/bundle_and_repos.env"
  printf 'TABERO_GPU_IDS=0,1\n'
  printf 'TABERO_FREE_DISK_KIB=%s\n' "${FREE_DISK_KIB}"
  printf 'TABERO_EPHEMERAL_PORT=%s\n' "${PORT}"
  printf 'TABERO_START_TIME_UTC=%s\n' "${START_TIME_UTC}"
} >"${START_METADATA}"
cleanup_preflight

server_command=(
  env
  CUDA_VISIBLE_DEVICES=0
  JAX_PLATFORMS=cuda
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  PYTHONUNBUFFERED=1
  "${T2_PYTHON}" "${SERVER_SCRIPT}"
  --port "${PORT}"
  --dsrl-bundle "${DSRL_BUNDLE}"
  policy:checkpoint
  --policy.config=pi0_lora_tacfield_tabero
  "--policy.dir=${BASE_MODEL}"
)
client_command=(
  conda run --no-capture-output -n tabero python -u "${CLIENT_SCRIPT}"
  --policy-model openpi
  --control-mode tactile
  --server-host 127.0.0.1
  --server-port "${PORT}"
  --num-total-experiments 50
  --num-success-steps 8
  --max-inference-steps 30
  --replan-steps 10
  --task-suites libero_object
  --task-ids "${TASK_ID}"
  --hdf5-folder "${HDF5_FOLDER}"
  --require-hdf5
  --output-dir "${OUTPUT_DIR}/raw"
  --output-format both
  --seed 11
  --prompt-seed 0
  --prompt-adverbs firmly tightly
  --send-dsrl-raw-image
  --sim-device cuda:1
  --sim-kit-args=--/renderer/activeGpu=1
  --headless
)
{
  printf 'Command:'
  printf ' %q' "${server_command[@]}"
  printf '\n'
} >"${OUTPUT_DIR}/server_command.txt"
{
  printf 'Command:'
  printf ' %q' "${client_command[@]}"
  printf '\n'
} >"${OUTPUT_DIR}/client_command.txt"
: >"${OUTPUT_DIR}/server.log"
: >"${OUTPUT_DIR}/client.log"
printf '%s\n' 'timestamp_utc,index,name,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_w,pstate' >"${OUTPUT_DIR}/gpu_samples.csv"
printf '%s\n' 'timestamp_utc,gpu_uuid,pid,process_name,used_gpu_memory_mib' >"${OUTPUT_DIR}/gpu_process_samples.csv"

if [[ "${DRY_RUN}" == true ]]; then
  : >"${OUTPUT_DIR}/server.pid"
  : >"${OUTPUT_DIR}/client.pid"
  : >"${OUTPUT_DIR}/gpu_sampler.pid"
  write_status dry_run 0
  STATUS_FINALIZED=true
  printf 'Output directory: %s\n' "${OUTPUT_DIR}"
  exit 0
fi

(
  while true; do
    sample_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while IFS= read -r row; do
      [[ -n "${row}" ]] && printf '%s,%s\n' "${sample_time}" "${row}"
    done < <(
      nvidia-smi --id=0,1 --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,pstate --format=csv,noheader,nounits
    ) >>"${OUTPUT_DIR}/gpu_samples.csv"
    while IFS= read -r row; do
      [[ -n "${row}" ]] && printf '%s,%s\n' "${sample_time}" "${row}"
    done < <(
      nvidia-smi --id=0,1 --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true
    ) >>"${OUTPUT_DIR}/gpu_process_samples.csv"
    sleep 5
  done
) &
GPU_SAMPLER_PID=$!
printf '%s\n' "${GPU_SAMPLER_PID}" >"${OUTPUT_DIR}/gpu_sampler.pid"

setsid "${server_command[@]}" >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${OUTPUT_DIR}/server.pid"
ready=false
for _ in $(seq 1 300); do
  kill -0 "${SERVER_PID}" 2>/dev/null || die "T2 server exited before becoming ready"
  if "${RLINF_PYTHON}" - "${PORT}" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
  then
    ready=true
    break
  fi
  sleep 2
done
[[ "${ready}" == true ]] || die "T2 server did not listen within 600 seconds"

setsid "${client_command[@]}" >"${OUTPUT_DIR}/client.log" 2>&1 &
CLIENT_PID=$!
printf '%s\n' "${CLIENT_PID}" >"${OUTPUT_DIR}/client.pid"
set +e
wait "${CLIENT_PID}"
CLIENT_STATUS=$?
set -e
CLIENT_PID=""
[[ "${CLIENT_STATUS}" -eq 0 ]] || die "Tabero client failed with exit status ${CLIENT_STATUS}"

PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${RLINF_PYTHON}" "${HELPER}" finalize \
  --raw-dir "${OUTPUT_DIR}/raw" \
  --output-dir "${OUTPUT_DIR}" \
  --bundle "${DSRL_BUNDLE}" \
  --task-id "${TASK_ID}" \
  --run-id "${WANDB_RUN_ID}"
write_status completed 0
STATUS_FINALIZED=true
exit 0
