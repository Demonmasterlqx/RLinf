#!/usr/bin/env bash
set -euo pipefail

readonly FIXED_PROJECT_ROOT="/data/home/sim6g/code/tabero"
readonly FIXED_BASE_MODEL="${FIXED_PROJECT_ROOT}/models/pi0_lora_tacfield_tabero_safetensors"
readonly DEFAULT_RESULTS_ROOT="${FIXED_PROJECT_ROOT}/results"
readonly DEFAULT_GPU_LOCK_DIR="/run/lock"
readonly MIN_FREE_DISK_KIB=52428800

usage() {
  echo "Usage: $0 dsrl <0|5> formal --dsrl-bundle ABS_PATH --training-profile PROFILE [--dry-run]" >&2
  echo "Profiles: formal_8gpu_50step | task5_4gpu_40step_small | task0_8gpu_60step | task0_8gpu_60step_selected_step{10,20,30,40,50}" >&2
}

die() {
  echo "error: $*" >&2
  exit 1
}

[[ $# -ge 7 ]] || { usage; die "missing required arguments"; }
METHOD="$1"
TASK_ID="$2"
RUN_MODE="$3"
shift 3
[[ "${METHOD}" == "dsrl" ]] || die "method must be exactly dsrl"
[[ "${TASK_ID}" == "0" || "${TASK_ID}" == "5" ]] || die "task must be exactly 0 or 5"
[[ "${RUN_MODE}" == "formal" ]] || die "run mode must be exactly formal"

DSRL_BUNDLE=""
TRAINING_PROFILE=""
DRY_RUN=false
while (($#)); do
  case "$1" in
    --dsrl-bundle)
      (($# >= 2)) || die "--dsrl-bundle requires an absolute path"
      [[ -z "${DSRL_BUNDLE}" ]] || die "--dsrl-bundle may be specified only once"
      DSRL_BUNDLE="$2"
      shift 2
      ;;
    --training-profile)
      (($# >= 2)) || die "--training-profile requires a profile name"
      [[ -z "${TRAINING_PROFILE}" ]] || die "--training-profile may be specified only once"
      TRAINING_PROFILE="$2"
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
[[ -n "${TRAINING_PROFILE}" ]] || die "--training-profile is required"
case "${TRAINING_PROFILE}" in
  formal_8gpu_50step)
    ;;
  task5_4gpu_40step_small)
    [[ "${TASK_ID}" == "5" ]] || \
      die "task5_4gpu_40step_small training profile supports only Task 5"
    ;;
  task0_8gpu_60step|\
  task0_8gpu_60step_selected_step10|\
  task0_8gpu_60step_selected_step20|\
  task0_8gpu_60step_selected_step30|\
  task0_8gpu_60step_selected_step40|\
  task0_8gpu_60step_selected_step50)
    [[ "${TASK_ID}" == "0" ]] || \
      die "${TRAINING_PROFILE} training profile supports only Task 0"
    ;;
  *)
    die "unsupported --training-profile: ${TRAINING_PROFILE}"
    ;;
esac
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
export PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${RLINF_PYTHON}" ]] || die "RLinf Python is missing: ${RLINF_PYTHON}"
[[ -x "${T2_PYTHON}" ]] || die "T2 Python is missing: ${T2_PYTHON}"
[[ -f "${SERVER_SCRIPT}" ]] || die "T2 server script is missing: ${SERVER_SCRIPT}"
[[ -f "${CLIENT_SCRIPT}" ]] || die "Tabero client script is missing: ${CLIENT_SCRIPT}"
[[ -d "${HDF5_FOLDER}" ]] || die "assembled HDF5 directory is missing: ${HDF5_FOLDER}"
[[ -f "${HELPER}" ]] || die "official-eval helper is missing: ${HELPER}"
[[ -f "${BASE_MODEL}/model.safetensors" ]] || die "fixed base weights are missing: ${BASE_MODEL}/model.safetensors"
command -v conda >/dev/null 2>&1 || die "conda is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"

# The client pins physical GPU 1 by index; inherited remapping would violate it.
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONDONTWRITEBYTECODE=1

LAUNCHER_START_TIME="$(awk '{print $22}' "/proc/$$/stat")"
[[ "${LAUNCHER_START_TIME}" =~ ^[0-9]+$ ]] || die "could not capture launcher process identity"

run_supervised() {
  local -a supervisor_command
  supervisor_command=(
    setsid "${RLINF_PYTHON}" "${HELPER}" supervise
    --parent-pid "$$"
    --parent-start-time "${LAUNCHER_START_TIME}"
    -- "$@"
  )
  # A background function already runs in a job subshell, so exec keeps $!
  # equal to the supervisor PID. Foreground calls need their own waiting shell.
  if [[ "${BASHPID}" != "$$" ]]; then
    exec "${supervisor_command[@]}"
  else
    (exec "${supervisor_command[@]}")
  fi
}

process_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

stop_supervised_process() {
  local pid="$1" exit_status
  [[ -n "${pid}" ]] || return 0
  if ! process_alive "${pid}"; then
    wait "${pid}" 2>/dev/null
    return $?
  fi
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 100); do
    process_alive "${pid}" || break
    sleep 0.1
  done
  if process_alive "${pid}"; then
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    return 1
  fi
  set +e
  wait "${pid}"
  exit_status=$?
  set -e
  [[ "${exit_status}" -eq 0 ]]
}

GPU_LOCK_GUARDIAN_PID=""
GPU_LOCK_STATE_DIR=""
cleanup_gpu_lease() {
  if [[ -n "${GPU_LOCK_GUARDIAN_PID}" ]]; then
    stop_supervised_process "${GPU_LOCK_GUARDIAN_PID}" || true
    GPU_LOCK_GUARDIAN_PID=""
  fi
  if [[ -n "${GPU_LOCK_STATE_DIR}" && -d "${GPU_LOCK_STATE_DIR}" ]]; then
    rm -f -- "${GPU_LOCK_STATE_DIR}/ready"
    rmdir -- "${GPU_LOCK_STATE_DIR}" 2>/dev/null || true
    GPU_LOCK_STATE_DIR=""
  fi
}
trap cleanup_gpu_lease EXIT

require_gpu_lease_guardian() {
  local guardian_status
  if process_alive "${GPU_LOCK_GUARDIAN_PID}"; then
    return 0
  fi
  set +e
  wait "${GPU_LOCK_GUARDIAN_PID}" 2>/dev/null
  guardian_status=$?
  set -e
  GPU_LOCK_GUARDIAN_PID=""
  die "GPU lease guardian exited unexpectedly with status ${guardian_status}"
}

run_supervised_with_gpu_lease() {
  local supervised_pid supervised_status
  require_gpu_lease_guardian
  run_supervised "$@" &
  supervised_pid=$!
  while process_alive "${supervised_pid}"; do
    if ! process_alive "${GPU_LOCK_GUARDIAN_PID}"; then
      stop_supervised_process "${supervised_pid}" || true
      require_gpu_lease_guardian
    fi
    sleep 0.1
  done
  set +e
  wait "${supervised_pid}"
  supervised_status=$?
  set -e
  require_gpu_lease_guardian
  [[ "${supervised_status}" -eq 0 ]]
}

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

if [[ "${DRY_RUN}" == true ]]; then
  [[ ! -L "${GPU_LOCK_DIR}" ]] || die "GPU lock directory must not be a symlink: ${GPU_LOCK_DIR}"
  mkdir -p -m 700 "${GPU_LOCK_DIR}"
  chmod 700 "${GPU_LOCK_DIR}"
fi
[[ -d "${GPU_LOCK_DIR}" && ! -L "${GPU_LOCK_DIR}" ]] || \
  die "GPU lock directory is missing or invalid: ${GPU_LOCK_DIR}"
GPU_LOCK_STATE_DIR="$(mktemp -d)"
run_supervised "${RLINF_PYTHON}" "${HELPER}" hold-gpu-locks \
  --lock-dir "${GPU_LOCK_DIR}" \
  --ready-file "${GPU_LOCK_STATE_DIR}/ready" &
GPU_LOCK_GUARDIAN_PID=$!
for _ in $(seq 1 100); do
  [[ -f "${GPU_LOCK_STATE_DIR}/ready" ]] && break
  process_alive "${GPU_LOCK_GUARDIAN_PID}" || break
  sleep 0.1
done
if [[ ! -f "${GPU_LOCK_STATE_DIR}/ready" ]] || ! process_alive "${GPU_LOCK_GUARDIAN_PID}"; then
  wait "${GPU_LOCK_GUARDIAN_PID}" 2>/dev/null || true
  GPU_LOCK_GUARDIAN_PID=""
  die "could not acquire host-wide GPU 0/1 leases"
fi
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
trap 'cleanup_preflight; cleanup_gpu_lease' EXIT
preflight_command=(
  "${RLINF_PYTHON}" "${HELPER}" preflight
  --bundle "${DSRL_BUNDLE}"
  --task-id "${TASK_ID}"
  --training-profile "${TRAINING_PROFILE}"
  --base-model "${BASE_MODEL}"
  --rlinf-repo "${RLINF_ROOT}"
  --t2-repo "${T2_ROOT}"
  --tabero-repo "${TABERO_ROOT}"
  --metadata-out "${PREFLIGHT_DIR}/bundle_and_repos.env"
)
if [[ "${DRY_RUN}" == true && "${TABERO_TEST_ALLOW_DIRTY:-}" == "1" ]]; then
  preflight_command+=(--allow-dirty)
fi
PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" run_supervised_with_gpu_lease "${preflight_command[@]}"

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
RUNTIME_VERIFY_DIR=""
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

write_running_status() {
  local temporary
  temporary="$(mktemp "${OUTPUT_DIR}/.run_status.env.XXXXXX")"
  {
    printf 'TABERO_RUN_STATUS=running\n'
    printf 'TABERO_EXIT_STATUS=\n'
    printf 'TABERO_START_TIME_UTC=%s\n' "${START_TIME_UTC}"
    printf 'TABERO_DURATION_SECONDS=\n'
  } >"${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

process_group_alive() {
  local pid="$1"
  kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null
}

terminate_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  stop_supervised_process "${pid}"
}

stop_gpu_sampler_successfully() {
  process_group_alive "${GPU_SAMPLER_PID}" || die "GPU sampler exited before shutdown"
  stop_supervised_process "${GPU_SAMPLER_PID}" || die "GPU sampler did not stop cleanly"
  GPU_SAMPLER_PID=""
  awk 'NR > 1 { found=1 } END { exit !found }' "${OUTPUT_DIR}/gpu_samples.csv" || \
    die "GPU sampler did not record an actual GPU sample"
}

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  terminate_group "${CLIENT_PID}" || true
  terminate_group "${SERVER_PID}" || true
  terminate_group "${GPU_SAMPLER_PID}" || true
  cleanup_gpu_lease
  cleanup_preflight
  if [[ -n "${RUNTIME_VERIFY_DIR}" && -d "${RUNTIME_VERIFY_DIR}" ]]; then
    rm -f -- "${RUNTIME_VERIFY_DIR}/current.env"
    rmdir -- "${RUNTIME_VERIFY_DIR}" 2>/dev/null || true
  fi
  if [[ "${STATUS_FINALIZED}" != true ]]; then
    write_status failed "${exit_status}"
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/output"
write_running_status
RUN_ID_SUFFIX="${RUN_STAMP/_/-}"
RUN_ID_SUFFIX="${RUN_ID_SUFFIX/_/-}"
WANDB_RUN_ID="tabero-official-dsrl-task${TASK_ID}-${RUN_ID_SUFFIX//_/-}"

PORT=""
if [[ "${DRY_RUN}" == true && -n "${TABERO_TEST_PORT:-}" ]]; then
  PORT="${TABERO_TEST_PORT}"
else
  PORT="$(run_supervised_with_gpu_lease "${RLINF_PYTHON}" - <<'PY'
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
PREFLIGHT_SNAPSHOT="${OUTPUT_DIR}/.preflight_snapshot.env"
RUN_ENV_TEMP="$(mktemp "${OUTPUT_DIR}/.run.env.XXXXXX")"
{
  printf 'TABERO_OFFICIAL_EVAL_RUN_ID=%s\n' "${RUN_STAMP}"
  printf 'WANDB_RUN_ID=%s\n' "${WANDB_RUN_ID}"
  printf 'TABERO_METHOD=dsrl\n'
  printf 'TABERO_TASK_ID=%s\n' "${TASK_ID}"
  printf 'TABERO_RUN_MODE=formal\n'
  printf 'TABERO_DSRL_TRAINING_PROFILE=%s\n' "${TRAINING_PROFILE}"
  printf 'TABERO_EXPECTED_EPISODES=50\n'
  printf 'TABERO_RECORD_STEP_TRACES=true\n'
  printf 'TABERO_DSRL_BUNDLE=%s\n' "${DSRL_BUNDLE}"
  printf 'TABERO_OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'TABERO_START_TIME_UTC=%s\n' "${START_TIME_UTC}"
} >"${RUN_ENV_TEMP}"
mv "${RUN_ENV_TEMP}" "${RUN_ENV}"
START_METADATA_TEMP="$(mktemp "${OUTPUT_DIR}/.start_metadata.env.XXXXXX")"
{
  cat "${PREFLIGHT_DIR}/bundle_and_repos.env"
  printf 'TABERO_GPU_IDS=0,1\n'
  printf 'TABERO_CUDA_DEVICE_ORDER=PCI_BUS_ID\n'
  printf 'TABERO_FREE_DISK_KIB=%s\n' "${FREE_DISK_KIB}"
  printf 'TABERO_EPHEMERAL_PORT=%s\n' "${PORT}"
  printf 'TABERO_START_TIME_UTC=%s\n' "${START_TIME_UTC}"
} >"${START_METADATA_TEMP}"
mv "${START_METADATA_TEMP}" "${START_METADATA}"
cp -- "${PREFLIGHT_DIR}/bundle_and_repos.env" "${PREFLIGHT_SNAPSHOT}"
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
  --record-step-traces
  --step-trace-dir "${OUTPUT_DIR}/step_traces"
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
  rm -f -- "${PREFLIGHT_SNAPSHOT}"
  write_status dry_run 0
  STATUS_FINALIZED=true
  printf 'Output directory: %s\n' "${OUTPUT_DIR}"
  exit 0
fi

require_gpu_lease_guardian
run_supervised "${RLINF_PYTHON}" "${HELPER}" sample-gpus \
  --gpu-file "${OUTPUT_DIR}/gpu_samples.csv" \
  --process-file "${OUTPUT_DIR}/gpu_process_samples.csv" \
  --interval 5 \
  --parent-pid "$$" &
GPU_SAMPLER_PID=$!
printf '%s\n' "${GPU_SAMPLER_PID}" >"${OUTPUT_DIR}/gpu_sampler.pid"

require_gpu_lease_guardian
run_supervised "${server_command[@]}" >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${OUTPUT_DIR}/server.pid"
ready=false
for _ in $(seq 1 300); do
  require_gpu_lease_guardian
  process_group_alive "${GPU_SAMPLER_PID}" || die "GPU sampler exited before server readiness"
  process_group_alive "${SERVER_PID}" || die "T2 server exited before becoming ready"
  if PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    run_supervised_with_gpu_lease "${RLINF_PYTHON}" "${HELPER}" listener-owned --pid "${SERVER_PID}" --port "${PORT}"
  then
    ready=true
    break
  fi
  sleep 2
done
[[ "${ready}" == true ]] || die "T2 server did not listen within 600 seconds"

require_gpu_lease_guardian
run_supervised "${client_command[@]}" >"${OUTPUT_DIR}/client.log" 2>&1 &
CLIENT_PID=$!
printf '%s\n' "${CLIENT_PID}" >"${OUTPUT_DIR}/client.pid"
while process_group_alive "${CLIENT_PID}"; do
  require_gpu_lease_guardian
  process_group_alive "${SERVER_PID}" || die "T2 server exited while client was running"
  process_group_alive "${GPU_SAMPLER_PID}" || die "GPU sampler exited while client was running"
  sleep 1
done
set +e
wait "${CLIENT_PID}"
CLIENT_STATUS=$?
set -e
CLIENT_PID=""
[[ "${CLIENT_STATUS}" -eq 0 ]] || die "Tabero client failed with exit status ${CLIENT_STATUS}"
require_gpu_lease_guardian
process_group_alive "${SERVER_PID}" || die "T2 server exited before controlled shutdown"
process_group_alive "${GPU_SAMPLER_PID}" || die "GPU sampler exited before controlled shutdown"
stop_gpu_sampler_successfully
stop_supervised_process "${SERVER_PID}" || die "T2 server did not stop cleanly"
SERVER_PID=""

RUNTIME_VERIFY_DIR="$(mktemp -d "${OUTPUT_DIR}/.runtime_verify.XXXXXX")"
runtime_preflight_command=(
  "${RLINF_PYTHON}" "${HELPER}" preflight
  --bundle "${DSRL_BUNDLE}"
  --task-id "${TASK_ID}"
  --training-profile "${TRAINING_PROFILE}"
  --base-model "${BASE_MODEL}"
  --rlinf-repo "${RLINF_ROOT}"
  --t2-repo "${T2_ROOT}"
  --tabero-repo "${TABERO_ROOT}"
  --metadata-out "${RUNTIME_VERIFY_DIR}/current.env"
)
PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" run_supervised_with_gpu_lease "${runtime_preflight_command[@]}"
cmp --silent "${PREFLIGHT_SNAPSHOT}" "${RUNTIME_VERIFY_DIR}/current.env" || \
  die "bundle, base model, or repository provenance changed during evaluation"
rm -f -- "${RUNTIME_VERIFY_DIR}/current.env" "${PREFLIGHT_SNAPSHOT}"
rmdir -- "${RUNTIME_VERIFY_DIR}"
RUNTIME_VERIFY_DIR=""

PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" run_supervised_with_gpu_lease "${RLINF_PYTHON}" "${HELPER}" finalize \
  --raw-dir "${OUTPUT_DIR}/raw" \
  --output-dir "${OUTPUT_DIR}" \
  --bundle "${DSRL_BUNDLE}" \
  --task-id "${TASK_ID}" \
  --base-model "${BASE_MODEL}" \
  --training-profile "${TRAINING_PROFILE}" \
  --run-id "${WANDB_RUN_ID}"
require_gpu_lease_guardian
stop_supervised_process "${GPU_LOCK_GUARDIAN_PID}" || die "GPU lease guardian did not stop cleanly"
GPU_LOCK_GUARDIAN_PID=""
rm -f -- "${GPU_LOCK_STATE_DIR}/ready"
rmdir -- "${GPU_LOCK_STATE_DIR}"
GPU_LOCK_STATE_DIR=""
trap '' INT TERM
write_status completed 0
STATUS_FINALIZED=true
exit 0
