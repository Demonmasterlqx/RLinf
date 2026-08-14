#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_DIR="${1:?model directory is required}"
readonly OUTPUT_ROOT="${2:?evaluation output directory is required}"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
readonly WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
readonly T2_ROOT="${WORKSPACE_ROOT}/T2-VLA"
readonly TABERO_ROOT="${WORKSPACE_ROOT}/Tabero"
readonly PROFILE_DIR="${TABERO_TASK0_EVAL_PROFILE_DIR:-${TABERO_ROOT}/benchmarks/datasets/libero/config_profiles/alltask_fixed_damage_1000000_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k}"
readonly SERVER_GPU=0
readonly CLIENT_GPU=1
readonly PORT="${TABERO_TASK0_EVAL_PORT:-8032}"
SERVER_PID=""
GPU_MONITOR_PID=""

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -f "${MODEL_DIR}/model.safetensors" ]] || die "model.safetensors is missing"
[[ -f "${MODEL_DIR}/config.json" ]] || die "model config is missing"
[[ -x "${T2_ROOT}/.venv/bin/python" ]] || die "T2-VLA Python is missing"
[[ -x "/data/home/sim6g/anaconda3/bin/conda" ]] || die "conda is missing"
[[ -f "${PROFILE_DIR}/libero_object.json" ]] || die "evaluation profile is missing"
[[ -d "${TABERO_ROOT}/benchmarks/datasets/libero/assembled_hdf5" ]] || \
  die "assembled HDF5 directory is missing"

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    for _wait in $(seq 1 60); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -0 "${SERVER_PID}" 2>/dev/null && kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  stop_server
}
trap cleanup EXIT INT TERM

wait_for_eval_gpus() {
  local deadline=$(( $(date +%s) + 600 )) gpu_id compute_pids used_memory busy
  while true; do
    busy=false
    for gpu_id in "${SERVER_GPU}" "${CLIENT_GPU}"; do
      compute_pids="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits)"
      used_memory="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ -n "${compute_pids//[[:space:]]/}" ]] || ((used_memory >= 1024)); then busy=true; fi
    done
    [[ "${busy}" == false ]] && return 0
    (( $(date +%s) < deadline )) || die "evaluation GPUs did not become idle"
    sleep 10
  done
}

mkdir -p "${OUTPUT_ROOT}/raw"
wait_for_eval_gpus
! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$" || die "port ${PORT} is busy"

{
  printf 'model=%s\nprofile=%s\nserver_gpu=%s\nclient_gpu=%s\nport=%s\n' \
    "${MODEL_DIR}" "${PROFILE_DIR}" "${SERVER_GPU}" "${CLIENT_GPU}" "${PORT}"
  printf 'task=libero_object:0\nepisodes=50\nseed=11\nprompt_mode=plain\n'
  printf 'success_hold_steps=8\nmax_inference_chunks=30\nreplan_steps=10\n'
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} >"${OUTPUT_ROOT}/launch_meta.txt"

(
  cd "${T2_ROOT}"
  export CUDA_VISIBLE_DEVICES="${SERVER_GPU}" JAX_PLATFORMS=cuda
  export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
  exec "${T2_ROOT}/.venv/bin/python" scripts/serve_policy.py \
    --port "${PORT}" policy:checkpoint \
    --policy.config=pi0_lora_tacfield_tabero --policy.dir="${MODEL_DIR}"
) >"${OUTPUT_ROOT}/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${OUTPUT_ROOT}/server.pid"

ready=false
for _ready_wait in $(seq 1 600); do
  if grep -q "server listening on 0.0.0.0:${PORT}" "${OUTPUT_ROOT}/server.log"; then ready=true; break; fi
  kill -0 "${SERVER_PID}" 2>/dev/null || break
  sleep 1
done
[[ "${ready}" == true ]] || die "policy server did not become ready"

(
  while kill -0 "${SERVER_PID}" 2>/dev/null; do
    timestamp="$(date --iso-8601=seconds)"
    nvidia-smi --id=0,1 --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | while IFS= read -r sample; do
        printf '%s,%s\n' "${timestamp}" "${sample}"
      done
    sleep 10
  done
) >"${OUTPUT_ROOT}/health.log" &
GPU_MONITOR_PID=$!

set +e
(
  cd "${TABERO_ROOT}"
  env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
    /data/home/sim6g/anaconda3/bin/conda run --no-capture-output -n tabero python -u \
    scripts/tools/run_task_evaluations.py \
    --policy-model openpi --control-mode tactile \
    --server-host 127.0.0.1 --server-port "${PORT}" \
    --num-total-experiments 50 --num-success-steps 8 \
    --max-inference-steps 30 --replan-steps 10 \
    --task-suites libero_object --task-ids 0 \
    --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 --require-hdf5 \
    --config-path "${PROFILE_DIR}" \
    --output-dir "${OUTPUT_ROOT}/raw" --output-format both \
    --seed 11 --prompt-seed 0 --prompt-adverbs \
    --sim-device "cuda:${CLIENT_GPU}" \
    --sim-kit-args="--/renderer/activeGpu=${CLIENT_GPU}" --headless
) >"${OUTPUT_ROOT}/client.log" 2>&1
client_exit=$?
set -e
printf '%s\n' "${client_exit}" >"${OUTPUT_ROOT}/client.exit_code"
((client_exit == 0)) || die "evaluation client exited ${client_exit}"

mapfile -t json_paths < <(find "${OUTPUT_ROOT}/raw" -maxdepth 1 -name 'success_rates_openpi_tactile_*.json' -type f | sort)
mapfile -t txt_paths < <(find "${OUTPUT_ROOT}/raw" -maxdepth 1 -name 'success_rates_openpi_tactile_*.txt' -type f | sort)
[[ ${#json_paths[@]} -eq 1 && ${#txt_paths[@]} -eq 1 ]] || die "raw result count mismatch"
PYTHONPATH="${REPO_ROOT}" "${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/toolkits/checkpoint/audit_tabero_task0_eval.py" \
  --json "${json_paths[0]}" --txt "${txt_paths[0]}" \
  --client-log "${OUTPUT_ROOT}/client.log" \
  --output "${OUTPUT_ROOT}/evaluation_audit.json" \
  >"${OUTPUT_ROOT}/evaluation_audit.log" 2>&1
printf 'status=completed_and_cross_checked\ncompleted_at=%s\n' \
  "$(date --iso-8601=seconds)" >"${OUTPUT_ROOT}/evaluation.state"
