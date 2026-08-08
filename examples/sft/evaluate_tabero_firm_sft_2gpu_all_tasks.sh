#!/usr/bin/env bash
set -euo pipefail

EXPORT_DIR="${1:?export directory is required}"
OUTPUT_ROOT="${2:?evaluation output directory is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
T2_ROOT="${WORKSPACE_ROOT}/T2-VLA"
TABERO_ROOT="${WORKSPACE_ROOT}/Tabero"
SERVER_GPU=0
CLIENT_GPU=1
PORT="${TABERO_FIRM_EVAL_PORT:-8016}"
SERVER_PID=""
GPU_MONITOR_PID=""

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ -f "${EXPORT_DIR}/model.safetensors" ]] || die "exported model is missing"
[[ -f "${EXPORT_DIR}/export_meta.json" ]] || die "export metadata is missing"
[[ -x "${T2_ROOT}/.venv/bin/python" ]] || die "T2-VLA Python is missing"
[[ -x "/data/home/sim6g/anaconda3/bin/conda" ]] || die "conda is missing"
[[ -f "${TABERO_ROOT}/scripts/tools/run_task_evaluations.py" ]] || \
  die "Tabero evaluator is missing"
[[ -d "${TABERO_ROOT}/benchmarks/datasets/libero/assembled_hdf5" ]] || \
  die "assembled HDF5 directory is missing"

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    for _wait in $(seq 1 60); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  GPU_MONITOR_PID=""
}

cleanup() {
  stop_gpu_monitor
  stop_server
}
trap cleanup EXIT INT TERM

wait_for_eval_gpus() {
  local deadline=$(( $(date +%s) + 600 ))
  local gpu_id compute_pids used_memory
  while true; do
    local busy=false
    for gpu_id in "${SERVER_GPU}" "${CLIENT_GPU}"; do
      compute_pids="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid \
        --format=csv,noheader,nounits)"
      used_memory="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ -n "${compute_pids//[[:space:]]/}" ]] || ((used_memory >= 1024)); then
        busy=true
      fi
    done
    [[ "${busy}" == false ]] && return 0
    (( $(date +%s) < deadline )) || die "evaluation GPUs did not become idle"
    sleep 10
  done
}

port_is_free() {
  ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"
}

run_attempt() {
  local attempt="$1"
  local attempt_dir="${OUTPUT_ROOT}/attempt${attempt}"
  local server_log="${attempt_dir}/server.log"
  local client_log="${attempt_dir}/client.log"
  local health_log="${attempt_dir}/health.log"
  local client_exit audit_exit json_path txt_path

  mkdir -p "${attempt_dir}/raw"
  wait_for_eval_gpus
  port_is_free || die "evaluation port ${PORT} is already in use"

  {
    printf 'attempt=%s\n' "${attempt}"
    printf 'method=sft_full_lora_tacfield\n'
    printf 'model=%s\n' "${EXPORT_DIR}"
    printf 'server_gpu=%s\n' "${SERVER_GPU}"
    printf 'client_gpu=%s\n' "${CLIENT_GPU}"
    printf 'port=%s\n' "${PORT}"
    printf 'tasks=0,1,2,3,5,6,7,8,9\n'
    printf 'episodes_per_task=50\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  } >"${attempt_dir}/launch_meta.txt"

  (
    cd "${T2_ROOT}"
    export CUDA_VISIBLE_DEVICES="${SERVER_GPU}"
    export JAX_PLATFORMS=cuda
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONUNBUFFERED=1
    exec "${T2_ROOT}/.venv/bin/python" scripts/serve_policy.py \
      --port "${PORT}" \
      policy:checkpoint \
      --policy.config=pi0_lora_tacfield_tabero \
      --policy.dir="${EXPORT_DIR}"
  ) >"${server_log}" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "${SERVER_PID}" >"${attempt_dir}/server.pid"

  local ready=false
  for _ready_wait in $(seq 1 600); do
    if grep -q "server listening on 0.0.0.0:${PORT}" "${server_log}"; then
      ready=true
      break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null || break
    sleep 1
  done
  if [[ "${ready}" != true ]]; then
    printf 'server_not_ready\n' >"${attempt_dir}/attempt_status"
    stop_server
    return 1
  fi

  (
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
      timestamp="$(date --iso-8601=seconds)"
      nvidia-smi --id=0,1 \
        --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | \
        while IFS= read -r sample; do
          printf '%s,%s\n' "${timestamp}" "${sample}"
        done
      sleep 10
    done
  ) >"${health_log}" &
  GPU_MONITOR_PID=$!

  set +e
  (
    cd "${TABERO_ROOT}"
    env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
      /data/home/sim6g/anaconda3/bin/conda run --no-capture-output \
      -n tabero python -u \
      "${TABERO_ROOT}/scripts/tools/run_task_evaluations.py" \
      --policy-model openpi \
      --control-mode tactile \
      --server-host 127.0.0.1 \
      --server-port "${PORT}" \
      --num-total-experiments 50 \
      --num-success-steps 8 \
      --max-inference-steps 30 \
      --replan-steps 10 \
      --task-suites libero_object \
      --task-ids 0 1 2 3 5 6 7 8 9 \
      --hdf5-folder "${TABERO_ROOT}/benchmarks/datasets/libero/assembled_hdf5" \
      --require-hdf5 \
      --output-dir "${attempt_dir}/raw" \
      --output-format both \
      --seed 11 \
      --prompt-seed 0 \
      --prompt-adverbs firmly tightly \
      --sim-device "cuda:${CLIENT_GPU}" \
      --sim-kit-args="--/renderer/activeGpu=${CLIENT_GPU}" \
      --headless
  ) >"${client_log}" 2>&1
  client_exit=$?
  set -e
  printf '%s\n' "${client_exit}" >"${attempt_dir}/client.exit_code"

  stop_gpu_monitor
  stop_server
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >>"${attempt_dir}/launch_meta.txt"
  ((client_exit == 0)) || {
    printf 'client_exit_%s\n' "${client_exit}" >"${attempt_dir}/attempt_status"
    return 1
  }

  mapfile -t json_paths < <(find "${attempt_dir}/raw" -maxdepth 1 \
    -name 'success_rates_openpi_tactile_*.json' -type f | sort)
  mapfile -t txt_paths < <(find "${attempt_dir}/raw" -maxdepth 1 \
    -name 'success_rates_openpi_tactile_*.txt' -type f | sort)
  [[ ${#json_paths[@]} -eq 1 && ${#txt_paths[@]} -eq 1 ]] || {
    printf 'raw_result_count_mismatch\n' >"${attempt_dir}/attempt_status"
    return 1
  }
  json_path="${json_paths[0]}"
  txt_path="${txt_paths[0]}"
  set +e
  "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/toolkits/checkpoint/audit_tabero_firm_all_task_eval.py" \
    --json "${json_path}" \
    --txt "${txt_path}" \
    --client-log "${client_log}" \
    --output "${attempt_dir}/evaluation_audit.json" \
    >"${attempt_dir}/evaluation_audit.log" 2>&1
  audit_exit=$?
  set -e
  if ((audit_exit != 0)); then
    printf 'audit_exit_%s\n' "${audit_exit}" >"${attempt_dir}/attempt_status"
    return 1
  fi
  printf 'completed_and_cross_checked\n' >"${attempt_dir}/attempt_status"
  printf 'attempt%s\n' "${attempt}" >"${OUTPUT_ROOT}/final_attempt"
  return 0
}

mkdir -p "${OUTPUT_ROOT}"
for attempt in 1 2; do
  if run_attempt "${attempt}"; then
    printf 'status=completed_and_cross_checked\nattempt=%s\ncompleted_at=%s\n' \
      "${attempt}" "$(date --iso-8601=seconds)" >"${OUTPUT_ROOT}/evaluation.state"
    exit 0
  fi
  printf '%s attempt=%s failed; preparing one full retry if available\n' \
    "$(date --iso-8601=seconds)" "${attempt}" >>"${OUTPUT_ROOT}/evaluation.log"
done

printf 'status=failed_after_one_retry\ncompleted_at=%s\n' \
  "$(date --iso-8601=seconds)" >"${OUTPUT_ROOT}/evaluation.state"
exit 1
