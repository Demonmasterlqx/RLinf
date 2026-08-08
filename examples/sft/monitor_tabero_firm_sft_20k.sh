#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?run directory is required}"
TRAIN_SESSION="${2:?tmux session is required}"
TRAIN_LOG="${RUN_DIR}/train.log"
HEALTH_LOG="${RUN_DIR}/health_monitor.log"
HEALTH_STATE="${RUN_DIR}/health_monitor.state"

while tmux list-panes -t "${TRAIN_SESSION}:train" >/dev/null 2>&1; do
  timestamp="$(date --iso-8601=seconds)"
  latest_step=""
  suspicious=""
  if [[ -f "${TRAIN_LOG}" ]]; then
    latest_step="$(tr '\r' '\n' <"${TRAIN_LOG}" | \
      grep 'Global Step:.*train/loss=' | tail -n 1 || true)"
    suspicious="$(grep -E \
      'Traceback|CUDA out of memory|OutOfMemoryError|NCCL[^[:space:]]* (error|failure)|train/(loss|grad_norm)=(nan|inf)' \
      "${TRAIN_LOG}" | tail -n 5 || true)"
  fi
  checkpoints="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type d \
    -name 'global_step_*' -printf '%f ' 2>/dev/null || true)"

  {
    printf 'timestamp=%s\n' "${timestamp}"
    printf 'tmux_session=%s\n' "${TRAIN_SESSION}"
    printf 'latest_progress=%s\n' "${latest_step}"
    printf 'checkpoints=%s\n' "${checkpoints}"
    printf 'suspicious_log_lines=%s\n' "${suspicious}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
    while IFS=, read -r gpu_uuid pid process_name used_memory; do
      pid="${pid//[[:space:]]/}"
      printf 'compute_process='
      ps -o user=,pid=,ppid=,etime=,cmd= -p "${pid}" || \
        printf 'missing pid=%s gpu_uuid=%s process=%s memory=%s\n' \
          "${pid}" "${gpu_uuid}" "${process_name}" "${used_memory}"
    done < <(nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits)
  } >"${HEALTH_STATE}.tmp"
  mv "${HEALTH_STATE}.tmp" "${HEALTH_STATE}"
  cat "${HEALTH_STATE}" >>"${HEALTH_LOG}"
  printf '\n' >>"${HEALTH_LOG}"
  sleep 300
done

printf 'timestamp=%s\nstatus=training_tmux_train_window_stopped\n' \
  "$(date --iso-8601=seconds)" >"${HEALTH_STATE}"
