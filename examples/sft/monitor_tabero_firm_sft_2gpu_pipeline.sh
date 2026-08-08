#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?run directory is required}"
TRAIN_SESSION="${2:?tmux session is required}"
HEALTH_LOG="${RUN_DIR}/health_monitor.log"
HEALTH_STATE="${RUN_DIR}/health_monitor.state"
GPU_SAMPLES="${RUN_DIR}/gpu_samples.csv"
LAST_PROGRESS=""
LAST_PROGRESS_EPOCH="$(date +%s)"

printf 'timestamp,gpu,memory_used_mib,memory_total_mib,utilization_pct\n' \
  >"${GPU_SAMPLES}"

while tmux list-panes -t "${TRAIN_SESSION}:train" >/dev/null 2>&1; do
  loop_start="$(date +%s)"
  for _sample in 1 2 3 4 5 6; do
    timestamp="$(date --iso-8601=seconds)"
    nvidia-smi --id=0,1 \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | \
      while IFS= read -r sample; do
        printf '%s,%s\n' "${timestamp}" "${sample}" >>"${GPU_SAMPLES}"
      done
    sleep 10
  done

  latest_line=""
  latest_progress=""
  for train_log in "${RUN_DIR}"/train*.log; do
    [[ -f "${train_log}" ]] || continue
    candidate_line="$(tr '\r' '\n' <"${train_log}" | \
      grep 'train/loss=' | tail -n 1 || true)"
    [[ -n "${candidate_line}" ]] && latest_line="${candidate_line}"
    candidate_progress="$(tr '\r' '\n' <"${train_log}" | \
      grep -oE '[0-9]+/20000' | tail -n 1 | cut -d/ -f1 || true)"
    [[ -n "${candidate_progress}" ]] && latest_progress="${candidate_progress}"
  done
  if [[ -n "${latest_progress}" && "${latest_progress}" != "${LAST_PROGRESS}" ]]; then
    LAST_PROGRESS="${latest_progress}"
    LAST_PROGRESS_EPOCH="$(date +%s)"
  fi
  now_epoch="$(date +%s)"
  stalled_seconds=$((now_epoch - LAST_PROGRESS_EPOCH))
  stalled=false
  ((stalled_seconds >= 900)) && stalled=true

  suspicious="$(grep -Eih \
    'Traceback|CUDA out of memory|OutOfMemoryError|NCCL.*(error|failure)|RayActorError|ActorDiedError|train/(loss|grad_norm)=(nan|inf)' \
    "${RUN_DIR}"/train*.log 2>/dev/null | tail -n 10 || true)"
  checkpoints="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type d \
    -name 'global_step_*' -printf '%f ' 2>/dev/null || true)"
  tmp_state="${HEALTH_STATE}.tmp"
  {
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'tmux_session=%s\n' "${TRAIN_SESSION}"
    printf 'latest_progress=%s\n' "${latest_progress}"
    printf 'latest_metric_line=%s\n' "${latest_line}"
    printf 'seconds_since_progress=%s\n' "${stalled_seconds}"
    printf 'stalled=%s\n' "${stalled}"
    printf 'checkpoints=%s\n' "${checkpoints}"
    printf 'suspicious_log_lines=%s\n' "${suspicious}"
    nvidia-smi --id=0,1 \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    while IFS=, read -r gpu_uuid pid process_name used_memory; do
      pid="${pid//[[:space:]]/}"
      printf 'compute_process='
      ps -o user=,pid=,ppid=,etime=,cmd= -p "${pid}" || \
        printf 'missing pid=%s gpu_uuid=%s process=%s memory=%s\n' \
          "${pid}" "${gpu_uuid}" "${process_name}" "${used_memory}"
    done < <(nvidia-smi --id=0,1 \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits)
    printf 'loop_seconds=%s\n' "$(( $(date +%s) - loop_start ))"
  } >"${tmp_state}"
  mv "${tmp_state}" "${HEALTH_STATE}"
  cat "${HEALTH_STATE}" >>"${HEALTH_LOG}"
  printf '\n' >>"${HEALTH_LOG}"
done

printf 'timestamp=%s\nstatus=training_tmux_train_window_stopped\n' \
  "$(date --iso-8601=seconds)" >"${HEALTH_STATE}"
