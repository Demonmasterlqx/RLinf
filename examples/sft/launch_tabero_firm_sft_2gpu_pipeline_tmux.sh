#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
RUN_STAMP="${TABERO_FIRM_2GPU_RUN_STAMP:-$(date +'%Y%m%d_%H%M%S')}"
SESSION="tabero_firm_rlinf_2gpu_mb16_20k_${RUN_STAMP}"
RUN_DIR="${WORKSPACE_ROOT}/results/tabero_firm_rlinf_2gpu_mb16_20k_${RUN_STAMP}"
TMUX_LOG="${WORKSPACE_ROOT}/results/${SESSION}_tmux.log"
TB_PORT="${TABERO_FIRM_TENSORBOARD_PORT:-6006}"

command -v tmux >/dev/null 2>&1 || {
  printf 'error: tmux is required\n' >&2
  exit 1
}
tmux has-session -t "${SESSION}" 2>/dev/null && {
  printf 'error: tmux session already exists: %s\n' "${SESSION}" >&2
  exit 1
}
ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${TB_PORT}$" && {
  printf 'error: TensorBoard port %s is already in use\n' "${TB_PORT}" >&2
  exit 1
}

mkdir -p "${WORKSPACE_ROOT}/results"
{
  printf 'session=%s\n' "${SESSION}"
  printf 'run_stamp=%s\n' "${RUN_STAMP}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'tensorboard_port=%s\n' "${TB_PORT}"
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
} >"${TMUX_LOG}"

tmux new-session -d -s "${SESSION}" -n train \
  "cd '${REPO_ROOT}' && exec '${SCRIPT_DIR}/run_tabero_firm_sft_2gpu_pipeline.sh' '${RUN_STAMP}'"
tmux pipe-pane -o -t "${SESSION}:train" "cat >>'${TMUX_LOG}'"

tmux new-window -t "${SESSION}" -n health \
  "while [[ ! -d '${RUN_DIR}' ]]; do tmux list-panes -t '${SESSION}:train' >/dev/null 2>&1 || exit 1; sleep 5; done; exec '${SCRIPT_DIR}/monitor_tabero_firm_sft_2gpu_pipeline.sh' '${RUN_DIR}' '${SESSION}'"

tmux new-window -t "${SESSION}" -n tensorboard \
  "exec '${SCRIPT_DIR}/serve_tabero_firm_sft_tensorboard_until_done.sh' '${RUN_DIR}' '${SESSION}' '${TB_PORT}'"

tmux new-window -t "${SESSION}" -n checkpoint_audit \
  "while [[ ! -d '${RUN_DIR}' ]]; do tmux list-panes -t '${SESSION}:train' >/dev/null 2>&1 || exit 1; sleep 5; done; exec '${SCRIPT_DIR}/audit_tabero_firm_sft_2gpu_checkpoints_until_done.sh' '${RUN_DIR}' '${SESSION}'"

tmux select-window -t "${SESSION}:train"
printf 'session=%s\nrun_stamp=%s\nrun_dir=%s\nattach=tmux attach -t %s\n' \
  "${SESSION}" "${RUN_STAMP}" "${RUN_DIR}" "${SESSION}"
