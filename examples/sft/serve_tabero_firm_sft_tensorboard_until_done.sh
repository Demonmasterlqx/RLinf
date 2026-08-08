#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?run directory is required}"
TRAIN_SESSION="${2:?tmux session is required}"
PORT="${3:-6006}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

while [[ ! -d "${RUN_DIR}/tensorboard" ]]; do
  tmux list-panes -t "${TRAIN_SESSION}:train" >/dev/null 2>&1 || exit 1
  sleep 5
done

"${REPO_ROOT}/.venv/bin/tensorboard" \
  --logdir "${RUN_DIR}/tensorboard" \
  --host 127.0.0.1 \
  --port "${PORT}" >"${RUN_DIR}/tensorboard_server.log" 2>&1 &
TB_PID=$!
printf '%s\n' "${TB_PID}" >"${RUN_DIR}/tensorboard_server.pid"
trap 'kill "${TB_PID}" 2>/dev/null || true; wait "${TB_PID}" 2>/dev/null || true' EXIT INT TERM

while tmux list-panes -t "${TRAIN_SESSION}:train" >/dev/null 2>&1; do
  kill -0 "${TB_PID}" 2>/dev/null || exit 1
  sleep 10
done
