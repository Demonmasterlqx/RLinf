#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?run directory is required}"
TRAIN_SESSION="${2:?tmux session is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
AUDIT_DIR="${RUN_DIR}/checkpoint_audits"
STATE_PATH="${RUN_DIR}/checkpoint_audit_monitor.state"
LOG_PATH="${RUN_DIR}/checkpoint_audit_monitor.log"
POLL_SECONDS="${TABERO_FIRM_CHECKPOINT_AUDIT_POLL_SECONDS:-30}"
EXPECTED_STEPS=(2000 4000 6000 8000 10000 12000 14000 16000 18000 20000)
declare -A LAST_SIZE

write_state() {
  local status="$1"
  local step="${2:-}"
  local detail="${3:-}"
  local tmp_path="${STATE_PATH}.tmp"
  {
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'status=%s\n' "${status}"
    printf 'step=%s\n' "${step}"
    printf 'detail=%s\n' "${detail}"
    printf 'audited='
    find "${AUDIT_DIR}" -maxdepth 1 -type f -name 'global_step_*.json' \
      -printf '%f ' 2>/dev/null | sort -V
    printf '\n'
  } >"${tmp_path}"
  mv "${tmp_path}" "${STATE_PATH}"
  cat "${STATE_PATH}" >>"${LOG_PATH}"
  printf '\n' >>"${LOG_PATH}"
}

layout_present() {
  local step_dir="$1"
  [[ -s "${step_dir}/actor/dcp_checkpoint/.metadata" && \
      -s "${step_dir}/actor/data_state.json" && \
      -s "${step_dir}/actor/model_state_dict/trainable_weights.pt" ]] || return 1
  [[ "$(find "${step_dir}/actor/dcp_checkpoint" -maxdepth 1 \
    -type f -name '*.distcp' -size +0c | wc -l)" -eq 2 ]]
}

checkpoint_size() {
  local step_dir="$1"
  find "${step_dir}" -type f -printf '%s\n' | \
    awk '{total += $1} END {printf "%.0f\n", total + 0}'
}

mkdir -p "${AUDIT_DIR}"
write_state waiting "" "waiting_for_global_step_2000"

while true; do
  all_audited=true
  for step in "${EXPECTED_STEPS[@]}"; do
    output_path="${AUDIT_DIR}/global_step_${step}.json"
    [[ -f "${output_path}" ]] && continue
    all_audited=false
    step_dir="${RUN_DIR}/checkpoints/global_step_${step}"
    [[ -d "${step_dir}" ]] || continue

    size_now="$(checkpoint_size "${step_dir}")"
    write_state writing "${step}" "bytes=${size_now}"
    if ! layout_present "${step_dir}" || \
       [[ "${LAST_SIZE[${step}]:-}" != "${size_now}" ]]; then
      LAST_SIZE[${step}]="${size_now}"
      continue
    fi

    tmp_output="${output_path}.tmp"
    audit_args=(
      --step-dir "${step_dir}"
      --expected-step "${step}"
      --output "${tmp_output}"
    )
    ((step == 20000)) && audit_args+=(--expect-final)
    if ! "${REPO_ROOT}/.venv/bin/python" \
      "${REPO_ROOT}/toolkits/checkpoint/audit_tabero_firm_sft_checkpoint_dir.py" \
      "${audit_args[@]}" >>"${LOG_PATH}" 2>&1; then
      rm -f -- "${tmp_output}"
      write_state audit_failed "${step}" "see=${LOG_PATH}"
      exit 1
    fi
    mv "${tmp_output}" "${output_path}"
    write_state audited "${step}" "output=${output_path}"
  done

  if [[ "${all_audited}" == true ]]; then
    write_state completed 20000 "all_10_checkpoints_audited"
    exit 0
  fi
  if ! tmux list-panes -t "${TRAIN_SESSION}:train" >/dev/null 2>&1; then
    write_state stopped_incomplete "" "training_pipeline_window_stopped"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done
