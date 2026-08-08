#!/usr/bin/env bash
set -uo pipefail

RUN_STAMP="${1:?run stamp is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
RUN_NAME="tabero_firm_rlinf_2gpu_mb16_20k_${RUN_STAMP}"
RUN_DIR="${WORKSPACE_ROOT}/results/${RUN_NAME}"
STATE_PATH="${WORKSPACE_ROOT}/results/${RUN_NAME}_supervisor.state"
LOG_PATH="${WORKSPACE_ROOT}/results/${RUN_NAME}_supervisor.log"

write_state() {
  local status="$1"
  local tmp_path="${STATE_PATH}.tmp"
  printf 'status=%s\nrun_dir=%s\nupdated_at=%s\n' \
    "${status}" "${RUN_DIR}" "$(date --iso-8601=seconds)" >"${tmp_path}"
  mv "${tmp_path}" "${STATE_PATH}"
}

latest_complete_checkpoint() {
  local candidate step
  while IFS= read -r step; do
    candidate="${RUN_DIR}/checkpoints/global_step_${step}"
    if [[ -f "${candidate}/actor/dcp_checkpoint/.metadata" && \
          -f "${candidate}/actor/data_state.json" && \
          -f "${candidate}/actor/model_state_dict/trainable_weights.pt" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type d \
    -name 'global_step_*' -printf '%f\n' 2>/dev/null | \
    sed 's/global_step_//' | sort -rn)
  return 1
}

write_state starting
printf '%s supervisor_start run_dir=%s\n' \
  "$(date --iso-8601=seconds)" "${RUN_DIR}" >>"${LOG_PATH}"

TABERO_FIRM_2GPU_RUN_STAMP="${RUN_STAMP}" \
  "${SCRIPT_DIR}/run_tabero_firm_sft_2gpu_selective_siglip_20k.sh" formal \
  >>"${LOG_PATH}" 2>&1
FIRST_EXIT=$?
[[ -d "${RUN_DIR}" ]] && printf '%s\n' "${FIRST_EXIT}" >"${RUN_DIR}/formal_exit_code_attempt1"

if ((FIRST_EXIT == 0)); then
  write_state completed
  printf '%s supervisor_exit=0 recovery_attempted=false\n' \
    "$(date --iso-8601=seconds)" >>"${LOG_PATH}"
  exit 0
fi

TRAIN_LOG="${RUN_DIR}/train.log"
FATAL_PATTERN='CUDA out of memory|OutOfMemoryError|train/(loss|grad_norm)=(nan|inf)|FloatingPointError|non-finite'
TRANSIENT_PATTERN='NCCL.*(error|failure)|RayActorError|ActorDiedError|raylet.*(died|lost)|Connection reset|Broken pipe'
if [[ ! -f "${TRAIN_LOG}" ]] || grep -Eiq "${FATAL_PATTERN}" "${TRAIN_LOG}" || \
   ! grep -Eiq "${TRANSIENT_PATTERN}" "${TRAIN_LOG}"; then
  write_state failed_no_retry
  printf '%s first_exit=%s recovery_attempted=false reason=fatal_or_non_transient\n' \
    "$(date --iso-8601=seconds)" "${FIRST_EXIT}" >>"${LOG_PATH}"
  exit "${FIRST_EXIT}"
fi

RESUME_DIR="$(latest_complete_checkpoint)" || {
  write_state failed_no_checkpoint
  printf '%s first_exit=%s recovery_attempted=false reason=no_complete_checkpoint\n' \
    "$(date --iso-8601=seconds)" "${FIRST_EXIT}" >>"${LOG_PATH}"
  exit "${FIRST_EXIT}"
}
RESUME_STEP="${RESUME_DIR##*_}"
"${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/toolkits/checkpoint/audit_tabero_firm_sft_checkpoint.py" \
  --checkpoint "${RESUME_DIR}/actor/model_state_dict/trainable_weights.pt" \
  --expected-step "${RESUME_STEP}" \
  --output "${RUN_DIR}/resume_checkpoint_audit.json" >>"${LOG_PATH}" 2>&1 || {
    write_state failed_checkpoint_audit
    exit "${FIRST_EXIT}"
  }

write_state recovering_once
printf '%s recovery_attempted=true resume_dir=%s\n' \
  "$(date --iso-8601=seconds)" "${RESUME_DIR}" >>"${LOG_PATH}"

TABERO_FIRM_2GPU_RESUME_DIR="${RESUME_DIR}" \
TABERO_FIRM_TRAIN_LOG_NAME="train_attempt2.log" \
  "${SCRIPT_DIR}/run_tabero_firm_sft_2gpu_selective_siglip_20k.sh" resume \
  >>"${LOG_PATH}" 2>&1
SECOND_EXIT=$?
printf '%s\n' "${SECOND_EXIT}" >"${RUN_DIR}/formal_exit_code_attempt2"
if ((SECOND_EXIT == 0)); then
  write_state completed_after_one_recovery
else
  write_state failed_after_one_recovery
fi
printf '%s supervisor_exit=%s recovery_attempted=true\n' \
  "$(date --iso-8601=seconds)" "${SECOND_EXIT}" >>"${LOG_PATH}"
exit "${SECOND_EXIT}"
