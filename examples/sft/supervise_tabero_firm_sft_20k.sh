#!/usr/bin/env bash
set -uo pipefail

RUN_STAMP="${1:?run stamp is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
RUN_NAME="tabero_firm_pi0_base_rlinf_dual_rank_bf16_20k_${RUN_STAMP}"
RUN_DIR="${WORKSPACE_ROOT}/results/${RUN_NAME}"
SUPERVISOR_LOG="${WORKSPACE_ROOT}/results/${RUN_NAME}_supervisor.log"
SUPERVISOR_STATE="${WORKSPACE_ROOT}/results/${RUN_NAME}_supervisor.state"

write_state() {
  printf 'status=%s\nrun_dir=%s\nupdated_at=%s\n' \
    "$1" "${RUN_DIR}" "$(date --iso-8601=seconds)" >"${SUPERVISOR_STATE}"
}

write_state starting
printf '%s supervisor_start run_dir=%s code_commit=%s\n' \
  "$(date --iso-8601=seconds)" "${RUN_DIR}" \
  "$(git -C "${REPO_ROOT}" rev-parse HEAD)" >>"${SUPERVISOR_LOG}"

TABERO_FIRM_20K_RUN_STAMP="${RUN_STAMP}" \
  "${SCRIPT_DIR}/run_tabero_firm_sft_dual_rank_t2_precision_20k.sh" formal \
  >>"${SUPERVISOR_LOG}" 2>&1
EXIT_CODE=$?

if [[ -d "${RUN_DIR}" ]]; then
  printf '%s\n' "${EXIT_CODE}" >"${RUN_DIR}/formal_exit_code"
fi
if ((EXIT_CODE == 0)) && \
  [[ -f "${RUN_DIR}/exports/step_20000/export_meta.json" ]]; then
  write_state completed
else
  write_state failed
fi
printf '%s supervisor_exit=%s\n' \
  "$(date --iso-8601=seconds)" "${EXIT_CODE}" >>"${SUPERVISOR_LOG}"
exit "${EXIT_CODE}"
