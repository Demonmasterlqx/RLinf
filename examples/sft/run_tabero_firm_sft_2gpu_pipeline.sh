#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP="${1:?run stamp is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"
PREFLIGHT_DIR="${WORKSPACE_ROOT}/results/tabero_firm_rlinf_2gpu_mb16_20k_preflight_${RUN_STAMP}"
RUN_DIR="${WORKSPACE_ROOT}/results/tabero_firm_rlinf_2gpu_mb16_20k_${RUN_STAMP}"
PIPELINE_STATE="${WORKSPACE_ROOT}/results/tabero_firm_rlinf_2gpu_mb16_20k_${RUN_STAMP}_pipeline.state"
PIPELINE_LOG="${WORKSPACE_ROOT}/results/tabero_firm_rlinf_2gpu_mb16_20k_${RUN_STAMP}_pipeline.log"

write_state() {
  local status="$1"
  local tmp_path="${PIPELINE_STATE}.tmp"
  printf 'status=%s\npreflight_dir=%s\nrun_dir=%s\nupdated_at=%s\n' \
    "${status}" "${PREFLIGHT_DIR}" "${RUN_DIR}" \
    "$(date --iso-8601=seconds)" >"${tmp_path}"
  mv "${tmp_path}" "${PIPELINE_STATE}"
}

write_state preflight_running
printf '%s pipeline_start stamp=%s\n' \
  "$(date --iso-8601=seconds)" "${RUN_STAMP}" >>"${PIPELINE_LOG}"
TABERO_FIRM_2GPU_RUN_STAMP="${RUN_STAMP}" \
  "${SCRIPT_DIR}/run_tabero_firm_sft_2gpu_selective_siglip_20k.sh" preflight \
  >>"${PIPELINE_LOG}" 2>&1

write_state training_running
TABERO_FIRM_EXISTING_NORM_STATS="${PREFLIGHT_DIR}/norm_stats/norm_stats.json" \
  "${SCRIPT_DIR}/supervise_tabero_firm_sft_2gpu_20k.sh" "${RUN_STAMP}" \
  >>"${PIPELINE_LOG}" 2>&1

write_state evaluation_running
"${SCRIPT_DIR}/evaluate_tabero_firm_sft_2gpu_all_tasks.sh" \
  "${RUN_DIR}/exports/step_20000" \
  "${RUN_DIR}/evaluation" >>"${PIPELINE_LOG}" 2>&1

write_state completed
printf '%s pipeline_complete run_dir=%s\n' \
  "$(date --iso-8601=seconds)" "${RUN_DIR}" >>"${PIPELINE_LOG}"
printf '%s\n' "${RUN_DIR}"
