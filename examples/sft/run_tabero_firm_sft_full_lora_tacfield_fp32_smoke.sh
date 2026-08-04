#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
RUN_STAMP="$(date +'%Y%m%d_%H%M%S')"
export TABERO_FIRM_SMOKE_RUN_NAME="tabero_firm_pi0_base_full_lora_fp32_smoke_${RUN_STAMP}"
RUN_DIR="${WORKSPACE_ROOT}/results/${TABERO_FIRM_SMOKE_RUN_NAME}"
export TABERO_FIRM_SMOKE_RUN_DIR="${RUN_DIR}"
NORM_DIR="${RUN_DIR}/norm_stats"
export TABERO_FIRM_NORM_STATS="${NORM_DIR}/norm_stats.json"
export EMBODIED_PATH="${SCRIPT_DIR}"
export REPO_PATH="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${NORM_DIR}"
{
  printf 'run_name=%s\n' "${TABERO_FIRM_SMOKE_RUN_NAME}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'norm_stats=%s\n' "${TABERO_FIRM_NORM_STATS}"
  printf 'NVIDIA_TF32_OVERRIDE=%s\n' "${NVIDIA_TF32_OVERRIDE}"
  printf 'WANDB_MODE=%s\n' "${WANDB_MODE}"
} | tee "${RUN_DIR}/launch_meta.txt"

if [[ -n "${TABERO_FIRM_EXISTING_NORM_STATS:-}" ]]; then
  cp "${TABERO_FIRM_EXISTING_NORM_STATS}" "${TABERO_FIRM_NORM_STATS}"
  printf 'Reused Firm-only norm stats from %s\n' \
    "${TABERO_FIRM_EXISTING_NORM_STATS}" | tee "${RUN_DIR}/norm_stats.log"
else
  "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/toolkits/lerobot/calculate_norm_stats.py" \
    --config-name pi0_lora_tacfield_tabero \
    --repo-id "${WORKSPACE_ROOT}/datas/tabero_firm" \
    --output-dir "${NORM_DIR}" \
    2>&1 | tee "${RUN_DIR}/norm_stats.log"
fi

"${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/toolkits/lerobot/audit_tabero_firm_sft_batch.py" \
  --dataset-path "${WORKSPACE_ROOT}/datas/tabero_firm" \
  --norm-stats-path "${TABERO_FIRM_NORM_STATS}" \
  --output "${RUN_DIR}/batch_audit.json" \
  2>&1 | tee "${RUN_DIR}/batch_audit.log"

"${REPO_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/train_vla_sft.py" \
  --config-path "${SCRIPT_DIR}/config" \
  --config-name tabero_firm_sft_full_lora_tacfield_fp32_smoke \
  2>&1 | tee "${RUN_DIR}/train.log"

printf '%s\n' "${RUN_DIR}"
