#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <1|5> <smoke|formal> [--resume-dir PATH] [--dry-run]" >&2
}

TASK_ID="${1:-}"
[[ "${TASK_ID}" == "1" || "${TASK_ID}" == "5" ]] || {
  usage
  exit 2
}
shift

readonly PROFILE_DIR="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/alltask_damage_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k"
case "${TASK_ID}" in
  1)
    readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task1_no_adverb_uniform_physics_2gpu_100step"
    readonly HDF5_PATH="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5/libero_object_task1_pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5"
    ;;
  5)
    readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task5_no_adverb_uniform_physics_2gpu_100step"
    readonly HDF5_PATH="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5/libero_object_task5_pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5"
    ;;
esac

readonly LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_LAUNCHER="${LAUNCHER_DIR}/run_tabero_task0_no_adverb_mass_friction_action_expert.sh"
[[ -x "${BASE_LAUNCHER}" ]] || {
  echo "error: base launcher is not executable: ${BASE_LAUNCHER}" >&2
  exit 1
}

export TABERO_PIRL_CONFIG_NAME="${CONFIG_NAME}"
export TABERO_PIRL_PROFILE_DIR="${PROFILE_DIR}"
export TABERO_PIRL_HDF5_PATH="${HDF5_PATH}"
export TABERO_PIRL_SMOKE_OUTPUT_PREFIX="tabero_task${TASK_ID}_no_adverb_uniform_physics_action_expert_lora_2gpu_capacity_smoke"
export TABERO_PIRL_FORMAL_OUTPUT_PREFIX="tabero_task${TASK_ID}_no_adverb_uniform_physics_action_expert_lora_2gpu_100step"
if [[ -n "${TABERO_PIRL_RUN_ID:-}" ]]; then
  export TABERO_TASK0_NO_ADVERB_RUN_ID="${TABERO_PIRL_RUN_ID}"
fi

exec "${BASE_LAUNCHER}" "$@"
