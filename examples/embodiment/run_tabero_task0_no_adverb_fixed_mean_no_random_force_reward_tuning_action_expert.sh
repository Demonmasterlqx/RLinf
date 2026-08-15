#!/usr/bin/env bash
set -euo pipefail

readonly LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_LAUNCHER="${LAUNCHER_DIR}/run_tabero_task0_no_adverb_mass_friction_action_expert.sh"
readonly PROFILE_DIR="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/task0_fixed_mean_mass_friction_no_control_override_from_rlinf_sft_20k"
readonly HDF5_PATH="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5/libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"

usage() {
  echo "Usage: $0 <coef10|coef20|coef30> <smoke|formal> [--resume-dir PATH] [--dry-run]" >&2
}

candidate="${1:-}"
case "${candidate}" in
  coef10)
    readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_fixed_mean_no_random_force_reward_coef10_maxbonus1_2gpu_100step"
    readonly GPU_PAIR="0,1"
    ;;
  coef20)
    readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_fixed_mean_no_random_force_reward_coef20_maxbonus1_2gpu_100step"
    readonly GPU_PAIR="2,3"
    ;;
  coef30)
    readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_fixed_mean_no_random_force_reward_coef30_maxbonus1_2gpu_100step"
    readonly GPU_PAIR="4,5"
    ;;
  *)
    usage
    exit 2
    ;;
esac
shift

mode="${1:-}"
[[ "${mode}" == "smoke" || "${mode}" == "formal" ]] || {
  usage
  exit 2
}

[[ -x "${BASE_LAUNCHER}" ]] || {
  echo "error: base launcher is not executable: ${BASE_LAUNCHER}" >&2
  exit 1
}

export TABERO_PIRL_CONFIG_NAME="${CONFIG_NAME}"
export TABERO_PIRL_PROFILE_DIR="${PROFILE_DIR}"
export TABERO_PIRL_HDF5_PATH="${HDF5_PATH}"
export TABERO_PIRL_SMOKE_OUTPUT_PREFIX="tabero_task0_no_adverb_fixed_mean_no_random_force_reward_${candidate}_maxbonus1_action_expert_lora_2gpu_capacity_smoke"
export TABERO_PIRL_FORMAL_OUTPUT_PREFIX="tabero_task0_no_adverb_fixed_mean_no_random_force_reward_${candidate}_maxbonus1_action_expert_lora_2gpu_100step"
export CUDA_VISIBLE_DEVICES="${GPU_PAIR}"
export LIBERO_RANDOMIZE_LIGHT=0

if [[ -n "${TABERO_TASK0_FORCE_RUN_ID:-}" ]]; then
  export TABERO_TASK0_NO_ADVERB_RUN_ID="${TABERO_TASK0_FORCE_RUN_ID}"
fi

exec "${BASE_LAUNCHER}" "$@"
