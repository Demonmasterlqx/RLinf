#!/usr/bin/env bash
set -euo pipefail

readonly LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_LAUNCHER="${LAUNCHER_DIR}/run_tabero_task0_no_adverb_mass_friction_action_expert.sh"
readonly CONFIG_NAME="isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_uniform_physics_no_damage_force_reward_coef1_maxbonus10_2gpu_100step"
readonly PROFILE_DIR="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/alltask_fixed_damage_1000000_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k"
readonly HDF5_PATH="/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5/libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"

[[ -x "${BASE_LAUNCHER}" ]] || {
  echo "error: base launcher is not executable: ${BASE_LAUNCHER}" >&2
  exit 1
}

export TABERO_PIRL_CONFIG_NAME="${CONFIG_NAME}"
export TABERO_PIRL_PROFILE_DIR="${PROFILE_DIR}"
export TABERO_PIRL_HDF5_PATH="${HDF5_PATH}"
export TABERO_PIRL_SMOKE_OUTPUT_PREFIX="tabero_task0_no_adverb_uniform_physics_no_damage_force_reward_coef1_maxbonus10_action_expert_lora_2gpu_capacity_smoke"
export TABERO_PIRL_FORMAL_OUTPUT_PREFIX="tabero_task0_no_adverb_uniform_physics_no_damage_force_reward_coef1_maxbonus10_action_expert_lora_2gpu_100step"
if [[ -n "${TABERO_TASK0_FORCE_RUN_ID:-}" ]]; then
  export TABERO_TASK0_NO_ADVERB_RUN_ID="${TABERO_TASK0_FORCE_RUN_ID}"
fi

exec "${BASE_LAUNCHER}" "$@"
