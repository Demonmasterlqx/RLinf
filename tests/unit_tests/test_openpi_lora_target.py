# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from rlinf.models import (
    _apply_openpi_lora,
    _get_openpi_lora_target_module,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


class DummyActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))


class DummyPaligemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


class DummyOpenPI(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = DummyPaligemma()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = DummyActionExpert()
        self.action_in_proj = nn.Linear(4, 4)
        self.tactile_prefix_encoder = nn.Linear(4, 4)
        self.value_head = nn.Linear(4, 1)


def _cfg(target: str):
    return OmegaConf.create(
        {
            "is_lora": True,
            "lora_rank": 2,
            "lora_path": None,
            "lora_target": target,
            "freeze_non_lora": True,
        }
    )


def test_get_openpi_lora_target_action_expert():
    model = DummyOpenPI()

    target, assign = _get_openpi_lora_target_module(model, "action_expert")

    assert target is model.paligemma_with_expert.gemma_expert.model
    replacement = DummyActionExpert()
    assign(replacement)
    assert model.paligemma_with_expert.gemma_expert.model is replacement


def test_get_openpi_lora_target_paligemma():
    model = DummyOpenPI()

    target, assign = _get_openpi_lora_target_module(model, "paligemma")

    assert target is model.paligemma_with_expert.paligemma
    replacement = DummyPaligemma()
    assign(replacement)
    assert model.paligemma_with_expert.paligemma is replacement


def test_apply_openpi_action_expert_lora_freezes_non_lora_params():
    model = DummyOpenPI()

    _apply_openpi_lora(model, _cfg("action_expert"))

    trainable = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    assert trainable
    assert any("lora_" in name for name in trainable)
    assert all("paligemma_with_expert.paligemma" not in name for name in trainable)
    assert all("lora_" in name or name.startswith("value_head.") for name in trainable)
    assert all(
        "paligemma_with_expert.gemma_expert.model" in name
        or name.startswith("value_head.")
        for name in trainable
    )


def test_apply_openpi_both_lora_wraps_vlm_and_action_expert_only():
    model = DummyOpenPI()

    _apply_openpi_lora(model, _cfg("both"))

    trainable = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    assert trainable
    assert any(
        name.startswith("paligemma_with_expert.paligemma") and "lora_" in name
        for name in trainable
    )
    assert any(
        name.startswith("paligemma_with_expert.gemma_expert.model") and "lora_" in name
        for name in trainable
    )
    assert all("lora_" in name or name.startswith("value_head.") for name in trainable)
    assert all(
        name.startswith("paligemma_with_expert.paligemma")
        or name.startswith("paligemma_with_expert.gemma_expert.model")
        or name.startswith("value_head.")
        for name in trainable
    )


def test_tabero_peft_config_uses_action_expert_lora_only():
    path = CONFIG_DIR / "isaaclab_pi0_peft_lora_tacfield_tabero.yaml"
    cfg = OmegaConf.load(path)

    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.fsdp_config.save_trainable_model_weights is True
    assert cfg.actor.fsdp_config.checkpoint_format == "none"


def test_tabero_multitask_peft_config_reads_subset_and_disables_force_key():
    path = CONFIG_DIR / "isaaclab_pi0_peft_lora_tacfield_tabero_multitask.yaml"
    cfg = OmegaConf.load(path)

    assert cfg.cluster.component_placement["actor,rollout"] == "all"
    assert cfg.cluster.component_placement.env == 0
    assert cfg.rollout.pipeline_stage_num == 9
    assert cfg.env.train.total_num_envs == 9
    assert cfg.env.train.init_params.tabero_task_subset_path.endswith(
        "Tabero/benchmarks/datasets/tabero/config/tabero_tasks.json"
    )
    assert cfg.env.train.init_params.force_key is None
    assert cfg.env.train.init_params.require_all_tasks_active is True
    assert cfg.actor.model.openpi.tactile_loss_weight == 0.0
    total_rollout_samples = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
    )
    assert total_rollout_samples % cfg.actor.global_batch_size == 0


def test_tabero_multitask_both_lora_config_reads_subset_and_trains_dual_adapters():
    path = CONFIG_DIR / "isaaclab_pi0_peft_lora_both_tacfield_tabero_multitask.yaml"
    cfg = OmegaConf.load(path)

    assert cfg.rollout.pipeline_stage_num == 9
    assert cfg.env.train.total_num_envs == 9
    assert cfg.env.train.init_params.tabero_task_subset_path.endswith(
        "Tabero/benchmarks/datasets/tabero/config/tabero_tasks.json"
    )
    assert cfg.env.train.init_params.force_key is None
    assert cfg.env.train.init_params.require_all_tasks_active is True
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "both"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.openpi.tactile_loss_weight == 0.0
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.actor.fsdp_config.save_trainable_model_weights is True
    total_rollout_samples = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
    )
    assert total_rollout_samples % cfg.actor.global_batch_size == 0


def test_tabero_multitask_both_lora_smoke_config_enables_two_task_video_smoke():
    path = (
        CONFIG_DIR / "isaaclab_pi0_peft_lora_both_tacfield_tabero_multitask_smoke.yaml"
    )
    cfg = OmegaConf.load(path)

    assert cfg.cluster.component_placement.actor == 0
    assert cfg.cluster.component_placement.rollout == 1
    assert cfg.cluster.component_placement.env == 2
    assert cfg.rollout.pipeline_stage_num == 2
    assert cfg.env.train.total_num_envs == 2
    assert len(cfg.env.train.init_params.tasks) == 2
    assert cfg.env.train.video_cfg.save_video is True
    assert cfg.env.train.video_cfg.image_keys == ["main_images", "wrist_images"]
    assert cfg.env.train.video_cfg.image_names == ["agentview", "eye_in_hand"]
    assert cfg.env.train.video_cfg.tactile_heatmap.enabled is True
    assert cfg.env.train.video_cfg.composite_views == [
        "agentview",
        "eye_in_hand",
        "tactile_heatmap",
    ]
    assert cfg.env.train.video_cfg.composite_name == "combined"
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "both"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.actor.model.openpi.train_expert_only is True


def test_tabero_stable_success_config_pairs_conditions_and_trains_expert_only():
    path = (
        CONFIG_DIR
        / "isaaclab_pi0_peft_lora_tacfield_tabero_multitask_stable_success.yaml"
    )
    cfg = OmegaConf.load(path)

    assert cfg.cluster.component_placement["actor,rollout"] == 7
    assert cfg.cluster.component_placement.env == "4:0-2,5:3-5,6:6-8"
    assert cfg.rollout.pipeline_stage_num == 1
    assert cfg.env.train.total_num_envs == 18
    assert cfg.env.train.rollout_epoch == 1
    assert cfg.env.train.max_steps_per_rollout_epoch == 320
    assert cfg.env.train.max_episode_steps == 320
    assert cfg.env.train.init_params.success.required_consecutive_steps == 8
    assert cfg.env.train.init_params.success.terminal_reward == 1.0
    assert cfg.env.train.init_params.prompt_conditions.enabled is True
    assert cfg.env.train.init_params.prompt_conditions.assignment == "paired"
    assert cfg.env.train.init_params.prompt_conditions.firm_adverbs == [
        "firmly",
        "tightly",
    ]
    assert cfg.env.train.init_params.prompt_conditions.gentle_adverbs == [
        "gently",
        "softly",
    ]
    assert cfg.env.train.init_params.force_key is None
    assert cfg.actor.global_batch_size == 18
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.openpi.tactile_loss_weight == 0.0
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.actor.optim.lr == 1.0e-6
    assert cfg.algorithm.update_epoch == 1
    assert cfg.algorithm.kl_beta == 0.02
    assert cfg.runner.save_interval == 1
    total_rollout_samples = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
    )
    assert total_rollout_samples % cfg.actor.global_batch_size == 0


def test_tabero_stable_success_smoke_config_has_two_tasks_and_condition_pairs():
    path = (
        CONFIG_DIR
        / "isaaclab_pi0_peft_lora_tacfield_tabero_multitask_stable_success_smoke.yaml"
    )
    cfg = OmegaConf.load(path)

    assert cfg.cluster.component_placement["actor,rollout"] == 7
    assert cfg.cluster.component_placement.env == "4-5"
    assert cfg.rollout.pipeline_stage_num == 1
    assert cfg.env.train.total_num_envs == 4
    assert cfg.env.train.rollout_epoch == 1
    assert cfg.env.train.max_steps_per_rollout_epoch == 10
    assert cfg.env.train.max_episode_steps == 10
    assert len(cfg.env.train.init_params.tasks) == 2
    assert cfg.env.train.init_params.success.required_consecutive_steps == 8
    assert cfg.env.train.init_params.prompt_conditions.enabled is True
    assert cfg.algorithm.normalize_advantages is False
    assert cfg.actor.global_batch_size == 4
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.env.train.video_cfg.save_video is True
    assert cfg.env.train.video_cfg.image_names == ["agentview", "eye_in_hand"]
    assert cfg.env.train.video_cfg.tactile_heatmap.enabled is True
    assert cfg.env.train.video_cfg.composite_name == "combined"


def test_tabero_full_lora_stable_success_config_balances_conditions():
    base_path = (
        CONFIG_DIR
        / "isaaclab_pi0_peft_lora_tacfield_tabero_multitask_stable_success.yaml"
    )
    path = (
        CONFIG_DIR
        / "isaaclab_pi0_peft_lora_both_tacfield_tabero_multitask_stable_success.yaml"
    )
    cfg = OmegaConf.merge(OmegaConf.load(base_path), OmegaConf.load(path))

    assert cfg.actor.model.lora_target == "both"
    assert cfg.actor.model.lora_rank == 32
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.openpi.tactile_loss_weight == 0.0
    assert cfg.actor.optim.lr == 1.0e-7
    assert cfg.algorithm.update_epoch == 1
    assert cfg.algorithm.kl_beta == 0.05
    assert cfg.env.train.total_num_envs == 18
    assert cfg.env.train.init_params.success.required_consecutive_steps == 8
    assert cfg.env.train.init_params.success.condition_reward_multipliers == {
        "firm": 1.0,
        "gentle": 3.0,
    }
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.actor.fsdp_config.save_trainable_model_weights is True


def test_tabero_full_lora_stable_success_smoke_composes_for_two_tasks(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name=(
                "isaaclab_pi0_peft_lora_both_tacfield_tabero_"
                "multitask_stable_success_smoke"
            )
        )

    assert cfg.actor.model.lora_target == "both"
    assert cfg.actor.optim.lr == 1.0e-7
    assert cfg.algorithm.kl_beta == 0.05
    assert cfg.algorithm.normalize_advantages is False
    assert cfg.env.train.total_num_envs == 4
    assert len(cfg.env.train.init_params.tasks) == 2
    assert cfg.env.train.init_params.success.condition_reward_multipliers == {
        "firm": 1.0,
        "gentle": 3.0,
    }
    assert cfg.env.train.video_cfg.save_video is True
