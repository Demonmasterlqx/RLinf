# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


class _NoShardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Parameter(torch.tensor([1.0]))
        self.critic = nn.Parameter(torch.tensor([1.0]))
        self._all_handles = [SimpleNamespace(uses_sharded_strategy=False)]


def test_fsdp_grad_clip_accepts_optimizer_parameter_subset():
    strategy = FSDPStrategy.__new__(FSDPStrategy)
    strategy.cfg = OmegaConf.create({"optim": {"clip_grad": 99.0}})
    model = _NoShardModel()
    model.actor.grad = torch.tensor([3.0])
    model.critic.grad = torch.tensor([4.0])

    grad_norm = strategy.clip_grad_norm_(
        model,
        max_norm=1.0,
        parameters=(model.actor,),
    )

    assert grad_norm == pytest.approx(3.0)
    torch.testing.assert_close(model.actor.grad, torch.tensor([1.0]))
    torch.testing.assert_close(model.critic.grad, torch.tensor([4.0]))
