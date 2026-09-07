DSRL：基于扩散模型的潜在空间强化学习
======================================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dsrl.png
   :align: center
   :width: 70%

   DSRL 在噪声空间中操控冻结的扩散策略。

使用 **DSRL（Diffusion Steering via Reinforcement Learning）** 对预训练的 **Pi0 扩散策略** 做强化学习微调。DSRL 在潜在噪声空间中训练轻量 SAC 智能体来引导冻结的 Pi0 策略，仅需约 500K 可训练参数。

相关论文： `Steering Your Diffusion Policy with Latent Space Reinforcement Learning <https://arxiv.org/abs/2506.15799>`_ （CoRL 2025, Wagenmaker et al.）

参考实现： `dsrl_pi0 <https://github.com/nakamotoo/dsrl_pi0>`_

核心思路：

1. **轻量级 SAC 智能体**：一个小型 SAC 智能体（约 500K 参数），配备紧凑的 CNN/MLP 编码器，处理观测并在潜在空间中生成噪声。
2. **噪声注入**：生成的噪声作为初始噪声输入到 Pi0 的扩散去噪器中，替代随机采样。
3. **冻结 VLM 主干**：预训练的 Pi0 VLM 和扩散专家模块保持冻结，保留泛化能力。
4. **噪声空间中的 SAC 训练**：SAC 智能体在噪声空间上使用环境奖励进行训练，采用 10 个 Q-head 集成的 Critic 实现稳定的价值估计。

概览
----------------------------------------

用一个轻量 SAC 智能体（约 50 万参数）在潜在噪声空间中操控冻结的 π₀ 扩散策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      DSRL (SAC)

   .. grid-item-card:: 模型
      :text-align: center

      π₀（冻结）

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      LIBERO-Spatial

   .. grid-item-card:: 训练
      :text-align: center

      ~500K 可训练参数

| **你将完成：** 安装（同 π₀）→ 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 预训练的 π₀ 检查点（见 :doc:`pi0`）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 环境
     - LIBERO-Spatial——强调空间推理的桌面操作。
   * - 观测
     - 8 维本体感知 + RGB 图像。
   * - 动作
     - 由 π₀ 冻结的扩散去噪器生成的连续动作，由 SAC 噪声操控。

DSRL 工作原理
----------------------------------------

**DSRL 流程**

1. **观测编码**：轻量级 CNN（64×64 → 64维）和状态编码器（8维 → 64维）处理观测数据。

2. **噪声生成**： ``GaussianPolicy`` （SquashedNormal 分布）为每个动作步生成 32 维噪声动作。

3. **扩散去噪**：噪声作为初始噪声注入 Pi0 的 ``sample_actions()``，冻结的扩散去噪器将噪声转换为真实动作。

4. **SAC 训练**：标准 SAC 配合自动熵调节训练噪声生成器：

   - **Actor**： ``GaussianPolicy`` ，3 层 MLP（128维隐藏层）
   - **Critic**： ``CompactMultiQHead`` — 10 个 Q 网络集成（共约 500K 参数）
   - **目标网络**：Float32 EMA 影子缓冲区，解决 bfloat16 精度问题

安装
----------------------------------------

DSRL 使用与 Pi0 相同的环境和模型依赖。请参考 :doc:`pi0` 获取完整的安装指南，包括 Docker 镜像配置、依赖安装和模型下载。

运行
----------------------------------------

**1. 配置文件**

- **DSRL 训练**： ``examples/embodiment/config/libero_spatial_dsrl_openpi.yaml``

**2. 关键参数配置**

**2.1 DSRL 模型参数**

.. code:: yaml

   actor:
     model:
       openpi:
         use_dsrl: True              # 启用 DSRL 模式
         dsrl_state_dim: 8           # 机器人本体感知维度
         dsrl_action_noise_dim: 32   # 每步噪声动作维度
         dsrl_num_q_heads: 10        # 集成 Critic 中的 Q-head 数量
         dsrl_image_latent_dim: 64   # 图像编码器输出维度
         dsrl_state_latent_dim: 64   # 状态编码器输出维度
         dsrl_hidden_dims: [128, 128, 128]  # MLP 隐藏层维度

**2.2 算法参数**

.. code:: yaml

   algorithm:
     adv_type: embodied_sac
     loss_type: embodied_sac
     gamma: 0.999             # 折扣因子
     tau: 0.005               # 目标网络软更新系数
     update_epoch: 200        # 每次交互后的训练步数
     train_actor_steps: 10    # Actor 训练延迟步数（先训练 Critic）
     entropy_tuning:
       alpha_type: softplus
       initial_alpha: 1.0
       target_entropy: -16
       optim:
         lr: 3.0e-4

**2.3 环境参数**

.. code:: yaml

   env:
     train:
       total_num_envs: 16
       use_step_penalty: True  # 使用 -1/0 奖励风格（步惩罚 + 终止奖励）
       max_episode_steps: 240
     eval:
       total_num_envs: 500
       use_step_penalty: True

**2.4 Tabero 触觉 Replay 契约**

Tabero 触觉 DSRL 使用每个 rank 独立的有界 transition ring。观测进入 replay
前即完成投影：主相机与腕部相机分别缩放到 64×64，并按固定顺序保存为一个
``[2, 3, 64, 64]`` bfloat16 张量。state、触觉 marker motion 和单个 32 维
SAC latent action 同样保存为 bfloat16；十个 primitive reward 保持 float32，
done 字段保持布尔类型。冻结 Pi0 的 diffusion chain、token、model action 以及
256×256 原图均不会写入 replay。

.. code:: yaml

   algorithm:
     dsrl_replay_semantics: main_wrist_bf16_64_transition_ring_v1
     replay_buffer:
       backend: compact_dsrl_ring
       capacity_transitions: 100000
       checkpoint_shard_transitions: 4096
       max_resident_gib: 12.0

ring checkpoint 按 shard 串行写入。上述语义属于 resume/export 的前置校验
契约；旧 trajectory replay checkpoint 会在恢复 model 或 optimizer 前被拒绝。

**2.5 Task820 PI0.5 TacField 配置**

Task820 的 marker motion 为 ``[B, 9, 440, 2]``，需要设置
``actor.model.openpi.dsrl_tactile_input_dim: 880``。该参数控制独立
actor/critic 触觉编码器的每帧输入宽度，默认值 396 保留原有 198-marker
模型行为。冻结的 PI0.5 TacField 前缀仍由原模型配置控制。
此配置使用通用 trajectory replay 和完整 trainable 参数同步；上节的
compact replay 及选择性同步契约仍固定为 198 markers，不能直接用于此配置。

提供两个本地实验配置，均从 Task820 firm_mixed SFT 的 step-18000 export
初始化，并使用 TensorBoard 和 W&B：

- ``isaaclab_pi05_dsrl_task820_tacfield_8gpu_smoke``：8 环境，2 次同步
  runner 调用，每次 1 次 SAC 更新；通过 ``train_embodied_agent.py`` 启动。
- ``isaaclab_pi05_dsrl_task820_tacfield_8gpu_benchmark``：96 环境，env 位于
  GPU 0–5、rollout 位于 GPU 6、actor 位于 GPU 7；micro batch 128、global
  batch 512、梯度累积 4，replay preload 开启、预取 2 批。

benchmark 必须通过 ``train_async.py`` 启动：异步 actor 更新与环境采样重叠，
``rollout.pipeline_stage_num: 2`` 开启两级采样流水线。
``runner.use_training_pipeline`` 是 PPO/PIRL 专用开关，SAC 必须保持 false。
两套配置均关闭 gradient checkpointing、checkpoint 保存、导出、视频与评测。
配置中的模型、资产和日志路径对应本地工作区；在其他机器上使用前需要调整。

当前 embodied runner 固定每个 epoch 为一次训练调用；若 ``max_steps >= 0``，
最终调用上限是 ``min(max_epochs, max_steps)``，设为 -1 则只由 max_epochs
限制。benchmark 两者均为 96，``algorithm.update_epoch: 8``，共执行
768 次 SAC 更新。这不是 96 次完整采样，也不是遍历数据集 96 次。
每 2 次 runner 调用请求权重同步，即每 16 次 SAC 更新。

在 8 张 RTX 5090（每卡 32607 MiB）上，该 benchmark 完成了 768 次更新、
两轮共 5760 个新 transition 入库，正常退出。丢弃前 4 次 runner 调用后的
replay 训练吞吐为 420.8 samples/s，新数据采样约 5.35–5.50 transitions/s
（每个 transition 执行 10 个控制步）。actor 显存抽样峰值 30846 MiB。
这是短程吞吐验证，不代表策略质量或 8 卡持续满载。
结果见 `W&B 确认实验 <https://wandb.ai/183842220-hkust/tabero-rlinf/runs/529eb1dc58>`_。

从 RLinf 根目录、配置好 W&B 认证后启动：

.. code-block:: bash

   # 进入持久会话后，在其中执行下面的环境设置及训练命令。
   tmux new-session -s task820_dsrl_benchmark
   source .venv/bin/activate
   export REPO_PATH="$PWD"
   export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
   # 此实验主机验证过的 NCCL 通信设置。
   export NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo
   # 每个新实验使用独立 run ID；相同实验重试时保持 ID 不变。
   export WANDB_RUN_ID=task820_dsrl_benchmark_run1 WANDB_RESUME=allow
   python -u examples/embodiment/train_async.py \
     --config-name isaaclab_pi05_dsrl_task820_tacfield_8gpu_benchmark

**3. 启动命令**

::

   bash examples/embodiment/run_embodiment.sh libero_spatial_dsrl_openpi

可视化与结果
----------------------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs

**2. 关键监控指标**

指标含义见 :doc:`训练指标 <../../reference/metrics>`。DSRL 相关指标：

- **环境指标**：

  - ``env/episode_len``：该回合实际经历的环境步数
  - ``env/return``：回合总回报
  - ``env/reward``：环境的 step-level 奖励
  - ``env/success_once``：回合中至少成功一次标志（0 或 1）

- **训练指标**：

  - ``train/sac/critic_loss``：Q 函数集成的损失
  - ``train/critic/grad_norm``：Q 函数的梯度范数

  - ``train/sac/actor_loss``：策略损失（噪声空间中的 GaussianPolicy）
  - ``train/actor/entropy``：策略熵
  - ``train/actor/grad_norm``：策略的梯度范数

  - ``train/sac/alpha_loss``：温度参数的损失
  - ``train/sac/alpha``：温度参数的值

  - ``train/replay_buffer/size``：当前重放缓冲区的大小
  - ``train/replay_buffer/utilization``：重放缓冲区的利用率
