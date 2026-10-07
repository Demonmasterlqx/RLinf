近端策略优化 (PPO)
==================================

1. 引言
---------------

近端策略优化 (PPO) 是目前应用最广泛的强化学习 (RL) 算法之一。  
它包含两个核心部分：  

- Actor（策略模型）：根据当前状态生成动作。  
- Critic（价值模型）：评估所选动作的价值。  

PPO 是一种稳定的策略梯度方法，它在原始策略梯度 (Policy Gradient) 的基础上改进。  
它通过限制策略更新的步长，提高了训练的稳定性和效率。  
此外，PPO 使用广义优势估计 (GAE) 来降低价值估计的方差。  

在 RLHF (Reinforcement Learning from Human Feedback) 的早期阶段，PPO 得到了广泛应用。  
然而，由于需要一个大型 Critic 模型（通常是另一种 LLM），它会带来高昂的计算成本和训练开销。  

更多细节请参考原始论文  
`PPO <https://arxiv.org/abs/1707.06347>`_ 以及它在 RLHF 中的应用  
`InstructGPT <https://arxiv.org/abs/2203.02155>`_。


2. 目标函数
----------------------

设策略为 :math:`\pi_\theta`。  
对于包含问答对 :math:`(q,a)` 的数据集 :math:`\mathcal{D}`，  
PPO 的目标函数定义如下：  

.. math::

   J_{\mathrm{PPO}}(\theta)
   = \mathbb{E}_{(q,a)\sim\mathcal{D},\, o_{\le t}\sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid q)}
   \Big[
     \min\!\Big(
       r_t(\theta)\,\hat{A}_t,\;
       \mathrm{clip}\,\big(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\,\hat{A}_t
     \Big)
   \Big],

其中：  

- :math:`r_t(\theta) = \dfrac{\pi_\theta(o_t \mid q, o_{<t})}
  {\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})}`  
  表示重要性采样比率，用来比较新旧策略。  

- :math:`\varepsilon` 是裁剪范围，一个超参数，用于防止更新过大。  

- :math:`\hat{A}_t` 是时间步 :math:`t` 的优势估计。  

使用广义优势估计 (GAE) 时，优势的计算公式为：  

.. math::

   \hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
   = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l},
   \qquad
   \delta_l = R_l + \gamma V(s_{l+1}) - V(s_l),
   \quad 0 \le \gamma, \lambda \le 1.

其中：  

- :math:`\gamma` （折扣因子）和 :math:`\lambda` （GAE 参数）是超参数。  
- :math:`V(s)` 是 Critic 模型给出的价值估计。  


3. 配置
-----------------

我们的框架支持在 LLM 推理任务和具身任务中使用 PPO。

3.1. 具身任务
~~~~~~~~~~~~~~~~~

下面首先给出一个具身任务的示例配置：

.. code-block:: yaml

   algorithm:

      # 核心 PPO 设置（建议不要修改）
      normalize_advantages: True
      group_size: 1
      adv_type: gae
      loss_type: actor_critic
      loss_agg_func: "token-mean"

      # 算法参数（通常需要调优）

      rollout_micro_batch_size: 256
      logprob_forward_micro_batch_size: 16  # 较大的 batch_size 可以提高稳定性。
                                            # 请根据算力和模型大小调整。

      entropy_bonus: 0          # 可选：鼓励探索
      clip_ratio_high: 0.2      # PPO 裁剪参数 (epsilon)
      clip_ratio_low: 0.2       # 应与 clip_ratio_high 保持一致
      value_clip: 0.2           # 稳定价值函数更新

      gamma: 0.99               # GAE 的折扣因子
      gae_lambda: 0.95          # GAE 的 Lambda 参数

      huber_delta: 10.0         # 价值训练中 Huber 损失的 Delta 参数

3.2. LLM 推理任务
~~~~~~~~~~~~~~~~~
LLM 推理任务的配置与具身任务的配置有相似之处。

.. code-block:: yaml

    algorithm:
       # 组数应为1
       group_size: 1

       # 优势函数
       adv_type: gae
       gamma: 1
       gae_lambda: 1
       normalize_advantages: True

       # actor loss 的类型。此处与具身任务不同，
       # 因为 actor 与 critic 不在同一个模型中，不能进行联合 backward
       loss_type: actor
       loss_agg_func: "token-mean"

       # 用于actor loss
       clip_ratio_c: 3.0
       clip_ratio_low: 0.2
       clip_ratio_high: 0.2

       # 用于critic loss
       value_clip: 0.2           # 稳定价值函数更新

另外，在 LLM 推理任务中，我们使用独立的 critic 模型，而不是让 actor 模型和 critic 模型共享 backbone；这意味着配置中除了像 GRPO 一样有 actor 组件，还需要添加一个 critic 组件，并为其设置相应的 placement:

.. code-block:: yaml

    cluster:
      num_nodes: 1
      component_placement:
        # 注意相比于 GRPO 新增了一个 critic 组件
        actor,critic,rollout,reward: all
                
    actor:
      group_name: "ActorGroup"
      training_backend: megatron
      ...
    
    critic:
      use_critic_model: true # 该参数用于指出 critic 是一个完整的模型，而非只有一个 value head
      group_name: "CriticGroup"
      training_backend: megatron
      # critic 的其余部分配置和 GRPO 配置中的 actor 配置非常类似
      ...
       

4. 注意事项
-----------

- 使用奖励归一化来稳定训练。  
- 监控 KL 散度以检测策略是否更新过度。  
- 对于大型 LLM，增加 batch size 可以减少方差。  

可选的多任务成功率加权
----------------------------

``algorithm.multi_task.enabled`` 默认关闭。开启后，使用同步入口
``train_embodied_agent.py``，在 ``env.train.multi_task.tasks`` 中用唯一
``name`` 和 ``init_params`` 覆盖声明任务。每个逻辑环境实例（worker × rollout
stage）固定分配一个任务，按任务列表循环分配，实例数必须覆盖全部任务。
配置和代码均位于 RLinf，任务资产只读，不修改 Tabero_X 或 IsaacLab。
评估使用相同顺序的任务名，可配置 ``env.eval.multi_task: ${env.train.multi_task}``。

首版支持已有 Libero/Tabero 和 Task820 的 terminal-safe OpenPI 适配、FSDP、GAE
和 chunk-level actor-critic PPO。同次训练要求模型、观测、动作、触觉、奖励及重置
接口兼容，保留原有单任务契约；不会自动将 Libero pi0 配置变成 pi05 配置。
开启功能时拒绝异步 PPO、training pipeline、外部 reward model 和 SFT co-training。

统计口径与 terminal-safe 训练一致：每个 rollout epoch 中，每个环境的首个完整
episode 只统计一次；终止后自动重置产生且已被训练 mask 排除的 episode 不计入。
跨 worker 先汇总成功数及完成数（含超时），再计算成功率；统计不受 reward filter
影响。采样窗口必须覆盖 episode horizon，缺少首个完整 episode 时在更新前报错。
评估只报告计数，不更新训练 EMA。

EMA 默认保留 0.9 的历史估计，首次直接采用观测成功率；没有新记录则保留旧值。
全部任务首次被观测前权重为 1，之后使用：

.. math::

   w_k=0.5+1.5\sigma(10(\bar p-p_k)),\qquad \bar p=K^{-1}\sum_k p_k.

参数分别为 ``weight_min``、``weight_max``、``sigmoid_scale``。每轮 rollout 更新
一次权重表，本轮所有 PPO epoch 复用。权重在跨 rank、包含全部梯度累积的完整
optimizer batch 内按有效 chunk 的平均值归一化，micro-batch 不单独归一化。
策略 loss、价值 loss 和 entropy 共同加权；GAE、return、模型缓存输入、终止 mask
及轨迹长度校正沿用现有实现。价值系数为 1；新示例的 ``entropy_bonus`` 为 0.005，
旧配置参数不变。

示例为 ``isaaclab_pi05_ppo_multitask_tacfield.yaml``，使用 pi05 TacField no-state。
必须提供审核后的 SFT ``actor.model.model_path`` 和已有的
``env.train.init_params.realworld_config_dir``；GPU 通过 RLinf placement 配置分配。
TensorBoard 和 W&B 同时记录任务计数、成功率、EMA、原始及归一化权重。
actor 的有效样本数为各 optimizer batch 的平均值，区别于 rollout 的 episode 计数。
启动日志记录任务分片分配。

checkpoint 在 ``actor/`` 旁保存原子写入的 ``multi_task_state.json``，包含配置记录、
EMA、初始化标志、累计计数及版本。续训要求状态文件存在、actor 步数匹配，并保持
``algorithm.multi_task.enabled: true``。任务列表及顺序由当前配置决定：同名任务继承
EMA、初始化标志及累计计数；新增任务从零开始；删除任务丢弃历史。改名视为删除后新增，
允许完全替换任务集合。重复名称、损坏状态或关闭控制器仍报错。存在未初始化任务时，
所有任务权重暂为 1。同名任务使用当前环境和奖励设置，历史统计不重算。

``contract`` 仅用于记录配置，恢复时不要求其存在或一致；已有 ``ppo_multitask_v1``
文件无需转换。GPU 分配、环境数量、stage、batch、奖励、评测或训练目标步数的变化
不再被多任务恢复层拦截，当前配置仍需合法，模型及 DCP 仍需兼容。历史 EMA 和计数保留，
后续权重使用当前 EMA decay、权重范围等设置。模型、优化器、scheduler 和 RNG 沿用原有
加载逻辑，不额外覆盖 checkpoint 恢复出的学习率。

多任务续训第一轮 rollout 前总会同步恢复后的 actor 权重，随后遵循配置的同步间隔。
旧单任务续训路径不变；从基础权重开始新实验不视为完整续训。
导出元数据列出全部任务，不将多任务模型描述为单个目标物体。

Tabero 分任务日志仅在 ``algorithm.multi_task.enabled`` 开启时生效，
保留全局曲线，并在 ``env/multi_task/<name>/`` 和
``eval/multi_task/<name>/`` 下记录各任务的 episode reward、return、长度、
力统计及已有 reward audit。``return`` 是 episode 未折扣累计奖励；``reward``
是各 episode 的 return/length 再取平均。跨 worker 先合并样本再计算统计量。
力均值和中位数沿用有效接触轨迹口径；``force_bonus`` 仍是包含失败轨迹的候选
bonus，不代表实际发放奖励。任务级 ``chunk_boundary/*`` 事件数跨 chunk 和
worker 求和。

PPO 增加 ``rollout/multi_task/<name>/rewards``、
``advantages_mean/min/max``、``returns_mean/min/max`` 和 ``values_mean/min/max``。
统计复用已有 loss mask 和动作对齐的任务标签，value 排除 bootstrap 行；
全局和任务级分别提供 ``rewards_count``、``advantages_count``、``returns_count``、
``values_count`` 分母。reward 沿用 chunk mask 下的 primitive 条目数，return、
advantage 和 value 按 chunk 条目计数；advantage 不做任务内重新归一化。
任务没有有效样本时只输出零计数，省略对应均值和极值。TensorBoard 与 W&B
使用相同键名和 step。单任务日志、任务权重、奖励计算和 checkpoint 保持原行为。
