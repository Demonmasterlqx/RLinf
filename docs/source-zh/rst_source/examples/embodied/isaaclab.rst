基于 IsaacLab 的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
   :align: center
   :width: 90%

   IsaacLab（图片来源：`IsaacLab <https://developer.nvidia.com/isaac/lab>`__）。

`IsaacLab <https://developer.nvidia.com/isaac/lab>`__ 是 NVIDIA 的 GPU 加速机器人学习仿真器。
你将使用 RLinf 在自定义 Franka 方块堆叠任务上，通过 PPO 微调 GR00T N1.5 或 OpenPI π₀.₅。

概览
----------------------------------------

先使用 SFT 检查点，再通过 PPO 在 IsaacLab Franka stack-cube 任务上微调 VLA。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GR00T N1.5 · π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      Franka stack-cube

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPUs

| **你将完成：** 安装 → 下载 Isaac Sim + SFT 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Isaac Sim · SFT 检查点（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0``
     - 将红色方块堆到蓝色方块上，再将绿色方块堆到红色方块上。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 第三人称相机和腕部相机 RGB（默认 256×256），以及机器人本体状态。
   * - 动作
     - 7 维连续动作：3D 位置（x, y, z）+ 3D 旋转（roll, pitch, yaw）+ 夹爪。
   * - 奖励
     - 稀疏 0/1 成功奖励。
   * - 提示词
     - ``Stack the red block on the blue block, then stack the green block on the red block.``

安装
----------------------------------------

.. include:: _setup_common.rst

**Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-isaaclab

   # 国内用户可使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-isaaclab

在镜像中切换到对应的虚拟环境：

.. code:: bash

   # GR00T N1.5
   source switch_env gr00t

   # OpenPI π₀.₅
   # source switch_env openpi

**自定义环境**

为你要运行的模型安装环境：

.. code:: bash

   # 国内用户可添加 --use-mirror。

   # GR00T N1.5
   bash requirements/install.sh embodied --model gr00t --env isaaclab
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env isaaclab
   # source .venv/bin/activate

下载 Isaac Sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 Isaac Sim 5.1.0 并初始化其 shell 环境：

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   每次在新终端中启动 IsaacLab 前，都需要运行 ``source ./setup_conda_env.sh``。

下载模型
----------------------------------------

下载你要微调的模型检查点。

**GR00T N1.5**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Stack-cube

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir RLinf-Gr00t-SFT-Stack-cube

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/YifWRobotics/RLinf-pi05-SFT-Stack-cube

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download YifWRobotics/RLinf-pi05-SFT-Stack-cube --local-dir RLinf-pi05-SFT-Stack-cube

.. include:: _model_path.rst

这些 SFT 检查点来自 IsaacLab stack-cube 任务的人类演示数据。
数据集已发布在 |huggingface|
`IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__。

运行
----------------------------------------

选择一个配置并启动训练：

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - 模型
     - 配置
     - 命令后缀
   * - GR00T N1.5
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``
     - ``isaaclab_franka_stack_cube_ppo_gr00t``
   * - OpenPI π₀.₅
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml``
     - ``isaaclab_franka_stack_cube_ppo_openpi_pi05``

.. code:: bash

   # GR00T N1.5
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

   # OpenPI π₀.₅
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05

这条命令会：

1. 使用选定的 Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 IsaacLab env 组件创建 Ray worker。
3. 运行 PPO rollout，计算稀疏任务奖励，并更新 VLA 策略。

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>`，
通过配置回退机制复用相同后缀：``isaaclab_franka_stack_cube_ppo_gr00t`` 和
``isaaclab_franka_stack_cube_ppo_openpi_pi05``。

.. note::

   GR00T 默认配置会分离 env、rollout 和 actor placement。OpenPI 默认配置使用
   ``actor,env,rollout: all`` 共置。请根据 GPU 显存预算调整
   ``cluster.component_placement``、``rollout.pipeline_stage_num`` 和
   ``actor.enable_offload``。

.. note::

   如需添加自定义 IsaacLab 任务，请在 ``rlinf/envs/isaaclab/tasks/`` 下实现任务，
   在 ``rlinf/envs/isaaclab/__init__.py`` 中注册任务，然后在
   ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` 等环境配置中，将
   ``init_params.id`` 指向新的 task id。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。完整指标说明见
:doc:`训练指标 <../../reference/metrics>`。

如需保存 rollout 视频，请在环境配置中启用 video：

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

如需启用 W&B 或 SwanLab，请添加 logger backend：

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - 模型阶段
     - 成功率
   * - GR00T N1.5 基础模型（无 SFT）
     - 0.000
   * - GR00T N1.5 SFT 模型
     - 0.654
   * - GR00T N1.5 RL 微调模型（SFT + RL）
     - 0.897
   * - OpenPI π₀.₅ SFT 模型
     - 0.859
   * - OpenPI π₀.₅ RL 微调模型（SFT + RL）
     - 0.953

致谢
----------------------------------------

感谢 `许明辉 <https://github.com/smallcracker>`__ 和
`杨楠 <https://github.com/AquaSage18>`__ 对 GR00T N1.5 示例的贡献与支持，也感谢
`Yifan Wu <https://github.com/YifWRobotics>`__ 对 OpenPI π₀.₅ 示例的贡献与支持。


Tabero RealWorld 首次轨迹指标
----------------------------

RealWorld 的 ``auto_reset=false`` rollout 每个环境只使用第一次终止之前的动作训练。
环境在 chunk 边界仍会局部 reset，因此指标在首次成功或超时时复制保存；
后续局部 reset 和替代轨迹不会覆盖这些记录。下一次整体 reset 开始新一轮计数。

``env/success_once`` 为首次轨迹的成功比例，``env/return`` 为累计奖励均值，
``env/reward`` 为各轨迹累计奖励除以实际长度后的均值。
``env/num_trajectories`` 按完成的 episode 记录计数，与 chunk 诊断数量分开。
记录覆盖成功动作所在的 chunk 中间和边界，并在完整 episode horizon 到达时检查遗漏。
这些记录不改变逐步 reward、终止标记、GAE 或 PPO mask。

可设置 ``env.train.init_params.episode_records_dir`` 保存逐轨迹 JSONL。
每个文件按 seed 和进程 ID 区分，记录包含 ``rollout_index``、``env_index``、
成功、奖励、长度、终止原因及可用的力指标。无有效力样本的非有限值写为 JSON ``null``。
视频可通过相同 seed 与 ``env_index`` 对应网格位置；首次轨迹结束后的画面属于替代轨迹。


Tabero 按任务选择力奖励
----------------------

普通 Tabero 与 RealWorld Tabero 均支持 ``success.force_bonus``（原倒数形式）和
``success.normalize_effort_reward``。每个任务最多启用一种，也可全部关闭；
同一次多任务训练可以混用两种形式。配置按公共 ``init_params`` 与任务覆盖深度合并，
因此切换奖励时必须显式关闭继承而来的另一种形式。其他成功判据和观测、动作、重置接口仍须一致。

新形式要求显式配置有限的 ``0 <= min_effort < mid_effort``，自动计算
``max_effort = mid_effort + (mid_effort - min_effort)``，不接受手动配置 ``max_effort``。

.. math::

   r_{\mathrm{effort}} = \operatorname{clip}\left(
   \frac{2(\mathrm{mid\_effort}-\mathrm{effort})}
   {\mathrm{max\_effort}-\mathrm{min\_effort}}, -1, 1\right)

min/mid/max 分别对应 +1/0/-1，超出区间时裁剪。effort 复用原奖励的有效双指接触
挤压力轨迹均值，包括原有抓取门控、多来源聚合、释放过滤和局部重置行为。
``min_valid_samples`` 默认 1，``contact_epsilon`` 默认 0.0001；不足有效样本时力分量为零。
只有有效成功时才发放 ``terminal_reward + 力分量``，失败和超时为零。
``[-1, 1]`` 限制的是力分量，后续仍应用环境 ``reward_coef`` 和已有条件倍率；
奖励项的 ``/ step_dt`` 与 IsaacLab RewardManager 的 ``* dt`` 抵消。

成功率使用重置前的成功终止标志，不从奖励正负推断。
例如 ``terminal_reward: 1.0`` 且 effort 达到 max 时，总奖励为零，但仍计为成功。
现有 ``force_bonus`` 诊断字段保留兼容，在新模式下表示选中的有符号力分量；
它是在成功门控及奖励缩放前的诊断值，不等于失败 episode 获得了力奖励。
总回报和成功指标继续通过现有 TensorBoard、W&B 通道记录。

以下为 Task820 多任务训练配置片段。数值仅为结构示例，不是物体力阈值的标定结果；
保留原训练配置的其他设置及任务描述。Vitasoy、Coca-Cola 使用不同范围，cookie 使用旧奖励。
现有 YAML 不会自动切换奖励形式。

.. code-block:: yaml

   runner:
     logger:
       logger_backends: [tensorboard, wandb]
   env:
     train:
       init_params:
         success:
           required_consecutive_steps: 8
           terminal_reward: 1.0
           force_bonus:
             enabled: false
           normalize_effort_reward:
             enabled: false
             min_valid_samples: 4
             contact_epsilon: 1.0
       multi_task:
         tasks:
           - name: vitasoy
             init_params:
               target_object: target_object_1
               task_description: pick up the Vitasoy and put it into the basket
               success:
                 normalize_effort_reward:
                   enabled: true
                   min_effort: 10.0
                   mid_effort: 20.0  # max_effort = 30
           - name: coca_cola
             init_params:
               target_object: target_object_2
               task_description: Pick up the Coca-Cola and put it into the basket
               success:
                 normalize_effort_reward:
                   enabled: true
                   min_effort: 15.0
                   mid_effort: 30.0  # max_effort = 45
           - name: cookie
             init_params:
               target_object: target_object_3
               task_description: pick up the cookie and put it into the basket
               success:
                 force_bonus:
                   enabled: true
                   coefficient: 10.0
                   epsilon: 1.0
                   max_bonus: 1.0
                   min_valid_samples: 4
                   contact_epsilon: 1.0
     eval:
       multi_task: ${env.train.multi_task}
       init_params:
         success: ${env.train.init_params.success}

单任务训练直接在 ``env.train.init_params.success`` 中设置同样的奖励块，无需任务列表。
改变力奖励属于训练目标调整，配置示例不修改模型、GPU、批量、训练步数或 ``resume_dir``。
