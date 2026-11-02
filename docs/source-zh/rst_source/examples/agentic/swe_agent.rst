Qwen3-30B-A3B 的 SWE-agent 训练
========================================

使用 `examples/agent/harbor/train.yaml` 训练 Qwen3-30B-A3B-Instruct-2507 作为软件工程 agent 解决 GitHub issue。mini-swe-agent 在 Harbor 管理的 Docker sandbox 中解题，题目自带的测试给最终的代码状态打分，带分数的轨迹再用 GRPO 训练模型。

概述
----------------------------------------

使用本配方在单机 8 卡上以 Megatron-Bridge actor 和 SGLang rollout，在 SWE-Gym Lite 题目上训练 MoE 模型。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3-30B-A3B-Instruct-2507（MoE，激活参数约 3B）

   .. grid-item-card:: 算法
      :text-align: center

      以测试结果为 reward 的多轮 GRPO

   .. grid-item-card:: 工具
      :text-align: center

      运行在 Harbor Docker sandbox 中的 mini-swe-agent

   .. grid-item-card:: 硬件
      :text-align: center

      参考运行使用 8×A100-80G

每个训练步在同一台机器上调动三个组件。Harbor 任务服务为每个 trial 启动一个 sandbox，先运行 mini-swe-agent，再运行题目的测试，最后删除 sandbox。mini-swe-agent 每一轮都向 ``HarborAgentLoopWorker`` 托管的 session gateway 请求模型输出；gateway 把请求转给 SGLang rollout，并原样记录生成的 token ID、logprob 和策略版本。测试结束后，worker 把测试分数和这些 token 拼成训练样本，因此 actor 训练用的就是策略实际生成的 token。agent 在两轮之间追加的文本（例如命令输出）loss mask 为 0。

环境
----------------------------------------

RLinf 环境
~~~~~~~~~~

按 :doc:`RLinf 安装 </rst_source/start/installation>` 搭建 RLinf 环境，使用带 Megatron 和 SGLang 的 agentic 环境。

Harbor 与 mini-swe-agent
~~~~~~~~~~~~~~~~~~~~~~~~

在同一个 Python 环境中安装：

1. ``harbor-miles-v0.20.0`` tag 对应的 Harbor（实测 commit 为 ``917b3c1be7cf6553a7a9fe25de8adb6f8e639287``），负责管理 Docker sandbox。
2. ``fastapi``、``uvicorn``、``httpx`` 和 ``PyYAML``。

机器上需要可用的 Docker。把 ``examples/agent/harbor/mini_swe_model.py`` 安装到 sandbox 内 agent 环境的 ``minisweagent/models/rlinf_text_model.py``。``examples/agent/harbor/mini_swe.yaml`` 选用其中的 ``TrajectoryTextModel``：回复里没有合法动作时，它把这条回复留在对话记录中，让模型看到格式错误的反馈；session 的 token 预算用完时，它以 ``LimitsExceeded`` 结束 agent。

数据
----------------------------------------

每道训练题对应一个 Harbor 任务目录，包含 ``task.toml``、``instruction.md``、``environment/``、``tests/`` 和 ``solution/``。本配方使用 ``SWE_GYM_LITE_TASKS``（`examples/agent/harbor/prepare_data.py`）中的 8 道题，取自 ``SWE-Gym/SWE-Gym-Lite`` 的 train split，revision 为 ``f70b1a29ab120eb0a0ee7a1deb029825e735b2b0``。用 Harbor 仓库里的 SWE-Gym 适配器（``adapters/swegym``）生成任务目录，放到 ``/data/prepared-swe``。

适配器根据 pytest 逐项测试的结果打分，所以要在每道题的 ``tests/test.sh`` 中设置 ``PYTEST_ADDOPTS="-v -rA --color=no"``；否则通过的测试不会输出名称，题目会被记为 0 分。训练前用 Harbor 的 ``oracle`` agent 逐题验收：空补丁应得 0 分，参考答案应得 1 分。

导出训练用的副本：

.. code-block:: bash

   python examples/agent/harbor/prepare_data.py \
     --source-tasks /data/prepared-swe --output-dir /data/swe-training

脚本把任务目录复制到 ``/data/swe-training/tasks`` 并去掉 ``solution/``，同时生成 ``/data/swe-training/train.jsonl``，每道题一行，数据集 ID 为 ``swe-gym-lite``。导出其他题目时，传入 ``--instance-ids`` 和 ``--dataset-id``。

配置
----------------------------------------

填写模型路径、数据路径和 Harbor 服务地址：

.. code-block:: yaml

   data:
     train_data_paths: ["/data/swe-training/train.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/Qwen3-30B-A3B-Instruct-2507
   agentloop:
     harbor:
       server_url: http://127.0.0.1:30200

SWE 相关的配置已经设置好：

.. code-block:: yaml

   agentloop:
     agentic_tokenizer: qwen
     chat_template_kwargs:
       enable_thinking: False
     harbor:
       max_concurrency: 32
       trial_timeout_s: 8400
       client_timeout_s: 8460
       cleanup_timeout_s: 40
       gateway_port: 30300
       gateway_advertise_host: null
   rollout:
     sglang:
       return_generation_errors: True

- ``agentloop.agentic_tokenizer`` 和 ``chat_template_kwargs`` 决定 gateway 如何渲染 agent 的对话记录。渲染结果必须与之前生成的文本完全一致，否则该 trial 记为前缀断裂并被丢弃。
- ``agentloop.harbor.max_concurrency`` 限制同时运行的 sandbox 数量。``examples/agent/harbor/server.yaml`` 中服务端的 ``max_concurrency`` 不能比它小。
- ``trial_timeout_s`` 包括排队、环境准备、agent 运行和测试的全部时间。``client_timeout_s`` 要大于 ``trial_timeout_s + cleanup_timeout_s``，保证客户端放弃之前，服务端能完成 trial 的取消。
- 每个 agent loop worker 在 ``gateway_port + rank`` 端口提供 gateway。sandbox 通过 ``gateway_advertise_host`` 访问它，默认使用 Ray 节点 IP。
- ``rollout.sglang.return_generation_errors: True`` 让生成失败以错误响应的形式返回 gateway，只让对应的 trial 失败，不会中断 rollout worker。

在 8×A100 上训练
----------------------------------------

8×A100-80G 上的并行方式（actor 与 rollout 共置）：

- Actor（Megatron）：tensor parallel 4、expert parallel 8、expert tensor parallel 1、data parallel 2，optimizer 状态 offload 到 CPU。
- Rollout（SGLang）：8 卡上一个引擎，tensor parallel 8、expert parallel 8、attention data parallel 2。
- 每步为 8 道题各采样 8 条轨迹，共 64 条，最多同时运行 32 个 sandbox。单条轨迹最多 32K token，每轮最多生成 4096 token，agent 最多 80 轮。

在仓库根目录启动 Harbor 服务。服务监听 ``127.0.0.1:30200``；``server.yaml`` 中 ``executor.tasks_dir`` 指向导出的题目，agent 日志和测试输出保存在 ``executor.trials_dir``。

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/serve.py \
     --config examples/agent/harbor/server.yaml

然后启动训练：

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/train.py \
     --config-path . --config-name train \
     rollout.model.model_path=/path/to/model/Qwen3-30B-A3B-Instruct-2507 \
     data.train_data_paths=[/data/swe-training/train.jsonl] \
     agentloop.harbor.server_url=http://127.0.0.1:30200

正式长跑前，可在命令末尾加上 ``runner.max_steps=2 runner.save_interval=1``，跑两步、每步都保存 checkpoint，确认整条链路可用。

只运行一个服务进程，因为 trial 注册表和并发限制都保存在进程内存中。需要认证时，启动服务前设置 ``RLINF_HARBOR_API_KEY``，并把 ``agentloop.harbor.api_key`` 设为相同的值。服务和 gateway 都不加密流量，应部署在可信网络内。

监控与排错
----------------------------------------

每个 trial 都会在 ``agentloop.harbor.artifacts_dir`` 下写一份 JSON 记录，包含各轮对话和最终结果。``harbor/*`` 指标按训练步汇总，比例类指标是 batch 内所有 trial 的平均值。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 指标
     - 含义与处理
   * - ``harbor/infrastructure_failure``
     - 环境准备、sandbox、测试、网络或超时导致的失败。持续升高通常说明 Docker 或服务有问题，查看 ``executor.trials_dir``。
   * - ``harbor/agent_format_failure``
     - 模型多次没有给出合法动作而结束的 trial，按 reward 0 参与训练。
   * - ``harbor/reward_fallback``
     - 轨迹完整但没有可用的测试分数，按 reward 0 参与训练。
   * - ``harbor/invalid_trajectory``
     - 无法用于训练的轨迹：为空、前缀断裂或 sandbox 清理未确认。最多重试 ``agentloop.harbor.max_retries`` 次。
   * - ``harbor/prefix_break``
     - 渲染出的对话记录不再以已记录的 token 开头，通常说明 chat template 无法还原模型自己的输出。
   * - ``harbor/timeout``
     - 达到服务端或客户端时限，或因测试超时等原因失败的 trial。
   * - ``harbor/turns``、``harbor/trial_elapsed_s``、``harbor/trial_elapsed_s_p50``、``harbor/trial_elapsed_s_p95``
     - 每个 trial 的模型调用次数和耗时。
   * - ``harbor/batch_aborted``、``harbor/aborted_groups``、``harbor/skipped_batches``
     - 只要有一道题凑不齐 ``group_size`` 条可用轨迹，所有 worker 都会丢弃整个 batch，训练步数不增加。

连续丢弃 ``agentloop.harbor.max_consecutive_aborted_batches`` 个 batch（配置中为 32）后，runner 会保存 checkpoint 并报错退出，这通常说明 Harbor 服务状态异常。

脱离训练运行 Trial
----------------------------------------

Harbor 服务也可以对接任意兼容 OpenAI 接口的模型服务运行 trial，例如独立部署的 SGLang。``HarborClient`` 提交 trial 并轮询，直到结束：

.. code-block:: python

   from rlinf.workers.agent.harbor.client import HarborClient
   from rlinf.workers.agent.harbor.protocol import HarborTrialRequest

   async def run_attempt():
       client = HarborClient("http://127.0.0.1:30200", api_key="task-service-key")
       request = HarborTrialRequest(
           trial_id="attempt-001", session_id="session-001",
           instance_id="getmoto__moto-4990",
           model_url="http://127.0.0.1:30000/v1",
           model="openai/swe-model", timeout_s=900,
       )
       try:
           return await client.run_trial(request)
       finally:
           await client.close()

每次尝试都要使用唯一的 ``trial_id`` 和 ``session_id``。``model_url`` 是 sandbox 能访问到的 OpenAI API 根地址。结果中的 ``reward`` 与 ``infrastructure_failure`` 相互独立：只要测试给出了有限的分数，``reward`` 就是这个分数，即使 agent 本身也失败了；否则为 ``None``。``client.cancel_trial(request)`` 取消单个 trial，并等待它的 sandbox 被删除。结果中 ``cleanup_confirmed=False`` 表示 sandbox 可能仍在运行，需要手动清理。
