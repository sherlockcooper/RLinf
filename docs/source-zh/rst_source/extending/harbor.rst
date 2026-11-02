通过 HTTP 运行 Harbor 任务
==========================

本文介绍如何在带 Docker 的 CPU 节点运行独立的 mini-swe-agent trial，并通过 Harbor 任务服务取回 verifier reward。先准备 Harbor 任务目录和可访问的模型接口，再启动服务、提交任务和管理取消；最后说明这一版与训练链路的接入边界。

启动服务
--------

任务服务管理 sandbox 容器，mini-swe-agent 向传入的模型地址发送请求，模型可以位于另一台 GPU 机器上。训练时应使用 session gateway 地址；独立评测可先使用 SGLang 接口。

使用独立 Python 环境，安装 ``harbor-miles-v0.20.0`` tag 对应的 Harbor（实测 commit 为 ``917b3c1be7cf6553a7a9fe25de8adb6f8e639287``），以及 ``fastapi``、``uvicorn``、``httpx``、``PyYAML`` 和 ``omegaconf``。CPU 服务按需导入 Harbor，不加载模型权重。

复制 ``examples/agent/harbor/server.yaml`` 并修改任务及输出目录。每个 ``instance_id`` 对应 ``tasks_dir`` 下包含 ``task.toml`` 的直接子目录。将 ``config_file`` 指向 mini-swe-agent 2.4.6 完整的 ``mini_textbased.yaml`` 配置，使用 ``model_class: litellm_textbased``，并设置有限的 agent step limit。在仓库根目录启动服务：

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/serve.py --config /data/harbor/server.yaml

默认监听 ``127.0.0.1:30200``。跨机器调用时，改为可访问的监听地址，在服务端设置 ``RLINF_HARBOR_API_KEY``，并向客户端传入相同密钥。只运行一个 Uvicorn 进程，trial 注册表和并发限制都在进程内维护。这套协议使用 ``/v1/trials``，与旧 miles 服务的 ``/run`` 接口不兼容。

提交并读取任务结果
------------------

每次尝试都需要独立的 trial id 和 session id，同一 prompt 的多个采样也不例外。``HarborClient.run_trial`` 提交请求并轮询至终态。``model_url`` 是 sandbox 能访问的 API 根地址；使用路径识别会话的网关时，地址中应包含 session 路径。适配器还会向 mini-swe-agent 传入 ``X-Session-ID``。

.. code-block:: python

   from rlinf.workers.agent.harbor.client import HarborClient
   from rlinf.workers.agent.harbor.protocol import HarborTrialRequest

   async def run_attempt():
       client = HarborClient("http://cpu-host:30200", api_key="task-service-key")
       request = HarborTrialRequest(
           trial_id="attempt-001", session_id="session-001",
           instance_id="psf__requests-5414-agent",
           model_url="http://gateway:30000/sessions/session-001/v1",
           model="openai/swe-model", timeout_s=900,
       )
       try:
           result = await client.run_trial(request)
           return result
       finally:
           await client.close()

结果分别记录 ``reward`` 和 ``infrastructure_failure``。verifier 正常给出的零分属于模型结果；安装、环境、verifier、传输及整体超时失败均返回 ``reward=None``。未知的 agent 退出状态保守地归类为基础设施失败。``eval_report.trial_dir`` 指向服务端产物目录，并非客户端本地路径。 agent 因 ``FormatError`` 或 ``RepeatedFormatError`` 退出时，返回 ``reward=0`` 和 ``failure_kind=model_format_error``，保留已有轨迹进入采样组，不重采样；如果同时发生基础设施故障，仍按基础设施失败处理。

复用依赖时，将固定版本的 agent 和 verifier 依赖预装进任务镜像，通过 ``agent_config.import_path`` 配置可信的预装 agent 适配器，并设置 ``public_name: mini-swe-agent``。适配器需要验证已安装版本并跳过安装；仅缓存镜像不会阻止 Harbor 默认的 agent setup 再次安装软件包。每个 trial 仍使用独立的临时 sandbox。

取消与关闭
----------

当轨迹过期时，调用 ``await client.cancel_trial(request)``，只取消该 trial，不会清空其他客户端的任务。``close`` 先取消当前客户端尚未结束的任务，再关闭连接。服务正常退出时也会取消未完成的 trial。

服务端超时包含排队、安装、agent 执行和验证。需要满足 ``client_timeout_s > timeout_s + cleanup_timeout_s``，且客户端的清理等待时间至少覆盖服务端清理等待时间。检查 ``cleanup_confirmed``；无法确认清理的情况属于基础设施失败，需要排查。仍未退出的执行器会继续占用并发名额。

相同提交在有界内存注册表保留记录期间具有幂等性。这一版没有进程崩溃后的持久化恢复，也不共享多进程注册表。服务重启后使用新的 id，异常退出后检查残留 Docker 资源。

用逐轮 token 记录训练
------------------------------

使用继承 ``AgentLoopWorker`` 的 ``HarborAgentLoopWorker`` 接入同步 ``AgentRunner``。每个 prompt group 并发启动独立 Harbor trial。worker 在自己的 CPU 节点托管 session gateway，mini-swe-agent 向网关发送普通 chat completions，网关通过 ``generate_input_channel`` 调用 RLinf rollout。Docker 仍由单独部署的 Harbor 服务管理。

先导出已验证的 Harbor 任务，再用配套 FSDP/SGLang 配置启动训练：

.. code-block:: bash

   python examples/agent/harbor/prepare_data.py \
     --source-tasks /data/prepared-swe --output-dir /data/swe-training \
     --instance-ids psf__requests-5414-agent --dataset-id swe-bench-verified
   PYTHONPATH=. python examples/agent/harbor/train.py \
     --config-path . --config-name train \
     actor.model.model_path=/models/Qwen2.5-Coder-1.5B \
     rollout.model.model_path=/models/Qwen2.5-Coder-1.5B \
     data.train_data_paths=[/data/swe-training/train.jsonl] \
     agentloop.harbor.server_url=http://cpu-host:30200

转换脚本复制任务目录并排除 oracle solution，同时生成 ``train.jsonl``，包含 ``instance_id``、``dataset_id``、``prompt`` 及供 ``ReasoningDataset`` 读取的 ``solutions`` 元数据。将 Harbor 服务的 ``tasks_dir`` 设置为 ``/data/swe-training/tasks``。原始 SWE-Bench 数据需要先转换为可执行、verifier 已验证的 Harbor 任务。示例的单实例用于链路测试；训练实验需要扩大子集。

RLinf 环境需要安装 ``fastapi``、``uvicorn`` 和 ``httpx``。通过 ``gateway_advertise_host`` 指定 sandbox 可访问的网关地址，默认使用 Ray 节点 IP；各 worker 监听 ``gateway_port + rank``。服务与网关应部署在可信网络内。示例将 actor 与 rollout 共置于 GPU 0，在不同阶段 offload。

配套 base 模型配置用于单轮 smoke test，请将 Harbor agent 的 ``step_limit`` 设为 1。多轮训练需要使用结束符与 chat template 兼容的 chat 模型，并保证 harness 在动作格式错误后仍保留 assistant 消息。mini-swe-agent 2.4.6 的文本模型可能省略这类消息，网关会检测到前缀断裂。

每轮请求中，网关渲染 chat template，检查其是否保留上一轮输入与实际输出的解码前缀，只对追加文本分词，并记录实际输出 token ID、logprob、结束原因和逐 token policy version。请求使用 ``/sessions/{session_id}/v1/chat/completions``，或向 ``/v1/chat/completions`` 传入 ``X-Session-ID``。GET 和 DELETE ``/sessions/{session_id}`` 分别读取记录和关闭会话；worker 创建会话，关闭前将记录保存到 ``artifacts_dir``。

verifier 完成后，worker 拼成单序列 ``RolloutResult``：模型输出 mask 为 1，追加的 harness 文本 mask 为 0，reward 来自 verifier，再通过现有 channel 交给 actor。前缀断裂、空轨迹或清理未确认的 trial 最多重试 ``max_retries`` 次；采样组仍不完整时跳过当前批次，继续下一批。执行或评分失败但轨迹完整的样本按下文规则补零保留。通过 ``harbor/*`` 指标及 JSON 轨迹文件排查失败。

当前 policy version 对应同步权重更新。R3 预留了 ``routing_replay`` 字段，尚未实现 routing 收集和异步暂停恢复。网关只接受普通、非流式文本 chat；扩大规模前需要用真实轨迹验证 chat template 兼容性。

只需远程调度任务时，使用 ``rlinf.workers.agent.harbor_task_worker`` 中的 ``HarborTaskWorker``。其 ``run_trials``、``cancel_trial`` 和 ``shutdown`` 管理任务结果，不组装训练样本。

长时间训练
~~~~~~~~~~

设置 ``runner.max_duration_hours: 12``，训练会在达到十二小时后的首个完整更新结束时退出。计时从 worker 初始化后的训练循环开始；不设置此字段则不限制运行时间。当 ``runner.save_interval`` 为正数时，到达时限会保存 checkpoint，再关闭 Harbor 会话。

使用 Megatron Bridge 时，设置 ``actor.megatron.save_hf_checkpoint: false`` 可跳过保存时的额外 HF 权重导出，仍保留用于续训的 Megatron 模型、优化器及调度器状态。默认值为 ``true``。关闭额外导出可减少保存阶段的内存开销。

mini-swe-agent 适配器将 session token 预算耗尽视为 ``LimitsExceeded``，结束 agent 循环，不重试已被拒绝的请求，然后由 Harbor 对当前仓库状态评分。

失败处理
~~~~~~~~

执行或验证失败（包括 verifier 超时）时，若轨迹 token 完整且环境清理已确认，保留轨迹并使用训练 reward 0。原始 verdict 和基础设施失败标记保留；训练元数据中的 ``training_reward`` 与 ``reward_source`` 标明补零。空轨迹、前缀断裂和清理未确认的样本按 ``max_retries`` 重试；仍有不完整组时，各 Worker 在向 actor 发送数据前共同丢弃当前批次，继续下一批，不增加更新步数。全对和全错组照常保留。

``harbor/reward_fallback``、``harbor/invalid_trajectory`` 和 ``harbor/skipped_batches`` 分别记录补零、无效轨迹和跳过批次。配置错误明确报错。连续跳过 ``agentloop.harbor.max_consecutive_aborted_batches`` 批（默认32）后保存 checkpoint 并报告持续服务故障，避免没有训练数据却无限运行。
