通过 HTTP 运行 Harbor 任务
==========================

本文介绍如何通过 Harbor 任务服务在 Docker sandbox 中运行 mini-swe-agent，并取回 verifier reward。先准备 Harbor 任务目录和可访问的模型接口，再启动服务、提交任务和管理取消，最后用记录的轨迹接入训练。

启动服务
--------

在同一台服务器运行 Harbor 和 RLinf。Harbor 管理 sandbox 容器，mini-swe-agent 向传入的模型地址发送请求。训练时使用 session gateway 地址，独立评测时使用 SGLang 接口。

使用已安装 ``torch`` 的 RLinf Python 环境，安装 ``harbor-miles-v0.20.0`` tag 对应的 Harbor（实测 commit 为 ``917b3c1be7cf6553a7a9fe25de8adb6f8e639287``），以及 ``fastapi``、``uvicorn``、``httpx``、``PyYAML`` 和 ``omegaconf``。

复制 ``examples/agent/harbor/server.yaml`` 并修改任务及输出目录。每个 ``instance_id`` 对应 ``tasks_dir`` 下包含 ``task.toml`` 的直接子目录。配套 ``mini_swe.yaml`` 使用 mini-swe-agent 2.4.6 的文本动作格式，最多运行 80 轮。将同目录的 ``mini_swe_model.py`` 安装到 sandbox 的 agent 环境中，路径为 ``minisweagent/models/rlinf_text_model.py``；配置中的 ``TrajectoryTextModel`` 会保留 assistant 消息，并将格式错误反馈交回模型。在仓库根目录启动服务：

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/serve.py --config /data/harbor/server.yaml

默认监听 ``127.0.0.1:30200``。在服务端设置 ``RLINF_HARBOR_API_KEY``，并向客户端传入相同密钥，即可启用认证。运行一个 Uvicorn 进程，trial 注册表和并发限制都在进程内维护。trial API 使用 ``/v1/trials``。

提交并读取任务结果
------------------

每次尝试都需要独立的 trial id 和 session id，同一 prompt 的多个采样也不例外。``HarborClient.run_trial`` 提交请求并轮询至终态。``model_url`` 是 sandbox 能访问的 API 根地址；使用路径识别会话的网关时，地址中应包含 session 路径。适配器还会向 mini-swe-agent 传入 ``X-Session-ID``。

.. code-block:: python

   from rlinf.workers.agent.harbor.client import HarborClient
   from rlinf.workers.agent.harbor.protocol import HarborTrialRequest

   async def run_attempt():
       client = HarborClient("http://127.0.0.1:30200", api_key="task-service-key")
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

结果独立记录 ``reward`` 和 ``infrastructure_failure``。执行器接受有限数值形式的 verifier 分数，包括零分；即使 agent 或 verifier 同时报告异常，也保留有效分数。缺失或非法分数记为 ``reward=None``；如果只是模型格式错误且没有基础设施故障，则返回 ``reward=0``，并标记 ``reward_source=model_format_error``。有效分数标记为 ``reward_source=verifier``。安装、环境、verifier、传输和整体超时异常归类为基础设施失败，未知的 agent 退出状态也标记为基础设施失败。agent 因 ``FormatError`` 或 ``RepeatedFormatError`` 退出时，记录 ``failure_kind=model_format_error``；若同时出现基础设施异常，则记录该异常类别。``eval_report.trial_dir`` 指向服务端产物目录，并非客户端本地路径。

复用依赖时，将固定版本的 agent 和 verifier 依赖预装进任务镜像，通过 ``agent_config.import_path`` 配置可信的预装 agent 适配器，并设置 ``public_name: mini-swe-agent``。适配器在 agent setup 中验证已安装版本并复用这些软件包。每个 trial 使用独立的临时 sandbox。

取消与关闭
----------

当轨迹过期时，调用 ``await client.cancel_trial(request)`` 取消该 trial。``close`` 先取消当前客户端尚未结束的任务，再关闭连接。服务正常退出时也会取消未完成的 trial。

服务端超时包含排队、安装、agent 执行和验证。需要满足 ``client_timeout_s > timeout_s + cleanup_timeout_s``，且客户端的清理等待时间至少覆盖服务端清理等待时间。检查 ``cleanup_confirmed``；无法确认清理的情况属于基础设施失败，需要排查。仍未退出的执行器会继续占用并发名额。

相同提交在有界内存注册表保留记录期间具有幂等性，注册表随服务进程存续。服务重启后使用新的 id，异常退出后检查残留 Docker 资源。

用逐轮 token 记录训练
------------------------------

使用继承 ``AgentLoopWorker`` 的 ``HarborAgentLoopWorker`` 接入同步 ``AgentRunner``。每个 prompt group 并发启动独立 Harbor trial。worker 托管 session gateway，mini-swe-agent 向网关发送普通 chat completions，网关通过 ``generate_input_channel`` 调用 RLinf rollout。Harbor 服务在同一台服务器上管理 Docker。

先导出准备好的 Harbor 任务，再使用 8 张 A100 80 GB 和 Qwen3-30B-A3B-Instruct-2507 启动同步 GRPO。配套配置使用 Megatron 训练、SGLang 生成。先将至少 8 个已准备实例的 ID 写入 ``/data/swe-instance-ids.txt``：

.. code-block:: bash

   python examples/agent/harbor/prepare_data.py \
     --source-tasks /data/prepared-swe --output-dir /data/swe-training \
     --instance-ids $(cat /data/swe-instance-ids.txt) --dataset-id swe-bench-verified
   PYTHONPATH=. python examples/agent/harbor/train.py \
     --config-path . --config-name train \
     rollout.model.model_path=/models/Qwen3-30B-A3B-Instruct-2507 \
     data.train_data_paths=[/data/swe-training/train.jsonl] \
     agentloop.harbor.server_url=http://127.0.0.1:30200

转换脚本复制任务目录并排除 oracle solution，同时生成 ``train.jsonl``，包含 ``instance_id``、``dataset_id``、``prompt`` 及供 ``ReasoningDataset`` 读取的 ``solutions`` 元数据。将 Harbor 服务的 ``tasks_dir`` 设置为 ``/data/swe-training/tasks``。原始 SWE-Bench 数据需要先转换为可执行、verifier 已验证的 Harbor 任务。命令导出 ``/data/swe-instance-ids.txt`` 中列出的实例。每批抽取 8 道题，每题采样 8 次，共 64 条轨迹，同时运行 32 个 sandbox。

RLinf 环境需要安装 ``fastapi``、``uvicorn`` 和 ``httpx``。通过 ``gateway_advertise_host`` 指定 sandbox 可访问的网关地址，默认使用 Ray 节点 IP；各 worker 监听 ``gateway_port + rank``。服务与网关应部署在可信网络内。示例将 actor 与 rollout 共置于 GPU 0–7，在不同阶段 offload。Megatron 使用 TP 4、PP 1、EP 8、expert TP 1；SGLang 使用 TP 8、EP 8、attention DP 2。

示例选择 ``qwen`` agentic tokenizer 并关闭 thinking，整条轨迹限制为 32K token，每次生成最多 4096 token，agent 最多运行 80 轮。训练采用 micro-batch 1、完整 activation recomputation、optimizer CPU offload fraction 1.0，学习率为 ``1e-6``，采样温度为 ``0.7``。

每轮请求中，网关渲染 chat template，检查其是否保留上一轮输入与实际输出的解码前缀，只对追加文本分词，并记录实际输出 token ID、logprob、结束原因和逐 token policy version。请求使用 ``/sessions/{session_id}/v1/chat/completions``，或向 ``/v1/chat/completions`` 传入 ``X-Session-ID``。GET 和 DELETE ``/sessions/{session_id}`` 分别读取记录和关闭会话。关闭会话会等待正在进行的生成结束，重复 DELETE 返回 ``closed=true``；worker 创建会话，关闭前将记录保存到 ``artifacts_dir``。

verifier 完成后，worker 拼成单序列 ``RolloutResult``：模型输出 mask 为 1，追加的 harness 文本 mask 为 0，reward 来自 verifier，再通过现有 channel 交给 actor。前缀断裂、空轨迹或清理未确认的 trial 最多重试 ``max_retries`` 次；采样组仍不完整时跳过当前批次，继续下一批。执行或评分失败但轨迹完整的样本按下文规则补零保留。通过 ``harbor/*`` 指标及 JSON 轨迹文件排查失败。

当前 policy version 对应同步权重更新。网关接受普通、非流式文本 chat；扩大规模前需要用真实轨迹验证 chat template 兼容性。


会话限制与失败处理
------------------

通过 ``algorithm.sampling_params.max_new_tokens`` 限制单次生成长度，``data.max_prompt_length`` 限制初始 prompt，``runner.seq_length`` 限制整条轨迹。``agentloop.chat_template_kwargs`` 将模型专用选项传给 agentic tokenizer。示例显式启用 ``rollout.sglang.return_generation_errors``，让生成异常返回网关，由 trial 流程处理；其他 SGLang 配置默认抛出异常。

mini-swe-agent 适配器将 session token 预算耗尽视为 ``LimitsExceeded``，结束 agent 循环，不重试已被拒绝的请求，然后由 Harbor 对当前仓库状态评分。

轨迹 token 完整且环境清理已确认时，保留有效 verifier reward；只有缺少有效分数时才使用训练 reward 0，包括 verifier 超时但没有分数的情况。原始 verdict 和基础设施失败标记保留；训练元数据中的 ``training_reward`` 与 ``reward_source`` 标明补零。空轨迹、前缀断裂和清理未确认的样本按 ``max_retries`` 重试；仍有不完整组时，各 Worker 在向 actor 发送数据前共同丢弃当前批次，继续下一批，不增加更新步数。全对和全错组照常保留。

``harbor/reward_fallback``、``harbor/invalid_trajectory`` 和 ``harbor/skipped_batches`` 分别记录缺分补零、无效轨迹和跳过批次。backend 对模型格式错误给出的零分计入 ``harbor/agent_format_failure``，不增加 ``harbor/reward_fallback``。配置错误明确报错。连续跳过 ``agentloop.harbor.max_consecutive_aborted_batches`` 批（默认 32）后保存 checkpoint 并报告持续服务故障，避免没有训练数据却无限运行。
