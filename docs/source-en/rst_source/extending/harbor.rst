Run Harbor Tasks Through HTTP
========================================

Use the Harbor task service to run mini-swe-agent in Docker sandboxes and retrieve verifier rewards. This guide covers service startup, client calls, cancellation, and training with recorded trajectories. Prepare your Harbor task directories and a reachable model endpoint first.

Start the Service
----------------------------------------

Run Harbor and RLinf on the same server. Harbor manages sandbox containers, and mini-swe-agent sends model requests to the supplied model URL. For training, supply a session gateway URL; for a standalone evaluation, supply a SGLang endpoint.

Use the RLinf Python environment with ``torch`` installed. Install Harbor from the ``harbor-miles-v0.20.0`` tag (tested commit ``917b3c1be7cf6553a7a9fe25de8adb6f8e639287``), along with ``fastapi``, ``uvicorn``, ``httpx``, ``PyYAML``, and ``omegaconf``.

Copy ``examples/agent/harbor/server.yaml`` and configure the task and output directories. Each ``instance_id`` resolves to one immediate child of ``tasks_dir`` containing ``task.toml``. The bundled ``mini_swe.yaml`` uses mini-swe-agent 2.4.6 text actions with an 80-turn limit. Install ``mini_swe_model.py`` from the same example directory as ``minisweagent/models/rlinf_text_model.py`` in the sandbox agent environment; the configured ``TrajectoryTextModel`` preserves assistant messages and sends format-error feedback back to the model. Start the service from the repository root:

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/serve.py --config /data/harbor/server.yaml

The default listener is ``127.0.0.1:30200``. Set ``RLINF_HARBOR_API_KEY`` on the server and pass the same key to the client to enable authentication. Run one Uvicorn process: the trial registry and concurrency limit are process-local. The trial API uses ``/v1/trials``.

Submit and Read a Trial
----------------------------------------

Give each attempt a unique trial id and session id, including samples from the same prompt group. ``HarborClient.run_trial`` submits the request and polls until a terminal result. ``model_url`` is the API root visible from the sandbox, including the session path when the gateway uses path-based routing. The adapter also passes ``X-Session-ID`` to mini-swe-agent.

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

The result records ``reward`` independently of ``infrastructure_failure``. The executor accepts finite numeric verifier scores, including zero, and preserves them even when the agent or verifier reports an error. A missing or invalid score becomes ``reward=None``, except that a model format error without an infrastructure failure receives ``reward=0`` and ``reward_source=model_format_error``. Valid scores have ``reward_source=verifier``. Setup, environment, verifier, transport, and overall deadline errors are classified as infrastructure failures; unknown agent exit statuses receive the same flag. Agent exits ``FormatError`` and ``RepeatedFormatError`` have ``failure_kind=model_format_error`` unless an infrastructure error also occurred. ``eval_report.trial_dir`` points to server-side artifacts, not a client-local directory.

To reuse dependencies, prepare task images with the pinned agent and verifier packages. Configure a trusted preinstalled-agent adapter through ``agent_config.import_path`` and set ``public_name: mini-swe-agent``. The adapter validates the installed version and reuses those packages during agent setup. Each trial gets its own disposable sandbox.

Cancel and Shut Down
----------------------------------------

When an attempt becomes stale, call ``await client.cancel_trial(request)`` to cancel that trial. ``close`` cancels this client's outstanding attempts before closing connections. The server also cancels outstanding trials during graceful shutdown.

The server deadline includes queueing, setup, agent execution, and verification. Configure ``client_timeout_s > timeout_s + cleanup_timeout_s`` and give the client cleanup allowance at least the server's cleanup allowance. Inspect ``cleanup_confirmed``: an unconfirmed cleanup is an infrastructure failure requiring operator attention. A pending executor retains its concurrency slot until it exits.

Identical submissions are idempotent while their completed records remain in the bounded in-memory registry. The registry lives for the duration of the service process. Use unique ids across service restarts and inspect leftover Docker resources after an ungraceful stop.

Train With Recorded Tokens
----------------------------------------

Use ``HarborAgentLoopWorker``, a subclass of ``AgentLoopWorker``, with the synchronous ``AgentRunner``. Each prompt group starts independent Harbor trials. The worker hosts a session gateway; mini-swe-agent sends text chat completions there, and the gateway calls RLinf through ``generate_input_channel``. The Harbor service manages Docker on the same server.

Export your prepared Harbor tasks and start synchronous GRPO with Qwen3-30B-A3B-Instruct-2507 on eight A100 80 GB GPUs. The bundled configuration uses Megatron for training and SGLang for rollout. List at least eight prepared instance IDs in ``/data/swe-instance-ids.txt``:

.. code-block:: bash

   python examples/agent/harbor/prepare_data.py \
     --source-tasks /data/prepared-swe --output-dir /data/swe-training \
     --instance-ids $(cat /data/swe-instance-ids.txt) --dataset-id swe-bench-verified
   PYTHONPATH=. python examples/agent/harbor/train.py \
     --config-path . --config-name train \
     rollout.model.model_path=/models/Qwen3-30B-A3B-Instruct-2507 \
     data.train_data_paths=[/data/swe-training/train.jsonl] \
     agentloop.harbor.server_url=http://127.0.0.1:30200

The exporter copies task directories without their oracle solutions and writes ``train.jsonl`` with ``instance_id``, ``dataset_id``, ``prompt``, and a ``solutions`` metadata mapping consumed by ``ReasoningDataset``. Set the Harbor server's ``tasks_dir`` to ``/data/swe-training/tasks``. Raw SWE-Bench records must first be converted into working Harbor tasks with validated verifiers. The command exports the instances listed in ``/data/swe-instance-ids.txt``. Each batch samples eight questions with eight trajectories per question, for 64 trajectories and 32 concurrent sandboxes.

Install ``fastapi``, ``uvicorn``, and ``httpx`` in the RLinf environment. Configure a gateway address reachable from the Harbor sandbox through ``gateway_advertise_host``; the default uses the Ray node IP. Each worker listens on ``gateway_port + rank``. Keep the service and gateway on a trusted network. The example places actor and rollout on GPUs 0–7 and offloads them between phases. Megatron uses TP 4, PP 1, EP 8, and expert TP 1; SGLang uses TP 8, EP 8, and attention DP 2.

The example selects the ``qwen`` agentic tokenizer with thinking disabled, a 32K-token trajectory budget, at most 4096 generated tokens per turn, and at most 80 agent turns. Training uses micro-batch 1, full activation recomputation, and optimizer CPU offload fraction 1.0. The learning rate is ``1e-6`` and sampling temperature is ``0.7``.

For each turn, the gateway renders the chat template, verifies that it preserves the decoded previous input and actual output, and tokenizes only the appended text. It records output token IDs, logprobs, finish reason, and per-token policy versions. Requests use ``/sessions/{session_id}/v1/chat/completions`` or ``/v1/chat/completions`` with ``X-Session-ID``. GET and DELETE on ``/sessions/{session_id}`` read and close a session. Closing waits for an in-flight generation, and repeated DELETE requests return ``closed=true``; the worker creates sessions and writes their records to ``artifacts_dir`` before closing them.

After verification, the worker assembles a single ``RolloutResult``: generated tokens have mask 1, appended harness text has mask 0, and the verifier supplies the reward. The existing channel transfers the result to the actor. Complete, aligned trajectories with a valid verifier score keep that score. When no usable score is available, the worker retains the trajectory with training reward zero, including after verifier timeout. Original verdicts and infrastructure flags remain in artifacts; ``training_reward`` and ``reward_source`` identify fallback scoring in training metadata. Prefix-broken, empty, or uncleared trials are retried up to ``max_retries``. An incomplete group discards the current batch on all workers before anything reaches the actor; the runner continues with the next batch without advancing its update counter. Inspect ``harbor/*`` metrics and the JSON artifacts to diagnose failures.

Policy versions identify synchronous weight updates. The gateway accepts ordinary, non-streaming text chat. Verify template compatibility on real trajectories before scaling up.


Session Limits and Failures
----------------------------------------

Set ``algorithm.sampling_params.max_new_tokens`` to cap each model call, ``data.max_prompt_length`` to bound the initial prompt, and ``runner.seq_length`` to bound the full trajectory. ``agentloop.chat_template_kwargs`` passes model-specific options to the agentic tokenizer. The example explicitly enables ``rollout.sglang.return_generation_errors`` so generation failures return to the gateway for trial handling; other SGLang configurations default to raising those errors.

The mini-swe-agent adapter treats an exhausted session token budget as ``LimitsExceeded``. It ends the agent loop without retrying the rejected request, allowing Harbor to verify the current repository state.

``harbor/reward_fallback``, ``harbor/invalid_trajectory``, and ``harbor/skipped_batches`` distinguish missing-score fallback from discarded batches. Model format errors scored zero by the backend count toward ``harbor/agent_format_failure`` and do not increment ``harbor/reward_fallback``. All-correct and all-wrong groups remain eligible. Configuration errors fail explicitly. After ``agentloop.harbor.max_consecutive_aborted_batches`` (default 32) consecutive discarded batches, the runner saves a checkpoint and reports persistent service failure. This protects against running indefinitely with no trainable data.
