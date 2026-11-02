Run Harbor Tasks Through HTTP
========================================

Use the Harbor task service to run independent mini-swe-agent attempts on a CPU host with Docker and retrieve verifier rewards. This guide covers service startup, client calls, cancellation, and the boundary with training. Prepare your Harbor task directories and a reachable model endpoint first.

Start the Service
----------------------------------------

The task service owns sandbox containers. mini-swe-agent sends model requests to the supplied model URL, so the model can run on a different GPU host. For training, supply a session gateway URL; for an evaluation smoke test, a standalone SGLang endpoint is sufficient.

Use a separate Python environment with Harbor from the ``harbor-miles-v0.20.0`` tag (tested commit ``917b3c1be7cf6553a7a9fe25de8adb6f8e639287``), ``fastapi``, ``uvicorn``, ``httpx``, ``PyYAML``, and ``omegaconf``. The CPU service imports Harbor lazily and does not load model weights.

Copy ``examples/agent/harbor/server.yaml`` and configure the task and output directories. Each ``instance_id`` resolves to one immediate child of ``tasks_dir`` containing ``task.toml``. Set ``config_file`` to the complete mini-swe-agent 2.4.6 ``mini_textbased.yaml`` configuration; use ``model_class: litellm_textbased`` and a finite agent step limit. Start the service from the repository root:

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/serve.py --config /data/harbor/server.yaml

The default listener is ``127.0.0.1:30200``. For remote clients, select a reachable interface and set ``RLINF_HARBOR_API_KEY`` on the server, then pass the same key to the client. Run one Uvicorn process: the trial registry and concurrency limit are process-local. This protocol uses ``/v1/trials`` and is not wire-compatible with the legacy miles ``/run`` service.

Submit and Read a Trial
----------------------------------------

Give each attempt a unique trial id and session id, including samples from the same prompt group. ``HarborClient.run_trial`` submits the request and polls until a terminal result. ``model_url`` is the API root visible from the sandbox, including the session path when the gateway uses path-based routing. The adapter also passes ``X-Session-ID`` to mini-swe-agent.

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

The result separates ``reward`` from ``infrastructure_failure``. A valid verifier reward of zero is a model outcome. Setup, environment, verifier, transport, and overall deadline failures carry ``reward=None``. Unknown agent exit statuses are treated conservatively as infrastructure failures. ``eval_report.trial_dir`` points to server-side artifacts; it is not a path on the client machine. Agent exits ``FormatError`` and ``RepeatedFormatError`` carry ``reward=0`` and ``failure_kind=model_format_error``; their recorded trajectories remain in the sampling group without resampling. A simultaneous infrastructure failure still takes precedence.

To reuse dependencies, prepare task images with the pinned agent and verifier packages. Configure a trusted preinstalled-agent adapter through ``agent_config.import_path`` and set ``public_name: mini-swe-agent``. The adapter must validate the installed version and skip installation; a cached image alone does not stop Harbor's default agent setup from installing packages. Each trial still gets its own disposable sandbox.

Cancel and Shut Down
----------------------------------------

When an attempt becomes stale, call ``await client.cancel_trial(request)``. This deletes only that trial; it does not flush other clients' work. ``close`` cancels this client's outstanding attempts before closing connections. The server also cancels outstanding trials during graceful shutdown.

The server deadline includes queueing, setup, agent execution, and verification. Configure ``client_timeout_s > timeout_s + cleanup_timeout_s`` and give the client cleanup allowance at least the server's cleanup allowance. Inspect ``cleanup_confirmed``: an unconfirmed cleanup is an infrastructure failure requiring operator attention. A pending executor retains its concurrency slot until it exits.

Identical submissions are idempotent while their completed records remain in the bounded in-memory registry. There is no durable recovery after a process crash and no multi-process registry. Use unique ids across service restarts and inspect leftover Docker resources after an ungraceful stop.

Train With Recorded Tokens
----------------------------------------

Use ``HarborAgentLoopWorker``, a subclass of ``AgentLoopWorker``, with the synchronous ``AgentRunner``. Each prompt group starts independent Harbor trials. The worker hosts a session gateway on its CPU node; mini-swe-agent sends text chat completions there, and the gateway calls RLinf through ``generate_input_channel``. The Harbor service still owns Docker on its separately configured host.

Export an existing, verified Harbor task directory and start training with the bundled FSDP/SGLang configuration:

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

The exporter copies task directories without their oracle solutions and writes ``train.jsonl`` with ``instance_id``, ``dataset_id``, ``prompt``, and a ``solutions`` metadata mapping consumed by ``ReasoningDataset``. Set the Harbor server's ``tasks_dir`` to ``/data/swe-training/tasks``. Raw SWE-Bench records must first be converted into working Harbor tasks with validated verifiers. The example uses one instance for integration testing; choose a larger subset for training experiments.

Install ``fastapi``, ``uvicorn``, and ``httpx`` in the RLinf environment. Configure a gateway address reachable from the Harbor sandbox through ``gateway_advertise_host``; the default uses the Ray node IP. Each worker listens on ``gateway_port + rank``. Keep the service and gateway on a trusted network. The example places actor and rollout on GPU 0 and offloads them between phases.

The bundled base-model configuration is a single-turn smoke test: set the Harbor agent's ``step_limit`` to 1. For multi-turn training, use a chat model with compatible template terminators and ensure the harness preserves assistant messages after action-format errors. mini-swe-agent 2.4.6's text model can omit those messages, which the gateway detects as a broken prefix.

For each turn, the gateway renders the chat template, verifies that it preserves the decoded previous input and actual output, and tokenizes only the appended text. It records output token IDs, logprobs, finish reason, and per-token policy versions. Requests use ``/sessions/{session_id}/v1/chat/completions`` or ``/v1/chat/completions`` with ``X-Session-ID``. GET and DELETE on ``/sessions/{session_id}`` read and close a session; the worker creates sessions and writes their records to ``artifacts_dir`` before closing them.

After verification, the worker assembles a single ``RolloutResult``: generated tokens have mask 1, appended harness text has mask 0, and the verifier supplies the reward. The existing channel transfers the result to the actor. Complete, aligned trajectories are retained with training reward zero when execution or verification fails, including verifier timeout. Original verdicts and infrastructure flags remain in artifacts; ``training_reward`` and ``reward_source`` identify fallback scoring in training metadata. Prefix-broken, empty, or uncleared trials are retried up to ``max_retries``. An incomplete group discards the current batch on all workers before anything reaches the actor; the runner continues with the next batch without advancing its update counter. Inspect ``harbor/*`` metrics and the JSON artifacts to diagnose failures.

Policy versions currently identify synchronous weight updates. R3 has a reserved ``routing_replay`` field; routing collection and asynchronous pause/resume are not implemented. Only ordinary, non-streaming text chat is accepted. Verify template compatibility on real trajectories before scaling up.

For task-only orchestration, use ``HarborTaskWorker`` in ``rlinf.workers.agent.harbor_task_worker``. Its ``run_trials``, ``cancel_trial``, and ``shutdown`` methods manage verdicts without assembling training samples.

Long-running training
~~~~~~~~~~~~~~~~~~~~~

Set ``runner.max_duration_hours: 12`` to stop at the first completed update
after twelve hours. The budget starts when training begins, after worker
initialization. Omit the field for unlimited wall time. With a positive
``runner.save_interval``, reaching the budget saves a checkpoint before
shutting down active Harbor sessions.

For Megatron Bridge, ``actor.megatron.save_hf_checkpoint: false`` skips the
additional HF weight export during checkpoint saves. Native Megatron model,
optimizer and scheduler state are still saved for resuming training. The
default is ``true``. This avoids the extra memory needed by HF export.

The mini-swe-agent adapter treats an exhausted session token budget as ``LimitsExceeded``. It ends the agent loop without retrying the rejected request, allowing Harbor to verify the current repository state.

``harbor/reward_fallback``, ``harbor/invalid_trajectory``, and ``harbor/skipped_batches`` distinguish fallback rewards from discarded batches. All-correct and all-wrong groups remain eligible. Configuration errors fail explicitly. After ``agentloop.harbor.max_consecutive_aborted_batches`` (default 32) consecutive discarded batches, the runner saves a checkpoint and reports persistent service failure. This protects against running indefinitely with no trainable data.
