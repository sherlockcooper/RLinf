SWE-agent Training for Qwen3-30B-A3B
====================================

Train Qwen3-30B-A3B-Instruct-2507 to resolve GitHub issues as a software engineering agent, using `examples/agent/harbor/train.yaml`. mini-swe-agent works on each task inside a Docker sandbox managed by Harbor, the task's tests score the final repository state, and the scored trajectories train the model with GRPO.

Overview
--------

Use this recipe to train an MoE model on SWE-Gym Lite tasks on a single 8-GPU node with the Megatron-Bridge actor and SGLang rollout.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen3-30B-A3B-Instruct-2507 (MoE, about 3B active parameters)

   .. grid-item-card:: Algorithm
      :text-align: center

      Multi-turn GRPO with verifier rewards

   .. grid-item-card:: Tools
      :text-align: center

      mini-swe-agent in Harbor Docker sandboxes

   .. grid-item-card:: Hardware
      :text-align: center

      Reference run on 8×A100-80G

Each training step runs three components on the same node. The Harbor task service starts a sandbox for each trial, runs mini-swe-agent and then the task's tests, and removes the sandbox. For every model turn, mini-swe-agent calls a session gateway hosted by ``HarborAgentLoopWorker``; the gateway forwards the request to the SGLang rollout and records the generated token IDs, logprobs, and policy version as sampled. After the tests finish, the worker joins the test score with these tokens into a training sample, so the actor trains on exactly the tokens the policy produced. Text that the agent appends between turns, such as command output, has loss mask 0.

Environment
-----------

RLinf Environment
~~~~~~~~~~~~~~~~~

RLinf environment setup follows :doc:`RLinf Installation </rst_source/start/installation>`. Use the agentic environment with Megatron and SGLang.

Harbor and mini-swe-agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Install into the same Python environment:

1. Harbor from the ``harbor-miles-v0.20.0`` tag (tested commit ``917b3c1be7cf6553a7a9fe25de8adb6f8e639287``), which manages the Docker sandboxes.
2. ``fastapi``, ``uvicorn``, ``httpx``, and ``PyYAML``.

Docker must be available on the node. Install ``examples/agent/harbor/mini_swe_model.py`` as ``minisweagent/models/rlinf_text_model.py`` in the agent environment used inside the sandbox. It provides ``TrajectoryTextModel``, which ``examples/agent/harbor/mini_swe.yaml`` selects: when a reply has no valid action, it keeps the reply in the history so the model sees the format-error feedback, and when the session token budget runs out, it ends the agent with ``LimitsExceeded``.

Data
----

Each training question is a Harbor task directory containing ``task.toml``, ``instruction.md``, ``environment/``, ``tests/``, and ``solution/``. The recipe uses the eight tasks in ``SWE_GYM_LITE_TASKS`` (`examples/agent/harbor/prepare_data.py`), from the ``SWE-Gym/SWE-Gym-Lite`` train split at revision ``f70b1a29ab120eb0a0ee7a1deb029825e735b2b0``. Generate their directories with the SWE-Gym adapter in the Harbor checkout (``adapters/swegym``) and place them under ``/data/prepared-swe``.

The adapter scores a task from per-test pytest results, so set ``PYTEST_ADDOPTS="-v -rA --color=no"`` in each task's ``tests/test.sh``; otherwise passing tests print no names and the task scores 0. Before training, check each task with Harbor's ``oracle`` agent: an empty patch must score 0 and the reference solution must score 1.

Export the training copy:

.. code-block:: bash

   python examples/agent/harbor/prepare_data.py \
     --source-tasks /data/prepared-swe --output-dir /data/swe-training

The script copies the task directories to ``/data/swe-training/tasks`` without ``solution/``, and writes ``/data/swe-training/train.jsonl`` with one question per task under dataset ID ``swe-gym-lite``. To export other tasks, pass ``--instance-ids`` and ``--dataset-id``.

Configuration
-------------

Fill in the model path, the data path, and the Harbor service address:

.. code-block:: yaml

   data:
     train_data_paths: ["/data/swe-training/train.jsonl"]
   rollout:
     model:
       model_path: /path/to/model/Qwen3-30B-A3B-Instruct-2507
   agentloop:
     harbor:
       server_url: http://127.0.0.1:30200

The SWE-specific settings are already set in the configuration:

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

- ``agentloop.agentic_tokenizer`` and ``chat_template_kwargs`` select how the gateway renders the agent's chat history. The rendering must reproduce the previously generated text exactly; otherwise the trial counts as a prefix break and is discarded.
- ``agentloop.harbor.max_concurrency`` caps concurrent sandboxes. Keep the service's ``max_concurrency`` in ``examples/agent/harbor/server.yaml`` at least as large.
- ``trial_timeout_s`` covers queueing, setup, the agent, and the tests. ``client_timeout_s`` must exceed ``trial_timeout_s + cleanup_timeout_s`` so the service can finish cancelling a trial before the client gives up.
- Each agent-loop worker serves its gateway on ``gateway_port + rank``. Sandboxes reach it through ``gateway_advertise_host``, which defaults to the Ray node IP.
- ``rollout.sglang.return_generation_errors: True`` returns a failed generation to the gateway as an error response, so only that trial fails instead of the rollout worker.

Training on 8×A100
------------------

Parallelism on 8×A100-80G (actor and rollout collocated):

- Actor (Megatron): tensor parallel 4, expert parallel 8, expert tensor parallel 1, data parallel 2, with optimizer states offloaded to CPU.
- Rollout (SGLang): one engine across the 8 GPUs with tensor parallel 8, expert parallel 8, and attention data parallel 2.
- Each step samples eight trajectories for each of the eight questions, 64 in total, with at most 32 sandboxes at a time. A trajectory is limited to 32K tokens, 4096 generated tokens per turn, and 80 agent turns.

Start the Harbor service from the repository root. It listens on ``127.0.0.1:30200``; ``server.yaml`` points ``executor.tasks_dir`` to the exported tasks and keeps agent logs and test output in ``executor.trials_dir``.

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/serve.py \
     --config examples/agent/harbor/server.yaml

Then run the training with:

.. code-block:: bash

   PYTHONPATH=. python examples/agent/harbor/train.py \
     --config-path . --config-name train \
     rollout.model.model_path=/path/to/model/Qwen3-30B-A3B-Instruct-2507 \
     data.train_data_paths=[/data/swe-training/train.jsonl] \
     agentloop.harbor.server_url=http://127.0.0.1:30200

To check the whole path before a long run, append ``runner.max_steps=2 runner.save_interval=1``: two steps run and both save a checkpoint.

Run one service process, because the trial registry and concurrency limit live in process memory. To require authentication, set ``RLINF_HARBOR_API_KEY`` before starting the service and set the same value as ``agentloop.harbor.api_key``. The service and gateways do not encrypt traffic, so keep them on a trusted network.

Monitoring and Troubleshooting
------------------------------

Every trial writes a JSON record with its turns and verdict to ``agentloop.harbor.artifacts_dir``. The ``harbor/*`` metrics summarize each step; rates are averaged over all trials in the batch.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Meaning and action
   * - ``harbor/infrastructure_failure``
     - Setup, sandbox, test, transport, or deadline failures. A rising value points to Docker or service problems; inspect ``executor.trials_dir``.
   * - ``harbor/agent_format_failure``
     - Trials that ended because the model repeatedly produced no valid action. They train with reward 0.
   * - ``harbor/reward_fallback``
     - Complete trajectories without a usable test score, trained with reward 0.
   * - ``harbor/invalid_trajectory``
     - Trajectories that cannot train: empty, prefix broken, or sandbox cleanup unconfirmed. They are retried up to ``agentloop.harbor.max_retries`` times.
   * - ``harbor/prefix_break``
     - Turns whose rendered history no longer starts with the recorded tokens, usually because the chat template does not reproduce the model's own output.
   * - ``harbor/timeout``
     - Trials that hit the service or client deadline, or failed with a timeout such as a test timeout.
   * - ``harbor/turns``, ``harbor/trial_elapsed_s``, ``harbor/trial_elapsed_s_p50``, ``harbor/trial_elapsed_s_p95``
     - Model calls per trial and trial duration.
   * - ``harbor/batch_aborted``, ``harbor/aborted_groups``, ``harbor/skipped_batches``
     - A question without ``group_size`` usable trajectories discards the whole batch on every worker; the step counter does not advance.

After ``agentloop.harbor.max_consecutive_aborted_batches`` discarded batches in a row (32 in the configuration), the runner saves a checkpoint and stops with an error, which usually means the Harbor service is unhealthy.

Running Trials without Training
-------------------------------

The Harbor service can also run trials against any OpenAI-compatible endpoint, such as a standalone SGLang server. ``HarborClient`` submits a trial and polls it until it finishes:

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

Give every attempt a unique ``trial_id`` and ``session_id``. ``model_url`` is the OpenAI API root as seen from the sandbox. The result keeps ``reward`` and ``infrastructure_failure`` separate: ``reward`` is the test score whenever the tests produced a finite one, even if the agent also failed, and ``None`` otherwise. ``client.cancel_trial(request)`` cancels one trial and waits for its sandbox to be removed. A result with ``cleanup_confirmed=False`` means a sandbox may still be running and needs manual cleanup.
