# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Harbor agent trajectories for synchronous RLinf training."""

import asyncio
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from rlinf.agents.tokenization import get_agentic_tokenizer
from rlinf.data.schema.reasoning_results import RolloutResult
from rlinf.scheduler import Worker
from rlinf.workers.agent.agent_loop import AgentLoopWorker
from rlinf.workers.agent.harbor.client import HarborClient
from rlinf.workers.agent.harbor.gateway import (
    SessionGateway,
    assemble_session,
    create_gateway_app,
)
from rlinf.workers.agent.harbor.protocol import HarborTrialRequest


class HarborAgentLoopWorker(AgentLoopWorker):
    """Run Harbor trials and join verifier rewards with exact generation records."""

    def __init__(self, cfg, placement):
        super().__init__(cfg, placement)
        self.agentic_tokenizer = get_agentic_tokenizer(
            self.tokenizer,
            tokenizer_type=cfg.agentloop.get("agentic_tokenizer", "default"),
            chat_template_kwargs=dict(cfg.agentloop.get("chat_template_kwargs", {})),
        )
        if cfg.rollout.rollout_backend != "sglang" or not cfg.rollout.return_logprobs:
            raise ValueError(
                "Harbor training requires SGLang with return_logprobs=True"
            )
        if cfg.agentloop.get("is_dynamic_rollout_batch", False):
            raise ValueError("Harbor emits single-sequence RolloutResult batches")
        self.settings = cfg.agentloop.harbor
        self._client = None
        self._server = None
        self._server_task = None
        self._attempts = set()
        self._pending_results = []
        self._limit = asyncio.Semaphore(int(self.settings.max_concurrency))
        self.gateway = SessionGateway(
            self.tokenizer,
            self.generate,
            int(cfg.data.max_prompt_length),
            int(cfg.runner.seq_length) - int(cfg.data.max_prompt_length),
            max_new_tokens=int(self.settings.max_new_tokens),
            max_sessions=int(self.settings.max_concurrency),
            chat_template_kwargs=dict(cfg.agentloop.get("chat_template_kwargs", {})),
            render_chat=self.agentic_tokenizer.render_session_chat,
            boundary_token_ids=getattr(
                self.agentic_tokenizer, "session_boundary_token_ids", frozenset()
            ),
        )
        self._artifact_dir = Path(self.settings.artifacts_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    async def init_worker(self, *args, **kwargs):
        import ray
        import uvicorn

        super().init_worker(*args, **kwargs)
        self._client = HarborClient(
            self.settings.server_url,
            client_timeout_s=float(self.settings.client_timeout_s),
            cleanup_timeout_s=float(self.settings.cleanup_timeout_s),
            api_key=self.settings.get("api_key"),
        )
        port = int(self.settings.gateway_port) + self._rank
        host = (
            self.settings.get("gateway_advertise_host")
            or ray.util.get_node_ip_address()
        )
        self.gateway_url = f"http://{host}:{port}"
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_gateway_app(self.gateway),
                host=self.settings.gateway_host,
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        self._server_task = asyncio.create_task(self._server.serve())
        deadline = time.monotonic() + 30
        while not self._server.started:
            if self._server_task.done():
                await self._server_task
                raise RuntimeError("Session gateway exited before startup")
            if time.monotonic() > deadline:
                raise TimeoutError("Session gateway startup timed out")
            await asyncio.sleep(0.05)

    async def _wait_for_trial(self, request, session):
        import httpx

        trial = asyncio.create_task(self._client.run_trial(request))
        broken = asyncio.create_task(session.prefix_break_event.wait())
        try:
            await asyncio.wait((trial, broken), return_when=asyncio.FIRST_COMPLETED)
            if broken.done() and not trial.done():
                try:
                    await self._client.cancel_trial(request)
                except (TimeoutError, httpx.HTTPError):
                    # run_trial owns cancellation retries and cleanup confirmation.
                    pass
            return await trial
        finally:
            for task in (trial, broken):
                if not task.done():
                    task.cancel()
            await asyncio.gather(trial, broken, return_exceptions=True)

    async def _run_attempt(self, metadata):
        async with self._limit:
            session_id = self.gateway.create_session()
            request = HarborTrialRequest(
                trial_id=uuid.uuid4().hex,
                session_id=session_id,
                instance_id=metadata["instance_id"],
                model_url=f"{self.gateway_url}/sessions/{session_id}/v1",
                model=self.settings.model_name,
                timeout_s=float(self.settings.trial_timeout_s),
                sampling_params={
                    "temperature": float(
                        self.cfg.algorithm.sampling_params.temperature
                    ),
                    "top_p": float(self.cfg.algorithm.sampling_params.top_p),
                    "max_tokens": int(self.settings.max_new_tokens),
                },
            )
            session = self.gateway.get_session(session_id)
            verdict = None
            try:
                verdict = await self._wait_for_trial(request, session)
                metrics = {
                    "infrastructure_failure": int(verdict.infrastructure_failure),
                    "agent_format_failure": int(
                        verdict.failure_kind == "model_format_error"
                    ),
                    "timeout": int(
                        verdict.status == "timed_out"
                        or verdict.failure_kind == "client_timeout"
                        or "timeout" in (verdict.failure_kind or "").lower()
                        or verdict.failure_kind == "trial_deadline"
                    ),
                    "prefix_break": int(session.prefix_broken),
                    "turns": len(session.turns),
                    "trial_elapsed_s": verdict.metrics.get("trial_elapsed_s", 0.0),
                }
                if verdict.failure_kind in {"unknown_task", "configuration_error"}:
                    raise ValueError(
                        f"Invalid Harbor configuration: {verdict.failure_kind}"
                    )
                metrics["reward_fallback"] = 0
                metrics["invalid_trajectory"] = 0
                if (
                    not verdict.terminal
                    or not verdict.cleanup_confirmed
                    or session.prefix_broken
                    or not session.turns
                ):
                    metrics["invalid_trajectory"] = 1
                    return None, metrics
                try:
                    sequence = assemble_session(session)
                except ValueError:
                    metrics["invalid_trajectory"] = 1
                    return None, metrics
                verifier = verdict.eval_report.get("verifier_result") or {}
                raw_reward = (verifier.get("rewards") or {}).get("reward")
                fallback = not (
                    isinstance(raw_reward, (int, float))
                    and not isinstance(raw_reward, bool)
                    and math.isfinite(raw_reward)
                )
                metrics["reward_fallback"] = int(fallback)
                sequence["reward"] = 0.0 if fallback else float(raw_reward)
                sequence["metadata"] = {
                    **metadata,
                    **asdict(verdict),
                    "num_turns": len(session.turns),
                    "training_reward": sequence["reward"],
                    "reward_source": "failure_fallback" if fallback else "verifier",
                }
                return sequence, metrics
            finally:
                payload = {
                    "session_id": session_id,
                    "instance_id": metadata["instance_id"],
                    "prefix_broken": session.prefix_broken,
                    "prefix_break_context": session.prefix_break_context,
                    "turns": [asdict(t) for t in session.turns],
                    "verdict": asdict(verdict) if verdict else None,
                }
                (self._artifact_dir / f"{session_id}.json").write_text(
                    json.dumps(payload)
                )
                await self.gateway.close_session(session_id)

    async def _run_group(self, group):
        metadata = group.answer
        if (
            not isinstance(metadata, dict)
            or not {"instance_id", "dataset_id"} <= metadata.keys()
        ):
            raise ValueError(
                "Harbor dataset answer must contain instance_id and dataset_id"
            )
        sequences, metrics = [], []
        for _ in range(int(self.settings.max_retries) + 1):
            attempts = [
                asyncio.create_task(self._run_attempt(metadata))
                for _ in range(group.group_size - len(sequences))
            ]
            self._attempts.update(attempts)
            try:
                outcomes = await asyncio.gather(*attempts)
            finally:
                for task in attempts:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await asyncio.gather(*attempts, return_exceptions=True)
                self._attempts.difference_update(attempts)
            for sequence, attempt_metrics in outcomes:
                metrics.append(attempt_metrics)
                if sequence is not None:
                    sequences.append(sequence)
            if len(sequences) == group.group_size:
                break
        if len(sequences) != group.group_size:
            return None, metrics
        result = RolloutResult(
            num_sequence=group.group_size,
            group_size=group.group_size,
            prompt_ids=[s["prompt_ids"] for s in sequences],
            prompt_lengths=[len(s["prompt_ids"]) for s in sequences],
            response_ids=[s["response_ids"] for s in sequences],
            response_lengths=[len(s["response_ids"]) for s in sequences],
            response_mask=[s["response_mask"] for s in sequences],
            rollout_logprobs=[s["rollout_logprobs"] for s in sequences],
            token_versions=[s["token_versions"] for s in sequences],
            trajectory_metadata=[s["metadata"] for s in sequences],
            rewards=torch.tensor([s["reward"] for s in sequences], dtype=torch.float32),
            answers=[metadata] * group.group_size,
            is_end=[True] * group.group_size,
        )
        return result, metrics

    @Worker.timer("run_agentloop_rollout")
    async def run_agentloop_rollout(self, input_channel, output_channel):
        request = await input_channel.get(async_op=True).async_wait()
        return await self._collect_rollout(request)

    async def _collect_rollout(self, request):
        tasks = [
            asyncio.create_task(self._run_group(g))
            for g in request.to_seq_group_infos()
        ]
        try:
            groups = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        metrics = [m for _, attempts in groups for m in attempts]
        aborted = any(result is None for result, _ in groups)
        self._pending_results = [] if aborted else [result for result, _ in groups]
        if not metrics:
            return {}
        summary = {
            f"harbor/{key}": sum(m[key] for m in metrics) / len(metrics)
            for key in (
                "infrastructure_failure",
                "agent_format_failure",
                "timeout",
                "prefix_break",
                "turns",
                "trial_elapsed_s",
                "reward_fallback",
                "invalid_trajectory",
            )
        }
        summary["harbor/batch_aborted"] = int(aborted)
        summary["harbor/aborted_groups"] = sum(result is None for result, _ in groups)
        durations = sorted(m["trial_elapsed_s"] for m in metrics)
        for percentile in (50, 95):
            summary[f"harbor/trial_elapsed_s_p{percentile}"] = durations[
                math.ceil(len(durations) * percentile / 100) - 1
            ]
        return summary

    async def publish_rollout(self, output_channel, accept: bool):
        """Publish only after every worker has collected a complete batch."""
        results, self._pending_results = self._pending_results, []
        if accept:
            for result in results:
                await output_channel.put(result, async_op=True).async_wait()

    async def shutdown(self):
        tasks = list(self._attempts)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.close()
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            await self._server_task
