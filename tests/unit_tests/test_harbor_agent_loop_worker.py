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

"""Harbor sampling groups retain model failures as trainable trajectories."""

import asyncio
import json

import httpx
import pytest
from omegaconf import OmegaConf
from test_session_gateway import Tokenizer

from rlinf.data.schema.reasoning_requests import SeqGroupInfo
from rlinf.workers.agent.harbor.backend import HarborExecutor
from rlinf.workers.agent.harbor.client import HarborClient
from rlinf.workers.agent.harbor.gateway import SessionGateway
from rlinf.workers.agent.harbor.server import create_app
from rlinf.workers.agent.harbor_agent_loop_worker import HarborAgentLoopWorker


@pytest.mark.parametrize("format_failures", [1, 4])
@pytest.mark.parametrize("verified_success_after_failure", [False, True])
@pytest.mark.parametrize(
    "failure_kind",
    [
        "RepeatedFormatError",
        "VerifierTimeoutError",
        "AgentTimeoutError",
        "RewardFileNotFoundError",
        "NonZeroAgentExitCodeError",
    ],
)
def test_model_format_failures_complete_group_without_resampling(
    tmp_path, format_failures, failure_kind, verified_success_after_failure
):
    async def scenario():
        async def generate(ids, params):
            return {
                "output_ids": [120, 126],
                "logprobs": [-0.2, -0.4],
                "token_versions": [25, 25],
                "finish_reason": "stop",
            }

        worker = object.__new__(HarborAgentLoopWorker)
        worker.cfg = OmegaConf.create(
            {"algorithm": {"sampling_params": {"temperature": 0.7, "top_p": 1.0}}}
        )
        worker.settings = OmegaConf.create(
            {
                "model_name": "openai/policy",
                "trial_timeout_s": 10,
                "max_new_tokens": 10,
                "max_retries": 1,
            }
        )
        worker.gateway = SessionGateway(Tokenizer(), generate, 200, 200)
        worker.gateway_url = "http://gateway"
        worker._artifact_dir = tmp_path / "records"
        worker._artifact_dir.mkdir()
        worker._limit = asyncio.Semaphore(4)
        worker._attempts = set()
        calls = []

        async def execute(req):
            index = len(calls)
            calls.append(req)
            await worker.gateway.complete(
                req.session_id, {"messages": [{"role": "user", "content": "fix"}]}
            )
            trial = tmp_path / req.trial_id
            (trial / "agent").mkdir(parents=True)
            status = failure_kind if index < format_failures else "Submitted"
            (trial / "agent/mini-swe-agent.trajectory.json").write_text(
                json.dumps({"info": {"exit_status": status}})
            )
            payload = {
                "verifier_result": {
                    "rewards": {"reward": 0 if index < format_failures else 1}
                }
            }
            if index < format_failures and failure_kind != "RepeatedFormatError":
                payload = {"exception_info": {"exception_type": failure_kind}}
            if verified_success_after_failure:
                payload["verifier_result"] = {"rewards": {"reward": 1.0}}
            return HarborExecutor._result(req, payload, trial)

        worker._client = HarborClient(
            "http://harbor",
            client_timeout_s=12,
            cleanup_timeout_s=1,
            poll_interval_s=0.001,
            transport=httpx.ASGITransport(create_app(execute)),
        )
        try:
            result, metrics = await worker._run_group(
                SeqGroupInfo(
                    id=0,
                    input_ids=[1],
                    answer={"instance_id": "instance", "dataset_id": "test"},
                    group_size=4,
                )
            )
            assert len(calls) == 4
            assert result.num_sequence == result.group_size == 4
            expected = (
                [1.0] * 4
                if verified_success_after_failure
                else ([0.0] * format_failures + [1.0] * (4 - format_failures))
            )
            assert result.rewards.tolist() == expected
            assert result.response_ids == [[120, 126]] * 4
            assert result.response_mask == [[1, 1]] * 4
            assert result.token_versions == [[25, 25]] * 4
            if failure_kind == "RepeatedFormatError":
                assert (
                    sum(m["agent_format_failure"] for m in metrics) == format_failures
                )
                assert all(m["infrastructure_failure"] == 0 for m in metrics)
            elif not verified_success_after_failure:
                assert sum(m["reward_fallback"] for m in metrics) == format_failures
                assert all(
                    s["reward_source"] == "failure_fallback"
                    for s in result.trajectory_metadata[:format_failures]
                )
            if verified_success_after_failure:
                assert all(
                    s["reward_source"] == "verifier" for s in result.trajectory_metadata
                )
                assert all(m["reward_fallback"] == 0 for m in metrics)
        finally:
            await worker._client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["empty", "prefix", "cleanup", "transport"])
def test_invalid_groups_do_not_publish_partial_batches(tmp_path, failure):
    from rlinf.data.schema.reasoning_requests import RolloutRequest
    from rlinf.workers.agent.harbor.protocol import HarborTrialResult

    async def scenario():
        async def generate(ids, params):
            return {
                "output_ids": [120, 126],
                "logprobs": [-0.2, -0.4],
                "token_versions": [30, 30],
                "finish_reason": "stop",
            }

        worker = object.__new__(HarborAgentLoopWorker)
        worker.cfg = OmegaConf.create(
            {"algorithm": {"sampling_params": {"temperature": 0.7, "top_p": 1.0}}}
        )
        worker.settings = OmegaConf.create(
            {
                "model_name": "openai/policy",
                "trial_timeout_s": 0.3,
                "max_new_tokens": 10,
                "max_retries": 1,
            }
        )
        worker.gateway = SessionGateway(Tokenizer(), generate, 200, 200)
        worker.gateway_url = "http://gateway"
        worker._artifact_dir = tmp_path
        worker._limit = asyncio.Semaphore(8)
        worker._attempts = set()
        worker._pending_results = []
        broken = True
        calls = []

        async def execute(req):
            calls.append(req.instance_id)
            invalid = broken and req.instance_id == "bad"
            if not invalid or failure != "empty":
                await worker.gateway.complete(
                    req.session_id, {"messages": [{"role": "user", "content": "fix"}]}
                )
            if invalid and failure == "prefix":
                worker.gateway.get_session(req.session_id).prefix_broken = True
            return HarborTrialResult(
                req.trial_id,
                req.session_id,
                "completed",
                reward=1,
                cleanup_confirmed=not (invalid and failure == "cleanup"),
            )

        app = create_app(execute)
        base = httpx.ASGITransport(app)

        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                if broken and failure == "transport":
                    raise httpx.ConnectError("injected outage")
                return await base.handle_async_request(req)

        worker._client = HarborClient(
            "http://harbor",
            client_timeout_s=1,
            cleanup_timeout_s=0.1,
            poll_interval_s=0.001,
            transport=Transport(),
        )
        request = RolloutRequest(
            n=4,
            input_ids=[[1], [1]],
            image_data=[[], []],
            answers=[{"instance_id": x, "dataset_id": "test"} for x in ["good", "bad"]],
            multi_modal_inputs=[None, None],
        )
        try:
            summary = await worker._collect_rollout(request)
            assert summary["harbor/batch_aborted"] == 1
            assert not worker._pending_results
            await worker.publish_rollout(None, accept=False)
            broken = False
            summary = await worker._collect_rollout(request)
            assert summary["harbor/batch_aborted"] == 0
            assert len(worker._pending_results) == 2
            assert all(r.num_sequence == 4 for r in worker._pending_results)
            assert not worker.gateway.sessions
            # Another worker may abort: none of this worker's samples reach the actor.
            await worker.publish_rollout(None, accept=False)
            assert not worker._pending_results
        finally:
            await worker._client.close()

    asyncio.run(scenario())
