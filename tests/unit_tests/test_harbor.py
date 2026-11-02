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

"""Harbor trial lifecycle, session token alignment, and training rewards."""

import asyncio
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.agents.tokenization import get_agentic_tokenizer
from rlinf.data.schema.reasoning_requests import RolloutRequest, SeqGroupInfo
from rlinf.workers.agent.harbor.backend import HarborExecutor
from rlinf.workers.agent.harbor.client import HarborClient
from rlinf.workers.agent.harbor.gateway import (
    SessionGateway,
    assemble_session,
    create_gateway_app,
)
from rlinf.workers.agent.harbor.protocol import HarborTrialRequest, HarborTrialResult
from rlinf.workers.agent.harbor.server import create_app
from rlinf.workers.agent.harbor_agent_loop_worker import HarborAgentLoopWorker


@pytest.fixture
def httpx():
    pytest.importorskip("fastapi")
    return pytest.importorskip("httpx")


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        text = "".join(chr(i) for i in ids)
        return text.replace("~", "") if skip_special_tokens else text

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join(m["role"] + ":" + m["content"] + "~" for m in messages)
            + "assistant:"
        )


def request(name="trial", timeout=0.3):
    return HarborTrialRequest(
        name,
        name,
        "instance",
        "http://gateway/session/v1",
        "openai/model",
        timeout_s=timeout,
    )


def client(app, httpx):
    return HarborClient(
        "http://harbor",
        client_timeout_s=1,
        cleanup_timeout_s=0.1,
        poll_interval_s=0.001,
        transport=httpx.ASGITransport(app),
    )


def test_result_and_idempotent_submission(httpx):
    async def scenario():
        calls = []

        async def execute(req):
            calls.append(req)
            return HarborTrialResult(
                req.trial_id,
                req.session_id,
                "completed",
                reward=0,
                cleanup_confirmed=True,
            )

        c = client(create_app(execute), httpx)
        try:
            result = await c.run_trial(request())
            repeated = await c.run_trial(request())
            assert result.reward == repeated.reward == 0
            assert not result.infrastructure_failure
            assert len(calls) == 1
        finally:
            await c.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_client", [False, True])
def test_scoped_cancellation_cleans_only_owned_trial(cancel_client, httpx):
    async def scenario():
        entered = {name: asyncio.Event() for name in ("a", "b")}
        release = asyncio.Event()
        cleaned = set()

        async def execute(req):
            entered[req.trial_id].set()
            try:
                await release.wait()
                return HarborTrialResult(
                    req.trial_id,
                    req.session_id,
                    "completed",
                    reward=1,
                    cleanup_confirmed=True,
                )
            finally:
                await asyncio.sleep(0.005)
                cleaned.add(req.trial_id)

        c = client(create_app(execute), httpx)
        a = asyncio.create_task(c.run_trial(request("a")))
        b = asyncio.create_task(c.run_trial(request("b")))
        await asyncio.gather(*(e.wait() for e in entered.values()))
        if cancel_client:
            a.cancel()
            with pytest.raises(asyncio.CancelledError):
                await a
        else:
            outcomes = await asyncio.gather(
                c.cancel_trial(request("a")), c.cancel_trial(request("a"))
            )
            assert all(r.cleanup_confirmed for r in outcomes)
            assert (await a).status == "cancelled"
        assert cleaned == {"a"}
        release.set()
        assert (await b).reward == 1
        await c.close()

    asyncio.run(scenario())


def test_server_deadline_and_cleanup_failure(httpx):
    async def scenario():
        async def execute(req):
            try:
                await asyncio.sleep(10)
            finally:
                raise RuntimeError("Docker daemon unavailable during cleanup")

        c = client(create_app(execute, cleanup_timeout_s=0.02), httpx)
        result = await c.run_trial(request(timeout=0.01))
        assert result.status == "timed_out"
        assert result.infrastructure_failure and result.reward is None
        assert not result.cleanup_confirmed
        assert result.failure_kind == "cleanup_failed"
        await c.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "exception,reward,infra",
    [
        (None, 0, False),
        (None, 1, False),
        (None, None, True),
        ("EnvironmentStartTimeoutError", 0, True),
        ("RewardFileNotFoundError", None, True),
        ("AgentTimeoutError", 0, False),
    ],
)
def test_verifier_failure_classification(tmp_path, exception, reward, infra):
    payload = {
        "exception_info": {"exception_type": exception},
        "verifier_result": {"rewards": {"reward": reward}},
    }
    result = HarborExecutor._result(request(), payload, tmp_path)
    assert result.infrastructure_failure is infra
    assert result.reward == reward


def test_multiturn_http_and_training_masks(httpx):
    async def scenario():
        calls = []

        async def generate(ids, params):
            calls.append(ids.copy())
            return {
                "output_ids": [ord("x"), ord("~")],
                "logprobs": [-0.2, -0.4],
                "token_versions": [7, 7],
                "finish_reason": "stop",
            }

        tokenizer = Tokenizer()
        gateway = SessionGateway(
            tokenizer,
            generate,
            200,
            200,
            render_chat=get_agentic_tokenizer(tokenizer).render_session_chat,
        )
        sid = gateway.create_session()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(create_gateway_app(gateway)),
            base_url="http://gateway",
        )
        messages = [{"role": "user", "content": "fix"}]
        response = await client.post(
            f"/sessions/{sid}/v1/chat/completions", json={"messages": messages}
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "x"
        messages += [
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "tool output"},
        ]
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Session-ID": sid},
            json={"messages": messages},
        )
        assert response.status_code == 200
        sequence = assemble_session(gateway.get_session(sid))
        assert calls[1][: len(calls[0]) + 2] == calls[0] + [120, 126]
        assert sequence["prompt_ids"] + sequence["response_ids"] == calls[1] + [
            120,
            126,
        ]
        assert [
            t for t, m in zip(sequence["response_ids"], sequence["response_mask"]) if m
        ] == [120, 126, 120, 126]
        assert [
            p
            for p, m in zip(sequence["rollout_logprobs"], sequence["response_mask"])
            if m
        ] == [-0.2, -0.4, -0.2, -0.4]
        assert [
            v
            for v, m in zip(sequence["token_versions"], sequence["response_mask"])
            if m
        ] == [7] * 4
        assert all(
            v == -1
            for v, m in zip(sequence["token_versions"], sequence["response_mask"])
            if not m
        )
        records = (await client.get(f"/sessions/{sid}")).json()
        assert len(records["turns"]) == 2
        assert (await client.delete(f"/sessions/{sid}")).status_code == 200
        assert not gateway.sessions
        await client.aclose()

    asyncio.run(scenario())


def test_prefix_break_is_recorded_and_cannot_train(httpx):
    async def scenario():
        async def generate(ids, params):
            return {
                "output_ids": [120, 126],
                "logprobs": [-1, -1],
                "token_versions": [0, 0],
                "finish_reason": "stop",
            }

        gateway = SessionGateway(Tokenizer(), generate, 200, 200)
        sid = gateway.create_session()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_gateway_app(gateway)),
            base_url="http://gateway",
        ) as client:
            url = f"/sessions/{sid}/v1/chat/completions"
            await client.post(
                url, json={"messages": [{"role": "user", "content": "old"}]}
            )
            result = await client.post(
                url, json={"messages": [{"role": "user", "content": "changed"}]}
            )
            assert result.status_code == 409
            assert gateway.get_session(sid).prefix_break_event.is_set()
            assert (
                gateway.get_session(sid).prefix_break_context["character_offset"] >= 0
            )
            with pytest.raises(ValueError, match="prefix-broken"):
                assemble_session(gateway.get_session(sid))

    asyncio.run(scenario())


def test_generated_tokens_are_not_reencoded():
    class AlternativeEncodingTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            expanded = [
                c for token in ids for c in ([120, 121] if token == 1000 else [token])
            ]
            return super().decode(expanded, **kwargs)

    async def scenario():
        calls = []

        async def generate(ids, params):
            calls.append(ids.copy())
            return {
                "output_ids": [1000, 126],
                "logprobs": [-1, -1],
                "token_versions": [0, 0],
                "finish_reason": "stop",
            }

        gateway = SessionGateway(AlternativeEncodingTokenizer(), generate, 200, 200)
        sid = gateway.create_session()
        messages = [{"role": "user", "content": "fix"}]
        first = await gateway.complete(sid, {"messages": messages})
        messages += [
            first["choices"][0]["message"],
            {"role": "user", "content": "continue"},
        ]
        await gateway.complete(sid, {"messages": messages})
        assert calls[1][len(calls[0]) : len(calls[0]) + 2] == [1000, 126]
        sequence = assemble_session(gateway.get_session(sid))
        assert sequence["prompt_ids"] + sequence["response_ids"] == calls[1] + [
            1000,
            126,
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_kind,format_failures,verified_success_after_failure",
    [
        ("RepeatedFormatError", 1, False),
        ("RepeatedFormatError", 4, False),
        ("RepeatedFormatError", 1, True),
        ("VerifierTimeoutError", 1, False),
        ("VerifierTimeoutError", 1, True),
        ("AgentTimeoutError", 1, False),
        ("RewardFileNotFoundError", 1, False),
        ("NonZeroAgentExitCodeError", 1, False),
    ],
)
def test_model_format_failures_complete_group_without_resampling(
    tmp_path, format_failures, failure_kind, verified_success_after_failure, httpx
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
            {
                "algorithm": {
                    "sampling_params": {
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "max_new_tokens": 10,
                    }
                }
            }
        )
        worker.settings = OmegaConf.create(
            {
                "model_name": "openai/policy",
                "trial_timeout_s": 10,
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
            if index < format_failures:
                payload = {"exception_info": {"exception_type": failure_kind}}
            if verified_success_after_failure:
                payload["verifier_result"] = {"rewards": {"reward": 1.0}}
            # Close through HTTP before the worker releases the session.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(create_gateway_app(worker.gateway)),
                base_url="http://gateway",
            ) as gateway_client:
                response = await gateway_client.delete(f"/sessions/{req.session_id}")
                assert response.status_code == 200
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
                assert all(m["reward_fallback"] == 0 for m in metrics)
                if not verified_success_after_failure:
                    assert all(
                        record["reward_source"] == "model_format_error"
                        for record in result.trajectory_metadata[:format_failures]
                    )
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


@pytest.mark.parametrize(
    "change,message",
    [
        ({"output_ids": []}, "Expected 1..2 output tokens, got 0"),
        ({"output_ids": [1, 2, 3]}, "Expected 1..2 output tokens, got 3"),
        ({"logprobs": []}, "Expected 2 logprobs entries for output tokens, got 0"),
        (
            {"token_versions": [0]},
            "Expected 2 token_versions entries for output tokens, got 1",
        ),
        (
            {"logprobs": [-0.2, float("nan")]},
            "Expected logprobs[1] to be finite, got nan",
        ),
        (
            {"token_versions": [0, -1]},
            "Expected token_versions[1] to be a nonnegative integer, got -1",
        ),
        (
            {"finish_reason": "abort"},
            "Expected finish_reason in ['length', 'stop'], got 'abort'",
        ),
    ],
)
def test_invalid_generation_does_not_commit_a_turn(change, message):
    async def scenario():
        valid = {
            "output_ids": [120, 126],
            "logprobs": [-0.2, -0.4],
            "token_versions": [0, 0],
            "finish_reason": "stop",
        }
        responses = iter([valid | change, valid])

        async def generate(ids, params):
            return next(responses)

        gateway = SessionGateway(Tokenizer(), generate, 200, 200, max_new_tokens=2)
        sid = gateway.create_session()
        body = {"messages": [{"role": "user", "content": "fix"}]}
        with pytest.raises(ValueError) as caught:
            await gateway.complete(sid, body)
        assert str(caught.value) == message
        session = gateway.get_session(sid)
        assert not session.turns and not session.token_ids
        await gateway.complete(sid, body)
        assert assemble_session(session)["response_ids"] == [120, 126]

    asyncio.run(scenario())


@pytest.mark.parametrize("raw_reward", [True, "1", float("nan"), float("inf")])
def test_invalid_verifier_score_has_no_training_reward(tmp_path, raw_reward):
    result = HarborExecutor._result(
        request(), {"verifier_result": {"rewards": {"reward": raw_reward}}}, tmp_path
    )
    assert result.reward is None
    assert result.infrastructure_failure
    assert result.failure_kind == "missing_reward"


def test_harbor_config_uses_only_selected_backend_and_one_token_limit():
    from rlinf.config import validate_harbor_cfg, validate_reasoning_cfg

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2] / "examples/agent/harbor/train.yaml"
    )
    cfg = validate_harbor_cfg(validate_reasoning_cfg(cfg))
    assert cfg.data.type == "reasoning"
    assert not {"reward", "inference"} & cfg.keys()
    assert "vllm" not in cfg.rollout
    assert "max_new_tokens" not in cfg.agentloop.harbor
    assert cfg.rollout.sglang.return_generation_errors is True
    cfg.algorithm.recompute_logprobs = True
    with pytest.raises(
        ValueError, match="algorithm.recompute_logprobs=False, got True"
    ):
        validate_harbor_cfg(cfg)


def test_concurrent_session_deletion_waits_for_generation_and_is_idempotent(httpx):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(ids, params):
            entered.set()
            await release.wait()
            return {
                "output_ids": [120],
                "logprobs": [-0.2],
                "token_versions": [0],
                "finish_reason": "stop",
            }

        gateway = SessionGateway(Tokenizer(), generate, 200, 200)
        sid = gateway.create_session()
        session = gateway.get_session(sid)
        completion = asyncio.create_task(
            gateway.complete(sid, {"messages": [{"role": "user", "content": "fix"}]})
        )
        await entered.wait()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_gateway_app(gateway)),
            base_url="http://gateway",
        ) as client:
            first = asyncio.create_task(client.delete(f"/sessions/{sid}"))
            second = asyncio.create_task(client.delete(f"/sessions/{sid}"))
            await asyncio.sleep(0)
            assert not first.done() and not second.done()
            release.set()
            await completion
            responses = await asyncio.gather(first, second)
            responses.append(await client.delete(f"/sessions/{sid}"))
            assert all(
                r.status_code == 200 and r.json() == {"closed": True} for r in responses
            )
        await gateway.close_session(sid)
        assert not gateway.sessions
        assert assemble_session(session)["response_ids"] == [120]

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled", [None, False, True])
def test_main_rollout_generation_error_policy(enabled):
    from rlinf.config import validate_rollout_cfg

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2] / "examples/agent/harbor/train.yaml"
    )
    if enabled is None:
        del cfg.rollout.sglang.return_generation_errors
    else:
        cfg.rollout.sglang.return_generation_errors = enabled
    OmegaConf.set_struct(cfg, True)
    validate_rollout_cfg(cfg.rollout, cfg.algorithm, cfg.actor)
    assert cfg.rollout.sglang.return_generation_errors is (enabled is True)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_main_rollout_rejects_nonboolean_generation_error_policy(value):
    from rlinf.config import validate_rollout_cfg

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2] / "examples/agent/harbor/train.yaml"
    )
    cfg.rollout.sglang.return_generation_errors = value
    with pytest.raises(ValueError) as caught:
        validate_rollout_cfg(cfg.rollout, cfg.algorithm, cfg.actor)
    assert str(caught.value) == (
        f"sglang.return_generation_errors must be a bool, got {value!r}"
    )


@pytest.mark.parametrize("failure", ["empty", "prefix", "cleanup", "transport"])
def test_invalid_groups_do_not_publish_partial_batches(tmp_path, failure, httpx):
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
            {
                "algorithm": {
                    "sampling_params": {
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "max_new_tokens": 10,
                    }
                }
            }
        )
        worker.settings = OmegaConf.create(
            {
                "model_name": "openai/policy",
                "trial_timeout_s": 0.3,
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
            # Apply the runner's rejection after this worker collected complete groups.
            await worker.publish_rollout(None, accept=False)
            assert not worker._pending_results
        finally:
            await worker._client.close()

    asyncio.run(scenario())


@pytest.fixture
def mini_swe():
    import importlib.util
    from types import SimpleNamespace

    pytest.importorskip("minisweagent")
    response = pytest.importorskip("litellm").ModelResponse
    agent = pytest.importorskip("minisweagent.agents.default").DefaultAgent
    path = (
        Path(__file__).resolve().parents[2] / "examples/agent/harbor/mini_swe_model.py"
    )
    spec = importlib.util.spec_from_file_location("mini_swe_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, ModelResponse=response, DefaultAgent=agent)


@pytest.mark.parametrize("retry_valid", [True, False])
def test_format_error_keeps_original_response_and_feedback(
    mini_swe, tmp_path, retry_valid
):
    original = "  <think>inspect the code</think>\nMissing a command.  \n"
    calls = []

    class Model(mini_swe.module.TrajectoryTextModel):
        def _query(self, messages, **kwargs):
            calls.append(messages)
            content = (
                original
                if len(calls) == 1 or not retry_valid
                else "```mswea_bash_command\necho ok\n```"
            )
            return mini_swe.ModelResponse(
                choices=[{"message": {"role": "assistant", "content": content}}]
            )

        def _calculate_cost(self, response):
            return {"cost": 0.25}

    class Environment:
        def get_template_vars(self):
            return {}

        def execute(self, action):
            return {"output": "ok", "returncode": 0}

        def serialize(self):
            return {}

    model = Model(
        model_name="openai/test",
        cost_tracking="ignore_errors",
        observation_template="{{output.output}}",
    )
    agent = mini_swe.DefaultAgent(
        model,
        Environment(),
        system_template="Execute a command.",
        instance_template="{{task}}",
        step_limit=3 if not retry_valid else 2,
        max_consecutive_format_errors=2,
        output_path=tmp_path / "trajectory.json",
    )
    result = agent.run("Fix the issue")
    assert result["exit_status"] == (
        "LimitsExceeded" if retry_valid else "RepeatedFormatError"
    )
    assert len(calls) == 2
    assert [m["role"] for m in calls[1]] == ["system", "user", "assistant", "user"]
    assert calls[1][2]["content"] == original
    assert calls[1][3]["content"]
    assert agent.cost == 0.5
    assert agent.messages[2]["content"] == original


def test_session_budget_stops_agent_without_retry(mini_swe, tmp_path, monkeypatch):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        raise mini_swe.module.BadRequestError(
            message="Error code: 400 - Session token budget exhausted",
            model="openai/test",
            llm_provider="openai",
        )

    monkeypatch.setattr("litellm.completion", complete)

    class Environment:
        def get_template_vars(self):
            return {}

        def serialize(self):
            return {}

    model = mini_swe.module.TrajectoryTextModel(
        model_name="openai/test", cost_tracking="ignore_errors"
    )
    agent = mini_swe.DefaultAgent(
        model,
        Environment(),
        system_template="Execute a command.",
        instance_template="{{task}}",
        step_limit=40,
        output_path=tmp_path / "trajectory.json",
    )

    result = agent.run("Fix the issue")

    assert result["exit_status"] == "LimitsExceeded"
    assert len(calls) == 1
    assert not any(message["role"] == "assistant" for message in agent.messages)


def test_other_bad_request_is_not_a_budget_limit(mini_swe, monkeypatch):
    error = mini_swe.module.BadRequestError(
        message="Unsupported parameter", model="openai/test", llm_provider="openai"
    )

    def complete(**kwargs):
        raise error

    monkeypatch.setattr("litellm.completion", complete)
    model = mini_swe.module.TrajectoryTextModel(
        model_name="openai/test", cost_tracking="ignore_errors"
    )
    with pytest.raises(mini_swe.module.BadRequestError) as caught:
        model._query([{"role": "user", "content": "Fix the issue"}])
    assert caught.value is error


def context_error(mini_swe, input_tokens, completion_tokens=4096):
    return mini_swe.module.BadRequestError(
        message=(
            "Requested token count exceeds the model's maximum context length of "
            f"32768 tokens. You requested a total of {input_tokens + completion_tokens} "
            f"tokens: {input_tokens} tokens from the input messages and "
            f"{completion_tokens} tokens for the completion."
        ),
        model="openai/test",
        llm_provider="openai",
    )


@pytest.mark.parametrize("token_key", ["max_tokens", "max_completion_tokens"])
def test_context_limit_retries_with_remaining_generation_budget(
    mini_swe, monkeypatch, token_key
):
    calls = []
    response = mini_swe.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "done"}}]
    )

    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise context_error(mini_swe, 28949)
        return response

    monkeypatch.setattr("litellm.completion", complete)
    model = mini_swe.module.TrajectoryTextModel(
        model_name="openai/test",
        model_kwargs={token_key: 4096},
        cost_tracking="ignore_errors",
    )
    messages = [{"role": "user", "content": "Continue"}]
    assert model._query(messages, temperature=0.7) is response
    assert len(calls) == 2
    assert calls[0][token_key] == 4096
    assert calls[1][token_key] == 3819
    assert calls[1]["messages"] == messages
    assert calls[1]["temperature"] == 0.7
    assert model.config.model_kwargs[token_key] == 4096


@pytest.mark.parametrize("input_tokens", [32768, 33000])
@pytest.mark.parametrize("input_only_error", [True, False])
def test_context_exhaustion_ends_without_retry(
    mini_swe, monkeypatch, input_tokens, input_only_error
):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if input_only_error:
            raise mini_swe.module.BadRequestError(
                message=(
                    f"The input ({input_tokens} tokens) is longer than the "
                    "model's context length (32768 tokens)."
                ),
                model="openai/test",
                llm_provider="openai",
            )
        raise context_error(mini_swe, input_tokens)

    monkeypatch.setattr("litellm.completion", complete)
    model = mini_swe.module.TrajectoryTextModel(
        model_name="openai/test", cost_tracking="ignore_errors"
    )
    with pytest.raises(mini_swe.module.LimitsExceeded):
        model._query([{"role": "user", "content": "Continue"}])
    assert len(calls) == 1


@pytest.mark.parametrize("boundary", ["<|observation|>", "<|user|>"])
def test_glm_feedback_trims_boundary_and_aligns_training_fields(boundary):
    import re

    from rlinf.agents.tokenization.agentic_tokenizer import GLMAgenticTokenizer
    from rlinf.workers.agent.harbor.gateway import PrefixBreakError

    class GLMTokenizer(Tokenizer):
        specials = {"<|observation|>": 1000, "<|user|>": 1001}

        def encode(self, text, **kwargs):
            ids = []
            for part in re.split(r"(<\|observation\|>|<\|user\|>)", text):
                if part in self.specials:
                    ids.append(self.specials[part])
                else:
                    ids.extend(map(ord, part))
            return ids

        def decode(self, ids, skip_special_tokens=False, **kwargs):
            inverse = {value: key for key, value in self.specials.items()}
            return "".join(
                ("" if skip_special_tokens else inverse[token])
                if token in inverse
                else chr(token)
                for token in ids
            )

        def apply_chat_template(self, messages, **kwargs):
            return (
                "[gMASK]<sop>"
                + "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
                + "<|assistant|></think>"
            )

    async def scenario():
        tokenizer = GLMTokenizer()
        renderer = object.__new__(GLMAgenticTokenizer)
        renderer.tokenizer = tokenizer
        renderer.chat_template_kwargs = {"enable_thinking": False}
        calls = []
        first_output = tokenizer.encode("bad" + boundary)
        last_output = tokenizer.encode("x" + boundary)

        async def generate(ids, params):
            calls.append(ids.copy())
            output = first_output if len(calls) == 1 else last_output
            return {
                "output_ids": output,
                "logprobs": [-0.1 * len(calls)] * len(output),
                "token_versions": [len(calls)] * len(output),
                "finish_reason": "stop",
            }

        gateway = SessionGateway(
            tokenizer,
            generate,
            200,
            200,
            render_chat=renderer.render_session_chat,
            boundary_token_ids=frozenset(tokenizer.specials.values()),
        )
        sid = gateway.create_session()
        messages = [{"role": "user", "content": "task"}]
        reply = await gateway.complete(sid, {"messages": messages})
        messages += [
            reply["choices"][0]["message"],
            {"role": "user", "content": "format error"},
        ]
        await gateway.complete(sid, {"messages": messages})
        session = gateway.get_session(sid)
        assert session.turns[0].output_ids == first_output
        assert session.turns[0].trimmed_output_tokens == 1
        assert session.turns[1].trimmed_output_tokens == 0
        assert calls[1] == tokenizer.encode(
            "[gMASK]<sop><|user|>task<|assistant|></think>bad"
            "<|user|>format error<|assistant|></think>"
        )
        sequence = assemble_session(session)
        assert (
            sequence["prompt_ids"] + sequence["response_ids"] == calls[1] + last_output
        )
        fields = ["response_ids", "response_mask", "rollout_logprobs", "token_versions"]
        assert len({len(sequence[field]) for field in fields}) == 1
        generated = [
            (token, logprob, version)
            for token, mask, logprob, version in zip(
                *(sequence[field] for field in fields)
            )
            if mask
        ]
        assert generated == [(token, -0.1, 1) for token in first_output[:-1]] + [
            (token, -0.2, 2) for token in last_output
        ]
        messages[1]["content"] = "changed"
        messages += [
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "next"},
        ]
        with pytest.raises(PrefixBreakError):
            await gateway.complete(sid, {"messages": messages})
        assert len(calls) == 2

    asyncio.run(scenario())
