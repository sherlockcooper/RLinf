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

"""Exercise the HTTP lifecycle against the real in-process service."""

import asyncio
import json
from dataclasses import asdict, replace

import pytest

from rlinf.workers.agent.harbor.backend import HarborExecutor
from rlinf.workers.agent.harbor.client import HarborClient
from rlinf.workers.agent.harbor.protocol import HarborTrialRequest, HarborTrialResult
from rlinf.workers.agent.harbor.server import create_app

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")


def request(name="trial", timeout=0.3):
    return HarborTrialRequest(
        name,
        name,
        "instance",
        "http://gateway/session/v1",
        "openai/model",
        timeout_s=timeout,
    )


def client(app):
    return HarborClient(
        "http://harbor",
        client_timeout_s=1,
        cleanup_timeout_s=0.1,
        poll_interval_s=0.001,
        transport=httpx.ASGITransport(app),
    )


def test_result_and_idempotent_submission():
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

        c = client(create_app(execute))
        try:
            result = await c.run_trial(request())
            repeated = await c.run_trial(request())
            assert result.reward == repeated.reward == 0
            assert not result.infrastructure_failure
            assert len(calls) == 1
        finally:
            await c.close()

    asyncio.run(scenario())


def test_conflict_does_not_cancel_existing_trial():
    async def scenario():
        running = asyncio.Event()
        release = asyncio.Event()

        async def execute(req):
            running.set()
            await release.wait()
            return HarborTrialResult(
                req.trial_id,
                req.session_id,
                "completed",
                reward=1,
                cleanup_confirmed=True,
            )

        app = create_app(execute)
        c, other = client(app), client(app)
        task = asyncio.create_task(c.run_trial(request()))
        await running.wait()
        conflict = await other.run_trial(replace(request(), session_id="different"))
        assert conflict.infrastructure_failure
        with pytest.raises(httpx.HTTPStatusError) as error:
            await other.cancel_trial(replace(request(), session_id="different"))
        assert error.value.response.status_code == 409
        release.set()
        assert (await task).reward == 1
        await c.close()
        await other.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_client", [False, True])
def test_scoped_cancellation_cleans_only_owned_trial(cancel_client):
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

        c = client(create_app(execute))
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


def test_server_deadline_and_cleanup_failure():
    async def scenario():
        async def execute(req):
            try:
                await asyncio.sleep(10)
            finally:
                raise RuntimeError("Docker daemon unavailable during cleanup")

        c = client(create_app(execute, cleanup_timeout_s=0.02))
        result = await c.run_trial(request(timeout=0.01))
        assert result.status == "timed_out"
        assert result.infrastructure_failure and result.reward is None
        assert not result.cleanup_confirmed
        assert result.failure_kind == "cleanup_failed"
        await c.close()

    asyncio.run(scenario())


def test_queued_cancel_and_shutdown():
    async def scenario():
        entered = asyncio.Event()
        cleaned = []

        async def execute(req):
            entered.set()
            try:
                await asyncio.sleep(10)
            finally:
                cleaned.append(req.trial_id)

        app = create_app(execute, max_concurrency=1)
        async with app.router.lifespan_context(app):
            c = client(app)
            await c.http.post("/v1/trials", json=asdict(request("a")))
            await entered.wait()
            await c.http.post("/v1/trials", json=asdict(request("b")))
            result = await c.cancel_trial(request("b"))
            assert result.status == "cancelled" and result.cleanup_confirmed
            assert cleaned == []
        assert cleaned == ["a"]
        await c.close()

    asyncio.run(scenario())


def test_timeout_validation_and_authentication():
    async def scenario():
        async def execute(req):
            raise AssertionError("Rejected request must not execute")

        c = client(create_app(execute, api_key="secret"))
        with pytest.raises(ValueError, match="cleanup allowance"):
            await c.run_trial(request(timeout=1))
        result = await c.run_trial(request())
        assert result.infrastructure_failure
        await c.close()

    asyncio.run(scenario())


def test_transport_failure_still_cancels_accepted_trial():
    async def scenario():
        cleaned = asyncio.Event()

        async def execute(req):
            try:
                await asyncio.sleep(10)
            finally:
                cleaned.set()

        transport = httpx.ASGITransport(create_app(execute))

        class DroppedPoll(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                if req.method == "GET":
                    raise httpx.ReadTimeout("Connection lost")
                return await transport.handle_async_request(req)

        c = HarborClient(
            "http://harbor",
            client_timeout_s=1,
            cleanup_timeout_s=0.1,
            poll_interval_s=0.001,
            transport=DroppedPoll(),
        )
        result = await c.run_trial(request())
        assert result.infrastructure_failure and result.reward is None
        assert result.cleanup_confirmed and cleaned.is_set()
        await c.close()

    asyncio.run(scenario())


def test_cancel_during_submission_waits_for_acceptance():
    async def scenario():
        submitting, release = asyncio.Event(), asyncio.Event()

        async def execute(req):
            await asyncio.sleep(10)

        transport = httpx.ASGITransport(create_app(execute))

        class DelayedSubmission(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                if req.method == "POST":
                    submitting.set()
                    await release.wait()
                return await transport.handle_async_request(req)

        c = HarborClient(
            "http://harbor",
            client_timeout_s=1,
            cleanup_timeout_s=0.1,
            poll_interval_s=0.001,
            transport=DelayedSubmission(),
        )
        task = asyncio.create_task(c.run_trial(request()))
        await submitting.wait()
        cancel = asyncio.create_task(c.cancel_trial(request()))
        await asyncio.sleep(0.005)
        assert not cancel.done()
        release.set()
        assert (await cancel).cleanup_confirmed
        assert (await task).status == "cancelled"
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
    assert result.reward == (None if infra else reward)


@pytest.mark.parametrize("exit_status", ["FormatError", "RepeatedFormatError"])
@pytest.mark.parametrize("reward", [None, 0, 1])
@pytest.mark.parametrize("exception", [None, "NonZeroAgentExitCodeError"])
def test_model_format_error_is_a_zero_reward_sample(
    tmp_path, exit_status, reward, exception
):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps({"info": {"exit_status": exit_status}})
    )
    result = HarborExecutor._result(
        request(),
        {
            "exception_info": {"exception_type": exception},
            "verifier_result": {"rewards": {"reward": reward}},
        },
        tmp_path,
    )
    assert result.status == "completed"
    assert result.reward == 0.0
    assert not result.infrastructure_failure
    assert result.failure_kind == "model_format_error"
    assert result.exit_status == exit_status
    assert result.cleanup_confirmed


@pytest.mark.parametrize(
    "exception", ["VerifierTimeoutError", "RewardFileNotFoundError", "HTTPStatusError"]
)
def test_model_format_status_does_not_hide_infrastructure_failure(tmp_path, exception):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps({"info": {"exit_status": "RepeatedFormatError"}})
    )
    result = HarborExecutor._result(
        request(),
        {"exception_info": {"exception_type": exception}},
        tmp_path,
    )
    assert result.infrastructure_failure
    assert result.reward is None
    assert result.failure_kind == exception


@pytest.mark.parametrize(
    "method,code,kind",
    [
        ("GET", 404, "transport_or_protocol"),
        ("GET", 503, "transport_or_protocol"),
        ("POST", 401, "configuration_error"),
        ("POST", 422, "configuration_error"),
    ],
)
def test_runtime_http_failure_is_distinct_from_bad_configuration(method, code, kind):
    async def scenario():
        async def execute(req):
            await asyncio.sleep(10)

        transport = httpx.ASGITransport(create_app(execute))

        class FailedResponse(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                if req.method == method:
                    return httpx.Response(code, json={"detail": "injected failure"})
                return await transport.handle_async_request(req)

        c = HarborClient(
            "http://harbor",
            client_timeout_s=1,
            cleanup_timeout_s=0.1,
            poll_interval_s=0.001,
            transport=FailedResponse(),
        )
        try:
            result = await c.run_trial(request())
            assert result.failure_kind == kind
            assert result.cleanup_confirmed
        finally:
            await c.close()

    asyncio.run(scenario())
