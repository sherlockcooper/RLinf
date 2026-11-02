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

"""Exercise format-error recovery through the real mini-swe-agent loop."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("minisweagent")
ModelResponse = pytest.importorskip("litellm").ModelResponse
DefaultAgent = pytest.importorskip("minisweagent.agents.default").DefaultAgent

path = Path(__file__).resolve().parents[2] / "examples/agent/harbor/mini_swe_model.py"
spec = importlib.util.spec_from_file_location("mini_swe_model", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("retry_valid", [True, False])
def test_format_error_keeps_original_response_and_feedback(tmp_path, retry_valid):
    original = "  <think>inspect the code</think>\nMissing a command.  \n"
    calls = []

    class Model(module.TrajectoryTextModel):
        def _query(self, messages, **kwargs):
            calls.append(messages)
            content = (
                original
                if len(calls) == 1 or not retry_valid
                else "```mswea_bash_command\necho ok\n```"
            )
            return ModelResponse(
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
    agent = DefaultAgent(
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


def test_session_budget_stops_agent_without_retry(tmp_path, monkeypatch):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        raise module.BadRequestError(
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

    model = module.TrajectoryTextModel(
        model_name="openai/test", cost_tracking="ignore_errors"
    )
    agent = DefaultAgent(
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


def test_other_bad_request_is_not_a_budget_limit(monkeypatch):
    error = module.BadRequestError(
        message="Unsupported parameter", model="openai/test", llm_provider="openai"
    )

    def complete(**kwargs):
        raise error

    monkeypatch.setattr("litellm.completion", complete)
    model = module.TrajectoryTextModel(
        model_name="openai/test", cost_tracking="ignore_errors"
    )
    with pytest.raises(module.BadRequestError) as caught:
        model._query([{"role": "user", "content": "Fix the issue"}])
    assert caught.value is error


def context_error(input_tokens, completion_tokens=4096):
    return module.BadRequestError(
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
def test_context_limit_retries_with_remaining_generation_budget(monkeypatch, token_key):
    calls = []
    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "done"}}]
    )

    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise context_error(28949)
        return response

    monkeypatch.setattr("litellm.completion", complete)
    model = module.TrajectoryTextModel(
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
    monkeypatch, input_tokens, input_only_error
):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if input_only_error:
            raise module.BadRequestError(
                message=(
                    f"The input ({input_tokens} tokens) is longer than the "
                    "model's context length (32768 tokens)."
                ),
                model="openai/test",
                llm_provider="openai",
            )
        raise context_error(input_tokens)

    monkeypatch.setattr("litellm.completion", complete)
    model = module.TrajectoryTextModel(
        model_name="openai/test", cost_tracking="ignore_errors"
    )
    with pytest.raises(module.LimitsExceeded):
        model._query([{"role": "user", "content": "Continue"}])
    assert len(calls) == 1
