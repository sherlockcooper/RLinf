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

"""Harbor SDK executor for trusted, locally prepared Docker tasks."""

import asyncio
import copy
import json
import math
import uuid
from pathlib import Path

from .protocol import HarborTrialRequest, HarborTrialResult

DOCKER_QUERY_TIMEOUT_S = 10
ALLOWED_SAMPLING_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "seed",
    }
)
MODEL_FORMAT_ERRORS = frozenset({"FormatError", "RepeatedFormatError"})


class HarborExecutor:
    """Execute a task with a pinned Harbor installation on the sandbox host.

    Args:
        tasks_dir: Trusted Harbor task directories, indexed by instance_id.
        trials_dir: Persistent output directory, outside disposable containers.
        agent_config: Administrator-provided Harbor AgentConfig fields.
        environment_config: Administrator-provided Docker resource settings.
    """

    def __init__(
        self,
        tasks_dir: str,
        trials_dir: str,
        agent_config: dict,
        environment_config: dict | None = None,
    ) -> None:
        self.tasks_dir = Path(tasks_dir).resolve()
        self.trials_dir = Path(trials_dir).resolve()
        self.agent_config = copy.deepcopy(agent_config)
        self.environment_config = dict(environment_config or {})
        if self.environment_config.get("type", "docker") != "docker":
            raise ValueError(
                f"Expected environment type docker, got {self.environment_config.get('type')!r}"
            )

    async def __call__(self, request: HarborTrialRequest) -> HarborTrialResult:
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            TrialConfig,
        )
        from harbor.trial.trial import Trial

        task_path = (self.tasks_dir / request.instance_id).resolve()
        if (
            task_path.parent != self.tasks_dir
            or not (task_path / "task.toml").is_file()
        ):
            return HarborTrialResult(
                trial_id=request.trial_id,
                session_id=request.session_id,
                status="failed",
                infrastructure_failure=True,
                failure_kind="unknown_task",
                cleanup_confirmed=True,
            )
        config = copy.deepcopy(self.agent_config)
        configured_name = config.pop(
            "public_name", config.get("name", "mini-swe-agent")
        )
        if request.agent_name != configured_name:
            raise ValueError(
                f"Expected configured agent {configured_name}, got {request.agent_name}"
            )
        config["model_name"] = request.model
        config.setdefault("env", {}).update(
            OPENAI_API_BASE=request.model_url,
            OPENAI_BASE_URL=request.model_url,
            OPENAI_API_KEY=request.model_api_key,
        )
        kwargs = config.setdefault("kwargs", {})
        if request.agent_name == "mini-swe-agent":
            if "config_file" in kwargs:
                import yaml

                if "config" in kwargs:
                    raise ValueError(
                        "Expected exactly one of config_file and config, got both keys"
                    )
                kwargs["config"] = yaml.safe_load(
                    Path(kwargs.pop("config_file")).read_text()
                )
            model_config = kwargs.setdefault("config", {}).setdefault("model", {})
            model_kwargs = model_config.setdefault("model_kwargs", {})
            model_kwargs["api_base"] = request.model_url
            unsupported = request.sampling_params.keys() - ALLOWED_SAMPLING_PARAMS
            if unsupported:
                raise ValueError(
                    f"Expected sampling parameters in {sorted(ALLOWED_SAMPLING_PARAMS)}, "
                    f"got unsupported keys {sorted(unsupported)}"
                )
            model_kwargs.update(request.sampling_params)
            model_kwargs.setdefault("extra_headers", {})["X-Session-ID"] = (
                request.session_id
            )
            if "max_tokens" in request.sampling_params:
                kwargs["max_tokens"] = request.sampling_params["max_tokens"]
        trial_name = f"rlinf-{uuid.uuid4().hex}"
        environment = dict(self.environment_config, type="docker", delete=True)
        trial = await Trial.create(
            TrialConfig(
                task=TaskConfig(path=task_path),
                trial_name=trial_name,
                trials_dir=self.trials_dir,
                agent=AgentConfig(**config),
                environment=EnvironmentConfig(**environment),
            )
        )
        # Harbor's adapter emits this header after the model config overrides.
        trial.agent.session_id = request.session_id
        payload = None
        try:
            payload = (await trial.run()).model_dump(mode="json")
        finally:
            # Confirm sandbox removal before reporting cleanup completion.
            await trial.agent_environment.stop(delete=True)
            process = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={trial_name}__env",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), DOCKER_QUERY_TIMEOUT_S
                )
            except BaseException:
                process.kill()
                await process.wait()
                raise
            if process.returncode or stdout.strip():
                raise RuntimeError(
                    f"Expected docker ps to exit 0 with no remaining containers, got exit={process.returncode}, containers={stdout.decode(errors='replace')!r}, stderr={stderr.decode(errors='replace')!r}"
                )
        return self._result(request, payload, trial.paths.trial_dir)

    @staticmethod
    def _result(
        request: HarborTrialRequest, payload: dict, trial_dir: Path
    ) -> HarborTrialResult:
        exception = payload.get("exception_info") or {}
        exception_type = exception.get("exception_type")
        reward = HarborExecutor._verifier_reward(payload)
        reward_source = "verifier" if reward is not None else None
        exit_status = exception_type
        trajectory = trial_dir / "agent/mini-swe-agent.trajectory.json"
        if trajectory.is_file():
            try:
                info = json.loads(trajectory.read_text()).get("info", {})
                exit_status = info.get("exit_status") or exit_status
            except (OSError, ValueError, AttributeError):
                pass
        infra, failure_kind = HarborExecutor._classify_failure(
            exception_type, exit_status, reward
        )
        if reward is None and failure_kind == "model_format_error":
            reward = 0.0
            reward_source = "model_format_error"
        return HarborTrialResult(
            trial_id=request.trial_id,
            session_id=request.session_id,
            status="failed" if infra else "completed",
            reward=reward,
            reward_source=reward_source,
            infrastructure_failure=infra,
            failure_kind=failure_kind,
            exit_status=exit_status,
            cleanup_confirmed=True,
            eval_report={
                "trial_dir": str(trial_dir),
                "verifier_result": payload.get("verifier_result"),
            },
        )

    @staticmethod
    def _verifier_reward(payload: dict) -> float | None:
        verifier = payload.get("verifier_result") or {}
        value = (verifier.get("rewards") or {}).get("reward")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _classify_failure(
        exception_type: str | None, exit_status: str | None, reward: float | None
    ) -> tuple[bool, str | None]:
        model_format_error = exit_status in MODEL_FORMAT_ERRORS
        model_exception = exception_type in MODEL_FORMAT_ERRORS
        if exception_type == "NonZeroAgentExitCodeError" and model_format_error:
            model_exception = True
        if (
            exception_type
            and exception_type != "AgentTimeoutError"
            and not model_exception
        ):
            return True, exception_type
        if exit_status and exit_status not in MODEL_FORMAT_ERRORS:
            if exit_status not in {"Submitted", "LimitsExceeded", "AgentTimeoutError"}:
                return True, exception_type or "agent_error"
        if model_format_error:
            return False, "model_format_error"
        if reward is None:
            return True, exception_type or "missing_reward"
        return False, None
