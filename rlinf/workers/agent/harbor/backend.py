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
    ):
        self.tasks_dir = Path(tasks_dir).resolve()
        self.trials_dir = Path(trials_dir).resolve()
        self.agent_config = copy.deepcopy(agent_config)
        self.environment_config = dict(environment_config or {})
        if self.environment_config.get("type", "docker") != "docker":
            raise ValueError("The Harbor executor currently supports Docker only")

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
                request.trial_id,
                request.session_id,
                "failed",
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
                    raise ValueError("Use either config_file or config, not both")
                kwargs["config"] = yaml.safe_load(
                    Path(kwargs.pop("config_file")).read_text()
                )
            model_config = kwargs.setdefault("config", {}).setdefault("model", {})
            model_kwargs = model_config.setdefault("model_kwargs", {})
            model_kwargs["api_base"] = request.model_url
            allowed = {"temperature", "top_p", "max_tokens", "stop", "seed"}
            if request.sampling_params.keys() - allowed:
                raise ValueError("Unsupported sampling parameters")
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
            # Harbor logs some cleanup errors instead of raising them. Confirm removal.
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
                stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            except BaseException:
                process.kill()
                await process.wait()
                raise
            if process.returncode or stdout.strip():
                raise RuntimeError(
                    f"Sandbox cleanup unconfirmed: {stderr.decode(errors='replace')}"
                )
        return self._result(request, payload, trial.paths.trial_dir)

    @staticmethod
    def _result(
        request: HarborTrialRequest, payload: dict, trial_dir: Path
    ) -> HarborTrialResult:
        exception = payload.get("exception_info") or {}
        exception_type = exception.get("exception_type")
        reward = (payload.get("verifier_result") or {}).get("rewards", {}).get("reward")
        exit_status = exception_type
        trajectory = trial_dir / "agent/mini-swe-agent.trajectory.json"
        if trajectory.is_file():
            try:
                info = json.loads(trajectory.read_text()).get("info", {})
                exit_status = info.get("exit_status") or exit_status
            except (OSError, ValueError, AttributeError):
                pass
        model_format_error = exit_status in MODEL_FORMAT_ERRORS
        model_exception = exception_type in MODEL_FORMAT_ERRORS or (
            exception_type == "NonZeroAgentExitCodeError" and model_format_error
        )
        infra = bool(
            exception_type
            and exception_type != "AgentTimeoutError"
            and not model_exception
        )
        # Keep malformed model replies as negative samples, even without a score.
        if model_format_error and not infra:
            reward = 0.0
        valid_reward = isinstance(reward, (int, float)) and math.isfinite(reward)
        infra = infra or not valid_reward
        if (
            exit_status
            and exit_status
            not in {
                "Submitted",
                "LimitsExceeded",
                "AgentTimeoutError",
            }
            | MODEL_FORMAT_ERRORS
        ):
            infra = True
        return HarborTrialResult(
            request.trial_id,
            request.session_id,
            "failed" if infra else "completed",
            reward=None if infra else float(reward),
            infrastructure_failure=infra,
            failure_kind=(
                exception_type
                or ("missing_reward" if not valid_reward else "agent_error")
            )
            if infra
            else ("model_format_error" if model_format_error else None),
            exit_status=exit_status,
            cleanup_confirmed=True,
            eval_report={
                "trial_dir": str(trial_dir),
                "verifier_result": payload.get("verifier_result"),
            },
        )
