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

"""Request and result types for Harbor trials."""

import math
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "timed_out"}


@dataclass(frozen=True)
class HarborTrialRequest:
    """Task instance, model endpoint, and session identity for one attempt."""

    trial_id: str
    session_id: str
    instance_id: str
    model_url: str
    model: str
    timeout_s: float = 900
    agent_name: str = "mini-swe-agent"
    sampling_params: dict[str, Any] = field(default_factory=dict)
    model_api_key: str = "dummy"

    def __post_init__(self):
        for name in ("trial_id", "session_id", "instance_id"):
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", getattr(self, name)
            ):
                raise ValueError(
                    f"{name} must be a safe identifier of at most 128 characters"
                )
        url = urlsplit(self.model_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("model_url must be an absolute HTTP(S) API root")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not self.model or not self.agent_name:
            raise ValueError("model and agent_name must be nonempty")


@dataclass
class HarborTrialResult:
    """Task verdict; infrastructure failure never supplies a training reward."""

    trial_id: str
    session_id: str
    status: str = "queued"
    reward: float | None = None
    infrastructure_failure: bool = False
    failure_kind: str | None = None
    exit_status: str | None = None
    cleanup_confirmed: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    eval_report: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in TERMINAL_STATUSES | {"queued", "running", "cancelling"}:
            raise ValueError(f"Unknown trial status: {self.status}")
        if self.infrastructure_failure and self.reward is not None:
            raise ValueError("Infrastructure failures must not carry a reward")
        if self.reward is not None and not math.isfinite(self.reward):
            raise ValueError("reward must be finite")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
