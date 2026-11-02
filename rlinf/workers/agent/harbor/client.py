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

"""HTTP client for submitting, querying, and cancelling Harbor trials."""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict
from http import HTTPStatus
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from .protocol import HarborTrialRequest, HarborTrialResult


class HarborClient:
    """Manage Harbor trials through the task service HTTP API.

    Args:
        server_url: Harbor task service address.
        client_timeout_s: Total submission/polling deadline.
        cleanup_timeout_s: Time reserved for cancellation acknowledgement.
        poll_interval_s: Delay between result queries.
        api_key: Optional task-service authentication token.
        transport: Optional httpx transport for embedding/testing.
    """

    def __init__(
        self,
        server_url: str,
        client_timeout_s: float = 960,
        cleanup_timeout_s: float = 30,
        poll_interval_s: float = 0.5,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        import httpx

        for name, value in {
            "client_timeout_s": client_timeout_s,
            "cleanup_timeout_s": cleanup_timeout_s,
            "poll_interval_s": poll_interval_s,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        self.client_timeout_s = client_timeout_s
        self.cleanup_timeout_s = cleanup_timeout_s
        self.poll_interval_s = poll_interval_s
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(
            base_url=server_url.rstrip("/"),
            headers=headers,
            timeout=cleanup_timeout_s,
            transport=transport,
        )
        self.active: dict[str, HarborTrialRequest] = {}
        self._submissions: dict[str, asyncio.Event] = {}

    async def cancel_trial(self, request: HarborTrialRequest) -> HarborTrialResult:
        """Cancel exactly one trial and wait for the server's cleanup acknowledgement."""
        submitted = self._submissions.get(request.trial_id)
        if submitted is not None:
            await asyncio.wait_for(submitted.wait(), self.cleanup_timeout_s)
        response = await self.http.delete(
            f"/v1/trials/{request.trial_id}", params={"session_id": request.session_id}
        )
        response.raise_for_status()
        return self._decode(response.json(), request)

    @staticmethod
    def _decode(payload: dict, request: HarborTrialRequest) -> HarborTrialResult:
        result = HarborTrialResult(**payload)
        if (result.trial_id, result.session_id) != (
            request.trial_id,
            request.session_id,
        ):
            raise ValueError(
                f"Expected trial/session {(request.trial_id, request.session_id)!r}, "
                f"got {(result.trial_id, result.session_id)!r}"
            )
        return result

    async def _cleanup(self, request: HarborTrialRequest) -> bool:
        try:
            result = await asyncio.wait_for(
                self.cancel_trial(request), self.cleanup_timeout_s
            )
            return result.terminal and result.cleanup_confirmed
        except Exception:
            return False

    async def run_trial(self, request: HarborTrialRequest) -> HarborTrialResult:
        """Submit one attempt; transport/deadline failures are infrastructure failures."""
        if self.client_timeout_s <= request.timeout_s + self.cleanup_timeout_s:
            raise ValueError(
                f"Expected client_timeout_s > {request.timeout_s + self.cleanup_timeout_s} "
                f"(trial={request.timeout_s}, cleanup={self.cleanup_timeout_s}), got {self.client_timeout_s}"
            )
        if request.trial_id in self.active:
            raise ValueError(
                f"Expected an inactive trial_id, got already active {request.trial_id!r}"
            )
        self.active[request.trial_id] = request
        submitted = asyncio.Event()
        self._submissions[request.trial_id] = submitted
        may_exist = True

        async def submit_and_wait():
            nonlocal may_exist
            try:
                response = await self.http.post("/v1/trials", json=asdict(request))
            finally:
                submitted.set()
            if (
                HTTPStatus.BAD_REQUEST
                <= response.status_code
                < HTTPStatus.INTERNAL_SERVER_ERROR
            ):
                may_exist = False
            response.raise_for_status()
            result = self._decode(response.json(), request)
            while not result.terminal:
                await asyncio.sleep(self.poll_interval_s)
                response = await self.http.get(f"/v1/trials/{request.trial_id}")
                response.raise_for_status()
                result = self._decode(response.json(), request)
            return result

        try:
            return await asyncio.wait_for(submit_and_wait(), self.client_timeout_s)
        except asyncio.CancelledError:
            submitted.set()
            await asyncio.shield(self._cleanup(request))
            raise
        except Exception as exc:
            cleaned = await self._cleanup(request) if may_exist else True
            return HarborTrialResult(
                trial_id=request.trial_id,
                session_id=request.session_id,
                status="failed",
                infrastructure_failure=True,
                cleanup_confirmed=cleaned,
                failure_kind=self._classify_failure(exc, may_exist),
                exit_status=type(exc).__name__,
            )
        finally:
            self.active.pop(request.trial_id, None)
            self._submissions.pop(request.trial_id, None)

    @staticmethod
    def _classify_failure(exc: Exception, may_exist: bool) -> str:
        import httpx

        if isinstance(exc, asyncio.TimeoutError):
            return "client_timeout"
        if not isinstance(exc, httpx.HTTPStatusError):
            return "transport_or_protocol"
        status = exc.response.status_code
        if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            return "configuration_error"
        if not may_exist and status in {
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.NOT_FOUND,
            HTTPStatus.UNPROCESSABLE_ENTITY,
        }:
            return "configuration_error"
        return "transport_or_protocol"

    async def close(self) -> None:
        """Cancel owned attempts before closing HTTP connections."""
        await asyncio.gather(*(self._cleanup(r) for r in list(self.active.values())))
        await self.http.aclose()
