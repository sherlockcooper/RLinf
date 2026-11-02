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

"""Run Harbor trials through an RLinf worker."""

import asyncio

from rlinf.scheduler import Worker
from rlinf.workers.agent.harbor.client import HarborClient
from rlinf.workers.agent.harbor.protocol import HarborTrialRequest, HarborTrialResult


class HarborTaskWorker(Worker):
    """Run independent Harbor attempts with externally supplied session model URLs.

    Returns verifier rewards, failure metadata, and sandbox cleanup status.
    Each request needs a unique trial_id and session_id, including group samples.
    """

    def __init__(
        self,
        server_url: str,
        max_concurrency: int = 8,
        client_timeout_s: float = 960,
        cleanup_timeout_s: float = 30,
        api_key: str | None = None,
    ):
        super().__init__()
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._settings = {
            "server_url": server_url,
            "client_timeout_s": client_timeout_s,
            "cleanup_timeout_s": cleanup_timeout_s,
            "api_key": api_key,
        }
        self._client = None
        self._limit = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task] = set()
        self._requests: dict[str, HarborTrialRequest] = {}
        self._queued: set[str] = set()
        self._cancelled: set[str] = set()
        self._closing = False

    def _get_client(self) -> HarborClient:
        if self._client is None:
            self._client = HarborClient(**self._settings)
        return self._client

    async def run_trials(
        self, requests: list[HarborTrialRequest]
    ) -> list[HarborTrialResult]:
        """Run a group of attempts, preserving input order and per-attempt identity."""
        if self._closing:
            raise RuntimeError("Harbor worker is shutting down")
        if len({r.trial_id for r in requests}) != len(requests):
            raise ValueError("trial_id must be unique within a group")
        if len({r.session_id for r in requests}) != len(requests):
            raise ValueError("session_id must be unique within a group")
        if any(r.trial_id in self._requests for r in requests):
            raise ValueError("trial_id already active in this worker")
        self._requests.update({r.trial_id: r for r in requests})
        self._queued.update(r.trial_id for r in requests)

        async def run(request):
            async with self._limit:
                self._queued.discard(request.trial_id)
                if request.trial_id in self._cancelled:
                    return HarborTrialResult(
                        request.trial_id,
                        request.session_id,
                        "cancelled",
                        cleanup_confirmed=True,
                    )
                return await self._get_client().run_trial(request)

        tasks = [asyncio.create_task(run(request)) for request in requests]
        self._tasks.update(tasks)
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.difference_update(tasks)
            for request in requests:
                self._requests.pop(request.trial_id, None)
                self._queued.discard(request.trial_id)
                self._cancelled.discard(request.trial_id)

    async def cancel_trial(self, request: HarborTrialRequest) -> HarborTrialResult:
        """Cancel a stale or unwanted trial without flushing unrelated attempts."""
        if request.trial_id in self._queued:
            if self._requests[request.trial_id] != request:
                raise ValueError("Cancellation does not match queued request")
            self._cancelled.add(request.trial_id)
            return HarborTrialResult(
                request.trial_id,
                request.session_id,
                "cancelled",
                cleanup_confirmed=True,
            )
        return await self._get_client().cancel_trial(request)

    async def shutdown(self) -> None:
        """Release this worker's active trials and HTTP resources."""
        self._closing = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.close()
            self._client = None
