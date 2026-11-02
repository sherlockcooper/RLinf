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

"""Independent task service with per-trial deadlines and scoped cancellation."""

import asyncio
import hmac
import math
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

from .protocol import HarborTrialRequest, HarborTrialResult


@dataclass
class _Entry:
    request: HarborTrialRequest
    result: HarborTrialResult
    task: asyncio.Task | None = None
    runner: asyncio.Task | None = None
    cancel_requested: bool = False


def create_app(
    execute: Callable[[HarborTrialRequest], Awaitable[HarborTrialResult]],
    max_concurrency: int = 8,
    max_timeout_s: float = 900,
    cleanup_timeout_s: float = 20,
    max_records: int = 1024,
    api_key: str | None = None,
):
    """Build a service. execute must finish sandbox cleanup before returning.

    A deadline includes queueing. An unconfirmed cancellation retains its slot
    until the runner exits, and is never reported as successful cleanup.
    Completed IDs remain idempotent until evicted by max_records.
    """
    from fastapi import FastAPI, Header, HTTPException

    if any(
        not math.isfinite(v) or v <= 0
        for v in (max_concurrency, max_timeout_s, cleanup_timeout_s, max_records)
    ):
        raise ValueError("Service limits must be positive")
    entries: dict[str, _Entry] = {}
    slots = asyncio.Semaphore(max_concurrency)

    async def cleanup(entry):
        if entry.runner is None:
            return True
        entry.runner.cancel()
        done, _ = await asyncio.wait([entry.runner], timeout=cleanup_timeout_s)
        if not done:
            return False
        try:
            value = entry.runner.result()
            return value.cleanup_confirmed
        except asyncio.CancelledError:
            # The backend must raise cancellation only after confirmed cleanup.
            return True
        except Exception:
            return False

    async def run(entry):
        start = time.monotonic()
        acquired = False
        result = entry.result
        try:
            await asyncio.wait_for(slots.acquire(), entry.request.timeout_s)
            acquired = True
            result.status = "running"
            entry.runner = asyncio.create_task(execute(entry.request))
            remaining = max(0, entry.request.timeout_s - (time.monotonic() - start))
            done, _ = await asyncio.wait([entry.runner], timeout=remaining)
            if not done:
                raise asyncio.TimeoutError
            result = entry.runner.result()
            if (result.trial_id, result.session_id) != (
                entry.request.trial_id,
                entry.request.session_id,
            ):
                raise ValueError("Executor returned a different trial/session identity")
            if not result.terminal:
                raise ValueError("Executor must return a terminal result")
        except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
            result.status = "cancelling"
            cleaned = await cleanup(entry)
            cancelled = isinstance(exc, asyncio.CancelledError)
            result = HarborTrialResult(
                trial_id=entry.request.trial_id,
                session_id=entry.request.session_id,
                status="cancelled" if cancelled else "timed_out",
                reward=None,
                infrastructure_failure=not cancelled or not cleaned,
                failure_kind="cancelled"
                if cancelled and cleaned
                else "trial_deadline"
                if cleaned
                else "cleanup_failed",
                cleanup_confirmed=cleaned,
            )
        except Exception as exc:
            cleaned = await cleanup(entry)
            result = HarborTrialResult(
                trial_id=entry.request.trial_id,
                session_id=entry.request.session_id,
                status="failed",
                infrastructure_failure=True,
                cleanup_confirmed=cleaned,
                failure_kind="executor_error",
                exit_status=type(exc).__name__,
            )
        finally:
            if acquired:
                if entry.runner is not None and not entry.runner.done():

                    def release(task):
                        slots.release()
                        if not task.cancelled():
                            task.exception()

                    entry.runner.add_done_callback(release)
                else:
                    slots.release()
            result.metrics["trial_elapsed_s"] = time.monotonic() - start
            entry.result = result

    async def cancel(entry):
        # Let the lifecycle coroutine enter its try/finally even on immediate cancellation.
        await asyncio.sleep(0)
        if (
            entry.task
            and not entry.task.done()
            and not entry.cancel_requested
            and entry.result.status != "cancelling"
        ):
            entry.cancel_requested = True
            entry.task.cancel()
        if entry.task:
            await asyncio.shield(entry.task)
        return asdict(entry.result)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.gather(
            *(cancel(e) for e in list(entries.values())), return_exceptions=True
        )

    app = FastAPI(lifespan=lifespan)

    def authorize(value):
        if api_key and not hmac.compare_digest(value or "", f"Bearer {api_key}"):
            raise HTTPException(401, "Invalid task service credentials")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "protocol": "rlinf-harbor-v1",
            "max_timeout_s": max_timeout_s,
            "cleanup_timeout_s": cleanup_timeout_s,
        }

    @app.post("/v1/trials")
    async def submit(
        request: HarborTrialRequest, authorization: str | None = Header(None)
    ):
        authorize(authorization)
        if request.timeout_s > max_timeout_s:
            raise HTTPException(422, "Trial timeout exceeds the server limit")
        if request.trial_id in entries:
            entry = entries[request.trial_id]
            if entry.request != request:
                raise HTTPException(
                    409, "trial_id already belongs to a different request"
                )
            return asdict(entry.result)
        if len(entries) >= max_records:
            finished = next(
                (
                    k
                    for k, e in entries.items()
                    if e.result.terminal and e.result.cleanup_confirmed
                ),
                None,
            )
            if finished is None:
                raise HTTPException(503, "Trial registry is full")
            del entries[finished]
        entry = _Entry(request, HarborTrialResult(request.trial_id, request.session_id))
        entries[request.trial_id] = entry
        entry.task = asyncio.create_task(run(entry))
        return asdict(entry.result)

    @app.get("/v1/trials/{trial_id}")
    async def get(trial_id: str, authorization: str | None = Header(None)):
        authorize(authorization)
        if trial_id not in entries:
            raise HTTPException(404, "Unknown trial_id")
        return asdict(entries[trial_id].result)

    @app.delete("/v1/trials/{trial_id}")
    async def delete(
        trial_id: str, session_id: str, authorization: str | None = Header(None)
    ):
        authorize(authorization)
        if trial_id not in entries:
            raise HTTPException(404, "Unknown trial_id")
        if entries[trial_id].request.session_id != session_id:
            raise HTTPException(409, "Cancellation session does not own this trial")
        return await cancel(entries[trial_id])

    return app
