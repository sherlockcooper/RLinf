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

"""Session-scoped chat requests and exact rollout token records."""

import asyncio
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from typing import Any, Awaitable, Callable


class PrefixBreakError(ValueError):
    """The harness changed a previously generated conversation prefix."""


@dataclass
class TurnRecord:
    input_delta: list[int]
    output_ids: list[int]
    logprobs: list[float]
    token_versions: list[int]
    finish_reason: str
    routing_replay: Any = None
    trimmed_output_tokens: int = 0


@dataclass
class Session:
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    prefix_broken: bool = False
    prefix_break_context: dict | None = None
    prefix_break_event: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionGateway:
    """Render chat messages and preserve every generated token without retokenizing it."""

    def __init__(
        self,
        tokenizer,
        generate: Callable[..., Awaitable[dict]],
        max_prompt_tokens: int,
        max_response_tokens: int,
        max_new_tokens: int = 768,
        max_sessions: int = 128,
        chat_template_kwargs: dict | None = None,
        render_chat: Callable[..., str] | None = None,
        boundary_token_ids: frozenset[int] = frozenset(),
    ):
        self.tokenizer = tokenizer
        self.generate = generate
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.max_new_tokens = max_new_tokens
        self.max_sessions = max_sessions
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.render_chat = render_chat
        self.boundary_token_ids = frozenset(boundary_token_ids)
        self.sessions: dict[str, Session] = {}

    def create_session(self) -> str:
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("Session capacity exhausted")
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = Session(session_id)
        return session_id

    def get_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            raise KeyError("Unknown session")
        return self.sessions[session_id]

    async def close_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        session.closed = True
        async with session.lock:
            self.sessions.pop(session_id, None)

    async def complete(self, session_id: str, body: dict) -> dict:
        if body.get("stream") or body.get("tools") or body.get("tool_choice"):
            raise ValueError("Only non-streaming text chat completions are supported")
        if body.get("n", 1) != 1:
            raise ValueError("Each session supports one completion per request")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or message.get("tool_calls")
                or message.get("function_call")
            ):
                raise ValueError(
                    "Expected plain-text system, user, or assistant messages"
                )
        session = self.get_session(session_id)
        async with session.lock:
            if session.closed or session.prefix_broken:
                raise ValueError("Session is closed or has a broken prefix")
            rendered = (
                self.render_chat(
                    messages,
                    **self.chat_template_kwargs,
                )
                if self.render_chat
                else self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            )
            trim = int(
                bool(
                    session.turns
                    and session.turns[-1].output_ids
                    and session.token_ids[-1] in self.boundary_token_ids
                )
            )
            prefix_ids = session.token_ids[:-trim] if trim else session.token_ids
            if session.turns:
                previous = self.tokenizer.decode(
                    prefix_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if not rendered.startswith(previous):
                    session.prefix_broken = True
                    offset = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(previous, rendered))
                            if a != b
                        ),
                        min(len(previous), len(rendered)),
                    )
                    session.prefix_break_context = {
                        "character_offset": offset,
                        "previous": previous[max(0, offset - 64) : offset + 128],
                        "rendered": rendered[max(0, offset - 64) : offset + 128],
                    }
                    session.prefix_break_event.set()
                    raise PrefixBreakError(
                        "Rendered conversation does not preserve generated tokens"
                    )
                delta = self.tokenizer.encode(
                    rendered[len(previous) :], add_special_tokens=False
                )
            else:
                delta = self.tokenizer.encode(rendered, add_special_tokens=False)
                if len(delta) > self.max_prompt_tokens:
                    raise ValueError("Initial prompt exceeds max_prompt_tokens")
            input_ids = prefix_ids + delta
            prompt_size = (
                len(session.turns[0].input_delta) if session.turns else len(delta)
            )
            remaining = self.max_response_tokens - (len(input_ids) - prompt_size)
            requested = body.get(
                "max_completion_tokens", body.get("max_tokens", self.max_new_tokens)
            )
            if (
                not isinstance(requested, int)
                or isinstance(requested, bool)
                or requested <= 0
            ):
                raise ValueError("max_tokens must be a positive integer")
            if remaining <= 0:
                raise ValueError("Session token budget exhausted")
            params = {
                key: body[key]
                for key in ("temperature", "top_p", "stop")
                if key in body
            }
            params["max_new_tokens"] = min(requested, remaining, self.max_new_tokens)
            result = await self.generate(input_ids, params)
            if "error" in result:
                raise RuntimeError(result["error"])
            output_ids = list(result["output_ids"])
            logprobs = list(result["logprobs"])
            versions = list(result["token_versions"])
            if (
                not output_ids
                or len(output_ids) > params["max_new_tokens"]
                or len(output_ids) != len(logprobs)
                or len(output_ids) != len(versions)
                or any(not math.isfinite(v) for v in logprobs)
                or any(not isinstance(v, int) or v < 0 for v in versions)
            ):
                raise ValueError("Rollout returned invalid token metadata")
            reason = result["finish_reason"]
            if reason not in {"stop", "length"}:
                raise ValueError(f"Unfinished generation: {reason}")
            if trim:
                session.turns[-1].trimmed_output_tokens = trim
            session.turns.append(
                TurnRecord(
                    delta,
                    output_ids,
                    logprobs,
                    versions,
                    reason,
                    result.get("routing_replay"),
                )
            )
            session.token_ids = input_ids + output_ids
            content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "policy"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": len(output_ids),
                    "total_tokens": len(input_ids) + len(output_ids),
                },
            }


def assemble_session(session: Session) -> dict:
    """Build one training sequence from verified turn records."""
    if session.prefix_broken or not session.turns:
        raise ValueError("Cannot train an empty or prefix-broken session")
    prompt_ids = list(session.turns[0].input_delta)
    response_ids, mask, logprobs, versions = [], [], [], []
    for index, turn in enumerate(session.turns):
        if index:
            response_ids.extend(turn.input_delta)
            mask.extend([0] * len(turn.input_delta))
            logprobs.extend([0.0] * len(turn.input_delta))
            versions.extend([-1] * len(turn.input_delta))
        trim = turn.trimmed_output_tokens
        if trim not in (0, 1) or (index == len(session.turns) - 1 and trim):
            raise ValueError("Only a continued turn may trim one boundary token")
        end = len(turn.output_ids) - trim
        response_ids.extend(turn.output_ids[:end])
        mask.extend([1] * end)
        logprobs.extend(turn.logprobs[:end])
        versions.extend(turn.token_versions[:end])
    if prompt_ids + response_ids != session.token_ids:
        raise ValueError("Assembled tokens differ from the generation history")
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_mask": mask,
        "rollout_logprobs": logprobs,
        "token_versions": versions,
    }


def create_gateway_app(gateway: SessionGateway):
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI()

    async def complete(session_id, body):
        try:
            return await gateway.complete(session_id, body)
        except KeyError as exc:
            raise HTTPException(
                HTTPStatus.NOT_FOUND, "Unknown session or missing rollout metadata"
            ) from exc
        except PrefixBreakError as exc:
            raise HTTPException(HTTPStatus.CONFLICT, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def session_completion(session_id: str, body: dict):
        return await complete(session_id, body)

    @app.post("/v1/chat/completions")
    async def header_completion(body: dict, x_session_id: str = Header(...)):
        return await complete(x_session_id, body)

    @app.get("/sessions/{session_id}")
    async def records(session_id: str):
        try:
            session = gateway.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, "Unknown session") from exc
        async with session.lock:
            return {
                "session_id": session_id,
                "prefix_broken": session.prefix_broken,
                "prefix_break_context": session.prefix_break_context,
                "turns": [asdict(turn) for turn in session.turns],
            }

    @app.delete("/sessions/{session_id}")
    async def close(session_id: str):
        try:
            await gateway.close_session(session_id)
        except KeyError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, "Unknown session") from exc
        return {"closed": True}

    return app
