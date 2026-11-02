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

from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from fastapi import FastAPI
    from transformers import PreTrainedTokenizerBase

PREFIX_CONTEXT_BEFORE = 64
PREFIX_CONTEXT_AFTER = 128


class PrefixBreakError(ValueError):
    """The harness changed a previously generated conversation prefix."""


@dataclass
class TurnRecord:
    """One turn's appended input and unmodified rollout output, aligned by token."""

    input_delta: list[int]
    output_ids: list[int]
    logprobs: list[float]
    token_versions: list[int]
    finish_reason: str
    trimmed_output_tokens: int = 0


@dataclass
class Session:
    """Mutable conversation history owned and serialized by one gateway."""

    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    prefix_broken: bool = False
    prefix_break_context: dict | None = None
    prefix_break_event: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionGateway:
    """Render chat messages and record the exact token IDs returned by rollout."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        generate: Callable[..., Awaitable[dict]],
        max_prompt_tokens: int,
        max_response_tokens: int,
        max_new_tokens: int = 768,
        max_sessions: int = 128,
        chat_template_kwargs: dict | None = None,
        render_chat: Callable[..., str] | None = None,
        boundary_token_ids: frozenset[int] = frozenset(),
    ) -> None:
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
        """Allocate an empty session and return its ID.

        Raises:
            RuntimeError: All configured session slots are in use.
        """
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(
                f"Session capacity exhausted: expected fewer than {self.max_sessions} "
                f"active sessions, got {len(self.sessions)}"
            )
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = Session(session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Session:
        """Return a live session; raise KeyError for an unknown ID."""
        if session_id not in self.sessions:
            raise KeyError(
                f"Expected an active session ID, got unknown session {session_id!r}"
            )
        return self.sessions[session_id]

    async def close_session(self, session_id: str) -> None:
        """Release a session after active generation finishes; repeated closes succeed."""
        session = self.sessions.get(session_id)
        if session is None:
            return
        session.closed = True
        async with session.lock:
            self.sessions.pop(session_id, None)

    @staticmethod
    def _validate_request(body: dict) -> list[dict]:
        for key in ("stream", "tools", "tool_choice"):
            if body.get(key):
                raise ValueError(
                    f"Expected plain, non-streaming chat with {key} unset or false, got {body[key]!r}"
                )
        if body.get("n", 1) != 1:
            raise ValueError(f"Expected n=1 per session request, got {body['n']!r}")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Expected a nonempty messages list, got {messages!r}")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Expected messages[{index}] to be a mapping, got {type(message).__name__}"
                )
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Expected messages[{index}].role in system/user/assistant, got {message.get('role')!r}"
                )
            if not isinstance(message.get("content"), str):
                raise ValueError(
                    f"Expected messages[{index}].content to be text, got {type(message.get('content')).__name__}"
                )
            for key in ("tool_calls", "function_call"):
                if message.get(key):
                    raise ValueError(
                        f"Expected text actions without {key}, got messages[{index}].{key}={message[key]!r}"
                    )
        return messages

    def _prefix_delta(self, session: Session, rendered: str) -> tuple[list[int], int]:
        trim = 0
        if (
            session.turns
            and session.turns[-1].output_ids
            and session.token_ids[-1] in self.boundary_token_ids
        ):
            trim = 1
        if not session.turns:
            delta = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(delta) > self.max_prompt_tokens:
                raise ValueError(
                    f"Expected initial prompt <= {self.max_prompt_tokens} tokens, got {len(delta)}"
                )
            return delta, trim
        prefix_ids = session.token_ids[:-trim] if trim else session.token_ids
        previous = self.tokenizer.decode(
            prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not rendered.startswith(previous):
            session.prefix_broken = True
            offset = next(
                (i for i, (a, b) in enumerate(zip(previous, rendered)) if a != b),
                min(len(previous), len(rendered)),
            )
            start, end = (
                max(0, offset - PREFIX_CONTEXT_BEFORE),
                offset + PREFIX_CONTEXT_AFTER,
            )
            session.prefix_break_context = {
                "character_offset": offset,
                "previous": previous[start:end],
                "rendered": rendered[start:end],
            }
            session.prefix_break_event.set()
            raise PrefixBreakError(
                f"Expected session {session.session_id!r} to preserve its generated prefix; "
                f"at character {offset}, expected {previous[start:end]!r}, got {rendered[start:end]!r}"
            )
        return self.tokenizer.encode(
            rendered[len(previous) :], add_special_tokens=False
        ), trim

    def _sampling_params(self, body: dict, remaining: int) -> dict:
        requested = body.get(
            "max_completion_tokens", body.get("max_tokens", self.max_new_tokens)
        )
        if (
            not isinstance(requested, int)
            or isinstance(requested, bool)
            or requested <= 0
        ):
            raise ValueError(
                f"Expected max_tokens/max_completion_tokens to be a positive integer, got {requested!r}"
            )
        if remaining <= 0:
            raise ValueError(
                f"Session token budget exhausted: expected remaining tokens > 0, got {remaining} (budget={self.max_response_tokens})"
            )
        params = {
            key: body[key] for key in ("temperature", "top_p", "stop") if key in body
        }
        params["max_new_tokens"] = min(requested, remaining, self.max_new_tokens)
        return params

    @staticmethod
    def _validate_generation(result: dict, max_new_tokens: int) -> None:
        required = {"output_ids", "logprobs", "token_versions", "finish_reason"}
        missing = required - result.keys()
        if missing:
            raise ValueError(
                f"Expected rollout fields {sorted(required)}, got {sorted(result)}; missing {sorted(missing)}"
            )
        output_ids = result["output_ids"]
        if not 0 < len(output_ids) <= max_new_tokens:
            raise ValueError(
                f"Expected 1..{max_new_tokens} output tokens, got {len(output_ids)}"
            )
        for index, token in enumerate(output_ids):
            if not isinstance(token, int) or isinstance(token, bool) or token < 0:
                raise ValueError(
                    f"Expected output_ids[{index}] to be a nonnegative integer, got {token!r}"
                )
        for name in ("logprobs", "token_versions"):
            if len(result[name]) != len(output_ids):
                raise ValueError(
                    f"Expected {len(output_ids)} {name} entries for output tokens, got {len(result[name])}"
                )
        for index, value in enumerate(result["logprobs"]):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"Expected logprobs[{index}] to be finite, got {value!r}"
                )
        for index, value in enumerate(result["token_versions"]):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"Expected token_versions[{index}] to be a nonnegative integer, got {value!r}"
                )
        if result["finish_reason"] not in {"stop", "length"}:
            raise ValueError(
                f"Expected finish_reason in ['length', 'stop'], got {result['finish_reason']!r}"
            )

    async def complete(self, session_id: str, body: dict) -> dict:
        """Generate one text completion and append its exact rollout token records.

        Args:
            session_id: ID returned by create_session.
            body: OpenAI chat request with text messages and optional sampling limits.

        Returns:
            A non-streaming OpenAI chat completion response.

        Raises:
            ValueError: The request, token budget, or generated metadata is invalid.
            RuntimeError: Rollout reported a generation failure.
        """
        messages = self._validate_request(body)
        session = self.get_session(session_id)
        async with session.lock:
            if session.closed or session.prefix_broken:
                raise ValueError(
                    f"Expected session {session_id!r} to be open with an intact prefix, "
                    f"got closed={session.closed}, prefix_broken={session.prefix_broken}"
                )
            if self.render_chat:
                rendered = self.render_chat(messages, **self.chat_template_kwargs)
            else:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            delta, trim = self._prefix_delta(session, rendered)
            prefix_ids = session.token_ids[:-trim] if trim else session.token_ids
            input_ids = prefix_ids + delta
            prompt_size = (
                len(session.turns[0].input_delta) if session.turns else len(delta)
            )
            remaining = self.max_response_tokens - (len(input_ids) - prompt_size)
            params = self._sampling_params(body, remaining)
            result = await self.generate(input_ids, params)
            if "error" in result:
                raise RuntimeError(
                    f"Expected a completed rollout for session {session_id!r}, got {result['error']}"
                )
            self._validate_generation(result, params["max_new_tokens"])
            turn = TurnRecord(
                input_delta=delta,
                output_ids=list(result["output_ids"]),
                logprobs=list(result["logprobs"]),
                token_versions=list(result["token_versions"]),
                finish_reason=result["finish_reason"],
            )
            if trim:
                session.turns[-1].trimmed_output_tokens = trim
            session.turns.append(turn)
            session.token_ids = input_ids + turn.output_ids
            return self._completion_response(body, input_ids, turn)

    def _completion_response(
        self, body: dict, input_ids: list[int], turn: TurnRecord
    ) -> dict:
        content = self.tokenizer.decode(
            turn.output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
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
                    "finish_reason": turn.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(turn.output_ids),
                "total_tokens": len(input_ids) + len(turn.output_ids),
            },
        }


def assemble_session(session: Session) -> dict:
    """Build one training sequence from verified turn records."""
    if session.prefix_broken or not session.turns:
        raise ValueError(
            f"Cannot train an empty or prefix-broken session {session.session_id!r}: "
            f"expected turns > 0 and intact prefix, got turns={len(session.turns)}, prefix_broken={session.prefix_broken}"
        )
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
            raise ValueError(
                f"Expected trim 0 or 1 on continued turns and 0 on the final turn, "
                f"got trim={trim} at turn {index + 1}/{len(session.turns)}"
            )
        end = len(turn.output_ids) - trim
        response_ids.extend(turn.output_ids[:end])
        mask.extend([1] * end)
        logprobs.extend(turn.logprobs[:end])
        versions.extend(turn.token_versions[:end])
    assembled = prompt_ids + response_ids
    if assembled != session.token_ids:
        offset = next(
            (i for i, (a, b) in enumerate(zip(session.token_ids, assembled)) if a != b),
            min(len(session.token_ids), len(assembled)),
        )
        raise ValueError(
            f"Expected {len(session.token_ids)} history tokens, assembled {len(assembled)}; "
            f"first mismatch at {offset}: expected {session.token_ids[offset : offset + 1]}, got {assembled[offset : offset + 1]}"
        )
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_mask": mask,
        "rollout_logprobs": logprobs,
        "token_versions": versions,
    }


def create_gateway_app(gateway: SessionGateway) -> FastAPI:
    """Expose text completions and session record retrieval/closure over HTTP."""
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI()

    async def complete(session_id, body):
        try:
            return await gateway.complete(session_id, body)
        except KeyError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc
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
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc
        async with session.lock:
            return {
                "session_id": session_id,
                "prefix_broken": session.prefix_broken,
                "prefix_break_context": session.prefix_break_context,
                "turns": [asdict(turn) for turn in session.turns],
            }

    @app.delete("/sessions/{session_id}")
    async def close(session_id: str):
        await gateway.close_session(session_id)
        return {"closed": True}

    return app
