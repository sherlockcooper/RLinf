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

"""Session API and exact multi-turn training token alignment."""

import asyncio

import pytest

from rlinf.data.schema.reasoning_results import RolloutResult
from rlinf.workers.agent.harbor.gateway import (
    SessionGateway,
    assemble_session,
    create_gateway_app,
)

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        text = "".join(chr(i) for i in ids)
        return text.replace("~", "") if skip_special_tokens else text

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join(m["role"] + ":" + m["content"] + "~" for m in messages)
            + "assistant:"
        )


def test_multiturn_http_and_training_masks():
    async def scenario():
        calls = []

        async def generate(ids, params):
            calls.append(ids.copy())
            return {
                "output_ids": [ord("x"), ord("~")],
                "logprobs": [-0.2, -0.4],
                "token_versions": [7, 7],
                "finish_reason": "stop",
            }

        gateway = SessionGateway(Tokenizer(), generate, 200, 200)
        sid = gateway.create_session()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(create_gateway_app(gateway)),
            base_url="http://gateway",
        )
        messages = [{"role": "user", "content": "fix"}]
        response = await client.post(
            f"/sessions/{sid}/v1/chat/completions", json={"messages": messages}
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "x"
        messages += [
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "tool output"},
        ]
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Session-ID": sid},
            json={"messages": messages},
        )
        assert response.status_code == 200
        sequence = assemble_session(gateway.get_session(sid))
        assert calls[1][: len(calls[0]) + 2] == calls[0] + [120, 126]
        assert sequence["prompt_ids"] + sequence["response_ids"] == calls[1] + [
            120,
            126,
        ]
        assert [
            t for t, m in zip(sequence["response_ids"], sequence["response_mask"]) if m
        ] == [120, 126, 120, 126]
        assert [
            v
            for v, m in zip(sequence["token_versions"], sequence["response_mask"])
            if m
        ] == [7] * 4
        assert all(
            v == -1
            for v, m in zip(sequence["token_versions"], sequence["response_mask"])
            if not m
        )
        records = (await client.get(f"/sessions/{sid}")).json()
        assert len(records["turns"]) == 2
        assert (await client.delete(f"/sessions/{sid}")).status_code == 200
        assert not gateway.sessions
        await client.aclose()

    asyncio.run(scenario())


def test_prefix_break_is_recorded_and_cannot_train():
    async def scenario():
        async def generate(ids, params):
            return {
                "output_ids": [120, 126],
                "logprobs": [-1, -1],
                "token_versions": [0, 0],
                "finish_reason": "stop",
            }

        gateway = SessionGateway(Tokenizer(), generate, 200, 200)
        sid = gateway.create_session()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_gateway_app(gateway)),
            base_url="http://gateway",
        ) as client:
            url = f"/sessions/{sid}/v1/chat/completions"
            await client.post(
                url, json={"messages": [{"role": "user", "content": "old"}]}
            )
            result = await client.post(
                url, json={"messages": [{"role": "user", "content": "changed"}]}
            )
            assert result.status_code == 409
            assert gateway.get_session(sid).prefix_break_event.is_set()
            assert (
                gateway.get_session(sid).prefix_break_context["character_offset"] >= 0
            )
            with pytest.raises(ValueError, match="prefix-broken"):
                assemble_session(gateway.get_session(sid))

    asyncio.run(scenario())


def test_generated_tokens_are_not_reencoded():
    class AlternativeEncodingTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            expanded = [
                c for token in ids for c in ([120, 121] if token == 1000 else [token])
            ]
            return super().decode(expanded, **kwargs)

    async def scenario():
        calls = []

        async def generate(ids, params):
            calls.append(ids.copy())
            return {
                "output_ids": [1000, 126],
                "logprobs": [-1, -1],
                "token_versions": [0, 0],
                "finish_reason": "stop",
            }

        gateway = SessionGateway(AlternativeEncodingTokenizer(), generate, 200, 200)
        sid = gateway.create_session()
        messages = [{"role": "user", "content": "fix"}]
        first = await gateway.complete(sid, {"messages": messages})
        messages += [
            first["choices"][0]["message"],
            {"role": "user", "content": "continue"},
        ]
        await gateway.complete(sid, {"messages": messages})
        assert calls[1][len(calls[0]) : len(calls[0]) + 2] == [1000, 126]
        sequence = assemble_session(gateway.get_session(sid))
        assert sequence["prompt_ids"] + sequence["response_ids"] == calls[1] + [
            1000,
            126,
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "output",
    [
        {
            "output_ids": [1],
            "logprobs": [],
            "token_versions": [0],
            "finish_reason": "stop",
        },
        {
            "output_ids": [1],
            "logprobs": [-1],
            "token_versions": [-1],
            "finish_reason": "stop",
        },
        {
            "output_ids": [1],
            "logprobs": [-1],
            "token_versions": [0],
            "finish_reason": "abort",
        },
    ],
)
def test_invalid_generation_is_not_recorded(output):
    async def scenario():
        async def generate(ids, params):
            return output

        gateway = SessionGateway(Tokenizer(), generate, 200, 200)
        sid = gateway.create_session()
        with pytest.raises(ValueError):
            await gateway.complete(
                sid, {"messages": [{"role": "user", "content": "fix"}]}
            )
        assert gateway.get_session(sid).turns == []

    asyncio.run(scenario())


def test_rollout_metadata_survives_merge_and_split():
    results = [
        RolloutResult(
            num_sequence=1,
            group_size=1,
            prompt_lengths=[1],
            prompt_ids=[[10]],
            response_lengths=[2],
            response_ids=[[11, 12]],
            is_end=[True],
            response_mask=[[1, 0]],
            rollout_logprobs=[[-1, 0]],
            token_versions=[[v, -1]],
            trajectory_metadata=[{"session_id": str(v)}],
            rewards=[float(v)],
        )
        for v in (0, 1)
    ]
    merged = RolloutResult.merge_result_list(results)
    split = RolloutResult.split_results(merged, 2)
    assert merged.token_versions == [[0, -1], [1, -1]]
    assert [r.trajectory_metadata for r in split] == [
        [{"session_id": "0"}],
        [{"session_id": "1"}],
    ]
    assert [r.response_mask for r in split] == [[[1, 0]], [[1, 0]]]


@pytest.mark.parametrize("boundary", ["|", "~"])
def test_feedback_trims_only_boundary_and_keeps_raw_record(boundary):
    class BoundaryTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return (
                "".join(
                    ("|" if m["role"] == "user" else ">") + m["content"]
                    for m in messages
                )
                + ">"
            )

        def decode(self, ids, skip_special_tokens=False, **kwargs):
            text = "".join(map(chr, ids))
            return (
                text.replace("|", "").replace("~", "") if skip_special_tokens else text
            )

    async def scenario():
        calls = []

        async def generate(ids, params):
            calls.append(ids.copy())
            return {
                "output_ids": list(map(ord, "bad" + boundary))
                if len(calls) == 1
                else [ord("x")],
                "logprobs": [-0.1] * 4 if len(calls) == 1 else [-0.2],
                "token_versions": [3] * 4 if len(calls) == 1 else [3],
                "finish_reason": "stop",
            }

        gateway = SessionGateway(
            BoundaryTokenizer(),
            generate,
            200,
            200,
            boundary_token_ids=frozenset(map(ord, "|~")),
        )
        sid = gateway.create_session()
        messages = [{"role": "user", "content": "task"}]
        reply = await gateway.complete(sid, {"messages": messages})
        messages += [
            reply["choices"][0]["message"],
            {"role": "user", "content": "format error"},
        ]
        await gateway.complete(sid, {"messages": messages})
        session = gateway.get_session(sid)
        assert session.turns[0].output_ids == list(map(ord, "bad" + boundary))
        assert session.turns[0].trimmed_output_tokens == 1
        assert calls[1] == list(map(ord, "|task>bad|format error>"))
        seq = assemble_session(session)
        assert sum(seq["response_mask"]) == 4
        assert [
            p for p, m in zip(seq["rollout_logprobs"], seq["response_mask"]) if m
        ] == [-0.1] * 3 + [-0.2]
        messages[1]["content"] = "changed"
        messages += [
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "next"},
        ]
        from rlinf.workers.agent.harbor.gateway import PrefixBreakError

        with pytest.raises(PrefixBreakError):
            await gateway.complete(sid, {"messages": messages})
        assert len(calls) == 2

    asyncio.run(scenario())
