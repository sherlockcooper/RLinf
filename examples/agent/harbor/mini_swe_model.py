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

"""Text-action model adapter that retains rejected assistant responses."""

import re
from typing import Any

from litellm.exceptions import BadRequestError
from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

# SGLang's token-input endpoint reports an oversized input through LiteLLM.
SGLANG_INPUT_LIMIT_ERROR = re.compile(
    r"The input \((\d+) tokens\) is longer than the model's "
    r"context length \((\d+) tokens\)"
)
# SGLang's OpenAI endpoint reports the combined prompt/completion budget.
SGLANG_COMPLETION_LIMIT_ERROR = re.compile(
    r"Requested token count exceeds the model's maximum context "
    r"length of (\d+) tokens\. You requested a total of \d+ tokens: "
    r"(\d+) tokens from the input messages and (\d+) tokens for the completion"
)


def _remaining_budget(message: str) -> int | None:
    if "Session token budget exhausted" in message:
        return 0
    input_limit = SGLANG_INPUT_LIMIT_ERROR.search(message)
    if input_limit and int(input_limit[1]) >= int(input_limit[2]):
        return 0
    budget = SGLANG_COMPLETION_LIMIT_ERROR.search(message)
    if budget is None:
        return None
    limit, input_tokens, requested = map(int, budget.groups())
    remaining = limit - input_tokens
    if remaining >= requested:
        return None
    return remaining


class TrajectoryTextModel(LitellmTextbasedModel):
    """Keep model output before format-error feedback in mini-swe-agent 2.4.6."""

    abort_exceptions = [*LitellmTextbasedModel.abort_exceptions, LimitsExceeded]

    def _query(self, messages, **kwargs):
        try:
            return super()._query(messages, **kwargs)
        except BadRequestError as error:
            message = " ".join(str(error).split())
            remaining = _remaining_budget(message)
            if remaining is None:
                raise
            if remaining <= 0:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": "Session token budget exhausted",
                        "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                    }
                ) from error
            options = self.config.model_kwargs | kwargs
            token_key = "max_tokens"
            if options.get("max_completion_tokens") is not None:
                token_key = "max_completion_tokens"
            # Retry within the server-reported token budget.
            return super()._query(messages, **(kwargs | {token_key: remaining}))

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict:
        """Return an action response or preserve it alongside format-error feedback."""
        try:
            return super().query(messages, **kwargs)
        except FormatError as error:
            feedback = error.messages[0]
            extra = feedback.get("extra", {})
            response = extra.get("response")
            if not isinstance(response, dict):
                raise RuntimeError(
                    f"Expected format-error feedback to include the original response mapping, got {type(response).__name__}; extra keys={list(extra)}"
                ) from error
            original = response["choices"][0]["message"]
            assistant = {"role": "assistant", "content": original.get("content") or ""}
            if original.get("reasoning_content") is not None:
                assistant["reasoning_content"] = original["reasoning_content"]
            # DefaultAgent charges the first message when handling FormatError.
            assistant["extra"] = {"cost": extra.pop("cost", 0.0), "response": response}
            error.messages = (assistant, *error.messages)
            raise
