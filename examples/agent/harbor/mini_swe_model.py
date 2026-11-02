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

from litellm.exceptions import BadRequestError
from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel


class TrajectoryTextModel(LitellmTextbasedModel):
    """Keep model output before format-error feedback in mini-swe-agent 2.4.6."""

    abort_exceptions = [*LitellmTextbasedModel.abort_exceptions, LimitsExceeded]

    def _query(self, messages, **kwargs):
        try:
            return super()._query(messages, **kwargs)
        except BadRequestError as error:
            message = " ".join(str(error).split())
            input_limit = re.search(
                r"The input \((\d+) tokens\) is longer than the model's "
                r"context length \((\d+) tokens\)",
                message,
            )
            exhausted = "Session token budget exhausted" in message or (
                input_limit is not None and int(input_limit[1]) >= int(input_limit[2])
            )
            if not exhausted:
                budget = re.search(
                    r"Requested token count exceeds the model's maximum context "
                    r"length of (\d+) tokens\. You requested a total of \d+ tokens: "
                    r"(\d+) tokens from the input messages and (\d+) tokens "
                    r"for the completion",
                    message,
                )
                if budget is None:
                    raise
                limit, input_tokens, requested = map(int, budget.groups())
                remaining = limit - input_tokens
                if remaining > 0:
                    if remaining >= requested:
                        raise
                    options = self.config.model_kwargs | kwargs
                    token_key = (
                        "max_completion_tokens"
                        if options.get("max_completion_tokens") is not None
                        else "max_tokens"
                    )
                    # A rejected request generated no tokens; retry within its exact budget.
                    return super()._query(messages, **(kwargs | {token_key: remaining}))
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "Session token budget exhausted",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            ) from error

    def query(self, messages, **kwargs):
        try:
            return super().query(messages, **kwargs)
        except FormatError as error:
            feedback = error.messages[0]
            extra = feedback.get("extra", {})
            response = extra.get("response")
            if not isinstance(response, dict):
                raise RuntimeError(
                    "Format error is missing the original model response"
                ) from error
            original = response["choices"][0]["message"]
            assistant = {"role": "assistant", "content": original.get("content") or ""}
            if original.get("reasoning_content") is not None:
                assistant["reasoning_content"] = original["reasoning_content"]
            # DefaultAgent charges the first message when handling FormatError.
            assistant["extra"] = {"cost": extra.pop("cost", 0.0), "response": response}
            error.messages = (assistant, *error.messages)
            raise
