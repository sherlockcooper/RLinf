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

"""Loss and gradient contracts for reinforcement learning algorithms."""

import pytest
import torch

from rlinf.algorithms.registry import policy_loss


@pytest.mark.parametrize("rows", [[True, True], [False, True], [False, False]])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_grpo_empty_microbatch_preserves_metrics_and_gradients(rows, diagnostics):
    mask = torch.tensor([[row, row, False] for row in rows])
    results = []
    for fast in [False, True]:
        logprobs = torch.full((2, 3), -1.0, requires_grad=True)
        loss, metrics = policy_loss(
            task_type="reasoning",
            loss_type="actor",
            logprobs=logprobs,
            old_logprobs=torch.full((2, 3), -1.05),
            advantages=torch.ones(2, 3),
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            loss_mask=mask,
            fast_path_zero_loss_mask=fast,
            log_logprob_diagnostics=diagnostics,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.equal(logprobs.grad[~mask], torch.zeros_like(logprobs.grad[~mask]))
        if mask.any():
            assert logprobs.grad[mask].abs().sum() > 0
        results.append((loss.detach(), metrics, logprobs.grad))
    expected, actual = results
    torch.testing.assert_close(actual[0], expected[0])
    assert list(actual[1]) == list(expected[1])
    for key in expected[1]:
        torch.testing.assert_close(actual[1][key], expected[1][key])
    torch.testing.assert_close(actual[2], expected[2])


def test_grpo_empty_microbatch_has_same_metrics_as_active_microbatch():
    metrics = []
    for active in [False, True]:
        _, result = policy_loss(
            task_type="reasoning",
            loss_type="actor",
            logprobs=torch.full((1, 3), -1.0, requires_grad=True),
            old_logprobs=torch.full((1, 3), -1.0),
            advantages=torch.ones(1, 3),
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            loss_mask=torch.full((1, 3), active),
            fast_path_zero_loss_mask=True,
        )
        metrics.append(result)
    assert list(metrics[0]) == list(metrics[1])
    for key in metrics[0]:
        assert torch.isfinite(torch.stack([row[key] for row in metrics]).mean())
