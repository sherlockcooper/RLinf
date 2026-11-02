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

"""Real CPU-offload optimizer restore and continuation checks."""

import copy

import pytest
import torch

pytest.importorskip("megatron")
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (  # noqa: E402
    HybridDeviceOptimizer,
)

from rlinf.hybrid_engines.megatron.checkpoint_utils import (  # noqa: E402
    reuse_hybrid_optimizer_buffers,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA offload")
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_restore_preserves_state_and_next_update_without_rebuilding_buffers(fraction):
    def create():
        params = [
            torch.nn.Parameter(torch.arange(16, device="cuda", dtype=torch.bfloat16))
            for _ in range(2)
        ]
        opt = HybridDeviceOptimizer(
            params,
            offload_fraction=fraction,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.Adam,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=False,
            lr=0.01,
        )
        return params, opt

    def step(params, opt):
        for index, param in enumerate(params):
            param.grad = torch.full_like(param, 0.125 * (index + 1))
        opt.step()
        torch.cuda.synchronize()
        opt.zero_grad()

    params, source = create()
    step(params, source)
    saved = copy.deepcopy(source.state_dict())
    expected_params, expected = create()
    restored_params, restored = create()
    for dst, ref in zip(expected_params, params):
        dst.data.copy_(ref)
    for dst, ref in zip(restored_params, params):
        dst.data.copy_(ref)
    expected.load_state_dict(copy.deepcopy(saved))
    before = {
        id(p): value.data_ptr() for p, value in restored.param_to_fp32_param.items()
    }
    with reuse_hybrid_optimizer_buffers(restored):
        restored.load_state_dict(copy.deepcopy(saved))
    assert "_init_sub_optimizers" not in restored.__dict__
    assert before == {
        id(p): value.data_ptr() for p, value in restored.param_to_fp32_param.items()
    }
    for left, right in zip(expected.state.values(), restored.state.values()):
        for key in ["step", "exp_avg", "exp_avg_sq", "master_param"]:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    step(expected_params, expected)
    step(restored_params, restored)
    for left, right in zip(expected_params, restored_params):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(expected.state.values(), restored.state.values()):
        for key in ["step", "exp_avg", "exp_avg_sq", "master_param"]:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
