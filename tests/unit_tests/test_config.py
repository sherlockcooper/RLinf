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

"""Rollout backend configuration requirements."""

import pytest
from omegaconf import OmegaConf

from rlinf.config import validate_rollout_cfg


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_rollout_accepts_selected_backend_without_inactive_config(backend):
    cfg = OmegaConf.create(
        {
            "model": {"model_type": "qwen2.5", "model_path": "/models/test"},
            "rollout_backend": backend,
            "return_logprobs": True,
            backend: {},
        }
    )
    OmegaConf.set_struct(cfg, True)
    result = validate_rollout_cfg(cfg, OmegaConf.create({}))
    assert result.rollout_backend == backend
    assert result[backend].attention_backend == (
        "triton" if backend == "sglang" else "FLASH_ATTN"
    )
    assert ("vllm" if backend == "sglang" else "sglang") not in result


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_rollout_requires_selected_backend_config(backend):
    inactive = "vllm" if backend == "sglang" else "sglang"
    cfg = OmegaConf.create(
        {
            "model": {"model_type": "qwen2.5", "model_path": "/models/test"},
            "rollout_backend": backend,
            "return_logprobs": True,
            inactive: {},
        }
    )
    with pytest.raises(AssertionError, match=f"{backend} config must be specified"):
        validate_rollout_cfg(cfg, OmegaConf.create({}))
