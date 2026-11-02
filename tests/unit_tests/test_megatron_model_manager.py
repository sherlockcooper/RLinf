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

"""Checkpoint metadata compatibility with Megatron."""

from argparse import Namespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("megatron")

from rlinf.hybrid_engines.megatron.megatron_model_manager import (  # noqa: E402
    _preprocess_checkpoint_common_state,
)


@pytest.mark.parametrize("config_type", [Namespace, OmegaConf.create])
def test_checkpoint_metadata_keeps_optimizer_flags_and_original_state(config_type):
    values = {
        "use_distributed_optimizer": True,
        "rank": 3,
        "local_rank": 3,
        "seed": 1234,
    }
    args = Namespace(**values) if config_type is Namespace else config_type(values)
    original = {"args": args, "iteration": 1}

    result = _preprocess_checkpoint_common_state(original)

    assert result["args"] == {"use_distributed_optimizer": True, "seed": 1234}
    assert result["iteration"] == 1
    assert original["args"] is args
    assert args.rank == 3
    assert args.local_rank == 3
