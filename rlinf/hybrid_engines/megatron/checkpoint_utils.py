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

"""Bound CPU-offload checkpoint restore memory when parameter layouts are stable."""

from contextlib import ExitStack, contextmanager


@contextmanager
def reuse_hybrid_optimizer_buffers(optimizer):
    """Keep existing master buffers while upstream hooks restore optimizer state."""
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )

    def visit(value):
        if isinstance(value, HybridDeviceOptimizer):
            yield value
        elif hasattr(value, "chained_optimizers"):
            for inner in value.chained_optimizers:
                yield from visit(inner)
        elif hasattr(value, "optimizer"):
            yield from visit(value.optimizer)

    @contextmanager
    def reuse_initializer(inner):
        original = inner._init_sub_optimizers
        had_override = "_init_sub_optimizers" in inner.__dict__
        layout = tuple(tuple(id(p) for p in g["params"]) for g in inner.param_groups)

        def initialize():
            current = tuple(
                tuple(id(p) for p in g["params"]) for g in inner.param_groups
            )
            if current != layout:
                original()

        inner._init_sub_optimizers = initialize
        try:
            yield
        finally:
            if had_override:
                inner._init_sub_optimizers = original
            else:
                del inner._init_sub_optimizers

    with ExitStack() as stack:
        for inner in visit(optimizer):
            stack.enter_context(reuse_initializer(inner))
        yield
