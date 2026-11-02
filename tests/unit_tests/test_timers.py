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

"""Wall-clock training budgets and step-boundary checkpoint decisions."""

import pytest

from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer


def test_hour_budget(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("rlinf.utils.timers.time.monotonic", lambda: now[0])
    timer = Timer.from_hours(0.5)
    timer.start_time()
    now[0] += 1799
    assert not timer.is_finished()
    now[0] += 1
    assert timer.is_finished()
    _, save, _ = check_progress(3, 100, -1, 5, 0, timer.is_finished())
    assert save


def test_unlimited_budget():
    timer = Timer.from_hours(None)
    timer.start_time()
    assert not timer.is_finished()


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf"), True])
def test_invalid_budget(hours):
    with pytest.raises(ValueError, match="max_duration_hours"):
        Timer.from_hours(hours)
