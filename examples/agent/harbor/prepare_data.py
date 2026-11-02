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

"""Export prepared Harbor SWE tasks and their RLinf dataset index."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

# SWE-Gym/SWE-Gym-Lite train split, revision f70b1a29ab120eb0a0ee7a1deb029825e735b2b0.
SWE_GYM_LITE_TASKS = (
    "Project-MONAI__MONAI-5908",
    "bokeh__bokeh-13289",
    "conan-io__conan-12397",
    "conan-io__conan-15422",
    "getmoto__moto-4990",
    "iterative__dvc-1661",
    "modin-project__modin-6298",
    "pandas-dev__pandas-56849",
)


def export_tasks(
    source_tasks: Path, output_dir: Path, instance_ids: list[str], dataset_id: str
) -> list[dict[str, Any]]:
    """Copy selected tasks, exclude oracle solutions, and write their dataset index.

    Returns:
        The same records written to output_dir/train.jsonl.
    """
    source_tasks = source_tasks.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError(f"Expected unique instance IDs, got {instance_ids!r}")
    for instance_id in instance_ids:
        task = (source_tasks / instance_id).resolve()
        if task.parent != source_tasks or not (task / "task.toml").is_file():
            raise ValueError(
                f"Expected {instance_id!r} directly under {source_tasks} with task.toml, got path={task}, task.toml exists={(task / 'task.toml').is_file()}"
            )
        instruction = (task / "instruction.md").read_text()
        target = output_dir / "tasks" / instance_id
        shutil.copytree(
            task, target, ignore=shutil.ignore_patterns("solution", "__pycache__")
        )
        records.append(
            {
                "instance_id": instance_id,
                "dataset_id": dataset_id,
                "prompt": instruction,
                "solutions": {"instance_id": instance_id, "dataset_id": dataset_id},
            }
        )
    (output_dir / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    return records


def main() -> None:
    """Export prepared Harbor tasks and a dataset index from CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        default=list(SWE_GYM_LITE_TASKS),
        help="Task IDs to export (default: SWE_GYM_LITE_TASKS).",
    )
    parser.add_argument("--dataset-id", default="swe-gym-lite")
    args = parser.parse_args()
    export_tasks(args.source_tasks, args.output_dir, args.instance_ids, args.dataset_id)


if __name__ == "__main__":
    main()
