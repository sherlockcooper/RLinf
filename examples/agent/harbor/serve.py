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

"""Start the independent Harbor service on a CPU host with Docker."""

import argparse
import importlib
import os
from pathlib import Path

from rlinf.workers.agent.harbor.backend import HarborExecutor
from rlinf.workers.agent.harbor.server import create_app


def main():
    import uvicorn
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    adapter = config.pop("environment_adapter", None)
    if adapter:
        module, function = adapter.split(":", 1)
        getattr(importlib.import_module(module), function)()
    listen = config.pop("listen", {"host": "127.0.0.1", "port": 30200})
    executor = HarborExecutor(**config.pop("executor"))
    app = create_app(executor, api_key=os.environ.get("RLINF_HARBOR_API_KEY"), **config)
    uvicorn.run(app, **listen, workers=1)


if __name__ == "__main__":
    main()
