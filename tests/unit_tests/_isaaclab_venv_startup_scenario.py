# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing
import os
import sys
import time

from rlinf.envs.isaaclab.venv import SubProcIsaacLabEnv


def raise_horizon_mismatch():
    raise ValueError(
        "Tabero IsaacLab episode horizon mismatch: expected 300 steps, got 160."
    )


def exit_without_handshake():
    os._exit(23)


SCENARIOS = {
    "value_error": raise_horizon_mismatch,
    "child_exit": exit_without_handshake,
}


def _resource_state(env):
    queues = (env.action_queue, env.obs_queue, env.reset_idx)
    return {
        "pipes_closed": env.parent_remote.closed and env.child_remote.closed,
        "queues_closed": all(queue._closed for queue in queues),
    }


def _repeat_cleanup(env):
    errors = []
    for _ in range(2):
        try:
            env.close()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    return errors


def main():
    scenario = sys.argv[1]
    env = SubProcIsaacLabEnv.__new__(SubProcIsaacLabEnv)
    started = time.monotonic()
    try:
        env.__init__(SCENARIOS[scenario])
        env.reset(seed=11)
    except Exception as error:
        elapsed = time.monotonic() - started
        process = env.isaac_lab_process
        report = {
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "elapsed": elapsed,
            "child_alive": process.is_alive(),
            "child_exitcode": process.exitcode,
            "resources_closed_by_init": _resource_state(env),
            "repeat_cleanup_errors": _repeat_cleanup(env),
            "active_child_pids": [
                child.pid for child in multiprocessing.active_children()
            ],
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0

    _repeat_cleanup(env)
    print(json.dumps({"error": "constructor unexpectedly succeeded"}), flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
