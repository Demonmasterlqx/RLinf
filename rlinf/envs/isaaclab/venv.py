# Copyright 2025 The RLinf Authors.
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

import traceback
from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp

from .utils import CloudpickleWrapper

_STARTUP_READY = "startup_ready"
_STARTUP_ERROR = "startup_error"
_STARTUP_POLL_SECONDS = 0.1
_FAILED_STARTUP_JOIN_SECONDS = 1.0


def _torch_worker(
    child_remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    action_queue: mp.Queue,
    obs_queue: mp.Queue,
    reset_idx_queue: mp.Queue,
):
    parent_remote.close()
    env_fn = env_fn_wrapper.x
    try:
        isaac_env, sim_app = env_fn()
        device = isaac_env.device
    except BaseException as error:
        error_payload = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            child_remote.send((_STARTUP_ERROR, error_payload))
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            child_remote.close()
        return

    try:
        child_remote.send((_STARTUP_READY, None))
        while True:
            try:
                cmd = child_remote.recv()
            except EOFError:
                child_remote.close()
                break
            if cmd == "reset":
                reset_index, reset_seed = reset_idx_queue.get()
                if reset_index is None:
                    reset_result = isaac_env.reset(seed=reset_seed)
                else:
                    reset_result = isaac_env.reset(
                        seed=reset_seed, env_ids=reset_index.to(device)
                    )
                obs_queue.put(reset_result)
            elif cmd == "step":
                input_action = action_queue.get()
                step_result = isaac_env.step(input_action)
                obs_queue.put(step_result)
            elif cmd == "close":
                isaac_env.close()
                child_remote.close()
                sim_app.close()
                break
            elif cmd == "device":
                child_remote.send(isaac_env.device)
            else:
                child_remote.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        child_remote.close()
    finally:
        try:
            isaac_env.close()
        except Exception as e:
            print(f"IsaacLab Env Closed with error: {e}")


class SubProcIsaacLabEnv:
    def __init__(self, env_fn):
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        self._closed = False
        self.parent_remote, self.child_remote = ctx.Pipe(duplex=True)
        self.action_queue = ctx.Queue()
        self.obs_queue = ctx.Queue()
        self.reset_idx = ctx.Queue()
        args = (
            self.child_remote,
            self.parent_remote,
            CloudpickleWrapper(env_fn),
            self.action_queue,
            self.obs_queue,
            self.reset_idx,
        )
        self.isaac_lab_process = ctx.Process(
            target=_torch_worker, args=args, daemon=True
        )
        try:
            self.isaac_lab_process.start()
            self.child_remote.close()
            self._wait_for_startup()
        except BaseException:
            self._cleanup_failed_startup()
            raise

    def _process_is_alive(self):
        try:
            return self.isaac_lab_process.is_alive()
        except (AssertionError, ValueError):
            return False

    def _startup_exit_error(self):
        try:
            self.isaac_lab_process.join(timeout=_STARTUP_POLL_SECONDS)
        except (AssertionError, ValueError):
            pass
        return RuntimeError(
            "IsaacLab subprocess exited before startup handshake "
            f"with exit code {self.isaac_lab_process.exitcode}"
        )

    def _wait_for_startup(self):
        while True:
            if self.parent_remote.poll(_STARTUP_POLL_SECONDS):
                try:
                    status, payload = self.parent_remote.recv()
                except EOFError:
                    raise self._startup_exit_error() from None
                if status == _STARTUP_READY:
                    return
                if status == _STARTUP_ERROR:
                    raise RuntimeError(
                        "IsaacLab subprocess initialization failed with "
                        f"{payload['exception_type']}: {payload['message']}\n"
                        f"Child traceback:\n{payload['traceback']}"
                    )
                raise RuntimeError(
                    f"IsaacLab subprocess sent invalid startup status: {status!r}"
                )
            if not self._process_is_alive():
                if self.parent_remote.poll():
                    continue
                raise self._startup_exit_error()

    def _close_parent_resources(self):
        for remote in (self.parent_remote, self.child_remote):
            try:
                remote.close()
            except (OSError, ValueError):
                pass
        for queue in (self.action_queue, self.obs_queue, self.reset_idx):
            try:
                queue.cancel_join_thread()
            except (OSError, ValueError):
                pass
            try:
                queue.close()
            except (OSError, ValueError):
                pass
        self._closed = True

    def _cleanup_failed_startup(self):
        if self._process_is_alive():
            self.isaac_lab_process.terminate()
        try:
            self.isaac_lab_process.join(timeout=_FAILED_STARTUP_JOIN_SECONDS)
        except (AssertionError, ValueError):
            pass
        if self._process_is_alive():
            self.isaac_lab_process.kill()
            self.isaac_lab_process.join()
        self._close_parent_resources()

    def reset(self, seed=None, env_ids=None):
        self.parent_remote.send("reset")
        self.reset_idx.put((env_ids, seed))
        obs, info = self.obs_queue.get()
        return obs, info

    def step(self, action: torch.Tensor):
        """
        action : (bs, action_dim)
        """
        self.parent_remote.send("step")
        self.action_queue.put(action)
        env_step_result = self.obs_queue.get()
        return env_step_result

    def close(self):
        if getattr(self, "_closed", False):
            return
        close_sent = False
        if self._process_is_alive() and not self.parent_remote.closed:
            try:
                self.parent_remote.send("close")
                close_sent = True
            except (BrokenPipeError, EOFError, OSError):
                pass
        if close_sent:
            self.isaac_lab_process.join()
        else:
            self._cleanup_failed_startup()
            return
        self._close_parent_resources()

    def device(self):
        self.parent_remote.send("device")
        return self.parent_remote.recv()
