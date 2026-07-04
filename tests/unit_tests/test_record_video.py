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

from concurrent.futures import Future

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.wrappers.record_video import RecordVideo


class _DummyEnv:
    seed = 42
    metadata = {"render_fps": 20}

    @property
    def is_start(self):
        return False

    @is_start.setter
    def is_start(self, value):
        pass


def _completed_future() -> Future:
    future = Future()
    future.set_result(None)
    return future


def test_record_video_legacy_single_view_path_is_unchanged(monkeypatch, tmp_path):
    saved = []
    cfg = OmegaConf.create(
        {
            "video_base_dir": str(tmp_path),
            "fps": 20,
            "info_on_video": False,
        }
    )
    wrapper = RecordVideo(_DummyEnv(), cfg)

    def fake_submit(frames, mp4_path):
        saved.append((frames, mp4_path))
        return _completed_future()

    monkeypatch.setattr(wrapper, "_submit_save", fake_submit)

    wrapper.add_new_frames(
        {"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)}
    )
    wrapper.flush_video()

    assert len(saved) == 1
    assert saved[0][1] == str(tmp_path / "seed_42" / "0.mp4")
    assert len(saved[0][0]) == 1


def test_record_video_writes_configured_views_to_named_subdirs(monkeypatch, tmp_path):
    saved = []
    cfg = OmegaConf.create(
        {
            "video_base_dir": str(tmp_path),
            "fps": 20,
            "info_on_video": False,
            "image_keys": ["main_images", "wrist_images"],
            "image_names": ["agentview", "eye_in_hand"],
        }
    )
    wrapper = RecordVideo(_DummyEnv(), cfg)

    def fake_submit(frames, mp4_path):
        saved.append((frames, mp4_path))
        return _completed_future()

    monkeypatch.setattr(wrapper, "_submit_save", fake_submit)

    wrapper.add_new_frames(
        {
            "main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "wrist_images": np.ones((1, 8, 8, 3), dtype=np.uint8),
        }
    )
    wrapper.flush_video()

    assert [path for _, path in saved] == [
        str(tmp_path / "seed_42" / "agentview" / "0.mp4"),
        str(tmp_path / "seed_42" / "eye_in_hand" / "0.mp4"),
    ]
    assert [len(frames) for frames, _ in saved] == [1, 1]
