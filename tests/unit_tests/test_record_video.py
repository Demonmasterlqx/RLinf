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
import pytest
from omegaconf import OmegaConf

from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.envs.wrappers.tactile_heatmap import (
    render_tactile_marker_motion_heatmap,
)


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


def test_tactile_heatmap_zero_input_returns_black_rgb_image():
    tactile = np.zeros((1, 9, 198, 2), dtype=np.float32)

    frames = render_tactile_marker_motion_heatmap(
        tactile,
        output_size=(8, 8),
    )

    assert len(frames) == 1
    assert frames[0].shape == (8, 8, 3)
    assert frames[0].dtype == np.uint8
    assert np.max(frames[0]) == 0


def test_tactile_heatmap_uses_latest_history_frame():
    tactile = np.zeros((1, 9, 198, 2), dtype=np.float32)
    tactile[0, 1, 0] = [10.0, 0.0]

    frames = render_tactile_marker_motion_heatmap(
        tactile,
        history_index=-1,
        output_size=(8, 8),
    )
    assert np.max(frames[0]) == 0

    tactile[0, -1, 0] = [1.0, 0.0]
    frames = render_tactile_marker_motion_heatmap(
        tactile,
        history_index=-1,
        output_size=(8, 8),
    )
    assert np.max(frames[0]) > 0


def test_tactile_heatmap_rejects_unexpected_marker_count():
    tactile = np.zeros((1, 9, 197, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="expected 198 markers"):
        render_tactile_marker_motion_heatmap(tactile)


def test_tactile_heatmap_normalizes_each_env_independently():
    tactile = np.zeros((2, 9, 198, 2), dtype=np.float32)
    tactile[0, -1, 0] = [1.0, 0.0]
    tactile[1, -1, 0] = [100.0, 0.0]

    frames = render_tactile_marker_motion_heatmap(
        tactile,
        history_index=-1,
        output_size=(8, 8),
    )

    assert np.max(frames[0]) == np.max(frames[1])


def test_record_video_writes_tactile_heatmap_and_combined_video(
    monkeypatch, tmp_path
):
    saved = []
    cfg = OmegaConf.create(
        {
            "video_base_dir": str(tmp_path),
            "fps": 20,
            "info_on_video": False,
            "image_keys": ["main_images", "wrist_images"],
            "image_names": ["agentview", "eye_in_hand"],
            "tactile_heatmap": {
                "enabled": True,
                "source_key": "tactile_marker_motion",
                "name": "tactile_heatmap",
                "history_index": -1,
                "marker_grid": [9, 11],
                "sensor_count": 2,
                "output_size": [8, 8],
                "clip_quantile": 0.99,
            },
            "composite_views": ["agentview", "eye_in_hand", "tactile_heatmap"],
            "composite_name": "combined",
        }
    )
    wrapper = RecordVideo(_DummyEnv(), cfg)

    def fake_submit(frames, mp4_path):
        saved.append((frames, mp4_path))
        return _completed_future()

    monkeypatch.setattr(wrapper, "_submit_save", fake_submit)

    tactile = np.zeros((1, 9, 198, 2), dtype=np.float32)
    tactile[0, -1, 0] = [1.0, 0.0]
    wrapper.add_new_frames(
        {
            "main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "wrist_images": np.ones((1, 8, 8, 3), dtype=np.uint8),
            "tactile_marker_motion": tactile,
        }
    )
    wrapper.flush_video()

    assert [path for _, path in saved] == [
        str(tmp_path / "seed_42" / "agentview" / "0.mp4"),
        str(tmp_path / "seed_42" / "eye_in_hand" / "0.mp4"),
        str(tmp_path / "seed_42" / "tactile_heatmap" / "0.mp4"),
        str(tmp_path / "seed_42" / "combined" / "0.mp4"),
    ]
    assert [len(frames) for frames, _ in saved] == [1, 1, 1, 1]
    assert saved[2][0][0].shape == (8, 8, 3)
    assert saved[3][0][0].shape == (8, 24, 3)


def test_record_video_missing_tactile_heatmap_source_warns_without_blocking_camera(
    monkeypatch, tmp_path
):
    saved = []
    cfg = OmegaConf.create(
        {
            "video_base_dir": str(tmp_path),
            "fps": 20,
            "info_on_video": False,
            "image_keys": ["main_images"],
            "image_names": ["agentview"],
            "tactile_heatmap": {
                "enabled": True,
                "source_key": "tactile_marker_motion",
                "name": "tactile_heatmap",
                "output_size": [8, 8],
            },
            "composite_views": ["agentview", "tactile_heatmap"],
            "composite_name": "combined",
        }
    )
    wrapper = RecordVideo(_DummyEnv(), cfg)

    def fake_submit(frames, mp4_path):
        saved.append((frames, mp4_path))
        return _completed_future()

    monkeypatch.setattr(wrapper, "_submit_save", fake_submit)

    with pytest.warns(UserWarning) as warnings_record:
        wrapper.add_new_frames(
            {"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)}
        )
        wrapper.flush_video()
    assert any(
        "tactile heatmap source key" in str(warning.message)
        for warning in warnings_record
    )

    assert [path for _, path in saved] == [
        str(tmp_path / "seed_42" / "agentview" / "0.mp4"),
    ]
