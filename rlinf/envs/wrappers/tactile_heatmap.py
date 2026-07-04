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

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _normalize_magnitude(values: np.ndarray, clip_quantile: float) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    max_value = float(np.max(values))
    if max_value <= 0:
        return np.zeros_like(values, dtype=np.float32)

    if 0 < clip_quantile < 1:
        scale = float(np.quantile(values, clip_quantile))
    else:
        scale = max_value
    if scale <= 0:
        scale = max_value
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def _render_sensor_grid(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    if np.max(grid) <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    gray = (np.clip(grid, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(heatmap_rgb, (width, height), interpolation=cv2.INTER_NEAREST)


def render_tactile_marker_motion_heatmap(
    marker_motion: Any,
    *,
    history_index: int = -1,
    marker_grid: tuple[int, int] | list[int] = (9, 11),
    sensor_count: int = 2,
    output_size: tuple[int, int] | list[int] = (256, 256),
    clip_quantile: float = 0.99,
) -> list[np.ndarray]:
    """Render Tabero marker-motion fields as per-env RGB heatmap frames.

    Expected input shape is ``(N, H, S * rows * cols, 2)``. ``H`` includes the
    reference frame at index 0 plus marker history. The rendered intensity is
    the latest marker displacement magnitude relative to the reference frame.
    """

    motion = _to_numpy(marker_motion).astype(np.float32, copy=False)
    if motion.ndim != 4:
        raise ValueError(
            "tactile marker motion must have shape (N, history, markers, xy); "
            f"got {tuple(motion.shape)}."
        )

    num_envs, history_len, markers, xy = motion.shape
    if xy != 2:
        raise ValueError(f"tactile marker motion xy dimension must be 2, got {xy}.")
    if history_len <= 0:
        raise ValueError("tactile marker motion history length must be positive.")

    rows, cols = (int(marker_grid[0]), int(marker_grid[1]))
    sensor_count = int(sensor_count)
    expected_markers = rows * cols * sensor_count
    if markers != expected_markers:
        raise ValueError(
            f"expected {expected_markers} markers from grid={rows}x{cols} "
            f"and sensor_count={sensor_count}, got {markers}."
        )

    if not -history_len <= int(history_index) < history_len:
        raise ValueError(
            f"history_index {history_index} is out of range for history length "
            f"{history_len}."
        )

    out_height, out_width = (int(output_size[0]), int(output_size[1]))
    if out_height <= 0 or out_width <= 0:
        raise ValueError(f"output_size must be positive, got {tuple(output_size)}.")

    selected = motion[:, int(history_index)]
    if history_len > 1 and int(history_index) != 0:
        selected = selected - motion[:, 0]

    magnitude = np.linalg.norm(selected, axis=-1)
    magnitude = np.stack(
        [
            _normalize_magnitude(magnitude[env_id], float(clip_quantile))
            for env_id in range(num_envs)
        ],
        axis=0,
    )
    magnitude = magnitude.reshape(num_envs, sensor_count, rows, cols)

    panel_widths = [out_width // sensor_count] * sensor_count
    panel_widths[-1] += out_width - sum(panel_widths)
    frames: list[np.ndarray] = []
    for env_id in range(num_envs):
        panels = [
            _render_sensor_grid(
                magnitude[env_id, sensor_id],
                out_height,
                panel_widths[sensor_id],
            )
            for sensor_id in range(sensor_count)
        ]
        frames.append(np.concatenate(panels, axis=1).astype(np.uint8, copy=False))
    return frames
