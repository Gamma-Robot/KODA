#!/usr/bin/env python3
"""EgoHumanoid-style binary hand open/close from native 3D finger keypoints.

Matches OpenDriveLab/EgoHumanoid ``add_hand_status.py``: fingertip distance +
finger curvature, enhancement, then a 0-1-0 square wave. Joint indices are
HaWoR/MANO 21 rather than PICO 26.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

MANO_TIPS = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Little": 20}
MANO_JOINTS = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Little": [17, 18, 19, 20],
}

def tip_avg_distance(hand: np.ndarray) -> np.ndarray:
    names = list(MANO_TIPS)
    tips = np.stack([hand[:, MANO_TIPS[name]] for name in names], axis=0)
    n_fingers = tips.shape[0]
    avg = np.zeros(hand.shape[0], dtype=np.float64)
    for i in range(n_fingers):
        distances = [
            np.linalg.norm(tips[i] - tips[j], axis=1)
            for j in range(n_fingers)
            if i != j
        ]
        avg += np.mean(np.stack(distances, axis=0), axis=0)
    return avg / n_fingers


def finger_curvature(hand: np.ndarray) -> np.ndarray:
    values = []
    for indices in MANO_JOINTS.values():
        points = np.stack([hand[:, idx] for idx in indices], axis=0)
        path = np.zeros(hand.shape[0], dtype=np.float64)
        for joint in range(len(indices) - 1):
            path += np.linalg.norm(points[joint + 1] - points[joint], axis=1)
        chord = np.linalg.norm(points[-1] - points[0], axis=1)
        values.append(np.where(chord > 1e-6, path / chord, 1.0))
    return np.mean(np.stack(values, axis=0), axis=0)


def _normalize(raw: np.ndarray) -> np.ndarray:
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    if hi <= lo:
        return np.zeros_like(raw, dtype=np.float64)
    return (raw - lo) / (hi - lo)


def weighted_trajectory(hand: np.ndarray) -> np.ndarray:
    tip = _normalize(tip_avg_distance(hand))
    curv = _normalize(finger_curvature(hand))
    return 0.5 * (1.0 - tip) + 0.5 * curv


def enhance_trajectory(trajectory: np.ndarray, offset: float = 0.25) -> np.ndarray:
    result = trajectory.copy()
    mean_height = float(np.clip(np.mean(trajectory) + offset, 0.0, 1.0))
    high = trajectory >= mean_height
    if np.any(high) and (1.0 - mean_height) > 0:
        normalized = (trajectory[high] - mean_height) / (1.0 - mean_height)
        enhanced = 1.0 - (1.0 - normalized) ** 100.0
        result[high] = mean_height + enhanced * (1.0 - mean_height)
    low = trajectory < mean_height
    if np.any(low) and mean_height > 0:
        normalized = trajectory[low] / mean_height
        result[low] = (normalized ** 100.0) * mean_height
    return result


def fit_square_wave_010(trajectory: np.ndarray) -> tuple[int, int, np.ndarray]:
    n_frame = int(trajectory.shape[0])
    weights = np.ones(n_frame, dtype=np.float64)
    weights[trajectory >= 0.8] = 1e18

    def objective(params: np.ndarray) -> float:
        start = int(np.floor(params[0]))
        end = int(np.floor(params[1]))
        start = max(0, min(n_frame - 2, start))
        end = max(start + 1, min(n_frame, end))
        wave = np.zeros(n_frame, dtype=np.float64)
        wave[start:end] = 1.0
        return float(np.mean(weights * (wave - trajectory) ** 2))

    best_mse = np.inf
    best_start, best_end = n_frame // 3, (2 * n_frame) // 3
    n_init = 12
    for i in range(n_init):
        for j in range(i + 1, n_init):
            x0 = int(i * (n_frame - 1) / max(n_init - 1, 1))
            y0 = int(j * (n_frame - 1) / max(n_init - 1, 1))
            result = minimize(objective, [x0, y0], method="Powell")
            if result.success and float(result.fun) < best_mse:
                best_mse = float(result.fun)
                best_start = int(np.floor(result.x[0]))
                best_end = int(np.floor(result.x[1]))
    best_start = max(0, min(n_frame - 2, best_start))
    best_end = max(best_start + 1, min(n_frame, best_end))
    wave = np.zeros(n_frame, dtype=np.float64)
    wave[best_start:best_end] = 1.0
    return best_start, best_end, wave


def past_only_sample(source: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    index = np.searchsorted(source_t, target_t, side="right") - 1
    index = np.clip(index, 0, source.shape[0] - 1)
    return source[index]


def compute_native_close_wave(joints: np.ndarray) -> np.ndarray:
    analog = enhance_trajectory(weighted_trajectory(np.asarray(joints, dtype=np.float64)))
    _start, _end, wave = fit_square_wave_010(analog)
    return wave.astype(np.float32)


def compute_hand_status(
    joints: np.ndarray,
    timestamps: np.ndarray,
    target_times: np.ndarray,
) -> dict[str, Any]:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    timestamps = timestamps - timestamps[0]
    wave = compute_native_close_wave(joints)
    analog = enhance_trajectory(weighted_trajectory(np.asarray(joints, dtype=np.float64)))
    start = int(np.flatnonzero(wave >= 0.5)[0]) if np.any(wave >= 0.5) else 0
    closed = np.flatnonzero(wave >= 0.5)
    end = int(closed[-1] + 1) if closed.size else 0
    close_50 = past_only_sample(wave, timestamps, target_times).astype(np.float32)
    analog_50 = past_only_sample(analog, timestamps, target_times).astype(np.float32)
    onset_s = float(timestamps[int(closed[0])]) if closed.size else None
    release_s = float(timestamps[int(closed[-1])]) if closed.size else None
    return {
        "close": close_50,
        "analog": analog_50,
        "wave_30": wave.astype(np.float32),
        "onset_s": onset_s,
        "release_s": release_s,
        "onset_frame_30": int(start) if closed.size else None,
        "release_frame_30": int(end) if closed.size else None,
    }
