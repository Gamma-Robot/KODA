# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure numpy coordinate/rotation helpers for viz."""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross products: skew(v) @ x == np.cross(v, x)."""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64)


def rotation_matrix_from_two_vec(v_from: np.ndarray, v_to: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return R such that R @ v_from ~= v_to (both treated as 3D vectors).

    Uses a Rodrigues-style construction, with special handling for near-parallel and near-opposite
    vectors for numerical stability.
    """
    a = np.asarray(v_from, dtype=np.float64).reshape(3)
    b = np.asarray(v_to, dtype=np.float64).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = b / nb

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))  # cos(theta)
    if c > 1.0 - eps:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + eps:
        # 180 deg rotation about any axis orthogonal to a:
        # R = -I + 2 * uu^T, where u is a unit axis orthogonal to a.
        axis_seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a, axis_seed))) > 0.9:
            axis_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(a, axis_seed)
        u = u / np.linalg.norm(u).clip(min=eps)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(u, u)

    v = np.cross(a, b)
    s2 = float(np.dot(v, v))  # ||v||^2 == sin^2(theta)
    K = skew(v)
    # R = I + K + K^2 * ((1 - c) / s^2)
    return np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / s2)
