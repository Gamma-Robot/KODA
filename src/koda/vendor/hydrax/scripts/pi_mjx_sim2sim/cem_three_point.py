"""Temporal CEM utilities for torso + dual-wrist tracking in MJX.

This module is intentionally separate from :mod:`cem_wrist`.  It keeps the
same sampling, local-quaternion tangent, interpolation, feasibility, and
candidate-selection semantics while extending every policy window to the
fixed link order ``torso, left wrist, right wrist``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from cem_wrist import (
    _normalize_quat,
    choose_candidate,
    local_tangent_from_quat,
    quat_error_magnitude,
    quat_from_local_tangent,
    quat_slerp_shortest,
)


DEFAULT_FUTURE_IDX: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
TRACKING_LINK_NAMES: tuple[str, ...] = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
LINK_COST_WEIGHTS = np.asarray((0.25, 1.0, 1.0), dtype=np.float64)
NUM_LINKS = len(TRACKING_LINK_NAMES)
POS_DIM = 3
ROT_DIM = 3
KNOT_FRAMES = len(DEFAULT_FUTURE_IDX)
NUM_KNOTS = 4
KNOT_STARTS: tuple[int, ...] = (0, 12, 24, 36)
HORIZON_STEPS = 36
EXECUTE_STEPS = 12


def window_nu(num_frames: int = KNOT_FRAMES) -> int:
    return int(num_frames) * NUM_LINKS * (POS_DIM + ROT_DIM)


def knot_nu(num_knots: int = NUM_KNOTS, num_frames: int = KNOT_FRAMES) -> int:
    return int(num_knots) * window_nu(num_frames)


@dataclass
class CEMConfig:
    num_samples: int = 256
    num_elites: int = 10
    num_frames: int = KNOT_FRAMES
    num_knots: int = NUM_KNOTS
    knot_starts: tuple[int, ...] = KNOT_STARTS
    horizon_steps: int = HORIZON_STEPS
    execute_steps: int = EXECUTE_STEPS
    sigma_start_pos: float = 0.03
    sigma_start_rot: float = 0.1
    sigma_min_pos: float = 0.005
    sigma_min_rot: float = 0.02
    explore_fraction: float = 0.5
    include_mean: bool = True
    w_rot: float = 0.1
    local_tangent_limit: float = math.pi
    link_cost_weights: tuple[float, ...] = (0.25, 1.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.knot_starts) != self.num_knots:
            raise ValueError("knot_starts must contain one start per knot")
        if tuple(self.knot_starts) != tuple(sorted(self.knot_starts)):
            raise ValueError("knot_starts must be sorted")
        if self.num_elites < 1 or self.num_elites > self.num_samples:
            raise ValueError("num_elites must be in [1, num_samples]")
        if self.horizon_steps <= 0 or self.execute_steps <= 0:
            raise ValueError("horizon_steps and execute_steps must be positive")
        if len(self.link_cost_weights) != NUM_LINKS:
            raise ValueError(f"link_cost_weights must have {NUM_LINKS} entries")
        if tuple(float(x) for x in self.link_cost_weights) != tuple(LINK_COST_WEIGHTS):
            raise ValueError("three-point cost weights are fixed to (0.25, 1.0, 1.0)")

    @property
    def window_nu(self) -> int:
        return window_nu(self.num_frames)

    @property
    def nu(self) -> int:
        return knot_nu(self.num_knots, self.num_frames)

    @property
    def adjusted_length(self) -> int:
        return max(self.knot_starts) + self.num_frames

    @property
    def num_explore(self) -> int:
        return int(self.num_samples * self.explore_fraction)


def pack_knots(knot_pos: np.ndarray, knot_tangent: np.ndarray) -> np.ndarray:
    """Pack ``(..., K, F, 3, 3)`` positions and local tangents."""
    pos = np.asarray(knot_pos, dtype=np.float64)
    tangent = np.asarray(knot_tangent, dtype=np.float64)
    if pos.shape != tangent.shape or pos.ndim < 4 or pos.shape[-3:] != (
        KNOT_FRAMES,
        NUM_LINKS,
        POS_DIM,
    ):
        raise ValueError("knot arrays must have shape (..., K, 6, 3, 3)")
    lead = pos.shape[:-4]
    num_knots = pos.shape[-4]
    return np.concatenate(
        [pos.reshape(*lead, num_knots, -1), tangent.reshape(*lead, num_knots, -1)],
        axis=-1,
    ).reshape(*lead, -1)


def unpack_knots(
    flat: np.ndarray,
    num_knots: int = NUM_KNOTS,
    num_frames: int = KNOT_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack ``(..., 432)`` into ``(..., K, F, 3, 3)`` arrays."""
    values = np.asarray(flat, dtype=np.float64)
    per_knot = int(num_frames) * NUM_LINKS * POS_DIM
    expected = int(num_knots) * 2 * per_knot
    if values.shape[-1] != expected:
        raise ValueError(f"expected last dimension {expected}, got {values.shape[-1]}")
    lead = values.shape[:-1]
    blocks = values.reshape(*lead, num_knots, 2 * per_knot)
    shape = (*lead, num_knots, num_frames, NUM_LINKS, POS_DIM)
    return blocks[..., :per_knot].reshape(shape), blocks[..., per_knot:].reshape(shape)


def knot_from_global_quat(
    knot_pos: np.ndarray,
    knot_quat: np.ndarray,
    reference_quat: np.ndarray,
) -> np.ndarray:
    return pack_knots(knot_pos, local_tangent_from_quat(reference_quat, knot_quat))


def make_sigma_vector(cfg: CEMConfig, use_start: bool = True) -> np.ndarray:
    sigma_pos = cfg.sigma_start_pos if use_start else cfg.sigma_min_pos
    sigma_rot = cfg.sigma_start_rot if use_start else cfg.sigma_min_rot
    per_knot = cfg.num_frames * NUM_LINKS * POS_DIM
    block = np.concatenate(
        [np.full(per_knot, sigma_pos), np.full(per_knot, sigma_rot)]
    )
    return np.tile(block, cfg.num_knots).astype(np.float64)


def make_sigma_min_vector(cfg: CEMConfig) -> np.ndarray:
    return make_sigma_vector(cfg, use_start=False)


def cem_sample(
    mean: np.ndarray,
    cov: np.ndarray,
    cfg: CEMConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample candidates in the same deterministic order as the wrist CEM."""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov, dtype=np.float64).reshape(-1)
    if mean.size != cfg.nu or cov.size != cfg.nu:
        raise ValueError("mean/cov dimension does not match CEM configuration")
    num_explore = min(cfg.num_explore, cfg.num_samples)
    num_main = cfg.num_samples - num_explore
    parts: list[np.ndarray] = []
    if num_main:
        parts.append(mean + rng.standard_normal((num_main, cfg.nu)) * cov)
    if num_explore:
        parts.append(
            mean
            + rng.standard_normal((num_explore, cfg.nu)) * make_sigma_vector(cfg)
        )
    samples = np.concatenate(parts, axis=0)[: cfg.num_samples]
    if cfg.include_mean:
        samples = samples.copy()
        samples[0] = mean
    return samples


def candidate_feasible(flat: np.ndarray, cfg: CEMConfig) -> np.ndarray:
    values = np.asarray(flat, dtype=np.float64)
    _, tangent = unpack_knots(values, cfg.num_knots, cfg.num_frames)
    if values.ndim == 1:
        return np.asarray(
            np.all(np.isfinite(values))
            and np.all(np.abs(tangent) <= cfg.local_tangent_limit)
        )
    tangent_axes = tuple(range(1, tangent.ndim))
    return np.all(np.isfinite(values), axis=-1) & np.all(
        np.abs(tangent) <= cfg.local_tangent_limit, axis=tangent_axes
    )


def cem_update_mean_cov(
    samples: np.ndarray,
    costs: np.ndarray,
    cfg: CEMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    valid = np.isfinite(costs) & np.all(np.isfinite(samples), axis=-1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return samples[0].copy(), make_sigma_vector(cfg)
    elite_indices = valid_indices[
        np.argsort(costs[valid_indices])[: min(cfg.num_elites, valid_indices.size)]
    ]
    elites = samples[elite_indices]
    mean = np.mean(elites, axis=0)
    cov = np.maximum(np.std(elites, axis=0, ddof=0), make_sigma_min_vector(cfg))
    return mean, cov


def adjusted_reference_from_knots(
    knot_flat: np.ndarray,
    reference_pos: np.ndarray,
    reference_quat: np.ndarray,
    knot_starts: Sequence[int] = KNOT_STARTS,
    num_frames: int = KNOT_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct all three global target trajectories from CEM knots."""
    flat = np.asarray(knot_flat, dtype=np.float64)
    squeeze = flat.ndim == 1
    if squeeze:
        flat = flat[None]
    ref_pos = np.asarray(reference_pos, dtype=np.float64)
    ref_quat = _normalize_quat(np.asarray(reference_quat, dtype=np.float64))
    if ref_pos.ndim != 3 or ref_pos.shape[1:] != (NUM_LINKS, POS_DIM):
        raise ValueError("reference_pos must have shape (L, 3, 3)")
    if ref_quat.shape != (ref_pos.shape[0], NUM_LINKS, 4):
        raise ValueError("reference_quat must have shape (L, 3, 4)")
    knot_pos, knot_tangent = unpack_knots(flat, len(knot_starts), num_frames)
    length = max(knot_starts) + num_frames
    if ref_pos.shape[0] < length:
        raise ValueError("reference motion is shorter than the knot coverage")
    frame_indices = np.asarray(knot_starts)[:, None] + np.arange(num_frames)[None]
    ref_knot_quat = ref_quat[frame_indices]
    knot_quat = quat_from_local_tangent(ref_knot_quat[None], knot_tangent)
    out_pos = np.empty((flat.shape[0], length, NUM_LINKS, POS_DIM), dtype=np.float64)
    out_quat = np.empty((flat.shape[0], length, NUM_LINKS, 4), dtype=np.float64)
    for knot_index, start in enumerate(knot_starts):
        out_pos[:, start : start + num_frames] = knot_pos[:, knot_index]
        out_quat[:, start : start + num_frames] = knot_quat[:, knot_index]
    for knot_index in range(len(knot_starts) - 1):
        left = knot_starts[knot_index] + num_frames - 1
        right = knot_starts[knot_index + 1]
        gap = np.arange(left, right + 1)
        alpha = (gap - left) / float(right - left)
        out_pos[:, gap] = (
            (1.0 - alpha)[None, :, None, None] * out_pos[:, left : left + 1]
            + alpha[None, :, None, None] * out_pos[:, right : right + 1]
        )
        out_quat[:, gap] = quat_slerp_shortest(
            out_quat[:, left, None],
            out_quat[:, right, None],
            alpha[None, :, None, None],
        )
    return (out_pos[0], out_quat[0]) if squeeze else (out_pos, out_quat)


def build_knot_reference_from_motion(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    t: int,
    link_indices: Sequence[int],
    knot_starts: Sequence[int] = KNOT_STARTS,
    num_frames: int = KNOT_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the zero-tangent 432D reference for the fixed three-link order."""
    indices = np.asarray(link_indices, dtype=np.int64)
    if indices.shape != (NUM_LINKS,):
        raise ValueError(f"link_indices must have shape ({NUM_LINKS},)")
    pos_w = np.asarray(body_pos_w, dtype=np.float64)
    quat_w = np.asarray(body_quat_w, dtype=np.float64)
    length = max(knot_starts) + num_frames
    frame_indices = np.clip(int(t) + np.arange(length), 0, pos_w.shape[0] - 1)
    ref_pos = pos_w[frame_indices][:, indices]
    ref_quat = _normalize_quat(quat_w[frame_indices][:, indices])
    knot_indices = np.asarray(knot_starts)[:, None] + np.arange(num_frames)[None]
    knot_pos = ref_pos[knot_indices]
    knot_quat = ref_quat[knot_indices]
    return knot_from_global_quat(knot_pos, knot_quat, knot_quat), ref_pos, ref_quat


def weighted_tracking_cost(
    robot_pos: np.ndarray,
    robot_quat: np.ndarray,
    ref_pos: np.ndarray,
    ref_quat: np.ndarray,
    w_rot: float = 0.1,
    link_weights: np.ndarray = LINK_COST_WEIGHTS,
) -> np.ndarray:
    """Apply unchanged wrist costs plus a 0.25-weight torso cost."""
    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    robot_quat = np.asarray(robot_quat, dtype=np.float64)
    ref_pos = np.asarray(ref_pos, dtype=np.float64)
    ref_quat = np.asarray(ref_quat, dtype=np.float64)
    weights_link = np.asarray(link_weights, dtype=np.float64)
    if weights_link.shape != (NUM_LINKS,):
        raise ValueError(f"link_weights must have shape ({NUM_LINKS},)")
    horizon = robot_pos.shape[1]
    weights_time = horizon - np.arange(horizon, dtype=np.float64)
    weights_time /= np.sum(weights_time)
    pos_sq = np.sum((robot_pos - ref_pos[None]) ** 2, axis=-1)
    rot_sq = quat_error_magnitude(ref_quat[None], robot_quat) ** 2
    per_step = np.sum((pos_sq + float(w_rot) * rot_sq) * weights_link[None, None], axis=-1)
    return np.sum(per_step * weights_time[None], axis=1)


__all__ = [
    "CEMConfig",
    "DEFAULT_FUTURE_IDX",
    "EXECUTE_STEPS",
    "HORIZON_STEPS",
    "KNOT_FRAMES",
    "KNOT_STARTS",
    "LINK_COST_WEIGHTS",
    "NUM_KNOTS",
    "NUM_LINKS",
    "TRACKING_LINK_NAMES",
    "adjusted_reference_from_knots",
    "build_knot_reference_from_motion",
    "candidate_feasible",
    "cem_sample",
    "cem_update_mean_cov",
    "choose_candidate",
    "knot_from_global_quat",
    "knot_nu",
    "make_sigma_min_vector",
    "make_sigma_vector",
    "pack_knots",
    "unpack_knots",
    "weighted_tracking_cost",
    "window_nu",
]
