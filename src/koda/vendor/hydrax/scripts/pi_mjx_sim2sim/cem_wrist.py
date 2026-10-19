"""Temporal CEM utilities for global dual-wrist tracking in MJX.

The CEM decision variable is four complete 72D wrist windows.  Position is
absolute world position.  Rotation is a local tangent around the original
world quaternion at each knot frame; reconstructed windows always contain
normalized global quaternions before they reach the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DEFAULT_FUTURE_IDX: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
NUM_WRISTS = 2
POS_DIM = 3
ROT_DIM = 3
KNOT_FRAMES = len(DEFAULT_FUTURE_IDX)
NUM_KNOTS = 4
KNOT_STARTS: tuple[int, ...] = (0, 12, 24, 36)
HORIZON_STEPS = 36
EXECUTE_STEPS = 3


def window_nu(num_frames: int = KNOT_FRAMES) -> int:
    return int(num_frames) * NUM_WRISTS * (POS_DIM + ROT_DIM)


def knot_nu(
    num_knots: int = NUM_KNOTS,
    num_frames: int = KNOT_FRAMES,
) -> int:
    return int(num_knots) * window_nu(num_frames)


@dataclass
class CEMConfig:
    num_samples: int = 1024
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

    def __post_init__(self) -> None:
        if len(self.knot_starts) != self.num_knots:
            raise ValueError("knot_starts must contain one start per knot")
        if tuple(self.knot_starts) != tuple(sorted(self.knot_starts)):
            raise ValueError("knot_starts must be sorted")
        if self.num_elites < 1 or self.num_elites > self.num_samples:
            raise ValueError("num_elites must be in [1, num_samples]")
        if self.horizon_steps <= 0 or self.execute_steps <= 0:
            raise ValueError("horizon_steps and execute_steps must be positive")

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


def _normalize_quat(quat_wxyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)):
        return np.full_like(q, np.nan)
    return q / np.maximum(norm, eps)


def _quat_conj(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(np.asarray(q1, dtype=np.float64), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(q2, dtype=np.float64), -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def axis_angle_from_quat(quat_wxyz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Quaternion (..., 4) wxyz to shortest-arc rotation vector."""
    q = _normalize_quat(quat_wxyz)
    flip = q[..., 0] < 0.0
    q = np.where(flip[..., None], -q, q)
    xyz = q[..., 1:]
    w = np.clip(q[..., 0], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = xyz / np.maximum(sin_half[..., None], eps)
    return np.where(sin_half[..., None] > eps, axis * angle[..., None], np.zeros_like(xyz))


def quat_from_rotvec(rotvec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rotation vector (..., 3) to normalized quaternion (..., 4) wxyz."""
    v = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1)
    half = 0.5 * angle
    scale = np.divide(np.sin(half), np.maximum(angle, eps))
    out = np.concatenate([np.cos(half)[..., None], v * scale[..., None]], axis=-1)
    identity = np.zeros_like(out)
    identity[..., 0] = 1.0
    return np.where(angle[..., None] > eps, out, identity)


def local_tangent_from_quat(
    reference_quat: np.ndarray,
    global_quat: np.ndarray,
) -> np.ndarray:
    """Return body/local tangent delta with global_q = ref_q * exp(delta)."""
    delta_q = _quat_mul(_quat_conj(_normalize_quat(reference_quat)), _normalize_quat(global_quat))
    return axis_angle_from_quat(delta_q)


def quat_from_local_tangent(
    reference_quat: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Reconstruct normalized global quaternion from a local tangent delta."""
    return _normalize_quat(_quat_mul(_normalize_quat(reference_quat), quat_from_rotvec(tangent)))


def quat_slerp_shortest(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    """Shortest-arc interpolation expressed in q0's local tangent."""
    q0n = _normalize_quat(q0)
    q1n = _normalize_quat(q1)
    delta = _quat_mul(_quat_conj(q0n), q1n)
    tangent = axis_angle_from_quat(delta)
    a = np.asarray(alpha, dtype=np.float64)
    while a.ndim < tangent.ndim:
        a = a[..., None]
    return quat_from_local_tangent(q0n, tangent * a)


def pack_window(u_pos: np.ndarray, u_rotvec: np.ndarray) -> np.ndarray:
    """Pack (..., F, 2, 3) position and rotation-vector arrays."""
    pos = np.asarray(u_pos, dtype=np.float64)
    rot = np.asarray(u_rotvec, dtype=np.float64)
    if pos.shape[:-3] != rot.shape[:-3] or pos.shape[-3:] != rot.shape[-3:]:
        raise ValueError("position and rotation shapes must match")
    return np.concatenate([pos.reshape(*pos.shape[:-3], -1), rot.reshape(*rot.shape[:-3], -1)], axis=-1)


def unpack_window(flat: np.ndarray, num_frames: int = KNOT_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Unpack (..., F*36) into (..., F, 2, 3) position and rotvec."""
    flat = np.asarray(flat, dtype=np.float64)
    n_pos = int(num_frames) * NUM_WRISTS * POS_DIM
    if flat.shape[-1] != 2 * n_pos:
        raise ValueError(f"expected last dimension {2 * n_pos}, got {flat.shape[-1]}")
    lead = flat.shape[:-1]
    pos = flat[..., :n_pos].reshape(*lead, num_frames, NUM_WRISTS, POS_DIM)
    rot = flat[..., n_pos:].reshape(*lead, num_frames, NUM_WRISTS, ROT_DIM)
    return pos, rot


def pack_knots(knot_pos: np.ndarray, knot_tangent: np.ndarray) -> np.ndarray:
    """Pack (..., K, F, 2, 3) positions and local tangents."""
    pos = np.asarray(knot_pos, dtype=np.float64)
    tan = np.asarray(knot_tangent, dtype=np.float64)
    if pos.shape != tan.shape or pos.ndim < 4 or pos.shape[-3:] != (
        KNOT_FRAMES,
        NUM_WRISTS,
        POS_DIM,
    ):
        raise ValueError("knot arrays must have shape (..., K, F, 2, 3)")
    lead = pos.shape[:-4]
    num_knots = pos.shape[-4]
    pos_flat = pos.reshape(*lead, num_knots, -1)
    tan_flat = tan.reshape(*lead, num_knots, -1)
    return np.concatenate([pos_flat, tan_flat], axis=-1).reshape(*lead, -1)


def unpack_knots(
    flat: np.ndarray,
    num_knots: int = NUM_KNOTS,
    num_frames: int = KNOT_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack (..., K*72) into (..., K, F, 2, 3) position and tangent."""
    flat = np.asarray(flat, dtype=np.float64)
    per_knot = int(num_frames) * NUM_WRISTS * POS_DIM
    expected = int(num_knots) * 2 * per_knot
    if flat.shape[-1] != expected:
        raise ValueError(f"expected last dimension {expected}, got {flat.shape[-1]}")
    lead = flat.shape[:-1]
    blocks = flat.reshape(*lead, num_knots, 2 * per_knot)
    pos = blocks[..., :per_knot].reshape(
        *lead, num_knots, num_frames, NUM_WRISTS, POS_DIM
    )
    tan = blocks[..., per_knot:].reshape(
        *lead, num_knots, num_frames, NUM_WRISTS, ROT_DIM
    )
    return pos, tan


def window_from_pos_quat(u_pos: np.ndarray, u_quat: np.ndarray) -> np.ndarray:
    """Pack a direct global-quaternion window for compatibility helpers."""
    return pack_window(u_pos, axis_angle_from_quat(u_quat))


def pos_quat_from_flat(flat: np.ndarray, num_frames: int = KNOT_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Decode the compatibility global-rotvec window representation."""
    pos, rotvec = unpack_window(flat, num_frames)
    return pos, quat_from_rotvec(rotvec)


def knot_from_global_quat(
    knot_pos: np.ndarray,
    knot_quat: np.ndarray,
    reference_quat: np.ndarray,
) -> np.ndarray:
    """Pack knot positions and local tangent rotations relative to reference."""
    tangent = local_tangent_from_quat(reference_quat, knot_quat)
    return pack_knots(knot_pos, tangent)


def make_sigma_vector(cfg: CEMConfig, use_start: bool = True) -> np.ndarray:
    sp = cfg.sigma_start_pos if use_start else cfg.sigma_min_pos
    sr = cfg.sigma_start_rot if use_start else cfg.sigma_min_rot
    per_knot = cfg.num_frames * NUM_WRISTS * POS_DIM
    return np.tile(
        np.concatenate([np.full(per_knot, sp), np.full(per_knot, sr)]), cfg.num_knots
    ).astype(np.float64)


def make_sigma_min_vector(cfg: CEMConfig) -> np.ndarray:
    return make_sigma_vector(cfg, use_start=False)


def cem_sample(
    mean: np.ndarray,
    cov: np.ndarray,
    cfg: CEMConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample diagonal-Gaussian knot candidates with deterministic ordering."""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov, dtype=np.float64).reshape(-1)
    if mean.size != cfg.nu or cov.size != cfg.nu:
        raise ValueError("mean/cov dimension does not match CEM configuration")
    n_explore = min(cfg.num_explore, cfg.num_samples)
    n_main = cfg.num_samples - n_explore
    parts = []
    if n_main:
        parts.append(mean + rng.standard_normal((n_main, cfg.nu)) * cov)
    if n_explore:
        parts.append(mean + rng.standard_normal((n_explore, cfg.nu)) * make_sigma_vector(cfg))
    samples = np.concatenate(parts, axis=0)[: cfg.num_samples]
    if cfg.include_mean:
        samples = samples.copy()
        samples[0] = mean
    return samples


def candidate_feasible(
    flat: np.ndarray,
    cfg: CEMConfig,
) -> np.ndarray:
    """Check only finite values and the requested local-tangent range."""
    x = np.asarray(flat, dtype=np.float64)
    _, tangent = unpack_knots(x, cfg.num_knots, cfg.num_frames)
    if x.ndim == 1:
        return np.asarray(
            np.all(np.isfinite(x)) and np.all(np.abs(tangent) <= cfg.local_tangent_limit)
        )
    return np.all(np.isfinite(x), axis=-1) & np.all(
        np.abs(tangent) <= cfg.local_tangent_limit, axis=tuple(range(1, tangent.ndim))
    )


def cem_update_mean_cov(
    samples: np.ndarray,
    costs: np.ndarray,
    cfg: CEMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a diagonal Gaussian to finite-cost elites."""
    samples = np.asarray(samples, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    valid = np.isfinite(costs) & np.all(np.isfinite(samples), axis=-1)
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size == 0:
        return samples[0].copy(), make_sigma_vector(cfg)
    idx = idx_valid[np.argsort(costs[idx_valid])[: min(cfg.num_elites, idx_valid.size)]]
    elites = samples[idx]
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
    """Construct complete adjusted global wrist motion from four knot windows.

    The returned length is ``max(knot_starts) + num_frames``.  Knot frames are
    copied exactly.  Every gap is interpolated between the preceding knot's
    tail and the following knot's head.
    """
    flat = np.asarray(knot_flat, dtype=np.float64)
    squeeze = flat.ndim == 1
    if squeeze:
        flat = flat[None, :]
    ref_pos = np.asarray(reference_pos, dtype=np.float64)
    ref_quat = _normalize_quat(np.asarray(reference_quat, dtype=np.float64))
    if ref_pos.ndim != 3 or ref_pos.shape[1:] != (NUM_WRISTS, POS_DIM):
        raise ValueError("reference_pos must have shape (L, 2, 3)")
    if ref_quat.shape != (ref_pos.shape[0], NUM_WRISTS, 4):
        raise ValueError("reference_quat must have shape (L, 2, 4)")
    knot_pos, knot_tangent = unpack_knots(flat, len(knot_starts), num_frames)
    length = max(knot_starts) + num_frames
    if ref_pos.shape[0] < length:
        raise ValueError("reference motion is shorter than the knot coverage")
    frame_idx = np.asarray(knot_starts)[:, None] + np.arange(num_frames)[None, :]
    ref_knot_quat = ref_quat[frame_idx]
    knot_quat = quat_from_local_tangent(ref_knot_quat[None, ...], knot_tangent)
    out_pos = np.empty((flat.shape[0], length, NUM_WRISTS, POS_DIM), dtype=np.float64)
    out_quat = np.empty((flat.shape[0], length, NUM_WRISTS, 4), dtype=np.float64)
    for k, start in enumerate(knot_starts):
        out_pos[:, start : start + num_frames] = knot_pos[:, k]
        out_quat[:, start : start + num_frames] = knot_quat[:, k]
    for k in range(len(knot_starts) - 1):
        left = knot_starts[k] + num_frames - 1
        right = knot_starts[k + 1]
        gap = np.arange(left, right + 1)
        alpha = (gap - left) / float(right - left)
        out_pos[:, gap] = (
            (1.0 - alpha)[None, :, None, None] * out_pos[:, left : left + 1]
            + alpha[None, :, None, None] * out_pos[:, right : right + 1]
        )
        out_quat[:, gap] = quat_slerp_shortest(
            out_quat[:, left, None, :, :],
            out_quat[:, right, None, :, :],
            alpha[None, :, None, None],
        )
    if squeeze:
        return out_pos[0], out_quat[0]
    return out_pos, out_quat


def adjusted_policy_windows(
    adjusted_pos: np.ndarray,
    adjusted_quat: np.ndarray,
    horizon: int = HORIZON_STEPS,
    num_frames: int = KNOT_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact window consumed at each rollout step h."""
    pos = np.asarray(adjusted_pos, dtype=np.float64)
    quat = np.asarray(adjusted_quat, dtype=np.float64)
    if pos.shape[:-1] != quat.shape[:-1] or pos.shape[-1] != 3 or quat.shape[-1] != 4:
        raise ValueError("adjusted position/quaternion shapes do not match")
    if pos.ndim == 3:
        pos = pos[None, ...]
        quat = quat[None, ...]
    if pos.shape[1] < horizon + num_frames:
        raise ValueError("adjusted motion is too short for the requested horizon")
    windows_pos = np.stack(
        [pos[:, h : h + num_frames] for h in range(horizon)], axis=1
    )
    windows_quat = np.stack(
        [quat[:, h : h + num_frames] for h in range(horizon)], axis=1
    )
    return windows_pos, windows_quat


def build_knot_reference_from_motion(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    t: int,
    lw_i: int,
    rw_i: int,
    knot_starts: Sequence[int] = KNOT_STARTS,
    num_frames: int = KNOT_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build zero-tangent knot reference and its 42-frame wrist reference."""
    pos_w = np.asarray(body_pos_w, dtype=np.float64)
    quat_w = np.asarray(body_quat_w, dtype=np.float64)
    length = max(knot_starts) + num_frames
    idx = np.clip(int(t) + np.arange(length), 0, pos_w.shape[0] - 1)
    ref_pos = np.stack([pos_w[idx, lw_i], pos_w[idx, rw_i]], axis=1)
    # Normalization is elementwise, so normalizing only the 42 wrist frames
    # is exactly equivalent to normalizing the entire motion at every replan.
    ref_quat = _normalize_quat(np.stack([quat_w[idx, lw_i], quat_w[idx, rw_i]], axis=1))
    knot_idx = np.asarray(knot_starts)[:, None] + np.arange(num_frames)[None, :]
    knot_pos = ref_pos[knot_idx]
    knot_quat = ref_quat[knot_idx]
    flat = knot_from_global_quat(knot_pos, knot_quat, knot_quat)
    return flat, ref_pos, ref_quat


def weighted_wrist_tracking_cost(
    robot_pos: np.ndarray,
    robot_quat: np.ndarray,
    ref_pos: np.ndarray,
    ref_quat: np.ndarray,
    w_rot: float,
) -> np.ndarray:
    """Time-weighted squared global wrist position/orientation cost."""
    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    robot_quat = np.asarray(robot_quat, dtype=np.float64)
    ref_pos = np.asarray(ref_pos, dtype=np.float64)
    ref_quat = np.asarray(ref_quat, dtype=np.float64)
    horizon = robot_pos.shape[1]
    weights = (horizon - np.arange(horizon, dtype=np.float64))
    weights /= np.sum(weights)
    dp = robot_pos - ref_pos[None, ...]
    pos_term = np.sum(dp * dp, axis=(-1, -2))
    rot_err = quat_error_magnitude(ref_quat[None, ...], robot_quat)
    per_step = pos_term + float(w_rot) * np.sum(rot_err * rot_err, axis=-1)
    return np.sum(per_step * weights[None, :], axis=1)


def quat_error_magnitude(
    q1_wxyz: np.ndarray,
    q2_wxyz: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Geodesic angle between normalized quaternions."""
    q1 = _normalize_quat(q1_wxyz)
    q2 = _normalize_quat(q2_wxyz)
    delta = _quat_mul(_quat_conj(q1), q2)
    xyz_norm = np.linalg.norm(delta[..., 1:], axis=-1)
    return 2.0 * np.arctan2(xyz_norm, np.abs(delta[..., 0]) + eps)


def choose_candidate(
    u_ref: np.ndarray,
    best_sample: np.ndarray | None,
    elite_mean: np.ndarray | None,
    cost_ref: float,
    cost_best_sample: float | None,
    cost_elite_mean: float | None,
    feasible_best_sample: bool = True,
    feasible_elite_mean: bool = True,
) -> tuple[np.ndarray, str, dict]:
    """Choose the lowest-cost validated plan, with reference as tie-breaker."""
    candidates: list[tuple[str, np.ndarray, float]] = [("reference", np.asarray(u_ref), float(cost_ref))]
    if best_sample is not None and feasible_best_sample and np.isfinite(cost_best_sample):
        candidates.append(("best_sample", np.asarray(best_sample), float(cost_best_sample)))
    if elite_mean is not None and feasible_elite_mean and np.isfinite(cost_elite_mean):
        candidates.append(("elite_mean", np.asarray(elite_mean), float(cost_elite_mean)))
    kind, plan, cost = min(enumerate(candidates), key=lambda item: (item[1][2], item[0]))[1]
    info = {
        "selected_candidate": kind,
        "selected_cost": float(cost),
        "accepted_cem": kind != "reference",
        "cost_ref": float(cost_ref),
        "cost_best_sample": None if cost_best_sample is None else float(cost_best_sample),
        "cost_elite_mean": None if cost_elite_mean is None else float(cost_elite_mean),
    }
    return plan.copy(), kind, info


def accept_execute_u(
    samples: np.ndarray,
    costs: np.ndarray,
    u_ref: np.ndarray,
    cfg: CEMConfig,
) -> tuple[np.ndarray, bool, dict]:
    """Compatibility wrapper using validated best sample as the only alternative."""
    idx = int(np.nanargmin(np.where(np.isfinite(costs), costs, np.inf)))
    mean, cov = cem_update_mean_cov(samples, costs, cfg)
    plan, _, info = choose_candidate(
        u_ref,
        samples[idx] if np.isfinite(costs[idx]) else None,
        None,
        float(costs[0]) if cfg.include_mean else float("inf"),
        float(costs[idx]) if np.isfinite(costs[idx]) else None,
        None,
    )
    info.update({"cost_min": float(np.nanmin(costs)), "cov_mean": float(np.mean(cov))})
    return plan, bool(info["accepted_cem"]), info
