"""Device-side JAX policy/MJX rollout for torso + dual-wrist CEM."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from cem_three_point import LINK_COST_WEIGHTS
from diagnostics.jax_closed_loop_push1 import (
    _make_weights,
    _policy_forward,
    _quat_apply_inverse,
    _quat_error_magnitude,
    _quat_inverse_left,
    _to_jax_history,
)
from mjx_pi_env_three_point import MjxPiEnvThreePoint

DEFAULT_REF_PAD_FRAMES = 2048


def pad_reference_arrays(
    ref_pos: np.ndarray,
    ref_quat: np.ndarray,
    pad_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad clip references to a static leading dimension by repeating the last frame."""
    ref_pos = np.asarray(ref_pos)
    ref_quat = np.asarray(ref_quat)
    if ref_pos.shape[0] != ref_quat.shape[0]:
        raise ValueError(
            f"ref_pos/ref_quat length mismatch: {ref_pos.shape[0]} vs {ref_quat.shape[0]}"
        )
    frames = int(ref_pos.shape[0])
    pad_frames = int(pad_frames)
    if pad_frames < 1:
        raise ValueError("pad_frames must be positive")
    if frames > pad_frames:
        raise ValueError(f"clip length {frames} exceeds ref pad {pad_frames}")
    if frames == pad_frames:
        return ref_pos, ref_quat
    pad = pad_frames - frames
    return (
        np.concatenate([ref_pos, np.repeat(ref_pos[-1:], pad, axis=0)], axis=0),
        np.concatenate([ref_quat, np.repeat(ref_quat[-1:], pad, axis=0)], axis=0),
    )


def next_ref_pad_frames(needed_frames: int, current_pad: int | None = None) -> int:
    pad = int(current_pad) if current_pad is not None else DEFAULT_REF_PAD_FRAMES
    if pad < 1:
        pad = DEFAULT_REF_PAD_FRAMES
    needed = int(needed_frames)
    while pad < needed:
        pad *= 2
    return pad


def prepare_rollout_refs(
    env: MjxPiEnvThreePoint,
    pad_frames: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    ref_pos = np.asarray(env.ref_body_pos)
    ref_quat = np.asarray(env.ref_body_quat)
    if pad_frames is not None:
        ref_pos, ref_quat = pad_reference_arrays(ref_pos, ref_quat, pad_frames)
    return (
        jnp.asarray(ref_pos),
        jnp.asarray(ref_quat),
        jnp.asarray(int(env.T) - 1, dtype=jnp.int32),
    )


def rollout_arguments(
    env: MjxPiEnvThreePoint,
    state: Any,
    hist: Any,
    adjusted_pos: np.ndarray,
    adjusted_quat: np.ndarray,
    time_step: int,
    pad_frames: int | None = None,
) -> tuple[Any, ...]:
    ref_pos, ref_quat, t_max = prepare_rollout_refs(env, pad_frames)
    return (
        state,
        _to_jax_history(hist),
        jnp.asarray(hist.head, dtype=jnp.int32),
        jnp.asarray(adjusted_pos),
        jnp.asarray(adjusted_quat),
        jnp.asarray(time_step, dtype=jnp.int32),
        ref_pos,
        ref_quat,
        t_max,
    )


def make_rollout(env: MjxPiEnvThreePoint, horizon_steps: int, w_rot: float):
    """JIT a complete batched three-point policy + PD + MJX horizon.

    Reference arrays and ``t_max`` are rollout arguments so clip length is not
    baked into the compiled trace. N and H remain static via separate production-N
    and N=1 compiled functions plus a closed-over horizon.
    """
    weights_np, constants_np, mask_np = _make_weights(env)
    weights = jax.tree_util.tree_map(jnp.asarray, weights_np)
    constants = jax.tree_util.tree_map(jnp.asarray, constants_np)
    self_mask = jnp.asarray(mask_np, dtype=bool)
    parents = tuple(int(x) for x in np.asarray(constants_np["parent_indices"]).tolist())
    future_idx = jnp.asarray(env.future_idx, dtype=jnp.int32)
    sim_idx = jnp.asarray(env.sim_idx_for_policy, dtype=jnp.int32)
    policy_to_sim = jnp.asarray(env.policy_idx_for_sim, dtype=jnp.int32)
    selected_idx = jnp.asarray(env.sel_idx, dtype=jnp.int32)
    track_slots = jnp.asarray(env.track_selected_slots, dtype=jnp.int32)
    track_motion_indices = jnp.asarray(env.track_motion_indices, dtype=jnp.int32)
    track_body_ids = jnp.asarray(env.track_mjx_body_ids, dtype=jnp.int32)
    link_cost_weights = jnp.asarray(LINK_COST_WEIGHTS, dtype=jnp.float32)
    context_len = int(env.context_len)
    order_table = jnp.asarray(
        (np.arange(context_len)[None] + np.arange(context_len)[:, None] + 1)
        % context_len,
        dtype=jnp.int32,
    )
    mode_indices = jnp.full((1024,), env.mode_index_val, dtype=jnp.int32)
    time_offsets = jnp.asarray(env.future_idx, dtype=jnp.float32)[None, :, None]

    def policy_fn(hist, target_pos, target_rot, modes, offsets):
        return _policy_forward(
            hist,
            target_pos,
            target_rot,
            modes,
            offsets,
            weights,
            constants,
            self_mask,
            4,
            parents,
        )

    def rollout(
        state: mjx.Data,
        hist: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        head: jax.Array,
        adjusted_pos: jax.Array,
        adjusted_quat: jax.Array,
        time_step: jax.Array,
        ref_pos_all: jax.Array,
        ref_quat_all: jax.Array,
        t_max: jax.Array,
    ):
        num_candidates = adjusted_pos.shape[0]
        modes = jnp.broadcast_to(mode_indices[:num_candidates], (num_candidates,))

        def body(carry: tuple[Any, ...], horizon_index: jax.Array):
            state0, hist0, head0 = carry
            qpos = state0.qpos
            qvel = state0.qvel
            root_pos = qpos[:, :3]
            root_quat = qpos[:, 3:7]
            dof_sim = qpos[:, 7 : 7 + 29]
            vel_sim = qvel[:, 6 : 6 + 29]
            dof_pol = dof_sim[:, sim_idx].astype(jnp.float32)
            vel_pol = vel_sim[:, sim_idx].astype(jnp.float32)
            base_ang = qvel[:, 3:6].astype(jnp.float32)

            head1 = (head0 + 1) % context_len
            root_hist, ang_hist, pos_hist, vel_hist, action_hist = hist0
            root_hist = root_hist.at[:, head1].set(root_quat.astype(jnp.float32))
            ang_hist = ang_hist.at[:, head1].set(base_ang)
            pos_hist = pos_hist.at[:, head1].set(dof_pol)
            vel_hist = vel_hist.at[:, head1].set(vel_pol)
            ordered = order_table[head1]
            policy_hist = (
                root_hist[:, ordered],
                ang_hist[:, ordered],
                pos_hist[:, ordered],
                vel_hist[:, ordered],
                action_hist[:, ordered],
            )

            indices = jnp.clip(time_step + horizon_index + future_idx, 0, t_max)
            base_pos = ref_pos_all[indices][:, selected_idx]
            base_quat = ref_quat_all[indices][:, selected_idx]
            target_pos = _quat_apply_inverse(
                root_quat[:, None, None],
                base_pos[None] - root_pos[:, None, None],
            ).astype(jnp.float32)
            target_rot = _quat_inverse_left(
                root_quat[:, None, None], base_quat[None]
            ).astype(jnp.float32)
            override_pos = jax.lax.dynamic_slice_in_dim(
                adjusted_pos, horizon_index, len(env.future_idx), axis=1
            )
            override_quat = jax.lax.dynamic_slice_in_dim(
                adjusted_quat, horizon_index, len(env.future_idx), axis=1
            )
            link_pos = _quat_apply_inverse(
                root_quat[:, None, None],
                override_pos - root_pos[:, None, None],
            ).astype(jnp.float32)
            link_quat = _quat_inverse_left(
                root_quat[:, None, None], override_quat
            ).astype(jnp.float32)
            target_pos = target_pos.at[:, :, track_slots].set(link_pos)
            target_rot = target_rot.at[:, :, track_slots].set(link_quat)

            target_dof, raw_action = policy_fn(
                policy_hist,
                target_pos,
                target_rot,
                modes,
                jnp.broadcast_to(
                    time_offsets, (num_candidates, len(env.future_idx), 1)
                ),
            )
            action_slot = (head1 + 1) % context_len
            action_hist = action_hist.at[:, action_slot].set(raw_action.astype(jnp.float32))
            target_sim = target_dof[:, policy_to_sim].astype(jnp.float32)
            state1 = env._pd_n_sub_batch(state0, target_sim)
            robot_pos = state1.xpos[:, track_body_ids]
            robot_quat = state1.xquat[:, track_body_ids]
            next_hist = (root_hist, ang_hist, pos_hist, vel_hist, action_hist)
            trace = (robot_pos, robot_quat, target_dof, raw_action)
            return (state1, next_hist, head1), trace

        (state_final, hist_final, head_final), trace = jax.lax.scan(
            body,
            (state, hist, head),
            jnp.arange(horizon_steps, dtype=jnp.int32),
        )
        robot_pos, robot_quat, target_seq, raw_seq = trace
        ref_indices = jnp.clip(
            time_step + jnp.arange(horizon_steps, dtype=jnp.int32) + 1, 0, t_max
        )
        ref_pos = ref_pos_all[ref_indices][:, track_motion_indices]
        ref_quat = ref_quat_all[ref_indices][:, track_motion_indices]
        robot_pos_n = robot_pos.transpose(1, 0, 2, 3)
        robot_quat_n = robot_quat.transpose(1, 0, 2, 3)
        position_sq = jnp.sum((robot_pos_n - ref_pos[None]) ** 2, axis=-1)
        rotation_sq = _quat_error_magnitude(ref_quat[None], robot_quat_n) ** 2
        per_step = jnp.sum(
            (position_sq + float(w_rot) * rotation_sq)
            * link_cost_weights[None, None],
            axis=-1,
        )
        time_weights = horizon_steps - jnp.arange(horizon_steps, dtype=jnp.float32)
        time_weights /= jnp.sum(time_weights)
        costs = jnp.sum(per_step * time_weights[None], axis=1)
        return (state_final, hist_final, head_final), (
            costs,
            robot_pos,
            robot_quat,
            target_seq,
            raw_seq,
        )

    return jax.jit(rollout)


__all__ = [
    "DEFAULT_REF_PAD_FRAMES",
    "_to_jax_history",
    "make_rollout",
    "next_ref_pad_frames",
    "pad_reference_arrays",
    "prepare_rollout_refs",
    "rollout_arguments",
]
