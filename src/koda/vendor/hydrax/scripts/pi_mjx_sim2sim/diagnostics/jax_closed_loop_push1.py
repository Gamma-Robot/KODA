#!/usr/bin/env python3
"""Isolated JAX closed-loop rollout diagnostic for one ScaleTrack setting.

This file is deliberately outside the production MPC path.  It reconstructs
the exported ScaleTrack policy with JAX primitives, keeps policy/history/target
data on the JAX device, and wraps the horizon in one ``jit``/``lax.scan``.
The script compares that path with the existing Torch/NumPy/MJX path on the
same push1 snapshot and candidate batch.

The implementation is a diagnostic, not a replacement for ``mjx_pi_env``.
Numerical parity is checked before any timing result is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax

# Deliberately use the same high-precision dot path for the policy and the MJX
# graph.  Enabling x64 here would also change MJX contact-index dtypes before
# the environment is constructed, so this isolated diagnostic must not do that.
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mujoco import mjx  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from cem_wrist import (  # noqa: E402
    CEMConfig,
    adjusted_reference_from_knots,
    build_knot_reference_from_motion,
    candidate_feasible,
    cem_sample,
    make_sigma_vector,
    weighted_wrist_tracking_cost,
)
from mjx_pi_env import MjxPiEnv  # noqa: E402


def _np_tensor(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu").numpy().copy()


def _quat_apply(q: jax.Array, v: jax.Array) -> jax.Array:
    xyz = q[..., 1:]
    t = 2.0 * jnp.cross(xyz, v)
    return v + q[..., :1] * t + jnp.cross(xyz, t)


def _quat_apply_inverse(q: jax.Array, v: jax.Array) -> jax.Array:
    return _quat_apply(jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1), v)


def _quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = jnp.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(q2, -1, 0)
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_inverse_left(q1: jax.Array, q2: jax.Array) -> jax.Array:
    return _quat_mul(jnp.concatenate([q1[..., :1], -q1[..., 1:]], axis=-1), q2)


def _quat_inverse_right(q1: jax.Array, q2: jax.Array) -> jax.Array:
    return _quat_mul(q1, jnp.concatenate([q2[..., :1], -q2[..., 1:]], axis=-1))


# These variants preserve the arithmetic ordering used by the TorchScript
# quaternion helpers.  The standard variants above remain for target building,
# which follows the NumPy reference implementation in policy_inputs.py.
def _torch_quat_apply(q: jax.Array, v: jax.Array) -> jax.Array:
    xyz = q[..., 1:]
    t = 2.0 * jnp.cross(xyz, v)
    return v + q[..., :1] * t + jnp.cross(xyz, t)


def _torch_quat_apply_inverse(q: jax.Array, v: jax.Array) -> jax.Array:
    xyz = q[..., 1:]
    t = 2.0 * jnp.cross(xyz, v)
    return v - q[..., :1] * t + jnp.cross(xyz, t)


def _torch_quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = jnp.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(q2, -1, 0)
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    return jnp.stack(
        [
            qq - ww + (z1 - y1) * (y2 - z2),
            qq - xx + (x1 + w1) * (x2 + w2),
            qq - yy + (w1 - x1) * (y2 + z2),
            qq - zz + (z1 + y1) * (w2 - x2),
        ],
        axis=-1,
    )


def _torch_quat_inverse_left(q1: jax.Array, q2: jax.Array) -> jax.Array:
    q1_inv = q1 * jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=q1.dtype)
    return _torch_quat_mul(q1_inv, q2)


def _torch_quat_inverse_right(q1: jax.Array, q2: jax.Array) -> jax.Array:
    q2_inv = q2 * jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=q2.dtype)
    return _torch_quat_mul(q1, q2_inv)


def _quat_error_magnitude(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """JAX counterpart of cem_wrist.quat_error_magnitude."""
    n1 = jnp.linalg.norm(q1, axis=-1, keepdims=True)
    n2 = jnp.linalg.norm(q2, axis=-1, keepdims=True)
    q1 = q1 / jnp.maximum(n1, 1.0e-12)
    q2 = q2 / jnp.maximum(n2, 1.0e-12)
    delta = _quat_mul(jnp.concatenate([q1[..., :1], -q1[..., 1:]], axis=-1), q2)
    xyz_norm = jnp.linalg.norm(delta[..., 1:], axis=-1)
    return 2.0 * jnp.arctan2(xyz_norm, jnp.abs(delta[..., 0]) + 1.0e-8)


def _rms_norm(x: jax.Array, scale: jax.Array) -> jax.Array:
    dim = x.shape[-1]
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True) / jnp.sqrt(float(dim))
    return scale * x / (norm + 1.0e-8)


def _linear(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    # ``matmul`` maps to the same batched GEMM pattern as torch.nn.Linear;
    # using highest dot precision avoids implicit TF32 rounding on NVIDIA.
    return jnp.matmul(x, jnp.swapaxes(weight, -1, -2)) + bias


def _rope(x: jax.Array, inv_freq: jax.Array) -> jax.Array:
    positions = jnp.arange(x.shape[1], dtype=jnp.float32)
    freq = positions[:, None] * inv_freq[None, :]
    c = jnp.cos(freq)[None, ...]
    s = jnp.sin(freq)[None, ...]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    out = jnp.zeros_like(x)
    out = out.at[..., 0::2].set(even * c - odd * s)
    out = out.at[..., 1::2].set(even * s + odd * c)
    return out


def _mha(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    prefix: str,
    weights: dict[str, jax.Array],
    mask: jax.Array | None,
    num_heads: int = 4,
    same_kv: bool = False,
) -> jax.Array:
    iw = weights[f"{prefix}.in_proj_weight"]
    ib = weights[f"{prefix}.in_proj_bias"]
    qw, kw, vw = jnp.split(iw, 3, axis=0)
    qb, kb, vb = jnp.split(ib, 3, axis=0)
    q = _linear(query, qw, qb)
    if same_kv:
        # torch.nn.MultiheadAttention uses one packed K/V projection when the
        # key and value tensors are the same object.  Keeping that combined
        # GEMM avoids an otherwise small but repeatable rounding difference.
        embed_dim = iw.shape[1]
        kv = _linear(key, iw[embed_dim:], ib[embed_dim:])
        k, v = jnp.split(kv, 2, axis=-1)
    else:
        k = _linear(key, kw, kb)
        v = _linear(value, vw, vb)
    batch, qlen, dim = q.shape
    klen = k.shape[1]
    head_dim = dim // num_heads
    q = q.reshape(batch, qlen, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, klen, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, klen, num_heads, head_dim).transpose(0, 2, 1, 3)
    q = q * (1.0 / jnp.sqrt(float(head_dim)))
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    if mask is not None:
        scores = jnp.where(mask[None, None, :, :], -jnp.inf, scores)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(probs, v)
    out = out.transpose(0, 2, 1, 3).reshape(batch, qlen, dim)
    return _linear(
        out,
        weights[f"{prefix}.out_proj.weight"],
        weights[f"{prefix}.out_proj.bias"],
    )


def _policy_forward(
    hist: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    target_pos: jax.Array,
    target_rot: jax.Array,
    mode_index: jax.Array,
    time_offsets: jax.Array,
    weights: dict[str, jax.Array],
    constants: dict[str, jax.Array],
    self_mask: jax.Array,
    num_heads: int,
    parents: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """JAX equivalent of HumanoidTransformerPolicyWrapperWithMode.forward."""
    root_hist, base_hist, dof_hist, vel_hist, action_hist = hist
    root_quat = root_hist[:, -1]
    dof_pos = dof_hist[:, -1]

    gravity = jnp.broadcast_to(constants["gravity"], root_hist.shape[:-1] + (3,))
    projected_gravity = _torch_quat_apply_inverse(root_hist, gravity)
    prop_obs = jnp.concatenate(
        [
            projected_gravity,
            base_hist,
            dof_hist - constants["default_dof_pos"],
            vel_hist * 0.05,
        ],
        axis=-1,
    )
    prop = _linear(
        prop_obs,
        weights["prop_embedder.weight"],
        weights["prop_embedder.bias"],
    )
    action = _linear(
        action_hist,
        weights["action_embedder.weight"],
        weights["action_embedder.bias"],
    )
    batch = prop.shape[0]
    x = jnp.zeros((batch, 2 * prop.shape[1], prop.shape[2]), dtype=prop.dtype)
    x = x.at[:, 0::2].set(prop)
    x = x.at[:, 1:-1:2].set(action[:, 1:])
    x = x.at[:, -1].set(weights["empty_embedding"][0, 0])

    # FK is intentionally written as a static loop.  The XML topology is fixed
    # for this diagnostic, so XLA unrolls it and keeps all body state on device.
    half_angles = dof_pos[:, constants["lab_to_xml"]] / 2.0
    joint_rot = jnp.concatenate(
        [
            jnp.cos(half_angles)[..., None],
            constants["joint_axis"][None, ...] * jnp.sin(half_angles)[..., None],
        ],
        axis=-1,
    )
    nbody = len(parents)
    body_pos = jnp.zeros((batch, nbody, 3), dtype=dof_pos.dtype)
    body_quat = jnp.zeros((batch, nbody, 4), dtype=dof_pos.dtype).at[..., 0].set(1.0)
    for j in range(1, nbody):
        parent = parents[j]
        parent_pos = body_pos[:, parent]
        parent_quat = body_quat[:, parent]
        local_trans = constants["local_translation"][j - 1]
        local_rot = constants["local_rotation"][j - 1]
        world_trans = _torch_quat_apply(parent_quat, jnp.broadcast_to(local_trans, (batch, 3)))
        curr_pos = parent_pos + world_trans
        curr_rot = _torch_quat_mul(
            jnp.broadcast_to(local_rot, (batch, 4)), joint_rot[:, j - 1]
        )
        curr_rot = _torch_quat_mul(parent_quat, curr_rot)
        body_pos = body_pos.at[:, j].set(curr_pos)
        body_quat = body_quat.at[:, j].set(curr_rot)

    selected = constants["selected_link_indices"]
    body_pos_sel = body_pos[:, selected]
    body_quat_sel = body_quat[:, selected]
    target_pos_rel = target_pos - body_pos_sel[:, None]
    tan_vec = constants["tan_vec"]
    norm_vec = constants["norm_vec"]
    target_rot_tn = jnp.concatenate(
        [_torch_quat_apply(target_rot, tan_vec), _torch_quat_apply(target_rot, norm_vec)], axis=-1
    )
    body_rot_broadcast = body_quat_sel[:, None]
    target_rot_rel = _torch_quat_inverse_right(target_rot, body_rot_broadcast)
    target_rot_rel_tn = jnp.concatenate(
        [_torch_quat_apply(target_rot_rel, tan_vec), _torch_quat_apply(target_rot_rel, norm_vec)],
        axis=-1,
    )
    task_obs = jnp.concatenate(
        [
            target_pos.reshape(batch, target_pos.shape[1], -1),
            target_pos_rel.reshape(batch, target_pos.shape[1], -1),
            target_rot_tn.reshape(batch, target_pos.shape[1], -1),
            target_rot_rel_tn.reshape(batch, target_pos.shape[1], -1),
            jnp.broadcast_to(time_offsets, (batch, time_offsets.shape[1], 1)),
        ],
        axis=-1,
    )
    mapping = constants["mode_mappings"][mode_index]
    masked = task_obs * mapping[:, None, :]
    mode_vec = constants["mode_vectors"][mode_index]
    task_input = jnp.concatenate(
        [masked, jnp.broadcast_to(mode_vec[:, None, :], (batch, masked.shape[1], mode_vec.shape[-1]))],
        axis=-1,
    )
    task_tokens = _linear(
        task_input,
        weights["task_embedder.task_projection.weight"],
        weights["task_embedder.task_projection.bias"],
    )

    for layer in range(4):
        prefix = f"transformer_blocks.{layer}"
        x_norm = _rms_norm(x, weights[f"{prefix}.rmsnorm1.scale"])
        x_rope = _rope(x_norm, weights[f"{prefix}.rope.inv_freq"])
        x = x + _mha(
            x_rope,
            x_rope,
            x_norm,
            f"{prefix}.self_attention",
            weights,
            self_mask,
            num_heads,
        )
        x_norm2 = _rms_norm(x, weights[f"{prefix}.rmsnorm2.scale"])
        c_norm = _rms_norm(task_tokens, weights[f"{prefix}.cond_norm.scale"])
        x = x + _mha(
            x_norm2,
            c_norm,
            c_norm,
            f"{prefix}.cross_attention",
            weights,
            None,
            num_heads,
            same_kv=True,
        )
        x_norm3 = _rms_norm(x, weights[f"{prefix}.rmsnorm3.scale"])
        ff_w = _linear(
            x_norm3,
            weights[f"{prefix}.feed_forward.w.weight"],
            jnp.zeros((x_norm3.shape[-1],), dtype=x_norm3.dtype),
        )
        ff_v = _linear(
            x_norm3,
            weights[f"{prefix}.feed_forward.v.weight"],
            jnp.zeros((x_norm3.shape[-1],), dtype=x_norm3.dtype),
        )
        gated = jax.nn.silu(ff_w) * ff_v
        x = x + _linear(
            gated,
            weights[f"{prefix}.feed_forward.output.weight"],
            jnp.zeros((x_norm3.shape[-1],), dtype=x_norm3.dtype),
        )
    x = _rms_norm(x, weights["final_norm.scale"])
    raw_action = _linear(x[:, -1], weights["projection_head.weight"], weights["projection_head.bias"])
    return raw_action * constants["action_scale"] + constants["default_dof_pos"], raw_action


def _make_weights(env: MjxPiEnv) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    sd = {k: _np_tensor(v) for k, v in env.policy.state_dict().items()}
    # Only the wrapper's buffers and parameters needed by forward are retained.
    names = (
        "empty_embedding",
        "prop_embedder.weight",
        "prop_embedder.bias",
        "action_embedder.weight",
        "action_embedder.bias",
        "task_embedder.task_projection.weight",
        "task_embedder.task_projection.bias",
        "final_norm.scale",
        "projection_head.weight",
        "projection_head.bias",
    )
    weights = {k: sd[k].astype(np.float32) for k in names}
    for layer in range(4):
        prefix = f"transformer_blocks.{layer}"
        for suffix in (
            "self_attention.in_proj_weight",
            "self_attention.in_proj_bias",
            "self_attention.out_proj.weight",
            "self_attention.out_proj.bias",
            "cross_attention.in_proj_weight",
            "cross_attention.in_proj_bias",
            "cross_attention.out_proj.weight",
            "cross_attention.out_proj.bias",
            "feed_forward.w.weight",
            "feed_forward.v.weight",
            "feed_forward.output.weight",
            "rmsnorm1.scale",
            "rmsnorm2.scale",
            "rmsnorm3.scale",
            "cond_norm.scale",
            "rope.inv_freq",
        ):
            key = f"{prefix}.{suffix}"
            weights[key] = sd[key].astype(np.float32)
    constants = {
        "gravity": sd["gravity_vec"][:, 0].astype(np.float32),
        "default_dof_pos": sd["default_dof_pos"][0].astype(np.float32),
        "action_scale": sd["action_scale"][0].astype(np.float32),
        "mode_mappings": sd["mode_mappings"].astype(np.float32),
        "mode_vectors": sd["mode_vectors"].astype(np.float32),
        "selected_link_indices": sd["selected_link_indices"].astype(np.int32),
        "tan_vec": sd["tan_vec"][0].astype(np.float32),
        "norm_vec": sd["norm_vec"][0].astype(np.float32),
        "local_translation": sd["local_translation"].astype(np.float32),
        "local_rotation": sd["local_rotation"].astype(np.float32),
        "parent_indices": sd["parent_indices"].astype(np.int32),
        "joint_axis": sd["joint_axis"].astype(np.float32),
        "lab_to_xml": sd["lab_to_xml_joint_indices"].astype(np.int32),
    }
    return weights, constants, sd["self_attn_mask"]


def _make_rollout(env: MjxPiEnv, h_steps: int, w_rot: float):
    weights_np, constants_np, mask_np = _make_weights(env)
    weights = jax.tree_util.tree_map(jnp.asarray, weights_np)
    constants = jax.tree_util.tree_map(jnp.asarray, constants_np)
    self_mask = jnp.asarray(mask_np, dtype=bool)
    parents = tuple(int(x) for x in np.asarray(constants_np["parent_indices"]).tolist())
    future_idx = jnp.asarray(env.future_idx, dtype=jnp.int32)
    sim_idx = jnp.asarray(env.sim_idx_for_policy, dtype=jnp.int32)
    policy_to_sim = jnp.asarray(env.policy_idx_for_sim, dtype=jnp.int32)
    selected_idx = jnp.asarray(env.sel_idx, dtype=jnp.int32)
    ref_pos_all = jnp.asarray(env.ref_body_pos)
    ref_quat_all = jnp.asarray(env.ref_body_quat)
    t_max = int(env.T - 1)
    context_len = int(env.context_len)
    order_table = jnp.asarray(
        (np.arange(context_len)[None, :] + np.arange(context_len)[:, None] + 1) % context_len,
        dtype=jnp.int32,
    )
    # The production wrapper's mode is fixed for this diagnostic.
    mode_index = jnp.asarray([env.mode_index_val] * 1024, dtype=jnp.int32)
    time_offsets = jnp.asarray(env.future_idx, dtype=jnp.float32)[None, :, None]

    # Keep XML topology as Python constants.  This lets the static FK loop be
    # unrolled by XLA instead of attempting a host conversion of a tracer.
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
    ) -> tuple[jax.Array, mjx.Data, tuple[jax.Array, ...]]:
        n = adjusted_pos.shape[0]
        modes = jnp.broadcast_to(mode_index[:n], (n,))

        def body(carry: tuple[Any, ...], h: jax.Array):
            st, hist0, head0 = carry
            qpos = st.qpos
            qvel = st.qvel
            root_pos = qpos[:, :3]
            root_quat = qpos[:, 3:7]
            dof_sim = qpos[:, 7 : 7 + 29]
            vel_sim = qvel[:, 6 : 6 + 29]
            dof_pol = dof_sim[:, sim_idx].astype(jnp.float32)
            vel_pol = vel_sim[:, sim_idx].astype(jnp.float32)
            base_ang = qvel[:, 3:6].astype(jnp.float32)
            head1 = (head0 + 1) % context_len
            rq, ba, dp, dv, ah = hist0
            rq = rq.at[:, head1].set(root_quat.astype(jnp.float32))
            ba = ba.at[:, head1].set(base_ang)
            dp = dp.at[:, head1].set(dof_pol)
            dv = dv.at[:, head1].set(vel_pol)
            ordered = order_table[head1]
            hist1 = (rq[:, ordered], ba[:, ordered], dp[:, ordered], dv[:, ordered], ah[:, ordered])

            idxs = jnp.clip(time_step + h + future_idx, 0, t_max)
            base_pos = ref_pos_all[idxs][:, selected_idx, :]
            base_quat = ref_quat_all[idxs][:, selected_idx, :]
            target_pos = _quat_apply_inverse(
                root_quat[:, None, None, :],
                base_pos[None, ...] - root_pos[:, None, None, :],
            ).astype(jnp.float32)
            target_rot = _quat_inverse_left(
                root_quat[:, None, None, :], base_quat[None, ...]
            ).astype(jnp.float32)
            up = jax.lax.dynamic_slice_in_dim(
                adjusted_pos, h, len(env.future_idx), axis=1
            )
            uq = jax.lax.dynamic_slice_in_dim(
                adjusted_quat, h, len(env.future_idx), axis=1
            )
            wrist_pos = _quat_apply_inverse(
                root_quat[:, None, None, :], up - root_pos[:, None, None, :]
            ).astype(jnp.float32)
            wrist_quat = _quat_inverse_left(
                root_quat[:, None, None, :], uq
            ).astype(jnp.float32)
            target_pos = target_pos.at[:, :, env.lw_sel, :].set(wrist_pos[:, :, 0])
            target_pos = target_pos.at[:, :, env.rw_sel, :].set(wrist_pos[:, :, 1])
            target_rot = target_rot.at[:, :, env.lw_sel, :].set(wrist_quat[:, :, 0])
            target_rot = target_rot.at[:, :, env.rw_sel, :].set(wrist_quat[:, :, 1])

            tgt_pol, raw_action = policy_fn(
                hist1,
                target_pos,
                target_rot,
                modes,
                jnp.broadcast_to(time_offsets, (n, len(env.future_idx), 1)),
            )
            action_slot = (head1 + 1) % context_len
            ah = ah.at[:, action_slot].set(raw_action.astype(jnp.float32))
            # The reference policy returns float32 targets; MJX will promote
            # only if its configured state dtype is wider.
            tgt_sim = tgt_pol[:, policy_to_sim].astype(jnp.float32)
            st1 = env._pd_n_sub_batch(st, tgt_sim)
            wrist_pos_out = st1.xpos[:, jnp.asarray([env.lw_bid, env.rw_bid])]
            wrist_quat_out = st1.xquat[:, jnp.asarray([env.lw_bid, env.rw_bid])]
            return (st1, (rq, ba, dp, dv, ah), head1), (wrist_pos_out, wrist_quat_out, tgt_pol, raw_action)

        (state_final, hist_final, head_final), trace = jax.lax.scan(
            body, (state, hist, head), jnp.arange(h_steps, dtype=jnp.int32)
        )
        robot_pos, robot_quat, tgt_seq, raw_seq = trace
        # Keep the reduction inside the compiled graph so the timing includes
        # the same cost computation as the reference rollout.  References are
        # indexed on device from the current replan time.
        ref_idx = jnp.clip(
            time_step + jnp.arange(h_steps, dtype=jnp.int32) + 1, 0, t_max
        )
        ref_pos = ref_pos_all[ref_idx][:, jnp.asarray([env.lw_i, env.rw_i])]
        ref_quat = ref_quat_all[ref_idx][:, jnp.asarray([env.lw_i, env.rw_i])]
        robot_pos_n = robot_pos.transpose(1, 0, 2, 3)
        robot_quat_n = robot_quat.transpose(1, 0, 2, 3)
        dp = robot_pos_n - ref_pos[None, ...]
        rot_err = _quat_error_magnitude(ref_quat[None, ...], robot_quat_n)
        per_step = jnp.sum(dp * dp, axis=(2, 3)) + float(w_rot) * jnp.sum(rot_err * rot_err, axis=2)
        weights_cost = (h_steps - jnp.arange(h_steps, dtype=jnp.float32))
        weights_cost = weights_cost / jnp.sum(weights_cost)
        costs = jnp.sum(per_step * weights_cost[None, :], axis=1)
        return (state_final, hist_final, head_final), (costs, robot_pos, robot_quat, tgt_seq, raw_seq)

    return jax.jit(rollout)


def _to_jax_history(hist: Any):
    return tuple(jnp.asarray(x) for x in (hist.root_quat, hist.base_ang, hist.dof_pos, hist.dof_vel, hist.action))


def _stats(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "samples_ms": a.tolist(),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def _sync_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = CEMConfig(num_samples=args.num_samples, num_elites=10, horizon_steps=args.horizon, execute_steps=3)
    xml = REPO_ROOT / "scalebfm/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml"
    motion = REPO_ROOT / "motion_data/processed/lafan/lafan1_push1_subject2.npz"
    art = SCRIPT_DIR / "artifacts"
    env = MjxPiEnv(str(art), str(xml), str(motion), mode_index=1, torch_device="cuda")
    env.warm_compile_batch(cfg.num_samples)
    env.warm_compile_batch(1)
    env.reset_rsi()
    snap = env.snapshot()
    u_ref, ref_pos, ref_quat = build_knot_reference_from_motion(
        env.ref_body_pos, env.ref_body_quat, 0, env.lw_i, env.rw_i,
        cfg.knot_starts, cfg.num_frames,
    )
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
    samples = cem_sample(u_ref.copy(), make_sigma_vector(cfg), cfg, rng)
    feasible = np.asarray(candidate_feasible(samples, cfg), dtype=bool)
    safe_samples = samples.copy()
    safe_samples[~feasible] = u_ref
    adjusted_pos, adjusted_quat = adjusted_reference_from_knots(
        safe_samples, ref_pos, ref_quat, cfg.knot_starts, cfg.num_frames
    )
    state_b, hist_b = env.broadcast_snapshot(snap, cfg.num_samples)
    hist_j = _to_jax_history(hist_b)
    head_j = jnp.asarray(hist_b.head, dtype=jnp.int32)
    adj_pos_j = jnp.asarray(adjusted_pos)
    adj_quat_j = jnp.asarray(adjusted_quat)
    rollout = _make_rollout(env, cfg.horizon_steps, cfg.w_rot)

    # Compile and check one complete device-side rollout.
    t_compile = time.perf_counter()
    out = rollout(state_b, hist_j, head_j, adj_pos_j, adj_quat_j, jnp.asarray(0, dtype=jnp.int32))
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t_compile

    # Baseline warm call, then identical steady-state repeats.
    env.rollout_horizon(
        snap, adjusted_pos, adjusted_quat, cfg.horizon_steps, cfg.w_rot,
        initial_batch=env.broadcast_snapshot(snap, cfg.num_samples),
    )
    baseline_ms: list[float] = []
    baseline_actions: list[np.ndarray] = []
    original_policy_forward = env._policy_forward
    capture = {"actions": []}

    def capture_policy(*a, **kw):
        tgt, action = original_policy_forward(*a, **kw)
        capture["actions"].append(np.asarray(action).copy())
        return tgt, action

    env._policy_forward = capture_policy  # type: ignore[method-assign]
    for _ in range(args.repeats):
        capture["actions"] = []
        t0 = time.perf_counter()
        costs = env.rollout_horizon(
            snap, adjusted_pos, adjusted_quat, cfg.horizon_steps, cfg.w_rot,
            initial_batch=env.broadcast_snapshot(snap, cfg.num_samples),
        )
        baseline_ms.append(1e3 * (time.perf_counter() - t0))
        baseline_actions.append(np.stack(capture["actions"], axis=1))
    env._policy_forward = original_policy_forward  # type: ignore[method-assign]

    jax_ms: list[float] = []
    jax_actions = None
    jax_cost = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        jout = rollout(state_b, hist_j, head_j, adj_pos_j, adj_quat_j, jnp.asarray(0, dtype=jnp.int32))
        jax.block_until_ready(jout)
        jax_ms.append(1e3 * (time.perf_counter() - t0))
        _, trace = jout
        _, robot_pos, robot_quat, tgt_seq, raw_seq = trace
        # Baseline capture stores the raw policy action, so compare it with
        # raw_seq rather than the PD target sequence.
        jax_actions = np.asarray(raw_seq).transpose(1, 0, 2)

    # Recompute JAX cost in the same device graph's output layout.  The
    # graph already computed costs; the host conversion here is outside the
    # timing interval and is used only for comparison.
    _, trace = rollout(state_b, hist_j, head_j, adj_pos_j, adj_quat_j, jnp.asarray(0, dtype=jnp.int32))
    jax_costs, robot_pos, robot_quat, tgt_seq, raw_seq = trace
    jax_costs = np.asarray(jax_costs)
    base_costs = np.asarray(costs)
    action_abs = np.abs(jax_actions - baseline_actions[-1])
    result = {
        "setting": {"motion": "push1", "N": cfg.num_samples, "H": cfg.horizon_steps, "cem_iters": 1, "seed": args.seed, "decimation": env.decimation},
        "compile_s": compile_s,
        "baseline_ms": _stats(baseline_ms),
        "jax_closed_loop_ms": _stats(jax_ms),
        "speedup": float(np.median(baseline_ms) / np.median(jax_ms)),
        "cost_abs_max": float(np.max(np.abs(jax_costs - base_costs))),
        "cost_abs_mean": float(np.mean(np.abs(jax_costs - base_costs))),
        "action_abs_max": float(np.max(action_abs)),
        "action_abs_mean": float(np.mean(action_abs)),
        "baseline_cost_mean": float(np.mean(base_costs)),
        "jax_cost_mean": float(np.mean(jax_costs)),
        "note": "isolated diagnostic; production files are not modified",
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
