"""MJX + TorchScript π closed-loop env with wrist-window command injection.

Supports single-env execute and batched CEM rollouts (host Torch π + vmap MJX PD).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch
from mujoco import mjx

from cem_wrist import (
    DEFAULT_FUTURE_IDX,
    pos_quat_from_flat,
    quat_error_magnitude,
    weighted_wrist_tracking_cost,
)
from policy_inputs import build_policy_targets_batch, build_policy_targets_single


_HISTORY_ORDER_CACHE: dict[int, np.ndarray] = {}


def quat_apply_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q_wxyz[..., :1], q_wxyz[..., 1:]
    t = np.cross(xyz, v) * 2.0
    return v + w * t + np.cross(xyz, t)


def quat_apply_inverse_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    qinv = np.stack([w, -x, -y, -z], axis=-1)
    return quat_apply_np(qinv, v)


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_mul_inverse_left_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w, x, y, z = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    qinv = np.stack([w, -x, -y, -z], axis=-1)
    return quat_mul_np(qinv, q2)


@dataclass
class PiHistory:
    root_quat: np.ndarray  # (1, C, 4) or (N, C, 4)
    base_ang: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    action: np.ndarray
    head: int = 0

    def copy(self) -> "PiHistory":
        return PiHistory(
            root_quat=self.root_quat.copy(),
            base_ang=self.base_ang.copy(),
            dof_pos=self.dof_pos.copy(),
            dof_vel=self.dof_vel.copy(),
            action=self.action.copy(),
            head=self.head,
        )

    def ordered(self, values: np.ndarray) -> np.ndarray:
        """Return history in oldest-to-newest order without rolling storage."""
        context_len = values.shape[1]
        all_indices = _HISTORY_ORDER_CACHE.get(context_len)
        if all_indices is None:
            offsets = np.arange(context_len, dtype=np.int64)
            all_indices = (offsets[None, :] + offsets[:, None] + 1) % context_len
            _HISTORY_ORDER_CACHE[context_len] = all_indices
        return values[:, all_indices[self.head]]


@dataclass
class EnvSnapshot:
    mjx_data: mjx.Data
    hist: PiHistory
    time_step: int


class MjxPiEnv:
    """Closed-loop UMI-2 on ScaleBridge MJX XML + Isaac-ordered TorchScript π."""

    def __init__(
        self,
        artifacts: str | Path,
        xml_path: str,
        motion_npz: str,
        mode_index: int | None = None,
        low_dt: float = 0.005,
        decimation: int = 4,
        torch_device: str = "cpu",
        profile_rollout: bool = False,
        artifact_manifest: str | Path | None = None,
    ):
        art = Path(artifacts)
        from load_eager_policy import load_eager_policy, resolve_policy_artifacts

        artifact_spec = resolve_policy_artifacts(art, artifact_manifest)
        meta = json.loads(artifact_spec.metadata_path.read_text())
        self.meta = meta
        self.artifact_spec = artifact_spec
        # Keep the exported policy input deterministic and identical to the
        # ScaleTrack/manual MuJoCo path.  Negative offsets use a random command
        # timestamp in ScaleTrack and are not a previous-frame convention.
        self.future_idx = list(DEFAULT_FUTURE_IDX)
        self.context_len = int(meta["history_buffer_size"])
        self.selected = list(meta["selected_body_names"])
        self.policy_joints = list(meta["joint_names"])
        self.body_names = list(meta["body_names"])
        stiff_pol = np.asarray(meta["stiffness"], dtype=np.float64)
        damp_pol = np.asarray(meta["damping"], dtype=np.float64)
        self.default_pol = np.asarray(meta["default_dof_pos"], dtype=np.float64)

        # Init JAX CUDA *before* moving Torch modules onto GPU to avoid cuDNN clashes.
        print(f"[mjx_pi] jax devices={jax.devices()}", flush=True)
        _ = jnp.zeros((1,), dtype=jnp.float32).block_until_ready()

        # Eager π on the requested device (GPU for CEM + execute).
        # TorchScript export has CPU-baked index buffers — keep it CPU-only as optional parity check.
        if torch_device == "cuda" and not torch.cuda.is_available():
            print("[mjx_pi] CUDA torch unavailable; falling back to cpu for π", flush=True)
            torch_device = "cpu"
        self.device = torch.device(torch_device)
        # This is deliberately opt-in: stage timing synchronizes JAX/Torch and
        # is useful for diagnosis, but would distort the normal pipeline.
        self.profile_rollout = bool(profile_rollout)
        self._rollout_stage_totals: dict[str, float] = {}
        self._rollout_stage_counts: dict[str, int] = {}
        self._rollout_stage_by_batch: dict[int, dict[str, dict[str, float | int]]] = {}
        self._rollout_runs: dict[int, dict[str, float | int]] = {}
        self.policy_ts = None
        ts_path = artifact_spec.torchscript_path
        if ts_path.is_file() and self.device.type == "cpu":
            self.policy_ts = torch.jit.load(str(ts_path), map_location="cpu")
            self.policy_ts.eval()

        self.policy, _ = load_eager_policy(
            art,
            device=torch_device,
            artifact_manifest=artifact_manifest,
            verify_checkpoint=False,
        )
        print(
            f"[mjx_pi] eager π on {self.device}"
            + (" (+ TorchScript cpu available)" if self.policy_ts is not None else " (CEM+execute)"),
            flush=True,
        )

        self.npz_body_index = {n: i for i, n in enumerate(self.body_names)}
        self.sel_idx = np.array([self.npz_body_index[n] for n in self.selected], dtype=np.int64)
        self.lw_i = self.npz_body_index["left_wrist_yaw_link"]
        self.rw_i = self.npz_body_index["right_wrist_yaw_link"]
        self.lw_sel = self.selected.index("left_wrist_yaw_link")
        self.rw_sel = self.selected.index("right_wrist_yaw_link")
        self.load_motion(motion_npz)

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.timestep = low_dt
        self.decimation = int(decimation)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sim_joints = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.mj_model.nu)
        ]
        assert len(self.sim_joints) == 29
        self.sim_idx_for_policy = np.array(
            [self.sim_joints.index(n) for n in self.policy_joints], dtype=np.int64
        )
        self.policy_idx_for_sim = np.array(
            [self.policy_joints.index(n) for n in self.sim_joints], dtype=np.int64
        )
        self.stiff_sim = stiff_pol[self.policy_idx_for_sim]
        self.damp_sim = damp_pol[self.policy_idx_for_sim]
        self.default_sim = self.default_pol[self.policy_idx_for_sim]
        self.lw_bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.rw_bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.mjx_model = mjx.put_model(self.mj_model)
        kp = jnp.asarray(self.stiff_sim)
        kd = jnp.asarray(self.damp_sim)
        n_sub = self.decimation

        def _pd_n_sub(st, tgt_sim):
            def body(_, s):
                q = s.qpos[7 : 7 + 29]
                qd = s.qvel[6 : 6 + 29]
                torque = (tgt_sim - q) * kp - qd * kd
                s = s.replace(ctrl=torque)
                return mjx.step(self.mjx_model, s)

            return jax.lax.fori_loop(0, n_sub, body, st)

        self._pd_n_sub = jax.jit(_pd_n_sub)
        self._pd_n_sub_batch = jax.jit(jax.vmap(_pd_n_sub, in_axes=(0, 0)))
        self._pd_compiled_ns: set[int] = set()
        self._policy_mode_cache: dict[int, torch.Tensor] = {}
        self._policy_offset_cache: dict[int, torch.Tensor] = {}

        st0 = mjx.put_data(self.mj_model, self.mj_data)
        _ = self._pd_n_sub(st0, jnp.asarray(self.default_sim))
        # Batch shapes are compiled by the runner after it knows the requested
        # CEM sample count.  Compiling an unconditional N=8 shape here adds
        # startup latency and memory pressure to every worker, including the
        # baseline path that never uses batched rollouts.
        print("[mjx_pi] PD single compile done", flush=True)

        self.mode_index_val = artifact_spec.mode_index if mode_index is None else int(mode_index)
        num_modes = int(self.policy.mode_vectors.shape[0])
        if not 0 <= self.mode_index_val < num_modes:
            raise ValueError(
                f"mode_index={self.mode_index_val} is outside artifact mode table [0, {num_modes})"
            )
        self.time_offsets = torch.as_tensor(
            self.future_idx, dtype=torch.long, device=self.device
        )[None, :, None]

        self.state: mjx.Data | None = None
        self.hist: PiHistory | None = None
        self.time_step = 0

    def load_motion(self, motion_npz: str | Path) -> None:
        """Swap clip arrays without reloading the policy or MJX model."""
        motion = np.load(motion_npz)
        if abs(float(motion["fps"]) - 50.0) >= 1e-6:
            raise ValueError(f"motion fps must be 50, got {motion['fps']}")
        self.T = int(motion["joint_pos"].shape[0])
        self.ref_body_pos = motion["body_pos_w"].astype(np.float64)
        self.ref_body_quat = motion["body_quat_w"].astype(np.float64)
        self.ref_joint_pos = motion["joint_pos"].astype(np.float64)
        self.ref_joint_vel = motion["joint_vel"].astype(np.float64)
        self.body_lin_vel_w = motion["body_lin_vel_w"].astype(np.float64)
        self.body_ang_vel_w = motion["body_ang_vel_w"].astype(np.float64)
        if self.ref_body_pos.shape[0] != self.T or self.ref_body_quat.shape[0] != self.T:
            raise ValueError(
                "motion body arrays must match joint_pos length: "
                f"T={self.T} pos={self.ref_body_pos.shape} quat={self.ref_body_quat.shape}"
            )
        expected_bodies = len(self.body_names)
        if self.ref_body_pos.shape[1] != expected_bodies:
            raise ValueError(
                f"motion body count {self.ref_body_pos.shape[1]} != artifact {expected_bodies}"
            )

    def warm_compile_batch(self, n: int) -> None:
        """JIT-compile batched PD+MJX for a given CEM sample count."""
        n = int(n)
        if n in self._pd_compiled_ns:
            return
        st0 = mjx.put_data(self.mj_model, self.mj_data)
        st_b = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], n, axis=0), st0
        )
        tg = jnp.repeat(jnp.asarray(self.default_sim)[None, :], n, axis=0)
        out = self._pd_n_sub_batch(st_b, tg)
        jax.block_until_ready(out.qpos)
        self._pd_compiled_ns.add(n)
        print(f"[mjx_pi] PD batch N={n} compiled", flush=True)

    def reset_rsi(self) -> None:
        qpos0 = np.zeros(self.mj_model.nq, dtype=np.float64)
        qvel0 = np.zeros(self.mj_model.nv, dtype=np.float64)
        pel = self.npz_body_index["pelvis"]
        qpos0[:3] = self.ref_body_pos[0, pel]
        qpos0[3:7] = self.ref_body_quat[0, pel]
        qpos0[7 : 7 + 29] = self.ref_joint_pos[0, self.policy_idx_for_sim]
        qvel0[:3] = self.body_lin_vel_w[0, pel]
        qvel0[3:6] = self.body_ang_vel_w[0, pel]
        qvel0[6 : 6 + 29] = self.ref_joint_vel[0, self.policy_idx_for_sim]
        self.mj_data.qpos[:] = qpos0
        self.mj_data.qvel[:] = qvel0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.state = mjx.put_data(self.mj_model, self.mj_data)
        dof0 = self.ref_joint_pos[0].astype(np.float32)
        self.hist = PiHistory(
            root_quat=np.repeat(qpos0[3:7][None, :], self.context_len, axis=0)[None, ...].astype(
                np.float32
            ),
            base_ang=np.zeros((1, self.context_len, 3), dtype=np.float32),
            dof_pos=np.repeat(dof0[None, :], self.context_len, axis=0)[None, ...],
            dof_vel=np.zeros((1, self.context_len, 29), dtype=np.float32),
            action=np.zeros((1, self.context_len, 29), dtype=np.float32),
            head=self.context_len - 1,
        )
        self.time_step = 0

    def snapshot(self) -> EnvSnapshot:
        assert self.state is not None and self.hist is not None
        return EnvSnapshot(
            mjx_data=self.state,
            hist=self.hist.copy(),
            time_step=int(self.time_step),
        )

    def restore(self, snap: EnvSnapshot) -> None:
        self.state = snap.mjx_data
        self.hist = snap.hist.copy()
        self.time_step = int(snap.time_step)

    def _read_robot_wrists(self, state: mjx.Data) -> tuple[np.ndarray, np.ndarray]:
        xpos = np.asarray(state.xpos)
        xquat = np.asarray(state.xquat)
        pos = np.stack([xpos[self.lw_bid], xpos[self.rw_bid]], axis=0)
        quat = np.stack([xquat[self.lw_bid], xquat[self.rw_bid]], axis=0)
        return pos, quat

    def true_wrist_at(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        ti = min(max(t, 0), self.T - 1)
        pos = np.stack([self.ref_body_pos[ti, self.lw_i], self.ref_body_pos[ti, self.rw_i]], axis=0)
        quat = np.stack(
            [self.ref_body_quat[ti, self.lw_i], self.ref_body_quat[ti, self.rw_i]], axis=0
        )
        return pos, quat

    def _build_targets(
        self,
        t: int,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        u_pos_w: np.ndarray | None,
        u_quat_w: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (F, B, 3/4) base-frame targets; optionally override wrists with world u."""
        return build_policy_targets_single(
            self.ref_body_pos,
            self.ref_body_quat,
            t,
            self.future_idx,
            self.sel_idx,
            root_pos,
            root_quat,
            self.lw_sel,
            self.rw_sel,
            u_pos_w,
            u_quat_w,
        )

    def _policy_forward(
        self,
        hist: PiHistory,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        *,
        profile: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """π forward on self.device. Prefer TorchScript only when on CPU and available."""
        if target_pos.ndim == 3:
            target_pos = target_pos[None, ...]
            target_rot = target_rot[None, ...]
        n = hist.root_quat.shape[0]
        device = self.device
        use_ts = n == 1 and self.policy_ts is not None and device.type == "cpu"
        t_static = time.perf_counter() if profile else 0.0
        mode, toff = self._policy_static_inputs(n)
        if profile:
            self._record_rollout_stage("policy_static", time.perf_counter() - t_static, n)

        t_prepare = time.perf_counter() if profile else 0.0
        host_args = (
            np.ascontiguousarray(hist.ordered(hist.root_quat)),
            np.ascontiguousarray(hist.ordered(hist.base_ang)),
            np.ascontiguousarray(hist.ordered(hist.dof_pos)),
            np.ascontiguousarray(hist.ordered(hist.dof_vel)),
            np.ascontiguousarray(hist.ordered(hist.action)),
            np.ascontiguousarray(target_pos),
            np.ascontiguousarray(target_rot),
        )
        if profile:
            self._record_rollout_stage("policy_host_prepare", time.perf_counter() - t_prepare, n)

        with torch.inference_mode():
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            t_h2d = time.perf_counter() if profile else 0.0
            args = (
                torch.from_numpy(host_args[0]).to(device),
                torch.from_numpy(host_args[1]).to(device),
                torch.from_numpy(host_args[2]).to(device),
                torch.from_numpy(host_args[3]).to(device),
                torch.from_numpy(host_args[4]).to(device),
                torch.from_numpy(host_args[5]).to(device),
                torch.from_numpy(host_args[6]).to(device),
                mode,
                toff,
            )
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            if profile:
                self._record_rollout_stage("policy_h2d", time.perf_counter() - t_h2d, n)

            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            t_policy = time.perf_counter() if profile else 0.0
            if use_ts:
                tgt_dof, action = self.policy_ts(*args)
            else:
                tgt_dof, action = self.policy(*args)
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            if profile:
                self._record_rollout_stage("policy_forward", time.perf_counter() - t_policy, n)

            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            t_d2h = time.perf_counter() if profile else 0.0
            tgt_dof_np = tgt_dof.detach().cpu().numpy().astype(np.float64)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            if profile:
                self._record_rollout_stage("policy_d2h", time.perf_counter() - t_d2h, n)
        return (
            tgt_dof_np,
            action_np,
        )

    def _record_rollout_stage(self, stage: str, elapsed_s: float, n: int) -> None:
        """Accumulate opt-in rollout stage timings without touching normal runs."""
        if not self.profile_rollout:
            return
        self._rollout_stage_totals[stage] = self._rollout_stage_totals.get(stage, 0.0) + elapsed_s
        self._rollout_stage_counts[stage] = self._rollout_stage_counts.get(stage, 0) + 1
        per_batch = self._rollout_stage_by_batch.setdefault(n, {})
        item = per_batch.setdefault(stage, {"total_s": 0.0, "calls": 0})
        item["total_s"] = float(item["total_s"]) + elapsed_s
        item["calls"] = int(item["calls"]) + 1

    def rollout_profile_summary(self) -> dict:
        """Return JSON-ready stage timing aggregates from profiled rollouts."""
        if not self.profile_rollout:
            return {"enabled": False}

        stages = {
            stage: {
                "total_s": total,
                "calls": self._rollout_stage_counts[stage],
                "mean_ms": 1e3 * total / max(self._rollout_stage_counts[stage], 1),
            }
            for stage, total in sorted(self._rollout_stage_totals.items())
        }
        batches = {}
        for n, data in sorted(self._rollout_stage_by_batch.items()):
            runs = self._rollout_runs.get(n, {})
            batches[str(n)] = {
                "rollout_calls": int(runs.get("calls", 0)),
                "horizon_steps": int(runs.get("steps", 0)),
                "wall_s": float(runs.get("wall_s", 0.0)),
                "stages": {
                    stage: {
                        "total_s": float(item["total_s"]),
                        "calls": int(item["calls"]),
                        "mean_ms": 1e3 * float(item["total_s"]) / max(int(item["calls"]), 1),
                    }
                    for stage, item in sorted(data.items())
                },
            }
        return {"enabled": True, "stages": stages, "by_batch": batches}

    def _policy_static_inputs(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached batch-shape-dependent policy inputs without eager allocation."""
        mode = self._policy_mode_cache.get(n)
        if mode is None:
            mode = torch.full(
                (n,), self.mode_index_val, dtype=torch.long, device=self.device
            )
            self._policy_mode_cache[n] = mode

        offsets = self._policy_offset_cache.get(n)
        if offsets is None:
            offsets = self.time_offsets.expand(n, -1, -1)
            self._policy_offset_cache[n] = offsets
        return mode, offsets

    def _push_hist(
        self,
        hist: PiHistory,
        root_quat: np.ndarray,
        base_ang: np.ndarray,
        dof_pol: np.ndarray,
        vel_pol: np.ndarray,
    ) -> PiHistory:
        h = hist
        h.head = (h.head + 1) % h.root_quat.shape[1]
        # Destination buffers are float32, so assignment performs the same
        # cast without allocating four temporary arrays at every step.
        h.root_quat[:, h.head] = root_quat
        h.base_ang[:, h.head] = base_ang
        h.dof_pos[:, h.head] = dof_pol
        h.dof_vel[:, h.head] = vel_pol
        return h

    def step_with_window(
        self,
        u_pos_w: np.ndarray | None,
        u_quat_w: np.ndarray | None,
        *,
        advance_time: bool = True,
    ) -> dict:
        """One control step using an optional global wrist policy window."""
        assert self.state is not None and self.hist is not None
        qpos = np.asarray(self.state.qpos)
        qvel = np.asarray(self.state.qvel)
        root_pos = qpos[:3]
        root_quat = qpos[3:7]
        dof_sim = qpos[7 : 7 + 29]
        vel_sim = qvel[6 : 6 + 29]
        dof_pol = dof_sim[self.sim_idx_for_policy]
        vel_pol = vel_sim[self.sim_idx_for_policy]
        base_ang = qvel[3:6]

        self.hist = self._push_hist(self.hist, root_quat, base_ang, dof_pol, vel_pol)

        target_pos, target_rot = self._build_targets(
            self.time_step, root_pos, root_quat, u_pos_w, u_quat_w
        )
        tgt_pol, act = self._policy_forward(self.hist, target_pos, target_rot)
        # State history advances before the policy call; action history advances
        # after it.  This preserves the legacy FIFO alignment without np.roll.
        action_slot = (self.hist.head + 1) % self.hist.action.shape[1]
        self.hist.action[0, action_slot] = act[0]
        tgt_sim = tgt_pol[0, self.policy_idx_for_sim]
        self.state = self._pd_n_sub(self.state, jnp.asarray(tgt_sim))

        ref_t = min(self.time_step + 1, self.T - 1)
        rpos, rquat = self.true_wrist_at(ref_t)
        xpos, xquat = self._read_robot_wrists(self.state)
        err_l = float(np.linalg.norm(xpos[0] - rpos[0]))
        err_r = float(np.linalg.norm(xpos[1] - rpos[1]))
        rot_l = float(quat_error_magnitude(rquat[0], xquat[0]))
        rot_r = float(quat_error_magnitude(rquat[1], xquat[1]))
        if advance_time:
            self.time_step += 1
        return {
            "err_pos_left": err_l,
            "err_pos_right": err_r,
            "err_pos_mean": 0.5 * (err_l + err_r),
            "err_rot_left": rot_l,
            "err_rot_right": rot_r,
            "err_rot_mean": 0.5 * (rot_l + rot_r),
            "robot_wrist_pos": xpos,
            "robot_wrist_quat": xquat,
            "ref_wrist_pos": rpos,
            "ref_wrist_quat": rquat,
        }

    def step_with_u(
        self,
        u_flat: np.ndarray | None,
        *,
        advance_time: bool = True,
    ) -> dict:
        """Compatibility wrapper for a direct global-rotvec window."""
        if u_flat is None:
            return self.step_with_window(None, None, advance_time=advance_time)
        pos, quat = pos_quat_from_flat(
            np.asarray(u_flat)[None, :], len(self.future_idx)
        )
        return self.step_with_window(pos[0], quat[0], advance_time=advance_time)

    def broadcast_snapshot(self, snap: EnvSnapshot, n: int) -> tuple[mjx.Data, PiHistory]:
        """Clone single-env snapshot to N parallel states."""
        def rep(x):
            x = jnp.asarray(x)
            return jnp.repeat(x[None, ...], n, axis=0)

        # mjx.Data fields used by step
        data = snap.mjx_data
        batch = jax.tree_util.tree_map(rep, data)
        h = snap.hist
        hist = PiHistory(
            root_quat=np.repeat(h.root_quat, n, axis=0),
            base_ang=np.repeat(h.base_ang, n, axis=0),
            dof_pos=np.repeat(h.dof_pos, n, axis=0),
            dof_vel=np.repeat(h.dof_vel, n, axis=0),
            action=np.repeat(h.action, n, axis=0),
            head=h.head,
        )
        return batch, hist

    def _build_targets_batch(
        self,
        t: int,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        u_pos_w: np.ndarray,
        u_quat_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized targets. root_*: (N,3/4), u_*: (N,F,2,3/4) → (N,F,B,3/4)."""
        return build_policy_targets_batch(
            self.ref_body_pos,
            self.ref_body_quat,
            t,
            self.future_idx,
            self.sel_idx,
            root_pos,
            root_quat,
            self.lw_sel,
            self.rw_sel,
            u_pos_w,
            u_quat_w,
        )

    def rollout_horizon(
        self,
        snap: EnvSnapshot,
        adjusted_pos: np.ndarray,
        adjusted_quat: np.ndarray,
        horizon: int,
        w_rot: float,
        initial_batch: tuple[mjx.Data, PiHistory] | None = None,
    ) -> np.ndarray:
        """Roll out one complete adjusted wrist motion per batched candidate."""
        t_rollout = time.perf_counter()
        adjusted_pos = np.asarray(adjusted_pos, dtype=np.float64)
        adjusted_quat = np.asarray(adjusted_quat, dtype=np.float64)
        if adjusted_pos.ndim == 3:
            adjusted_pos = adjusted_pos[None, ...]
            adjusted_quat = adjusted_quat[None, ...]
        n = adjusted_pos.shape[0]
        if adjusted_pos.shape[1] < horizon + len(self.future_idx):
            raise ValueError("adjusted reference does not cover rollout windows")
        t_setup = time.perf_counter()
        if initial_batch is None:
            batch_state, hist = self.broadcast_snapshot(snap, n)
        else:
            batch_state, hist = initial_batch
            hist = hist.copy()
        if self.profile_rollout:
            jax.block_until_ready(batch_state.qpos)
        self._record_rollout_stage("rollout_setup", time.perf_counter() - t_setup, n)
        t0 = snap.time_step
        robot_pos_steps: list[np.ndarray] = []
        robot_quat_steps: list[np.ndarray] = []
        ref_pos_steps: list[np.ndarray] = []
        ref_quat_steps: list[np.ndarray] = []
        num_frames = len(self.future_idx)

        steps_run = 0
        for h in range(horizon):
            t = t0 + h
            if t >= self.T - 1:
                break
            t_state = time.perf_counter()
            qpos = np.asarray(batch_state.qpos)
            qvel = np.asarray(batch_state.qvel)
            self._record_rollout_stage("state_d2h", time.perf_counter() - t_state, n)
            root_pos = qpos[:, :3]
            root_quat = qpos[:, 3:7]
            dof_sim = qpos[:, 7 : 7 + 29]
            vel_sim = qvel[:, 6 : 6 + 29]
            dof_pol = dof_sim[:, self.sim_idx_for_policy]
            vel_pol = vel_sim[:, self.sim_idx_for_policy]
            base_ang = qvel[:, 3:6]

            t_history = time.perf_counter()
            hist = self._push_hist(hist, root_quat, base_ang, dof_pol, vel_pol)
            self._record_rollout_stage("history_update", time.perf_counter() - t_history, n)

            t_targets = time.perf_counter()
            target_pos, target_rot = self._build_targets_batch(
                t,
                root_pos,
                root_quat,
                adjusted_pos[:, h : h + num_frames],
                adjusted_quat[:, h : h + num_frames],
            )
            self._record_rollout_stage("target_build", time.perf_counter() - t_targets, n)
            tgt_pol, act = self._policy_forward(
                hist, target_pos, target_rot, profile=self.profile_rollout
            )
            action_slot = (hist.head + 1) % hist.action.shape[1]
            hist.action[:, action_slot] = act
            tgt_sim = tgt_pol[:, self.policy_idx_for_sim]
            t_mjx_input = time.perf_counter()
            tgt_sim_jax = jnp.asarray(tgt_sim)
            if self.profile_rollout:
                jax.block_until_ready(tgt_sim_jax)
            self._record_rollout_stage("mjx_target_h2d", time.perf_counter() - t_mjx_input, n)

            t_mjx = time.perf_counter()
            batch_state = self._pd_n_sub_batch(batch_state, tgt_sim_jax)
            if self.profile_rollout:
                # Make all fields consumed in this loop ready here.  Otherwise
                # a lazy xpos/xquat computation would be charged to the next
                # host transfer rather than to the MJX step itself.
                jax.block_until_ready(
                    (batch_state.qpos, batch_state.qvel, batch_state.xpos, batch_state.xquat)
                )
            self._record_rollout_stage("mjx_pd_4substeps", time.perf_counter() - t_mjx, n)

            t_wrist = time.perf_counter()
            ref_t = min(t + 1, self.T - 1)
            rpos, rquat = self.true_wrist_at(ref_t)
            xpos = np.asarray(batch_state.xpos)
            xquat = np.asarray(batch_state.xquat)
            robot_pos = np.stack([xpos[:, self.lw_bid], xpos[:, self.rw_bid]], axis=1)
            robot_quat = np.stack([xquat[:, self.lw_bid], xquat[:, self.rw_bid]], axis=1)
            self._record_rollout_stage("wrist_state_d2h", time.perf_counter() - t_wrist, n)
            robot_pos_steps.append(robot_pos)
            robot_quat_steps.append(robot_quat)
            ref_pos_steps.append(rpos)
            ref_quat_steps.append(rquat)
            steps_run += 1

        if not robot_pos_steps:
            return np.full(n, np.inf, dtype=np.float64)
        t_cost = time.perf_counter()
        robot_pos = np.stack(robot_pos_steps, axis=1)
        robot_quat = np.stack(robot_quat_steps, axis=1)
        ref_pos = np.stack(ref_pos_steps, axis=0)
        ref_quat = np.stack(ref_quat_steps, axis=0)
        cost = weighted_wrist_tracking_cost(
            robot_pos, robot_quat, ref_pos, ref_quat, w_rot
        )
        self._record_rollout_stage("cost_aggregation", time.perf_counter() - t_cost, n)
        if self.profile_rollout:
            run = self._rollout_runs.setdefault(n, {"calls": 0, "steps": 0, "wall_s": 0.0})
            run["calls"] = int(run["calls"]) + 1
            run["steps"] = int(run["steps"]) + steps_run
            run["wall_s"] = float(run["wall_s"]) + (time.perf_counter() - t_rollout)
        return cost
