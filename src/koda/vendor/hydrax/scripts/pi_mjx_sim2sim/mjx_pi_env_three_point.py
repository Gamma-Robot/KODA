"""Isolated torso + dual-wrist extension of the MJX ScaleTrack environment."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np

from cem_wrist import quat_error_magnitude
from mjx_pi_env import MjxPiEnv
from policy_inputs import _quat_apply_inverse_np, _quat_mul_inverse_left_np


TRACKING_LINK_NAMES: tuple[str, ...] = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
EXPECTED_SELECTED_SLOTS: tuple[int, ...] = (7, 10, 13)


class MjxPiEnvThreePoint(MjxPiEnv):
    """Use a three-point artifact without changing the dual-wrist environment."""

    def __init__(
        self,
        artifacts: str | Path,
        xml_path: str,
        motion_npz: str,
        *,
        artifact_manifest: str | Path,
        mode_index: int | None = None,
        low_dt: float = 0.005,
        decimation: int = 4,
        torch_device: str = "cpu",
        profile_rollout: bool = False,
    ) -> None:
        super().__init__(
            artifacts,
            xml_path,
            motion_npz,
            mode_index=mode_index,
            low_dt=low_dt,
            decimation=decimation,
            torch_device=torch_device,
            profile_rollout=profile_rollout,
            artifact_manifest=artifact_manifest,
        )
        self.track_motion_indices = np.asarray(
            [self.npz_body_index[name] for name in TRACKING_LINK_NAMES], dtype=np.int32
        )
        self.track_selected_slots = np.asarray(
            [self.selected.index(name) for name in TRACKING_LINK_NAMES], dtype=np.int32
        )
        self.track_mjx_body_ids = np.asarray(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in TRACKING_LINK_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.track_mjx_body_ids < 0):
            missing = [
                name
                for name, body_id in zip(TRACKING_LINK_NAMES, self.track_mjx_body_ids)
                if body_id < 0
            ]
            raise ValueError(f"three-point MuJoCo bodies not found: {missing}")
        if tuple(self.track_selected_slots.tolist()) != EXPECTED_SELECTED_SLOTS:
            raise ValueError(
                "three-point artifact selected slots changed: "
                f"expected {EXPECTED_SELECTED_SLOTS}, got {tuple(self.track_selected_slots.tolist())}"
            )
        if self.artifact_spec.manifest_path is None:
            raise ValueError("three-point environment requires an explicit artifact manifest")
        manifest = json.loads(self.artifact_spec.manifest_path.read_text())
        manifest_names = tuple(manifest["policy"]["selected_body_names"])
        if manifest_names != TRACKING_LINK_NAMES:
            raise ValueError(
                f"three-point artifact manifest links must be {TRACKING_LINK_NAMES}, got {manifest_names}"
            )

        self.torso_i = int(self.track_motion_indices[0])
        self.torso_sel = int(self.track_selected_slots[0])
        self.torso_bid = int(self.track_mjx_body_ids[0])

    def _read_robot_tracking(self, state) -> tuple[np.ndarray, np.ndarray]:
        xpos = np.asarray(state.xpos)
        xquat = np.asarray(state.xquat)
        return xpos[self.track_mjx_body_ids], xquat[self.track_mjx_body_ids]

    def true_tracking_at(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        ti = min(max(int(t), 0), self.T - 1)
        return (
            self.ref_body_pos[ti, self.track_motion_indices],
            self.ref_body_quat[ti, self.track_motion_indices],
        )

    def _build_targets(
        self,
        t: int,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        u_pos_w: np.ndarray | None,
        u_quat_w: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a policy window and override only the three active slots."""
        if (u_pos_w is None) != (u_quat_w is None):
            raise ValueError("three-point position and quaternion overrides must be provided together")
        idxs = np.clip(
            int(t) + np.asarray(self.future_idx, dtype=np.int64), 0, self.ref_body_pos.shape[0] - 1
        )
        body_pos = np.asarray(self.ref_body_pos[idxs][:, self.sel_idx], dtype=np.float64).copy()
        body_quat = np.asarray(self.ref_body_quat[idxs][:, self.sel_idx], dtype=np.float64).copy()
        if u_pos_w is not None:
            pos = np.asarray(u_pos_w, dtype=np.float64)
            quat = np.asarray(u_quat_w, dtype=np.float64)
            expected_pos = (len(self.future_idx), len(TRACKING_LINK_NAMES), 3)
            expected_quat = (len(self.future_idx), len(TRACKING_LINK_NAMES), 4)
            if pos.shape != expected_pos or quat.shape != expected_quat:
                raise ValueError(
                    f"three-point override shapes must be {expected_pos}/{expected_quat}, "
                    f"got {pos.shape}/{quat.shape}"
                )
            body_pos[:, self.track_selected_slots] = pos
            body_quat[:, self.track_selected_slots] = quat
        target_pos = _quat_apply_inverse_np(
            root_quat[None, None], body_pos - root_pos[None, None]
        ).astype(np.float32)
        target_rot = _quat_mul_inverse_left_np(root_quat[None, None], body_quat).astype(
            np.float32
        )
        return target_pos, target_rot

    def step_with_window(
        self,
        u_pos_w: np.ndarray | None,
        u_quat_w: np.ndarray | None,
        *,
        advance_time: bool = True,
    ) -> dict:
        """Execute one control step with torso/left-wrist/right-wrist targets."""
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
        action_slot = (self.hist.head + 1) % self.hist.action.shape[1]
        self.hist.action[0, action_slot] = act[0]
        tgt_sim = tgt_pol[0, self.policy_idx_for_sim]
        self.state = self._pd_n_sub(self.state, jnp.asarray(tgt_sim))

        ref_t = min(self.time_step + 1, self.T - 1)
        ref_pos, ref_quat = self.true_tracking_at(ref_t)
        robot_pos, robot_quat = self._read_robot_tracking(self.state)
        pos_error = np.linalg.norm(robot_pos - ref_pos, axis=-1)
        rot_error = quat_error_magnitude(ref_quat, robot_quat)
        if advance_time:
            self.time_step += 1
        return {
            "err_pos_torso": float(pos_error[0]),
            "err_pos_left": float(pos_error[1]),
            "err_pos_right": float(pos_error[2]),
            "err_pos_wrist_mean": float(np.mean(pos_error[1:])),
            "err_rot_torso": float(rot_error[0]),
            "err_rot_left": float(rot_error[1]),
            "err_rot_right": float(rot_error[2]),
            "err_rot_wrist_mean": float(np.mean(rot_error[1:])),
            "robot_tracking_pos": robot_pos,
            "robot_tracking_quat": robot_quat,
            "ref_tracking_pos": ref_pos,
            "ref_tracking_quat": ref_quat,
        }
