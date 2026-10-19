"""Dual-wrist Mink IK for the G1 29-DoF body model.

This replaces the historical bytecode loader. Callers freeze the floating
base and non-selected hinges, track both wrist yaw bodies, and keep a
posture regularizer on the seed configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mink
import numpy as np
import qpsolvers

from st_wrist_ik import ARMS_ONLY, bare_joint_name, resolve_joint

__all__ = [
    "ARMS_ONLY",
    "MinkDualWristIK",
    "MinkSolveResult",
    "bare_joint_name",
    "build_model_qpos",
    "extract_joint_qpos",
    "resolve_joint",
]


DEFAULT_COLLISION_PAIRS = (
    (
        ("left_hand_coll", "left_forearm_coll"),
        ("torso_coll_l", "torso_coll_r", "pelvis_coll", "head_coll"),
    ),
    (
        ("right_hand_coll", "right_forearm_coll"),
        ("torso_coll_l", "torso_coll_r", "pelvis_coll", "head_coll"),
    ),
    (
        ("left_hand_coll", "left_forearm_coll"),
        ("right_hand_coll", "right_forearm_coll"),
    ),
)


def build_model_qpos(
    model: mujoco.MjModel,
    *,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_qpos: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64)
    free_id = -1
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            free_id = joint_id
            break
    if free_id < 0:
        raise ValueError("model has no free joint")
    adr = int(model.jnt_qposadr[free_id])
    q[adr : adr + 3] = np.asarray(root_pos, dtype=np.float64).reshape(3)
    quat = np.asarray(root_quat_wxyz, dtype=np.float64).reshape(4)
    q[adr + 3 : adr + 7] = quat / max(float(np.linalg.norm(quat)), 1e-12)
    for index, name in enumerate(joint_names):
        joint_id = resolve_joint(model, name)
        q[int(model.jnt_qposadr[joint_id])] = float(joint_qpos[index])
    return q


def extract_joint_qpos(
    model: mujoco.MjModel, qpos: np.ndarray, joint_names: list[str]
) -> np.ndarray:
    out = np.zeros(len(joint_names), dtype=np.float64)
    qpos = np.asarray(qpos, dtype=np.float64)
    for index, name in enumerate(joint_names):
        joint_id = resolve_joint(model, name)
        out[index] = qpos[int(model.jnt_qposadr[joint_id])]
    return out


def _se3(pos: np.ndarray, quat_wxyz: np.ndarray) -> mink.SE3:
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    return mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=quat), pos)


def _ori_error_rad(current: np.ndarray, target: np.ndarray) -> float:
    q1 = current / max(float(np.linalg.norm(current)), 1e-12)
    q2 = target / max(float(np.linalg.norm(target)), 1e-12)
    dot = float(np.abs(np.clip(np.dot(q1, q2), -1.0, 1.0)))
    return float(2.0 * np.arccos(min(1.0, dot)))


def _geom_exists(model: mujoco.MjModel, name: str) -> bool:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0


def _solve_ik_with_frozen_dofs(
    configuration: mink.Configuration,
    tasks: list,
    dt: float,
    solver: str,
    damping: float,
    limits: list,
    frozen_dof_indices: np.ndarray,
) -> np.ndarray:
    """Solve Mink IK while keeping non-arm DoFs fixed across Mink versions.

    Mink 1.x exposes ``DofFreezingTask`` and accepts it through the
    ``constraints`` argument. The project uses MuJoCo 3.5, for which the
    last compatible Mink release is 0.0.13; that API has no freezing task and
    no ``constraints`` parameter. Build the same QP directly in that case and
    add ``A @ delta_q = 0`` rows for the frozen tangent coordinates.
    """
    if hasattr(mink, "DofFreezingTask"):
        freeze_task = mink.DofFreezingTask(
            configuration.model,
            frozen_dof_indices.tolist(),
            gain=1.0,
        )
        return mink.solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            damping=damping,
            safety_break=False,
            limits=limits,
            constraints=[freeze_task],
        )

    configuration.check_limits(safety_break=False)
    problem = mink.build_ik(
        configuration,
        tasks,
        dt,
        damping=damping,
        limits=limits,
    )
    frozen = np.asarray(frozen_dof_indices, dtype=int).reshape(-1)
    if frozen.size:
        freeze_matrix = np.eye(configuration.model.nv, dtype=float)[frozen]
        freeze_rhs = np.zeros(frozen.size, dtype=float)
        problem.A = (
            freeze_matrix
            if problem.A is None
            else np.vstack((np.asarray(problem.A), freeze_matrix))
        )
        problem.b = (
            freeze_rhs
            if problem.b is None
            else np.concatenate((np.asarray(problem.b), freeze_rhs))
        )
    result = qpsolvers.solve_problem(problem, solver=solver)
    if not result.found or result.x is None:
        raise mink.NoSolutionFound(solver)
    return np.asarray(result.x, dtype=np.float64) / float(dt)


@dataclass
class MinkSolveResult:
    qpos: np.ndarray
    success: bool
    n_iters: int
    max_arm_delta_rad: float
    left_pos_err_m: float
    right_pos_err_m: float
    left_ori_err_rad: float
    right_ori_err_rad: float


class MinkDualWristIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        left_frame: str,
        right_frame: str,
        position_cost: float = 1.0,
        orientation_cost: float = 0.08,
        posture_cost: float = 0.12,
        dt: float = 0.02,
        max_iters: int = 80,
        pos_tol_m: float = 0.01,
        ori_tol_rad: float = 0.2,
        use_collision: bool = False,
        collision_margin_m: float = 0.015,
        arm_vel_limit: float = 8.0,
        max_arm_delta_rad: float = 0.2,
        joint_names: list[str] | None = None,
        solver: str = "daqp",
    ) -> None:
        self.model = model
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.dt = float(dt)
        self.max_iters = int(max_iters)
        self.pos_tol_m = float(pos_tol_m)
        self.ori_tol_rad = float(ori_tol_rad)
        self.max_arm_delta_rad = float(max_arm_delta_rad)
        self.solver_name = solver
        self.joint_names = list(joint_names or ARMS_ONLY)
        self.configuration = mink.Configuration(model)
        self.left_task = mink.FrameTask(
            left_frame,
            "body",
            position_cost=float(position_cost),
            orientation_cost=float(orientation_cost),
        )
        self.right_task = mink.FrameTask(
            right_frame,
            "body",
            position_cost=float(position_cost),
            orientation_cost=float(orientation_cost),
        )
        self.posture_task = mink.PostureTask(model, cost=float(posture_cost))
        movable: list[int] = []
        for name in self.joint_names:
            joint_id = resolve_joint(model, name)
            dof = int(model.jnt_dofadr[joint_id])
            width = 1 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE else int(
                model.jnt_dofadr[joint_id + 1] - dof if joint_id + 1 < model.njnt else model.nv - dof
            )
            movable.extend(range(dof, dof + max(width, 1)))
        self._arm_dof_indices = sorted(set(movable))
        frozen = [index for index in range(model.nv) if index not in set(self._arm_dof_indices)]
        self._frozen_dof_indices = np.asarray(frozen, dtype=int)
        velocities = {name: np.asarray([float(arm_vel_limit)]) for name in self.joint_names}
        self.limits: list[mink.Limit] = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, velocities),
        ]
        if use_collision:
            pairs = []
            for left, right in DEFAULT_COLLISION_PAIRS:
                left_ok = [name for name in left if _geom_exists(model, name)]
                right_ok = [name for name in right if _geom_exists(model, name)]
                if left_ok and right_ok:
                    pairs.append((left_ok, right_ok))
            if pairs:
                self.limits.append(
                    mink.CollisionAvoidanceLimit(
                        model,
                        pairs,
                        gain=0.85,
                        minimum_distance_from_collisions=float(collision_margin_m),
                        collision_detection_distance=max(0.08, float(collision_margin_m)),
                    )
                )

    def solve(
        self,
        *,
        seed_qpos: np.ndarray,
        left_pos: np.ndarray,
        left_quat_wxyz: np.ndarray,
        right_pos: np.ndarray,
        right_quat_wxyz: np.ndarray,
        joint_names: list[str] | None = None,
    ) -> MinkSolveResult:
        del joint_names
        seed = np.asarray(seed_qpos, dtype=np.float64).reshape(self.model.nq).copy()
        self.configuration.update(seed)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.left_task.set_target(_se3(left_pos, left_quat_wxyz))
        self.right_task.set_target(_se3(right_pos, right_quat_wxyz))
        tasks = [self.left_task, self.right_task, self.posture_task]
        n_iters = 0
        success = False
        for n_iters in range(1, self.max_iters + 1):
            try:
                velocity = _solve_ik_with_frozen_dofs(
                    self.configuration,
                    tasks,
                    self.dt,
                    solver=self.solver_name,
                    damping=1e-3,
                    limits=self.limits,
                    frozen_dof_indices=self._frozen_dof_indices,
                )
            except mink.NoSolutionFound:
                break
            self.configuration.integrate_inplace(velocity, self.dt)
            errors = self._errors(left_pos, left_quat_wxyz, right_pos, right_quat_wxyz)
            if (
                errors["left_pos_err_m"] <= self.pos_tol_m
                and errors["right_pos_err_m"] <= self.pos_tol_m
                and errors["left_ori_err_rad"] <= self.ori_tol_rad
                and errors["right_ori_err_rad"] <= self.ori_tol_rad
            ):
                success = True
                break
        qpos = np.asarray(self.configuration.q, dtype=np.float64).copy()
        delta = self._selected_delta(seed, qpos)
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_delta > self.max_arm_delta_rad > 0:
            scale = self.max_arm_delta_rad / max_delta
            qpos = self._scale_selected(seed, qpos, scale)
            self.configuration.update(qpos)
            max_delta = float(self.max_arm_delta_rad)
        errors = self._errors(left_pos, left_quat_wxyz, right_pos, right_quat_wxyz)
        success = success and errors["left_pos_err_m"] <= max(0.05, 5.0 * self.pos_tol_m)
        return MinkSolveResult(
            qpos=qpos,
            success=bool(success),
            n_iters=int(n_iters),
            max_arm_delta_rad=max_delta,
            left_pos_err_m=errors["left_pos_err_m"],
            right_pos_err_m=errors["right_pos_err_m"],
            left_ori_err_rad=errors["left_ori_err_rad"],
            right_ori_err_rad=errors["right_ori_err_rad"],
        )

    def _selected_delta(self, seed: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        deltas = []
        for name in self.joint_names:
            joint_id = resolve_joint(self.model, name)
            adr = int(self.model.jnt_qposadr[joint_id])
            deltas.append(qpos[adr] - seed[adr])
        return np.asarray(deltas, dtype=np.float64)

    def _scale_selected(self, seed: np.ndarray, qpos: np.ndarray, scale: float) -> np.ndarray:
        out = qpos.copy()
        for name in self.joint_names:
            joint_id = resolve_joint(self.model, name)
            adr = int(self.model.jnt_qposadr[joint_id])
            out[adr] = seed[adr] + scale * (qpos[adr] - seed[adr])
        return out

    def _errors(
        self,
        left_pos: np.ndarray,
        left_quat: np.ndarray,
        right_pos: np.ndarray,
        right_quat: np.ndarray,
    ) -> dict[str, float]:
        left = self.configuration.get_transform_frame_to_world(self.left_frame, "body")
        right = self.configuration.get_transform_frame_to_world(self.right_frame, "body")
        return {
            "left_pos_err_m": float(np.linalg.norm(left.translation() - np.asarray(left_pos))),
            "right_pos_err_m": float(np.linalg.norm(right.translation() - np.asarray(right_pos))),
            "left_ori_err_rad": _ori_error_rad(left.rotation().wxyz, left_quat),
            "right_ori_err_rad": _ori_error_rad(right.rotation().wxyz, right_quat),
        }


def main() -> None:
    raise SystemExit("st_mink_wrist_ik is a library; use refine_three_point_kinematics_mink.py")


if __name__ == "__main__":
    main()
