"""Batched actuatorless MuJoCo-Warp world for G1 rollouts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from sil_trackers import DevicePDCommand, MotionGuidance, PDCommand, RobotState
from sil_tts.simulation.randomization import ScenarioBatch
from sil_tts.simulation.execution_state import ExecutionState

try:  # Keep CPU-only tests importable; construction still requires the runtime.
    import mujoco
    import mujoco_warp
    import warp as wp
except ImportError:  # pragma: no cover - exercised by the normal CPU environment
    mujoco = None
    mujoco_warp = None
    wp = None


if wp is not None:

    @wp.kernel
    def _gather_joints(
        qpos: wp.array2d(dtype=wp.float32),
        qvel: wp.array2d(dtype=wp.float32),
        qpos_addresses: wp.array(dtype=wp.int32),
        dof_addresses: wp.array(dtype=wp.int32),
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
    ):
        lane, joint = wp.tid()
        joint_position[lane, joint] = qpos[lane, qpos_addresses[joint]]
        joint_velocity[lane, joint] = qvel[lane, dof_addresses[joint]]


    @wp.kernel
    def _gather_bodies(
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        body_ids: wp.array(dtype=wp.int32),
        body_position: wp.array2d(dtype=wp.vec3),
        body_orientation: wp.array2d(dtype=wp.quat),
        body_linear_velocity: wp.array2d(dtype=wp.vec3),
        body_angular_velocity: wp.array2d(dtype=wp.vec3),
    ):
        lane, body = wp.tid()
        source = body_ids[body]
        velocity = cvel[lane, source]
        body_position[lane, body] = xpos[lane, source]
        body_orientation[lane, body] = xquat[lane, source]
        body_angular_velocity[lane, body] = wp.vec3(
            velocity[0], velocity[1], velocity[2]
        )
        body_linear_velocity[lane, body] = wp.vec3(
            velocity[3], velocity[4], velocity[5]
        )


    @wp.kernel
    def _apply_pd_force(
        qpos: wp.array2d(dtype=wp.float32),
        qvel: wp.array2d(dtype=wp.float32),
        position_target: wp.array2d(dtype=wp.float32),
        velocity_target: wp.array2d(dtype=wp.float32),
        feedforward_torque: wp.array2d(dtype=wp.float32),
        proportional_gain: wp.array2d(dtype=wp.float32),
        derivative_gain: wp.array2d(dtype=wp.float32),
        effort_limit: wp.array(dtype=wp.float32),
        actuation_strength: wp.array(dtype=wp.float32),
        qpos_addresses: wp.array(dtype=wp.int32),
        dof_addresses: wp.array(dtype=wp.int32),
        qfrc_applied: wp.array2d(dtype=wp.float32),
    ):
        lane, joint = wp.tid()
        qpos_address = qpos_addresses[joint]
        dof_address = dof_addresses[joint]
        force = (
            feedforward_torque[lane, joint]
            + proportional_gain[lane, joint]
            * (position_target[lane, joint] - qpos[lane, qpos_address])
            + derivative_gain[lane, joint]
            * (velocity_target[lane, joint] - qvel[lane, dof_address])
        )
        limit = effort_limit[joint]
        strength = actuation_strength[lane % actuation_strength.shape[0]]
        qfrc_applied[lane, dof_address] = strength * wp.clamp(
            force, -limit, limit
        )


    @wp.kernel
    def _scatter_kinematic(
        kinematic_qpos: wp.array2d(dtype=wp.float32),
        kinematic_qvel: wp.array2d(dtype=wp.float32),
        qpos_addresses: wp.array(dtype=wp.int32),
        dof_addresses: wp.array(dtype=wp.int32),
        qpos: wp.array2d(dtype=wp.float32),
        qvel: wp.array2d(dtype=wp.float32),
    ):
        lane, joint = wp.tid()
        qpos[lane, qpos_addresses[joint]] = kinematic_qpos[lane, joint]
        qvel[lane, dof_addresses[joint]] = kinematic_qvel[lane, joint]


@dataclass(frozen=True)
class MjwarpRobotStateView:
    """Borrowed declared-layout device arrays consumed by a batched tracker."""

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_position: Any
    joint_velocity: Any
    body_position_world: Any
    body_orientation_world: Any
    body_linear_velocity_world: Any
    body_angular_velocity_world: Any


def _runtime() -> tuple[Any, Any, Any]:
    if mujoco is None or mujoco_warp is None or wp is None:
        raise RuntimeError(
            "MjwarpWorld requires mujoco, mujoco-warp, and warp-lang"
        )
    return mujoco, mujoco_warp, wp


def _names(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{field} must contain unique, non-empty names")
    return names


def prepare_mjcf_spec(
    mj: Any,
    path: Path,
    *,
    joint_names: tuple[str, ...],
    joint_armature: np.ndarray | None,
    joint_frictionloss: np.ndarray | None,
    floor_friction: np.ndarray | None,
) -> Any:
    """Load one MJCF, drop actuators, and reuse or add the named floor."""

    spec = mj.MjSpec.from_file(str(path))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    if floor_friction is not None:
        existing = spec.geom("floor")
        if existing is None:
            spec.worldbody.add_geom(
                name="floor",
                type=mj.mjtGeom.mjGEOM_PLANE,
                size=[0.0, 0.0, 0.05],
                friction=floor_friction,
            )
        else:
            existing.friction = floor_friction
    for index, name in enumerate(joint_names):
        joint = spec.joint(name)
        if joint is None:
            raise ValueError(f"MJCF is missing joint {name!r}")
        if joint_armature is not None:
            joint.armature = float(joint_armature[index])
        if joint_frictionloss is not None:
            joint.frictionloss = float(joint_frictionloss[index])
    return spec


def _scaled_scenario_rows(
    nominal: np.ndarray, scales: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Repeat one nominal model field and scale selected entries per scenario."""
    values = np.repeat(np.asarray(nominal, dtype=np.float32)[None], len(scales), axis=0)
    trailing = (1,) * (values.ndim - 1)
    values[:, indices] *= np.asarray(scales, dtype=np.float32).reshape(
        (len(scales), *trailing)
    )
    return np.ascontiguousarray(values)


def _quat_velocity(
    before: np.ndarray, after: np.ndarray, dt: np.ndarray
) -> np.ndarray:
    """Vectorized ``mju_subQuat(after, before) / dt`` for unit wxyz quats."""

    aw, ax, ay, az = np.moveaxis(after, -1, 0)
    bw, bx, by, bz = np.moveaxis(before, -1, 0)
    relative = np.stack(
        (
            aw * bw + ax * bx + ay * by + az * bz,
            bw * ax - bx * aw - by * az + bz * ay,
            bw * ay + bx * az - by * aw - bz * ax,
            bw * az - bx * ay + by * ax - bz * aw,
        ),
        axis=-1,
    )
    relative /= np.linalg.norm(relative, axis=-1, keepdims=True)
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    sine = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sine, relative[..., :1])
    scale = np.divide(
        angle,
        sine,
        out=np.full_like(angle, 2.0),
        where=sine > 1.0e-12,
    )
    return vector * scale / dt[None, :, None]


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by unit wxyz quaternions."""

    xyz = quaternion[..., 1:]
    cross = np.cross(xyz, vector)
    return vector + 2.0 * (
        quaternion[..., :1] * cross + np.cross(xyz, cross)
    )


class MjwarpWorld:
    """One fixed-size CUDA batch using a deterministically composed MJCF.

    Actuators are stripped at spec load so PD force can be written to
    ``qfrc_applied``. Optional kinematic joints are overwritten by name at
    every physics substep and are not part of the PD command.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int,
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        physics_timestep: float,
        joint_effort_limit: np.ndarray,
        floor_friction: tuple[float, float, float] | None = None,
        joint_armature: np.ndarray | None = None,
        joint_frictionloss: np.ndarray | None = None,
        kinematic_joint_names: tuple[str, ...] = (),
        guidance_velocity_stencil: str = "centered",
        scenario_count: int = 1,
        device: str = "cuda:0",
        nconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        mj, mjw, warp = _runtime()
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MJCF does not exist: {path}")
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(scenario_count, bool)
            or not isinstance(scenario_count, int)
            or scenario_count < 1
        ):
            raise ValueError("scenario_count must be a positive integer")
        if int(batch_size) % scenario_count:
            raise ValueError("batch_size must be divisible by scenario_count")
        timestep = float(physics_timestep)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("physics_timestep must be finite and positive")
        if guidance_velocity_stencil not in {"centered", "backward"}:
            raise ValueError(
                "guidance_velocity_stencil must be 'centered' or 'backward'"
            )

        joints = _names(joint_names, "joint_names")
        bodies = _names(body_names, "body_names")
        armature = self._joint_parameter(joint_armature, len(joints), "joint_armature")
        frictionloss = self._joint_parameter(
            joint_frictionloss, len(joints), "joint_frictionloss"
        )
        effort_limit = self._joint_parameter(
            joint_effort_limit,
            len(joints),
            "joint_effort_limit",
            strictly_positive=True,
        )
        assert effort_limit is not None
        kinematic = (
            ()
            if not kinematic_joint_names
            else _names(kinematic_joint_names, "kinematic_joint_names")
        )
        overlap = sorted(set(joints).intersection(kinematic))
        if overlap:
            raise ValueError(
                "kinematic_joint_names cannot overlap controlled joints: "
                f"{overlap}"
            )
        floor = None
        if floor_friction is not None:
            floor = np.asarray(floor_friction, dtype=np.float64)
            if (
                floor.shape != (3,)
                or not np.isfinite(floor).all()
                or (floor < 0.0).any()
            ):
                raise ValueError(
                    "floor_friction must contain three finite non-negative values"
                )
        spec = prepare_mjcf_spec(
            mj,
            path,
            joint_names=joints,
            joint_armature=armature,
            joint_frictionloss=frictionloss,
            floor_friction=floor,
        )
        model = spec.compile()
        if model.nu != 0:
            raise ValueError(
                "MjwarpWorld requires the locked actuatorless model (nu == 0)"
            )
        model.opt.timestep = timestep
        joint_ids = self._object_ids(model, mj.mjtObj.mjOBJ_JOINT, joints, "joint")
        body_ids = self._object_ids(model, mj.mjtObj.mjOBJ_BODY, bodies, "body")

        supported = {int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)}
        unsupported = [
            joints[index]
            for index, joint_id in enumerate(joint_ids)
            if int(model.jnt_type[joint_id]) not in supported
        ]
        if unsupported:
            raise ValueError(
                "controlled joints must be scalar hinge/slide joints: "
                f"{unsupported}"
            )
        qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
        dof_addresses = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
        if len(set(qpos_addresses.tolist())) != len(joints) or len(
            set(dof_addresses.tolist())
        ) != len(joints):
            raise ValueError("controlled joint qpos/dof mappings must be unique")
        kinematic_qpos_addresses = np.zeros(0, dtype=np.int32)
        kinematic_dof_addresses = np.zeros(0, dtype=np.int32)
        if kinematic:
            kinematic_ids = self._object_ids(
                model, mj.mjtObj.mjOBJ_JOINT, kinematic, "kinematic joint"
            )
            unsupported_kinematic = [
                kinematic[index]
                for index, joint_id in enumerate(kinematic_ids)
                if int(model.jnt_type[joint_id]) not in supported
            ]
            if unsupported_kinematic:
                raise ValueError(
                    "kinematic joints must be scalar hinge/slide joints: "
                    f"{unsupported_kinematic}"
                )
            kinematic_qpos_addresses = np.asarray(
                model.jnt_qposadr[kinematic_ids], dtype=np.int32
            )
            kinematic_dof_addresses = np.asarray(
                model.jnt_dofadr[kinematic_ids], dtype=np.int32
            )

        free_joint_ids = np.flatnonzero(
            np.asarray(model.jnt_type) == int(mj.mjtJoint.mjJNT_FREE)
        )
        if free_joint_ids.size != 1:
            raise ValueError("kinematic projection requires exactly one free root joint")

        warp.init()
        resolved_device = warp.get_device(device)
        if not resolved_device.is_cuda:
            raise ValueError(f"MjwarpWorld requires a CUDA device, got {device!r}")

        self._model = model
        self._batch_size = int(batch_size)
        self._scenario_count = scenario_count
        self._guidance_velocity_stencil = guidance_velocity_stencil
        self._joint_names = joints
        self._body_names = bodies
        self._kinematic_joint_names = kinematic
        self._qpos_addresses_np = qpos_addresses
        self._dof_addresses_np = dof_addresses
        self._kinematic_qpos_addresses_np = kinematic_qpos_addresses
        self._kinematic_dof_addresses_np = kinematic_dof_addresses
        self._body_ids_np = np.asarray(body_ids, dtype=np.int32)
        self._collidable_geom_ids_np = np.flatnonzero(
            (np.asarray(model.geom_contype) != 0)
            | (np.asarray(model.geom_conaffinity) != 0)
        ).astype(np.int32)
        self._nominal_geom_friction = np.asarray(
            model.geom_friction, dtype=np.float32
        ).copy()
        self._nominal_dof_armature = np.asarray(
            model.dof_armature, dtype=np.float32
        ).copy()
        self._nominal_dof_frictionloss = np.asarray(
            model.dof_frictionloss, dtype=np.float32
        ).copy()
        root = int(free_joint_ids[0])
        self._root_qpos_address = int(model.jnt_qposadr[root])
        self._root_dof_address = int(model.jnt_dofadr[root])
        root_body_id = int(model.jnt_bodyid[root])
        root_body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, root_body_id)
        if root_body_name not in bodies:
            raise ValueError(
                f"body_names must include the free-root body {root_body_name!r}"
            )
        self._root_body_index = bodies.index(root_body_name)
        self._device = resolved_device
        self._command_ready = False
        self._kinematic_ready = not bool(kinematic)
        self._kinematic_position = np.zeros((0, len(kinematic)), dtype=np.float64)
        self._kinematic_velocity = np.zeros((0, len(kinematic)), dtype=np.float64)
        self._step_graphs: dict[int, Any] = {}
        self._projection_data: dict[int, Any] = {}
        self._const_data: Any | None = None
        self._armature_scale = np.ones(scenario_count, dtype=np.float32)

        initial = mj.MjData(model)
        mj.mj_forward(model, initial)
        with warp.ScopedDevice(resolved_device):
            self._model_wp = mjw.put_model(model)
            self._expand_model_scenarios(self._model_wp, scenario_count)
            self._data = mjw.put_data(
                model,
                initial,
                nworld=self._batch_size,
                nconmax=nconmax,
                njmax=njmax,
            )
            self._qpos_addresses = warp.array(
                qpos_addresses, dtype=warp.int32, device=resolved_device
            )
            self._dof_addresses = warp.array(
                dof_addresses, dtype=warp.int32, device=resolved_device
            )
            self._body_ids = warp.array(
                self._body_ids_np, dtype=warp.int32, device=resolved_device
            )
            self._effort_limit = warp.array(
                effort_limit, dtype=warp.float32, device=resolved_device
            )
            self._actuation_strength = warp.ones(
                scenario_count, dtype=warp.float32, device=resolved_device
            )
            shape = (self._batch_size, len(joints))
            self._joint_position = warp.empty(shape, dtype=warp.float32)
            self._joint_velocity = warp.empty(shape, dtype=warp.float32)
            self._position_target = warp.empty(shape, dtype=warp.float32)
            self._velocity_target = warp.empty(shape, dtype=warp.float32)
            self._feedforward_torque = warp.empty(shape, dtype=warp.float32)
            self._proportional_gain = warp.empty(shape, dtype=warp.float32)
            self._derivative_gain = warp.empty(shape, dtype=warp.float32)
            kinematic_shape = (self._batch_size, max(len(kinematic), 1))
            self._kinematic_qpos_addresses = warp.array(
                kinematic_qpos_addresses
                if kinematic
                else np.zeros(1, dtype=np.int32),
                dtype=warp.int32,
                device=resolved_device,
            )
            self._kinematic_dof_addresses = warp.array(
                kinematic_dof_addresses
                if kinematic
                else np.zeros(1, dtype=np.int32),
                dtype=warp.int32,
                device=resolved_device,
            )
            self._kinematic_qpos = warp.empty(kinematic_shape, dtype=warp.float32)
            self._kinematic_qvel = warp.empty(kinematic_shape, dtype=warp.float32)
            body_shape = (self._batch_size, len(bodies))
            self._body_position = warp.empty(body_shape, dtype=warp.vec3)
            self._body_orientation = warp.empty(body_shape, dtype=warp.quat)
            self._body_linear_velocity = warp.empty(body_shape, dtype=warp.vec3)
            self._body_angular_velocity = warp.empty(body_shape, dtype=warp.vec3)
            self._refresh_state_view()

        self._state_view = MjwarpRobotStateView(
            joint_names=joints,
            body_names=bodies,
            joint_position=self._joint_position,
            joint_velocity=self._joint_velocity,
            body_position_world=self._body_position,
            body_orientation_world=self._body_orientation,
            body_linear_velocity_world=self._body_linear_velocity,
            body_angular_velocity_world=self._body_angular_velocity,
        )
        self._command_view = DevicePDCommand(
            joint_names=joints,
            position_target=self._position_target,
            velocity_target=self._velocity_target,
            feedforward_torque=self._feedforward_torque,
            proportional_gain=self._proportional_gain,
            derivative_gain=self._derivative_gain,
        )

    @staticmethod
    def _object_ids(
        model: Any,
        object_type: Any,
        names: tuple[str, ...],
        label: str,
    ) -> np.ndarray:
        ids = np.asarray(
            [mujoco.mj_name2id(model, object_type, name) for name in names],
            dtype=np.int32,
        )
        missing = [
            name
            for name, object_id in zip(names, ids, strict=True)
            if object_id < 0
        ]
        if missing:
            raise ValueError(f"MJCF is missing {label}s: {missing}")
        return ids

    @staticmethod
    def _joint_parameter(
        value: np.ndarray | None,
        count: int,
        field: str,
        *,
        strictly_positive: bool = False,
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (count,):
            raise ValueError(f"{field} must have shape ({count},), got {array.shape}")
        lower_invalid = array <= 0.0 if strictly_positive else array < 0.0
        if not np.isfinite(array).all() or lower_invalid.any():
            condition = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{field} must be finite and {condition}")
        return array

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def scenario_count(self) -> int:
        return self._scenario_count

    @property
    def physics_timestep(self) -> float:
        return float(self._model.opt.timestep)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def body_names(self) -> tuple[str, ...]:
        return self._body_names

    @property
    def kinematic_joint_names(self) -> tuple[str, ...]:
        return self._kinematic_joint_names

    @property
    def command_qpos_addresses(self) -> np.ndarray:
        return np.array(self._qpos_addresses_np, dtype=np.int32, copy=True)

    @property
    def command_dof_addresses(self) -> np.ndarray:
        return np.array(self._dof_addresses_np, dtype=np.int32, copy=True)

    @property
    def kinematic_qpos_addresses(self) -> np.ndarray:
        return np.array(self._kinematic_qpos_addresses_np, dtype=np.int32, copy=True)

    @property
    def kinematic_dof_addresses(self) -> np.ndarray:
        return np.array(self._kinematic_dof_addresses_np, dtype=np.int32, copy=True)

    @property
    def root_qpos_address(self) -> int:
        return int(self._root_qpos_address)

    @property
    def root_dof_address(self) -> int:
        return int(self._root_dof_address)

    @property
    def state_view(self) -> MjwarpRobotStateView:
        return self._state_view

    @property
    def command_view(self) -> DevicePDCommand:
        """Borrow world-owned device buffers that remain valid for its lifetime."""
        return self._command_view

    def configure_scenarios(self, scenarios: ScenarioBatch) -> None:
        """Install one fixed-size physics ensemble without replacing device buffers."""
        if not isinstance(scenarios, ScenarioBatch):
            raise TypeError("scenarios must be a ScenarioBatch")
        if scenarios.count != self._scenario_count:
            raise ValueError(
                "scenario count must match the world's fixed scenario_count"
            )
        if scenarios.num_joints != len(self._joint_names):
            raise ValueError("scenario joint count must match world.joint_names")
        if self._model.npair and not np.array_equal(
            scenarios.contact_friction_scale, np.ones(scenarios.count)
        ):
            raise ValueError(
                "contact friction randomization does not support explicit pairs"
            )

        friction = _scaled_scenario_rows(
            self._nominal_geom_friction,
            scenarios.contact_friction_scale,
            self._collidable_geom_ids_np,
        )
        armature = _scaled_scenario_rows(
            self._nominal_dof_armature,
            scenarios.joint_armature_scale,
            self._dof_addresses_np,
        )
        frictionloss = _scaled_scenario_rows(
            self._nominal_dof_frictionloss,
            scenarios.joint_frictionloss_scale,
            self._dof_addresses_np,
        )
        actuation_strength = np.asarray(
            scenarios.actuation_strength_scale, dtype=np.float32
        )

        with wp.ScopedDevice(self._device):
            self._copy_device_array(self._model_wp.geom_friction, friction, wp.vec3)
            self._copy_device_array(self._model_wp.dof_armature, armature, wp.float32)
            self._copy_device_array(
                self._model_wp.dof_frictionloss, frictionloss, wp.float32
            )
            self._copy_device_array(
                self._actuation_strength, actuation_strength, wp.float32
            )
            if not np.array_equal(
                self._armature_scale,
                np.asarray(scenarios.joint_armature_scale, dtype=np.float32),
            ):
                if self._const_data is None:
                    self._const_data = mujoco_warp.make_data(
                        self._model,
                        nworld=self._scenario_count,
                        nconmax=0,
                        nccdmax=0,
                        njmax=0,
                    )
                mujoco_warp.set_const_0(self._model_wp, self._const_data)
        self._armature_scale = np.asarray(
            scenarios.joint_armature_scale, dtype=np.float32
        ).copy()

    def _expand_model_scenarios(self, model: Any, count: int) -> None:
        if count == 1:
            return
        for field in fields(model):
            value = getattr(model, field.name)
            if not isinstance(value, wp.array) or not getattr(
                value, "_is_batched", False
            ):
                continue
            repeated = np.repeat(np.asarray(value.numpy()), count, axis=0)
            setattr(
                model,
                field.name,
                wp.array(repeated, dtype=value.dtype, device=self._device),
            )
        meaninertia = model.stat.meaninertia
        model.stat.meaninertia = wp.array(
            np.repeat(np.asarray(meaninertia.numpy()), count, axis=0),
            dtype=meaninertia.dtype,
            device=self._device,
        )

    def _copy_device_array(
        self, destination: Any, values: np.ndarray, dtype: Any
    ) -> None:
        wp.copy(
            destination,
            wp.array(values, dtype=dtype, device=self._device),
        )

    def reset(self, execution_state: ExecutionState) -> None:
        """Restore the physical portion of a ``ExecutionState`` exactly."""
        qpos = self._physical(
            execution_state.qpos, (self._batch_size, self._model.nq), "qpos"
        )
        qvel = self._physical(
            execution_state.qvel, (self._batch_size, self._model.nv), "qvel"
        )
        times = np.full(
            self._batch_size, float(execution_state.time), dtype=np.float32
        )
        if not np.isfinite(times).all() or (times < 0.0).any():
            raise ValueError("execution_state time must be finite and non-negative")
        with wp.ScopedDevice(self._device):
            mujoco_warp.reset_data(self._model_wp, self._data)
            wp.copy(
                self._data.qpos,
                wp.array(qpos, dtype=wp.float32, device=self._device),
            )
            wp.copy(
                self._data.qvel,
                wp.array(qvel, dtype=wp.float32, device=self._device),
            )
            wp.copy(
                self._data.time,
                wp.array(times, dtype=wp.float32, device=self._device),
            )
            mujoco_warp.forward(self._model_wp, self._data)
            self._refresh_state_view()
        self._command_ready = False
        self._kinematic_ready = not bool(self._kinematic_joint_names)

    def apply_pd(self, command: PDCommand | DevicePDCommand) -> None:
        """Bind one canonical command; ``step`` reevaluates its PD law."""
        if command.joint_names != self._joint_names:
            raise ValueError("PDCommand joint_names must match MjwarpWorld joint_names")
        shape = (self._batch_size, len(self._joint_names))
        fields = (
            ("position_target", self._position_target),
            ("velocity_target", self._velocity_target),
            ("feedforward_torque", self._feedforward_torque),
            ("proportional_gain", self._proportional_gain),
            ("derivative_gain", self._derivative_gain),
        )
        with wp.ScopedDevice(self._device):
            for name, destination in fields:
                source = getattr(command, name)
                if isinstance(command, DevicePDCommand):
                    self._validate_device_command_array(source, shape, name)
                    if source.ptr != destination.ptr:
                        wp.copy(destination, source)
                else:
                    values = self._physical(source, shape, name)
                    wp.copy(
                        destination,
                        wp.array(values, dtype=wp.float32, device=self._device),
                    )
        self._command_ready = True

    def bind_kinematic_trajectory(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> None:
        """Store one open-loop kinematic trajectory shared by every batch lane."""

        if not self._kinematic_joint_names:
            raise ValueError("world has no kinematic joints to bind")
        count = len(self._kinematic_joint_names)
        pos = np.asarray(position, dtype=np.float64)
        vel = np.asarray(velocity, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != count or pos.shape[0] < 1:
            raise ValueError(
                f"kinematic position must have shape (T, {count}), got {pos.shape}"
            )
        if vel.shape != pos.shape:
            raise ValueError("kinematic velocity must match position")
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise ValueError("kinematic trajectory must be finite")
        self._kinematic_position = np.ascontiguousarray(pos)
        self._kinematic_velocity = np.ascontiguousarray(vel)

    def apply_kinematic_frame(self, frame_index: int) -> None:
        """Copy one clamped trajectory frame onto every world in the batch."""

        if not self._kinematic_joint_names:
            return
        if self._kinematic_position.shape[0] < 1:
            raise RuntimeError("bind_kinematic_trajectory must be called first")
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
            raise ValueError("frame_index must be an integer")
        index = int(np.clip(int(frame_index), 0, self._kinematic_position.shape[0] - 1))
        count = len(self._kinematic_joint_names)
        position = np.broadcast_to(
            self._kinematic_position[index], (self._batch_size, count)
        )
        velocity = np.broadcast_to(
            self._kinematic_velocity[index], (self._batch_size, count)
        )
        with wp.ScopedDevice(self._device):
            wp.copy(
                self._kinematic_qpos,
                wp.array(position, dtype=wp.float32, device=self._device),
            )
            wp.copy(
                self._kinematic_qvel,
                wp.array(velocity, dtype=wp.float32, device=self._device),
            )
        self._kinematic_ready = True

    def overlay_kinematic_trajectory(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        start_frame: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Write bound kinematic joints into a dense physical trajectory."""

        if not self._kinematic_joint_names:
            return qpos, qvel
        if self._kinematic_position.shape[0] < 1:
            raise RuntimeError("bind_kinematic_trajectory must be called first")
        owned_qpos = np.array(qpos, dtype=np.float64, copy=True, order="C")
        owned_qvel = np.array(qvel, dtype=np.float64, copy=True, order="C")
        expected = (owned_qpos.shape[0], owned_qpos.shape[1], self._model.nq)
        owned_qpos = self._physical(owned_qpos, expected, "qpos", dtype=np.float64)
        owned_qvel = self._physical(
            owned_qvel,
            (owned_qpos.shape[0], owned_qpos.shape[1], self._model.nv),
            "qvel",
            dtype=np.float64,
        )
        if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
            raise ValueError("start_frame must be a non-negative integer")
        last = self._kinematic_position.shape[0] - 1
        frames = np.clip(
            start_frame + np.arange(owned_qpos.shape[1]), 0, last
        )
        owned_qpos[:, :, self._kinematic_qpos_addresses_np] = self._kinematic_position[
            frames
        ][None]
        owned_qvel[:, :, self._kinematic_dof_addresses_np] = self._kinematic_velocity[
            frames
        ][None]
        return owned_qpos, owned_qvel

    def host_command(self) -> PDCommand:
        """Materialize the current world-owned command at a execution boundary."""
        if not self._command_ready:
            raise RuntimeError("no PD command has been applied")
        wp.synchronize_device(self._device)
        arrays = {
            name: np.asarray(getattr(self._command_view, name).numpy()).copy()
            for name in (
                "position_target",
                "velocity_target",
                "feedforward_torque",
                "proportional_gain",
                "derivative_gain",
            )
        }
        return PDCommand(joint_names=self._joint_names, **arrays)

    def step(self, num_substeps: int = 1) -> None:
        """Advance physics, recomputing PD force before every substep."""
        if not self._command_ready:
            raise RuntimeError("apply_pd must be called before step")
        if self._kinematic_joint_names and not self._kinematic_ready:
            raise RuntimeError("apply_kinematic_frame must be called before step")
        if int(num_substeps) != num_substeps or num_substeps < 1:
            raise ValueError("num_substeps must be a positive integer")
        n = int(num_substeps)
        with wp.ScopedDevice(self._device):
            graph = self._step_graphs.get(n)
            if graph is None:
                # Eager warm-up loads MJWarp modules; then capture once —
                # hundreds of small kernels per substep make launch overhead dominate.
                self._run_substeps(n)
                with wp.ScopedCapture() as capture:
                    self._run_substeps(n)
                self._step_graphs[n] = capture.graph
            else:
                wp.capture_launch(graph)
            self._refresh_state_view()

    def _run_substeps(self, num_substeps: int) -> None:
        for _ in range(num_substeps):
            self._data.qfrc_applied.zero_()
            wp.launch(
                _apply_pd_force,
                dim=(self._batch_size, len(self._joint_names)),
                inputs=[
                    self._data.qpos,
                    self._data.qvel,
                    self._position_target,
                    self._velocity_target,
                    self._feedforward_torque,
                    self._proportional_gain,
                    self._derivative_gain,
                    self._effort_limit,
                    self._actuation_strength,
                    self._qpos_addresses,
                    self._dof_addresses,
                    self._data.qfrc_applied,
                ],
                device=self._device,
            )
            if self._kinematic_joint_names:
                self._scatter_kinematic_state()
            mujoco_warp.step(self._model_wp, self._data)
        if self._kinematic_joint_names:
            self._scatter_kinematic_state()

    def _scatter_kinematic_state(self) -> None:
        wp.launch(
            _scatter_kinematic,
            dim=(self._batch_size, len(self._kinematic_joint_names)),
            inputs=[
                self._kinematic_qpos,
                self._kinematic_qvel,
                self._kinematic_qpos_addresses,
                self._kinematic_dof_addresses,
                self._data.qpos,
                self._data.qvel,
            ],
            device=self._device,
        )

    def robot_state(self) -> RobotState:
        """Copy the current declared robot state to host NumPy arrays."""
        view = self._state_view

        def copy(value: Any) -> np.ndarray:
            return np.asarray(value.numpy()).copy()

        return RobotState(
            joint_names=view.joint_names,
            body_names=view.body_names,
            joint_position=copy(view.joint_position),
            joint_velocity=copy(view.joint_velocity),
            body_position_world=copy(view.body_position_world),
            body_orientation_world=copy(view.body_orientation_world),
            body_linear_velocity_world=copy(view.body_linear_velocity_world),
            body_angular_velocity_world=copy(view.body_angular_velocity_world),
        )

    def physical_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return owned host copies of qpos, qvel, and per-lane simulation time."""
        wp.synchronize_device(self._device)
        return (
            np.asarray(self._data.qpos.numpy()).copy(),
            np.asarray(self._data.qvel.numpy()).copy(),
            np.asarray(self._data.time.numpy()).copy(),
        )

    def project(
        self,
        *,
        query_offsets: tuple[int, ...],
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        root_position_world: np.ndarray,
        root_orientation_world: np.ndarray,
        joint_position: np.ndarray,
        control_period: float,
    ) -> MotionGuidance:
        """Project dense root/joint qpos with the exact model used by simulation."""
        if joint_names != self._joint_names or body_names != self._body_names:
            raise ValueError("projection joint/body names must match the world")
        joint_array = np.asarray(joint_position)
        if joint_array.ndim != 3:
            raise ValueError("joint_position must have shape (batch, horizon, joints)")
        batch, horizon = joint_array.shape[:2]
        if batch < 1 or horizon < 1:
            raise ValueError("projection batch and horizon must be positive")
        if query_offsets != tuple(range(horizon)):
            raise ValueError("projection query_offsets must be a dense zero-based grid")
        root_position = self._physical(
            root_position_world,
            (batch, horizon, 3),
            "root_position_world",
            dtype=np.float64,
        )
        root_orientation = self._physical(
            root_orientation_world,
            (batch, horizon, 4),
            "root_orientation_world",
            dtype=np.float64,
        )
        joints = self._physical(
            joint_array,
            (batch, horizon, len(self._joint_names)),
            "joint_position",
            dtype=np.float64,
        )
        if not np.allclose(
            np.linalg.norm(root_orientation, axis=-1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("root_orientation_world must contain unit quaternions")
        period = float(control_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("control_period must be finite and positive")

        qpos = self._assemble_qpos(root_position, root_orientation, joints)
        qvel = self._differentiate(qpos, period)

        frames = batch * horizon
        with wp.ScopedDevice(self._device):
            data = self._projection_data.get(frames)
            if data is None:
                data = mujoco_warp.make_data(
                    self._model,
                    nworld=frames,
                    nconmax=0,
                    nccdmax=0,
                    njmax=0,
                )
                self._projection_data[frames] = data
            wp.copy(
                data.qpos,
                wp.array(
                    qpos.reshape(frames, -1),
                    dtype=wp.float32,
                    device=self._device,
                ),
            )
            wp.copy(
                data.qvel,
                wp.array(
                    qvel.reshape(frames, -1),
                    dtype=wp.float32,
                    device=self._device,
                ),
            )
            mujoco_warp.kinematics(self._model_wp, data)
            mujoco_warp.com_pos(self._model_wp, data)
            mujoco_warp.com_vel(self._model_wp, data)
        wp.synchronize_device(self._device)
        body_position = np.asarray(data.xpos.numpy())[:, self._body_ids_np]
        body_orientation = np.asarray(data.xquat.numpy())[:, self._body_ids_np]
        cvel = np.asarray(data.cvel.numpy())[:, self._body_ids_np]
        shape3 = (batch, horizon, len(self._body_names), 3)
        shape4 = (batch, horizon, len(self._body_names), 4)
        return MotionGuidance(
            query_offsets=query_offsets,
            joint_names=joint_names,
            body_names=body_names,
            joint_position=joints.astype(np.float64, copy=True),
            joint_velocity=qvel[:, :, self._dof_addresses_np].astype(np.float64),
            body_position_world=body_position.reshape(shape3).astype(np.float64),
            body_orientation_world=body_orientation.reshape(shape4).astype(np.float64),
            body_linear_velocity_world=cvel[:, :, 3:]
            .reshape(shape3)
            .astype(np.float64),
            body_angular_velocity_world=cvel[:, :, :3]
            .reshape(shape3)
            .astype(np.float64),
        )

    def physical_trajectory(
        self, guidance: MotionGuidance, control_period: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct dense qpos/qvel for initialization or persisted replay."""
        if (
            guidance.joint_names != self._joint_names
            or guidance.body_names != self._body_names
        ):
            raise ValueError("guidance joint/body names must match the world")
        horizon = len(guidance.query_offsets)
        if guidance.query_offsets != tuple(range(horizon)):
            raise ValueError("guidance must use a dense zero-based query grid")
        batch = guidance.joint_position.shape[0]
        period = float(control_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("control_period must be finite and positive")
        root_position = self._physical(
            guidance.body_position_world[:, :, self._root_body_index],
            (batch, horizon, 3),
            "root_position_world",
            dtype=np.float64,
        )
        root_orientation = self._physical(
            guidance.body_orientation_world[:, :, self._root_body_index],
            (batch, horizon, 4),
            "root_orientation_world",
            dtype=np.float64,
        )
        joints = self._physical(
            guidance.joint_position,
            (batch, horizon, len(self._joint_names)),
            "joint_position",
            dtype=np.float64,
        )
        qpos = self._assemble_qpos(root_position, root_orientation, joints)
        return qpos, self._differentiate(qpos, period)

    def _assemble_qpos(
        self,
        root_position: np.ndarray,
        root_orientation: np.ndarray,
        joints: np.ndarray,
    ) -> np.ndarray:
        batch, horizon = joints.shape[:2]
        qpos = np.broadcast_to(
            np.asarray(self._model.qpos0, dtype=np.float64),
            (batch, horizon, self._model.nq),
        ).copy()
        root = self._root_qpos_address
        qpos[:, :, root : root + 3] = root_position
        qpos[:, :, root + 3 : root + 7] = root_orientation
        qpos[:, :, self._qpos_addresses_np] = joints
        return qpos

    def _differentiate(self, qpos: np.ndarray, period: float) -> np.ndarray:
        batch, horizon, _ = qpos.shape
        velocity = np.zeros((batch, horizon, self._model.nv), dtype=np.float64)
        if horizon == 1:
            return velocity
        steps = np.arange(horizon)
        if self._guidance_velocity_stencil == "centered":
            left = np.maximum(steps - 1, 0)
            right = np.minimum(steps + 1, horizon - 1)
        else:
            left = np.maximum(steps - 1, 0)
            right = steps.copy()
            right[0] = 1
        dt = (right - left).astype(np.float64) * period
        before = qpos[:, left]
        after = qpos[:, right]
        root_qpos = self._root_qpos_address
        root_dof = self._root_dof_address
        velocity[:, :, root_dof : root_dof + 3] = (
            after[:, :, root_qpos : root_qpos + 3]
            - before[:, :, root_qpos : root_qpos + 3]
        ) / dt[None, :, None]
        before_orientation = before[:, :, root_qpos + 3 : root_qpos + 7]
        angular_velocity = _quat_velocity(
            before_orientation,
            after[:, :, root_qpos + 3 : root_qpos + 7],
            dt,
        )
        if self._guidance_velocity_stencil == "backward":
            angular_velocity = _quat_rotate(before_orientation, angular_velocity)
        velocity[:, :, root_dof + 3 : root_dof + 6] = angular_velocity
        velocity[:, :, self._dof_addresses_np] = (
            after[:, :, self._qpos_addresses_np]
            - before[:, :, self._qpos_addresses_np]
        ) / dt[None, :, None]
        return velocity

    def _refresh_state_view(self) -> None:
        wp.launch(
            _gather_joints,
            dim=(self._batch_size, len(self._joint_names)),
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._qpos_addresses,
                self._dof_addresses,
                self._joint_position,
                self._joint_velocity,
            ],
            device=self._device,
        )
        wp.launch(
            _gather_bodies,
            dim=(self._batch_size, len(self._body_names)),
            inputs=[
                self._data.xpos,
                self._data.xquat,
                self._data.cvel,
                self._body_ids,
                self._body_position,
                self._body_orientation,
                self._body_linear_velocity,
                self._body_angular_velocity,
            ],
            device=self._device,
        )

    @staticmethod
    def _physical(
        value: np.ndarray,
        shape: tuple[int, ...],
        field: str,
        *,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            raise ValueError(f"{field} must have shape {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{field} must be finite")
        return np.ascontiguousarray(array)

    def _validate_device_command_array(
        self, value: Any, shape: tuple[int, ...], field: str
    ) -> None:
        if (
            getattr(value, "shape", None) != shape
            or getattr(value, "dtype", None) != wp.float32
            or getattr(value, "device", None) != self._device
            or not bool(getattr(value, "is_contiguous", False))
        ):
            raise ValueError(
                f"{field} must be a contiguous float32 Warp array with shape "
                f"{shape} on {self._device}"
            )
