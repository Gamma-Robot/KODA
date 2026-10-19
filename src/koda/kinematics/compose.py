#!/usr/bin/env python3
"""Compose 29-DoF MPC body states and the render model's hand states offline.

The hand archive supplies one value per independently actuated joint (14 for
Dex3, 12 for the BrainCo hands).  Joints that the render XML couples through a
``<equality><joint>`` mimic rule are evaluated here, because MuJoCo resolves
those constraints in the dynamics solver rather than in ``mj_forward``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from koda.kinematics.dex3 import (
    COMPILER_BOUNDINERTIA,
    COMPILER_BOUNDMASS,
    DEFAULT_BODY_XML,
    DEFAULT_DEX3_XML,
    DEFAULT_G1_MESHDIR,
    RECORDING_MODEL_CONTRACTS,
    d435_calibration,
    evaluate_mimic_rule,
    hinge_joint_names,
    joint_mimic_rules,
    load_dex3_model,
    nonworld_body_names,
    require_object_id,
)


ROOT_BODY_NAME = "pelvis"
EXPECTED_BODY_JOINTS = 29
EXPECTED_HAND_JOINTS = 14
EXPECTED_FPS = 50.0


def _string_list(values: np.ndarray | Iterable[Any]) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result


def _require_unique(names: list[str], label: str) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label} contains duplicate names: {duplicates}")


def _require_array(
    arrays: dict[str, np.ndarray],
    keys: tuple[str, ...],
    label: str,
) -> np.ndarray:
    for key in keys:
        if key in arrays:
            return np.asarray(arrays[key])
    raise ValueError(f"{label} is missing; expected one of {keys}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _load_adjacent_meta(path: Path) -> dict[str, Any]:
    metadata_path = path.parent / "meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"body kinematics metadata is required for name mapping: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata root must be an object: {metadata_path}")
    return metadata


def _scalar_fps(arrays: dict[str, np.ndarray], label: str) -> float:
    if "fps" not in arrays:
        raise ValueError(f"{label} is missing scalar fps")
    fps_array = np.asarray(arrays["fps"])
    if fps_array.size != 1:
        raise ValueError(f"{label} fps must be scalar, got {fps_array.shape}")
    fps = float(fps_array.reshape(()))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{label} fps must be finite and positive, got {fps}")
    return fps


def _timestamps(
    arrays: dict[str, np.ndarray],
    frame_count: int,
    fps: float,
    label: str,
) -> np.ndarray:
    if "timestamps" in arrays:
        timestamps = np.asarray(arrays["timestamps"], dtype=np.float64)
        if timestamps.shape != (frame_count,):
            raise ValueError(
                f"{label} timestamps shape is {timestamps.shape}, expected {(frame_count,)}"
            )
        timestamps = timestamps - timestamps[0]
    else:
        timestamps = np.arange(frame_count, dtype=np.float64) / fps
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{label} timestamps must be finite and strictly increasing")
    return timestamps


def _validate_numeric(name: str, values: np.ndarray, shape: tuple[int, ...]) -> None:
    if values.shape != shape:
        raise ValueError(f"{name} shape is {values.shape}, expected {shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")


def _ids_and_addresses(
    model: mujoco.MjModel,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [require_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
        dtype=np.int32,
    )
    invalid = [
        name
        for name, joint_id in zip(names, joint_ids)
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    if invalid:
        raise ValueError(f"expected 1-DoF hinge joints: {invalid}")
    return (
        joint_ids,
        np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
        np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
    )


def _free_joint_addresses(model: mujoco.MjModel, label: str) -> tuple[int, int]:
    free_joint_ids = np.flatnonzero(
        np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if free_joint_ids.shape != (1,):
        raise ValueError(f"{label} must have exactly one free joint")
    joint_id = int(free_joint_ids[0])
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _reconstruct_source_base_qvel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_id: int,
    root_linear_velocity_w: np.ndarray,
    root_angular_velocity_w: np.ndarray,
    free_dof_address: int,
) -> np.ndarray:
    """Recover free-joint qvel from the recorded pelvis COM 6D velocity."""

    jacobian_pos = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rot = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(
        model,
        data,
        jacobian_pos,
        jacobian_rot,
        root_body_id,
    )
    jacobian = np.concatenate((jacobian_pos, jacobian_rot), axis=0)
    base_slice = slice(free_dof_address, free_dof_address + 6)
    target_velocity = np.concatenate(
        (root_linear_velocity_w, root_angular_velocity_w), axis=0
    )
    # qvel already contains every known hinge velocity and a zero base.
    residual = target_velocity - jacobian @ data.qvel
    base_jacobian = jacobian[:, base_slice]
    try:
        base_qvel = np.linalg.solve(base_jacobian, residual)
    except np.linalg.LinAlgError as error:
        raise ValueError("pelvis free-joint Jacobian is singular") from error
    if not np.all(np.isfinite(base_qvel)):
        raise ValueError("reconstructed free-joint qvel is non-finite")
    return base_qvel


def _rotate_by_quaternion(quaternion_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    scalar = quaternion_wxyz[..., 0:1]
    vector = quaternion_wxyz[..., 1:4]
    intermediate = 2.0 * np.cross(vector, vectors)
    return vectors + scalar * intermediate + np.cross(vector, intermediate)


def _quaternion_angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dots = np.abs(np.sum(a_norm * b_norm, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def compose(args: argparse.Namespace) -> dict[str, Any]:
    body_path = args.body_kinematics.expanduser().resolve()
    hand_path = args.hand_kinematics.expanduser().resolve()
    body_model_path = args.body_model.expanduser().resolve()
    dex3_model_path = args.dex3_model.expanduser().resolve()
    meshdir = args.meshdir.expanduser().resolve()
    # The body model and the render model resolve their mesh paths differently
    # when they come from different vendors, so keep the two roots separable.
    target_meshdir = (
        args.dex3_meshdir.expanduser().resolve()
        if args.dex3_meshdir is not None
        else meshdir
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.output_dir.expanduser().resolve() / "kinematics.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    body = _load_npz(body_path)
    hand = _load_npz(hand_path)
    # New tracking archives are self-describing. Keep the legacy sidecar
    # requirement when names are absent, but do not reject a valid archive
    # merely because its published final directory omits meta.json.
    if (body_path.parent / "meta.json").is_file():
        body_meta = _load_adjacent_meta(body_path)
    elif "joint_names" in body and "body_names" in body:
        body_meta = {}
    else:
        body_meta = _load_adjacent_meta(body_path)
    body_joint_names = _string_list(
        body["joint_names"]
        if "joint_names" in body
        else body_meta.get("joint_names", [])
    )
    source_body_names = _string_list(
        body["body_names"] if "body_names" in body else body_meta.get("body_names", [])
    )
    hand_joint_names = _string_list(
        _require_array(hand, ("joint_names", "hand_joint_names"), "hand joint names")
    )
    _require_unique(body_joint_names, "body joint names")
    _require_unique(source_body_names, "body names")
    _require_unique(hand_joint_names, "hand joint names")

    body_joint_pos = _require_array(body, ("joint_pos",), "body joint_pos")
    body_joint_vel = _require_array(body, ("joint_vel",), "body joint_vel")
    source_body_pos = _require_array(body, ("body_pos_w",), "body body_pos_w")
    source_body_quat = _require_array(body, ("body_quat_w",), "body body_quat_w")
    source_body_lin_vel = _require_array(
        body, ("body_lin_vel_w",), "body body_lin_vel_w"
    )
    source_body_ang_vel = _require_array(
        body, ("body_ang_vel_w",), "body body_ang_vel_w"
    )
    hand_qpos_input = _require_array(hand, ("qpos", "hand_qpos"), "hand qpos")
    hand_qvel_input = _require_array(hand, ("qvel", "hand_qvel"), "hand qvel")

    frame_count = int(body_joint_pos.shape[0]) if body_joint_pos.ndim == 2 else -1
    if frame_count <= 0:
        raise ValueError("body trajectory must contain at least one frame")
    _validate_numeric(
        "body joint_pos", body_joint_pos, (frame_count, EXPECTED_BODY_JOINTS)
    )
    _validate_numeric(
        "body joint_vel", body_joint_vel, (frame_count, EXPECTED_BODY_JOINTS)
    )
    _validate_numeric(
        "body body_pos_w", source_body_pos, (frame_count, len(source_body_names), 3)
    )
    _validate_numeric(
        "body body_quat_w", source_body_quat, (frame_count, len(source_body_names), 4)
    )
    _validate_numeric(
        "body body_lin_vel_w",
        source_body_lin_vel,
        (frame_count, len(source_body_names), 3),
    )
    _validate_numeric(
        "body body_ang_vel_w",
        source_body_ang_vel,
        (frame_count, len(source_body_names), 3),
    )
    expected_hand_joints = int(args.expected_hand_joints)
    _validate_numeric("hand qpos", hand_qpos_input, (frame_count, expected_hand_joints))
    _validate_numeric("hand qvel", hand_qvel_input, (frame_count, expected_hand_joints))
    if len(body_joint_names) != EXPECTED_BODY_JOINTS:
        raise ValueError(
            f"body metadata has {len(body_joint_names)} joints, expected "
            f"{EXPECTED_BODY_JOINTS}"
        )
    if len(hand_joint_names) != expected_hand_joints:
        raise ValueError(
            f"hand file has {len(hand_joint_names)} joint names, expected "
            f"{expected_hand_joints}"
        )
    if ROOT_BODY_NAME not in source_body_names:
        raise ValueError(f"body metadata is missing root body {ROOT_BODY_NAME!r}")
    root_source_index = source_body_names.index(ROOT_BODY_NAME)
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(source_body_quat, axis=-1) - 1.0))
    )
    if quaternion_norm_error > 1.0e-4:
        raise ValueError(
            f"source body quaternion norm error is too large: {quaternion_norm_error}"
        )

    body_fps = _scalar_fps(body, "body kinematics")
    hand_fps = _scalar_fps(hand, "hand kinematics")
    if abs(body_fps - EXPECTED_FPS) > 1.0e-9:
        raise ValueError(f"body trajectory must be 50 Hz, got {body_fps}")
    if abs(hand_fps - body_fps) > 1.0e-9:
        raise ValueError(f"body/hand fps differ: {body_fps} vs {hand_fps}")
    body_timestamps = _timestamps(body, frame_count, body_fps, "body kinematics")
    hand_timestamps = _timestamps(hand, frame_count, hand_fps, "hand kinematics")
    timestamp_error = float(np.max(np.abs(body_timestamps - hand_timestamps)))
    if timestamp_error > 1.0e-6:
        raise ValueError(
            f"body/hand timestamps are not aligned; max error is {timestamp_error:.9g} s"
        )

    # Compile both source and target XMLs through the same in-memory meshdir
    # and compiler bounds.  The visual Dex3 XML uses a relative ``assets/``
    # meshdir and cannot be loaded safely with from_xml_path from arbitrary
    # worker directories.
    source_model = load_dex3_model(
        body_model_path,
        meshdir=meshdir,
        with_d435=False,
    )
    target_model = load_dex3_model(
        dex3_model_path,
        meshdir=target_meshdir,
        with_d435=True,
        hand_mode=args.hand_mode,
    )
    source_hinges = hinge_joint_names(source_model)
    target_hinges = hinge_joint_names(target_model)
    # Joint equalities in the render XML are mimic couplings that the offline
    # kinematics pass has to expand itself; mj_forward alone leaves the driven
    # joint at its reference value.  Dex3 has none, so this stays a no-op.
    mimic_rules = joint_mimic_rules(target_model)
    mimic_by_dependent = {str(rule["dependent"]): rule for rule in mimic_rules}
    target_body_names = nonworld_body_names(target_model)
    if not target_body_names or target_body_names[0] != ROOT_BODY_NAME:
        raise ValueError(
            f"Dex3 body order must start with {ROOT_BODY_NAME!r}, got "
            f"{target_body_names[:1]}"
        )
    if not set(body_joint_names).issubset(set(source_hinges)):
        raise ValueError(
            "body joint-name set is not covered by the source model: "
            f"missing={sorted(set(source_hinges) - set(body_joint_names))}, "
            f"extra={sorted(set(body_joint_names) - set(source_hinges))}"
        )
    if set(body_joint_names).intersection(hand_joint_names):
        raise ValueError("body and hand joint-name sets overlap")
    supplied_joint_set = set(body_joint_names).union(hand_joint_names)
    derived_joint_set = set(mimic_by_dependent) - supplied_joint_set
    if supplied_joint_set.union(derived_joint_set) != set(target_hinges):
        raise ValueError(
            "body+hand+mimic joint-name sets do not match the Dex3 XML: "
            f"missing={sorted(set(target_hinges) - supplied_joint_set - derived_joint_set)}, "
            f"extra={sorted(supplied_joint_set - set(target_hinges))}"
        )
    unresolved_mimic = sorted(
        name
        for name in derived_joint_set
        if str(mimic_by_dependent[name]["reference"]) not in supplied_joint_set
    )
    if unresolved_mimic:
        raise ValueError(
            "mimic joints reference values that were not supplied: "
            f"{unresolved_mimic}"
        )
    if not set(source_body_names).issubset(target_body_names):
        raise ValueError(
            "source body names are not a subset of the Dex3 XML: "
            f"{sorted(set(source_body_names) - set(target_body_names))}"
        )

    _, source_qpos_addresses, source_qvel_addresses = _ids_and_addresses(
        source_model, body_joint_names
    )
    target_joint_ids, target_qpos_addresses, target_qvel_addresses = _ids_and_addresses(
        target_model, target_hinges
    )
    source_free_qpos, source_free_qvel = _free_joint_addresses(
        source_model, "body model"
    )
    target_free_qpos, target_free_qvel = _free_joint_addresses(
        target_model, "Dex3 model"
    )
    if (source_free_qpos, source_free_qvel, target_free_qpos, target_free_qvel) != (
        0,
        0,
        0,
        0,
    ):
        raise ValueError(
            "both models must place their free joint at qpos/qvel address 0"
        )

    body_value_index = {name: index for index, name in enumerate(body_joint_names)}
    hand_value_index = {name: index for index, name in enumerate(hand_joint_names)}
    target_body_indices = np.asarray(
        [target_body_names.index(name) for name in source_body_names], dtype=np.int32
    )
    inertial_offset_body = np.asarray(
        [
            target_model.body_ipos[
                require_object_id(target_model, mujoco.mjtObj.mjOBJ_BODY, name)
            ]
            - source_model.body_ipos[
                require_object_id(source_model, mujoco.mjtObj.mjOBJ_BODY, name)
            ]
            for name in source_body_names
        ],
        dtype=np.float64,
    )
    inertial_offset_max_m = float(np.max(np.abs(inertial_offset_body)))
    inertial_offset_bodies = [
        name
        for name, offset in zip(source_body_names, inertial_offset_body)
        if float(np.max(np.abs(offset))) > 0.0
    ]
    target_hand_names = [
        name
        for name in target_hinges
        if name in hand_value_index or name in derived_joint_set
    ]

    joint_dtype = body_joint_pos.dtype
    velocity_dtype = body_joint_vel.dtype
    joint_pos = np.empty((frame_count, len(target_hinges)), dtype=joint_dtype)
    joint_vel = np.empty((frame_count, len(target_hinges)), dtype=velocity_dtype)
    for output_index, name in enumerate(target_hinges):
        if name in body_value_index:
            input_index = body_value_index[name]
            joint_pos[:, output_index] = body_joint_pos[:, input_index]
            joint_vel[:, output_index] = body_joint_vel[:, input_index]
        elif name in hand_value_index:
            input_index = hand_value_index[name]
            joint_pos[:, output_index] = hand_qpos_input[:, input_index]
            joint_vel[:, output_index] = hand_qvel_input[:, input_index]
        else:
            rule = mimic_by_dependent[name]
            reference = str(rule["reference"])
            if reference in body_value_index:
                reference_index = body_value_index[reference]
                reference_qpos = body_joint_pos[:, reference_index]
                reference_qvel = body_joint_vel[:, reference_index]
            else:
                reference_index = hand_value_index[reference]
                reference_qpos = hand_qpos_input[:, reference_index]
                reference_qvel = hand_qvel_input[:, reference_index]
            dependent_qpos, dependent_qvel = evaluate_mimic_rule(
                rule, reference_qpos, reference_qvel
            )
            joint_pos[:, output_index] = dependent_qpos.astype(joint_dtype)
            joint_vel[:, output_index] = dependent_qvel.astype(velocity_dtype)

    # Validate the hand states against the exact per-joint ranges in the target XML.
    limits = np.asarray(target_model.jnt_range[target_joint_ids], dtype=np.float64)
    limited = np.asarray(target_model.jnt_limited[target_joint_ids], dtype=bool)

    # A mimic polynomial and its driven joint's range come from the same URDF
    # but are rounded independently, so the coupled joint can overshoot its own
    # limit by a fraction of a milliradian at full flexion.  Saturate to the
    # model range, which is what the hardware coupling does, and fail loudly if
    # the overshoot is large enough to mean the polynomial is actually wrong.
    mimic_clip_max_rad = 0.0
    if derived_joint_set:
        derived_output_indices = np.asarray(
            [target_hinges.index(name) for name in sorted(derived_joint_set)],
            dtype=np.int32,
        )
        derived_limited = limited[derived_output_indices]
        derived_limits = limits[derived_output_indices]
        values = np.asarray(joint_pos[:, derived_output_indices], dtype=np.float64)
        clipped = np.where(
            derived_limited[None, :],
            np.clip(values, derived_limits[:, 0][None, :], derived_limits[:, 1][None, :]),
            values,
        )
        mimic_clip_max_rad = float(np.max(np.abs(clipped - values)))
        if mimic_clip_max_rad > float(args.max_mimic_clip_rad):
            raise ValueError(
                "mimic-expanded joint exceeds its XML range by "
                f"{mimic_clip_max_rad:.9g} rad, above the "
                f"{float(args.max_mimic_clip_rad):.9g} rad rounding allowance"
            )
        joint_pos[:, derived_output_indices] = clipped.astype(joint_pos.dtype)

    violations = np.maximum(
        limits[:, 0][None, :] - np.asarray(joint_pos, dtype=np.float64),
        np.asarray(joint_pos, dtype=np.float64) - limits[:, 1][None, :],
    )
    hand_output_indices = np.asarray(
        [target_hinges.index(name) for name in target_hand_names], dtype=np.int32
    )
    hand_violation = float(
        np.max(
            np.where(
                limited[hand_output_indices][None, :],
                violations[:, hand_output_indices],
                -np.inf,
            )
        )
    )
    if hand_violation > 1.0e-6:
        raise ValueError(
            f"hand qpos exceeds a Dex3 XML joint range by {hand_violation:.9g} rad"
        )

    body_count = len(target_body_names)
    body_pos_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
    body_quat_w = np.empty((frame_count, body_count, 4), dtype=np.float32)
    body_lin_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
    body_ang_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)

    source_data = mujoco.MjData(source_model)
    target_data = mujoco.MjData(target_model)
    source_root_id = require_object_id(
        source_model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME
    )
    target_body_ids = np.asarray(
        [
            require_object_id(target_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in target_body_names
        ],
        dtype=np.int32,
    )
    object_velocity = np.empty(6, dtype=np.float64)

    max_base_velocity_reconstruction_error = 0.0
    for frame_index in range(frame_count):
        root_position = source_body_pos[frame_index, root_source_index]
        root_quaternion = source_body_quat[frame_index, root_source_index]

        source_data.qpos[:] = source_model.qpos0
        source_data.qvel[:] = 0.0
        source_data.qpos[source_free_qpos : source_free_qpos + 3] = root_position
        source_data.qpos[source_free_qpos + 3 : source_free_qpos + 7] = root_quaternion
        source_data.qpos[source_qpos_addresses] = body_joint_pos[frame_index]
        source_data.qvel[source_qvel_addresses] = body_joint_vel[frame_index]
        mujoco.mj_forward(source_model, source_data)
        base_qvel = _reconstruct_source_base_qvel(
            source_model,
            source_data,
            source_root_id,
            source_body_lin_vel[frame_index, root_source_index],
            source_body_ang_vel[frame_index, root_source_index],
            source_free_qvel,
        )
        source_data.qvel[source_free_qvel : source_free_qvel + 6] = base_qvel
        mujoco.mj_forward(source_model, source_data)
        mujoco.mj_objectVelocity(
            source_model,
            source_data,
            mujoco.mjtObj.mjOBJ_BODY,
            source_root_id,
            object_velocity,
            0,
        )
        reconstructed_velocity = np.concatenate(
            (object_velocity[3:], object_velocity[:3]), axis=0
        )
        requested_velocity = np.concatenate(
            (
                source_body_lin_vel[frame_index, root_source_index],
                source_body_ang_vel[frame_index, root_source_index],
            ),
            axis=0,
        )
        max_base_velocity_reconstruction_error = max(
            max_base_velocity_reconstruction_error,
            float(np.max(np.abs(reconstructed_velocity - requested_velocity))),
        )
        target_data.qpos[:] = target_model.qpos0
        target_data.qvel[:] = 0.0
        target_data.qpos[target_free_qpos : target_free_qpos + 3] = root_position
        target_data.qpos[target_free_qpos + 3 : target_free_qpos + 7] = root_quaternion
        target_data.qpos[target_qpos_addresses] = joint_pos[frame_index]
        target_data.qvel[target_free_qvel : target_free_qvel + 6] = base_qvel
        target_data.qvel[target_qvel_addresses] = joint_vel[frame_index]
        mujoco.mj_forward(target_model, target_data)

        body_pos_w[frame_index] = target_data.xpos[target_body_ids]
        body_quat_w[frame_index] = target_data.xquat[target_body_ids]
        for body_index, body_id in enumerate(target_body_ids):
            # MuJoCo uses angular:linear ordering; local=0 returns world axes.
            mujoco.mj_objectVelocity(
                target_model,
                target_data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                object_velocity,
                0,
            )
            body_ang_vel_w[frame_index, body_index] = object_velocity[:3]
            body_lin_vel_w[frame_index, body_index] = object_velocity[3:]

    if max_base_velocity_reconstruction_error > 1.0e-8:
        raise RuntimeError(
            "free-joint velocity reconstruction failed; max pelvis COM velocity "
            f"error={max_base_velocity_reconstruction_error:.9g}"
        )

    position_error = np.abs(
        body_pos_w[:, target_body_indices]
        - np.asarray(source_body_pos, dtype=np.float32)
    )
    pose_position_max_error_m = float(np.max(position_error))
    pose_orientation_max_error_rad = float(
        np.max(
            _quaternion_angle_error(
                np.asarray(body_quat_w[:, target_body_indices], dtype=np.float64),
                np.asarray(source_body_quat, dtype=np.float64),
            )
        )
    )
    # MuJoCo reports body velocity at the inertial frame origin.  Two builds of
    # the same kinematic chain can place that point differently -- BrainCo splits
    # the rubber hand out of wrist_yaw_link, moving its centre of mass by ~2 cm --
    # so the rigid-body transport term is the exact expected difference, not an
    # error to tolerate.  Identical inertial frames leave the term at zero and
    # the comparison unchanged.
    expected_lin_vel = np.asarray(source_body_lin_vel, dtype=np.float32)
    if inertial_offset_max_m > 0.0:
        offset_w = _rotate_by_quaternion(
            np.asarray(body_quat_w[:, target_body_indices], dtype=np.float64),
            np.broadcast_to(
                inertial_offset_body[None, :, :],
                (frame_count, len(source_body_names), 3),
            ),
        )
        expected_lin_vel = (
            expected_lin_vel
            + np.cross(
                np.asarray(source_body_ang_vel, dtype=np.float64), offset_w
            ).astype(np.float32)
        ).astype(np.float32)
    velocity_linear_max_error = float(
        np.max(np.abs(body_lin_vel_w[:, target_body_indices] - expected_lin_vel))
    )
    velocity_angular_max_error = float(
        np.max(
            np.abs(
                body_ang_vel_w[:, target_body_indices]
                - np.asarray(source_body_ang_vel, dtype=np.float32)
            )
        )
    )
    if pose_position_max_error_m > float(args.max_source_body_position_error_m):
        raise RuntimeError(
            f"Dex3 FK changed a source body position by {pose_position_max_error_m:.9g} m"
        )
    if pose_orientation_max_error_rad > 2.0e-3:
        raise RuntimeError(
            "Dex3 FK changed a source body orientation by "
            f"{pose_orientation_max_error_rad:.9g} rad"
        )
    if velocity_linear_max_error > 1.0e-4 or velocity_angular_max_error > 1.0e-4:
        raise RuntimeError(
            "render model changed a source body velocity beyond the inertial "
            f"frame transport term: linear={velocity_linear_max_error:.9g} m/s, "
            f"angular={velocity_angular_max_error:.9g} rad/s"
        )

    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos_w": body_pos_w[:, 0].copy(),
        "root_quat_wxyz": body_quat_w[:, 0].copy(),
        "root_lin_vel_w": body_lin_vel_w[:, 0].copy(),
        "root_ang_vel_w": body_ang_vel_w[:, 0].copy(),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.asarray(target_hinges),
        "body_names": np.asarray(target_body_names),
        "timestamps": body_timestamps,
        "fps": np.asarray(int(round(body_fps)), dtype=np.int64),
        "hand_qpos": joint_pos[:, hand_output_indices].copy(),
        "hand_qvel": joint_vel[:, hand_output_indices].copy(),
        "hand_joint_names": np.asarray(target_hand_names),
        "source_body_kinematics": np.asarray(str(body_path)),
        "source_hand_kinematics": np.asarray(str(hand_path)),
        "source_body_model": np.asarray(str(body_model_path)),
        "source_dex3_model": np.asarray(str(dex3_model_path)),
    }
    if mimic_rules:
        # Only emitted for render models that actually declare joint equalities,
        # so the Dex3 archive keeps its historical field set byte for byte.
        arrays["mimic_joint_names"] = np.asarray(
            [str(rule["dependent"]) for rule in mimic_rules]
        )
        arrays["mimic_reference_names"] = np.asarray(
            [str(rule["reference"]) for rule in mimic_rules]
        )
        arrays["mimic_polycoef"] = np.asarray(
            [np.asarray(rule["polycoef"], dtype=np.float64) for rule in mimic_rules]
        )
        arrays["mimic_qpos0"] = np.asarray(
            [
                [float(rule["dependent_qpos0"]), float(rule["reference_qpos0"])]
                for rule in mimic_rules
            ],
            dtype=np.float64,
        )
    for field, values in arrays.items():
        if np.issubdtype(np.asarray(values).dtype, np.number) and not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(f"output field {field} contains non-finite values")

    body_output_indices = [target_hinges.index(name) for name in body_joint_names]
    if not np.array_equal(joint_pos[:, body_output_indices], body_joint_pos):
        raise RuntimeError("body joint positions were not preserved exactly")
    if not np.array_equal(joint_vel[:, body_output_indices], body_joint_vel):
        raise RuntimeError("body joint velocities were not preserved exactly")

    validation = {
        "body_joint_positions_exact": True,
        "body_joint_velocities_exact": True,
        "max_source_body_position_error_m": pose_position_max_error_m,
        "max_source_body_orientation_error_rad": pose_orientation_max_error_rad,
        "max_source_body_linear_velocity_error_m_per_s": velocity_linear_max_error,
        "max_source_body_angular_velocity_error_rad_per_s": velocity_angular_max_error,
        "max_base_velocity_reconstruction_error": max_base_velocity_reconstruction_error,
        "max_body_hand_timestamp_error_s": timestamp_error,
        "source_body_quaternion_norm_error": quaternion_norm_error,
        "max_hand_joint_limit_violation_rad": max(0.0, hand_violation),
    }
    if mimic_rules:
        validation["max_mimic_range_clip_rad"] = mimic_clip_max_rad
    if inertial_offset_bodies:
        validation["inertial_frame_offset_bodies"] = inertial_offset_bodies
        validation["max_inertial_frame_offset_m"] = inertial_offset_max_m
    metadata = {
        "format": "Dex3-composed MuJoCo kinematics NPZ",
        "format_version": 1,
        "generation": {
            "method": "offline state assignment followed by mujoco.mj_forward",
            "physics_integration": False,
            "policy_or_mpc_execution": False,
            "body_motion_modified": False,
        },
        "sources": {
            "body_kinematics": str(body_path),
            "hand_kinematics": str(hand_path),
            "body_model_xml": str(body_model_path),
            "dex3_model_xml": str(dex3_model_path),
            "dex3_meshdir": str(target_meshdir),
        },
        "model": {
            "nq": target_model.nq,
            "nv": target_model.nv,
            "hinge_joint_count": len(target_hinges),
            "nonworld_body_count": len(target_body_names),
            "in_memory_compiler_patch": {
                "meshdir": str(target_meshdir),
                "boundmass": COMPILER_BOUNDMASS,
                "boundinertia": COMPILER_BOUNDINERTIA,
                "source_xml_edited": False,
            },
            "d435": d435_calibration(),
        },
        "sampling": {
            "frames": frame_count,
            "fps": body_fps,
            "period_s": 1.0 / body_fps,
            "duration_s": float(body_timestamps[-1] - body_timestamps[0]),
            "timestamps": "kinematics.npz:timestamps, relative seconds from frame 0",
        },
        "coordinate_conventions": {
            "world_suffix": "fields ending in _w are expressed in MuJoCo world axes",
            "cartesian_axes": ["x", "y", "z"],
            "quaternion_order": "wxyz",
            "body_pose_point": "MuJoCo body-frame origin",
            "body_velocity_point": "MuJoCo body center of mass",
            "body_velocity_source": "mj_objectVelocity(local=0), reordered to linear then angular fields",
            "joint_position_unit": "rad",
            "joint_velocity_unit": "rad/s",
        },
        "dimensions": {
            "T": frame_count,
            "J": len(target_hinges),
            "B": len(target_body_names),
            "H": len(target_hand_names),
        },
        "joint_names": target_hinges,
        "body_names": target_body_names,
        "hand_joint_names": target_hand_names,
        "source_body_joint_names": body_joint_names,
        "source_body_names": source_body_names,
        "joint_index_to_name": {
            str(index): name for index, name in enumerate(target_hinges)
        },
        "body_index_to_name": {
            str(index): name for index, name in enumerate(target_body_names)
        },
        "hand_joint_index_to_name": {
            str(index): name for index, name in enumerate(target_hand_names)
        },
        "root": {
            "body_index": 0,
            "body_name": ROOT_BODY_NAME,
            "position": "root_pos_w",
            "orientation": "root_quat_wxyz",
            "linear_velocity": "root_lin_vel_w",
            "angular_velocity": "root_ang_vel_w",
        },
        "fields": {
            "joint_pos": {
                "shape": ["T", "J"],
                "unit": "rad",
                "axes": ["time", "joint_names"],
            },
            "joint_vel": {
                "shape": ["T", "J"],
                "unit": "rad/s",
                "axes": ["time", "joint_names"],
            },
            "root_pos_w": {
                "shape": ["T", 3],
                "unit": "m",
                "axes": ["time", ["x", "y", "z"]],
            },
            "root_quat_wxyz": {
                "shape": ["T", 4],
                "unit": "unit quaternion",
                "axes": ["time", ["w", "x", "y", "z"]],
            },
            "root_lin_vel_w": {
                "shape": ["T", 3],
                "unit": "m/s",
                "axes": ["time", ["x", "y", "z"]],
            },
            "root_ang_vel_w": {
                "shape": ["T", 3],
                "unit": "rad/s",
                "axes": ["time", ["x", "y", "z"]],
            },
            "body_pos_w": {
                "shape": ["T", "B", 3],
                "unit": "m",
                "axes": ["time", "body_names", ["x", "y", "z"]],
            },
            "body_quat_w": {
                "shape": ["T", "B", 4],
                "unit": "unit quaternion",
                "axes": ["time", "body_names", ["w", "x", "y", "z"]],
            },
            "body_lin_vel_w": {
                "shape": ["T", "B", 3],
                "unit": "m/s",
                "axes": ["time", "body_names", ["x", "y", "z"]],
            },
            "body_ang_vel_w": {
                "shape": ["T", "B", 3],
                "unit": "rad/s",
                "axes": ["time", "body_names", ["x", "y", "z"]],
            },
            "hand_qpos": {
                "shape": ["T", "H"],
                "unit": "rad",
                "axes": ["time", "hand_joint_names"],
            },
            "hand_qvel": {
                "shape": ["T", "H"],
                "unit": "rad/s",
                "axes": ["time", "hand_joint_names"],
            },
            "timestamps": {"shape": ["T"], "unit": "s", "origin": "frame 0"},
            "fps": {"shape": [], "unit": "Hz"},
        },
        "validation": validation,
    }
    if mimic_rules:
        metadata["supplied_hand_joint_names"] = hand_joint_names
        metadata["mimic_joints"] = [
            {
                "dependent": str(rule["dependent"]),
                "reference": str(rule["reference"]),
                "polycoef": np.asarray(rule["polycoef"], dtype=np.float64).tolist(),
                "dependent_qpos0": float(rule["dependent_qpos0"]),
                "reference_qpos0": float(rule["reference_qpos0"]),
            }
            for rule in mimic_rules
        ]

    _atomic_save_npz(output_path, arrays)
    metadata_path = output_path.parent / "meta.json"
    _atomic_save_json(metadata_path, metadata)
    result = {
        "output": str(output_path),
        "metadata": str(metadata_path),
        "frames": frame_count,
        "fps": body_fps,
        "joints": len(target_hinges),
        "bodies": len(target_body_names),
        "validation": validation,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-kinematics", type=Path, required=True)
    parser.add_argument("--hand-kinematics", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; writes kinematics.npz and meta.json.",
    )
    destination.add_argument(
        "--output",
        type=Path,
        help="Explicit output NPZ path; writes meta.json beside it.",
    )
    parser.add_argument("--body-model", type=Path, default=DEFAULT_BODY_XML)
    parser.add_argument("--dex3-model", type=Path, default=DEFAULT_DEX3_XML)
    parser.add_argument("--meshdir", type=Path, default=DEFAULT_G1_MESHDIR)
    parser.add_argument(
        "--dex3-meshdir",
        type=Path,
        default=None,
        help="Mesh root for --dex3-model when it differs from --meshdir.",
    )
    parser.add_argument(
        "--hand-mode",
        choices=sorted(RECORDING_MODEL_CONTRACTS),
        default="dex3",
        help="Render-model contract to enforce for --dex3-model.",
    )
    parser.add_argument(
        "--expected-hand-joints",
        type=int,
        default=EXPECTED_HAND_JOINTS,
        help=(
            "Width of the supplied hand archive. Joints that the render XML "
            "drives through a joint equality are expanded, not supplied."
        ),
    )
    parser.add_argument(
        "--max-mimic-clip-rad",
        type=float,
        default=1.0e-3,
        help=(
            "Rounding allowance when a mimic polynomial overshoots the driven "
            "joint's own XML range at full flexion."
        ),
    )
    parser.add_argument(
        "--max-source-body-position-error-m",
        type=float,
        default=1.0e-5,
        help=(
            "Maximum allowed source-vs-Dex3 FK position difference. The strict "
            "default catches model mismatches; seeded contact view can pass an "
            "audited tolerance when Mink and render XML have different meshes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    compose(parse_args())


if __name__ == "__main__":
    main()
