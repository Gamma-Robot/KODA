#!/usr/bin/env python3
"""Shared MuJoCo model helpers for offline Dex3 kinematics tools."""

from __future__ import annotations

from pathlib import Path
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


DEFAULT_DEX3_XML = Path(os.environ.get("KODA_DEX3_XML", "assets/robots/g1_with_visual_dex3_hands.xml"))
DEFAULT_BODY_XML = Path(os.environ.get("KODA_BODY_XML", "assets/robots/g1_29dof_mjx.xml"))
DEFAULT_G1_MESHDIR = Path(os.environ.get("KODA_G1_MESHDIR", "assets/robots/g1/meshes"))

D435_BODY_NAME = "d435_link"
D435_CAMERA_NAME = "front_cam"
D435_PARENT_BODY = "torso_link"
D435_POSITION = (0.0576235, 0.01753, 0.41987)
D435_QUATERNION_WXYZ = (0.91496, 0.0, 0.403545, 0.0)
D435_CAMERA_POSITION = (0.0, 0.0, 0.0)
D435_CAMERA_QUATERNION_WXYZ = (0.5, 0.5, -0.5, -0.5)
D435_WIDTH = 640
D435_HEIGHT = 480
D435_FX = 606.5891
D435_FY = 605.7623
D435_CX = 320.0
D435_CY = 240.0
D435_PRINCIPALPIXEL_OFFSET = (
    D435_CX - D435_WIDTH / 2.0,
    D435_CY - D435_HEIGHT / 2.0,
)

COMPILER_BOUNDMASS = 1.0e-6
COMPILER_BOUNDINERTIA = 1.0e-8

# Recording-model contracts keyed by hand mode.  Dex3 keeps its historical
# literals; g1_brainco adds five mimic distal joints per hand on top of the six
# actively retargeted ones.
RECORDING_MODEL_CONTRACTS = {
    "dex3": {"nq": 50, "nv": 49, "hinges": 43, "bodies": 45},
    "g1_brainco": {"nq": 58, "nv": 57, "hinges": 51, "bodies": 69},
}


def _vector_text(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def build_dex3_xml(
    xml_path: str | Path = DEFAULT_DEX3_XML,
    *,
    meshdir: str | Path = DEFAULT_G1_MESHDIR,
    with_d435: bool = True,
) -> str:
    """Return a compilable in-memory Dex3 MJCF without editing the source XML."""

    xml_path = Path(xml_path).expanduser().resolve()
    meshdir = Path(meshdir).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"Dex3 XML does not exist: {xml_path}")
    if not meshdir.is_dir():
        raise FileNotFoundError(f"G1 mesh directory does not exist: {meshdir}")

    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(meshdir))
    compiler.set("boundmass", f"{COMPILER_BOUNDMASS:g}")
    compiler.set("boundinertia", f"{COMPILER_BOUNDINERTIA:g}")

    if with_d435:
        existing_bodies = root.findall(f".//body[@name='{D435_BODY_NAME}']")
        existing_cameras = root.findall(f".//camera[@name='{D435_CAMERA_NAME}']")
        if existing_bodies or existing_cameras:
            raise ValueError(
                f"cannot inject {D435_BODY_NAME}/{D435_CAMERA_NAME}: "
                "one of those names already exists in the source XML"
            )
        torso_bodies = root.findall(f".//body[@name='{D435_PARENT_BODY}']")
        if len(torso_bodies) != 1:
            raise ValueError(
                f"expected exactly one {D435_PARENT_BODY!r} body, found "
                f"{len(torso_bodies)}"
            )
        d435 = ET.SubElement(
            torso_bodies[0],
            "body",
            name=D435_BODY_NAME,
            pos=_vector_text(D435_POSITION),
            quat=_vector_text(D435_QUATERNION_WXYZ),
        )
        ET.SubElement(
            d435,
            "camera",
            name=D435_CAMERA_NAME,
            pos=_vector_text(D435_CAMERA_POSITION),
            quat=_vector_text(D435_CAMERA_QUATERNION_WXYZ),
            resolution=f"{D435_WIDTH} {D435_HEIGHT}",
            focalpixel=f"{D435_FX:.10g} {D435_FY:.10g}",
            # MuJoCo expects principalpixel relative to the image center, not
            # the absolute OpenCV principal point stored in K.
            principalpixel=_vector_text(D435_PRINCIPALPIXEL_OFFSET),
            # The size only converts pixel calibration to MuJoCo's physical
            # intrinsic representation; one unit per sensor axis is enough.
            sensorsize="1 1",
        )

    return ET.tostring(root, encoding="unicode")


def load_dex3_model(
    xml_path: str | Path = DEFAULT_DEX3_XML,
    *,
    meshdir: str | Path = DEFAULT_G1_MESHDIR,
    with_d435: bool = True,
    hand_mode: str = "dex3",
) -> mujoco.MjModel:
    """Compile the visual Dex3 model after applying only in-memory patches."""

    xml = build_dex3_xml(xml_path, meshdir=meshdir, with_d435=with_d435)
    model = mujoco.MjModel.from_xml_string(xml)
    if with_d435:
        validate_dex3_recording_model(model, hand_mode=hand_mode)
    return model


def object_names(model: mujoco.MjModel, object_type: mujoco.mjtObj) -> list[str]:
    """Return all named objects of a type in MuJoCo ID order."""

    if object_type == mujoco.mjtObj.mjOBJ_BODY:
        count = model.nbody
    elif object_type == mujoco.mjtObj.mjOBJ_JOINT:
        count = model.njnt
    elif object_type == mujoco.mjtObj.mjOBJ_CAMERA:
        count = model.ncam
    else:
        raise ValueError(f"unsupported object type: {object_type}")
    names = [mujoco.mj_id2name(model, object_type, index) for index in range(count)]
    if any(name is None for name in names):
        raise ValueError(f"model has unnamed objects of type {object_type}")
    return [str(name) for name in names]


def hinge_joint_names(model: mujoco.MjModel) -> list[str]:
    names = object_names(model, mujoco.mjtObj.mjOBJ_JOINT)
    return [
        name
        for joint_id, name in enumerate(names)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]


def nonworld_body_names(model: mujoco.MjModel) -> list[str]:
    names = object_names(model, mujoco.mjtObj.mjOBJ_BODY)
    if not names or names[0] != "world":
        raise ValueError("MuJoCo body ID 0 is not named 'world'")
    return names[1:]


def require_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"model is missing {name!r} ({object_type})")
    return object_id


def joint_mimic_rules(model: mujoco.MjModel) -> list[dict[str, object]]:
    """Return the model's joint equality constraints as explicit mimic rules.

    MuJoCo resolves ``mjEQ_JOINT`` through the constraint solver during
    dynamics, so an offline kinematics pass that only calls ``mj_forward`` must
    evaluate the polynomial itself.  A model without joint equalities (Dex3)
    returns an empty list, which keeps every caller on its historical path.
    """

    rules: list[dict[str, object]] = []
    for equality_id in range(int(model.neq)):
        if int(model.eq_type[equality_id]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        if not bool(model.eq_active0[equality_id]):
            raise ValueError(
                f"joint equality {equality_id} is inactive; offline mimic "
                "expansion has no defined semantics for it"
            )
        dependent_id = int(model.eq_obj1id[equality_id])
        reference_id = int(model.eq_obj2id[equality_id])
        if reference_id < 0:
            raise ValueError(
                f"joint equality {equality_id} has no reference joint; a "
                "single-joint constraint cannot be expanded offline"
            )
        for joint_id in (dependent_id, reference_id):
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(
                    "offline mimic expansion supports hinge joints only, got "
                    f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)}"
                )
        rules.append(
            {
                "dependent": str(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, dependent_id)
                ),
                "reference": str(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, reference_id)
                ),
                "polycoef": np.asarray(
                    model.eq_data[equality_id][:5], dtype=np.float64
                ).copy(),
                "dependent_qpos0": float(
                    model.qpos0[int(model.jnt_qposadr[dependent_id])]
                ),
                "reference_qpos0": float(
                    model.qpos0[int(model.jnt_qposadr[reference_id])]
                ),
            }
        )
    dependents = [str(rule["dependent"]) for rule in rules]
    duplicates = sorted({name for name in dependents if dependents.count(name) > 1})
    if duplicates:
        raise ValueError(f"joint equalities drive the same joint twice: {duplicates}")
    references = {str(rule["reference"]) for rule in rules}
    chained = sorted(references.intersection(dependents))
    if chained:
        raise ValueError(f"chained joint equalities are not supported: {chained}")
    return rules


def evaluate_mimic_rule(
    rule: dict[str, object],
    reference_qpos: np.ndarray,
    reference_qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one MuJoCo joint equality and its time derivative.

    MuJoCo defines ``y - y0 = sum_k a_k * (x - x0)^k`` on qpos relative to the
    model reference pose, so the velocity follows from the chain rule.
    """

    coefficients = np.asarray(rule["polycoef"], dtype=np.float64)
    offset = np.asarray(reference_qpos, dtype=np.float64) - float(
        rule["reference_qpos0"]
    )
    powers = np.stack([offset**exponent for exponent in range(5)], axis=-1)
    dependent_qpos = float(rule["dependent_qpos0"]) + powers @ coefficients
    derivative_powers = np.stack(
        [exponent * offset ** max(exponent - 1, 0) for exponent in range(5)], axis=-1
    )
    dependent_qvel = (derivative_powers @ coefficients) * np.asarray(
        reference_qvel, dtype=np.float64
    )
    return dependent_qpos, dependent_qvel


def d435_calibration() -> dict[str, object]:
    return {
        "body_name": D435_BODY_NAME,
        "camera_name": D435_CAMERA_NAME,
        "parent_body": D435_PARENT_BODY,
        "mount_position_m": list(D435_POSITION),
        "mount_quaternion_wxyz": list(D435_QUATERNION_WXYZ),
        "camera_position_m": list(D435_CAMERA_POSITION),
        "camera_quaternion_wxyz": list(D435_CAMERA_QUATERNION_WXYZ),
        "width": D435_WIDTH,
        "height": D435_HEIGHT,
        "fx": D435_FX,
        "fy": D435_FY,
        "cx": D435_CX,
        "cy": D435_CY,
        "mujoco_principalpixel_offset": list(D435_PRINCIPALPIXEL_OFFSET),
        "K": [
            [D435_FX, 0.0, D435_CX],
            [0.0, D435_FY, D435_CY],
            [0.0, 0.0, 1.0],
        ],
    }


def validate_dex3_recording_model(
    model: mujoco.MjModel,
    *,
    hand_mode: str = "dex3",
) -> None:
    """Fail early if the shared recording model contract changes."""

    try:
        contract = RECORDING_MODEL_CONTRACTS[hand_mode]
    except KeyError:
        raise ValueError(
            f"unknown hand mode {hand_mode!r}; expected one of "
            f"{sorted(RECORDING_MODEL_CONTRACTS)}"
        ) from None
    hinges = hinge_joint_names(model)
    bodies = nonworld_body_names(model)
    if (model.nq, model.nv) != (contract["nq"], contract["nv"]):
        raise ValueError(
            f"{hand_mode} recording model has nq/nv={model.nq}/{model.nv}, "
            f"expected {contract['nq']}/{contract['nv']}"
        )
    if len(hinges) != contract["hinges"]:
        raise ValueError(
            f"{hand_mode} recording model has {len(hinges)} hinges, "
            f"expected {contract['hinges']}"
        )
    if len(bodies) != contract["bodies"]:
        raise ValueError(
            f"{hand_mode} recording model has {len(bodies)} non-world bodies, "
            f"expected {contract['bodies']}"
        )
    require_object_id(model, mujoco.mjtObj.mjOBJ_BODY, D435_BODY_NAME)
    camera_id = require_object_id(model, mujoco.mjtObj.mjOBJ_CAMERA, D435_CAMERA_NAME)
    if not np.array_equal(
        np.asarray(model.cam_resolution[camera_id]),
        np.asarray([D435_WIDTH, D435_HEIGHT]),
    ):
        raise ValueError("front_cam resolution does not match the D435 calibration")
    expected_intrinsic = np.asarray(
        [
            D435_FX / D435_WIDTH,
            D435_FY / D435_HEIGHT,
            D435_PRINCIPALPIXEL_OFFSET[0] / D435_WIDTH,
            D435_PRINCIPALPIXEL_OFFSET[1] / D435_HEIGHT,
        ]
    )
    if not np.allclose(model.cam_intrinsic[camera_id], expected_intrinsic, atol=1.0e-7):
        raise ValueError(
            "front_cam compiled intrinsics do not match the requested D435 calibration"
        )
