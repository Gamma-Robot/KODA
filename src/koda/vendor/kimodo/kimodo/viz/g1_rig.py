# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G1 robot rig: mesh loading, joint mapping, and viser scene setup for G1 skeleton."""

import os
import xml.etree.ElementTree as ET
from typing import Any, Optional, Tuple

import numpy as np
import trimesh

import viser
import viser.transforms as tf
from kimodo.assets import skeleton_asset_path
from kimodo.skeleton import G1Skeleton34

# MuJoCo (z-up, x-forward) -> kimodo (y-up, z-forward)
MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

# MuJoCo (z-up, x-forward) -> kimodo (y-up, z-forward)
MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

G1_MESH_JOINT_MAP = {
    "pelvis_skel": ["pelvis.STL", "pelvis_contour_link.STL"],
    "left_hip_pitch_skel": ["left_hip_pitch_link.STL"],
    "left_hip_roll_skel": ["left_hip_roll_link.STL"],
    "left_hip_yaw_skel": ["left_hip_yaw_link.STL"],
    "left_knee_skel": ["left_knee_link.STL"],
    "left_ankle_pitch_skel": ["left_ankle_pitch_link.STL"],
    "left_ankle_roll_skel": ["left_ankle_roll_link.STL"],
    "right_hip_pitch_skel": ["right_hip_pitch_link.STL"],
    "right_hip_roll_skel": ["right_hip_roll_link.STL"],
    "right_hip_yaw_skel": ["right_hip_yaw_link.STL"],
    "right_knee_skel": ["right_knee_link.STL"],
    "right_ankle_pitch_skel": ["right_ankle_pitch_link.STL"],
    "right_ankle_roll_skel": ["right_ankle_roll_link.STL"],
    "waist_yaw_skel": ["waist_yaw_link_rev_1_0.STL", "waist_yaw_link.STL"],
    "waist_roll_skel": ["waist_roll_link_rev_1_0.STL", "waist_roll_link.STL"],
    "waist_pitch_skel": [
        "torso_link_rev_1_0.STL",
        "torso_link.STL",
        "logo_link.STL",
        "head_link.STL",
    ],
    "left_shoulder_pitch_skel": ["left_shoulder_pitch_link.STL"],
    "left_shoulder_roll_skel": ["left_shoulder_roll_link.STL"],
    "left_shoulder_yaw_skel": ["left_shoulder_yaw_link.STL"],
    "left_elbow_skel": ["left_elbow_link.STL"],
    "left_wrist_roll_skel": ["left_wrist_roll_link.STL"],
    "left_wrist_pitch_skel": ["left_wrist_pitch_link.STL"],
    "left_wrist_yaw_skel": ["left_wrist_yaw_link.STL", "left_rubber_hand.STL"],
    "right_shoulder_pitch_skel": ["right_shoulder_pitch_link.STL"],
    "right_shoulder_roll_skel": ["right_shoulder_roll_link.STL"],
    "right_shoulder_yaw_skel": ["right_shoulder_yaw_link.STL"],
    "right_elbow_skel": ["right_elbow_link.STL"],
    "right_wrist_roll_skel": ["right_wrist_roll_link.STL"],
    "right_wrist_pitch_skel": ["right_wrist_pitch_link.STL"],
    "right_wrist_yaw_skel": ["right_wrist_yaw_link.STL", "right_rubber_hand.STL"],
}

# Joint axis/limits from g1.xml (used by exports, e.g. MujocoQposConverter)
_G1_JOINT_AXIS_INDEX_CACHE: Optional[dict[str, int]] = None
_G1_JOINT_LIMITS_CACHE: Optional[dict[str, tuple[float, float]]] = None


def _get_g1_joint_axis_indices() -> dict[str, int]:
    """Return a map from G1 joint names to a single rotation axis index."""
    global _G1_JOINT_AXIS_INDEX_CACHE
    if _G1_JOINT_AXIS_INDEX_CACHE is not None:
        return _G1_JOINT_AXIS_INDEX_CACHE

    xml_path = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    if not os.path.exists(xml_path):
        _G1_JOINT_AXIS_INDEX_CACHE = {}
        return _G1_JOINT_AXIS_INDEX_CACHE

    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_axes = {}
    for xml_class in tree.findall(".//default"):
        if "class" not in xml_class.attrib:
            continue
        joint_nodes = xml_class.findall("joint")
        if joint_nodes:
            joint_axes[xml_class.get("class")] = joint_nodes[0].get("axis")

    axis_indices_by_name: dict[str, int] = {}
    for joint in root.find("worldbody").findall(".//joint"):
        axis_str = joint.get("axis") or joint_axes.get(joint.get("class"))
        if axis_str is None:
            continue
        axis_vals = np.array([float(x) for x in axis_str.split()], dtype=np.float64)
        if not np.any(axis_vals):
            continue
        axis_kimodo = MUJOCO_TO_KIMODO @ axis_vals
        axis_idx = int(np.argmax(np.abs(axis_kimodo)))
        axis_indices_by_name[joint.get("name").replace("_joint", "_skel")] = axis_idx

    _G1_JOINT_AXIS_INDEX_CACHE = axis_indices_by_name
    return _G1_JOINT_AXIS_INDEX_CACHE


def _get_g1_joint_limits() -> dict[str, tuple[float, float]]:
    """Return a map from G1 joint names to (min, max) angle limits in radians."""
    global _G1_JOINT_LIMITS_CACHE
    if _G1_JOINT_LIMITS_CACHE is not None:
        return _G1_JOINT_LIMITS_CACHE

    xml_path = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    if not os.path.exists(xml_path):
        _G1_JOINT_LIMITS_CACHE = {}
        return _G1_JOINT_LIMITS_CACHE

    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_ranges: dict[str, tuple[float, float]] = {}
    for xml_class in tree.findall(".//default"):
        class_name = xml_class.get("class")
        if not class_name:
            continue
        joint_nodes = xml_class.findall("joint")
        if not joint_nodes:
            continue
        range_str = joint_nodes[0].get("range")
        if not range_str:
            continue
        range_vals = [float(x) for x in range_str.split()]
        if len(range_vals) != 2:
            continue
        class_ranges[class_name] = (range_vals[0], range_vals[1])

    joint_limits: dict[str, tuple[float, float]] = {}
    worldbody = root.find("worldbody")
    if worldbody is None:
        _G1_JOINT_LIMITS_CACHE = {}
        return _G1_JOINT_LIMITS_CACHE

    for joint in worldbody.findall(".//joint"):
        range_str = joint.get("range") or class_ranges.get(joint.get("class"))
        if range_str is None:
            continue
        if isinstance(range_str, tuple):
            joint_range = range_str
        else:
            range_vals = [float(x) for x in range_str.split()]
            if len(range_vals) != 2:
                continue
            joint_range = (range_vals[0], range_vals[1])
        joint_name = joint.get("name")
        if not joint_name:
            continue
        joint_limits[joint_name.replace("_joint", "_skel")] = joint_range

    _G1_JOINT_LIMITS_CACHE = joint_limits
    return _G1_JOINT_LIMITS_CACHE


_G1_JOINT_F2Q_DATA_CACHE: Optional[dict[str, dict[str, Any]]] = None


def get_g1_joint_f2q_data(
    skeleton: G1Skeleton34,
) -> dict[str, dict[str, Any]]:
    """Return per-hinge-joint f2q data for correct 1-DoF + limits in offset space.

    Each entry is for a G1 hinge joint (by name) and contains:
      - "offset_f2q": (3, 3) matrix such that R_f2q = offset_f2q @ R_local (kimodo).
      - "axis_f2q": (3,) unit axis in f2q space; angle = dot(axis_angle(R_f2q), axis_f2q).
      - "rest_dof_axis_angle": angle (rad) at T-pose in f2q space; MuJoCo q = angle_f2q - this.

    Limits from the XML apply to q = angle_f2q - rest_dof_axis_angle.
    """
    global _G1_JOINT_F2Q_DATA_CACHE
    if _G1_JOINT_F2Q_DATA_CACHE is not None:
        return _G1_JOINT_F2Q_DATA_CACHE

    from kimodo.exports.mujoco import MujocoQposConverter

    converter = MujocoQposConverter(skeleton)
    # converter: _rot_offsets_f2q[kimodo_idx], _mujoco_joint_axis_values_f2q_space[hinge_idx],
    # _rest_dofs_axis_angle[hinge_idx], _kimodo_indices_to_mujoco_indices[kimodo_idx] = hinge_idx+1 or 0
    out: dict[str, dict[str, Any]] = {}
    for j in range(skeleton.nbjoints):
        mujoco_one_based = converter._kimodo_indices_to_mujoco_indices[j].item()
        if mujoco_one_based <= 0:
            continue
        hinge_idx = mujoco_one_based - 1
        joint_name = skeleton.bone_order_names[j]
        offset_f2q = converter._rot_offsets_f2q[j].detach().cpu().numpy().astype(np.float64)
        axis_f2q = converter._mujoco_joint_axis_values_f2q_space[hinge_idx].detach().cpu().numpy().astype(np.float64)
        n = np.linalg.norm(axis_f2q)
        if n > 1e-10:
            axis_f2q = axis_f2q / n
        rest_dof = float(converter._rest_dofs_axis_angle[hinge_idx].detach().cpu().numpy())
        out[joint_name] = {
            "offset_f2q": offset_f2q,
            "axis_f2q": axis_f2q,
            "rest_dof_axis_angle": rest_dof,
        }
    _G1_JOINT_F2Q_DATA_CACHE = out
    return out


# -----------------------------------------------------------------------------
# Mesh loading cache (shared across G1 rig instances; each rig gets its own scene meshes)
# -----------------------------------------------------------------------------
_G1_MESH_DATA_CACHE: dict[str, list[dict]] = {}


def _load_g1_mesh_data(
    mesh_dir: str,
    skeleton: G1Skeleton34,
) -> list[dict]:
    """Load STL meshes and XML transforms once per mesh_dir; shared across rig instances."""
    if mesh_dir in _G1_MESH_DATA_CACHE:
        return _G1_MESH_DATA_CACHE[mesh_dir]

    mesh_geom_cache = G1MeshRig._mesh_geom_cache
    mesh_transform_cache = G1MeshRig._mesh_transform_cache

    # Load XML-derived transforms (cached inside _get_mesh_local_transforms_impl)
    mesh_file_transforms = _get_mesh_local_transforms_impl(mesh_dir, mesh_transform_cache)

    data_list: list[dict] = []
    for joint_name, mesh_files in G1_MESH_JOINT_MAP.items():
        if joint_name not in skeleton.bone_index:
            continue
        joint_idx = skeleton.bone_index[joint_name]
        for mesh_file in mesh_files:
            mesh_path = os.path.join(mesh_dir, mesh_file)
            if not os.path.exists(mesh_path):
                continue
            vertices, faces = _get_mesh_geometry_impl(mesh_file, mesh_path, mesh_dir, mesh_geom_cache)
            if vertices is None:
                continue
            geom_pos, geom_rot = mesh_file_transforms.get(
                mesh_file,
                (np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
            )
            data_list.append(
                {
                    "mesh_file": mesh_file,
                    "vertices": vertices,
                    "faces": faces,
                    "joint_idx": joint_idx,
                    "geom_pos": geom_pos.copy(),
                    "geom_rot": geom_rot.copy(),
                }
            )

    _G1_MESH_DATA_CACHE[mesh_dir] = data_list
    return data_list


def _get_mesh_geometry_impl(
    mesh_file: str,
    mesh_path: str,
    mesh_dir: str,
    mesh_geom_cache: dict,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load one STL; result cached per mesh_dir and shared across rigs."""
    cached = mesh_geom_cache.get(mesh_dir)
    if cached is not None and mesh_file in cached:
        vertices, faces = cached[mesh_file]
        return vertices.copy(), faces.copy()

    mesh = trimesh.load_mesh(mesh_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    vertices = mesh.vertices @ MUJOCO_TO_KIMODO.T
    faces = mesh.faces

    if mesh_dir not in mesh_geom_cache:
        mesh_geom_cache[mesh_dir] = {}
    mesh_geom_cache[mesh_dir][mesh_file] = (vertices, faces)
    return vertices.copy(), faces.copy()


def _get_mesh_local_transforms_impl(
    mesh_dir: str,
    mesh_transform_cache: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse g1.xml once per mesh_dir; result shared across G1 rig instances."""
    cached = mesh_transform_cache.get(mesh_dir)
    if cached is not None:
        return {mesh_file: (pos.copy(), rot.copy()) for mesh_file, (pos, rot) in cached.items()}

    xml_path = os.path.abspath(os.path.join(mesh_dir, "..", "..", "xml", "g1.xml"))
    if not os.path.exists(xml_path):
        return {}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mesh_file_to_mesh_name = {}
    for mesh in root.findall(".//asset/mesh"):
        mesh_name = mesh.get("name")
        mesh_file = mesh.get("file")
        if mesh_name and mesh_file:
            mesh_file_to_mesh_name[mesh_file] = mesh_name

    mesh_name_to_transform = {}
    for geom in root.findall(".//geom"):
        mesh_name = geom.get("mesh")
        if mesh_name is None:
            continue
        pos = geom.get("pos")
        quat = geom.get("quat")
        if pos is None:
            geom_pos = np.zeros(3, dtype=np.float64)
        else:
            geom_pos = np.array([float(x) for x in pos.split()], dtype=np.float64)
        if quat is None:
            geom_rot = np.eye(3, dtype=np.float64)
        else:
            wxyz = np.array([float(x) for x in quat.split()], dtype=np.float64)
            geom_rot = tf.SO3(wxyz=wxyz).as_matrix()
        mesh_name_to_transform[mesh_name] = (geom_pos, geom_rot)

    mesh_file_transforms = {}
    for mesh_file, mesh_name in mesh_file_to_mesh_name.items():
        geom_pos, geom_rot = mesh_name_to_transform.get(
            mesh_name,
            (np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
        )
        geom_pos = MUJOCO_TO_KIMODO @ geom_pos
        geom_rot = MUJOCO_TO_KIMODO @ geom_rot @ MUJOCO_TO_KIMODO.T
        mesh_file_transforms[mesh_file] = (geom_pos, geom_rot)

    mesh_transform_cache[mesh_dir] = {mf: (pos.copy(), rot.copy()) for mf, (pos, rot) in mesh_file_transforms.items()}
    return mesh_file_transforms


class G1MeshRig:
    """Rig for G1 STL meshes.

    Each instance has its own scene meshes (so clear() only removes one character). Loading is
    shared: STL files and g1.xml are cached per mesh_dir via _load_g1_mesh_data() and the class-
    level _mesh_*_cache dicts.
    """

    _mesh_geom_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    _mesh_transform_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        skeleton: G1Skeleton34,
        mesh_dir: str,
        color: Tuple[int, int, int],
    ):
        self.server = server
        self.skeleton = skeleton
        self.mesh_dir = mesh_dir
        self.color = color
        self.mesh_handles: list[viser.SceneHandle] = []
        self.mesh_items: list[dict[str, object]] = []
        self._defer_initial_visibility = True

        data_list = _load_g1_mesh_data(mesh_dir, skeleton)

        for item in data_list:
            mesh_file = item["mesh_file"]
            vertices = item["vertices"]
            faces = item["faces"]
            joint_idx = item["joint_idx"]
            geom_pos = item["geom_pos"]
            geom_rot = item["geom_rot"]

            handle = self.server.scene.add_mesh_simple(
                f"/{name}/g1_mesh/{os.path.splitext(mesh_file)[0]}",
                vertices=vertices,
                faces=faces,
                opacity=None,
                color=self.color,
                wireframe=False,
                visible=not self._defer_initial_visibility,
            )
            self.mesh_handles.append(handle)
            self.mesh_items.append(
                {
                    "handle": handle,
                    "joint_idx": joint_idx,
                    "geom_pos": geom_pos,
                    "geom_rot": geom_rot,
                }
            )

        if self._defer_initial_visibility:
            for handle in self.mesh_handles:
                handle.visible = True

    def set_visibility(self, visible: bool) -> None:
        for handle in self.mesh_handles:
            handle.visible = visible

    def set_opacity(self, opacity: float) -> None:
        for handle in self.mesh_handles:
            handle.opacity = opacity

    def set_wireframe(self, wireframe: bool) -> None:
        for handle in self.mesh_handles:
            handle.wireframe = wireframe

    def set_color(self, color: Tuple[int, int, int]) -> None:
        self.color = color
        for handle in self.mesh_handles:
            handle.color = color

    def set_pose(self, joints_pos: np.ndarray, joints_rot: np.ndarray) -> None:
        for item in self.mesh_items:
            handle = item["handle"]
            joint_idx = item["joint_idx"]
            geom_pos = item["geom_pos"]
            geom_rot = item["geom_rot"]

            joint_pos = joints_pos[joint_idx]
            joint_rot = joints_rot[joint_idx]
            mesh_pos = joint_pos + joint_rot @ geom_pos
            mesh_rot = joint_rot @ geom_rot

            handle.position = mesh_pos
            handle.wxyz = tf.SO3.from_matrix(mesh_rot).wxyz

    def clear(self) -> None:
        for handle in self.mesh_handles:
            self.server.scene.remove_by_name(handle.name)
        self.mesh_handles = []
        self.mesh_items = []
