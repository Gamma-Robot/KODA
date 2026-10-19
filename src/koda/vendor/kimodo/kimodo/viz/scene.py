# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viser scene entities: waypoints, skeleton mesh, and character."""

import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh

import viser
import viser.transforms as tf
from kimodo.skeleton import (
    G1Skeleton34,
    SkeletonBase,
    SMPLXSkeleton22,
    SOMASkeleton30,
    SOMASkeleton77,
)

from .coords import rotation_matrix_from_two_vec
from .g1_rig import (
    G1MeshRig,
)
from .smplx_skin import SMPLXSkin
from .soma_skin import SOMASkin


class WaypointMesh:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        position: np.ndarray,
        heading: Optional[np.ndarray] = None,
        color: Optional[Tuple[int, int, int]] = (255, 0, 0),
    ):
        self.server = server

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.025)
        annulus = trimesh.creation.annulus(r_min=0.1, r_max=0.2, height=0.005)

        z_to_y_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        annulus_vertices = annulus.vertices @ z_to_y_up

        self.sphere = self.server.scene.add_mesh_simple(
            name=f"{name}/sphere",
            vertices=sphere.vertices,
            faces=sphere.faces,
            position=position,
            color=color,
        )
        self.annulus = self.server.scene.add_mesh_simple(
            name=f"{name}/annulus",
            vertices=annulus_vertices,
            faces=annulus.faces,
            position=position,
            color=color,
        )

        self.arrow_base = None
        self.arrow_head = None
        if heading is not None:
            assert heading.shape == (2,), "Heading must be a 2D vector"
            heading = 0.3 * (heading / np.linalg.norm(heading))
            heading_3d = np.array([heading[0], 0, heading[1]])
            arrow_base = trimesh.creation.cylinder(radius=0.01, height=0.3)
            arrow_head = trimesh.creation.cone(radius=0.03, height=0.075)
            arrow_base_vertices = arrow_base.vertices
            arrow_head_vertices = arrow_head.vertices
            self.arrow_base = self.server.scene.add_mesh_simple(
                name=f"{name}/arrow_base",
                vertices=arrow_base_vertices,
                faces=arrow_base.faces,
                position=position + (heading_3d / 2),
                color=color,
            )
            self.arrow_head = self.server.scene.add_mesh_simple(
                name=f"{name}/arrow_head",
                vertices=arrow_head_vertices,
                faces=arrow_head.faces,
                position=position + heading_3d,
                color=color,
            )

    def update_position(self, position: np.ndarray, heading: Optional[np.ndarray] = None):
        self.sphere.position = position
        self.annulus.position = position
        if heading is not None:
            assert heading.shape == (2,), "Heading must be a 2D vector"
            heading = 0.3 * (heading / np.linalg.norm(heading))
            heading_3d = np.array([heading[0], 0, heading[1]])
            if self.arrow_base is not None:
                self.arrow_base.position = position + (heading_3d / 2)
            if self.arrow_head is not None:
                self.arrow_head.position = position + heading_3d

    def clear(self):
        self.server.scene.remove_by_name(self.sphere.name)
        self.server.scene.remove_by_name(self.annulus.name)
        if self.arrow_base is not None:
            self.server.scene.remove_by_name(self.arrow_base.name)
        if self.arrow_head is not None:
            self.server.scene.remove_by_name(self.arrow_head.name)

    def set_visible(self, visible: bool) -> None:
        self.sphere.visible = visible
        self.annulus.visible = visible
        if self.arrow_base is not None:
            self.arrow_base.visible = visible
        if self.arrow_head is not None:
            self.arrow_head.visible = visible


class SkeletonMesh:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        joint_color: Optional[Tuple[float, float, float] | np.ndarray] = (
            255,
            235,
            0,
        ),
        bone_color: Optional[Tuple[float, float, float] | np.ndarray] = (
            27,
            106,
            0,
        ),
        starting_joints_pos: Optional[torch.Tensor] = None,
    ):
        """
        name: str, name of the skeleton mesh
        server: viser.ViserServer, server to add the skeleton mesh to
        skeleton: SkeletonBase, skeleton to visualize
        joint_color: Optional[Tuple[float, float, float] | np.ndarray], color of the joints
        bone_color: Optional[Tuple[float, float, float] | np.ndarray], color of the bones
        starting_joints_pos: Optional[torch.Tensor], starting joint positions
        """
        self.server = server
        self.skeleton = skeleton
        joint_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.02)
        bone_mesh = trimesh.creation.cylinder(radius=0.01, height=1.0)

        init_joints_pos = skeleton.neutral_joints.clone()
        self.num_joints = init_joints_pos.shape[0]
        num_bones = self.num_joints - 1
        non_root_bones = [
            joint_name
            for joint_name, parent_name in self.skeleton.bone_order_names_with_parents
            if parent_name is not None
        ]
        self.bone_to_idx = {bone_name: idx for idx, bone_name in enumerate(non_root_bones)}

        # initialize meshes
        init_joints_wxyzs = np.concatenate([np.ones((self.num_joints, 1)), np.zeros((self.num_joints, 3))], axis=1)
        if isinstance(joint_color, tuple):
            self.joint_colors = np.full((self.num_joints, 3), joint_color)
        elif isinstance(joint_color, np.ndarray):
            assert joint_color.shape == (
                self.num_joints,
                3,
            ), "Joint colors must be (J, 3)"
            self.joint_colors = joint_color
        joint_scales = np.ones((self.num_joints, 3))
        hand_roots = {"LeftHand", "RightHand"}
        finger_joint_names = set(skeleton.left_hand_joint_names + skeleton.right_hand_joint_names) - hand_roots
        for jname in finger_joint_names:
            if jname in skeleton.bone_index:
                joint_scales[skeleton.bone_index[jname]] = 0.6
        self.joint_scales = joint_scales

        self.joints_batched_mesh = server.scene.add_batched_meshes_simple(
            f"{name}/joints",
            vertices=joint_mesh.vertices,
            faces=joint_mesh.faces,
            batched_wxyzs=init_joints_wxyzs,
            batched_positions=np.zeros((self.num_joints, 3)),
            batched_scales=joint_scales,
            batched_colors=self.joint_colors,
        )
        init_bones_wxyzs = np.concatenate([np.ones((num_bones, 1)), np.zeros((num_bones, 3))], axis=1)
        if isinstance(bone_color, tuple):
            bone_color = np.full((num_bones, 3), bone_color)
        elif isinstance(bone_color, np.ndarray):
            assert bone_color.shape == (num_bones, 3), "Bone colors must be (J-1, 3)"
            bone_color = bone_color
        self.bones_batched_mesh = server.scene.add_batched_meshes_simple(
            f"{name}/bones",
            vertices=bone_mesh.vertices,
            faces=bone_mesh.faces,
            batched_wxyzs=init_bones_wxyzs,
            batched_positions=np.zeros((num_bones, 3)),
            batched_scales=np.ones((num_bones, 3)),
            batched_colors=bone_color,
        )

        self.mesh_info_cache = None

        if starting_joints_pos is not None:
            self.set_pose(starting_joints_pos)
        else:
            if isinstance(skeleton, SOMASkeleton77):
                skel30 = SOMASkeleton30(load=True)
                min_height = skel30.neutral_joints[:, 1].min().item()
            else:
                min_height = init_joints_pos[:, 1].min().item()
            init_joints_pos[:, 1] -= min_height  # move to be on ground
            self.set_pose(init_joints_pos)

    def compute_single_pose(self, joints_pos: np.ndarray):
        """Compute the mesh for a single frame.

        joints_pos: [J, 3] global joint positions.
        """
        new_batched_positions = np.zeros((self.skeleton.nbjoints - 1, 3))
        new_batched_wxyzs = np.zeros((self.skeleton.nbjoints - 1, 4))
        new_batched_scales = np.ones((self.skeleton.nbjoints - 1, 3))
        for joint_name, parent_name in self.skeleton.bone_order_names_with_parents:
            if parent_name is None:
                continue
            joint_idx = self.skeleton.bone_index[joint_name]
            parent_idx = self.skeleton.bone_index[parent_name]
            joint_pos = joints_pos[joint_idx]
            parent_pos = joints_pos[parent_idx]

            bone_pos = (joint_pos + parent_pos) / 2.0
            bone_scale = np.linalg.norm(joint_pos - parent_pos)
            if bone_scale < 1e-8:
                bone_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                bone_dir = (joint_pos - parent_pos) / bone_scale
                R = rotation_matrix_from_two_vec(np.array([0.0, 0.0, 1.0], dtype=np.float64), bone_dir)
                bone_wxyz = tf.SO3.from_matrix(R).wxyz

            bone_idx = self.bone_to_idx[joint_name]
            new_batched_positions[bone_idx] = bone_pos
            new_batched_wxyzs[bone_idx] = bone_wxyz
            new_batched_scales[bone_idx] = np.array([1.0, 1.0, bone_scale], dtype=float)

        return new_batched_positions, new_batched_wxyzs, new_batched_scales

    def precompute_mesh_info(self, joints_pos: torch.Tensor):
        """Precompute the meshes for all frames at once.

        joints_pos: [T, J, 3].
        """
        joints_pos = joints_pos.cpu().numpy()
        num_frames = joints_pos.shape[0]
        self.mesh_info_cache = {
            "positions": np.zeros((num_frames, self.skeleton.nbjoints - 1, 3)),
            "wxyzs": np.zeros((num_frames, self.skeleton.nbjoints - 1, 4)),
            "scales": np.ones((num_frames, self.skeleton.nbjoints - 1, 3)),
        }
        for i in range(num_frames):
            new_batched_positions, new_batched_wxyzs, new_batched_scales = self.compute_single_pose(joints_pos[i])
            self.mesh_info_cache["positions"][i] = new_batched_positions
            self.mesh_info_cache["wxyzs"][i] = new_batched_wxyzs
            self.mesh_info_cache["scales"][i] = new_batched_scales

    def update_mesh_info_cache(self, joints_pos: torch.Tensor, frame_idx: int):
        """Update the mesh info cache for the given frame."""
        assert self.mesh_info_cache is not None
        new_batched_positions, new_batched_wxyzs, new_batched_scales = self.compute_single_pose(
            joints_pos.cpu().numpy()
        )
        self.mesh_info_cache["positions"][frame_idx] = new_batched_positions
        self.mesh_info_cache["wxyzs"][frame_idx] = new_batched_wxyzs
        self.mesh_info_cache["scales"][frame_idx] = new_batched_scales

    def set_pose(
        self,
        joints_pos: torch.Tensor,
        foot_contacts: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
    ):
        """Set pose from [J, 3] global joint positions."""
        self.cur_joints_pos = joints_pos
        joints_pos = joints_pos.cpu().numpy()

        if self.mesh_info_cache is not None:
            assert frame_idx is not None
            new_batched_positions = self.mesh_info_cache["positions"][frame_idx]
            new_batched_wxyzs = self.mesh_info_cache["wxyzs"][frame_idx]
            new_batched_scales = self.mesh_info_cache["scales"][frame_idx]
        else:
            new_batched_positions, new_batched_wxyzs, new_batched_scales = self.compute_single_pose(joints_pos)

        self.bones_batched_mesh.batched_positions = new_batched_positions
        self.bones_batched_mesh.batched_wxyzs = new_batched_wxyzs
        self.bones_batched_mesh.batched_scales = new_batched_scales
        self.joints_batched_mesh.batched_positions = joints_pos

        if foot_contacts is not None:
            cur_joint_colors = self.joint_colors.copy()
            foot_contacts = foot_contacts.bool().cpu().numpy().astype(bool)
            foot_joints = np.array(self.skeleton.foot_joint_idx, dtype=int)
            contact_idx = foot_joints[foot_contacts]
            cur_joint_colors[contact_idx] = (255, 0, 0)
            self.joints_batched_mesh.batched_colors = cur_joint_colors
        else:
            self.joints_batched_mesh.batched_colors = self.joint_colors

    def set_visibility(self, visible: bool):
        self.joints_batched_mesh.visible = visible
        self.bones_batched_mesh.visible = visible

    def get_pose(self) -> np.ndarray:
        return self.cur_joints_pos

    def clear(self):
        names = [mesh.name for mesh in [self.joints_batched_mesh, self.bones_batched_mesh]]
        for name in names:
            self.server.scene.remove_by_name(name)


LIGHT_THEME = dict(
    mesh=(152, 189, 255),
)

DARK_THEME = dict(
    mesh=(100, 135, 195),
)

SKIN_CACHE = {}


class Character:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        skeleton: SkeletonBase,
        create_skeleton_mesh: bool = True,
        create_skinned_mesh: bool = True,
        visible_skeleton: bool = False,
        visible_skinned_mesh: bool = True,
        skinned_mesh_opacity: float = 1.0,
        show_foot_contacts: bool = True,
        dark_mode: bool = False,
        mesh_mode: Optional[str] = None,
        gui_use_soma_layer_checkbox: Optional[viser.GuiCheckboxHandle] = None,
    ):
        self.server = server
        self.name = name
        self.skeleton = skeleton
        self.cur_joints_pos = None
        self.cur_joints_rot = None
        self.cur_foot_contacts = None

        self.skeleton_mesh = None
        self.show_foot_contacts = show_foot_contacts
        if create_skeleton_mesh:
            self.skeleton_mesh = SkeletonMesh(f"/{name}/skeleton", server, skeleton)
            self.cur_joints_pos = self.skeleton_mesh.get_pose()
            self.skeleton_mesh.set_visibility(visible_skeleton)

        self.skinned_mesh = None
        self.skin = None
        self.mesh_mode = mesh_mode
        self.g1_mesh_rig = None
        if create_skinned_mesh:
            if isinstance(self.skeleton, (SOMASkeleton30, SOMASkeleton77)) and mesh_mode in [
                "soma_skin",
                "soma_layer_skin",
            ]:
                if mesh_mode in SKIN_CACHE:
                    # already okay
                    pass
                else:
                    if mesh_mode == "soma_layer_skin":
                        try:
                            # try importing the lib
                            from .soma_layer_skin import SOMASkin as SOMASkin_SOMA

                            if mesh_mode not in SKIN_CACHE:
                                SKIN_CACHE[mesh_mode] = SOMASkin_SOMA(self.skeleton)

                        except (ModuleNotFoundError, FileNotFoundError) as e:
                            if isinstance(e, ModuleNotFoundError):
                                msg = "SOMA layer skin is unavailable: the soma package is not installed."
                            else:
                                msg = "SOMA layer skin is unavailable: SOMA asset files are missing."
                            traceback.print_exc()
                            if hasattr(self.server, "add_notification"):
                                self.server.add_notification(
                                    "SOMA layer skin unavailable",
                                    msg,
                                    auto_close_seconds=5.0,
                                    with_close_button=True,
                                )
                            if gui_use_soma_layer_checkbox is not None:
                                gui_use_soma_layer_checkbox.value = False
                            mesh_mode = "soma_skin"

                    # another if, in case mesh_mode changed
                    if mesh_mode == "soma_skin" and mesh_mode not in SKIN_CACHE:
                        SKIN_CACHE[mesh_mode] = SOMASkin(self.skeleton)

                self.skin = SKIN_CACHE[mesh_mode]
                self.skinned_mesh = server.scene.add_mesh_simple(
                    f"/{name}/simple_skinned",
                    vertices=self.skin.bind_vertices.cpu().numpy(),
                    faces=self.skin.faces.cpu().numpy(),
                    opacity=None,
                    color=LIGHT_THEME["mesh"] if not dark_mode else DARK_THEME["mesh"],
                    wireframe=False,
                    visible=False,
                )
                self.skinned_verts_cache = None

                bind_pos = self.skeleton.neutral_joints.clone()
                if isinstance(self.skeleton, SOMASkeleton77):
                    skel30 = SOMASkeleton30(load=True)
                    min_height = skel30.neutral_joints[:, 1].min().item()
                else:
                    min_height = bind_pos[:, 1].min().item()
                bind_pos[:, 1] -= min_height
                bind_pos[:, 1] += 0.02
                bind_rotmat = torch.eye(3, device=bind_pos.device).repeat(bind_pos.shape[0], 1, 1)
                self.set_pose(bind_pos, bind_rotmat)
                self.skinned_mesh.visible = True
                self.set_skinned_mesh_visibility(visible_skinned_mesh)
                self.set_skinned_mesh_opacity(skinned_mesh_opacity)
            elif isinstance(self.skeleton, SMPLXSkeleton22) and mesh_mode == "smplx_skin":
                if mesh_mode not in SKIN_CACHE:
                    SKIN_CACHE[mesh_mode] = SMPLXSkin(self.skeleton)
                self.skin = SKIN_CACHE[mesh_mode]
                self.skinned_mesh = server.scene.add_mesh_simple(
                    f"/{name}/simple_skinned",
                    vertices=self.skin.bind_vertices.cpu().numpy(),
                    faces=self.skin.faces.cpu().numpy(),
                    opacity=None,
                    color=LIGHT_THEME["mesh"] if not dark_mode else DARK_THEME["mesh"],
                    wireframe=False,
                    visible=False,
                )
                self.skinned_verts_cache = None

                bind_pos = self.skeleton.neutral_joints.clone()
                min_height = bind_pos[:, 1].min().item()
                bind_pos[:, 1] -= min_height
                bind_rotmat = torch.eye(3, device=bind_pos.device).repeat(bind_pos.shape[0], 1, 1)
                self.set_pose(bind_pos, bind_rotmat)
                self.skinned_mesh.visible = True
                self.set_skinned_mesh_visibility(visible_skinned_mesh)
                self.set_skinned_mesh_opacity(skinned_mesh_opacity)
            elif isinstance(self.skeleton, G1Skeleton34) and mesh_mode == "g1_stl":
                g1_mesh_dir = Path(self.skeleton.folder) / "meshes/g1"
                if not os.path.exists(g1_mesh_dir):
                    raise ValueError(f"G1 mesh directory not found: {g1_mesh_dir}")
                self.g1_mesh_rig = G1MeshRig(
                    name,
                    server,
                    self.skeleton,
                    str(g1_mesh_dir),
                    DARK_THEME["mesh"] if dark_mode else LIGHT_THEME["mesh"],
                )
                init_joints_rot = self.skeleton.rest_pose_local_rot.clone()
                init_global_joint_rots, _, init_joints_pos = self.skeleton.fk(
                    init_joints_rot,
                    torch.zeros(3, device=init_joints_rot.device, dtype=init_joints_rot.dtype),
                )
                min_height = init_joints_pos[:, 1].min().item()
                init_joints_pos[:, 1] -= min_height
                self.set_pose(init_joints_pos, init_global_joint_rots)
                self.set_skinned_mesh_visibility(visible_skinned_mesh)
                self.set_skinned_mesh_opacity(skinned_mesh_opacity)
            else:
                raise ValueError(
                    "Unsupported mesh mode for skeleton type: "
                    f"{type(self.skeleton).__name__} with mesh_mode={mesh_mode}"
                )

    def change_theme(self, is_dark_mode):
        color = DARK_THEME["mesh"] if is_dark_mode else LIGHT_THEME["mesh"]
        if self.skinned_mesh is not None:
            self.skinned_mesh.color = color
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_color(color)

    def set_skeleton_visibility(self, visible: bool):
        if self.skeleton_mesh is not None:
            self.skeleton_mesh.set_visibility(visible)

    def set_show_foot_contacts(self, show: bool, frame_idx: Optional[int] = None):
        self.show_foot_contacts = show
        if self.skeleton_mesh is not None and self.cur_joints_pos is not None:
            fc = self.cur_foot_contacts if show else None
            self.skeleton_mesh.set_pose(self.cur_joints_pos, foot_contacts=fc, frame_idx=frame_idx)

    def set_skinned_mesh_visibility(self, visible: bool):
        if self.skinned_mesh is not None:
            self.skinned_mesh.visible = visible
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_visibility(visible)

    def set_skinned_mesh_opacity(self, opacity: float):
        if self.skinned_mesh is not None:
            self.skinned_mesh.opacity = opacity
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_opacity(opacity)

    def set_skinned_mesh_wireframe(self, wireframe: bool):
        if self.skinned_mesh is not None:
            self.skinned_mesh.wireframe = wireframe
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.set_wireframe(wireframe)

    def precompute_skinning(self, joints_pos: torch.Tensor, joints_rot: torch.Tensor, chunk_size: int = 64):
        """Precompute skinning for all frames, processing in chunks to avoid OOM.

        joints_pos: [T, J, 3], joints_rot: [T, J, 3, 3].

        The LBS gather intermediate is ~V*W*48 bytes per frame (V=18k, W=8 for SOMA
        gives ~7 MB/frame), so a chunk of 64 peaks around ~1 GB -- safe alongside
        a loaded text encoder + diffusion model on a typical 24 GB GPU.
        """
        assert self.skin is not None
        T = joints_pos.shape[0]
        with torch.no_grad():
            if T <= chunk_size:
                self.skinned_verts_cache = self.skin.skin(joints_rot, joints_pos, rot_is_global=True).cpu().numpy()
            else:
                chunks = []
                for start in range(0, T, chunk_size):
                    end = min(start + chunk_size, T)
                    verts = self.skin.skin(joints_rot[start:end], joints_pos[start:end], rot_is_global=True).cpu().numpy()
                    chunks.append(verts)
                self.skinned_verts_cache = np.concatenate(chunks, axis=0)

    def update_skinning_cache(self, joints_pos: torch.Tensor, joints_rot: torch.Tensor, frame_idx: int):
        """Update skinning cache for one frame."""
        if self.skinned_verts_cache is None:
            return
        with torch.no_grad():
            new_skinned_verts = self.skin.skin(joints_rot[None], joints_pos[None], rot_is_global=True)[0].cpu().numpy()
        self.skinned_verts_cache[frame_idx] = new_skinned_verts

    def set_pose(
        self,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
        foot_contacts: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
    ):
        if self.skeleton_mesh is not None:
            self.cur_foot_contacts = foot_contacts
            display_fc = foot_contacts if self.show_foot_contacts else None
            self.skeleton_mesh.set_pose(joints_pos, foot_contacts=display_fc, frame_idx=frame_idx)

        if self.skinned_mesh is not None:
            if self.skinned_verts_cache is not None:
                assert frame_idx is not None
                skinned_verts = self.skinned_verts_cache[frame_idx]
            else:
                with torch.no_grad():
                    skinned_verts = self.skin.skin(joints_rot[None], joints_pos[None], rot_is_global=True)[0].cpu().numpy()
            self.skinned_mesh.vertices = skinned_verts
        if self.g1_mesh_rig is not None:
            joints_pos_np = joints_pos.detach().cpu().numpy()
            joints_rot_np = joints_rot.detach().cpu().numpy()
            self.g1_mesh_rig.set_pose(joints_pos_np, joints_rot_np)

        self.cur_joints_pos = joints_pos
        self.cur_joints_rot = joints_rot

    def get_pose(self) -> torch.Tensor:
        return self.cur_joints_pos, self.cur_joints_rot

    def clear(self):
        if self.skeleton_mesh is not None:
            self.skeleton_mesh.clear()
        if self.skinned_mesh is not None:
            self.server.scene.remove_by_name(self.skinned_mesh.name)
        if self.g1_mesh_rig is not None:
            self.g1_mesh_rig.clear()
