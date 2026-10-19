# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Playback and motion editing: CharacterMotion."""

from typing import Callable, Literal, Optional

import numpy as np
import torch

import viser.transforms as tf
from kimodo.skeleton import (
    G1Skeleton34,
    SOMASkeleton30,
    SOMASkeleton77,
    batch_rigid_transform,
    global_rots_to_local_rots,
)
from kimodo.tools import to_numpy, to_torch

from .g1_rig import (
    _get_g1_joint_axis_indices,
    _get_g1_joint_limits,
    get_g1_joint_f2q_data,
)
from .scene import Character


class CharacterMotion:
    def __init__(
        self,
        character: Character,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
        foot_contacts: Optional[torch.Tensor] = None,
    ):
        self.character = character
        self.server = character.server
        self.skeleton = character.skeleton
        self.name = character.name

        # [T, J, 3] global joint positions
        self.joints_pos = joints_pos
        # [T, J, 3, 3] global joint rotation matrices
        self.joints_rot = joints_rot
        assert joints_pos.shape[0] == joints_rot.shape[0]
        # keep track of local rots as well for convenience during pose editing
        self.joints_local_rot = global_rots_to_local_rots(joints_rot, self.skeleton)

        self.length = joints_pos.shape[0]
        self.cur_frame_idx = None

        self.foot_contacts = foot_contacts
        if foot_contacts is not None:
            assert foot_contacts.shape[0] == self.length

        self.precompute_mesh_info()

        # gizmos for pose editing
        self.root_translation_gizmo = None
        self.updating_root_translation_gizmo = False
        self.joint_gizmos = None
        self.updating_joint_gizmos = False
        self.gizmo_space: Literal["world", "local"] = "local"
        self._drag_start_world_rot: list = []
        self._joint_gizmo_dragging: list[bool] = []

    def precompute_mesh_info(self):
        if self.character.skeleton_mesh is not None:
            print("Caching skeleton mesh info...")
            self.character.skeleton_mesh.precompute_mesh_info(self.joints_pos)
        if self.character.skinned_mesh is not None:
            print("Caching skinning info...")
            self.character.precompute_skinning(self.joints_pos, self.joints_rot)

    def set_frame(self, idx: int):
        """Sets the pose of the character to the given frame index."""
        idx = min(idx, self.length - 1)  # clamp to last frame
        cur_foot_contacts = self.foot_contacts[idx] if self.foot_contacts is not None else None
        self.character.set_pose(
            self.joints_pos[idx],
            self.joints_rot[idx],
            frame_idx=idx,
            foot_contacts=cur_foot_contacts,
        )
        self.cur_frame_idx = idx

        # update gizmos if frame has changed due to playback
        cur_root_pos = self.joints_pos[self.cur_frame_idx, self.skeleton.root_idx].clone()
        cur_root_pos[1] = 0.0
        if self.root_translation_gizmo is not None and not self.updating_root_translation_gizmo:
            self.root_translation_gizmo.position = cur_root_pos.cpu().numpy()
        if self.joint_gizmos is not None:
            for i, joint_gizmo in enumerate(self.joint_gizmos):
                # Do not push wxyz/position while this gizmo is being dragged;
                # otherwise the client receives e.g. identity and the gizmo snaps back.
                if not self.updating_joint_gizmos and not self._joint_gizmo_dragging[i]:
                    joint_gizmo.position = self.joints_pos[self.cur_frame_idx, i].cpu().numpy()
                    if self.gizmo_space == "world":
                        joint_gizmo.wxyz = (1.0, 0.0, 0.0, 0.0)
                    else:
                        joint_gizmo.wxyz = tf.SO3.from_matrix(self.joints_rot[self.cur_frame_idx, i].cpu().numpy()).wxyz

    def update_pose_at_frame(
        self,
        frame_idx: int,
        joints_pos: Optional[torch.Tensor] = None,
        joints_rot: Optional[torch.Tensor] = None,
        joints_local_rot: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ):
        """Overwrites one or more of the pose components at the given frame.

        If only a subset of joints_pos, joints_rot, or joints_local_rot are provided, the other
        components will be updated with FK.
        """
        if joints_pos is not None:
            joints_pos = to_torch(joints_pos, device=self.joints_pos.device, dtype=self.joints_pos.dtype)
            self.joints_pos[frame_idx] = joints_pos
            if joints_local_rot is None and joints_rot is None:
                raise NotImplementedError("No IK to update joint rotations accordingly.")
        if joints_rot is not None:
            joints_rot = to_torch(joints_rot, device=self.joints_rot.device, dtype=self.joints_rot.dtype)
            self.joints_rot[frame_idx] = joints_rot
            if joints_local_rot is None:
                # update local rots from global rots
                self.joints_local_rot[frame_idx] = global_rots_to_local_rots(joints_rot, self.skeleton)
            if joints_pos is None:
                # need to update with FK
                new_posed_joints, _ = batch_rigid_transform(
                    self.joints_local_rot[frame_idx : frame_idx + 1],
                    self.skeleton.neutral_joints[None].to(self.joints_local_rot.device),
                    self.skeleton.joint_parents.to(self.joints_local_rot.device),
                    self.skeleton.root_idx,
                )
                new_posed_joints = (
                    new_posed_joints[0]
                    + self.joints_pos[frame_idx, self.skeleton.root_idx : self.skeleton.root_idx + 1]
                    - self.skeleton.neutral_joints[[self.skeleton.root_idx]]
                )
                self.joints_pos[frame_idx] = new_posed_joints
        if joints_local_rot is not None:
            joints_local_rot = to_torch(joints_local_rot, device=self.joints_local_rot.device).to(
                dtype=self.joints_local_rot.dtype
            )
            self.joints_local_rot[frame_idx] = joints_local_rot
            if joints_rot is None or joints_pos is None:
                # need to update with FK
                new_posed_joints, new_global_rots = batch_rigid_transform(
                    self.joints_local_rot[frame_idx : frame_idx + 1],
                    self.skeleton.neutral_joints[None].to(self.joints_local_rot.device),
                    self.skeleton.joint_parents.to(self.joints_local_rot.device),
                    self.skeleton.root_idx,
                )
                new_posed_joints = (
                    new_posed_joints[0]
                    + self.joints_pos[frame_idx, self.skeleton.root_idx : self.skeleton.root_idx + 1]
                    - self.skeleton.neutral_joints[[self.skeleton.root_idx]]
                )
                if joints_rot is None:
                    self.joints_rot[frame_idx] = new_global_rots[0]
                if joints_pos is None:
                    self.joints_pos[frame_idx] = new_posed_joints
        if foot_contacts is not None:
            foot_contacts = to_torch(foot_contacts, device=self.foot_contacts.device).to(dtype=self.foot_contacts.dtype)
            self.foot_contacts[frame_idx] = foot_contacts

        if self.character.skeleton_mesh is not None:
            self.character.skeleton_mesh.update_mesh_info_cache(self.joints_pos[frame_idx], frame_idx)
        if self.character.skinned_mesh is not None:
            self.character.update_skinning_cache(self.joints_pos[frame_idx], self.joints_rot[frame_idx], frame_idx)

    def clear(self):
        self.character.clear()

    #
    # Editing helpers
    #
    def get_current_projected_root_pos(self) -> np.ndarray:
        """Get the projected root position on the ground at the current frame."""
        root_pos = self.joints_pos[self.cur_frame_idx, self.skeleton.root_idx].clone()
        root_pos[1] = 0.0
        return to_numpy(root_pos)

    def get_projected_root_pos(self, start_frame_idx: int, end_frame_idx: int = None) -> np.ndarray:
        """If requested frames are out of range, simply pads with the last frame to get expected
        length."""
        if end_frame_idx is None:
            expected_len = 1
        else:
            expected_len = end_frame_idx - start_frame_idx + 1
        if start_frame_idx >= self.length:
            start_frame_idx = self.length - 1
        if end_frame_idx is None or expected_len == 1:
            root_pos = self.joints_pos[start_frame_idx, self.skeleton.root_idx].clone()
            root_pos[1] = 0.0
            return to_numpy(root_pos)
        else:
            if end_frame_idx >= self.length:
                end_frame_idx = self.length - 1
            root_pos = self.joints_pos[start_frame_idx : end_frame_idx + 1, self.skeleton.root_idx].clone()
            root_pos[:, 1] = 0.0
            if root_pos.shape[0] < expected_len:
                # pad with the last root position
                root_pos = torch.cat(
                    [
                        root_pos,
                        root_pos[-1:].repeat(expected_len - root_pos.shape[0], 1),
                    ],
                    dim=0,
                )
            return to_numpy(root_pos)

    def set_projected_root_pos_path(
        self,
        root_pos_path: np.ndarray | torch.Tensor,
        min_frame_idx: int = None,
        max_frame_idx: int = None,
    ):
        """Sets the projected root position path for the character motion. Can set only a subset of
        the path by providing min_frame_idx and max_frame_idx. If not provided, will set the full
        path.

        Args:
            root_pos_path: torch.Tensor, [T, 2] projected root positions
            min_frame_idx: int, optional, minimum frame index to set the path at
            max_frame_idx: int, optional, maximum frame index to set the path at
        """
        if min_frame_idx is not None or max_frame_idx is not None:
            assert (
                min_frame_idx is not None and max_frame_idx is not None
            ), "min_frame_idx and max_frame_idx must be provided if setting path at specific frames"
            if min_frame_idx >= self.length:
                # both are out of bounds
                return
            max_frame_idx = min(max_frame_idx, self.length - 1)
            root_pos_path = root_pos_path[min_frame_idx : max_frame_idx + 1]
        else:
            assert root_pos_path.shape[0] == self.length
            min_frame_idx = 0
            max_frame_idx = self.length - 1

        cur_joints_pos = self.joints_pos.clone()[min_frame_idx : max_frame_idx + 1]
        root_pos_tensor = to_torch(root_pos_path, device=cur_joints_pos.device, dtype=cur_joints_pos.dtype)
        diff = root_pos_tensor - cur_joints_pos[:, self.skeleton.root_idx, [0, 2]]
        cur_joints_pos[:, :, [0, 2]] += diff.unsqueeze(1)
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            rel_idx = frame_idx - min_frame_idx
            self.update_pose_at_frame(
                frame_idx,
                joints_pos=cur_joints_pos[rel_idx],
                joints_rot=self.joints_rot[frame_idx],
                joints_local_rot=self.joints_local_rot[frame_idx],
            )
        # update immediately to show changes
        self.set_frame(self.cur_frame_idx)

    def get_joints_pos(self, start_frame_idx: int, end_frame_idx: int = None) -> np.ndarray:
        """If requested frames are out of range, simply pads with the last frame to get expected
        length."""
        if end_frame_idx is None:
            expected_len = 1
        else:
            expected_len = end_frame_idx - start_frame_idx + 1
        if start_frame_idx >= self.length:
            start_frame_idx = self.length - 1
        if end_frame_idx is None or expected_len == 1:
            return to_numpy(self.joints_pos[start_frame_idx].clone())
        else:
            if end_frame_idx >= self.length:
                end_frame_idx = self.length - 1
            return_joints_pos = self.joints_pos[start_frame_idx : end_frame_idx + 1].clone()
            if return_joints_pos.shape[0] < expected_len:
                # pad with the last pose
                return_joints_pos = torch.cat(
                    [
                        return_joints_pos,
                        return_joints_pos[-1:].repeat(expected_len - return_joints_pos.shape[0], 1, 1),
                    ],
                    dim=0,
                )
            return to_numpy(return_joints_pos)

    def get_joints_rot(self, start_frame_idx: int, end_frame_idx: int = None) -> np.ndarray:
        """If requested frames are out of range, simply pads with the last frame to get expected
        length."""
        if end_frame_idx is None:
            expected_len = 1
        else:
            expected_len = end_frame_idx - start_frame_idx + 1
        if start_frame_idx >= self.length:
            start_frame_idx = self.length - 1
        if end_frame_idx is None or expected_len == 1:
            return to_numpy(self.joints_rot[start_frame_idx].clone())
        else:
            if end_frame_idx >= self.length:
                end_frame_idx = self.length - 1
            return_joints_rot = self.joints_rot[start_frame_idx : end_frame_idx + 1].clone()
            if return_joints_rot.shape[0] < expected_len:
                # pad with the last pose
                return_joints_rot = torch.cat(
                    [
                        return_joints_rot,
                        return_joints_rot[-1:].repeat(expected_len - return_joints_rot.shape[0], 1, 1, 1),
                    ],
                    dim=0,
                )
            return to_numpy(return_joints_rot)

    def get_current_joints_pos(self) -> torch.Tensor:
        return self.joints_pos[self.cur_frame_idx].clone()

    def get_current_joints_rot(self) -> torch.Tensor:
        return self.joints_rot[self.cur_frame_idx].clone()

    def add_root_translation_gizmo(
        self,
        constraints: dict,
        on_2d_root_drag_end: Optional[Callable[[], None]] = None,
        on_drag_start: Optional[Callable[[], None]] = None,
    ):
        """Create and initialize gizmo to control the root translation.

        When the user drags the root 2D gizmo, path updates are skipped until release. Optional
        on_2d_root_drag_end is called when the drag ends (e.g. to refresh dense path). on_drag_start
        is called when the drag begins (e.g. to snapshot state for undo).
        """
        # TODO: could also allow rotation around y-axis
        self.root_translation_gizmo = self.server.scene.add_transform_controls(
            f"/{self.name}/gizmo_root_translation",
            scale=0.5,
            line_width=2.5,
            active_axes=(True, False, True),  # only allow translation on xz plane
            disable_axes=False,
            disable_sliders=False,
            disable_rotations=True,
            depth_test=False,  # render even when occluded
        )
        init_position = self.get_current_projected_root_pos()
        self.root_translation_gizmo.position = init_position

        @self.root_translation_gizmo.on_drag_start
        def _(_):
            if on_drag_start is not None:
                on_drag_start()

        @self.root_translation_gizmo.on_update
        def _(_):
            self.updating_root_translation_gizmo = True
            # translate to gizmo position
            new_root_pos = to_torch(
                self.root_translation_gizmo.position,
                device=self.joints_pos.device,
            ).to(dtype=self.joints_pos.dtype)
            cur_joints_pos = self.joints_pos[self.cur_frame_idx].clone()
            root_diff = new_root_pos - cur_joints_pos[self.skeleton.root_idx]
            root_diff[1] = 0.0  # don't change height
            cur_joints_pos += root_diff[None]
            self.update_pose_at_frame(
                self.cur_frame_idx,
                joints_pos=cur_joints_pos,
                joints_rot=self.joints_rot[self.cur_frame_idx],
                joints_local_rot=self.joints_local_rot[self.cur_frame_idx],
            )

            self.updating_root_translation_gizmo = False
            # update immediately to show user changes
            self.set_frame(self.cur_frame_idx)
            # update the 2D waypoint constraints as well if there is one
            if "2D Root" in constraints:
                root_2d_contraints = constraints["2D Root"]
                # if there is a constraint at that frame, we want to update it
                frame_idx = self.cur_frame_idx
                if frame_idx in root_2d_contraints.keyframes:
                    for keyframe_id in root_2d_contraints.frame2keyid[frame_idx]:
                        # add will modify the existing constraint
                        # update_path=False during drag to avoid lag; path refreshes on_drag_end
                        root_2d_contraints.add_keyframe(
                            keyframe_id,
                            frame_idx,
                            root_pos=new_root_pos,
                            exists_ok=True,
                            update_path=False,
                        )
            if "Full-Body" in constraints:
                full_body_constraints = constraints["Full-Body"]
                # if there is a constraint at that frame, we want to update it
                frame_idx = self.cur_frame_idx
                if frame_idx in full_body_constraints.keyframes:
                    current_dict = full_body_constraints.keyframes[frame_idx]
                    for keyframe_id in full_body_constraints.frame2keyid[frame_idx]:
                        # add will modify the existing constraint
                        full_body_constraints.add_keyframe(
                            keyframe_id,
                            frame_idx,
                            joints_pos=cur_joints_pos,
                            joints_rot=current_dict["joints_rot"],
                            exists_ok=True,
                        )
            if "End-Effectors" in constraints:
                end_effector_constraints = constraints["End-Effectors"]
                # if there is a constraint at that frame, we want to update it
                frame_idx = self.cur_frame_idx
                if frame_idx in end_effector_constraints.keyframes:
                    current_dict = end_effector_constraints.keyframes[frame_idx]
                    for keyframe_id, _ in end_effector_constraints.frame2keyid[frame_idx]:
                        # add will modify the existing constraint
                        end_effector_constraints.add_keyframe(
                            keyframe_id,
                            frame_idx,
                            joints_pos=cur_joints_pos,
                            joints_rot=current_dict["joints_rot"],
                            joint_names=current_dict["joint_names"],
                            end_effector_type=current_dict["end_effector_type"],
                            exists_ok=True,
                        )

        @self.root_translation_gizmo.on_drag_end
        def _on_drag_end(_):
            # Refresh path visualization and dense path after release.
            if "2D Root" in constraints:
                root_2d = constraints["2D Root"]
                if root_2d.line_segments is not None:
                    root_2d.update_line_segments()
            if on_2d_root_drag_end is not None:
                on_2d_root_drag_end()

    def add_joint_gizmos(
        self,
        constraints: dict,
        space: Literal["world", "local"] = "local",
        on_drag_start: Optional[Callable[[], None]] = None,
    ):
        # Remove existing joint gizmos first so the client gets remove then add,
        # avoiding in-place update that can briefly show duplicate gizmos.
        if self.joint_gizmos is not None:
            for joint_gizmo in self.joint_gizmos:
                self.server.scene.remove_by_name(joint_gizmo.name)
            self.joint_gizmos = None

        self.joint_gizmos = []
        self.gizmo_space = space
        # For world mode: store joint world rotation at drag start to compose with
        # PivotControls' cumulative-from-identity drag rotation.
        self._drag_start_world_rot = [None] * self.skeleton.nbjoints
        # Skip pushing wxyz/position in set_frame while a gizmo is being dragged,
        # so the client does not receive "snap back" (e.g. identity for world mode).
        self._joint_gizmo_dragging = [False] * self.skeleton.nbjoints

        joint_axis_indices = None
        joint_limits = None
        joint_f2q_data = None
        hidden_gizmo_joints = None
        if isinstance(self.skeleton, G1Skeleton34):
            joint_axis_indices = _get_g1_joint_axis_indices()
            joint_limits = _get_g1_joint_limits()
            joint_f2q_data = get_g1_joint_f2q_data(self.skeleton)
            hidden_gizmo_joints = {
                "left_hand_roll_skel",
                "right_hand_roll_skel",
                "left_toe_base",
                "right_toe_base",
            }
        elif isinstance(self.skeleton, SOMASkeleton77):
            skel30_names = {name for name, _ in SOMASkeleton30.bone_order_names_with_parents}
            hidden_gizmo_joints = {name for name in self.skeleton.bone_order_names if name not in skel30_names}
            hidden_gizmo_joints |= {
                "RightHandThumbEnd",
                "RightHandMiddleEnd",
                "LeftHandThumbEnd",
                "LeftHandMiddleEnd",
                "LeftEye",
                "RightEye",
                "Jaw",
            }
        elif isinstance(self.skeleton, SOMASkeleton30):
            hidden_gizmo_joints = {
                "RightHandThumbEnd",
                "RightHandMiddleEnd",
                "LeftHandThumbEnd",
                "LeftHandMiddleEnd",
                "LeftEye",
                "RightEye",
                "Jaw",
            }

        if space == "world":
            # World mode: gizmo rings stay scene-axis-aligned (identity).
            joints_wxyzs = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                (self.skeleton.nbjoints, 1),
            )
        else:
            # Local mode: gizmo shows joint world rotation so rings follow the joint.
            joints_wxyzs = tf.SO3.from_matrix(self.joints_rot[self.cur_frame_idx].cpu().numpy()).wxyz
        for joint_idx in range(self.skeleton.nbjoints):
            disable_axes = True  # by default, only rotation controls
            disable_sliders = True
            if joint_idx == self.skeleton.root_idx:
                disable_axes = False  # allow translation for root
                disable_sliders = False
            active_axes = (True, True, True)
            if joint_axis_indices is not None:
                joint_name = self.skeleton.bone_order_names[joint_idx]
                axis_idx = joint_axis_indices.get(joint_name)
                if axis_idx is not None:
                    # PivotControls shows rotation handles when a plane is active.
                    # To allow rotation about one axis, enable the other two axes.
                    active_axes = (
                        axis_idx != 0,
                        axis_idx != 1,
                        axis_idx != 2,
                    )
            joint_visible = True
            if hidden_gizmo_joints is not None:
                joint_name = self.skeleton.bone_order_names[joint_idx]
                joint_visible = joint_name not in hidden_gizmo_joints
            cur_joint_gizmo = self.server.scene.add_transform_controls(
                f"/{self.name}/gizmo_joint_{joint_idx}",
                scale=0.075,
                line_width=4.0,
                active_axes=active_axes,
                disable_axes=disable_axes,
                disable_sliders=disable_sliders,
                disable_rotations=False,
                depth_test=False,  # render even when occluded
                position=self.joints_pos[self.cur_frame_idx, joint_idx].cpu().numpy(),
                wxyz=joints_wxyzs[joint_idx],
                visible=joint_visible,
                space=space,
            )
            self.joint_gizmos.append(cur_joint_gizmo)

            def set_callback_in_closure(i: int) -> None:
                @cur_joint_gizmo.on_drag_start
                def _on_drag_start(_) -> None:
                    if on_drag_start is not None:
                        on_drag_start()
                    self._joint_gizmo_dragging[i] = True
                    if self.gizmo_space == "world":
                        self._drag_start_world_rot[i] = self.joints_rot[self.cur_frame_idx, i].clone().cpu().numpy()

                @cur_joint_gizmo.on_drag_end
                def _on_drag_end(_) -> None:
                    self._joint_gizmo_dragging[i] = False
                    # Force-sync so the client always receives the reset (viser setter skips on allclose).
                    # Use self.joint_gizmos[i] (not cur_joint_gizmo) to avoid the
                    # closure-in-loop bug: cur_joint_gizmo would point to the last handle.
                    gizmo = self.joint_gizmos[i]
                    gizmo.sync_position(self.joints_pos[self.cur_frame_idx, i].cpu().numpy())
                    if self.gizmo_space == "world":
                        gizmo.sync_wxyz((1.0, 0.0, 0.0, 0.0))
                    else:
                        gizmo.sync_wxyz(tf.SO3.from_matrix(self.joints_rot[self.cur_frame_idx, i].cpu().numpy()).wxyz)
                    self.set_frame(self.cur_frame_idx)

                @cur_joint_gizmo.on_update
                def _(_) -> None:
                    self.updating_joint_gizmos = True
                    new_local_joint_rots = self.joints_local_rot[self.cur_frame_idx].clone()
                    # Gizmo parent is identity; client sends rotation as wxyz.
                    # World mode: wxyz is cumulative from identity, compose with
                    # stored initial world rotation. Local mode: wxyz is new world rotation.
                    gizmo_rot_mat = tf.SO3(self.joint_gizmos[i].wxyz).as_matrix()
                    if self.gizmo_space == "world" and self._drag_start_world_rot[i] is not None:
                        new_world_rot_mat = gizmo_rot_mat @ self._drag_start_world_rot[i]
                    else:
                        new_world_rot_mat = gizmo_rot_mat
                    parent_idx = self.skeleton.joint_parents[i].item()
                    if parent_idx >= 0:
                        R_parent_world = self.joints_rot[self.cur_frame_idx, parent_idx].detach().cpu().numpy()
                        new_local_rot_mat_np = (R_parent_world.T @ new_world_rot_mat).astype(np.float32)
                    else:
                        new_local_rot_mat_np = new_world_rot_mat.astype(np.float32)
                    new_local_rot = tf.SO3.from_matrix(new_local_rot_mat_np)
                    joint_name = self.skeleton.bone_order_names[i]
                    if joint_f2q_data is not None and joint_name in joint_f2q_data:
                        # G1 hinge: use offset (f2q) space so 1-DoF and limits match the robot.
                        # R_f2q = offset_f2q @ R_local; angle_f2q = dot(axis_angle(R_f2q), axis_f2q);
                        # MuJoCo q = angle_f2q - rest_dof; limits apply to q.
                        f2q = joint_f2q_data[joint_name]
                        offset_f2q = f2q["offset_f2q"]
                        axis_f2q = f2q["axis_f2q"]
                        rest_dof = f2q["rest_dof_axis_angle"]
                        R_local = new_local_rot_mat_np.astype(np.float64)
                        R_f2q = offset_f2q @ R_local
                        rotvec = tf.SO3.from_matrix(R_f2q).log()
                        angle_f2q = float(np.dot(rotvec, axis_f2q))
                        # Keep angle continuous relative to current pose.
                        current_R_f2q = offset_f2q @ (
                            self.joints_local_rot[self.cur_frame_idx, i].detach().cpu().numpy().astype(np.float64)
                        )
                        current_angle_f2q = float(np.dot(tf.SO3.from_matrix(current_R_f2q).log(), axis_f2q))
                        two_pi = 2.0 * np.pi
                        angle_f2q = angle_f2q + two_pi * np.round((current_angle_f2q - angle_f2q) / two_pi)
                        q = angle_f2q - rest_dof
                        if joint_limits is not None:
                            joint_limit = joint_limits.get(joint_name)
                            if joint_limit is not None:
                                q = float(np.clip(q, joint_limit[0], joint_limit[1]))
                        angle_f2q = q + rest_dof
                        R_f2q_new = tf.SO3.exp(angle_f2q * axis_f2q).as_matrix()
                        new_local_rot_mat_np = (offset_f2q.T @ R_f2q_new).astype(np.float32)
                    elif joint_axis_indices is not None:
                        axis_idx = joint_axis_indices.get(joint_name)
                        if axis_idx is not None:
                            rotvec = new_local_rot.log()
                            axis = np.zeros(3, dtype=np.float64)
                            axis[axis_idx] = 1.0
                            angle = float(rotvec[axis_idx])
                            # Keep angle continuous relative to current pose.
                            current_rot = tf.SO3.from_matrix(
                                self.joints_local_rot[self.cur_frame_idx, i].detach().cpu().numpy()
                            )
                            current_angle = float(current_rot.log()[axis_idx])
                            two_pi = 2.0 * np.pi
                            angle = angle + two_pi * np.round((current_angle - angle) / two_pi)
                            if joint_limits is not None:
                                joint_limit = joint_limits.get(joint_name)
                                if joint_limit is not None:
                                    angle = float(np.clip(angle, joint_limit[0], joint_limit[1]))
                            new_local_rot_mat_np = tf.SO3.exp(angle * axis).as_matrix()
                    new_local_rot_mat = torch.tensor(new_local_rot_mat_np).to(new_local_joint_rots.device)
                    new_local_joint_rots[i] = new_local_rot_mat

                    self.update_pose_at_frame(
                        self.cur_frame_idx,
                        joints_local_rot=new_local_joint_rots,
                    )

                    # handle root translation separately
                    cur_joints_pos = self.joints_pos[self.cur_frame_idx].clone()
                    if i == self.skeleton.root_idx:
                        new_root_pos = to_torch(
                            self.joint_gizmos[i].position,
                            device=self.joints_pos.device,
                        ).to(dtype=self.joints_pos.dtype)
                        root_diff = new_root_pos - self.joints_pos[self.cur_frame_idx, i]
                        if torch.norm(root_diff) > 1e-3:
                            # the root translation has been changed
                            # translate to gizmo position
                            cur_joints_pos += root_diff[None]
                            self.update_pose_at_frame(
                                self.cur_frame_idx,
                                joints_pos=cur_joints_pos,
                                joints_rot=self.joints_rot[self.cur_frame_idx],
                                joints_local_rot=self.joints_local_rot[self.cur_frame_idx],
                            )

                    # update immediately to show user changes. Keep updating_joint_gizmos
                    # True so set_frame does not overwrite gizmo wxyz mid-drag.
                    self.set_frame(self.cur_frame_idx)
                    self.updating_joint_gizmos = False

                    if i == self.skeleton.root_idx:
                        # update the 2D waypoint constraints as well if there is one
                        if "2D Root" in constraints:
                            root_2d_contraints = constraints["2D Root"]
                            # if there is a constraint at that frame, we want to update it
                            frame_idx = self.cur_frame_idx
                            if frame_idx in root_2d_contraints.keyframes:
                                new_root_pos[1] = 0.0  # force y to 0
                                for keyframe_id in root_2d_contraints.frame2keyid[frame_idx]:
                                    # add will modify the existing constraint
                                    root_2d_contraints.add_keyframe(
                                        keyframe_id,
                                        frame_idx,
                                        root_pos=new_root_pos,
                                        exists_ok=True,
                                        update_path=False,
                                    )

                    if "Full-Body" in constraints:
                        full_body_constraints = constraints["Full-Body"]
                        # if there is a constraint at that frame, we want to update it
                        frame_idx = self.cur_frame_idx
                        if frame_idx in full_body_constraints.keyframes:
                            for keyframe_id in full_body_constraints.frame2keyid[frame_idx]:
                                # add will modify the existing constraint
                                full_body_constraints.add_keyframe(
                                    keyframe_id,
                                    frame_idx,
                                    joints_pos=self.joints_pos[frame_idx],
                                    joints_rot=self.joints_rot[frame_idx],
                                    exists_ok=True,
                                )
                    if "End-Effectors" in constraints:
                        end_effector_constraints = constraints["End-Effectors"]
                        # if there is a constraint at that frame, we want to update it
                        frame_idx = self.cur_frame_idx
                        if frame_idx in end_effector_constraints.keyframes:
                            current_dict = end_effector_constraints.keyframes[frame_idx]
                            for keyframe_id, _ in end_effector_constraints.frame2keyid[frame_idx]:
                                # add will modify the existing constraint
                                end_effector_constraints.add_keyframe(
                                    keyframe_id,
                                    frame_idx,
                                    joints_pos=self.joints_pos[frame_idx],
                                    joints_rot=self.joints_rot[frame_idx],
                                    joint_names=current_dict["joint_names"],
                                    end_effector_type=current_dict["end_effector_type"],
                                    exists_ok=True,
                                )

            set_callback_in_closure(joint_idx)

    def clear_all_gizmos(self):
        self.updating_root_translation_gizmo = True
        self.updating_joint_gizmos = True
        if self.root_translation_gizmo is not None:
            self.server.scene.remove_by_name(self.root_translation_gizmo.name)
            self.root_translation_gizmo = None
        if self.joint_gizmos is not None:
            for joint_gizmo in self.joint_gizmos:
                self.server.scene.remove_by_name(joint_gizmo.name)
            self.joint_gizmos = None
        self._drag_start_world_rot = []
        self._joint_gizmo_dragging = []
        self.updating_root_translation_gizmo = False
        self.updating_joint_gizmos = False
