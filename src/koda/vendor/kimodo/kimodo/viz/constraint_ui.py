# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint visualization and frame indexing for the viz UI."""

from typing import List, Optional

import numpy as np
import torch

import viser
import viser.transforms as tf
from kimodo.motion_rep.smooth_root import get_smooth_root_pos
from kimodo.skeleton import SkeletonBase
from kimodo.tools import to_numpy, to_torch

from .scene import SkeletonMesh, WaypointMesh


def update_interval(interval_start, interval_end, start_frame_idx, end_frame_idx):
    """Updates an interval after removing the range from start_frame_idx to end_frame_idx."""
    # Calculate new range after removing [start_frame_idx, end_frame_idx]
    # Case 1: Removal fully contains the interval -> delete entirely
    if start_frame_idx <= interval_start and end_frame_idx >= interval_end:
        return None, None  # Already removed, don't recreate
    # Case 2: Removal is at the start of interval -> shrink from start
    elif start_frame_idx <= interval_start and end_frame_idx < interval_end:
        new_start = end_frame_idx + 1
        new_end = interval_end
    # Case 3: Removal is at the end of interval -> shrink from end
    elif start_frame_idx > interval_start and end_frame_idx >= interval_end:
        new_start = interval_start
        new_end = start_frame_idx - 1
    # Case 4: Removal is in the middle -> keep the larger portion
    else:  # start_frame_idx > interval_start and end_frame_idx < interval_end
        left_size = start_frame_idx - interval_start
        right_size = interval_end - end_frame_idx
        if left_size >= right_size:
            new_start = interval_start
            new_end = start_frame_idx - 1
        else:
            new_start = end_frame_idx + 1
            new_end = interval_end
    return new_start, new_end


class ConstraintSet:
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        self.name = name
        self.server = server
        self.skeleton = skeleton
        self.display_name = display_name if display_name is not None else name

        self.keyframes = dict()  # frame_idx -> poses
        self.frame2keyid = dict()  # frame_idx -> list of keyframe ids at this frame
        self.scene_elements = dict()  # frame_idx -> meshes, labels, etc.
        self.interval_labels = dict()  # (start_frame_idx, end_frame_idx) -> interval_label
        self.labels_visible = True

    def set_label_visibility(self, visible: bool) -> None:
        """Show or hide constraint labels without deleting them."""
        self.labels_visible = visible
        for scene_data in self.scene_elements.values():
            label = scene_data.get("label")
            if label is not None:
                label.visible = visible
        for interval_label in self.interval_labels.values():
            interval_label.visible = visible

    def set_overlay_visibility(self, only_frame: Optional[int] = None) -> None:
        """Show all overlay elements, or only those at the given frame.

        Args:
            only_frame: If None, show all overlays. If int, show only overlays at that frame.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def add_keyframe(self, keyframe_id: str, frame_idx: int, pose_data: torch.Tensor):
        """Adds a single keyframe at the given frame with the given pose data.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            pose_data: torch.Tensor, e.g. full-body pose, EE pose, 2D root pose, etc.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        pose_seq_data: torch.Tensor,
    ):
        """Adds a keyframe interval between the given start and end frames with the given pose data.

        Args:
            interval_id: str, id for the interval. Must be unique within the given start_frame_idx and end_frame_idx.
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
            pose_seq_data: torch.Tensor, data for constrained interval, e.g. full-body poses, EE poses, 2D root poses, etc.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _add_interval_label(self, start_frame_idx: int, end_frame_idx: int):
        """
        Adds an interval label between the given start and end frames
        Args:
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
        """
        mid = int((start_frame_idx + end_frame_idx) / 2)
        interval_label_pos = self._get_label_pos(mid)
        interval_label = self.server.scene.add_label(
            name=f"/{self.name}/interval_label_{start_frame_idx}_{end_frame_idx}",
            text=f"{self.display_name} @ [{start_frame_idx}, {end_frame_idx}]",
            position=interval_label_pos,
            font_size_mode="screen",
            font_screen_scale=0.7,
            anchor="center-center",
        )
        interval_label.visible = self.labels_visible
        self.interval_labels[(start_frame_idx, end_frame_idx)] = interval_label

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        """
        Removes a keyframe at the given frame
        Args:
            keyframe_id: str, id for the keyframe to remove
            frame_idx: int, frame index to remove the keyframe at
        """
        raise NotImplementedError("Subclasses must implement this method")

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        """
        Removes an interval between the given start and end frames
        Args:
            interval_id: str, id for the interval to remove
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_label_pos(self, frame_idx: int):
        """
        Returns the position of where to place the displayed label for the given frame index
        Args:
            frame_idx: int, frame index to get the label position for
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _remove_interval_and_update_label(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        """
        Removes an interval between the given start and end frames and updates the label
        Args:
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
        """
        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            self.remove_keyframe(interval_id, frame_idx)

        # Update interval labels that overlap with the removed range
        intervals_to_update = []
        for (interval_start, interval_end), label in list(self.interval_labels.items()):
            # Check if intervals overlap
            if interval_start <= end_frame_idx and interval_end >= start_frame_idx:
                intervals_to_update.append((interval_start, interval_end, label))

        for interval_start, interval_end, label in intervals_to_update:
            # Remove old label from scene and dict
            self.server.scene.remove_by_name(label.name)
            del self.interval_labels[(interval_start, interval_end)]

            new_start, new_end = update_interval(interval_start, interval_end, start_frame_idx, end_frame_idx)

            if new_start is None or new_end is None:
                continue

            # Create updated label with new range
            if new_start <= new_end:
                # Position label at midpoint - these keyframes are guaranteed to exist
                # since the new range is outside the removal range
                mid_frame = (new_start + new_end) // 2
                label_pos = self._get_label_pos(mid_frame)
                new_label = self.server.scene.add_label(
                    name=f"/{self.name}/interval_label_{new_start}_{new_end}",
                    text=f"{self.display_name} @ [{new_start}, {new_end}]",
                    position=label_pos,
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="center-center",
                )
                new_label.visible = self.labels_visible
                self.interval_labels[(new_start, new_end)] = new_label

    def get_constraint_info(self, device: Optional[str] = None):
        """Returns constraint information for generation (torch) or UI (numpy)."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_frame_idx(self):
        """Returns all constrained frame indices in the set."""
        return [frame_idx for frame_idx in list(self.keyframes.keys())]

    def clear(self, frame_idx: Optional[int] = None):
        """
        Clears all keyframes and intervals from the constraint set
        Args:
            frame_idx: int, sing frame index to clear if given
        """
        raise NotImplementedError("Subclasses must implement this method")


def build_constraint_set_table_markdown(constraint_list: List[ConstraintSet]):
    markdown = "| Track | Frame Num |\n"
    markdown += "|------|----------|\n"

    # Sort constraints by frame_idx
    for constraint in constraint_list:
        frame_info = constraint.get_frame_idx()
        if len(frame_info) > 0:
            frame_info = ", ".join([str(frame) for frame in sorted(frame_info)])
        else:
            frame_info = "-"
        markdown += f"| {constraint.display_name} | {frame_info} |\n"

    return markdown


class FullbodyKeyframeSet(ConstraintSet):
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        super().__init__(name, server, skeleton, display_name=display_name)

    def add_keyframe(
        self,
        keyframe_id: str,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        viz_label: bool = True,
        exists_ok: bool = False,
    ):
        """Adds a single full-body keyframe at the given frame or updates the existing one at this
        frame. Note if a keyframe already exists at this frame, it will be updated to the given
        pose.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            joints_pos: torch.Tensor, [J, 3] joints positions to add the keyframe at
        """
        # create/update scene elements
        if frame_idx in self.keyframes:
            skeleton_mesh = self.scene_elements[frame_idx]["skeleton_mesh"]
            skeleton_mesh.set_pose(to_torch(joints_pos))
            if viz_label and "label" in self.scene_elements[frame_idx]:
                label = self.scene_elements[frame_idx]["label"]
                label.position = to_numpy(joints_pos)[self.skeleton.root_idx]
                label.visible = self.labels_visible
        else:
            # create skeleton to visualize the full-body constraint
            skeleton_mesh = SkeletonMesh(
                f"/{self.name}/skeleton_{frame_idx}",
                self.server,
                self.skeleton,
                joint_color=(255, 235, 0),
                bone_color=(255, 0, 0),
                starting_joints_pos=to_torch(joints_pos),
            )
            self.scene_elements[frame_idx] = {
                "skeleton_mesh": skeleton_mesh,
            }
            if viz_label:
                label = self.server.scene.add_label(
                    name=f"/{self.name}/label_{frame_idx}",
                    text=f"{self.display_name} @ {frame_idx}",
                    position=to_numpy(joints_pos)[self.skeleton.root_idx],
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="center-center",
                )
                label.visible = self.labels_visible
                self.scene_elements[frame_idx]["label"] = label

        # set/update data
        self.keyframes[frame_idx] = {
            "joints_pos": to_numpy(joints_pos),
            "joints_rot": to_numpy(joints_rot),
        }

        if frame_idx not in self.frame2keyid:
            self.frame2keyid[frame_idx] = []

        if keyframe_id in self.frame2keyid[frame_idx]:
            if not exists_ok:
                raise AssertionError("keyframe_id already exists in this frame!")
        else:
            self.frame2keyid[frame_idx].append(keyframe_id)

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
    ):
        """Adds a full-body keyframe interval between the given start and end frames.

        Args:
            start_frame_idx: int, start frame index of the interval
            end_frame_idx: int, end frame index of the interval
            joints_pos: torch.Tensor, [T, J, 3] joints positions within the interval
        """
        assert joints_pos.shape[0] == end_frame_idx - start_frame_idx + 1
        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            rel_idx = frame_idx - start_frame_idx
            self.add_keyframe(
                interval_id,
                frame_idx,
                joints_pos[rel_idx],
                joints_rot[rel_idx],
                viz_label=False,
            )

        # add separate interval label
        self._add_interval_label(start_frame_idx, end_frame_idx)

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        if frame_idx not in self.keyframes:
            return
        if keyframe_id not in self.frame2keyid[frame_idx]:
            return
        self.frame2keyid[frame_idx].remove(keyframe_id)
        if len(self.frame2keyid[frame_idx]) == 0:
            del self.frame2keyid[frame_idx]
            self.clear(frame_idx)

    def _get_label_pos(self, frame_idx: int):
        return self.keyframes[frame_idx]["joints_pos"][self.skeleton.root_idx]

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        self._remove_interval_and_update_label(interval_id, start_frame_idx, end_frame_idx)

    def get_constraint_info(self, device: Optional[str] = None):
        all_joints_pos = []
        all_joints_rot = []
        for v in self.keyframes.values():
            joints_pos = to_torch(v["joints_pos"], device=device)
            joints_rot = to_torch(v["joints_rot"], device=device)
            if len(joints_pos.shape) == 2:
                all_joints_pos.append(joints_pos[None])
            else:
                all_joints_pos.append(joints_pos)
            if len(joints_rot.shape) == 3:
                all_joints_rot.append(joints_rot[None])
            else:
                all_joints_rot.append(joints_rot)

        all_joints_pos = torch.cat(all_joints_pos, dim=0) if len(all_joints_pos) > 0 else None
        all_joints_rot = torch.cat(all_joints_rot, dim=0) if len(all_joints_rot) > 0 else None

        return {
            "frame_idx": self.get_frame_idx(),
            "joints_pos": all_joints_pos,
            "joints_rot": all_joints_rot,
        }

    def clear(self, frame_idx: Optional[int] = None):
        frame_idx_list = list(self.keyframes.keys()) if frame_idx is None else [frame_idx]
        for fidx in frame_idx_list:
            self.scene_elements[fidx]["skeleton_mesh"].clear()
            if "ee_rotation_axes" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["ee_rotation_axes"].name)
            if "label" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["label"].name)

            self.keyframes.pop(fidx)
            self.scene_elements.pop(fidx)
            self.frame2keyid.pop(fidx, None)

        if frame_idx is None:
            # clear all interval labels if clearing all keyframes
            for interval_label in list(self.interval_labels.values()):
                self.server.scene.remove_by_name(interval_label.name)
            self.interval_labels.clear()
            self.frame2keyid.clear()

    def set_overlay_visibility(self, only_frame: Optional[int] = None) -> None:
        show_all = only_frame is None
        for fidx, scene_data in self.scene_elements.items():
            visible = show_all or fidx == only_frame
            scene_data["skeleton_mesh"].set_visibility(visible)
            label = scene_data.get("label")
            if label is not None:
                label.visible = visible and self.labels_visible
        for interval_label in self.interval_labels.values():
            interval_label.visible = show_all and self.labels_visible


class EEJointsKeyframeSet(ConstraintSet):
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        super().__init__(name, server, skeleton, display_name=display_name)

        # frame_idx -> list of (keyframe_id, joint_names) at this frame
        self.frame2keyid = dict()

    def create_scene_elements(
        self,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: Optional[torch.Tensor | np.ndarray],
        joint_names: List[str],
        viz_label: bool = True,
    ):
        # create skeleton to visualize the full-body constraint
        ee_joint_indices = []
        ee_gizmo_indices = []
        constrained_bone_idx = []
        for joint_name in joint_names:
            if joint_name == "Hips":
                continue
            elif joint_name in ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]:
                expanded_joint_names = {
                    "LeftHand": self.skeleton.left_hand_joint_names,
                    "RightHand": self.skeleton.right_hand_joint_names,
                    "LeftFoot": self.skeleton.left_foot_joint_names,
                    "RightFoot": self.skeleton.right_foot_joint_names,
                }[joint_name]
                ee_joint_indices.extend([self.skeleton.bone_order_names_index[joint] for joint in expanded_joint_names])
                if len(expanded_joint_names) > 1:
                    ee_gizmo_indices.extend(
                        [self.skeleton.bone_order_names_index[joint] for joint in expanded_joint_names[:1]]
                    )
                constrained_bone_idx.extend(
                    [self.skeleton.bone_order_names_index[joint] - 1 for joint in expanded_joint_names[1:]]
                )
            else:
                raise ValueError(f"Invalid joint name: {joint_name}")

        # de-duplicate while preserving order
        ee_joint_indices = list(dict.fromkeys(ee_joint_indices))
        ee_gizmo_indices = list(dict.fromkeys(ee_gizmo_indices))
        constrained_bone_idx = list(dict.fromkeys(constrained_bone_idx))

        constrained_idx = [self.skeleton.root_idx] + ee_joint_indices

        constrained_idx = np.array(constrained_idx)
        constrained_bone_idx = np.array(constrained_bone_idx)

        # create skeleton to visualize the full-body constraint
        joint_color = np.full((self.skeleton.nbjoints, 3), (220, 220, 220))
        bone_color = np.full((self.skeleton.nbjoints - 1, 3), (220, 220, 220))
        # color constrained joints differently
        joint_color[constrained_idx] = (255, 0, 0)
        bone_color[constrained_bone_idx] = (255, 0, 0)
        skeleton_mesh = SkeletonMesh(
            f"/{self.name}/skeleton_{frame_idx}",
            self.server,
            self.skeleton,
            joint_color=joint_color,
            bone_color=bone_color,
            starting_joints_pos=to_torch(joints_pos),
        )

        self.scene_elements[frame_idx] = {
            "skeleton_mesh": skeleton_mesh,
        }
        joints_pos_np = to_numpy(joints_pos)
        joints_rot_np = to_numpy(joints_rot) if joints_rot is not None else None
        if joints_rot_np is not None and len(ee_gizmo_indices) > 0:
            ee_axes = self.server.scene.add_batched_axes(
                f"/{self.name}/ee_rot_axes_{frame_idx}",
                batched_wxyzs=tf.SO3.from_matrix(joints_rot_np[ee_gizmo_indices]).wxyz,
                batched_positions=joints_pos_np[ee_gizmo_indices],
                axes_length=0.07,
                axes_radius=0.007,
            )
            self.scene_elements[frame_idx]["ee_rotation_axes"] = ee_axes
        if viz_label:
            label = self.server.scene.add_label(
                name=f"/{self.name}/label_{frame_idx}",
                text=f"{self.display_name} @ {frame_idx}",
                position=joints_pos_np[self.skeleton.root_idx] + np.array([0.0, 0.05, 0.0]),
                font_size_mode="screen",
                font_screen_scale=0.7,
                anchor="bottom-center",
            )
            label.visible = self.labels_visible
            self.scene_elements[frame_idx]["label"] = label

    def add_keyframe(
        self,
        keyframe_id: str,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        joint_names: List[str],
        end_effector_type: str,
        viz_label: bool = True,
        exists_ok: bool = False,
    ):
        """Adds a single EE keyframe at the given frame or updates the existing one at this frame.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            joints_pos: torch.Tensor, [J, 3] joints positions to add the keyframe at
            joints_rot: torch.Tensor, [J, 3, 3] joints rotation matrices to add the keyframe at
            joint_names: List[str], names of the joints to add the keyframe at
        """
        need_create_viz = True
        joint_names_input = joint_names

        if not isinstance(end_effector_type, set):
            end_effector_type = set([end_effector_type])

        # create/update scene elements
        if frame_idx in self.keyframes:
            if joint_names != self.keyframes[frame_idx]["joint_names"]:
                # merge together with existing constraint if needed
                joint_names = set(joint_names)
                joint_names.update(set(self.keyframes[frame_idx]["joint_names"]))
                joint_names = list(joint_names)
                end_effector_type.update(self.keyframes[frame_idx]["end_effector_type"])
                # need to re-create viz elements
                self.clear(frame_idx)
            else:
                need_create_viz = False
                # overwrite the pose with the latest one
                skeleton_mesh = self.scene_elements[frame_idx]["skeleton_mesh"]
                skeleton_mesh.set_pose(to_torch(joints_pos))
                if "ee_rotation_axes" in self.scene_elements[frame_idx]:
                    ee_gizmo_indices = []
                    for joint_name in joint_names:
                        if joint_name == "Hips":
                            continue
                        elif joint_name in [
                            "LeftHand",
                            "RightHand",
                            "LeftFoot",
                            "RightFoot",
                        ]:
                            expanded_joint_names = {
                                "LeftHand": self.skeleton.left_hand_joint_names,
                                "RightHand": self.skeleton.right_hand_joint_names,
                                "LeftFoot": self.skeleton.left_foot_joint_names,
                                "RightFoot": self.skeleton.right_foot_joint_names,
                            }[joint_name]
                            if len(expanded_joint_names) > 0:
                                ee_gizmo_indices.extend(
                                    [self.skeleton.bone_order_names_index[joint] for joint in expanded_joint_names[:1]]
                                    # take only the base joint of the end effector (to avoid clutter)
                                )
                        else:
                            raise ValueError(f"Invalid joint name: {joint_name}")
                    ee_gizmo_indices = list(dict.fromkeys(ee_gizmo_indices))
                    if len(ee_gizmo_indices) > 0:
                        ee_axes = self.scene_elements[frame_idx]["ee_rotation_axes"]
                        joints_pos_np = to_numpy(joints_pos)
                        joints_rot_np = to_numpy(joints_rot)
                        ee_axes.batched_positions = joints_pos_np[ee_gizmo_indices]
                        ee_axes.batched_wxyzs = tf.SO3.from_matrix(joints_rot_np[ee_gizmo_indices]).wxyz
                if viz_label and "label" in self.scene_elements[frame_idx]:
                    label = self.scene_elements[frame_idx]["label"]
                    label.position = to_numpy(joints_pos)[self.skeleton.root_idx]
                    label.visible = self.labels_visible

        if need_create_viz:
            self.create_scene_elements(frame_idx, joints_pos, joints_rot, joint_names, viz_label=viz_label)

        # set/update data
        self.keyframes[frame_idx] = {
            "joints_pos": to_numpy(joints_pos),
            "joints_rot": to_numpy(joints_rot),
            "joint_names": joint_names,
            "end_effector_type": end_effector_type,
        }

        if frame_idx not in self.frame2keyid:
            self.frame2keyid[frame_idx] = []

        known_keyframe_ids = {k: idx for idx, (k, _) in enumerate(self.frame2keyid[frame_idx])}

        if keyframe_id in known_keyframe_ids.keys():
            if not exists_ok:
                raise AssertionError("keyframe_id already exists in this frame!")
            idx = known_keyframe_ids[keyframe_id]
            # override previous exisiting keyframe
            self.frame2keyid[frame_idx][idx] = (keyframe_id, joint_names_input)
        else:
            # track which subset of joints are constrained by this keyframe_id
            self.frame2keyid[frame_idx].append((keyframe_id, joint_names_input))

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        joint_names: List[str],
        end_effector_type: str,
    ):
        """Adds an interval of EE keyframes at the given frame or updates the existing one at this
        frame.

        Args:
            interval_id: str, id for the interval. Must be unique within the given start_frame_idx and end_frame_idx.
            start_frame_idx: int, start frame index to add the interval at
            end_frame_idx: int, end frame index to add the interval at
            joints_pos: torch.Tensor, [T, J, 3] joints positions to add the interval at
            joints_rot: torch.Tensor, [T, J, 3, 3] joints rotation matrices to add the interval at
            joint_names: List[str], names of the joints to add for the entire interval
        """
        num_frames = end_frame_idx - start_frame_idx + 1
        joints_pos_np = to_numpy(joints_pos)
        joints_rot_np = to_numpy(joints_rot)
        assert joints_pos_np.shape[0] == num_frames
        assert joints_rot_np.shape[0] == num_frames

        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            rel_idx = frame_idx - start_frame_idx
            self.add_keyframe(
                interval_id,
                frame_idx,
                joints_pos_np[rel_idx],
                joints_rot_np[rel_idx],
                joint_names,
                end_effector_type,
                viz_label=False,
            )
        self._add_interval_label(start_frame_idx, end_frame_idx)

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        """Removes a keyframe at the given frame or updates the existing one at this frame by
        removing the specified joints.

        Args:
            keyframe_id: str, id for the keyframe to remove. This determines which joints to remove.
            frame_idx: int, frame index to remove the keyframe at
        """
        if frame_idx not in self.keyframes:
            return

        remaining_joint_names = set()
        delete_idx = None
        for i, (keyid, joint_names) in enumerate(self.frame2keyid[frame_idx]):
            if keyid == keyframe_id:
                delete_idx = i
            else:
                remaining_joint_names.update(joint_names)
        if delete_idx is None:
            # this keyframe_id is not in the specified frame
            return

        self.frame2keyid[frame_idx].pop(delete_idx)
        if len(remaining_joint_names) == 0:
            # no more keyframes in this frame, clear the frame
            del self.frame2keyid[frame_idx]
            self.clear(frame_idx)
            return

        # only deleting part of keyframe (potentially some subset of joints)
        # delete the old visualization and add a new one with the updated joint set
        new_joint_names = list(remaining_joint_names)
        self.clear(frame_idx, scene_elements_only=True)
        joints_pos = self.keyframes[frame_idx]["joints_pos"]
        joints_rot = self.keyframes[frame_idx]["joints_rot"]
        self.create_scene_elements(frame_idx, joints_pos, joints_rot, new_joint_names)
        self.keyframes[frame_idx]["joint_names"] = new_joint_names

    def _get_label_pos(self, frame_idx: int):
        return self.keyframes[frame_idx]["joints_pos"][self.skeleton.root_idx]

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        self._remove_interval_and_update_label(interval_id, start_frame_idx, end_frame_idx)

    def get_constraint_info(self, device: Optional[str] = None):
        all_joints_pos = []
        all_joints_rot = []
        all_joints_names = []
        all_end_effector_type = []
        for v in self.keyframes.values():
            joints_pos = to_torch(v["joints_pos"], device=device)
            joints_rot = to_torch(v["joints_rot"], device=device)
            if len(joints_pos.shape) == 2:
                all_joints_pos.append(joints_pos[None])
            else:
                all_joints_pos.append(joints_pos)
            if len(joints_rot.shape) == 3:
                all_joints_rot.append(joints_rot[None])
            else:
                all_joints_rot.append(joints_rot)
            all_joints_names.append(v["joint_names"])
            all_end_effector_type.append(v["end_effector_type"])

        all_joints_pos = torch.cat(all_joints_pos, dim=0) if len(all_joints_pos) > 0 else None
        all_joints_rot = torch.cat(all_joints_rot, dim=0) if len(all_joints_rot) > 0 else None

        return {
            "frame_idx": self.get_frame_idx(),
            "joints_pos": all_joints_pos,
            "joints_rot": all_joints_rot,
            "joint_names": all_joints_names,
            "end_effector_type": all_end_effector_type,
        }

    def clear(self, frame_idx: Optional[int] = None, scene_elements_only: bool = False):
        frame_idx_list = list(self.keyframes.keys()) if frame_idx is None else [frame_idx]
        for fidx in frame_idx_list:
            self.scene_elements[fidx]["skeleton_mesh"].clear()
            if "ee_rotation_axes" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["ee_rotation_axes"].name)
            if "label" in self.scene_elements[fidx]:
                self.server.scene.remove_by_name(self.scene_elements[fidx]["label"].name)
            self.scene_elements.pop(fidx)
            if not scene_elements_only:
                self.keyframes.pop(fidx)

        if frame_idx is None:
            # clear all interval labels if clearing all keyframes
            for interval_label in list(self.interval_labels.values()):
                self.server.scene.remove_by_name(interval_label.name)
            self.interval_labels.clear()

    def set_overlay_visibility(self, only_frame: Optional[int] = None) -> None:
        show_all = only_frame is None
        for fidx, scene_data in self.scene_elements.items():
            visible = show_all or fidx == only_frame
            scene_data["skeleton_mesh"].set_visibility(visible)
            if "ee_rotation_axes" in scene_data:
                scene_data["ee_rotation_axes"].visible = visible
            label = scene_data.get("label")
            if label is not None:
                label.visible = visible and self.labels_visible
        for interval_label in self.interval_labels.values():
            interval_label.visible = show_all and self.labels_visible


class RootKeyframe2DSet(ConstraintSet):
    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        skeleton: SkeletonBase,
        display_name: Optional[str] = None,
    ):
        super().__init__(name, server, skeleton, display_name=display_name)
        self.dense_path = False
        self.smooth_path = True
        self.line_segments = None  # visualization of dense path
        self.interval_line_segments = {}

    def add_keyframe(
        self,
        keyframe_id: str,
        frame_idx: int,
        root_pos: torch.Tensor | np.ndarray,
        viz_label: bool = True,
        update_path: bool = True,
        viz_waypoint: bool = True,
        exists_ok: bool = False,
    ):
        """Adds a single 2D root keyframe at the given frame or updates the existing one at this
        frame.

        Args:
            keyframe_id: str, id for the keyframe. Must be unique within the given frame_idx.
            frame_idx: int, frame index to add the keyframe at
            root_pos: torch.Tensor, [3] root position to add the keyframe at, y entry (index 1) should be 0
            viz_label: bool, whether to visualize the label for the keyframe
        """
        root_pos_np = to_numpy(root_pos)
        if frame_idx not in self.scene_elements:
            self.scene_elements[frame_idx] = {}

        scene_data = self.scene_elements[frame_idx]
        if frame_idx in self.keyframes:
            waypoint = scene_data.get("waypoint")
            if waypoint is not None:
                waypoint.update_position(root_pos_np)
            elif viz_waypoint:
                waypoint = WaypointMesh(
                    f"/{self.name}/waypoint_{frame_idx}",
                    self.server,
                    position=root_pos_np,
                )
                scene_data["waypoint"] = waypoint

            label = scene_data.get("label")
            if viz_label and label is not None:
                label.position = root_pos_np
                label.visible = self.labels_visible
            elif viz_label and label is None:
                label = self.server.scene.add_label(
                    name=f"/{self.name}/label_{frame_idx}",
                    text=f"{self.display_name} @ {frame_idx}",
                    position=root_pos_np,
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="bottom-left",
                )
                label.visible = self.labels_visible
                scene_data["label"] = label
        else:
            if viz_waypoint:
                waypoint = WaypointMesh(
                    f"/{self.name}/waypoint_{frame_idx}",
                    self.server,
                    position=root_pos_np,
                )
                scene_data["waypoint"] = waypoint
            if viz_label:
                label = self.server.scene.add_label(
                    name=f"/{self.name}/label_{frame_idx}",
                    text=f"{self.display_name} @ {frame_idx}",
                    position=root_pos_np,
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="bottom-left",
                )
                label.visible = self.labels_visible
                scene_data["label"] = label

        # set/update data
        self.keyframes[frame_idx] = root_pos_np
        if frame_idx not in self.frame2keyid:
            self.frame2keyid[frame_idx] = []

        if keyframe_id in self.frame2keyid[frame_idx]:
            if not exists_ok:
                raise AssertionError("keyframe_id already exists in this frame!")
        else:
            self.frame2keyid[frame_idx].append(keyframe_id)

        # need to update path visualization
        if self.line_segments is not None and update_path:
            self.update_line_segments()

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        root_pos: torch.Tensor | np.ndarray,
    ):
        """Adds an interval of 2D root keyframes between the given start and end frames.

        Args:
            interval_id: str, id for the interval. Must be unique within the given start_frame_idx and end_frame_idx.
            start_frame_idx: int, start frame index to add the interval at
            end_frame_idx: int, end frame index to add the interval at
            root_pos: torch.Tensor, [T, 3] root positions to add the interval at
        """
        root_pos_np = to_numpy(root_pos)
        assert root_pos_np.shape[0] == end_frame_idx - start_frame_idx + 1
        if root_pos_np.shape[0] >= 2:
            points = np.zeros((root_pos_np.shape[0] - 1, 2, 3))
            points[:, 0] = root_pos_np[:-1]
            points[:, 1] = root_pos_np[1:]
            if interval_id in self.interval_line_segments:
                self.server.scene.remove_by_name(self.interval_line_segments[interval_id].name)
            self.interval_line_segments[interval_id] = self.server.scene.add_line_segments(
                name=f"/{self.name}/interval_{interval_id}_line",
                points=points,
                colors=(255, 0, 0),
                line_width=5.0,
            )

        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            rel_idx = frame_idx - start_frame_idx
            self.add_keyframe(
                interval_id,
                frame_idx,
                root_pos_np[rel_idx],
                viz_label=False,
                update_path=False,
                viz_waypoint=False,
            )
        self._add_interval_label(start_frame_idx, end_frame_idx)
        if self.line_segments is not None:
            self.update_line_segments()

    def set_smooth_path(self, smooth_path: bool):
        self.smooth_path = smooth_path
        if self.line_segments is not None:
            self.update_line_segments()

    def set_dense_path(self, dense_path: bool):
        """If dense_path is True, will make the path dense by interpolated between added keyframes.

        Args:
            dense_path: bool, whether to make the path dense
        """
        self.dense_path = dense_path
        if self.dense_path:
            # visualize dense path with line segments
            self.line_segments = self.server.scene.add_line_segments(
                name=f"/{self.name}/line_segments",
                points=np.zeros((1, 2, 3)),
                colors=(255, 0, 0),
                line_width=5.0,
            )
            self.update_line_segments()
        else:
            if self.line_segments is not None:
                self.server.scene.remove_by_name(self.line_segments.name)
                self.line_segments = None

    def interpolate_path(self, t: np.ndarray):
        """Interpolates the path between the given frame indices.

        Args:
            t: np.ndarray, frame indices to interpolate at
        """
        from scipy.interpolate import interp1d

        cur_info = self._get_sparse_constraint_info()
        frame_idx = cur_info["frame_idx"]
        all_root_pos = cur_info["root_pos"]

        x = all_root_pos[:, 0]
        z = all_root_pos[:, 2]

        kind = "linear"
        # if self.smooth_path and len(frame_idx) >= 3:
        # kind = "quadratic"

        interp_x = interp1d(frame_idx, x, kind=kind)
        interp_z = interp1d(frame_idx, z, kind=kind)

        x_new = interp_x(t)
        z_new = interp_z(t)

        path3d = np.stack([x_new, np.zeros_like(x_new), z_new], axis=1)

        if self.smooth_path and len(frame_idx) >= 3:
            path3d = get_smooth_root_pos(torch.from_numpy(path3d[None]))[0].numpy()
        return path3d

    def update_line_segments(self):
        if len(self.keyframes) < 2:
            return

        t = np.array(sorted(self.get_frame_idx()))
        if self.smooth_path:
            # more points for smoothed curve
            t = np.linspace(t[0], t[-1], 100)

        path3d = self.interpolate_path(t)

        points = np.zeros((len(path3d) - 1, 2, 3))
        points[:, 0] = path3d[:-1]
        points[:, 1] = path3d[1:]

        self.line_segments.points = points

    def remove_keyframe(self, keyframe_id: str, frame_idx: int):
        if frame_idx not in self.keyframes:
            return
        if keyframe_id not in self.frame2keyid[frame_idx]:
            return
        self.frame2keyid[frame_idx].remove(keyframe_id)
        if len(self.frame2keyid[frame_idx]) == 0:
            del self.frame2keyid[frame_idx]
            self.clear(frame_idx)
            if self.line_segments is not None:
                self.update_line_segments()

    def _get_label_pos(self, frame_idx: int):
        return self.keyframes[frame_idx]

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int):
        if interval_id in self.interval_line_segments:
            self.server.scene.remove_by_name(self.interval_line_segments[interval_id].name)
            del self.interval_line_segments[interval_id]
        self._remove_interval_and_update_label(interval_id, start_frame_idx, end_frame_idx)

    def _get_sparse_constraint_info(self):
        all_root_pos = []
        for v in self.keyframes.values():
            v_np = to_numpy(v)
            if len(v_np.shape) == 1:
                all_root_pos.append(v_np[None])
            else:
                all_root_pos.append(v_np)
        if len(all_root_pos) > 0:
            all_root_pos = np.concatenate(all_root_pos, axis=0)
        else:
            all_root_pos = None
        return {
            "frame_idx": self.get_frame_idx(),
            "root_pos": all_root_pos,
        }

    def get_constraint_info(self, device: Optional[str] = None):
        if not self.dense_path or len(self.keyframes) == 0:
            info = self._get_sparse_constraint_info()
            return {
                "frame_idx": info["frame_idx"],
                "root_pos": to_torch(info["root_pos"], device=device, dtype=torch.float32),
            }
        else:
            frame_idx_list = self.get_frame_idx()
            min_frame_idx = min(frame_idx_list)
            max_frame_idx = max(frame_idx_list)
            t = np.arange(min_frame_idx, max_frame_idx + 1)
            path3d = self.interpolate_path(t)
            return {
                "frame_idx": t.tolist(),
                "root_pos": to_torch(path3d, device=device, dtype=torch.float32),
            }

    def clear(self, frame_idx: Optional[int] = None):
        frame_idx_list = list(self.keyframes.keys()) if frame_idx is None else [frame_idx]
        for fidx in frame_idx_list:
            scene_data = self.scene_elements.get(fidx, {})
            waypoint = scene_data.get("waypoint")
            if waypoint is not None:
                waypoint.clear()
            label = scene_data.get("label")
            if label is not None:
                self.server.scene.remove_by_name(label.name)

            self.keyframes.pop(fidx)
            self.scene_elements.pop(fidx)

        if frame_idx is None:
            # clear all interval labels if clearing all keyframes
            for interval_label in list(self.interval_labels.values()):
                self.server.scene.remove_by_name(interval_label.name)
            self.interval_labels.clear()

            # clear line segments if turning off dense path
            if self.line_segments is not None:
                self.server.scene.remove_by_name(self.line_segments.name)
                self.line_segments = None

            for interval_line in list(self.interval_line_segments.values()):
                self.server.scene.remove_by_name(interval_line.name)
            self.interval_line_segments.clear()

    def set_overlay_visibility(self, only_frame: Optional[int] = None) -> None:
        show_all = only_frame is None
        for fidx, scene_data in self.scene_elements.items():
            visible = show_all or fidx == only_frame
            waypoint = scene_data.get("waypoint")
            if waypoint is not None:
                waypoint.set_visible(visible)
            label = scene_data.get("label")
            if label is not None:
                label.visible = visible and self.labels_visible
        if self.line_segments is not None:
            self.line_segments.visible = show_all
        for line_handle in self.interval_line_segments.values():
            line_handle.visible = show_all
        for interval_label in self.interval_labels.values():
            interval_label.visible = show_all and self.labels_visible


#
# GUI Elements that need to be tracked
