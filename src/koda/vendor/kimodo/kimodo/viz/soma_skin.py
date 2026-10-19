# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA skeleton skinning for visualization (SOMASkeleton30 / SOMASkeleton77)."""

from pathlib import Path

import numpy as np
import torch

from kimodo.skeleton import (
    SOMASkeleton30,
    SOMASkeleton77,
    batch_rigid_transform,
    global_rots_to_local_rots,
)

# Skin for SOMASkeleton77
SKEL_PATH = "somaskel77"
SKIN_NAME = "skin_standard.npz"


class SOMASkin:
    def __init__(self, skeleton):
        skel_path = Path(skeleton.folder).parent / SKEL_PATH
        skin_data_path = skel_path / SKIN_NAME

        self.skeleton_input = skeleton
        assert isinstance(
            skeleton, (SOMASkeleton30, SOMASkeleton77)
        ), "SOMASkin currently only supports SOMASkeleton30 or SOMASkeleton77"
        assert skeleton.neutral_joints is not None, "The skeleton must have neutral joints instantiated"
        device = skeleton.neutral_joints.device

        # the skin is always the 77-joint skeleton
        #   if user is using the 30-joint skeleton, we will pad it when skinning is called
        self.skeleton_skin = SOMASkeleton77(skel_path).to(device)

        # bind_rig_transform: [R, 4, 4]
        # bind_vertices: [V, 3]
        # faces: [F, 3]
        # lbs indices, lbs weights: [V, W] (W = max (num joints vertice is related to), in our case W=5)
        skin_data = np.load(skin_data_path)
        bind_rig_np = np.array(skin_data["bind_rig_transform"], dtype=np.float32)
        self.bind_rig_transform = torch.from_numpy(bind_rig_np).to(device=device, dtype=torch.float)
        # Precompute the inverse in numpy to avoid torch lazy evaluation issues
        bind_rig_inv_np = np.linalg.inv(bind_rig_np)
        self.bind_rig_transform_inv = torch.from_numpy(bind_rig_inv_np).to(device=device, dtype=torch.float)
        self.bind_vertices = torch.tensor(skin_data["bind_vertices"], device=device, dtype=torch.float)
        self.faces = torch.tensor(skin_data["faces"], device=device, dtype=torch.long)
        self.lbs_indices = torch.tensor(skin_data["lbs_indices"], device=device, dtype=torch.long)
        self.lbs_weights = torch.tensor(skin_data["lbs_weights"], device=device, dtype=torch.float)

        # double check the rig matches expected skeleton
        rig_joint_names = list(skin_data["rig_joint_names"])  # list(str) : [R]
        for sname, rname in zip(self.skeleton_skin.bone_order_names, rig_joint_names):
            if sname != rname:
                raise ValueError(f"MISMATCH in skinnging rig: expected='{sname}' vs rig='{rname}'")

    def lbs(self, posed_transform):
        bind_rig_transform_inv = self.bind_rig_transform_inv
        bind_vertices = self.bind_vertices
        lbs_weights = self.lbs_weights
        # posed_transform: [B, F, J, 4, 4] or [B, J, 4, 4] or [J, 4, 4]
        # unsqueeze to match posed_transform dim
        for _ in range(posed_transform.dim() - 3):
            bind_rig_transform_inv = bind_rig_transform_inv.unsqueeze(0)
            bind_vertices = bind_vertices.unsqueeze(0)
            lbs_weights = lbs_weights.unsqueeze(0)
            # bind_rig_transform_inv: [..., R, 4, 4]
            # bind_vertices: [..., V, 3]
            # lbs_weights: [..., V, W]

        affine_mat = (posed_transform @ bind_rig_transform_inv)[..., :3, :]  # [..., J, 3, 4]
        vs = (
            affine_mat[..., self.lbs_indices, :, :]
            @ torch.concat([bind_vertices, torch.ones_like(bind_vertices[..., 0:1])], dim=-1)[..., None, :, None]
        )  # [..., V, W, 3, 1]
        ws = lbs_weights[..., None, None]
        resv = (vs * ws).sum(dim=-3).squeeze(-1)  # [..., V, 3]
        return resv

    def skin(self, joint_rotmat, joint_pos, rot_is_global=False):
        """
        joint_rotmat: [T, J, 3, 3] local or global joint rotation matrices
        joint_pos: [T, J, 3] global joint positions
        rot_is_global: bool, if True, joint_rotmat is global rotation matrices, otherwise it is local rotation matrices and FK is performed internally
        """
        nF, nJ = joint_pos.shape[:2]
        device = joint_rotmat.device

        if nJ != self.skeleton_skin.nbjoints:
            assert nJ == 30, "SOMASkin currently only supports 30-joint or 77-joint skeletons"

            # make sure we have local joint rotations
            if rot_is_global:
                local_joint_rots_mats_subset = global_rots_to_local_rots(joint_rotmat, self.skeleton_input)
            else:
                local_joint_rots_mats_subset = joint_rotmat

            local_joint_rots_mats = self.skeleton_input.to_SOMASkeleton77(local_joint_rots_mats_subset)

            # FK to get the global joint pos and rot
            neutral_joints_seq = self.skeleton_skin.neutral_joints[None].repeat((nF, 1, 1)).to(device)
            new_joint_pos, joint_rotmat = batch_rigid_transform(
                local_joint_rots_mats,
                neutral_joints_seq,
                self.skeleton_skin.joint_parents.to(device),
                self.skeleton_skin.root_idx,
            )
            joint_pos = new_joint_pos + joint_pos[:, self.skeleton_input.root_idx : self.skeleton_input.root_idx + 1]
            nJ = self.skeleton_skin.nbjoints
            rot_is_global = True

        # prepare full transformation matrices
        fk_transform = torch.eye(4, device=device)[None, None].repeat(nF, nJ, 1, 1)
        fk_transform[..., :3, 3] = joint_pos
        if rot_is_global:
            fk_transform[..., :3, :3] = joint_rotmat
        else:
            neutral_joints_seq = self.skeleton_skin.neutral_joints[None].repeat((nF, 1, 1)).to(device)
            # FK to get the global rotations
            _, global_joint_rotmat = batch_rigid_transform(
                joint_rotmat,
                neutral_joints_seq,
                self.skeleton_skin.joint_parents.to(device),
                self.skeleton_skin.root_idx,
            )
            fk_transform[..., :3, :3] = global_joint_rotmat

        vertices = self.lbs(fk_transform)
        return vertices
