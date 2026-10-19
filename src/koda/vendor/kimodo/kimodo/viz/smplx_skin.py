# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SMPL-X skinning and joint mapping for visualization."""

import os
import warnings
from pathlib import Path

import numpy as np
import torch

from kimodo.geometry import axis_angle_to_matrix
from kimodo.skeleton import SMPLXSkeleton22, batch_rigid_transform

SKIN_NAME = "SMPLX_NEUTRAL.npz"
BETA_NAME = "beta.npy"
MEAN_HANDS_NAME = "mean_hands.npy"

SMPLX_BODY_JOINT_NAME_MAP = {
    "pelvis": "Pelvis",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "spine1": "Spine1",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "spine2": "Spine2",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
    "spine3": "Spine3",
    "left_foot": "L_Foot",
    "right_foot": "R_Foot",
    "neck": "Neck",
    "left_collar": "L_Collar",
    "right_collar": "R_Collar",
    "head": "Head",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
}

# SMPL-X hand pose order (15 joints per hand) matching SMPL-X index order.
SMPLX_HAND_JOINT_ORDER = [
    "Index1",
    "Index2",
    "Index3",
    "Middle1",
    "Middle2",
    "Middle3",
    "Pinky1",
    "Pinky2",
    "Pinky3",
    "Ring1",
    "Ring2",
    "Ring3",
    "Thumb1",
    "Thumb2",
    "Thumb3",
]

SMPLX_FACE_JOINT_NAMES = ["Jaw", "L_Eye", "R_Eye"]


class SMPLXSkin:
    def __init__(
        self,
        skeleton,
        use_mean_hands=True,
    ):
        skel_dir = Path(skeleton.folder)
        skin_data_path = skel_dir / SKIN_NAME

        if not skin_data_path.exists():
            raise FileExistsError(
                f"You should download the {SKIN_NAME} from the smplx website, and put it there: {skin_data_path}"
            )

        beta_path = skel_dir / BETA_NAME
        mean_hands_path = skel_dir / MEAN_HANDS_NAME

        self.skeleton = skeleton
        assert isinstance(skeleton, SMPLXSkeleton22), "SMPLXSkin only supports SMPLXSkeleton22"
        assert skeleton.neutral_joints is not None, "SMPLXSkeleton22 must have neutral joints instantiated"

        device = skeleton.neutral_joints.device
        with warnings.catch_warnings():
            # Ignore legacy object-dtype warning emitted while unpickling old SMPL-X assets.
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=Warning,
                module=r"numpy\.lib\._format_impl",
            )
            # np.load on .npz is lazy; materialize all fields while filter is active.
            with np.load(skin_data_path, allow_pickle=True) as skin_npz:
                skin_data = {key: skin_npz[key] for key in skin_npz.files}

        joint2num = skin_data["joint2num"]
        if isinstance(joint2num, np.ndarray):
            joint2num = joint2num.item()
        self.full_joint_count = int(skin_data["weights"].shape[1])
        kintree_table = np.array(skin_data["kintree_table"], dtype=np.int64)
        parents = kintree_table[0].copy()
        parents[parents > 1_000_000_000] = -1
        self.full_joint_parents = torch.tensor(parents, device=device, dtype=torch.long)
        root_candidates = np.where(parents == -1)[0]
        self.full_root_idx = int(root_candidates[0]) if root_candidates.size else 0
        self.joint_regressor = torch.tensor(
            np.array(skin_data["J_regressor"], dtype=np.float32),
            device=device,
            dtype=torch.float,
        )

        rig_joint_names = []
        rig_joint_indices = []
        for joint_name in self.skeleton.bone_order_names:
            mapped_name = SMPLX_BODY_JOINT_NAME_MAP.get(joint_name)
            if mapped_name is None or mapped_name not in joint2num:
                raise ValueError(f"Missing SMPL-X joint mapping for '{joint_name}'")
            rig_joint_names.append(mapped_name)
            rig_joint_indices.append(int(joint2num[mapped_name]))
        self.body_joint_indices = np.array(rig_joint_indices, dtype=np.int64)

        # Prepare mean hand pose rotations for joints not produced by the model.
        if use_mean_hands and mean_hands_path is not None and os.path.exists(mean_hands_path):
            mean_hands = np.array(np.load(mean_hands_path), dtype=np.float32)
        else:
            mean_hands = np.zeros(90, dtype=np.float32)
        if mean_hands.shape[0] != 90:
            raise ValueError(f"Expected mean_hands shape (90,), got {mean_hands.shape}")
        mean_hands = mean_hands.reshape(30, 3)
        mean_hands_rotmats = axis_angle_to_matrix(torch.tensor(mean_hands, device=device, dtype=torch.float))
        left_hand_joint_names = [f"L_{name}" for name in SMPLX_HAND_JOINT_ORDER]
        right_hand_joint_names = [f"R_{name}" for name in SMPLX_HAND_JOINT_ORDER]
        left_indices = [joint2num[name] for name in left_hand_joint_names]
        right_indices = [joint2num[name] for name in right_hand_joint_names]
        self.hand_joint_indices = np.array(left_indices + right_indices, dtype=np.int64)
        self.mean_hand_rotmats = mean_hands_rotmats
        face_indices = [joint2num[name] for name in SMPLX_FACE_JOINT_NAMES if name in joint2num]
        self.face_joint_indices = np.array(face_indices, dtype=np.int64)
        self.mean_face_rotmats = torch.eye(3, device=device).repeat(len(self.face_joint_indices), 1, 1)

        # bind_rig_transform: [J, 4, 4]
        # bind_vertices: [V, 3]
        # faces: [F, 3]
        # lbs indices, lbs weights: [V, W] (W = number of joints)
        v_template = np.array(skin_data["v_template"], dtype=np.float32)
        faces = np.array(skin_data["f"], dtype=np.int64)
        weights = np.array(skin_data["weights"], dtype=np.float32)

        shapedirs = np.array(skin_data["shapedirs"], dtype=np.float32)
        posedirs = np.array(skin_data["posedirs"], dtype=np.float32)

        if beta_path is not None and os.path.exists(beta_path):
            betas = np.array(np.load(beta_path), dtype=np.float32)
        else:
            betas = np.zeros(300, dtype=np.float32)

        num_shape_coeffs = shapedirs.shape[2]  # 400 = 300 + 100 (shape + expression)
        if betas.shape[0] < num_shape_coeffs:
            betas = np.pad(betas, (0, num_shape_coeffs - betas.shape[0]), mode="constant")
        elif betas.shape[0] > num_shape_coeffs:
            betas = betas[:num_shape_coeffs]

        v_shaped = v_template + np.tensordot(shapedirs, betas, axes=[2, 0])
        self.v_shaped = torch.tensor(v_shaped, device=device, dtype=torch.float)
        self.posedirs = torch.tensor(posedirs, device=device, dtype=torch.float)
        self.joint_rest = torch.einsum("jv,vc->jc", self.joint_regressor, self.v_shaped)

        # Align SMPL-X body rest joints to the model skeleton rest pose.
        body_rest = self.skeleton.neutral_joints.to(device=device, dtype=torch.float)
        if body_rest.shape[0] == self.body_joint_indices.shape[0]:
            # Treat mismatches as a warning and align to the skeleton pose anyway.
            max_delta = (self.joint_rest[self.body_joint_indices] - body_rest).abs().max()
            if max_delta > 1e-6:
                print(
                    "Warning: SMPL-X rest pose mismatch (max_delta="
                    f"{max_delta:.2e}); aligning to skeleton neutral joints."
                )
            self.joint_rest[self.body_joint_indices] = body_rest

        # Renormalize weights to avoid numerical issues.
        weight_sums = weights.sum(axis=1, keepdims=True)
        zero_mask = weight_sums[:, 0] < 1e-8
        weights = weights / np.clip(weight_sums, 1e-8, None)
        if np.any(zero_mask):
            weights[zero_mask, :] = 0.0
            weights[zero_mask, self.full_root_idx] = 1.0

        joint_indices = np.arange(self.full_joint_count, dtype=np.int64)
        lbs_indices = np.tile(joint_indices[None, :], (v_template.shape[0], 1))

        bind_rig_np = np.zeros((self.full_joint_count, 4, 4), dtype=np.float32)
        bind_rig_np[:, 3, 3] = 1.0
        bind_rig_np[:, :3, :3] = np.eye(3, dtype=np.float32)
        bind_rig_np[:, :3, 3] = self.joint_rest.detach().cpu().numpy()

        self.bind_rig_transform = torch.from_numpy(bind_rig_np).to(device=device, dtype=torch.float)
        bind_rig_inv_np = np.linalg.inv(bind_rig_np)
        self.bind_rig_transform_inv = torch.from_numpy(bind_rig_inv_np).to(device=device, dtype=torch.float)
        self.bind_vertices = torch.tensor(v_shaped, device=device, dtype=torch.float)
        self.faces = torch.tensor(faces, device=device, dtype=torch.long)
        self.lbs_indices = torch.tensor(lbs_indices, device=device, dtype=torch.long)
        self.lbs_weights = torch.tensor(weights, device=device, dtype=torch.float)

        # double check the rig matches expected skeleton order
        for sname, rname in zip(self.skeleton.bone_order_names, rig_joint_names):
            mapped_name = SMPLX_BODY_JOINT_NAME_MAP.get(sname)
            if mapped_name != rname:
                raise ValueError(f"MISMATCH in skinning rig: expected='{mapped_name}' vs rig='{rname}'")

    def lbs(self, posed_transform, bind_vertices=None):
        bind_rig_transform_inv = self.bind_rig_transform_inv
        if bind_vertices is None:
            bind_vertices = self.bind_vertices
        lbs_weights = self.lbs_weights
        # posed_transform: [B, F, J, 4, 4] or [B, J, 4, 4] or [J, 4, 4]
        # unsqueeze to match posed_transform batch dims
        batch_dims = posed_transform.shape[:-3]
        if bind_vertices.dim() == 2:
            for _ in batch_dims:
                bind_vertices = bind_vertices.unsqueeze(0)
        elif bind_vertices.dim() == 3:
            if len(batch_dims) == 1:
                if bind_vertices.shape[0] != batch_dims[0]:
                    bind_vertices = bind_vertices.unsqueeze(0)
            elif len(batch_dims) > 1:
                for _ in range(len(batch_dims) - 1):
                    bind_vertices = bind_vertices.unsqueeze(0)
        for _ in batch_dims:
            bind_rig_transform_inv = bind_rig_transform_inv.unsqueeze(0)
            lbs_weights = lbs_weights.unsqueeze(0)
        # bind_rig_transform_inv: [..., J, 4, 4]
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
        rot_is_global: bool, if True, joint_rotmat is global rotation matrices,
        otherwise it is local rotation matrices and FK is performed internally
        """
        nF, nJ = joint_pos.shape[:2]
        device = joint_rotmat.device

        # import ipdb; ipdb.set_trace()
        if rot_is_global:
            if joint_rotmat.shape[1] == self.full_joint_count:
                local_rotmat_full = joint_rotmat.clone()
                parents = self.full_joint_parents.to(device)
                parent_rot_mats = local_rotmat_full[:, parents]
                parent_rot_mats[:, self.full_root_idx] = torch.eye(3, device=device)
                parent_rot_mats_inv = parent_rot_mats.transpose(2, 3)
                local_rotmat_full = torch.einsum(
                    "T N m n, T N n o -> T N m o",
                    parent_rot_mats_inv,
                    local_rotmat_full,
                )
            else:
                local_rotmat = self.skeleton.global_rots_to_local_rots(joint_rotmat)
        else:
            local_rotmat = joint_rotmat

        if rot_is_global and joint_rotmat.shape[1] == self.full_joint_count:
            full_local = local_rotmat_full
        else:
            full_local = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(nF, self.full_joint_count, 1, 1)
            full_local[:, self.body_joint_indices] = local_rotmat
        if self.mean_hand_rotmats is not None:
            full_local[:, self.hand_joint_indices] = self.mean_hand_rotmats[None]
        if self.mean_face_rotmats is not None:
            full_local[:, self.face_joint_indices] = self.mean_face_rotmats[None]
        pose_feature = (full_local[:, 1:] - torch.eye(3, device=device)[None, None]).reshape(nF, -1)

        pose_offsets = torch.einsum("vcp,tp->tvc", self.posedirs, pose_feature)
        v_posed = self.v_shaped[None] + pose_offsets
        joints_rest = self.joint_rest[None].repeat(nF, 1, 1)
        posed_joints, global_joint_rots = batch_rigid_transform(
            full_local,
            joints_rest,
            self.full_joint_parents.to(device),
            self.full_root_idx,
        )
        # remove the skeleton offset of the root joint
        root_trans = joint_pos[:, self.skeleton.root_idx] - self.skeleton.neutral_joints[0:1]
        posed_joints = posed_joints + root_trans[:, None, :]

        fk_transform = torch.eye(4, device=device)[None, None].repeat(nF, self.full_joint_count, 1, 1)
        fk_transform[..., :3, :3] = global_joint_rots
        fk_transform[..., :3, 3] = posed_joints

        vertices = self.lbs(fk_transform, bind_vertices=v_posed)
        return vertices
