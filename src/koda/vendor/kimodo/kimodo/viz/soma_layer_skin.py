# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA layer-based skinning for visualization (SOMASkeleton30 / SOMASkeleton77)."""

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from soma import SomaLayer as SOMALayer

from kimodo.assets import SOMA_ASSETS_ROOT
from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77, global_rots_to_local_rots

SOMA_MHR_NEUTRAL_PATH = "somaskel30/soma_base_fit_mhr_params.npz"


class SOMASkin:
    def __init__(
        self,
        skeleton,
    ):
        self.skeleton = skeleton

        assert isinstance(
            skeleton, (SOMASkeleton30, SOMASkeleton77)
        ), "SOMASkin currently only supports SOMASkeleton30 or SOMASkeleton77"
        assert skeleton.neutral_joints is not None, "The skeleton must have neutral joints instantiated"

        device = skeleton.neutral_joints.device
        device = "cpu"
        self.device = device

        self._soma_model = SOMALayer(
            identity_model_type="mhr",
            device=device,
        )
        self.faces = self._soma_model.faces

        neutral_mhr_path = Path(skeleton.folder).parent / SOMA_MHR_NEUTRAL_PATH
        neutral_mhr = np.load(neutral_mhr_path)

        # one time call to prepare the identity
        self.soma_identity = torch.from_numpy(neutral_mhr["identity_params"])
        self.scale_params = torch.from_numpy(neutral_mhr["scale_params"])
        self._soma_model.prepare_identity(self.soma_identity.to(device), scale_params=self.scale_params.to(device))

        # dummy output to get bind_vertices
        transl = torch.zeros(1, 3, device=device)

        self._full_skeleton = SOMASkeleton77()
        self.skel_slice = self.skeleton.get_skel_slice(self._full_skeleton)

        self.bind_vertices = self.soma_model_pose(
            self._full_skeleton.relaxed_hands_rest_pose[None],
            transl=transl,
            pose2rot=False,
        )["vertices"][0]

    def soma_model_pose(self, *args, **kwargs):
        with torch.inference_mode():
            return self._soma_model.pose(*args, **kwargs)

    def skin(self, joint_rotmat, joint_pos, rot_is_global=False):
        """
        joint_rotmat: [T, J, 3, 3] local or global joint rotation matrices
        joint_pos: [T, J, 3] global joint positions
        rot_is_global: bool, if True, joint_rotmat is global rotation matrices, otherwise it is local rotation matrices and FK is performed internally
        """

        nF, nJ = joint_pos.shape[:2]

        if rot_is_global:
            local_joint_rots_mats_subset = global_rots_to_local_rots(joint_rotmat, self.skeleton)
        else:
            local_joint_rots_mats_subset = joint_rotmat

        if nJ != self._full_skeleton.nbjoints:
            local_joint_rots_mats = self.skeleton.to_SOMASkeleton77(local_joint_rots_mats_subset)
        else:
            local_joint_rots_mats = local_joint_rots_mats_subset

        # remove the skeleton offset of the root joint
        transl = joint_pos[:, self.skeleton.root_idx] - self.skeleton.neutral_joints[0:1]

        output = self.soma_model_pose(
            local_joint_rots_mats.to(device=self.device, dtype=torch.float32),
            transl=transl.to(device=self.device, dtype=torch.float32),
            pose2rot=False,
        )
        return output["vertices"]
