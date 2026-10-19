# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert kimodo motion to AMASS/SMPL-X compatible parameters (axis-angle, Y-up or Z-up)."""

import os
from typing import Optional

import einops
import numpy as np
import torch

from kimodo.assets import skeleton_asset_path
from kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from kimodo.tools import ensure_batched, to_numpy, to_torch


def kimodo_y_up_to_amass_coord_rotation_matrix() -> np.ndarray:
    """3x3 rotation mapping Kimodo Y-up (+Z forward) to AMASS Z-up (+Y forward).

    Used by :func:`get_amass_parameters` and :func:`amass_arrays_to_kimodo_motion` (inverse).
    """
    y_up_to_z_up = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_z_180 = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.matmul(rot_z_180, y_up_to_z_up).astype(np.float32)


@ensure_batched(local_rot_mats=5, root_positions=3, lengths=1)
def get_amass_parameters(
    local_rot_mats,
    root_positions,
    skeleton,
    z_up=True,
):
    """Convert local rot mats and root positions to AMASS-style trans and pose_body; optional z_up
    coordinate transform.

    Our method generates motions with Y-up and +Z forward; if z_up=True, transform to Z-up and +Y
    forward as in AMASS.
    """
    # Our method generate motions with Y-up and +Z forward
    # if z_up = True, we transform this to: Z-up with +Y forward, as in AMASS
    # Remove the root offset; SMPL-X FK adds pelvis offset back.
    pelvis_offset = skeleton.neutral_joints[skeleton.root_idx].cpu().numpy()
    trans = root_positions - pelvis_offset

    root_rot_mats = to_numpy(local_rot_mats[:, :, 0])
    local_rot_axis_angle = to_numpy(matrix_to_axis_angle(to_torch(local_rot_mats)))
    pose_body = einops.rearrange(local_rot_axis_angle[:, :, 1:], "b t j d -> b t (j d)")

    # Optionally convert from Y-up to Z-up coordinates.
    if z_up:
        y_up_to_z_up = kimodo_y_up_to_amass_coord_rotation_matrix()
        root_rot_mats = np.matmul(y_up_to_z_up, root_rot_mats)
        trans = np.matmul(trans + pelvis_offset, y_up_to_z_up.T) - pelvis_offset

    root_orient = to_numpy(matrix_to_axis_angle(to_torch(root_rot_mats)))
    return trans, root_orient, pose_body


def amass_arrays_to_kimodo_motion(
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    skeleton,
    source_fps: float,
    *,
    z_up: bool = True,
):
    """Inverse of :func:`get_amass_parameters` for a single sequence (AMASS → Kimodo motion dict).

    Args:
        trans: ``(T, 3)`` AMASS root translation (same as ``trans`` in AMASS NPZ).
        root_orient: ``(T, 3)`` axis-angle root orientation in AMASS coordinates (z-up when ``z_up``).
        pose_body: ``(T, 63)`` body pose axis-angle (21 joints × 3).
        skeleton: :class:`~kimodo.skeleton.definitions.SMPLXSkeleton22` instance.
        source_fps: Source frame rate (Hz) of the AMASS recording.
        z_up: If ``True``, invert the same Y-up↔Z-up transform as ``get_amass_parameters(..., z_up=True)``.

    Returns:
        Motion dict compatible with :func:`kimodo.exports.motion_io.save_kimodo_npz`.
    """
    from kimodo.exports.motion_io import complete_motion_dict

    trans = np.asarray(trans, dtype=np.float32)
    root_orient = np.asarray(root_orient, dtype=np.float32)
    pose_body = np.asarray(pose_body, dtype=np.float32)
    if trans.ndim != 2 or trans.shape[-1] != 3:
        raise ValueError(f"trans must be (T, 3); got {trans.shape}")
    if root_orient.shape != trans.shape:
        raise ValueError(f"root_orient shape {root_orient.shape} must match trans {trans.shape}")
    t = trans.shape[0]
    if pose_body.shape != (t, 63):
        raise ValueError(f"pose_body must be (T, 63); got {pose_body.shape}")

    pelvis_offset = skeleton.neutral_joints[skeleton.root_idx].detach().cpu().numpy().astype(np.float32)
    device = skeleton.neutral_joints.device
    dtype = torch.float32

    Y_np = kimodo_y_up_to_amass_coord_rotation_matrix()
    if z_up:
        y_up_to_z_up = torch.from_numpy(Y_np).to(device=device, dtype=dtype)
        # trans_amass = root_kimodo @ Y.T - pelvis_offset  =>  root_kimodo = (trans_amass + pelvis_offset) @ Y
        root_positions_np = (trans + pelvis_offset) @ Y_np
    else:
        root_positions_np = trans + pelvis_offset

    root_positions = torch.from_numpy(root_positions_np).to(device=device, dtype=dtype)

    R_amass_root = axis_angle_to_matrix(torch.from_numpy(root_orient).to(device=device, dtype=dtype))
    if z_up:
        R_kimodo_root = torch.einsum("ij,tjk->tik", y_up_to_z_up.T, R_amass_root)
    else:
        R_kimodo_root = R_amass_root

    nb = skeleton.nbjoints
    if nb != 22:
        raise ValueError(f"Expected SMPL-X body skeleton with 22 joints; got {nb}")

    local_rot_mats = torch.zeros((t, nb, 3, 3), device=device, dtype=dtype)
    local_rot_mats[:, 0] = R_kimodo_root

    pose_aa = torch.from_numpy(pose_body.reshape(t, 21, 3)).to(device=device, dtype=dtype)
    local_rot_mats[:, 1:] = axis_angle_to_matrix(pose_aa.reshape(-1, 3)).reshape(t, 21, 3, 3)

    return complete_motion_dict(local_rot_mats, root_positions, skeleton, source_fps)


def amass_npz_to_kimodo_motion(npz_path: str, skeleton, source_fps: Optional[float] = None, *, z_up: bool = True):
    """Load an AMASS-style ``.npz`` and return a Kimodo motion dict.

    Args:
        npz_path: Path to AMASS NPZ (``trans``, ``root_orient``, ``pose_body``, ...).
        skeleton: SMPL-X skeleton instance.
        source_fps: Source frame rate (Hz); if ``None``, uses ``mocap_frame_rate``
            from the file when present, else ``30.0``.
        z_up: Same meaning as :func:`amass_arrays_to_kimodo_motion`.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        trans = np.asarray(data["trans"], dtype=np.float32)
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)
        pose_body = np.asarray(data["pose_body"], dtype=np.float32)
        if source_fps is None:
            source_fps = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data.files else 30.0

    return amass_arrays_to_kimodo_motion(trans, root_orient, pose_body, skeleton, source_fps, z_up=z_up)


class AMASSConverter:
    def __init__(
        self,
        fps,
        skeleton,
        beta_path=str(skeleton_asset_path("smplx22", "beta.npy")),
        mean_hands_path=str(skeleton_asset_path("smplx22", "mean_hands.npy")),
    ):
        self.fps = fps
        self.skeleton = skeleton
        # Load betas
        if os.path.exists(beta_path):
            # only use first 16 betas to match AMASS
            betas = np.load(beta_path)[:16]
        else:
            betas = np.zeros(16)

        # Load mean hands
        if os.path.exists(mean_hands_path):
            mean_hands = np.load(mean_hands_path)
        else:
            mean_hands = np.zeros(90)

        self.default_frame_params = {
            "pose_jaw": np.zeros(3),
            "pose_eye": np.zeros(6),
            "pose_hand": mean_hands,
        }
        self.output_dict_base = {
            "gender": "neutral",
            "surface_model_type": "smplx",
            "betas": betas,
            "num_betas": len(betas),
            "mocap_frame_rate": float(fps),
        }

    def convert_save_npz(self, output: dict, npz_path, z_up=True):
        trans, root_orient, pose_body = get_amass_parameters(
            output["local_rot_mats"],
            output["root_positions"],
            self.skeleton,
            z_up=z_up,
        )
        nb_frames = trans.shape[-2]

        amass_output_base = self.output_dict_base.copy()
        for key, val in self.default_frame_params.items():
            amass_output_base[key] = einops.repeat(val, "d -> t d", t=nb_frames)

        amass_output_base["mocap_time_length"] = nb_frames / self.fps
        self.save_npz(trans, root_orient, pose_body, amass_output_base, npz_path)

    def save_npz(self, trans, root_orient, pose_body, base_output, npz_path):
        shape = trans.shape
        if len(shape) == 3 and shape[0] == 1:
            # if only one motion, squeeze the data
            trans = trans[0]
            root_orient = root_orient[0]
            pose_body = pose_body[0]
            shape = trans.shape
        if len(shape) == 2:
            amass_output = {
                "trans": trans,
                "root_orient": root_orient,
                "pose_body": pose_body,
            } | base_output
            np.savez(npz_path, **amass_output)

        elif len(shape) == 3:
            # real batch of motions
            npz_path_base, ext = os.path.splitext(npz_path)
            for i in range(shape[0]):
                npz_path_i = npz_path_base + "_" + str(i).zfill(2) + ext
                self.save_npz(trans[i], root_orient[i], pose_body[i], base_output, npz_path_i)


# amass_output = {
#     "gender": "neutral",
#     "surface_model_type": "smplx",
#     "mocap_frame_rate": float(fps),
#     "mocap_time_length": len(motion) / float(fps)
#     "trans": trans,
#     "betas": betas,
#     "num_betas": len(betas),
#     "root_orient": np.array([T, 3]), # axis angle
#     "pose_body": np.array([T, 63]), # 63=21*3, axis angle 21 = 22 - root
#     "pose_hand": np.array([T, 90]), # 90=30*3=15*2*3 axis angle (load from mean_hands)
#     "pose_jaw": np.array([T, 3]), # all zeros is fine
#     "pose_eye": np.array([T, 6]), # all zeros is fine`
# }
