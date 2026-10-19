# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Foot skate and contact consistency metrics."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from kimodo.motion_rep.feature_utils import compute_vel_xyz
from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
from kimodo.skeleton import SkeletonBase
from kimodo.tools import ensure_batched

from .base import Metric


def get_four_contacts(fidx: list):
    if len(fidx) == 4:
        return fidx
    if len(fidx) == 6:
        # For soma77
        # remove "LeftToeEnd" and "RightToeEnd"
        fidx = fidx[:2] + fidx[3:5]
        return fidx
    raise ValueError("Expects 4 or 6 foot joints (heel/toe per foot)")


class FootSkateFromHeight(Metric):
    """When toe joint is near the floor, measures mean velocity of the toes."""

    def __init__(
        self,
        skeleton: SkeletonBase,
        fps: float,
        height_thresh: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height_thresh = height_thresh
        self.skeleton = skeleton
        self.fps = fps

    @ensure_batched(posed_joints=4, lengths=1)
    def _compute(
        self,
        posed_joints: Tensor,
        lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        fidx = self.skeleton.foot_joint_idx
        fidx = get_four_contacts(fidx)

        feet_pos = posed_joints[:, :, fidx]
        toe_pos = feet_pos[:, :, [1, 3]]

        toe_on_floor = (toe_pos[..., 1] < self.height_thresh)[:, :-1]  # y-up [B, T, 2] where [left right]

        dt = 1.0 / self.fps
        toe_vel = torch.norm(toe_pos[:, 1:] - toe_pos[:, :-1], dim=-1) / dt  # [B, nframes-1, 2]

        # compute err
        contact_toe_vel = toe_vel * toe_on_floor  # vel when corresponding toe is on ground

        # account for generated length
        # since they are velocities use length-1 to avoid inaccurate vel going one frame past len
        device = toe_on_floor.device
        len_mask = torch.arange(toe_on_floor.shape[1], device=device)[None, :, None].expand(toe_on_floor.shape) < (
            lengths[:, None, None] - 1
        )
        toe_on_floor = toe_on_floor * len_mask
        contact_toe_vel = contact_toe_vel * len_mask

        mean_vel = torch.sum(contact_toe_vel, (1, 2)) / (torch.sum(toe_on_floor, (1, 2)) + 1e-6)
        return {"foot_skate_from_height": mean_vel}


class FootSkateFromContacts(Metric):
    """Measures velocity of the toes and ankles when predicted to be in contact."""

    def __init__(
        self,
        skeleton: SkeletonBase,
        fps: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skeleton = skeleton
        self.fps = fps

    @ensure_batched(posed_joints=4, foot_contacts=3, lengths=1)
    def _compute(
        self,
        posed_joints: Tensor,
        foot_contacts: Tensor,
        lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        fidx = self.skeleton.foot_joint_idx
        fidx = get_four_contacts(fidx)

        feet_pos = posed_joints[:, :, fidx]
        dt = 1.0 / self.fps
        foot_vel = torch.norm(feet_pos[:, 1:] - feet_pos[:, :-1], dim=-1) / dt

        if foot_contacts.shape[-1] == 6:
            # For soma77
            # remove "LeftToeEnd" and "RightToeEnd"
            foot_contacts = foot_contacts[..., [0, 1, 3, 4]]

        foot_contacts = foot_contacts[:, :-1]
        vel_err = foot_vel * foot_contacts

        # account for generated length
        # since they are velocities use length-1 to avoid inaccurate vel going one frame past len
        device = foot_contacts.device
        len_mask = torch.arange(foot_contacts.shape[1], device=device)[None, :, None].expand(foot_contacts.shape) < (
            lengths[:, None, None] - 1
        )
        foot_contacts = foot_contacts * len_mask
        vel_err = vel_err * len_mask

        mean_vel = torch.sum(vel_err, (1, 2)) / (torch.sum(foot_contacts, (1, 2)) + 1e-6)  # mean over contacting frames

        # Compute max velocity error across all feet and frames (per batch)
        max_vel = vel_err.amax(dim=(1, 2))  # [B]

        return {
            "foot_skate_from_pred_contacts": mean_vel,
            "foot_skate_max_vel": max_vel,
        }


class FootSkateRatio(Metric):
    """Compute fraction of frames where the foot skates when it is on the ground.

    Inspired by GMD: https://github.com/korrawe/guided-motion-diffusion/blob/main/data_loaders/humanml/utils/metrics.py#L204
    """

    def __init__(
        self,
        skeleton: SkeletonBase,
        fps: float,
        height_thresh=0.05,
        vel_thresh=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height_thresh = height_thresh
        self.vel_thresh = vel_thresh

        self.skeleton = skeleton
        self.fps = fps

    @ensure_batched(posed_joints=4, foot_contacts=3, lengths=1)
    def _compute(
        self,
        posed_joints: Tensor,
        foot_contacts: Tensor,
        lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        fidx = self.skeleton.foot_joint_idx
        fidx = get_four_contacts(fidx)

        feet_pos = posed_joints[:, :, fidx]
        toe_pos = feet_pos[:, :, [1, 3]]

        toe_on_floor = toe_pos[..., 1] < self.height_thresh  # y-up [B, T, 2] where [left right]
        # current and next frame on floor to consider it in contact
        toe_on_floor = torch.logical_and(toe_on_floor[:, :-1], toe_on_floor[:, 1:])  # [B, T-1, 2]

        dt = 1.0 / self.fps
        toe_vel = torch.norm(toe_pos[:, 1:] - toe_pos[:, :-1], dim=-1) / dt  # [B, nframes-1, 2]

        # compute err
        contact_toe_vel = toe_vel * toe_on_floor  # vel when corresponding toe is on ground

        # account for generated length
        # since they are velocities use length-1 to avoid inaccurate vel going one frame past len
        device = toe_on_floor.device
        len_mask = torch.arange(toe_on_floor.shape[1], device=device)[None, :, None].expand(toe_on_floor.shape) < (
            lengths[:, None, None] - 1
        )
        toe_on_floor = toe_on_floor * len_mask
        contact_toe_vel = contact_toe_vel * len_mask

        # skating if velocity during contact > thresh
        toe_skate = contact_toe_vel > self.vel_thresh
        skate_ratio = torch.sum(toe_skate, (1, 2)) / (torch.sum(toe_on_floor, (1, 2)) + 1e-6)
        return {"foot_skate_ratio": skate_ratio}


class FootContactConsistency(Metric):
    """Measures consistency between heuristic detected foot contacts (from height and velocity) and
    predicted foot contacts.

    i.e. accuracy of how well predicted matches heuristic.
    """

    def __init__(
        self,
        skeleton: SkeletonBase,
        fps: float,
        vel_thresh: float = 0.15,
        height_thresh: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vel_thresh = vel_thresh
        self.height_thresh = height_thresh

        self.skeleton = skeleton
        self.fps = fps

    @ensure_batched(posed_joints=4, foot_contacts=3, lengths=1)
    def _compute(
        self,
        posed_joints: Tensor,
        foot_contacts: Tensor,
        lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        velocity = compute_vel_xyz(posed_joints, float(self.fps), lengths=lengths)
        heuristic_contacts = foot_detect_from_pos_and_vel(
            posed_joints,
            velocity,
            self.skeleton,
            self.vel_thresh,
            self.height_thresh,
        )

        if foot_contacts.shape[-1] == 6:
            # For soma77
            # remove "LeftToeEnd" and "RightToeEnd"
            foot_contacts = foot_contacts[..., [0, 1, 3, 4]]

        num_contacts = foot_contacts.shape[-1]
        incorrect = torch.logical_xor(heuristic_contacts, foot_contacts)
        # account for generated length
        # since they are velocities, use length-1 to avoid inaccurate vel going one frame past len
        device = foot_contacts.device
        len_mask = torch.arange(foot_contacts.shape[1], device=device)[None, :, None].expand(foot_contacts.shape) < (
            lengths[:, None, None] - 1
        )
        incorrect = incorrect * len_mask

        incorrect_ratio = torch.sum(incorrect, (1, 2)) / (num_contacts * (lengths - 1))
        accuracy = 1 - incorrect_ratio

        return {"foot_contact_consistency": accuracy}
