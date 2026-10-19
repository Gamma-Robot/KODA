# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint-following metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import Tensor

from kimodo.constraints import (
    EndEffectorConstraintSet,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from kimodo.tools import ensure_batched

from .base import Metric


class ContraintFollow(Metric):
    """Constraint-following metric dispatcher for kimodo constraint sets."""

    def __init__(
        self,
        skeleton,
        root_threshold: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skeleton = skeleton
        self.root_threshold = root_threshold

    @ensure_batched(posed_joints=4, constraints_lst=2, lengths=1)
    def _compute(
        self,
        posed_joints: Tensor,
        constraints_lst: Optional[List],
        lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        if not constraints_lst:
            return {}

        root_idx = self.skeleton.root_idx
        output = defaultdict(list)

        for posed_joints_s, constraint_lst_s, lengths_s in zip(posed_joints, constraints_lst, lengths):
            output_seq = defaultdict(list)
            for constraint in constraint_lst_s:
                frame_idx = constraint.frame_indices.to(device=posed_joints_s.device, dtype=torch.long)
                assert frame_idx.max() < lengths_s, "The constraint is defined outsite the lenght of the motion."
                if frame_idx.numel() == 0:
                    continue

                if isinstance(constraint, Root2DConstraintSet):
                    pred_root2d = posed_joints_s[frame_idx, root_idx][:, [0, 2]]
                    target = constraint.smooth_root_2d.to(posed_joints_s.device)

                    dist = torch.norm(pred_root2d - target, dim=-1)
                    output_seq["constraint_root2d_err"].append(dist)
                    hit = (dist <= self.root_threshold).float()
                    output_seq["constraint_root2d_acc"].append(hit)

                elif isinstance(constraint, FullBodyConstraintSet):
                    pred = posed_joints_s[frame_idx]
                    target = constraint.global_joints_positions.to(posed_joints_s.device)
                    err = torch.norm(pred - target, dim=-1)
                    output_seq["constraint_fullbody_keyframe"].append(err)

                elif isinstance(constraint, EndEffectorConstraintSet):
                    pos_idx = constraint.pos_indices.to(device=posed_joints_s.device, dtype=torch.long)
                    pred = posed_joints_s[frame_idx].index_select(1, pos_idx)
                    target = constraint.global_joints_positions.to(posed_joints_s.device).index_select(1, pos_idx)
                    err = torch.norm(pred - target, dim=-1)
                    output_seq["constraint_end_effector"].append(err)

            # in case we have several same constraints in the list
            for key, val in output_seq.items():
                output[key].append(torch.cat(val).mean())

        reduced = {}
        for key, vals in output.items():
            reduced[key] = torch.stack(vals, dim=0)
        return reduced
