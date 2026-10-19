# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import viser
from kimodo.constraints import (
    TYPE_TO_CLASS,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from kimodo.exports.mujoco import apply_g1_real_robot_projection
from kimodo.skeleton import G1Skeleton34, SOMASkeleton30
from kimodo.tools import seed_everything

from .embedding_cache import CachedTextEncoder
from .state import ClientSession, ModelBundle


def compute_model_constraints_lst(
    session: ClientSession,
    model_bundle: ModelBundle,
    num_frames: int,
    device: str,
):
    """Compute the lst of constraints for the model based on the constraints in viser."""
    assert len(session.motions) == 1, "Only one motion allowed for constrained generation"
    if not session.constraints:
        return []

    model_skeleton = model_bundle.model.skeleton
    # For SOMA, UI uses somaskel77; extract 30-joint subset for the model
    use_skel_slice = isinstance(model_skeleton, SOMASkeleton30) and session.skeleton.nbjoints != model_skeleton.nbjoints
    skel_slice = model_skeleton.get_skel_slice(session.skeleton) if use_skel_slice else None

    dense_smooth_root_pos_2d = None
    if session.constraints["2D Root"].dense_path:
        # get the full 2d root
        dense_smooth_root_pos_2d = session.constraints["2D Root"].get_constraint_info(device=device)["root_pos"][
            :, [0, 2]
        ]

    model_constraints = []
    for track_name, constraint in session.constraints.items():
        constraint_info = constraint.get_constraint_info(device=device)
        frame_idx = constraint_info["frame_idx"]
        # drop any constraints outside the generation range
        valid_info = [(i, fi) for i, fi in enumerate(frame_idx) if fi < num_frames]
        valid_idx = [i for i, _ in valid_info]
        valid_frame_idx = [fi for _, fi in valid_info]

        if len(valid_frame_idx) == 0:
            continue

        frame_indices = torch.tensor(valid_frame_idx)
        if track_name == "2D Root":
            smooth_root_pos_2d = constraint_info["root_pos"][valid_idx][:, [0, 2]].to(device)
            # same as "smooth_root_2d"
            model_constraints.append(
                Root2DConstraintSet(
                    model_skeleton,
                    frame_indices,
                    smooth_root_pos_2d,
                )
            )
        elif track_name == "Full-Body":
            constraint_joints_pos = constraint_info["joints_pos"][valid_idx].to(device)
            constraint_joints_rot = constraint_info["joints_rot"][valid_idx].to(device)
            if skel_slice is not None:
                constraint_joints_pos = constraint_joints_pos[:, skel_slice]
                constraint_joints_rot = constraint_joints_rot[:, skel_slice]

            smooth_root_pos_2d = None
            if dense_smooth_root_pos_2d is not None:
                smooth_root_pos_2d = dense_smooth_root_pos_2d[frame_indices]

            model_constraints.append(
                FullBodyConstraintSet(
                    model_skeleton,
                    frame_indices,
                    constraint_joints_pos,
                    constraint_joints_rot,
                    smooth_root_2d=smooth_root_pos_2d,
                )
            )
        elif track_name == "End-Effectors":
            constraint_joints_pos = constraint_info["joints_pos"][valid_idx].to(device)
            constraint_joints_rot = constraint_info["joints_rot"][valid_idx].to(device)
            if skel_slice is not None:
                constraint_joints_pos = constraint_joints_pos[:, skel_slice]
                constraint_joints_rot = constraint_joints_rot[:, skel_slice]

            end_effector_type_set_lst = [
                end_effector_type_set
                for i, end_effector_type_set in enumerate(constraint_info["end_effector_type"])
                if i in valid_idx
            ]

            # regroup the end effector data by type
            cls_idx = defaultdict(list)
            for idx, end_effector_type_set in enumerate(end_effector_type_set_lst):
                for end_effector_type in end_effector_type_set:
                    cls_idx[TYPE_TO_CLASS[end_effector_type]].append(idx)

            for cls, lst_idx in cls_idx.items():
                frame_indices_cls = frame_indices[lst_idx]
                smooth_root_pos_2d = None
                if dense_smooth_root_pos_2d is not None:
                    smooth_root_pos_2d = dense_smooth_root_pos_2d[frame_indices_cls]

                constraint_joints_pos_el = constraint_joints_pos[lst_idx]
                constraint_joints_rot_el = constraint_joints_rot[lst_idx]

                model_constraints.append(
                    cls(
                        model_skeleton,
                        frame_indices_cls,
                        constraint_joints_pos_el,
                        constraint_joints_rot_el,
                        smooth_root_2d=smooth_root_pos_2d,
                    )
                )
        else:
            raise ValueError(f"Unsupported constraint type: {constraint.display_name}")
    return model_constraints


def generate(
    *,
    client: viser.ClientHandle,
    session: ClientSession,
    model_bundle: ModelBundle,
    prompts: list[str],
    num_frames: list[int],
    num_samples: int,
    seed: int,
    diffusion_steps: int,
    cfg_weight: Optional[list[float]] = None,
    cfg_type: Optional[str] = None,
    postprocess_parameters: Optional[dict] = None,
    transitions_parameters: Optional[dict] = None,
    real_robot_rotations: bool = False,
    device: str,
    clear_motions,
    add_character_motion,
) -> None:
    client_id = client.client_id
    print(
        f"Generating {num_samples} samples for a total of {sum(num_frames)} frames with those prompt: {prompts} (client {client_id})"
    )

    seed_everything(seed)

    model_constraints = compute_model_constraints_lst(session, model_bundle, sum(num_frames), device)
    cfg_weight = cfg_weight or [2.0, 2.0]
    postprocess_parameters = postprocess_parameters or {}
    transitions_parameters = transitions_parameters or {}

    encoder = getattr(model_bundle.model, "text_encoder", None)
    if isinstance(encoder, CachedTextEncoder):
        with encoder.session_context(session):
            pred_joints_output = model_bundle.model(
                prompts,
                num_frames,
                diffusion_steps,
                multi_prompt=True,
                constraint_lst=model_constraints,
                cfg_weight=cfg_weight,
                num_samples=num_samples,
                cfg_type=cfg_type,
                **(postprocess_parameters | transitions_parameters),
            )  # [B, T, motion_rep_dim]
    else:
        pred_joints_output = model_bundle.model(
            prompts,
            num_frames,
            diffusion_steps,
            multi_prompt=True,
            constraint_lst=model_constraints,
            cfg_weight=cfg_weight,
            num_samples=num_samples,
            cfg_type=cfg_type,
            **(postprocess_parameters | transitions_parameters),
        )  # [B, T, motion_rep_dim]

    joints_pos = pred_joints_output["posed_joints"]  # [B, T, J, 3]
    joints_rot = pred_joints_output["global_rot_mats"]
    foot_contacts = pred_joints_output.get("foot_contacts")

    # Optionally project G1 to real robot DoF (1-DoF per joint, clamped) for display.
    if real_robot_rotations and isinstance(session.skeleton, G1Skeleton34):
        joints_pos, joints_rot = apply_g1_real_robot_projection(
            session.skeleton,
            pred_joints_output["posed_joints"],
            pred_joints_output["global_rot_mats"],
            clamp_to_limits=True,
        )

    # Display on characters (callbacks keep this module UI-agnostic).
    clear_motions(client_id)
    # Keep one sample centered at the origin so constraints align.
    spread_factor = 1.0  # meters
    center_idx = num_samples // 2
    x_trans = (np.arange(num_samples) - center_idx) * spread_factor
    for i in range(num_samples):
        cur_joints_pos = joints_pos[i]
        cur_joints_pos[..., 0] += x_trans[i]
        add_character_motion(
            client,
            session.skeleton,
            cur_joints_pos,
            joints_rot[i],
            foot_contacts[i],
        )
