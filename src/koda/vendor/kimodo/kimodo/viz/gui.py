# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GUI element handles for the demo app."""

from dataclasses import dataclass

import viser


@dataclass
class GuiElements:
    gui_play_pause_button: viser.GuiInputHandle
    gui_next_frame_button: viser.GuiInputHandle
    gui_prev_frame_button: viser.GuiInputHandle
    gui_generate_button: viser.GuiInputHandle
    gui_model_fps: viser.GuiInputHandle[int]
    gui_timeline: viser.GuiInputHandle[int]
    gui_viz_skeleton_checkbox: viser.GuiInputHandle[bool]
    gui_viz_foot_contacts_checkbox: viser.GuiInputHandle[bool]
    gui_viz_skinned_mesh_checkbox: viser.GuiInputHandle[bool]
    gui_viz_skinned_mesh_opacity_slider: viser.GuiInputHandle[float]
    gui_camera_fov_slider: viser.GuiInputHandle[float]

    # generation controls
    gui_duration_slider: viser.GuiInputHandle[float]
    gui_num_samples_slider: viser.GuiInputHandle[int]
    gui_cfg_checkbox: viser.GuiCheckboxHandle
    gui_cfg_text_weight_slider: viser.GuiInputHandle[float]
    gui_cfg_constraint_weight_slider: viser.GuiInputHandle[float]
    gui_diffusion_steps_slider: viser.GuiInputHandle[int]
    gui_seed: viser.GuiInputHandle[int]
    gui_postprocess_checkbox: viser.GuiCheckboxHandle
    gui_root_margin: viser.GuiInputHandle[float]
    gui_real_robot_rotations_checkbox: viser.GuiInputHandle[bool]
    # appearance
    gui_dark_mode_checkbox: viser.GuiCheckboxHandle

    # which skinning method to use for SOMA
    gui_use_soma_layer_checkbox: viser.GuiCheckboxHandle
