# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viser-based 3D viz: re-exports from viz submodules for backward compatibility."""

import os

from .constraint_ui import (
    ConstraintSet,
    EEJointsKeyframeSet,
    FullbodyKeyframeSet,
    RootKeyframe2DSet,
    build_constraint_set_table_markdown,
    update_interval,
)
from .gui import GuiElements
from .playback import CharacterMotion
from .scene import (
    DARK_THEME,
    LIGHT_THEME,
    SKIN_CACHE,
    Character,
    SkeletonMesh,
    WaypointMesh,
)


def load_example_cases(examples_base_dir):
    """List subdirectories of examples_base_dir as a name -> path dict."""
    example_dirs = os.listdir(examples_base_dir)
    example_names = sorted([d for d in example_dirs if os.path.isdir(os.path.join(examples_base_dir, d))])
    return {name: os.path.join(examples_base_dir, name) for name in example_names}


__all__ = [
    "Character",
    "CharacterMotion",
    "ConstraintSet",
    "DARK_THEME",
    "EEJointsKeyframeSet",
    "FullbodyKeyframeSet",
    "GuiElements",
    "LIGHT_THEME",
    "RootKeyframe2DSet",
    "SKIN_CACHE",
    "SkeletonMesh",
    "WaypointMesh",
    "build_constraint_set_table_markdown",
    "load_example_cases",
    "update_interval",
]
