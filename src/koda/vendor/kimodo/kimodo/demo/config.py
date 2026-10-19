# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from kimodo.assets import DEMO_EXAMPLES_ROOT
from kimodo.model.registry import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    FRIENDLY_NAMES,
    get_datasets,
    get_model_info,
    get_models_for_dataset_skeleton,
    get_short_key_from_display_name,
    get_skeleton_display_name,
    get_skeleton_display_names_for_dataset,
    get_skeleton_key_from_display_name,
    get_skeletons_for_dataset,
    get_versions_for_dataset_skeleton,
    resolve_to_short_key,
)

SERVER_NAME = os.environ.get("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "7860"))
HF_MODE = os.environ.get("HF_MODE", False)

# HF mode: user queue and session limit (override via env in Spaces)
MAX_ACTIVE_USERS = int(os.environ.get("MAX_ACTIVE_USERS", "5"))
MAX_SESSION_MINUTES = float(os.environ.get("MAX_SESSION_MINUTES", "5.0"))

DEFAULT_PLAYBACK_SPEED = 1.0
# default start duration is 6.0 sec, but model can handle up to 10 sec
DEFAULT_CUR_DURATION = 6.0
DEFAULT_PROMPT = "A person walks forward."
MIN_DURATION = 2.0
MAX_DURATION = 10.0

SHOW_TRANSITION_PARAMS = True
INIT_POSTPROCESSING = True
NB_TRANSITION_FRAMES = 5

LIGHT_THEME = dict(
    floor=(220, 220, 220),
    grid=(180, 180, 180),
)

# Dark theme: slightly lighter grid and floor for better visibility and less flat black
DARK_THEME = dict(
    floor=(48, 48, 52),
    grid=(105, 105, 110),
)

EXAMPLES_ROOT_DIR = str(DEMO_EXAMPLES_ROOT)

# Model list and paths from kimodo registry (all models: Kimodo + TMR)
MODEL_NAMES = tuple(AVAILABLE_MODELS)
MODEL_EXAMPLES_DIRS = {name: os.path.join(EXAMPLES_ROOT_DIR, name) for name in MODEL_NAMES}
# Display labels for backward compatibility (short_key -> display name)
MODEL_LABELS = {name: FRIENDLY_NAMES.get(name, f"Model ({name})") for name in MODEL_NAMES}
MODEL_LABEL_TO_NAME = {label: name for name, label in MODEL_LABELS.items()}

# -----------------------------------------------------------------------------
# Demo UI copy
# -----------------------------------------------------------------------------

DEMO_UI_QUICK_START_CORE_MD = """
### Camera
- **Left-drag**: rotate
- **Right-drag**: pan
- **Scroll**: zoom

### Playback
- **Space** to play/pause
- **←/→** to step frames, or click the frame number.
- **Scroll up/down** in the timeline: move left/right
- **Shift + scroll** in the timeline: zoom in/out

### Prompts
- **Double-click** a text prompt to edit it.
- **Click and drag** the right edge of a prompt box to extend/shorten it.
- **Click empty space** to add a prompt.
- **Right-click** a prompt to delete it.

### Generate
- Go to the **Generate** tab to modify options
- It is also possible to **load** examples
- Click **Generate** to generate a motion

### Constraints
- This is **optional**: should be use after a first generation
- **Click** in the timeline tracks (Full-Body / 2D root etc) to add a constraint.
- **Right-click** on a constraint to delete it.
- To **edit** a constraint:
    - Move playback to the target frame
    - Click **Enter Editing Mode** in the Constraints tab.
"""

DEMO_UI_QUICK_START_MODAL_MD = (
    DEMO_UI_QUICK_START_CORE_MD
    + """

See the **Instructions** tab for the full user manual.
"""
)

DEMO_UI_INSTRUCTIONS_TAB_MD = (
    """
## How to Use This Demo

"""
    + DEMO_UI_QUICK_START_CORE_MD
    + """

---

### Generating Motion (step-by-step)

1. **Edit the text prompts** in the timeline (e.g., "A person walks forward.")
2. **Modify the duration** by moving the right edge of each prompts (2–10 seconds)
3. **Add constraints** (optional) to control the motion:
   - Click **Enter Editing Mode** to adjust the character pose
   - Use the timeline to place keyframes or intervals in constraint tracks (see below)
4. **Click Generate** to create the motion
5. If generating multiple samples, **click on a mesh** to select which one to keep

### Timeline Editing

**Adding Constraints:**
1. Click anywhere on the timeline to add a keyframe at that frame. The keyframe is created based on the current character motion.
2. Ctrl/Cmd+click+drag to add an interval constraint, or expand a keyframe into an interval
3. Enter editing mode with the **Enter Editing Mode** button to adjust character pose before/after adding constraints.

**Constraint Types:**
- **Full-Body**: constrains the entire character pose
- **2D Root**: constrains the character's path on the ground plane
  - Enable **Densify** to create a continuous path
- **End-Effectors**: constrains hands and feet positions
  - Use separate tracks for Left/Right Hand/Foot


**Moving & Deleting:**
- **Drag keyframes/intervals** to move them to different frames
- **Right-click** a keyframe or interval to delete it
- Use **Clear All Constraints** to remove everything

**Tips:**
- The posing skeleton becomes visible in editing mode for precise positioning
- Use **Snap to constraint** to align the current frame to a constraint

### Saving & Loading

You can save the current constraints or current motion to load in later from the Load/Save menu.
Saving an **Example** will save the full constraints, motion, and generation metadata.

### Visualization Options

Switch to the **Visualize** tab to:
- Toggle mesh and skeleton visibility
- Adjust mesh opacity
- Show/hide foot contact indicators
- Switch between light and dark modes
"""
)
