# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import mujoco
import mujoco.viewer
import numpy as np

from kimodo.assets import skeleton_asset_path

qpos = np.loadtxt("motion.csv", delimiter=",")
model = mujoco.MjModel.from_xml_path(str(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
data = mujoco.MjData(model)

fps = 30  # adjust to your intended playback rate

with mujoco.viewer.launch_passive(model, data) as viewer:
    # loop the motion
    while viewer.is_running():
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / fps)
