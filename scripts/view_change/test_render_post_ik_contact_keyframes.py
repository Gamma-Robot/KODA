#!/usr/bin/env python3

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from render_post_ik_contact_keyframes import all_object_ids


class ObjectIdDecodeTest(unittest.TestCase):
    def test_preserves_all_geom_ids_and_reserves_zero(self) -> None:
        segmentation = np.zeros((1, 3, 2), dtype=np.int32)
        segmentation[0, 0] = [4, int(mujoco.mjtObj.mjOBJ_GEOM)]
        segmentation[0, 1] = [8, int(mujoco.mjtObj.mjOBJ_BODY)]
        segmentation[0, 2] = [0, -1]
        result = all_object_ids(segmentation)
        self.assertEqual(result.tolist(), [[5, 0, 0]])


if __name__ == "__main__":
    unittest.main()
