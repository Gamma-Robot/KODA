import unittest

import mujoco
import numpy as np

from mujoco_robot_id_capture import robot_geom_ids, robot_id_from_segmentation


class RobotIdCaptureTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="pelvis"><geom name="robot_geom" type="sphere" size=".1"/></body>
                <body name="prop"><geom name="prop_geom" type="box" size=".1 .1 .1"/></body>
              </worldbody>
            </mujoco>
            """
        )

    def test_robot_subtree_and_geom_segmentation(self):
        geom_ids, body_ids, root_names = robot_geom_ids(self.model)
        robot_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_geom")
        prop_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "prop_geom")
        self.assertIn(int(robot_geom), geom_ids.tolist())
        self.assertNotIn(int(prop_geom), geom_ids.tolist())
        self.assertTrue(body_ids)
        self.assertTrue(any(name.endswith("pelvis") for name in root_names))

        seg = np.zeros((2, 3, 2), dtype=np.int32)
        seg[..., 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
        seg[0, 0, 0] = robot_geom
        seg[0, 1, 0] = prop_geom
        out = robot_id_from_segmentation(seg, geom_ids)
        self.assertEqual(int(out[0, 0]), int(robot_geom) + 1)
        self.assertEqual(int(out[0, 1]), 0)


if __name__ == "__main__":
    unittest.main()
