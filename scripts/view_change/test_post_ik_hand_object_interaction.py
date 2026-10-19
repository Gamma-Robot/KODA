#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_post_ik_hand_object_interaction import interpolate_pose


class PoseInterpolationTest(unittest.TestCase):
    def test_interpolates_position_and_orientation(self) -> None:
        source_time = np.asarray([0.0, 1.0])
        position = np.asarray([[0.0, 0.0, 0.0], [2.0, -1.0, 4.0]])
        rotation = Rotation.from_euler("z", [0.0, 90.0], degrees=True).as_matrix()
        result_position, result_rotation = interpolate_pose(
            source_time, position, rotation, np.asarray([0.5])
        )
        self.assertTrue(np.allclose(result_position[0], [1.0, -0.5, 2.0]))
        expected = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
        self.assertTrue(np.allclose(result_rotation[0], expected, atol=1e-9))

    def test_rejects_extrapolation(self) -> None:
        with self.assertRaises(ValueError):
            interpolate_pose(
                np.asarray([0.0, 1.0]),
                np.zeros((2, 3)),
                np.repeat(np.eye(3)[None], 2, axis=0),
                np.asarray([1.1]),
            )


if __name__ == "__main__":
    unittest.main()
