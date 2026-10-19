#!/usr/bin/env python3

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from smooth_three_point_reference import repair_transient_wrist_outliers


class TransientWristOutlierRepairTest(unittest.TestCase):
    def repair(self, timestamps, position, rotation):
        return repair_transient_wrist_outliers(
            np.asarray(timestamps, dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.asarray(rotation, dtype=np.float64),
            gap_factor=3.0,
            max_event_duration_s=1.1,
            position_floor_m=0.05,
            orientation_floor_deg=45.0,
        )

    def test_repairs_bounded_jump_out_and_return(self):
        timestamps = np.arange(8, dtype=np.float64) * 0.02
        position = np.column_stack((timestamps, np.zeros((8, 2))))
        position[3] += np.asarray([0.20, 0.10, 0.0])
        rotation = Rotation.from_euler("z", np.zeros(8), degrees=True).as_matrix()
        rotation[3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

        repaired_position, repaired_rotation, report = self.repair(
            timestamps, position, rotation
        )

        self.assertEqual(report["repaired_frame_indices"], [3])
        np.testing.assert_allclose(repaired_position[3], [0.06, 0.0, 0.0])
        np.testing.assert_allclose(repaired_rotation[3], np.eye(3), atol=1.0e-12)

    def test_does_not_repair_persistent_pose_change(self):
        timestamps = np.arange(8, dtype=np.float64) * 0.02
        position = np.zeros((8, 3), dtype=np.float64)
        position[3:, 0] = 0.20
        rotation = Rotation.from_euler(
            "z", [0, 0, 0, 90, 90, 90, 90, 90], degrees=True
        ).as_matrix()

        repaired_position, repaired_rotation, report = self.repair(
            timestamps, position, rotation
        )

        self.assertEqual(report["repaired_frame_indices"], [])
        np.testing.assert_array_equal(repaired_position, position)
        np.testing.assert_allclose(repaired_rotation, rotation, atol=1.0e-12)

    def test_does_not_pair_anomalous_edges_across_timestamp_gap(self):
        timestamps = np.asarray([0.00, 0.02, 0.04, 1.00, 1.02, 1.04])
        position = np.zeros((6, 3), dtype=np.float64)
        position[2:4, 0] = 0.20
        rotation = np.repeat(np.eye(3)[None], 6, axis=0)

        repaired_position, _, report = self.repair(timestamps, position, rotation)

        self.assertEqual(report["repaired_frame_indices"], [])
        np.testing.assert_array_equal(repaired_position, position)


if __name__ == "__main__":
    unittest.main()
