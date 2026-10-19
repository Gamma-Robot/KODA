#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_contact_aware_ik_targets import (  # noqa: E402
    matrix_to_wxyz,
    resample_rotation,
    wxyz_to_matrix,
)


class ContactAwareTargetTest(unittest.TestCase):
    def test_rotation_resampling_and_wxyz(self) -> None:
        source_time = np.asarray([0.0, 1.0])
        rotations = Rotation.from_euler("z", [0.0, 90.0], degrees=True).as_matrix()
        result = resample_rotation(source_time, rotations, np.asarray([0.5]))
        angle = Rotation.from_matrix(result[0]).as_euler("zyx", degrees=True)[0]
        self.assertAlmostEqual(angle, 45.0, places=6)
        quaternion = matrix_to_wxyz(result)[0]
        expected_xyzw = Rotation.from_euler("z", 45.0, degrees=True).as_quat()
        self.assertTrue(np.allclose(quaternion, expected_xyzw[[3, 0, 1, 2]]))
        self.assertTrue(np.allclose(wxyz_to_matrix(quaternion[None]), result))


if __name__ == "__main__":
    unittest.main()
