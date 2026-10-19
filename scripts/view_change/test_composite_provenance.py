import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from composite_st_depth_overlay import _load_robot_mask, _neighbor_dbg_median
from run_remaining_hole_eraser import _load_provenance, _remaining_holes


class CompositeProvenanceTest(unittest.TestCase):
    def test_vectorized_neighbor_median_matches_border_window(self):
        depth = np.arange(16, dtype=np.float32).reshape(4, 4)
        hole = np.zeros((4, 4), dtype=bool)
        hole[0, 0] = True
        median = _neighbor_dbg_median(depth, hole, radius=1)
        # The valid 2x2 window is [1, 4, 5], whose median is 4.
        self.assertEqual(float(median[0, 0]), 4.0)
        self.assertTrue(np.isnan(median[1, 1]))

    def test_16bit_object_id_mask_preserves_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "000000.png"
            ids = np.zeros((3, 3), dtype=np.uint16)
            ids[1, 1] = 1024
            self.assertTrue(cv2.imwrite(str(path), ids))
            robot = _load_robot_mask(path, ids.shape, {1024})
        self.assertTrue(robot[1, 1])
        self.assertEqual(int(robot.sum()), 1)

    def test_remaining_hole_provenance_uses_shared_codes(self):
        hole = np.zeros((3, 4), dtype=np.uint8)
        hole[0, 0] = 255
        hole[1, 1] = 255
        choice = np.zeros_like(hole)
        choice[1, 1] = 255
        remaining = _remaining_holes(hole, choice) > 127
        provenance = _load_provenance(None, hole.shape, hole > 127, choice > 127)
        provenance[remaining] = 4
        self.assertEqual(int(provenance[0, 0]), 4)  # synthetic
        self.assertEqual(int(provenance[1, 1]), 3)  # robot
        self.assertEqual(int(provenance[2, 2]), 1)  # observed


if __name__ == "__main__":
    unittest.main()
