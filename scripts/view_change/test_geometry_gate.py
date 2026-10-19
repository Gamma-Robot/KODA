import unittest

import numpy as np

from geometry_gate import compute_depth_confidence, compute_depth_confidence_batch, evaluate_geometry_gate
from quality_report import aggregate_episode


class GeometryGateTest(unittest.TestCase):
    def test_confidence_rejects_invalid_and_edges(self):
        depth = np.ones((12, 12), dtype=np.float32)
        depth[5, 5] = 0
        depth[:, 6:] = 3.0
        confidence, valid, edge = compute_depth_confidence(depth)
        self.assertEqual(confidence.shape, depth.shape)
        self.assertFalse(valid[5, 5])
        self.assertTrue(edge[:, 6].any())

    def test_batch_confidence_matches_per_frame(self):
        rng = np.random.default_rng(2)
        depth = rng.uniform(0.4, 4.0, size=(5, 12, 16)).astype(np.float32)
        depth[0, 3, 3] = 0
        depth[2, :, 8:] = 3.5
        rgb = rng.integers(0, 255, size=(5, 12, 16, 3), dtype=np.uint8)
        sensor = rng.uniform(0.0, 1.0, size=depth.shape).astype(np.float32)
        sensor[1] *= 200.0

        derived = [compute_depth_confidence(depth[i], rgb=rgb[i], previous_depth=None if i == 0 else depth[i - 1]) for i in range(5)]
        batched = compute_depth_confidence_batch(depth, rgb=rgb, use_temporal=True)
        for i, (conf, valid, edge) in enumerate(derived):
            np.testing.assert_allclose(batched[0][i], conf, rtol=0, atol=0)
            np.testing.assert_array_equal(batched[1][i], valid)
            np.testing.assert_array_equal(batched[2][i], edge)

        sensor_rows = [compute_depth_confidence(depth[i], sensor_confidence=sensor[i]) for i in range(5)]
        sensor_batch = compute_depth_confidence_batch(depth, sensor_confidence=sensor)
        for i, (conf, valid, edge) in enumerate(sensor_rows):
            np.testing.assert_allclose(sensor_batch[0][i], conf, rtol=0, atol=0)
            np.testing.assert_array_equal(sensor_batch[1][i], valid)
            np.testing.assert_array_equal(sensor_batch[2][i], edge)

    def test_gate_passes_good_metrics(self):
        report = evaluate_geometry_gate(
            [{
                "valid_ratio": 0.8,
                "front_ratio": 0.99,
                "in_bounds_ratio": 0.8,
                "reprojection_error_px": 1.0,
                "timing_offset_ms": 2.0,
            }],
            timing_available=True,
        )
        self.assertEqual(report["status"], "pass")

    def test_gate_fails_front_ratio_and_bad_run(self):
        rows = [{"valid_ratio": 0.8, "front_ratio": 0.2, "in_bounds_ratio": 0.8, "bad": True} for _ in range(6)]
        report = evaluate_geometry_gate(rows, timing_available=True)
        self.assertEqual(report["status"], "fail")
        self.assertIn("target_front_ratio", report["failed_checks"])
        self.assertIn("max_bad_frame_run", report["failed_checks"])

    def test_quality_report_accepts_bypassed_gate(self):
        report = aggregate_episode(
            [{"valid_ratio": 0.9, "depth_relative_error": 0.05, "reprojection_error_px": 1.0}],
            calibration_status="complete",
            geometry_status="bypassed",
        )
        self.assertNotIn("geometry_status", report["hard_failures"])


if __name__ == "__main__":
    unittest.main()

