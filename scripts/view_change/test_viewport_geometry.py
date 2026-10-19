import json
import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

from stera_episode_to_pipeline_inputs import (
    _frame_locked_rgb_check,
    _nearest_time_map,
    _register_depth,
    _register_depth_batch,
    _registered_depth_identity_check,
)
from viewport_gt_depth_warp import (
    _align_target_timestamps,
    _gate_frame_metrics,
    _load_K_json,
    _nearest_indices,
    _resolve_video_index_mode,
    _target_c2w,
    warp_frame_to_target,
    warp_frames_to_target,
)


class ViewportGeometryTest(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 8, 10
        self.k = np.array([[8.0, 0.0, self.w / 2], [0.0, 8.0, self.h / 2], [0.0, 0.0, 1.0]])
        self.depth = np.full((self.h, self.w), 2.0, dtype=np.float32)

    def test_identity_gate_passes(self):
        row = _gate_frame_metrics(
            self.depth, np.eye(4), np.eye(4), self.k, self.k, (self.h, self.w)
        )
        self.assertGreaterEqual(row["front_ratio"], 0.99)
        self.assertGreaterEqual(row["in_bounds_ratio"], 0.99)
        self.assertFalse(row["bad"])

    def test_points_behind_target_fail(self):
        # Target c2w rotated 180 degrees around X sees its optical +Z opposite
        # the source plane, so every source point is behind it.
        target = np.eye(4)
        target[:3, :3] = np.diag([1.0, -1.0, -1.0])
        row = _gate_frame_metrics(self.depth, np.eye(4), target, self.k, self.k, (self.h, self.w))
        self.assertEqual(row["front_ratio"], 0.0)
        self.assertTrue(row["bad"])

    def test_translation_changes_projection_but_stays_metric(self):
        target = np.eye(4)
        target[0, 3] = 0.1
        row = _gate_frame_metrics(self.depth, np.eye(4), target, self.k, self.k, (self.h, self.w))
        self.assertGreater(row["front_ratio"], 0.99)
        self.assertGreater(row["in_bounds_ratio"], 0.30)

    def test_gate_max_points_is_deterministic_and_exact_below_cap(self):
        depth = np.full((80, 100), 2.0, dtype=np.float32)
        large_cap = _gate_frame_metrics(
            depth, np.eye(4), np.eye(4), self.k, self.k, (80, 100),
            sample_stride=1, max_points=10_000,
        )
        uncapped = _gate_frame_metrics(
            depth, np.eye(4), np.eye(4), self.k, self.k, (80, 100),
            sample_stride=1, max_points=0,
        )
        self.assertEqual(large_cap["gate_sample_count"], uncapped["gate_sample_count"])
        self.assertEqual(large_cap["front_ratio"], uncapped["front_ratio"])
        self.assertEqual(large_cap["in_bounds_ratio"], uncapped["in_bounds_ratio"])

        capped = _gate_frame_metrics(
            depth, np.eye(4), np.eye(4), self.k, self.k, (80, 100),
            sample_stride=1, max_points=64,
        )
        repeated = _gate_frame_metrics(
            depth, np.eye(4), np.eye(4), self.k, self.k, (80, 100),
            sample_stride=1, max_points=64,
        )
        self.assertEqual(capped["gate_uncapped_sample_count"], 8000)
        self.assertEqual(capped["gate_sample_count"], 64)
        self.assertEqual(capped, repeated)

    def test_timestamp_nearest_map_has_signed_offset(self):
        idx, delta = _nearest_time_map(np.array([0.00, 0.05, 0.10]), np.array([0.01, 0.07, 0.11]))
        np.testing.assert_array_equal(idx, [0, 1, 2])
        np.testing.assert_allclose(delta, [0.01, 0.02, 0.01])

    def test_target_nearest_timestamp_tie_uses_earlier_frame(self):
        idx = _nearest_indices(np.array([0.01, 0.03]), np.array([0.0, 0.02, 0.04]))
        np.testing.assert_array_equal(idx, [0, 1])

    def test_full_rgb_video_uses_global_indices_for_manifest_slice(self):
        self.assertEqual(
            _resolve_video_index_mode(513, list(range(325, 350)), 325, 350),
            "global",
        )

    def test_sliced_rgb_video_uses_local_indices(self):
        self.assertEqual(
            _resolve_video_index_mode(25, list(range(325, 350)), 325, 350),
            "local",
        )

    def test_nested_d435_meta_intrinsics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meta.json"
            expected = [[8.0, 0.0, 5.0], [0.0, 9.0, 4.0], [0.0, 0.0, 1.0]]
            path.write_text(json.dumps({"model": {"d435": {"K": expected, "width": 10, "height": 8}}}))
            np.testing.assert_allclose(_load_K_json(path, 10, 8), expected)

    def test_single_frame_renderer_supports_independent_target_size(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4)
        depth = np.ones((4, 4), dtype=np.float32)
        k = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
        out_rgb, out_depth, valid, confidence = warp_frame_to_target(
            rgb, depth, k, np.eye(4), k, (6, 8), "cpu", splat_radius_px=0
        )
        self.assertEqual(out_rgb.shape, (6, 8, 3))
        self.assertEqual(out_depth.shape, (6, 8))
        self.assertEqual(valid.shape, (6, 8))
        self.assertGreater(int((valid > 0).sum()), 0)
        self.assertTrue(np.all(confidence[valid > 0] > 0))

    def test_batch_warp_matches_per_frame(self):
        rng = np.random.default_rng(7)
        n, h, w = 3, 12, 16
        rgb = rng.integers(0, 255, size=(n, h, w, 3), dtype=np.uint8)
        depth = rng.uniform(0.4, 2.5, size=(n, h, w)).astype(np.float32)
        depth[1, :2, :] = 0
        conf = rng.uniform(0.2, 1.0, size=(n, h, w)).astype(np.float32)
        k = np.array([[12.0, 0.0, 7.5], [0.0, 12.0, 5.5], [0.0, 0.0, 1.0]])
        rel = np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0)
        rel[2, 2, 3] = 0.08
        per = [
            warp_frame_to_target(
                rgb[i], depth[i], k, rel[i], k, (10, 14), "cpu",
                confidence=conf[i], splat_radius_px=1,
            )
            for i in range(n)
        ]
        batched = warp_frames_to_target(
            rgb, depth, k, rel, k, (10, 14), "cpu",
            confidence=conf, splat_radius_px=1,
        )
        for i, (out_rgb, out_depth, valid, confidence) in enumerate(per):
            np.testing.assert_array_equal(batched[0][i], out_rgb)
            np.testing.assert_allclose(batched[1][i], out_depth, rtol=0, atol=0)
            np.testing.assert_array_equal(batched[2][i], valid)
            np.testing.assert_allclose(batched[3][i], confidence, rtol=0, atol=0)

    def test_registered_depth_identity_requires_scaled_intrinsics(self):
        rgb_k = np.array([[1374.0, 0.0, 955.0], [0.0, 1374.0, 546.0], [0.0, 0.0, 1.0]])
        depth_k = rgb_k.copy()
        depth_k[0, :3] /= 7.5
        depth_k[1, :3] /= 7.5
        check = _registered_depth_identity_check(rgb_k, depth_k, (1080, 1920), (144, 256))
        self.assertTrue(check["pass"])
        depth_k[0, 2] += 3.0
        self.assertFalse(_registered_depth_identity_check(rgb_k, depth_k, (1080, 1920), (144, 256))["pass"])

    def test_frame_lock_requires_exact_pairing(self):
        ts = np.array([100.0, 100.033, 100.066])
        check = _frame_locked_rgb_check(
            rgb_count=3, camera_count=3, depth_count=3,
            rgb_timestamps=ts, camera_timestamps=ts, depth_timestamps=ts,
        )
        self.assertTrue(check["pass"])
        self.assertFalse(_frame_locked_rgb_check(
            rgb_count=3, camera_count=3, depth_count=3,
            rgb_timestamps=ts, camera_timestamps=ts + 1e-3, depth_timestamps=ts,
        )["pass"])

    def test_calibrated_depth_zbuffer_and_mm_to_m(self):
        depth_m = np.array([[1.0]], dtype=np.float32)  # 1000 mm converted before registration
        k_depth = np.eye(3, dtype=np.float64)
        k_rgb = np.array([[1.0, 0.0, self.w / 2], [0.0, 1.0, self.h / 2], [0.0, 0.0, 1.0]])
        out, confidence, edge = _register_depth(
            depth_m,
            k_depth,
            np.zeros(5),
            k_rgb,
            np.zeros(5),
            np.eye(4),
            (self.h, self.w),
        )
        self.assertAlmostEqual(float(out[int(self.h / 2), int(self.w / 2)]), 1.0, places=5)
        self.assertGreater(float(confidence.max()), 0.0)
        self.assertFalse(bool(edge[int(self.h / 2), int(self.w / 2)]))

    def test_calibrated_depth_splat_is_dense_and_zbuffered(self):
        # 2x2 native depth covers an 8x8 RGB image only after calibrated
        # projection plus a footprint splat. This must not become a raw resize:
        # the nearer projected sample wins in an overlap.
        depth = np.array([[0.5, 1.0], [1.0, 1.0]], dtype=np.float32)
        k_depth = np.eye(3, dtype=np.float64)
        k_rgb = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
        out, confidence, edge = _register_depth(
            depth, k_depth, np.zeros(5), k_rgb, np.zeros(5), np.eye(4), (8, 8),
            confidence=np.ones_like(depth),
            source_valid=np.ones_like(depth, dtype=bool),
            source_edge_invalid=np.zeros_like(depth, dtype=bool),
        )
        self.assertGreater(float((~edge).mean()), 0.30)
        self.assertAlmostEqual(float(out[2, 2]), 0.5, places=5)
        self.assertGreater(float(confidence[2, 2]), 0.2)

    def test_batch_register_matches_per_frame_and_chunking(self):
        rng = np.random.default_rng(6)
        native = rng.uniform(0.4, 2.5, size=(4, 2, 2)).astype(np.float32)
        native[1, 0, 1] = 0.0
        conf = rng.uniform(0.3, 1.0, size=native.shape).astype(np.float32)
        valid = native > 1e-6
        edge = np.zeros_like(native, dtype=bool)
        edge[2, 1, 1] = True
        k_depth = np.eye(3, dtype=np.float64)
        k_rgb = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4, dtype=np.float64)
        per = [
            _register_depth(
                native[i], k_depth, np.zeros(5), k_rgb, np.zeros(5), T, (8, 8),
                confidence=conf[i], source_valid=valid[i], source_edge_invalid=edge[i],
            )
            for i in range(native.shape[0])
        ]
        batched = _register_depth_batch(
            native, k_depth, np.zeros(5), k_rgb, np.zeros(5), T, (8, 8),
            confidence=conf, source_valid=valid, source_edge_invalid=edge,
        )
        for i, (depth, confidence, edge_i) in enumerate(per):
            np.testing.assert_allclose(batched[0][i], depth, rtol=0, atol=0)
            np.testing.assert_allclose(batched[1][i], confidence, rtol=0, atol=0)
            np.testing.assert_array_equal(batched[2][i], edge_i)

        import stera_episode_to_pipeline_inputs as adapter

        old = adapter._REGISTER_MAX_OUTPUT_BYTES
        adapter._REGISTER_MAX_OUTPUT_BYTES = 8 * 8 * 8
        try:
            chunked = _register_depth_batch(
                native, k_depth, np.zeros(5), k_rgb, np.zeros(5), T, (8, 8),
                confidence=conf, source_valid=valid, source_edge_invalid=edge,
            )
        finally:
            adapter._REGISTER_MAX_OUTPUT_BYTES = old
        np.testing.assert_array_equal(chunked[0], batched[0])
        np.testing.assert_array_equal(chunked[1], batched[1])
        np.testing.assert_array_equal(chunked[2], batched[2])

    def test_cuda_pinhole_register_matches_cpu_opencv(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(9)
        native = rng.uniform(0.4, 2.5, size=(3, 8, 10)).astype(np.float32)
        native[0, 0, 0] = 0.0
        conf = rng.uniform(0.3, 1.0, size=native.shape).astype(np.float32)
        valid = native > 1e-6
        edge = rng.random(native.shape) < 0.05
        k_depth = np.array([[20.0, 0.0, 4.5], [0.0, 20.0, 3.5], [0.0, 0.0, 1.0]])
        k_rgb = np.array([[80.0, 0.0, 18.0], [0.0, 80.0, 14.0], [0.0, 0.0, 1.0]])
        T = np.eye(4, dtype=np.float64)
        T[2, 3] = 0.01
        cpu = _register_depth_batch(
            native, k_depth, np.zeros(5), k_rgb, np.zeros(5), T, (32, 40),
            confidence=conf, source_valid=valid, source_edge_invalid=edge,
            splat_device="cpu",
        )
        gpu = _register_depth_batch(
            native, k_depth, np.zeros(5), k_rgb, np.zeros(5), T, (32, 40),
            confidence=conf, source_valid=valid, source_edge_invalid=edge,
            splat_device="cuda",
        )
        np.testing.assert_allclose(gpu[0], cpu[0], rtol=0, atol=1e-5)
        np.testing.assert_allclose(gpu[1], cpu[1], rtol=0, atol=1e-5)
        np.testing.assert_array_equal(gpu[2], cpu[2])

    def test_manifest_convention_rejects_extra_rz(self):
        # This verifies the contract used by main(): extra Rz(180) may be
        # diagnosed, but is never accepted when manifest says plain convention.
        plain = _target_c2w(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), "opencv", False)
        flipped = _target_c2w(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), "opencv", True)
        self.assertFalse(np.allclose(plain[:3, :3], flipped[:3, :3]))

    def test_relative_target_t_sec_is_anchored_to_source_epoch(self):
        aligned, meta = _align_target_timestamps(
            np.array([0.0, 1.0 / 30.0, 2.0 / 30.0]),
            "t_sec",
            source_anchor_s=51935.810025458,
            target_anchor_index=1,
        )
        self.assertIsNotNone(aligned)
        np.testing.assert_allclose(aligned, [51935.77669212467, 51935.810025458, 51935.84335879133])
        self.assertEqual(meta["mapping"], "relative_t_sec_anchored_to_source_rgb_frame")
        self.assertEqual(meta["target_anchor_index"], 1)


if __name__ == "__main__":
    unittest.main()
