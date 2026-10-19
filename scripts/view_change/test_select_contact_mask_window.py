#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_contact_mask_window import (  # noqa: E402
    apply_blue_cup_filter,
    gather_counts_from_uv,
    nearest_depth_indices,
    point_count,
    point_count_batch,
    project_uv_batch,
    runs,
    windows_from_counts,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _intrinsics(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    fx = fy = 200.0
    depth_k = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rgb_k = np.array(
        [[400.0, 0.0, 16.0], [0.0, 400.0, 12.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return depth_k, rgb_k


class MaskWindowBatchTest(unittest.TestCase):
    def test_nearest_depth_indices_match_serial_argmin(self) -> None:
        hand_t = np.asarray([0.00, 0.04, 0.07, 0.11], dtype=np.float64)
        depth_t = np.asarray([0.00, 0.03, 0.09], dtype=np.float64)
        expected = np.array(
            [int(np.argmin(np.abs(depth_t - query))) for query in hand_t],
            dtype=np.int64,
        )
        self.assertTrue(np.array_equal(nearest_depth_indices(hand_t, depth_t, 4), expected))

    def test_windows_and_keyframe_contract(self) -> None:
        # Gap of 8 empty frames (> max_gap=6) splits two otherwise-valid runs.
        counts = np.asarray(
            [0, 120, 130, 140, 10, 0, 0, 0, 0, 0, 0, 0, 150, 160, 170, 180],
            dtype=np.int64,
        )
        windows = windows_from_counts(counts, min_points=100, max_gap=6, min_frames=3)
        self.assertEqual([item[2:] for item in windows], [(1, 3), (12, 15)])
        support, length, start, end = max(windows, key=lambda item: (item[1], item[0]))
        self.assertEqual((start, end, length), (12, 15, 4))
        keyframe = start + int(np.argmax(counts[start : end + 1]))
        self.assertEqual(keyframe, 15)
        self.assertEqual(runs([1, 2, 3, 11], max_gap=6), [(1, 3), (11, 11)])

    def test_blue_hsv_matches_open_cv_split_gate(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[1, 1] = (20, 80, 180)  # saturated blue-ish BGR would differ; RGB path
        rgb[2, 2] = (200, 200, 200)
        mask = np.full((4, 4), 255, dtype=np.uint8)
        filtered = apply_blue_cup_filter(mask, rgb, None)
        import cv2

        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        blue = (
            (hue >= 85) & (hue <= 130) & (saturation >= 35) & (value >= 20)
        )
        expected = np.where(blue & (mask > 0), 255, 0).astype(np.uint8)
        self.assertTrue(np.array_equal(filtered, expected))
        self.assertTrue(np.array_equal(apply_blue_cup_filter(mask, None, None), mask))

    def test_numpy_batch_matches_serial_point_count(self) -> None:
        rng = np.random.default_rng(20260830)
        height, width = 12, 16
        mask_h, mask_w = 24, 32
        depth_k, rgb_k = _intrinsics(width, height)
        rows, columns = np.indices((height, width), dtype=np.float64)
        rows = rows.reshape(-1)
        columns = columns.reshape(-1)
        depths = rng.uniform(200.0, 1800.0, size=(7, height, width))
        depths[2, 3, 4] = np.nan
        depths[4] = 50.0
        masks = (rng.random((7, mask_h, mask_w)) > 0.55).astype(np.uint8) * 255
        serial = np.asarray(
            [
                point_count(masks[i], depths[i], depth_k, rgb_k, columns, rows)
                for i in range(7)
            ],
            dtype=np.int64,
        )
        batched = point_count_batch(masks, depths, depth_k, rgb_k)
        self.assertTrue(np.array_equal(serial, batched), f"{serial} vs {batched}")
        uv, valid = project_uv_batch(depths, depth_k, rgb_k)
        gathered = gather_counts_from_uv(masks, uv, valid)
        self.assertTrue(np.array_equal(serial, gathered))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA required to compare TorchPointCounter batch vs per-frame",
    )
    def test_cuda_batch_matches_per_frame_count(self) -> None:
        from select_contact_mask_window import TorchPointCounter

        rng = np.random.default_rng(7)
        height, width = 10, 14
        mask_h, mask_w = 20, 28
        depth_k, rgb_k = _intrinsics(width, height)
        depths = rng.uniform(150.0, 2200.0, size=(5, height, width))
        depths[1, 0, 0] = np.nan
        masks = (rng.random((5, mask_h, mask_w)) > 0.5).astype(np.uint8) * 255
        counter = TorchPointCounter(
            depth_k, rgb_k, (height, width), torch.device("cuda")
        )
        serial = np.asarray(
            [counter.count(masks[i], depths[i]) for i in range(5)],
            dtype=np.int64,
        )
        batched = counter.count_batch(masks, depths)
        self.assertTrue(np.array_equal(serial, batched), f"{serial} vs {batched}")
        uv, valid = counter.project_uv(depths)
        gathered = counter.gather_counts(masks, uv, valid)
        self.assertTrue(np.array_equal(serial, gathered))
        extra = (rng.random((5, mask_h, mask_w)) > 0.6).astype(np.uint8) * 255
        extra_serial = np.asarray(
            [counter.count(extra[i], depths[i]) for i in range(5)],
            dtype=np.int64,
        )
        extra_gathered = counter.gather_counts(extra, uv, valid)
        self.assertTrue(np.array_equal(extra_serial, extra_gathered))


if __name__ == "__main__":
    unittest.main()
