#!/usr/bin/env python3
"""Packed source RGB, optional composite mp4, and batched paste/median tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from composite_st_depth_overlay import _neighbor_dbg_median
from frame_volume import is_packed_stream, open_volume
from packed_source_rgb import ensure_packed_source_rgb
from paste_mode import paste_mask_mode_ab_numpy
from viewport_gt_depth_warp import PackedRgbReader, warp_frame_to_target


class PackedSourceRgbTest(unittest.TestCase):
    def test_ensure_is_idempotent_and_matches_cv2(self) -> None:
        if cv2 is None:
            self.skipTest("OpenCV unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "rgb.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (8, 6)
            )
            self.assertTrue(writer.isOpened())
            for index in range(4):
                bgr = np.zeros((6, 8, 3), dtype=np.uint8)
                bgr[..., 0] = 10 + index
                bgr[..., 1] = 20
                bgr[..., 2] = 30
                writer.write(bgr)
            writer.release()
            dest = root / "source_rgb"
            first = ensure_packed_source_rgb(video, dest)
            second = ensure_packed_source_rgb(video, dest)
            self.assertEqual(first, second)
            self.assertTrue(is_packed_stream(dest))
            cap = cv2.VideoCapture(str(video))
            ok, bgr0 = cap.read()
            cap.release()
            self.assertTrue(ok and bgr0 is not None)
            decoded = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
            volume = open_volume(dest, color_space="rgb")
            self.assertGreaterEqual(len(volume), 1)
            self.assertEqual(tuple(volume.frame_shape[:2]), decoded.shape[:2])
            np.testing.assert_array_equal(volume[0], decoded)
            meta = json.loads((dest / "volume_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta.get("decoder"), "opencv_bgr2rgb")
            pngs = list(dest.glob("*.png"))
            self.assertEqual(pngs, [])


class NeighborMedianBatchTest(unittest.TestCase):
    def test_batched_median_matches_per_frame(self) -> None:
        rng = np.random.default_rng(3)
        h, w = 12, 16
        d_bg = rng.uniform(0.4, 2.0, size=(3, h, w)).astype(np.float32)
        hole = rng.random((3, h, w)) > 0.65
        per_frame = np.stack(
            [_neighbor_dbg_median(d_bg[i], hole[i], 2) for i in range(3)]
        )
        batched = _neighbor_dbg_median(d_bg, hole, 2)
        np.testing.assert_allclose(batched, per_frame, equal_nan=True)

    def test_hole_only_median_matches_cpu_device(self) -> None:
        rng = np.random.default_rng(8)
        d_bg = rng.uniform(0.4, 2.0, size=(2, 10, 12)).astype(np.float32)
        hole = rng.random((2, 10, 12)) > 0.6
        cpu = _neighbor_dbg_median(d_bg, hole, 2, device="cpu")
        auto = _neighbor_dbg_median(d_bg, hole, 2)
        np.testing.assert_allclose(auto, cpu, equal_nan=True, rtol=1e-5, atol=1e-5)


class WarpPackedRgbTest(unittest.TestCase):
    def test_packed_reader_and_empty_frame_hold_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            from frame_volume import create_volume

            packed = Path(temp) / "source_rgb"
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
            rgb[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4)
            vol = create_volume(packed, 2, rgb.shape, np.uint8, color_space="rgb")
            vol[0] = rgb
            vol[1] = rgb + 1
            vol.flush()
            reader = PackedRgbReader(packed)
            np.testing.assert_array_equal(reader.get_rgb(0), rgb)
            k = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
            depth = np.zeros((4, 4), dtype=np.float32)
            out_rgb, out_depth, valid, _ = warp_frame_to_target(
                reader.get_rgb(0), depth, k, np.eye(4), k, (6, 8), "cpu", splat_radius_px=0
            )
            self.assertFalse(bool(np.any(valid)))
            self.assertEqual(out_rgb.shape, (6, 8, 3))
            self.assertEqual(out_depth.shape, (6, 8))
            reader.close()


class OptionalMp4ContractTest(unittest.TestCase):
    def test_composite_out_is_optional(self) -> None:
        composite = Path(__file__).with_name("composite_st_depth_overlay.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('default=None', composite)
        self.assertIn("Formal path writes packed --rgb-out-dir only", composite)
        eraser = Path(__file__).with_name("run_remaining_hole_eraser.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("out-mp4", eraser)

    def test_sam_writes_packed_masks_not_png_triples(self) -> None:
        sam = Path(__file__).with_name("sam3_efficientsam3_hand_masks.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"layout": "packed_memmap"', sam)
        self.assertIn("def _forward_prompts", sam)
        self.assertIn("forward_image(images_b)", sam)


class BatchedPasteMaskTest(unittest.TestCase):
    def test_numpy_3d_boolean_indexing(self) -> None:
        rng = np.random.default_rng(4)
        b, h, w = 2, 8, 10
        d_st = rng.uniform(0.3, 1.5, size=(b, h, w)).astype(np.float32)
        d_bg = rng.uniform(0.3, 1.5, size=(b, h, w)).astype(np.float32)
        robot = rng.random((b, h, w)) > 0.4
        hole = rng.random((b, h, w)) > 0.7
        neighbor = np.where(hole, d_bg, np.nan).astype(np.float32)
        batched = paste_mask_mode_ab_numpy(
            d_st, d_bg, robot, hole, paste_mode="B", neighbor_depth=neighbor, st_bg_depth=1e9
        )
        for index in range(b):
            per_frame = paste_mask_mode_ab_numpy(
                d_st[index],
                d_bg[index],
                robot[index],
                hole[index],
                paste_mode="B",
                neighbor_depth=neighbor[index],
                st_bg_depth=1e9,
            )
            np.testing.assert_array_equal(batched[index], per_frame)


if __name__ == "__main__":
    unittest.main()
