#!/usr/bin/env python3
"""Packed FrameVolume round-trip and splat/paste CPU vs GPU alignment."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from depth_splat import (
    splat_offsets,
    splat_zbuffer,
    splat_zbuffer_batch,
    splat_zbuffer_cpu,
    splat_zbuffer_cpu_batch,
    splat_zbuffer_torch,
)
from frame_volume import (
    as_mask,
    create_volume,
    hardlink_packed,
    is_packed_stream,
    load_mask_bool_volume,
    open_volume,
    packed_path,
)
from paste_mode import paste_mask_mode_ab, paste_mask_mode_ab_numpy, paste_mask_mode_ab_torch


class FrameVolumeRoundTripTest(unittest.TestCase):
    def test_packed_float_roundtrip_matches_legacy_npy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packed_dir = root / "packed"
            legacy_dir = root / "legacy"
            legacy_dir.mkdir()
            count, h, w = 3, 8, 10
            vol = create_volume(packed_dir, count, (h, w), np.float32, fill=0)
            for i in range(count):
                frame = np.full((h, w), float(i) + 0.25, dtype=np.float32)
                vol[i] = frame
                np.save(legacy_dir / f"{i:06d}.npy", frame, allow_pickle=False)
            vol.flush()
            reopened = open_volume(packed_dir)
            legacy = open_volume(legacy_dir)
            self.assertTrue(is_packed_stream(packed_dir))
            self.assertFalse(is_packed_stream(legacy_dir))
            self.assertEqual(len(reopened), count)
            self.assertEqual(len(legacy), count)
            for i in range(count):
                np.testing.assert_array_equal(reopened[i], legacy[i])
            np.testing.assert_array_equal(reopened.array, np.stack([legacy[i] for i in range(count)]))

    def test_packed_rgb_matches_png_roundtrip(self) -> None:
        if cv2 is None:
            self.skipTest("OpenCV unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packed_dir = root / "packed"
            legacy_dir = root / "legacy"
            legacy_dir.mkdir()
            rgb = np.zeros((4, 6, 3), dtype=np.uint8)
            rgb[..., 0] = 10
            rgb[..., 1] = 20
            rgb[..., 2] = 30
            vol = create_volume(packed_dir, 1, rgb.shape, np.uint8, color_space="rgb")
            vol[0] = rgb
            vol.flush()
            self.assertTrue(cv2.imwrite(str(legacy_dir / "000000.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
            packed = open_volume(packed_dir, color_space="rgb")
            legacy = open_volume(legacy_dir, color_space="rgb")
            np.testing.assert_array_equal(packed[0], rgb)
            np.testing.assert_array_equal(legacy[0], rgb)

    def test_mask_uint8_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "mask"
            vol = create_volume(directory, 1, (3, 3), np.uint8, fill=0)
            frame = np.array([[0, 127, 255], [1, 128, 200], [0, 0, 255]], dtype=np.uint8)
            vol[0] = frame
            vol.flush()
            loaded = open_volume(directory)[0]
            np.testing.assert_array_equal(as_mask(loaded), frame > 127)

    def test_as_mask_keeps_packed_thw_and_drops_color_channel(self) -> None:
        packed = np.zeros((4, 5, 6), dtype=np.uint8)
        packed[1, 2, 3] = 255
        masked = as_mask(packed)
        self.assertEqual(masked.shape, (4, 5, 6))
        self.assertTrue(masked[1, 2, 3])
        rgb = np.zeros((5, 6, 3), dtype=np.uint8)
        rgb[1, 2] = 255
        self.assertEqual(as_mask(rgb).shape, (5, 6))
        self.assertTrue(as_mask(rgb)[1, 2])

    def test_as_mask_keeps_packed_thw_and_drops_color_channel(self) -> None:
        packed = np.zeros((4, 5, 6), dtype=np.uint8)
        packed[1, 2, 3] = 255
        masked = as_mask(packed)
        self.assertEqual(masked.shape, (4, 5, 6))
        self.assertTrue(masked[1, 2, 3])
        rgb = np.zeros((5, 6, 3), dtype=np.uint8)
        rgb[1, 2] = 255
        self.assertEqual(as_mask(rgb).shape, (5, 6))
        self.assertTrue(as_mask(rgb)[1, 2])

    def test_hardlink_packed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "src"
            dest = root / "dst"
            vol = create_volume(source, 2, (2, 2), np.uint8, fill=7)
            vol.flush()
            mode = hardlink_packed(source, dest)
            self.assertIn(mode, ("hardlink", "copy"))
            np.testing.assert_array_equal(open_volume(dest).array, open_volume(source).array)
            self.assertTrue(packed_path(dest).is_file())

    def test_load_mask_bool_volume_packed_without_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "human_mask"
            count, h, w = 4, 6, 8
            vol = create_volume(directory, count, (h, w), np.uint8, fill=0)
            frame = np.zeros((h, w), dtype=np.uint8)
            frame[1:3, 2:5] = 255
            vol[2] = frame
            vol.flush()
            self.assertFalse(any(directory.glob("*.png")))
            stacked = load_mask_bool_volume(directory, count=count, shape=(h, w))
            self.assertEqual(stacked.shape, (count, h, w))
            self.assertFalse(np.any(stacked[0]))
            np.testing.assert_array_equal(stacked[2], frame > 127)
            self.assertGreater(float(stacked.mean()), 0.0)

    def test_load_mask_bool_volume_legacy_png(self) -> None:
        if cv2 is None:
            self.skipTest("OpenCV unavailable")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "masks"
            directory.mkdir()
            frame = np.zeros((3, 4), dtype=np.uint8)
            frame[1, 2] = 255
            self.assertTrue(cv2.imwrite(str(directory / "000001.png"), frame))
            stacked = load_mask_bool_volume(directory, count=3, shape=(3, 4))
            self.assertFalse(np.any(stacked[0]))
            np.testing.assert_array_equal(stacked[1], frame > 0)
            self.assertFalse(np.any(stacked[2]))


class SplatAlignmentTest(unittest.TestCase):
    def _sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list]:
        rng = np.random.default_rng(0)
        h, w = 32, 40
        n = 80
        uv = np.stack(
            [rng.integers(0, w, size=n), rng.integers(0, h, size=n)],
            axis=1,
        ).astype(np.int64)
        z = rng.uniform(0.2, 3.0, size=n).astype(np.float32)
        conf = rng.uniform(0.2, 1.0, size=n).astype(np.float32)
        return uv, z, conf, h, w, splat_offsets(2)

    def test_cpu_splat_is_deterministic(self) -> None:
        uv, z, conf, h, w, offsets = self._sample()
        a = splat_zbuffer_cpu(uv, z, conf, h, w, offsets)
        b = splat_zbuffer_cpu(uv[::-1], z[::-1], conf[::-1], h, w, offsets)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)

    def test_auto_splat_matches_cpu(self) -> None:
        uv, z, conf, h, w, offsets = self._sample()
        cpu = splat_zbuffer_cpu(uv, z, conf, h, w, offsets)
        auto = splat_zbuffer(uv, z, conf, h, w, offsets, device="cpu")
        for left, right in zip(cpu, auto):
            np.testing.assert_array_equal(left, right)

    def test_cuda_splat_matches_cpu_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        uv, z, conf, h, w, offsets = self._sample()
        cpu = splat_zbuffer_cpu(uv, z, conf, h, w, offsets)
        gpu = splat_zbuffer_torch(uv, z, conf, h, w, offsets, device="cuda")
        np.testing.assert_array_equal(cpu[0], gpu[0])
        np.testing.assert_array_equal(cpu[1], gpu[1])
        np.testing.assert_array_equal(cpu[2], gpu[2])

    def test_batched_splat_matches_per_frame(self) -> None:
        rng = np.random.default_rng(4)
        h, w, n_frames = 24, 28, 5
        offsets = splat_offsets(2)
        uv, z, conf, frame_id = [], [], [], []
        per_frame = []
        for i in range(n_frames):
            n = 40
            frame_uv = np.stack(
                [rng.integers(0, w, size=n), rng.integers(0, h, size=n)],
                axis=1,
            ).astype(np.int64)
            frame_z = rng.uniform(0.2, 3.0, size=n).astype(np.float32)
            frame_c = rng.uniform(0.2, 1.0, size=n).astype(np.float32)
            per_frame.append(splat_zbuffer_cpu(frame_uv, frame_z, frame_c, h, w, offsets))
            uv.append(frame_uv)
            z.append(frame_z)
            conf.append(frame_c)
            frame_id.append(np.full(n, i, dtype=np.int64))
        batched = splat_zbuffer_cpu_batch(
            np.concatenate(uv),
            np.concatenate(z),
            np.concatenate(conf),
            np.concatenate(frame_id),
            n_frames,
            h,
            w,
            offsets,
        )
        for i, (depth, conf_i, edge) in enumerate(per_frame):
            np.testing.assert_array_equal(batched[0][i], depth)
            np.testing.assert_array_equal(batched[1][i], conf_i)
            np.testing.assert_array_equal(batched[2][i], edge)
        auto = splat_zbuffer_batch(
            np.concatenate(uv),
            np.concatenate(z),
            np.concatenate(conf),
            np.concatenate(frame_id),
            n_frames,
            h,
            w,
            offsets,
            device="cpu",
        )
        for left, right in zip(batched, auto):
            np.testing.assert_array_equal(left, right)

    def test_cuda_batched_splat_matches_cpu_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(5)
        h, w, n_frames = 20, 22, 4
        n = 60
        uv = np.stack(
            [rng.integers(0, w, size=n * n_frames), rng.integers(0, h, size=n * n_frames)],
            axis=1,
        ).astype(np.int64)
        z = rng.uniform(0.2, 3.0, size=n * n_frames).astype(np.float32)
        conf = rng.uniform(0.2, 1.0, size=n * n_frames).astype(np.float32)
        frame_id = np.repeat(np.arange(n_frames, dtype=np.int64), n)
        offsets = splat_offsets(1)
        cpu = splat_zbuffer_cpu_batch(uv, z, conf, frame_id, n_frames, h, w, offsets)
        gpu = splat_zbuffer_batch(uv, z, conf, frame_id, n_frames, h, w, offsets, device="cuda")
        np.testing.assert_array_equal(cpu[0], gpu[0])
        np.testing.assert_array_equal(cpu[1], gpu[1])
        np.testing.assert_array_equal(cpu[2], gpu[2])


class PasteAlignmentTest(unittest.TestCase):
    def _sample(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(1)
        h, w = 24, 32
        d_bg = rng.uniform(0.4, 2.0, size=(h, w)).astype(np.float32)
        d_st = d_bg + rng.uniform(-0.3, 0.3, size=(h, w)).astype(np.float32)
        robot = rng.random((h, w)) > 0.4
        hole = rng.random((h, w)) > 0.7
        neighbor = np.where(hole, d_bg + rng.uniform(-0.1, 0.1, size=(h, w)), np.nan).astype(np.float32)
        d_bg[rng.random((h, w)) > 0.9] = np.nan
        return {
            "d_st_raw": d_st,
            "d_bg": d_bg,
            "robot": robot,
            "hole": hole,
            "neighbor_depth": neighbor,
        }

    def test_mode_a_and_b_cpu(self) -> None:
        sample = self._sample()
        mode_a = paste_mask_mode_ab_numpy(
            sample["d_st_raw"],
            sample["d_bg"],
            sample["robot"],
            sample["hole"],
            paste_mode="A",
            neighbor_depth=None,
            st_bg_depth=1e9,
        )
        mode_b = paste_mask_mode_ab_numpy(st_bg_depth=1e9, paste_mode="B", **sample)
        self.assertEqual(mode_a.dtype, bool)
        self.assertEqual(mode_b.shape, sample["d_st_raw"].shape)
        in_hole_robot = sample["hole"] & sample["robot"]
        self.assertTrue(np.all(mode_a[in_hole_robot]))

    def test_cuda_paste_matches_cpu_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        sample = self._sample()
        cpu = paste_mask_mode_ab_numpy(st_bg_depth=1e9, paste_mode="B", **sample)
        gpu = paste_mask_mode_ab_torch(
            st_bg_depth=1e9, paste_mode="B", device="cuda", **sample
        )
        np.testing.assert_array_equal(cpu, gpu)
        auto = paste_mask_mode_ab(st_bg_depth=1e9, paste_mode="B", device="cuda", **sample)
        np.testing.assert_array_equal(cpu, auto)

    def test_batched_mode_b_matches_per_frame(self) -> None:
        sample = self._sample()
        per_frame = paste_mask_mode_ab_numpy(st_bg_depth=1e9, paste_mode="B", **sample)
        batched = paste_mask_mode_ab_numpy(
            sample["d_st_raw"][None],
            sample["d_bg"][None],
            sample["robot"][None],
            sample["hole"][None],
            paste_mode="B",
            neighbor_depth=sample["neighbor_depth"][None],
            st_bg_depth=1e9,
        )
        np.testing.assert_array_equal(batched[0], per_frame)


if __name__ == "__main__":
    unittest.main()
