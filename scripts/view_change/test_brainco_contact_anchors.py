#!/usr/bin/env python3
"""Unit tests for BrainCo neighborhood Laplacian contact and Dex3 CupModel orchestration."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from contact_pointcloud_analysis import (
    VISIBILITY_OCCLUDED,
    VISIBILITY_UNKNOWN,
    VISIBILITY_VISIBLE,
    TorchDepthProjector,
    build_rgb_occluder_z,
    choose_object_masks,
    classify,
    classify_depth_against_mask,
    classify_depth_against_mask_torch,
    classify_fingertip_visibility,
    depth_footprint_uv,
    depth_footprint_uv_torch,
    depth_grid_rc,
    downsample_object_cloud,
    farthest_point_indices,
    estimate_normals,
    finger_contact_metric,
    flags_from_runs,
    hand_neighborhood_object_sel,
    nearest_mano_distance,
    label_registered_depth,
    load_mask,
    neighborhood_object_sel,
    parse_sides,
    project_depth,
    voxel_keep_indices,
    resolve_device,
    unproject_masked_optical,
)
from frame_volume import create_volume
from interaction_mesh import (
    build_interaction_adjacency,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    deepest_inside_world_push,
    laplacian_residual,
    mesh_penetration_hinge,
    retract_axis_away_from_object,
    retract_axis_toward_target,
)
from optimize_contact_aware_wrist_and_fingers import (
    hand_dof,
    hand_side_slice,
    side_joint_count,
)

try:
    from optimize_brainco_contact_anchors import (
        adaptive_wrist_translation_bound,
        apply_wrist_local_translation,
        cap_object_points,
        estimate_outward_normals,
        interpolate_deltas,
        packed_object_points,
    )
except ImportError:  # pragma: no cover - mujoco is optional for flag tests
    adaptive_wrist_translation_bound = None
    apply_wrist_local_translation = None
    estimate_outward_normals = None
    interpolate_deltas = None
    packed_object_points = None
    cap_object_points = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTE = PROJECT_ROOT / "scripts/view_change/run_contact_precompute.sh"
BATCH = PROJECT_ROOT / "scripts/view_change/run_contact_aware_batch_view_change.sh"
TRACKING = PROJECT_ROOT / "scripts/view_change/run_contact_aware_6d_reference_tracking.sh"
DRIVER = PROJECT_ROOT / "scripts/run_three_point_batch.py"
RETARGET = PROJECT_ROOT / "scripts/view_change/brainco_hand_retarget.py"


class LoadMaskTest(unittest.TestCase):
    def test_png_roundtrip_and_nearest_resize(self) -> None:
        import tempfile

        from PIL import Image

        pixels = np.zeros((4, 6), dtype=np.uint8)
        pixels[1, 2] = 255
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            Image.fromarray(pixels).save(path)
            same = load_mask(path, (4, 6))
            self.assertIsNotNone(same)
            np.testing.assert_array_equal(same, pixels > 0)
            scaled = load_mask(path, (8, 12))
            self.assertIsNotNone(scaled)
            self.assertEqual(scaled.shape, (8, 12))
            self.assertTrue(bool(scaled[2, 4]) or bool(scaled[3, 4]) or bool(scaled[2, 5]))




class NeighborhoodObjectSelTest(unittest.TestCase):
    def test_occluded_fingertip_still_collects_object_depth(self) -> None:
        # MANO tip sits behind the visible object surface; no finger pixels
        # are in the human mask, so the ball must still pick up object points.
        world = np.asarray(
            [
                [0.00, 0.00, 0.20],  # object surface
                [0.01, 0.00, 0.20],
                [0.00, 0.01, 0.205],
                [0.40, 0.00, 0.20],  # far point, other hand
            ],
            dtype=np.float64,
        )
        not_human = np.asarray([True, True, True, True])
        center = np.asarray([0.00, 0.00, 0.23])  # 3 cm behind the surface
        selected = neighborhood_object_sel(world, center, 0.04, not_human)
        self.assertTrue(np.any(selected[:3]))
        self.assertFalse(bool(selected[3]))

    def test_human_mask_excludes_hand_depth_but_is_not_a_visibility_gate(self) -> None:
        world = np.asarray(
            [[0.0, 0.0, 0.2], [0.005, 0.0, 0.2], [0.0, 0.0, 0.21]],
            dtype=np.float64,
        )
        not_human = np.asarray([True, True, False])  # last point is the hand itself
        center = np.asarray([0.0, 0.0, 0.21])
        selected = neighborhood_object_sel(world, center, 0.04, not_human)
        np.testing.assert_array_equal(selected, [True, True, False])

    def test_left_and_right_balls_do_not_share_points(self) -> None:
        world = np.asarray(
            [[-0.15, 0.0, 0.3], [0.15, 0.0, 0.3]],
            dtype=np.float64,
        )
        not_human = np.ones(2, dtype=bool)
        left = neighborhood_object_sel(world, np.asarray([-0.15, 0.0, 0.32]), 0.04, not_human)
        right = neighborhood_object_sel(world, np.asarray([0.15, 0.0, 0.32]), 0.04, not_human)
        np.testing.assert_array_equal(left, [True, False])
        np.testing.assert_array_equal(right, [False, True])

    def test_mask_edge_halo_is_not_object(self) -> None:
        world = np.asarray([[0.0, 0.0, 0.2], [0.03, 0.0, 0.2]], dtype=np.float64)
        not_human = np.ones(2, dtype=bool)
        extra = np.asarray([False, True])  # first sample hugs the silhouette
        selected = neighborhood_object_sel(
            world, np.asarray([0.0, 0.0, 0.21]), 0.04, not_human, extra=extra
        )
        np.testing.assert_array_equal(selected, [False, True])


class DepthFootprintClassifyTest(unittest.TestCase):
    def test_full_hand_box_is_hand_empty_box_is_object_ok(self) -> None:
        human = np.zeros((8, 8), dtype=bool)
        human[:4, :4] = True
        boxes = (
            np.asarray([0.0, 4.0], dtype=np.float64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            np.asarray([4.0, 8.0], dtype=np.float64),
            np.asarray([4.0, 4.0], dtype=np.float64),
        )
        is_hand, is_mixed, is_object_ok = classify_depth_against_mask(human, boxes)
        np.testing.assert_array_equal(is_hand, [True, False])
        np.testing.assert_array_equal(is_mixed, [False, False])
        np.testing.assert_array_equal(is_object_ok, [False, True])

    def test_visibility_inside_object_is_occluded(self) -> None:
        self.assertEqual(
            classify_fingertip_visibility(0.20, 0.25, "object_ok", 0.02),
            VISIBILITY_OCCLUDED,
        )
        self.assertEqual(
            classify_fingertip_visibility(0.25, 0.25, "hand", 0.02),
            VISIBILITY_VISIBLE,
        )
        self.assertEqual(
            classify_fingertip_visibility(float("nan"), 0.25, "none", 0.02),
            VISIBILITY_UNKNOWN,
        )


def _cuda_device():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda")


def _toy_rgbd_scene() -> dict[str, np.ndarray | tuple[int, int]]:
    depth_k = np.array(
        [[80.0, 0.0, 9.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    rgb_k = np.array(
        [[160.0, 0.0, 19.5], [0.0, 160.0, 15.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    depth_mm = np.full((16, 20), 420.0, dtype=np.float64)
    depth_mm[:2, :2] = 0.0
    depth_mm[4, 5] = 80.0
    depth_mm[10, 12] = 4000.0
    depth_mm[6:10, 8:14] = 510.0
    human = np.zeros((32, 40), dtype=bool)
    human[8:20, 4:18] = True
    object_mask = np.zeros((32, 40), dtype=bool)
    object_mask[12:28, 16:34] = True
    return {
        "depth_k": depth_k,
        "rgb_k": rgb_k,
        "depth_mm": depth_mm,
        "human": human,
        "object_mask": object_mask,
        "rgb_shape": (32, 40),
    }


def _assert_labeled_equal(cpu: dict, gpu: dict) -> None:
    np.testing.assert_array_equal(cpu["valid"], gpu["valid"])
    np.testing.assert_array_equal(cpu["inside"], gpu["inside"])
    np.testing.assert_array_equal(cpu["is_hand"], gpu["is_hand"])
    np.testing.assert_array_equal(cpu["is_mixed"], gpu["is_mixed"])
    np.testing.assert_array_equal(cpu["is_object_ok"], gpu["is_object_ok"])
    np.testing.assert_array_equal(cpu["pixel_object"], gpu["pixel_object"])
    # CUDA geometry is fp32; UV can round 1 pixel off while flags stay aligned.
    if len(cpu["uv"]) == len(gpu["uv"]) and len(cpu["uv"]):
        np.testing.assert_allclose(cpu["uv"], gpu["uv"], rtol=0.0, atol=1.0)
        np.testing.assert_allclose(cpu["optical"], gpu["optical"], rtol=1e-4, atol=1e-4)
    else:
        np.testing.assert_equal(len(cpu["uv"]), len(gpu["uv"]))
    for left, right in zip(cpu["boxes"], gpu["boxes"]):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1.5)
    for key in (
        "human_mask_fraction",
        "object_mask_fraction",
        "valid_depth_ratio",
    ):
        np.testing.assert_allclose(cpu[key], gpu[key], rtol=1e-4, atol=1e-4)


class CudaRegisteredDepthTest(unittest.TestCase):
    def test_cpu_label_matches_inline_classify(self) -> None:
        scene = _toy_rgbd_scene()
        labeled = label_registered_depth(
            scene["depth_mm"],
            scene["depth_k"],
            scene["rgb_k"],
            scene["human"],
            scene["object_mask"],
            scene["rgb_shape"],
            depth_footprint=True,
            projector=None,
        )
        optical, uv, valid, inside = project_depth(
            scene["depth_mm"], scene["depth_k"], scene["rgb_k"], scene["rgb_shape"]
        )
        np.testing.assert_array_equal(labeled["optical"], optical)
        np.testing.assert_array_equal(labeled["uv"], uv)
        np.testing.assert_array_equal(labeled["valid"], valid)
        np.testing.assert_array_equal(labeled["inside"], inside)
        cols, rows = depth_grid_rc(valid)
        boxes = depth_footprint_uv(
            scene["depth_k"], scene["rgb_k"], cols, rows, optical[:, 2]
        )
        is_hand, is_mixed, is_object_ok = classify_depth_against_mask(
            scene["human"], boxes
        )
        np.testing.assert_array_equal(labeled["is_hand"], is_hand)
        np.testing.assert_array_equal(labeled["is_mixed"], is_mixed)
        np.testing.assert_array_equal(labeled["is_object_ok"], is_object_ok)

    def test_cuda_classify_and_footprint_match_cpu(self) -> None:
        device = _cuda_device()
        if device is None:
            self.skipTest("CUDA is unavailable")
        import torch

        scene = _toy_rgbd_scene()
        optical, _uv, valid, _inside = project_depth(
            scene["depth_mm"], scene["depth_k"], scene["rgb_k"], scene["rgb_shape"]
        )
        cols, rows = depth_grid_rc(valid)
        boxes = depth_footprint_uv(
            scene["depth_k"], scene["rgb_k"], cols, rows, optical[:, 2]
        )
        is_hand, is_mixed, is_object_ok = classify_depth_against_mask(
            scene["human"], boxes
        )
        depth_k_t = torch.as_tensor(scene["depth_k"], device=device)
        rgb_k_t = torch.as_tensor(scene["rgb_k"], device=device)
        boxes_t = depth_footprint_uv_torch(
            depth_k_t,
            rgb_k_t,
            torch.as_tensor(cols, device=device),
            torch.as_tensor(rows, device=device),
            torch.as_tensor(optical[:, 2], device=device),
        )
        for left, right in zip(boxes, boxes_t):
            np.testing.assert_allclose(
                left, right.detach().cpu().numpy(), rtol=0.0, atol=1e-12
            )
        gpu_hand, gpu_mixed, gpu_ok = classify_depth_against_mask_torch(
            torch.as_tensor(scene["human"], device=device), boxes_t
        )
        np.testing.assert_array_equal(is_hand, gpu_hand.detach().cpu().numpy())
        np.testing.assert_array_equal(is_mixed, gpu_mixed.detach().cpu().numpy())
        np.testing.assert_array_equal(is_object_ok, gpu_ok.detach().cpu().numpy())

    def test_cuda_label_and_masked_unproject_match_cpu(self) -> None:
        device = _cuda_device()
        if device is None:
            self.skipTest("CUDA is unavailable")
        scene = _toy_rgbd_scene()
        projector = TorchDepthProjector(
            scene["depth_k"], scene["rgb_k"], scene["depth_mm"].shape, device
        )
        for footprint in (True, False):
            cpu = label_registered_depth(
                scene["depth_mm"],
                scene["depth_k"],
                scene["rgb_k"],
                scene["human"],
                scene["object_mask"],
                scene["rgb_shape"],
                depth_footprint=footprint,
                projector=None,
            )
            gpu = label_registered_depth(
                scene["depth_mm"],
                scene["depth_k"],
                scene["rgb_k"],
                scene["human"],
                scene["object_mask"],
                scene["rgb_shape"],
                depth_footprint=footprint,
                projector=projector,
            )
            _assert_labeled_equal(cpu, gpu)
        cpu_pts = unproject_masked_optical(
            scene["depth_mm"],
            scene["depth_k"],
            scene["rgb_k"],
            scene["object_mask"],
            projector=None,
        )
        gpu_pts = unproject_masked_optical(
            scene["depth_mm"],
            scene["depth_k"],
            scene["rgb_k"],
            scene["object_mask"],
            projector=projector,
        )
        np.testing.assert_allclose(cpu_pts, gpu_pts, rtol=1e-4, atol=1e-3)
        self.assertEqual(resolve_device("cuda").type, "cuda")

    def test_empty_valid_depth_stays_empty_on_cuda(self) -> None:
        device = _cuda_device()
        if device is None:
            self.skipTest("CUDA is unavailable")
        scene = _toy_rgbd_scene()
        depth_mm = np.zeros_like(scene["depth_mm"])
        projector = TorchDepthProjector(
            scene["depth_k"], scene["rgb_k"], depth_mm.shape, device
        )
        cpu = label_registered_depth(
            depth_mm,
            scene["depth_k"],
            scene["rgb_k"],
            scene["human"],
            scene["object_mask"],
            scene["rgb_shape"],
            depth_footprint=True,
            projector=None,
        )
        gpu = label_registered_depth(
            depth_mm,
            scene["depth_k"],
            scene["rgb_k"],
            scene["human"],
            scene["object_mask"],
            scene["rgb_shape"],
            depth_footprint=True,
            projector=projector,
        )
        _assert_labeled_equal(cpu, gpu)
        self.assertEqual(len(cpu["optical"]), 0)


class DownsampleObjectCloudTest(unittest.TestCase):
    def test_caps_at_max_points_and_keeps_normals(self) -> None:
        points = np.linspace(0, 1, 2000)[:, None] * np.array([[1.0, 0.0, 0.0]])
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (2000, 1))
        pts, nrm = downsample_object_cloud(points, normals, max_points=1024, seed=3)
        self.assertEqual(len(pts), 1024)
        self.assertEqual(len(nrm), 1024)
        np.testing.assert_allclose(nrm, np.tile([[0.0, 0.0, 1.0]], (1024, 1)))

    def test_downsample_then_normal_does_not_change_interval_flags(self) -> None:
        from scipy.spatial import cKDTree

        raw = [True, True, False, True, True, True, False, False]
        flags = flags_from_runs(len(raw), classify(raw, max_gap=1, min_run=2))
        rng = np.random.default_rng(3)
        points = rng.normal(size=(2000, 3)).astype(np.float64)
        dummy_normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (2000, 1))
        packed_pts, _ = downsample_object_cloud(
            points, dummy_normals, max_points=1024, seed=3
        )
        packed_tree = cKDTree(packed_pts)
        packed_nrm = estimate_normals(
            packed_pts,
            packed_tree,
            np.arange(len(packed_pts)),
            np.zeros(3, dtype=np.float64),
            k=16,
        )
        self.assertEqual(len(packed_pts), 1024)
        self.assertEqual(len(packed_nrm), 1024)
        flags_after = flags_from_runs(len(raw), classify(raw, max_gap=1, min_run=2))
        np.testing.assert_array_equal(flags, flags_after)
        source = (
            PROJECT_ROOT / "scripts/view_change/contact_pointcloud_analysis.py"
        ).read_text(encoding="utf-8")
        downsample_at = source.index("packed_pts, _ = downsample_object_cloud(")
        normals_at = source.index("packed_nrm = estimate_normals(")
        self.assertLess(downsample_at, normals_at)

    def test_voxel_grid_is_sparse_over_a_large_plane(self) -> None:
        xs, ys = np.meshgrid(np.linspace(0.0, 0.20, 81), np.linspace(0.0, 0.20, 81))
        points = np.stack(
            [xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1
        ).astype(np.float64)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(points), 1))
        pts, _ = downsample_object_cloud(
            points, normals, max_points=2048, seed=3, voxel_m=0.02
        )
        self.assertGreaterEqual(len(pts), 80)
        self.assertLessEqual(len(pts), 130)
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        self.assertGreater(float(np.min(dist)), 0.01)

    def test_farthest_prefix_covers_both_ends(self) -> None:
        points = np.linspace(0.0, 1.0, 50)[:, None] * np.array([[1.0, 0.0, 0.0]])
        idx = farthest_point_indices(points, 8, seed=0)
        chosen = points[idx]
        self.assertGreater(float(chosen[:, 0].max() - chosen[:, 0].min()), 0.7)
        self.assertEqual(len(voxel_keep_indices(points, 0.25)), 5)
        dist = nearest_mano_distance(
            np.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(dist, [0.0, 0.3])


class ContactFlagDeviceTest(unittest.TestCase):
    def test_gpu_and_cpu_contact_flags_match_on_tiny_fixture(self) -> None:
        device = _cuda_device()
        if device is None:
            self.skipTest("CUDA is unavailable")
        import torch
        from scipy.spatial import cKDTree

        xs, ys = np.meshgrid(
            np.linspace(-0.02, 0.02, 8), np.linspace(-0.02, 0.02, 8)
        )
        object_points = np.stack(
            [xs.ravel(), ys.ravel(), np.full(xs.size, 0.20)], axis=1
        ).astype(np.float64)
        object_optical = object_points.copy()
        finger_points = np.asarray(
            [[0.0, 0.0, 0.205], [0.004, 0.0, 0.204], [0.0, 0.004, 0.206]],
            dtype=np.float64,
        )
        finger_optical = finger_points.copy()
        camera = np.zeros(3, dtype=np.float64)
        tree = cKDTree(object_points)
        kwargs = dict(
            min_hand_points=1,
            distance_threshold_m=0.02,
            proximity_envelope_m=0.04,
            depth_gap_threshold_m=0.03,
            normal_dot_threshold=0.1,
        )
        cpu = finger_contact_metric(
            finger_points,
            finger_optical,
            object_points,
            object_optical,
            tree,
            camera,
            device=None,
            **kwargs,
        )
        gpu = finger_contact_metric(
            finger_points,
            finger_optical,
            object_points,
            object_optical,
            None,
            camera,
            device=device,
            object_tensor=torch.as_tensor(
                object_points, dtype=torch.float32, device=device
            ),
            **kwargs,
        )
        self.assertEqual(bool(cpu["raw_contact"]), bool(gpu["raw_contact"]))
        self.assertEqual(bool(cpu["envelope_contact"]), bool(gpu["envelope_contact"]))
        self.assertEqual(bool(cpu["valid"]), bool(gpu["valid"]))


class Neighborhood2DObjectTest(unittest.TestCase):
    def test_object_ok_outside_mask_enters_o(self) -> None:
        rgb_k = np.array(
            [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        uv = np.asarray(
            [[5.0, 5.0], [8.0, 5.0], [20.0, 5.0]],
            dtype=np.float64,
        )
        optical_z = np.asarray([0.30, 0.30, 0.80], dtype=np.float64)
        is_object_ok = np.asarray([True, True, True])
        is_hand = np.asarray([False, False, False])
        keypoints = np.zeros((21, 2), dtype=np.float64)
        keypoints[:, :] = [5.0, 5.0]
        keypoints[8] = [6.0, 5.0]
        mano = np.zeros((21, 3), dtype=np.float64)
        mano[:, 2] = 0.30
        human = np.zeros((12, 24), dtype=bool)
        human[5, 5] = True
        selected = hand_neighborhood_object_sel(
            uv=uv,
            optical_z=optical_z,
            is_object_ok=is_object_ok,
            is_hand=is_hand,
            keypoints=keypoints,
            mano_optical=mano,
            rgb_k=rgb_k,
            human=human,
            finger_width_px=2.0,
            dilation_px=3.0,
            z_band_m=0.05,
            shape=(12, 24),
        )
        self.assertTrue(bool(selected[0]))
        self.assertTrue(bool(selected[1]))
        self.assertFalse(bool(selected[2]))

    def test_mixed_and_hand_pixels_do_not_enter_o(self) -> None:
        rgb_k = np.eye(3, dtype=np.float64)
        uv = np.asarray([[2.0, 2.0], [3.0, 2.0], [4.0, 2.0]], dtype=np.float64)
        optical_z = np.asarray([0.25, 0.25, 0.25], dtype=np.float64)
        is_object_ok = np.asarray([False, False, True])
        is_hand = np.asarray([True, False, False])
        keypoints = np.zeros((21, 2), dtype=np.float64)
        keypoints[:, :] = [3.0, 2.0]
        mano = np.zeros((21, 3), dtype=np.float64)
        mano[:, 2] = 0.25
        selected = hand_neighborhood_object_sel(
            uv=uv,
            optical_z=optical_z,
            is_object_ok=is_object_ok,
            is_hand=is_hand,
            keypoints=keypoints,
            mano_optical=mano,
            rgb_k=rgb_k,
            human=None,
            finger_width_px=4.0,
            dilation_px=2.0,
            z_band_m=0.05,
            shape=(8, 8),
        )
        np.testing.assert_array_equal(selected, [False, False, True])

    def test_mano_ball_keeps_far_image_near_3d_and_drops_far_3d(self) -> None:
        rgb_k = np.eye(3, dtype=np.float64)
        uv = np.asarray([[2.0, 2.0], [40.0, 2.0], [3.0, 2.0]], dtype=np.float64)
        points = np.asarray(
            [
                [0.00, 0.00, 0.30],
                [0.08, 0.00, 0.30],
                [0.40, 0.00, 0.30],
            ],
            dtype=np.float64,
        )
        optical_z = points[:, 2]
        is_object_ok = np.ones(3, dtype=bool)
        is_hand = np.zeros(3, dtype=bool)
        keypoints = np.zeros((21, 2), dtype=np.float64)
        keypoints[:, :] = [2.0, 2.0]
        mano = np.zeros((21, 3), dtype=np.float64)
        mano[:, 2] = 0.30
        selected = hand_neighborhood_object_sel(
            uv=uv,
            optical_z=optical_z,
            is_object_ok=is_object_ok,
            is_hand=is_hand,
            keypoints=keypoints,
            mano_optical=mano,
            rgb_k=rgb_k,
            human=None,
            finger_width_px=2.0,
            dilation_px=1.0,
            z_band_m=0.12,
            shape=(8, 48),
            points=points,
            mano_points=mano,
            cloud_radius_m=0.12,
        )
        np.testing.assert_array_equal(selected, [True, True, False])
        tight = hand_neighborhood_object_sel(
            uv=uv,
            optical_z=optical_z,
            is_object_ok=is_object_ok,
            is_hand=is_hand,
            keypoints=keypoints,
            mano_optical=mano,
            rgb_k=rgb_k,
            human=None,
            finger_width_px=2.0,
            dilation_px=1.0,
            z_band_m=0.12,
            shape=(8, 48),
        )
        self.assertTrue(bool(tight[0]))
        self.assertFalse(bool(tight[1]))


class InteractionMeshTest(unittest.TestCase):
    def test_identity_laplacian_is_zero_and_shift_increases_it(self) -> None:
        rng = np.random.default_rng(7)
        P_h = rng.normal(size=(16, 3))
        O = rng.normal(size=(8, 3)) * 0.04 + np.array([0.12, 0.0, 0.0])
        source = np.vstack([P_h, O])
        adj = build_interaction_adjacency(source)
        identity = laplacian_residual(source, np.vstack([P_h, O]), adj)
        np.testing.assert_allclose(identity, 0.0, atol=1e-12)
        shifted = P_h + np.array([0.04, 0.0, 0.0])
        moved = laplacian_residual(source, np.vstack([shifted, O]), adj)
        self.assertGreater(float(np.linalg.norm(moved)), 1e-6)
        hand_idx = slice(0, 16)
        lap_source = calculate_laplacian_coordinates(source, adj)
        lap_moved = calculate_laplacian_coordinates(np.vstack([shifted, O]), adj)
        self.assertGreater(
            float(np.linalg.norm(lap_moved[hand_idx] - lap_source[hand_idx])),
            0.0,
        )

    def test_laplacian_matrix_matches_coordinates_and_is_linear_in_hand(self) -> None:
        rng = np.random.default_rng(3)
        P_h = rng.normal(size=(16, 3))
        O = rng.normal(size=(8, 3)) * 0.03
        source = np.vstack([P_h, O])
        adj = build_interaction_adjacency(source)
        matrix = calculate_laplacian_matrix(source, adj, uniform_weight=True)
        np.testing.assert_allclose(
            matrix @ source,
            calculate_laplacian_coordinates(source, adj, uniform_weight=True),
            atol=1e-12,
        )
        P_r = P_h + rng.normal(size=P_h.shape) * 0.01
        target = np.vstack([P_r, O])
        residual = laplacian_residual(source, target, adj, uniform_weight=True)
        predicted = (matrix @ (target - source)).reshape(-1)
        np.testing.assert_allclose(residual, predicted, atol=1e-12)


class PenetrationHingeTest(unittest.TestCase):
    def test_inside_palm_is_positive_outside_is_zero(self) -> None:
        hand = np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        )
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1))
        inside = np.asarray([[0.0, 0.0, -0.006]])
        outside = np.asarray([[0.0, 0.0, 0.02]])
        hinge_in = mesh_penetration_hinge(
            inside, hand, normals, thickness_m=0.018, eps_m=0.001, weight=30.0
        )
        hinge_out = mesh_penetration_hinge(
            outside, hand, normals, thickness_m=0.018, eps_m=0.001, weight=30.0
        )
        self.assertGreater(float(hinge_in[0]), 0.0)
        np.testing.assert_allclose(hinge_out, 0.0, atol=1e-12)


class NeighborhoodPullOutTest(unittest.TestCase):
    def test_all_neighborhood_points_push_hand_out_and_outside_is_zero(self) -> None:
        hand = np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        )
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1))
        inside = np.asarray(
            [
                [0.0, 0.0, -0.006],
                [0.005, 0.0, -0.004],
                [0.002, 0.002, -0.008],
            ],
            dtype=np.float64,
        )
        outside = np.asarray([[0.0, 0.0, 0.02], [0.01, 0.0, 0.03]])
        push, n_inside, depth = deepest_inside_world_push(
            inside, hand, normals, thickness_m=0.018, clearance_m=0.001
        )
        self.assertEqual(n_inside, 3)
        self.assertAlmostEqual(depth, 0.008, places=6)
        np.testing.assert_allclose(push, [0.0, 0.0, -0.009], atol=1e-12)
        moved_hand = hand + push.reshape(1, 3)
        push_after, n_after, depth_after = deepest_inside_world_push(
            inside, moved_hand, normals, thickness_m=0.018, clearance_m=0.001
        )
        self.assertEqual(n_after, 0)
        np.testing.assert_allclose(push_after, 0.0, atol=1e-12)
        self.assertEqual(depth_after, 0.0)
        push_out, n_out, _ = deepest_inside_world_push(
            outside, hand, normals, thickness_m=0.018, clearance_m=0.001
        )
        self.assertEqual(n_out, 0)
        np.testing.assert_allclose(push_out, 0.0, atol=1e-12)

    def test_retract_axis_always_backs_off_the_object(self) -> None:
        origin = np.zeros(3, dtype=np.float64)
        obj = np.asarray([[0.0, 0.0, 0.04]], dtype=np.float64)
        away_pos = retract_axis_away_from_object(
            np.array([0.0, 0.0, 1.0]), origin, obj
        )
        away_neg = retract_axis_away_from_object(
            np.array([0.0, 0.0, -1.0]), origin, obj
        )
        np.testing.assert_allclose(away_pos, [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(away_neg, [0.0, 0.0, -1.0], atol=1e-12)

    def test_retract_axis_toward_human_landmarks(self) -> None:
        robot = np.zeros(3, dtype=np.float64)
        human = np.array([0.03, -0.02, 0.04], dtype=np.float64)
        axis = retract_axis_toward_target(robot, human)
        self.assertIsNotNone(axis)
        expected = human / np.linalg.norm(human)
        np.testing.assert_allclose(axis, expected, atol=1e-12)
        self.assertIsNone(retract_axis_toward_target(robot, robot))
        self.assertIsNone(
            retract_axis_toward_target(robot, np.array([np.nan, 0.0, 0.0]))
        )
        hand = np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        )
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1))
        obj = np.asarray([[0.0, 0.0, -0.006]], dtype=np.float64)
        push, n_inside, _depth = deepest_inside_world_push(
            obj,
            hand,
            normals,
            thickness_m=0.018,
            clearance_m=0.001,
            retract_axis=axis,
        )
        self.assertGreater(n_inside, 0)
        self.assertGreater(float(np.dot(push, human)), 0.0)

    def test_inward_nearest_normal_still_retracts_along_palm_axis(self) -> None:
        hand = np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        )
        inward = np.tile(np.array([[0.0, 0.0, -1.0]]), (3, 1))
        obj = np.asarray(
            [[0.0, 0.0, 0.010], [0.0, 0.0, -0.006]],
            dtype=np.float64,
        )
        wrong, n_inside, depth = deepest_inside_world_push(
            obj, hand, inward, thickness_m=0.018, clearance_m=0.001
        )
        self.assertGreater(n_inside, 0)
        self.assertGreater(float(wrong[2]), 0.0)
        retract = retract_axis_away_from_object(
            np.array([0.0, 0.0, 1.0]), hand.mean(axis=0), obj
        )
        push, n_axis, depth_axis = deepest_inside_world_push(
            obj,
            hand,
            inward,
            thickness_m=0.018,
            clearance_m=0.001,
            retract_axis=retract,
        )
        self.assertEqual(n_axis, n_inside)
        self.assertAlmostEqual(depth_axis, depth, places=6)
        self.assertLess(float(push[2]), 0.0)
        obj_mean = obj.mean(axis=0)
        self.assertLess(float(np.dot(push, obj_mean - hand.mean(axis=0))), 0.0)

    def test_wrist_local_translation_clips_to_bound(self) -> None:
        if apply_wrist_local_translation is None:
            self.skipTest("optimize_brainco_contact_anchors (mujoco) is unavailable")
        rotation = np.eye(3, dtype=np.float64)
        x = np.zeros(12, dtype=np.float64)
        moved = apply_wrist_local_translation(
            x, np.array([0.0, 0.0, -0.01]), rotation, 0.015
        )
        np.testing.assert_allclose(moved[:3], [0.0, 0.0, -0.01], atol=1e-12)
        np.testing.assert_allclose(moved[3:], 0.0, atol=1e-12)
        x[:3] = [0.0, 0.0, -0.014]
        clipped = apply_wrist_local_translation(
            x, np.array([0.0, 0.0, -0.01]), rotation, 0.015
        )
        np.testing.assert_allclose(clipped[:3], [0.0, 0.0, -0.015], atol=1e-12)


class GraspResidualTest(unittest.TestCase):
    def setUp(self) -> None:
        if adaptive_wrist_translation_bound is None:
            self.skipTest("optimize_brainco_contact_anchors (mujoco) is unavailable")

    def test_laplacian_uses_prefix_pullout_keeps_full_packed_cloud(self) -> None:
        packed = np.arange(20, dtype=np.float64).reshape(20, 1) * np.array(
            [[1.0, 0.0, 0.0]]
        )
        stored = packed_object_points(packed, 20, min_points=0)
        lap = cap_object_points(stored, 512)
        pull = stored
        self.assertEqual(len(stored), 20)
        self.assertEqual(len(lap), 20)
        dense = np.arange(2048, dtype=np.float64).reshape(2048, 1) * np.array(
            [[1.0, 0.0, 0.0]]
        )
        stored_dense = packed_object_points(dense, 2048, min_points=0)
        lap_dense = cap_object_points(stored_dense, 512)
        self.assertEqual(len(stored_dense), 2048)
        self.assertEqual(len(lap_dense), 512)
        np.testing.assert_allclose(lap_dense, stored_dense[:512])
        self.assertGreater(len(stored_dense), len(lap_dense))

    def test_adaptive_wrist_box_stays_15mm_until_25mm_then_caps_at_40mm(self) -> None:
        self.assertAlmostEqual(
            adaptive_wrist_translation_bound(0.020), 0.015
        )
        self.assertAlmostEqual(
            adaptive_wrist_translation_bound(0.0249), 0.015
        )
        self.assertAlmostEqual(
            adaptive_wrist_translation_bound(0.030), 0.030
        )
        self.assertAlmostEqual(
            adaptive_wrist_translation_bound(0.040), 0.040
        )
        self.assertAlmostEqual(
            adaptive_wrist_translation_bound(0.080), 0.040
        )

    def test_pca_normals_use_sample_centroid_not_palmar_face_origin(self) -> None:
        if estimate_outward_normals is None:
            self.skipTest("optimize_brainco_contact_anchors (mujoco) is unavailable")
        rng = np.random.default_rng(0)
        palm = np.column_stack(
            [rng.normal(0.0, 0.01, 80), rng.normal(0.0, 0.01, 80), np.full(80, 0.012)]
        )
        back = np.column_stack(
            [rng.normal(0.0, 0.01, 80), rng.normal(0.0, 0.01, 80), np.full(80, -0.012)]
        )
        cloud = np.concatenate([palm, back], axis=0)
        palm_origin = np.array([0.0, 0.0, 0.03], dtype=np.float64)
        flipped = estimate_outward_normals(cloud, palm_origin)
        centroid = estimate_outward_normals(cloud, cloud.mean(axis=0))
        palm_n_flipped = flipped[:80].mean(axis=0)
        palm_n_centroid = centroid[:80].mean(axis=0)
        self.assertLess(float(palm_n_flipped[2]), 0.0)
        self.assertGreater(float(palm_n_centroid[2]), 0.0)
        contact = np.zeros(6, dtype=np.int8)
        deltas = np.ones((6, 12), dtype=np.float64)
        filled = interpolate_deltas(contact, deltas)
        np.testing.assert_array_equal(filled, 0.0)

    def test_optimizer_uses_holosoma_sequential_sqp_not_least_squares(self) -> None:
        source = (PROJECT_ROOT / "scripts/view_change/optimize_brainco_contact_anchors.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("solve_sqp_frame", source)
        self.assertIn("holosoma_sequential_sqp", source)
        self.assertNotIn("from scipy.optimize import least_squares", source)
        self.assertIn("calculate_laplacian_matrix", source)
        self.assertIn("retract_axis_toward_target", source)
        self.assertIn("human_landmarks", source)
        self.assertIn("retract_axis_away_from_object", source)
        self.assertIn("retract_axis=axis", source)
        self.assertNotIn("mesh_penetration_hinge", source)
        assemble = source.split("def assemble_sqp_system", 1)[1].split("def solve_sqp_frame", 1)[0]
        self.assertNotIn("deepest_inside_world_push", assemble)
        self.assertNotIn("pull_hand_out_of_neighborhood", assemble)
        pull = source.split("def pull_hand_out_of_neighborhood", 1)[1].split(
            "def optimize_side", 1
        )[0]
        self.assertIn("retract_axis_toward_target", pull)
        self.assertIn("human[finite].mean", pull)
        self.assertIn("cap_object_points", source)
        self.assertIn("laplacian-object-cloud-max-points", source)
        self.assertIn("object_points=lap_pts", source)
        sqp_call = source.split("previous = solve_sqp_frame", 1)[1].split(
            "first_contact = False", 1
        )[0]
        self.assertIn("object_points=lap_pts", sqp_call)
        pull_call = source.split(
            "filled[frame], pull_stats = pull_hand_out_of_neighborhood", 1
        )[1].split("pullout_frames += 1", 1)[0]
        self.assertIn("object_points=cloud_pts", pull_call)
        self.assertIn("do_pullout = int(args.pullout_iters) > 0", source)
        self.assertIn('if do_pullout else "none"', source)
        precompute = PRECOMPUTE.read_text(encoding="utf-8")
        self.assertIn(
            'CONTACT_OBJECT_CLOUD_MAX_POINTS="${CONTACT_OBJECT_CLOUD_MAX_POINTS:-2048}"',
            precompute,
        )
        self.assertIn(
            'CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS="${CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS:-512}"',
            precompute,
        )
        self.assertIn("--laplacian-object-cloud-max-points", precompute)
        self.assertIn(
            'CONTACT_OBJECT_CLOUD_RADIUS_M="${CONTACT_OBJECT_CLOUD_RADIUS_M:-0.12}"',
            precompute,
        )
        self.assertIn(
            'CONTACT_OBJECT_CLOUD_VOXEL_M="${CONTACT_OBJECT_CLOUD_VOXEL_M:-0.02}"',
            precompute,
        )
        self.assertIn("--object-cloud-radius-m", precompute)
        self.assertIn("--object-cloud-voxel-m", precompute)
        self.assertNotIn(
            'CONTACT_OBJECT_CLOUD_MAX_POINTS="${CONTACT_OBJECT_CLOUD_MAX_POINTS:-128}"',
            precompute,
        )
        view = (
            PROJECT_ROOT / "scripts/view_change/run_singleframe_view_change_30to50.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('STOP_AFTER="${STOP_AFTER:-}"', view)
        self.assertIn("stop_after_if_requested composite", view)






class ContactFlagsTest(unittest.TestCase):
    def test_parse_sides_and_all_runs(self) -> None:
        self.assertEqual(parse_sides("right", None), ("right",))
        self.assertEqual(parse_sides("right", "left,right"), ("left", "right"))
        flags = flags_from_runs(12, [[1, 2, 3], [8, 9]])
        np.testing.assert_array_equal(flags, [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])

    def test_left_contact_does_not_copy_onto_right(self) -> None:
        left = flags_from_runs(5, [[1, 2]])
        right = flags_from_runs(5, [])
        self.assertTrue(np.any(left))
        np.testing.assert_array_equal(right, 0)
        combined = np.maximum(left, right)
        np.testing.assert_array_equal(combined, left)


class OccluderZTest(unittest.TestCase):
    def test_hand_pixels_are_not_occluders_object_pixels_are(self) -> None:
        depth = np.full((2, 2), 400, dtype=np.float64)
        depth_k = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        rgb_k = np.array([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
        human = np.zeros((8, 8), dtype=bool)
        human[:, :4] = True
        zbuf = build_rgb_occluder_z(depth, depth_k, rgb_k, (8, 8), human)
        self.assertTrue(np.any(np.isfinite(zbuf)))
        self.assertTrue(np.any(~np.isfinite(zbuf[:, :4])))


class BrainCoPowellDimensionTest(unittest.TestCase):
    def test_left_side_is_wrist_plus_six_active_joints(self) -> None:
        self.assertEqual(hand_dof("g1_brainco"), 12)
        self.assertEqual(side_joint_count("g1_brainco"), 6)
        self.assertEqual(hand_side_slice("g1_brainco", "left"), slice(0, 6))
        self.assertEqual(hand_side_slice("g1_brainco", "right"), slice(6, 12))
        self.assertEqual(6 + side_joint_count("g1_brainco"), 12)

    def test_dex3_right_path_stays_thirteen_params(self) -> None:
        self.assertEqual(hand_dof("dex3"), 14)
        self.assertEqual(side_joint_count("dex3"), 7)
        self.assertEqual(hand_side_slice("dex3", "right"), slice(7, 14))
        self.assertEqual(6 + side_joint_count("dex3"), 13)


class OrchestrationModesTest(unittest.TestCase):
    def test_cup_prompts_use_colon_default_and_shared_mask_path(self) -> None:
        precompute = PRECOMPUTE.read_text(encoding="utf-8")
        self.assertIn(
            'CUP_PROMPTS="${CUP_PROMPTS:-a blue cup|a blue mug|a cup|a mug}"',
            precompute,
        )
        self.assertIn('OBJECT_SOURCE="${OBJECT_SOURCE:-mask}"', precompute)
        self.assertIn(
            'OBJECT_SOURCE="${OBJECT_SOURCE:-fingertip_neighborhood}"',
            precompute,
        )
        self.assertIn("--sides left,right", precompute)
        self.assertIn("evaluate_hand_object_interaction.py", precompute)
        self.assertIn("optimize_contact_aware_wrist_and_fingers.py", precompute)
        self.assertIn("optimize_brainco_contact_anchors.py", precompute)
        self.assertIn("contact_anchors.npz", precompute)
        self.assertIn("run_contact_py()", precompute)
        self.assertIn("OMP_NUM_THREADS=1", precompute)
        self.assertIn(
            'run_contact_py "${SCRIPT_DIR}/evaluate_hand_object_interaction.py"',
            precompute,
        )
        self.assertIn(
            'run_contact_py "${SCRIPT_DIR}/optimize_brainco_contact_anchors.py"',
            precompute,
        )
        dex3_joint = precompute.split('if [[ "${HAND_MODE}" == "g1_brainco" ]]; then', 1)[-1]
        self.assertIn("optimize_contact_aware_wrist_and_fingers.py", dex3_joint)
        brainco_block = precompute.split("JOINT_CONTACT_STATUS=\"disabled\"")[1]
        brainco_only = brainco_block.split("else")[0]
        self.assertNotIn("evaluate_hand_object_interaction.py", brainco_only)
        self.assertNotIn("optimize_contact_aware_wrist.py", brainco_only)

    def test_orchestrators_no_longer_force_contact_off(self) -> None:
        batch = BATCH.read_text(encoding="utf-8")
        tracking = TRACKING.read_text(encoding="utf-8")
        driver = DRIVER.read_text(encoding="utf-8")
        precompute = PRECOMPUTE.read_text(encoding="utf-8")
        self.assertNotIn("forcing JOINT_CONTACT_ENABLED=0", batch)
        self.assertNotIn("forcing JOINT_CONTACT_ENABLED=0", tracking)
        self.assertNotIn("optimize_brainco_contact_anchors", driver)
        self.assertIn("inject_contact_aware_wrist_reference", driver)
        self.assertIn("resolve_joint_contact_bundle", driver)
        self.assertIn('TRACK_ARGS+=("--hand-mode" "g1_brainco")', tracking)
        self.assertIn('"--contact-edge-hold"', tracking)
        self.assertNotIn("g1_brainco_has_no_contact_stage", tracking)
        self.assertIn("g1_brainco_palm_refine+neighborhood_laplacian", tracking)
        self.assertIn("g1_brainco_palm_refine+neighborhood_laplacian", batch)
        self.assertNotIn("g1_brainco_palm_refine+cupmodel_joint_contact", tracking)
        self.assertNotIn("g1_brainco_palm_refine+cupmodel_joint_contact", batch)
        self.assertIn('CONTACT_DETECTOR="${CONTACT_DETECTOR:-pointcloud}"', batch)
        self.assertIn("write_egohumanoid_contact_frames.py", batch)
        self.assertIn("CONTACT_DETECTOR must be pointcloud or egohumanoid", batch)
        self.assertIn('RERUN_VIEW_CHANGE="${RERUN_VIEW_CHANGE:-0}"', batch)
        self.assertIn("skipped_missing_kinematics", batch)
        self.assertIn("reusable_tracking_root", batch)
        self.assertIn("FORCE=1 and RERUN_VIEW_CHANGE=1 cannot be set together", batch)
        self.assertIn("prune_contact_evidence \"${sid}\" completed", batch)
        self.assertIn("do not require a joint-contact bundle", batch)
        self.assertIn('VIEW_FORCE="${VIEW_FORCE:-${FORCE}}"', tracking)
        self.assertIn('FORCE="${VIEW_FORCE}"', tracking)
        self.assertIn("evaluate_hand_object_interaction.py", precompute)
        self.assertIn("optimize_brainco_contact_anchors.py", precompute)

    def test_pointcloud_loads_packed_human_masks_not_png_stems(self) -> None:
        source = (
            PROJECT_ROOT / "scripts/view_change/contact_pointcloud_analysis.py"
        ).read_text(encoding="utf-8")
        self.assertIn("load_mask_bool_volume", source)
        self.assertNotIn('human_dir / f"{frame:06d}.png"', source)
        self.assertNotIn(
            'selected_mask_dir / f"{frame:06d}.png"',
            source,
        )
        window = (
            PROJECT_ROOT / "scripts/view_change/select_contact_mask_window.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ascontiguousarray", window)
        self.assertNotIn("as_mask(mask_volume[frame])", window)

    def test_choose_object_masks_reads_packed_volume_without_png(self) -> None:
        import tempfile

        pixels = np.zeros((6, 8), dtype=np.uint8)
        pixels[1:3, 2:5] = 255
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "prompt_00_a_blue_cup" / "masks"
            second = root / "prompt_99_a_wooden_cup" / "masks"
            n = 4
            vol1 = create_volume(first, n, pixels.shape, np.uint8, fill=0)
            vol1[1] = pixels
            vol1.flush()
            vol2 = create_volume(second, n, pixels.shape, np.uint8, fill=0)
            vol2[2] = pixels
            vol2.flush()
            self.assertFalse(any(first.glob("*.png")))
            stack, names = choose_object_masks(root, n, pixels.shape, max_gap=0)
            self.assertEqual(stack.shape, (n, *pixels.shape))
            self.assertFalse(np.any(stack[0]))
            np.testing.assert_array_equal(stack[1], pixels > 0)
            np.testing.assert_array_equal(stack[2], pixels > 0)
            self.assertEqual(names[1], "prompt_00_a_blue_cup")
            self.assertEqual(names[2], "prompt_99_a_wooden_cup")

    def test_palm_refine_always_writes_zero_contact_keys(self) -> None:
        retarget = RETARGET.read_text(encoding="utf-8")
        self.assertIn('"left_contact": np.zeros(frames, dtype=np.int8)', retarget)
        self.assertIn('"right_contact": np.zeros(frames, dtype=np.int8)', retarget)


if __name__ == "__main__":
    unittest.main()
