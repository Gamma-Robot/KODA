#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from hand_retarget_experiment import (
    INDEX_MCP,
    LITTLE_MCP,
    MCP_INDICES,
    MIDDLE_MCP,
    RING_MCP,
    WRIST,
    batch_frames,
    calibrated_mcp_center,
    landmark_frame,
    mano_palm_vertex_centroid,
    mano_palm_surface_centroid,
    palm_polygon_centroid,
    projected_convex_hull_centroid,
    rigid_fit,
    robot_landmarks,
)


class RigidFitTest(unittest.TestCase):
    def test_recovers_known_transform(self) -> None:
        source = np.asarray(
            [[0.0, 0.0, 0.0], [0.12, 0.0, 0.03], [0.12, 0.0, -0.03]]
        )
        expected_rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
        expected_translation = np.asarray([0.4, -0.7, 1.2])
        target = (expected_rotation @ source.T).T + expected_translation
        rotation, translation = rigid_fit(source, target)
        self.assertTrue(np.allclose(rotation, expected_rotation, atol=1e-9))
        self.assertTrue(np.allclose(translation, expected_translation, atol=1e-9))

    def test_robot_landmarks_are_mirrored_only_in_thickness(self) -> None:
        left = robot_landmarks("left")
        right = robot_landmarks("right")
        self.assertTrue(np.allclose(left.palm, right.palm))
        self.assertAlmostEqual(left.index_root[1], -right.index_root[1])
        self.assertAlmostEqual(left.middle_root[1], -right.middle_root[1])
        self.assertTrue(np.allclose(left.index_root[[0, 2]], right.index_root[[0, 2]]))

    def test_palm_polygon_centroid_is_area_weighted(self) -> None:
        hand = np.zeros((25, 3), dtype=np.float64)
        hand[WRIST] = [0.0, 0.0, 0.0]
        hand[INDEX_MCP] = [1.0, 0.0, 1.0]
        hand[MIDDLE_MCP] = [1.0, 0.0, 0.3]
        hand[RING_MCP] = [1.0, 0.0, -0.3]
        hand[LITTLE_MCP] = [1.0, 0.0, -1.0]
        self.assertTrue(
            np.allclose(palm_polygon_centroid(hand), [2.0 / 3.0, 0.0, 0.0])
        )

    def test_template_palm_kabsch_tracks_known_rigid_motion(self) -> None:
        base = np.zeros((25, 3), dtype=np.float64)
        base[WRIST] = [0.0, 0.0, 0.0]
        base[INDEX_MCP] = [0.10, 0.0, 0.04]
        base[MIDDLE_MCP] = [0.105, 0.0, 0.014]
        base[RING_MCP] = [0.102, 0.0, -0.014]
        base[LITTLE_MCP] = [0.095, 0.0, -0.04]
        transforms = [
            (Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix(), np.asarray([0.4, -0.2, 0.8])),
            (Rotation.from_euler("xyz", [-0.1, 0.25, -0.2]).as_matrix(), np.asarray([0.45, -0.15, 0.82])),
        ]
        hands = np.stack([(rotation @ base.T).T + translation for rotation, translation in transforms])
        positions, rotations = batch_frames(hands, "template_palm_kabsch", "right")
        robot_palm = robot_landmarks("right").palm
        tracked_palm = np.einsum("tij,j->ti", rotations, robot_palm) + positions
        expected_palm = np.stack([palm_polygon_centroid(hand) for hand in hands])
        expected_rotation = np.stack(
            [landmark_frame(hand, INDEX_MCP, LITTLE_MCP) for hand in hands]
        )
        self.assertTrue(np.allclose(tracked_palm, expected_palm, atol=1e-9))
        self.assertTrue(np.allclose(rotations, expected_rotation, atol=1e-9))

    def test_projected_hull_centroid_is_translation_and_rotation_equivariant(self) -> None:
        base = np.zeros((25, 3), dtype=np.float64)
        base[WRIST] = [0.0, 0.0, 0.0]
        base[INDEX_MCP] = [0.10, 0.02, 0.04]
        base[MIDDLE_MCP] = [0.105, 0.01, 0.014]
        base[RING_MCP] = [0.102, -0.01, -0.014]
        base[LITTLE_MCP] = [0.095, -0.02, -0.04]
        rotation = Rotation.from_euler("xyz", [0.25, -0.15, 0.2]).as_matrix()
        translation = np.asarray([0.4, -0.2, 0.8])
        transformed = (rotation @ base.T).T + translation
        expected = (rotation @ projected_convex_hull_centroid(base)) + translation
        self.assertTrue(
            np.allclose(projected_convex_hull_centroid(transformed), expected, atol=1e-9)
        )

    def test_calibrated_mcp_center_reports_alpha(self) -> None:
        hand = np.zeros((4, 25, 3), dtype=np.float64)
        hand[:, WRIST] = [0.0, 0.0, 0.0]
        for frame, alpha in enumerate((0.2, 0.35, 0.5, 0.65)):
            hand[frame, INDEX_MCP] = [1.0, 0.0, 0.7]
            hand[frame, MIDDLE_MCP] = [1.0, 0.0, 0.25]
            hand[frame, RING_MCP] = [1.0, 0.0, -0.25]
            hand[frame, LITTLE_MCP] = [1.0, 0.0, -0.7]
        centers, alpha = calibrated_mcp_center(hand)
        self.assertTrue(np.isfinite(alpha))
        self.assertEqual(centers.shape, (4, 3))
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)

    def test_mano_vertex_palm_centroid_uses_only_palm_region(self) -> None:
        hand = np.zeros((25, 3), dtype=np.float64)
        hand[WRIST] = [0.0, 0.0, 0.0]
        hand[INDEX_MCP] = [1.0, 0.0, 1.0]
        hand[MIDDLE_MCP] = [1.0, 0.0, 0.3]
        hand[RING_MCP] = [1.0, 0.0, -0.3]
        hand[LITTLE_MCP] = [1.0, 0.0, -1.0]
        palm_vertices = [
            [0.65 + 0.01 * i, 0.001 * ((i % 2) * 2 - 1), -0.12 + 0.03 * i]
            for i in range(10)
        ]
        vertices = np.asarray(palm_vertices + [[2.0, 0.0, 0.0]], dtype=np.float64)
        center = mano_palm_vertex_centroid(hand, vertices)
        self.assertLess(float(np.linalg.norm(center - [0.69, 0.0, 0.0])), 0.08)

    def test_mano_surface_centroid_is_area_weighted_and_rejects_finger_face(self) -> None:
        hand = np.zeros((25, 3), dtype=np.float64)
        hand[WRIST] = [0.0, 0.0, 0.0]
        hand[INDEX_MCP] = [1.0, 0.0, 1.0]
        hand[MIDDLE_MCP] = [1.0, 0.0, 0.3]
        hand[RING_MCP] = [1.0, 0.0, -0.3]
        hand[LITTLE_MCP] = [1.0, 0.0, -1.0]
        vertices = np.asarray(
            [
                [0.65, 0.0, 0.0],
                [0.75, 0.0, 0.0],
                [0.65, 0.0, 0.2],
                [0.75, 0.0, 0.2],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
                [2.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 1, 2], [1, 3, 2]] * 4 + [[4, 5, 6]], dtype=np.int64
        )
        center = mano_palm_surface_centroid(hand, vertices, faces)
        self.assertTrue(np.allclose(center, [0.7, 0.0, 0.1], atol=1e-9))


if __name__ == "__main__":
    unittest.main()
