#!/usr/bin/env python3
"""Palm-axis signs must use the MJCF correction, not causal latching."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from brainco_hand_retarget import (
    PALM_SIGN_ALIGN_MIN_ABS_DOT,
    _compose_palm_correction,
    _enforce_palm_normal_continuity,
    _raw_palm_sign_confidence,
    estimate_palm_poses_from_keypoints,
)


def _hand_along(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """21 MANO-like points: MCPs along +z, laterals along +x in local frame."""
    local = np.zeros((21, 3), dtype=np.float64)
    local[5] = [0.03, 0.0, 0.08]
    local[9] = [0.01, 0.0, 0.09]
    local[13] = [-0.01, 0.0, 0.087]
    local[17] = [-0.03, 0.0, 0.075]
    return local @ np.asarray(rotation, dtype=np.float64).T + translation


def _end_angle_deg(rotations: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(rotations[0].T @ rotations[-1]) - 1) / 2, -1, 1)
            )
        )
    )


class PalmSignAlignmentTest(unittest.TestCase):
    def test_compose_correction_is_a_z_rotation(self) -> None:
        correction = _compose_palm_correction(91.0, None)
        np.testing.assert_allclose(correction.T @ correction, np.eye(3), atol=1e-12)
        self.assertGreater(np.linalg.det(correction), 0.0)
        self.assertAlmostEqual(
            abs(np.degrees(np.arctan2(correction[1, 0], correction[0, 0]))),
            91.0,
            places=6,
        )

    def test_quarter_turn_reference_is_orthogonal_until_correction_is_applied(self) -> None:
        frames = 6
        keypoints = np.stack([_hand_along(np.eye(3), np.zeros(3)) for _ in range(frames)])
        distal = keypoints[:, (5, 9, 13, 17), :].mean(axis=1) - keypoints[:, 0]
        lateral = keypoints[:, 5] - keypoints[:, 17]
        robot_palm = np.repeat(Rotation.from_euler("z", np.deg2rad(90.0)).as_matrix()[None], frames, axis=0)
        unaligned_x, unaligned_z = _raw_palm_sign_confidence(
            lateral, distal, robot_palm[:, :, 0], robot_palm[:, :, 2]
        )
        aligned = np.einsum("tij,jk->tik", robot_palm, _compose_palm_correction(90.0, None))
        aligned_x, aligned_z = _raw_palm_sign_confidence(
            lateral, distal, aligned[:, :, 0], aligned[:, :, 2]
        )
        self.assertLess(float(np.median(unaligned_x)), PALM_SIGN_ALIGN_MIN_ABS_DOT)
        self.assertGreater(float(np.median(unaligned_z)), 0.9)
        self.assertGreater(float(np.median(aligned_x)), 0.9)
        self.assertGreater(float(np.median(aligned_z)), 0.9)

    def test_causal_continuity_latches_a_return_from_a_slow_ramp(self) -> None:
        yaws = np.concatenate(
            [
                np.zeros(4),
                np.linspace(0.0, np.pi, 6),
                np.full(2, np.pi),
                np.zeros(6),
            ]
        )
        rotations = Rotation.from_euler("z", yaws).as_matrix()
        latched = _enforce_palm_normal_continuity(rotations)
        self.assertGreater(_end_angle_deg(latched), 90.0)

        gated = _enforce_palm_normal_continuity(
            rotations, degenerate=np.zeros(len(yaws), dtype=bool)
        )
        np.testing.assert_allclose(gated, rotations)
        self.assertLess(_end_angle_deg(gated), 1.0)

    def test_absolute_signs_recover_after_interpolated_orientation_ramp(self) -> None:
        frames = 24
        keypoints = np.zeros((frames, 21, 3), dtype=np.float64)
        spike = Rotation.from_euler("z", np.deg2rad(140.0))
        for index in range(frames):
            if 8 <= index < 16:
                alpha = (index - 8) / 8.0
                rotation = Rotation.from_rotvec(alpha * spike.as_rotvec()).as_matrix()
            else:
                rotation = np.eye(3)
            keypoints[index] = _hand_along(rotation, np.zeros(3))
        wrist = np.repeat(np.eye(4)[None], frames, axis=0)
        poses = estimate_palm_poses_from_keypoints(
            keypoints,
            wrist,
            np.eye(4),
            smooth_radius=2,
            local_z_correction_deg=0.0,
        )
        self.assertLess(_end_angle_deg(poses[:, :3, :3]), 20.0)

    def test_brainco_mounting_plus_ramp_does_not_latch(self) -> None:
        """Robot palm is a quarter-turn from MANO; C is that same quarter-turn."""
        frames = 24
        robot_palm = Rotation.from_euler("z", np.deg2rad(90.0)).as_matrix()
        keypoints = np.zeros((frames, 21, 3), dtype=np.float64)
        spike = Rotation.from_euler("z", np.deg2rad(140.0))
        for index in range(frames):
            if 8 <= index < 16:
                alpha = (index - 8) / 8.0
                constructed = Rotation.from_rotvec(alpha * spike.as_rotvec()).as_matrix()
            else:
                constructed = np.eye(3)
            keypoints[index] = _hand_along(constructed, np.zeros(3))
        wrist = np.repeat(np.eye(4)[None], frames, axis=0)
        wrist[:, :3, :3] = robot_palm
        poses = estimate_palm_poses_from_keypoints(
            keypoints,
            wrist,
            np.eye(4),
            smooth_radius=2,
            local_z_correction_deg=90.0,
        )
        self.assertLess(_end_angle_deg(poses[:, :3, :3]), 25.0)

    def test_guardrail_rejects_a_still_degenerate_x_axis(self) -> None:
        frames = 6
        keypoints = np.stack([_hand_along(np.eye(3), np.zeros(3)) for _ in range(frames)])
        wrist = np.repeat(np.eye(4)[None], frames, axis=0)
        wrist[:, :3, :3] = Rotation.from_euler("z", np.deg2rad(90.0)).as_matrix()
        with self.assertRaises(ValueError) as ctx:
            estimate_palm_poses_from_keypoints(
                keypoints,
                wrist,
                np.eye(4),
                smooth_radius=0,
                local_z_correction_deg=0.1,
            )
        self.assertIn("degenerate", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
