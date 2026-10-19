#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_contact_aware_wrist import (  # noqa: E402
    annotate_finger_contact_metrics,
    apply_residual,
    physical_loss,
    residual_bounds,
    robust_grasp,
    selected_grasp_valid,
)


class ContactAwareWristTest(unittest.TestCase):
    @staticmethod
    def args(require_all: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            require_all_finger_contact=require_all,
            optimization_contact_target_m=0.004,
            optimization_penetration_target_m=0.004,
            optimization_penetration_ratio_target=0.015,
            robust_contact_m=0.007,
            robust_penetration_m=0.0045,
            robust_penetration_ratio=0.018,
            penetration_tolerance_m=0.005,
            penetration_ratio_tolerance=0.02,
        )

    @staticmethod
    def metric(index_contact: bool = True) -> dict:
        groups = {
            name: {
                "surface_distance_min_m": 0.003 if contact else 0.012,
                "contact": contact,
            }
            for name, contact in (
                ("thumb", True),
                ("index", index_contact),
                ("middle", True),
            )
        }
        return {
            "groups": groups,
            "thumb_surface_distance_m": 0.003,
            "opposition_surface_distance_m": 0.003,
            "opposition_normal_dot": -0.8,
            "bilateral_contact": True,
            "valid_grasp": True,
            "hand_penetration_p95_m": 0.001,
            "hand_penetration_ratio": 0.001,
        }

    def test_local_residual_composition(self) -> None:
        base_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        base_position = np.asarray([1.0, 2.0, 3.0])
        residual = np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, np.pi / 4])
        position, rotation = apply_residual(
            base_position, base_rotation, residual
        )
        self.assertTrue(np.allclose(position, [1.0, 2.01, 3.0]))
        expected = base_rotation @ Rotation.from_euler(
            "z", 45, degrees=True
        ).as_matrix()
        self.assertTrue(np.allclose(rotation, expected))

    def test_bounds_cover_six_dof(self) -> None:
        bounds = residual_bounds(0.04, 25.0)
        self.assertEqual(len(bounds), 6)
        self.assertEqual(bounds[0], (-0.04, 0.04))
        self.assertAlmostEqual(bounds[3][1], np.deg2rad(25.0))

    def test_all_finger_mode_rejects_missing_index_contact(self) -> None:
        args = self.args(require_all=True)
        metric = annotate_finger_contact_metrics(
            self.metric(index_contact=False), args
        )
        self.assertFalse(metric["all_finger_contact"])
        self.assertFalse(metric["all_finger_valid_grasp"])
        self.assertFalse(robust_grasp(metric, args))
        self.assertFalse(selected_grasp_valid(metric, args))

    def test_legacy_mode_keeps_best_opposition_contract(self) -> None:
        args = self.args(require_all=False)
        metric = annotate_finger_contact_metrics(
            self.metric(index_contact=False), args
        )
        self.assertTrue(robust_grasp(metric, args))
        self.assertTrue(selected_grasp_valid(metric, args))

    def test_all_finger_loss_penalizes_missing_index(self) -> None:
        all_args = self.args(require_all=True)
        legacy_args = self.args(require_all=False)
        metric = annotate_finger_contact_metrics(
            self.metric(index_contact=False), all_args
        )
        self.assertGreater(
            physical_loss(metric, all_args), physical_loss(metric, legacy_args)
        )


if __name__ == "__main__":
    unittest.main()
