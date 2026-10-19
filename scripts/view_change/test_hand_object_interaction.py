#!/usr/bin/env python3

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_hand_object_interaction import (
    BRAINCO_HAND_JOINT_NAMES,
    CupModel,
    Dex3MeshSampler,
    copy_cup_model,
    cup_temporal_error,
    evaluate_configuration,
    fit_upright_tapered_cylinder,
    interpolate_cup_model,
    nearest_index,
    point_group_metrics,
)


class CupGeometryTest(unittest.TestCase):
    def test_nearest_index_chooses_earlier_on_tie(self) -> None:
        time = np.asarray([0.0, 0.02, 0.04])
        self.assertEqual(nearest_index(time, 0.03), 1)

    def test_robust_tapered_cylinder_fit_rejects_handle(self) -> None:
        rng = np.random.default_rng(7)
        center = np.asarray([0.25, -0.40])
        z_reference = 0.9
        z = rng.uniform(0.84, 0.96, size=1800)
        angle = rng.uniform(-2.7, 2.7, size=1800)
        radius = 0.052 + 0.05 * (z - z_reference)
        body = np.column_stack(
            [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                z,
            ]
        )
        body += rng.normal(scale=0.001, size=body.shape)
        handle = np.column_stack(
            [
                rng.uniform(center[0] + 0.06, center[0] + 0.11, 350),
                rng.uniform(center[1] - 0.018, center[1] + 0.018, 350),
                rng.uniform(0.86, 0.95, 350),
            ]
        )
        model, inliers = fit_upright_tapered_cylinder(
            np.concatenate([body, handle]),
            frame=0,
            seed=11,
            ransac_iterations=700,
        )
        self.assertTrue(np.allclose(model.center_xy, center, atol=0.004))
        self.assertAlmostEqual(model.radius_reference, 0.052, delta=0.004)
        self.assertAlmostEqual(model.radius_slope, 0.05, delta=0.035)
        self.assertGreater(np.count_nonzero(inliers), 1500)

    def test_contact_and_penetration_are_separate(self) -> None:
        cup = CupModel(
            frame=0,
            center_xy=(0.0, 0.0),
            z_reference=0.5,
            radius_reference=0.05,
            radius_slope=0.0,
            z_min=0.4,
            z_max=0.6,
            points=100,
            inliers=100,
            inlier_fraction=1.0,
            residual_p50_m=0.0,
            residual_p95_m=0.0,
            depth_frame=0,
            depth_time_offset_ms=0.0,
            camera_time_offset_ms=0.0,
        )
        contact = np.asarray(
            [[0.051, 0.0, 0.5], [0.052, 0.0, 0.5], [0.053, 0.0, 0.5]]
        )
        contact_metric, _ = point_group_metrics(contact, cup, 0.008)
        self.assertTrue(contact_metric["contact"])
        self.assertEqual(contact_metric["penetrating_points"], 0)
        inside = np.asarray(
            [[0.030, 0.0, 0.5], [0.031, 0.0, 0.5], [0.032, 0.0, 0.5]]
        )
        inside_metric, _ = point_group_metrics(inside, cup, 0.008)
        self.assertFalse(inside_metric["contact"])
        self.assertEqual(inside_metric["penetrating_points"], 3)
        self.assertGreater(inside_metric["penetration_max_m"], 0.015)

    def test_failed_fit_is_interpolated_only_between_valid_models(self) -> None:
        def model(frame: int, x: float, radius: float) -> CupModel:
            return CupModel(
                frame=frame,
                center_xy=(x, 2.0 * x),
                z_reference=0.5,
                radius_reference=radius,
                radius_slope=0.0,
                z_min=0.4,
                z_max=0.6,
                points=100,
                inliers=80,
                inlier_fraction=0.8,
                residual_p50_m=0.001,
                residual_p95_m=0.003,
                depth_frame=frame,
                depth_time_offset_ms=0.0,
                camera_time_offset_ms=0.0,
            )

        result = interpolate_cup_model(
            2,
            model(0, 0.0, 0.03),
            model(4, 0.04, 0.05),
            {
                "depth_frame": 12,
                "depth_time_offset_ms": 1.5,
                "camera_time_offset_ms": -2.0,
            },
        )
        self.assertTrue(np.allclose(result.center_xy, [0.02, 0.04]))
        self.assertAlmostEqual(result.radius_reference, 0.04)
        self.assertEqual(result.depth_frame, 12)

    def test_temporal_gate_rejects_depth_layer_jump(self) -> None:
        base = CupModel(
            frame=10,
            center_xy=(0.4, 1.39),
            z_reference=1.02,
            radius_reference=0.036,
            radius_slope=0.0,
            z_min=0.96,
            z_max=1.08,
            points=100,
            inliers=80,
            inlier_fraction=0.8,
            residual_p50_m=0.001,
            residual_p95_m=0.003,
            depth_frame=10,
            depth_time_offset_ms=0.0,
            camera_time_offset_ms=0.0,
        )
        jumped = copy_cup_model(
            11,
            base,
            {
                "depth_frame": 11,
                "depth_time_offset_ms": 0.0,
                "camera_time_offset_ms": 0.0,
            },
        )
        jumped = CupModel(
            **{
                **jumped.__dict__,
                "center_xy": (0.48, 1.40),
                "z_reference": 0.75,
            }
        )
        error = cup_temporal_error(
            base,
            jumped,
            center_step_m=0.05,
            radius_step_m=0.012,
            height_step_m=0.04,
        )
        self.assertIsNotNone(error)


class BrainCoSamplerTest(unittest.TestCase):
    def test_brainco_sampler_accepts_twelve_dof_qpos(self) -> None:
        self.assertEqual(len(BRAINCO_HAND_JOINT_NAMES), 12)
        sampler = Dex3MeshSampler(
            "left", samples_per_geom=8, seed=3, hand_mode="g1_brainco"
        )
        samples = sampler.local_samples(np.zeros(12, dtype=np.float64))
        for group in ("palm", "thumb", "index", "middle"):
            self.assertIn(group, samples)
            self.assertGreater(len(samples[group]), 0)
        self.assertEqual(len(sampler.hand_joint_names), 12)
        with self.assertRaises(ValueError):
            sampler.local_samples(np.zeros(14, dtype=np.float64))


def _reference_evaluate(local_samples, wrist_position, wrist_rotation, cup, contact_m=0.008):
    world_samples = {
        group: points @ wrist_rotation.T + wrist_position
        for group, points in local_samples.items()
    }
    groups = {}
    penetrations = []
    for group, points in world_samples.items():
        metric, penetration = point_group_metrics(points, cup, contact_m)
        groups[group] = metric
        penetrations.append(penetration)
    all_penetration = np.concatenate(penetrations)
    positive = all_penetration[all_penetration > 0.0]
    thumb = groups["thumb"]
    opposition = min(
        (groups["index"], groups["middle"]),
        key=lambda value: (
            not bool(value["contact"]),
            float(value["surface_distance_min_m"]),
        ),
    )
    normal_dot = float(
        thumb["closest_radial_normal_x"] * opposition["closest_radial_normal_x"]
        + thumb["closest_radial_normal_y"] * opposition["closest_radial_normal_y"]
    )
    return {
        "groups": groups,
        "thumb_surface_distance_m": float(thumb["surface_distance_min_m"]),
        "opposition_surface_distance_m": float(opposition["surface_distance_min_m"]),
        "opposition_normal_dot": normal_dot,
        "bilateral_contact": bool(
            thumb["contact"] and opposition["contact"] and normal_dot <= -0.20
        ),
        "hand_penetration_ratio": float(np.mean(all_penetration > 0.0)),
        "hand_penetration_mean_positive_m": (
            float(np.mean(positive)) if positive.size else 0.0
        ),
        "hand_penetration_p95_m": (
            float(np.percentile(positive, 95)) if positive.size else 0.0
        ),
        "hand_penetration_max_m": float(np.max(all_penetration)),
    }, world_samples


class FastCupEvalTest(unittest.TestCase):
    @staticmethod
    def _cup() -> CupModel:
        return CupModel(
            frame=0,
            center_xy=(0.02, -0.04),
            z_reference=0.5,
            radius_reference=0.045,
            radius_slope=0.02,
            z_min=0.38,
            z_max=0.62,
            points=100,
            inliers=100,
            inlier_fraction=1.0,
            residual_p50_m=0.0,
            residual_p95_m=0.0,
            depth_frame=0,
            depth_time_offset_ms=0.0,
            camera_time_offset_ms=0.0,
        )

    def test_evaluate_configuration_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(20260830)
        cup = self._cup()
        local = {
            name: rng.normal(scale=0.04, size=(count, 3))
            + np.asarray([0.04, 0.0, 0.5])
            for name, count in (
                ("palm", 400),
                ("thumb", 220),
                ("index", 180),
                ("middle", 180),
                ("ring", 90),
                ("pinky", 90),
            )
        }
        position = np.asarray([0.11, -0.07, 0.02], dtype=np.float64)
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        fast, world = evaluate_configuration(local, position, rotation, cup, 0.008, 0.005, 0.02)
        ref, ref_world = _reference_evaluate(local, position, rotation, cup)
        for key in (
            "thumb_surface_distance_m",
            "opposition_surface_distance_m",
            "opposition_normal_dot",
            "hand_penetration_ratio",
            "hand_penetration_mean_positive_m",
            "hand_penetration_p95_m",
            "hand_penetration_max_m",
        ):
            self.assertAlmostEqual(float(fast[key]), float(ref[key]), places=12, msg=key)
        self.assertEqual(bool(fast["bilateral_contact"]), bool(ref["bilateral_contact"]))
        for group, metric in ref["groups"].items():
            for field, value in metric.items():
                got = fast["groups"][group][field]
                if isinstance(value, bool):
                    self.assertEqual(bool(got), value, msg=f"{group}.{field}")
                elif isinstance(value, int):
                    self.assertEqual(int(got), value, msg=f"{group}.{field}")
                else:
                    self.assertAlmostEqual(
                        float(got), float(value), places=12, msg=f"{group}.{field}"
                    )
            self.assertTrue(np.allclose(world[group], ref_world[group], atol=1e-12))

    def test_packed_local_samples_match_dict(self) -> None:
        sampler = Dex3MeshSampler(
            "right", samples_per_geom=32, seed=3, hand_mode="g1_brainco"
        )
        qpos = np.linspace(-0.2, 0.2, 12)
        packed = np.empty((sampler.packed_count, 3), dtype=np.float64)
        sampler.fill_packed_local_samples(qpos, packed)
        unpacked = sampler.unpack_packed_samples(packed)
        local = sampler.local_samples(qpos)
        self.assertEqual(tuple(unpacked.keys()), tuple(local.keys()))
        self.assertEqual(int(sampler.packed_offsets[-1]), sum(len(v) for v in local.values()))
        for group, points in local.items():
            self.assertTrue(np.allclose(unpacked[group], points, atol=1e-12))

    def test_packed_geom_transform_matches_python_loop(self) -> None:
        import mujoco

        sampler = Dex3MeshSampler(
            "right", samples_per_geom=32, seed=3, hand_mode="g1_brainco"
        )
        qpos = np.linspace(-0.2, 0.2, 12)
        sampler._set_hand_qpos(qpos)
        mujoco.mj_kinematics(sampler.model, sampler.data)
        wrist_position = np.asarray(sampler.data.xpos[sampler.wrist_body], dtype=np.float64)
        wrist_rotation = np.asarray(
            sampler.data.xmat[sampler.wrist_body], dtype=np.float64
        ).reshape(3, 3)
        expected = np.empty((sampler.packed_count, 3), dtype=np.float64)
        for geom_id, vertices, start, end in sampler._packed_geoms:
            geom_position = np.asarray(sampler.data.geom_xpos[geom_id], dtype=np.float64)
            geom_rotation = np.asarray(
                sampler.data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            world = vertices @ geom_rotation.T + geom_position
            expected[start:end] = (world - wrist_position) @ wrist_rotation
        packed = np.empty((sampler.packed_count, 3), dtype=np.float64)
        sampler.fill_packed_local_samples(qpos, packed)
        self.assertTrue(np.allclose(packed, expected, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
