#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inject_contact_aware_wrist_reference import (  # noqa: E402
    continuous_wxyz,
    inject_contact_reference,
    resample_pose,
    wxyz_to_matrix,
)


class InjectContactAwareReferenceTest(unittest.TestCase):
    def test_resample_pose_repeats_single_identity_candidate(self) -> None:
        position, rotation = resample_pose(
            np.asarray([0.0]),
            np.asarray([[0.2, -0.1, 0.8]]),
            np.asarray([np.eye(3)]),
            np.asarray([0.013]),
        )
        np.testing.assert_allclose(position, [[0.2, -0.1, 0.8]])
        np.testing.assert_allclose(rotation, [np.eye(3)])

    def test_single_contact_frame_maps_to_nearest_target_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = 20
            reference = {
                "joint_pos": np.zeros((frames, 1), dtype=np.float32),
                "joint_vel": np.zeros((frames, 1), dtype=np.float32),
                "body_pos_w": np.zeros((frames, 2, 3), dtype=np.float32),
                "body_quat_w": np.zeros((frames, 2, 4), dtype=np.float32),
                "body_lin_vel_w": np.zeros((frames, 2, 3), dtype=np.float32),
                "body_ang_vel_w": np.zeros((frames, 2, 3), dtype=np.float32),
                "body_names": np.asarray(["pelvis", "right_wrist_yaw_link"]),
                "joint_names": np.asarray(["joint_a"]),
                "fps": np.asarray(50.0, dtype=np.float32),
            }
            reference["body_quat_w"][..., 0] = 1.0
            identity = np.tile(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
            )
            targets = {
                "left_wrist": np.zeros((frames, 3), dtype=np.float32),
                "right_wrist": np.zeros((frames, 3), dtype=np.float32),
                "left_wrist_quat_wxyz": identity.copy(),
                "right_wrist_quat_wxyz": identity.copy(),
                "kimodo_pelvis_heading_quat_wxyz": identity.copy(),
                "timestamps": np.arange(frames, dtype=np.float64) / 50.0,
                "fps_session": np.asarray(50.0, dtype=np.float32),
            }
            native_frames = 10
            candidate_position = np.zeros((native_frames, 3), dtype=np.float64)
            candidate_position[1] = [0.2, -0.1, 0.8]
            contact = {
                "timestamps": np.arange(native_frames, dtype=np.float64) / 30.0,
                "frame_indices": np.asarray([1], dtype=np.int64),
                "causal_position": candidate_position,
                "causal_rotation": np.tile(np.eye(3), (native_frames, 1, 1)),
            }
            reference_path = root / "reference.npz"
            targets_path = root / "targets.npz"
            meta_path = root / "meta.json"
            contact_path = root / "contact.npz"
            output_reference = root / "output_reference.npz"
            output_targets = root / "output_targets.npz"
            output_meta = root / "output_meta.json"
            np.savez_compressed(reference_path, **reference)
            np.savez_compressed(targets_path, **targets)
            np.savez_compressed(contact_path, **contact)
            meta_path.write_text(json.dumps({"method": "template_palm_kabsch"}))

            report = inject_contact_reference(
                base_reference=reference_path,
                base_targets=targets_path,
                base_meta=meta_path,
                contact_candidates=contact_path,
                output_reference=output_reference,
                output_targets=output_targets,
                output_meta=output_meta,
                edge_hold_policy="edge_hold",
            )

            selected = report["contact_aware_injection"][
                "target_frame_range_inclusive"
            ]
            self.assertEqual(selected, [2, 2])
            with np.load(output_reference, allow_pickle=False) as output:
                np.testing.assert_allclose(
                    output["body_pos_w"][selected[0], 1], [0.2, -0.1, 0.8]
                )

    def test_edge_hold_composes_residual_with_moving_template_palm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = 8
            timestamps = np.arange(frames, dtype=np.float64) / 10.0
            base_position = np.stack(
                [timestamps, 2.0 * timestamps, np.ones(frames)], axis=1
            )
            base_angle = np.arange(frames, dtype=np.float64) * 5.0
            base_rotation = Rotation.from_euler("z", base_angle, degrees=True).as_matrix()
            reference = {
                "joint_pos": np.zeros((frames, 1), dtype=np.float32),
                "joint_vel": np.zeros((frames, 1), dtype=np.float32),
                "body_pos_w": base_position[:, None].astype(np.float32),
                "body_quat_w": continuous_wxyz(base_rotation)[:, None].astype(np.float32),
                "body_lin_vel_w": np.zeros((frames, 1, 3), dtype=np.float32),
                "body_ang_vel_w": np.zeros((frames, 1, 3), dtype=np.float32),
                "body_names": np.asarray(["right_wrist_yaw_link"]),
                "joint_names": np.asarray(["joint_a"]),
                "fps": np.asarray(10.0, dtype=np.float32),
            }
            targets = {
                "left_wrist": np.zeros((frames, 3), dtype=np.float32),
                "right_wrist": base_position.astype(np.float32),
                "left_wrist_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "right_wrist_quat_wxyz": continuous_wxyz(base_rotation).astype(np.float32),
                "kimodo_pelvis_heading_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "timestamps": timestamps,
                "fps_session": np.asarray(10.0, dtype=np.float32),
            }
            contact_frames = np.asarray([3, 4], dtype=np.int64)
            translation_offset = np.asarray([0.03, -0.02, 0.01])
            rotation_offset = Rotation.from_euler("x", 12.0, degrees=True).as_matrix()
            causal_position = base_position.copy()
            causal_rotation = base_rotation.copy()
            causal_position[contact_frames] += translation_offset
            causal_rotation[contact_frames] = np.einsum(
                "tij,jk->tik", base_rotation[contact_frames], rotation_offset
            )
            contact = {
                "timestamps": timestamps,
                "frame_indices": contact_frames,
                "causal_position": causal_position,
                "causal_rotation": causal_rotation,
            }
            reference_path = root / "reference.npz"
            targets_path = root / "targets.npz"
            meta_path = root / "meta.json"
            contact_path = root / "contact.npz"
            output_reference = root / "output_reference.npz"
            output_targets = root / "output_targets.npz"
            output_meta = root / "output_meta.json"
            np.savez_compressed(reference_path, **reference)
            np.savez_compressed(targets_path, **targets)
            np.savez_compressed(contact_path, **contact)
            meta_path.write_text(json.dumps({"method": "template_palm_kabsch"}))

            inject_contact_reference(
                base_reference=reference_path,
                base_targets=targets_path,
                base_meta=meta_path,
                contact_candidates=contact_path,
                output_reference=output_reference,
                output_targets=output_targets,
                output_meta=output_meta,
                edge_hold_policy="edge_hold",
            )

            with np.load(output_reference, allow_pickle=False) as output:
                actual_position = np.asarray(output["body_pos_w"][:, 0])
                actual_rotation = wxyz_to_matrix(output["body_quat_w"][:, 0])
            np.testing.assert_allclose(
                actual_position, base_position + translation_offset, atol=1.0e-6
            )
            expected_rotation = np.einsum(
                "tij,jk->tik", base_rotation, rotation_offset
            )
            np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1.0e-6)

    def test_injection_changes_only_selected_wrist_and_matching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = 20
            body_names = np.asarray(
                ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]
            )
            reference = {
                "joint_pos": np.zeros((frames, 2), dtype=np.float32),
                "joint_vel": np.zeros((frames, 2), dtype=np.float32),
                "body_pos_w": np.zeros((frames, 3, 3), dtype=np.float32),
                "body_quat_w": np.zeros((frames, 3, 4), dtype=np.float32),
                "body_lin_vel_w": np.zeros((frames, 3, 3), dtype=np.float32),
                "body_ang_vel_w": np.zeros((frames, 3, 3), dtype=np.float32),
                "body_names": body_names,
                "joint_names": np.asarray(["joint_a", "joint_b"]),
                "fps": np.asarray(10.0, dtype=np.float32),
            }
            reference["body_quat_w"][..., 0] = 1.0
            targets = {
                "left_wrist": np.zeros((frames, 3), dtype=np.float32),
                "right_wrist": np.zeros((frames, 3), dtype=np.float32),
                "left_wrist_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "right_wrist_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "kimodo_pelvis_heading_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "timestamps": np.arange(frames, dtype=np.float64) / 10.0,
                "fps_session": np.asarray(10.0, dtype=np.float32),
            }
            native_frames = 10
            contact_rotation = Rotation.from_rotvec(
                np.tile([0.0, 0.0, np.deg2rad(30.0)], (native_frames, 1))
            ).as_matrix()
            contact = {
                "timestamps": np.arange(native_frames, dtype=np.float64) / 5.0,
                "frame_indices": np.arange(3, 7, dtype=np.int64),
                "causal_position": np.tile(
                    np.asarray([0.2, -0.1, 0.8]), (native_frames, 1)
                ),
                "causal_rotation": contact_rotation,
                "reference_causal_position": np.tile(
                    np.asarray([9.0, 9.0, 9.0]), (native_frames, 1)
                ),
                "reference_causal_rotation": np.tile(
                    np.eye(3), (native_frames, 1, 1)
                ),
            }
            reference_path = root / "reference.npz"
            targets_path = root / "targets.npz"
            meta_path = root / "meta.json"
            contact_path = root / "contact.npz"
            output_reference = root / "output_reference.npz"
            output_targets = root / "output_targets.npz"
            output_meta = root / "output_meta.json"
            np.savez_compressed(reference_path, **reference)
            np.savez_compressed(targets_path, **targets)
            np.savez_compressed(contact_path, **contact)
            meta_path.write_text(json.dumps({"method": "template_palm_kabsch"}))

            report = inject_contact_reference(
                base_reference=reference_path,
                base_targets=targets_path,
                base_meta=meta_path,
                contact_candidates=contact_path,
                output_reference=output_reference,
                output_targets=output_targets,
                output_meta=output_meta,
                transition_frames=2,
            )
            with np.load(output_reference, allow_pickle=False) as output:
                np.testing.assert_array_equal(output["joint_pos"], reference["joint_pos"])
                np.testing.assert_array_equal(
                    output["body_pos_w"][:, :2], reference["body_pos_w"][:, :2]
                )
                selected = np.asarray(output["contact_aware_target_indices"])
                np.testing.assert_allclose(
                    output["body_pos_w"][selected, 2],
                    np.tile([0.2, -0.1, 0.8], (len(selected), 1)),
                )
                angle = Rotation.from_matrix(
                    wxyz_to_matrix(output["body_quat_w"][selected, 2])
                ).as_euler("zyx", degrees=True)[:, 0]
                np.testing.assert_allclose(angle, 30.0, atol=1.0e-5)
                self.assertGreater(
                    np.linalg.norm(output["body_lin_vel_w"][:, 2]), 0.0
                )
            with np.load(output_targets, allow_pickle=False) as output:
                np.testing.assert_array_equal(output["left_wrist"], targets["left_wrist"])
                np.testing.assert_allclose(
                    output["right_wrist"][selected],
                    np.tile([0.2, -0.1, 0.8], (len(selected), 1)),
                )
            self.assertEqual(
                report["contact_aware_injection"]["tracking_fields"],
                "causal_position/causal_rotation",
            )

    def test_left_injection_changes_only_left_wrist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = 20
            body_names = np.asarray(
                ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]
            )
            reference = {
                "joint_pos": np.zeros((frames, 2), dtype=np.float32),
                "joint_vel": np.zeros((frames, 2), dtype=np.float32),
                "body_pos_w": np.zeros((frames, 3, 3), dtype=np.float32),
                "body_quat_w": np.zeros((frames, 3, 4), dtype=np.float32),
                "body_lin_vel_w": np.zeros((frames, 3, 3), dtype=np.float32),
                "body_ang_vel_w": np.zeros((frames, 3, 3), dtype=np.float32),
                "body_names": body_names,
                "joint_names": np.asarray(["joint_a", "joint_b"]),
                "fps": np.asarray(10.0, dtype=np.float32),
            }
            reference["body_quat_w"][..., 0] = 1.0
            targets = {
                "left_wrist": np.zeros((frames, 3), dtype=np.float32),
                "right_wrist": np.zeros((frames, 3), dtype=np.float32),
                "left_wrist_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "right_wrist_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "kimodo_pelvis_heading_quat_wxyz": np.tile(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)
                ),
                "timestamps": np.arange(frames, dtype=np.float64) / 10.0,
                "fps_session": np.asarray(10.0, dtype=np.float32),
            }
            native_frames = 10
            contact_rotation = Rotation.from_rotvec(
                np.tile([0.0, 0.0, np.deg2rad(30.0)], (native_frames, 1))
            ).as_matrix()
            contact = {
                "timestamps": np.arange(native_frames, dtype=np.float64) / 5.0,
                "frame_indices": np.arange(3, 7, dtype=np.int64),
                "causal_position": np.tile(
                    np.asarray([0.2, -0.1, 0.8]), (native_frames, 1)
                ),
                "causal_rotation": contact_rotation,
                "reference_causal_position": np.tile(
                    np.asarray([9.0, 9.0, 9.0]), (native_frames, 1)
                ),
                "reference_causal_rotation": np.tile(
                    np.eye(3), (native_frames, 1, 1)
                ),
            }
            reference_path = root / "reference.npz"
            targets_path = root / "targets.npz"
            meta_path = root / "meta.json"
            contact_path = root / "contact.npz"
            output_reference = root / "output_reference.npz"
            output_targets = root / "output_targets.npz"
            output_meta = root / "output_meta.json"
            np.savez_compressed(reference_path, **reference)
            np.savez_compressed(targets_path, **targets)
            np.savez_compressed(contact_path, **contact)
            meta_path.write_text(json.dumps({"method": "template_palm_kabsch"}))

            inject_contact_reference(
                base_reference=reference_path,
                base_targets=targets_path,
                base_meta=meta_path,
                contact_candidates=contact_path,
                output_reference=output_reference,
                output_targets=output_targets,
                output_meta=output_meta,
                side="left",
                transition_frames=2,
            )
            with np.load(output_reference, allow_pickle=False) as output:
                np.testing.assert_array_equal(output["joint_pos"], reference["joint_pos"])
                np.testing.assert_array_equal(
                    output["body_pos_w"][:, 0], reference["body_pos_w"][:, 0]
                )
                np.testing.assert_array_equal(
                    output["body_pos_w"][:, 2], reference["body_pos_w"][:, 2]
                )
                selected = np.asarray(output["contact_aware_target_indices"])
                np.testing.assert_allclose(
                    output["body_pos_w"][selected, 1],
                    np.tile([0.2, -0.1, 0.8], (len(selected), 1)),
                )
            with np.load(output_targets, allow_pickle=False) as output:
                np.testing.assert_array_equal(output["right_wrist"], targets["right_wrist"])
                np.testing.assert_allclose(
                    output["left_wrist"][selected],
                    np.tile([0.2, -0.1, 0.8], (len(selected), 1)),
                )


if __name__ == "__main__":
    unittest.main()
