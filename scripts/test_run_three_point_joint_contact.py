from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_three_point_batch import (
    HAND_MODE_BRAINCO,
    Q0MinkOptions,
    ReferenceSmoothingOptions,
    SessionProbe,
    joint_contact_bundle_sha256,
    resolve_joint_contact_bundle,
    sha256,
    write_persistent_kinematics,
)


def test_publish_attaches_synchronized_joint_contact_hand(tmp_path: Path) -> None:
    frames = 5
    body_time = np.arange(frames, dtype=np.float64) / 50.0
    source = tmp_path / "body.npz"
    np.savez_compressed(
        source,
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        timestamps=body_time,
        fps=np.asarray(50.0, dtype=np.float32),
    )

    hand = tmp_path / "hand.npz"
    hand_qpos = np.arange(frames * 14, dtype=np.float32).reshape(frames, 14)
    np.savez_compressed(
        hand,
        qpos=hand_qpos,
        qvel=np.ones_like(hand_qpos),
        hand_joint_names=np.asarray([f"hand_{index}" for index in range(14)]),
        timestamps=body_time,
    )

    candidate = tmp_path / "candidate.npz"
    source_time = np.arange(4, dtype=np.float64) / 30.0
    np.savez_compressed(candidate, timestamps=source_time)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "contact_pointcloud_2finger",
                "contact_detector": "source_rgbd_pointcloud",
            }
        )
    )
    contact_json = tmp_path / "contact.json"
    contact_json.write_text(json.dumps({"contact_frames": [1, 2]}))

    session_dir = tmp_path / "session_test"
    calibration = session_dir / "calibrations/R_optical_to_link.npy"
    calibration.parent.mkdir(parents=True)
    np.save(calibration, np.eye(3, dtype=np.float32))
    (session_dir / "annotation.hdf5").write_bytes(b"annotation")
    destination = tmp_path / "published.npz"
    write_persistent_kinematics(
        source,
        destination,
        session=SessionProbe("session_test", session_dir, 4, source_time[-1]),
        artifact_manifest_hash="artifact",
        dependency_manifest={"sha256": "dependency"},
        reference_preparation={"method": "template_palm_kabsch"},
        orientation_offset_json=None,
        contact_aware_candidates=candidate,
        joint_contact_wrist_candidate=candidate,
        joint_contact_hand=hand,
        contact_manifest=manifest,
        contact_json=contact_json,
        q0_mink=Q0MinkOptions(),
        reference_smoothing=ReferenceSmoothingOptions(),
        reference_smoothing_meta=None,
    )

    with np.load(destination, allow_pickle=False) as archive:
        assert archive["joint_pos"].shape == (frames, 29)
        np.testing.assert_array_equal(archive["hand_qpos"], hand_qpos)
        np.testing.assert_array_equal(archive["contact_30hz"], [0, 1, 1, 0])
        assert archive["contact_50hz"].shape == (frames,)
        assert set(archive["contact_50hz"].tolist()) <= {0, 1}
        assert int(archive["contact_50hz"].sum()) > 0
        np.testing.assert_array_equal(archive["left_contact"], 0)
        np.testing.assert_array_equal(archive["right_contact"], archive["contact_50hz"])
        np.testing.assert_array_equal(
            archive["contact_50hz"],
            np.maximum(archive["left_contact"], archive["right_contact"]),
        )
        assert str(archive["joint_contact_bundle_sha256"]) == (
            joint_contact_bundle_sha256(candidate, hand, manifest)
        )

    manifest.write_text(
        json.dumps(
            {
                "status": "no_contact_detected",
                "contact_detector": "source_rgbd_pointcloud",
            }
        )
    )
    no_contact_destination = tmp_path / "published_no_contact.npz"
    write_persistent_kinematics(
        source,
        no_contact_destination,
        session=SessionProbe("session_test", session_dir, 4, source_time[-1]),
        artifact_manifest_hash="artifact",
        dependency_manifest={"sha256": "dependency"},
        reference_preparation={"method": "template_palm_kabsch"},
        orientation_offset_json=None,
        contact_aware_candidates=None,
        joint_contact_wrist_candidate=candidate,
        joint_contact_hand=hand,
        contact_manifest=manifest,
        contact_json=contact_json,
        q0_mink=Q0MinkOptions(),
        reference_smoothing=ReferenceSmoothingOptions(),
        reference_smoothing_meta=None,
    )
    with np.load(no_contact_destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["hand_qpos"], hand_qpos)
        np.testing.assert_array_equal(archive["contact_30hz"], 0)
        np.testing.assert_array_equal(archive["contact_50hz"], 0)
        np.testing.assert_array_equal(archive["left_contact"], 0)
        np.testing.assert_array_equal(archive["right_contact"], 0)
        np.testing.assert_array_equal(archive["left_contact"], 0)
        np.testing.assert_array_equal(archive["right_contact"], 0)


def test_resolve_precomputed_joint_contact_bundle(tmp_path: Path) -> None:
    session_id = "session_test"
    bundle_root = tmp_path / session_id / "contact_aware_auto"
    joint_root = bundle_root / "joint_contact"
    joint_root.mkdir(parents=True)
    np.savez_compressed(
        bundle_root / "contact_aware_wrist_candidates_native.npz",
        timestamps=np.arange(3, dtype=np.float64) / 30.0,
    )
    for path in (
        joint_root / "base_hand_dex3_50hz.npz",
        bundle_root / "hand_dex3_50hz_joint_contact.npz",
    ):
        np.savez_compressed(path, qpos=np.zeros((5, 14), dtype=np.float32))
    (bundle_root / "contact_frames.json").write_text(
        json.dumps({"contact_frames": [1]})
    )
    manifest = bundle_root / "contact_precompute_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "contact_pointcloud_2finger",
                "joint_contact_optimization": {"status": "enabled"},
            }
        )
    )

    contact = resolve_joint_contact_bundle(tmp_path, session_id)
    assert contact is not None
    assert contact.inject_wrist_residual
    assert contact.hand_kinematics.name == "hand_dex3_50hz_joint_contact.npz"

    manifest.write_text(
        json.dumps(
            {
                "status": "no_contact_detected",
                "joint_contact_optimization": {"status": "skipped_no_contact"},
            }
        )
    )
    no_contact = resolve_joint_contact_bundle(tmp_path, session_id)
    assert no_contact is not None
    assert not no_contact.inject_wrist_residual
    assert no_contact.hand_kinematics.name == "base_hand_dex3_50hz.npz"


def test_publish_brainco_palm_only_zeros_contact(tmp_path: Path) -> None:
    frames = 6
    body_time = np.arange(frames, dtype=np.float64) / 50.0
    source = tmp_path / "body.npz"
    np.savez_compressed(
        source,
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        timestamps=body_time,
        fps=np.asarray(50.0, dtype=np.float32),
    )
    hand = tmp_path / "hand.npz"
    np.savez_compressed(
        hand,
        qpos=np.arange(frames * 12, dtype=np.float32).reshape(frames, 12),
        qvel=np.zeros((frames, 12), dtype=np.float32),
        hand_joint_names=np.asarray([f"hand_{index}" for index in range(12)]),
        timestamps=body_time,
        left_contact=np.zeros(frames, dtype=np.int8),
        right_contact=np.zeros(frames, dtype=np.int8),
        left_contact_30hz=np.zeros(frames, dtype=np.int8),
        right_contact_30hz=np.zeros(frames, dtype=np.int8),
        native_timestamps=body_time,
    )
    brainco_reference = tmp_path / "brainco_reference.npz"
    np.savez_compressed(brainco_reference, placeholder=np.asarray(1))
    brainco_meta = tmp_path / "brainco_meta.json"
    brainco_meta.write_text(json.dumps({"method": "g1_brainco_palm_refine"}))

    session_dir = tmp_path / "session_test"
    calibration = session_dir / "calibrations/R_optical_to_link.npy"
    calibration.parent.mkdir(parents=True)
    np.save(calibration, np.eye(3, dtype=np.float32))
    (session_dir / "annotation.hdf5").write_bytes(b"annotation")
    destination = tmp_path / "published.npz"
    write_persistent_kinematics(
        source,
        destination,
        session=SessionProbe("session_test", session_dir, frames, body_time[-1]),
        artifact_manifest_hash="artifact",
        dependency_manifest={"sha256": "dependency"},
        reference_preparation={"method": "g1_brainco_palm_refine"},
        orientation_offset_json=None,
        contact_aware_candidates=None,
        joint_contact_wrist_candidate=None,
        joint_contact_hand=hand,
        contact_manifest=None,
        contact_json=None,
        q0_mink=Q0MinkOptions(),
        reference_smoothing=ReferenceSmoothingOptions(),
        reference_smoothing_meta=None,
        hand_mode=HAND_MODE_BRAINCO,
        brainco_reference=brainco_reference,
        brainco_meta=brainco_meta,
    )
    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["left_contact"], 0)
        np.testing.assert_array_equal(archive["right_contact"], 0)
        np.testing.assert_array_equal(archive["contact_50hz"], 0)
        assert archive["hand_qpos"].shape == (frames, 12)


def test_publish_brainco_bundle_left_right_contact(tmp_path: Path) -> None:
    frames = 5
    body_time = np.arange(frames, dtype=np.float64) / 50.0
    source = tmp_path / "body.npz"
    np.savez_compressed(
        source,
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        timestamps=body_time,
        fps=np.asarray(50.0, dtype=np.float32),
    )
    hand = tmp_path / "hand.npz"
    np.savez_compressed(
        hand,
        qpos=np.arange(frames * 12, dtype=np.float32).reshape(frames, 12),
        qvel=np.zeros((frames, 12), dtype=np.float32),
        hand_joint_names=np.asarray([f"hand_{index}" for index in range(12)]),
        timestamps=body_time,
    )
    source_time = np.arange(4, dtype=np.float64) / 30.0
    left_candidate = tmp_path / "left.npz"
    right_candidate = tmp_path / "right.npz"
    for path in (left_candidate, right_candidate):
        np.savez_compressed(path, timestamps=source_time)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "hand_mode": HAND_MODE_BRAINCO,
                "status": "contact_pointcloud_2finger",
                "contact_detector": "source_rgbd_pointcloud",
            }
        )
    )
    contact_json = tmp_path / "contact.json"
    contact_json.write_text(
        json.dumps(
            {
                "contact_frames": [1, 2],
                "left_contact_frames": [1],
                "right_contact_frames": [2],
            }
        )
    )
    brainco_reference = tmp_path / "brainco_reference.npz"
    np.savez_compressed(brainco_reference, placeholder=np.asarray(1))
    brainco_meta = tmp_path / "brainco_meta.json"
    brainco_meta.write_text(json.dumps({"method": "g1_brainco_palm_refine"}))

    session_dir = tmp_path / "session_test"
    calibration = session_dir / "calibrations/R_optical_to_link.npy"
    calibration.parent.mkdir(parents=True)
    np.save(calibration, np.eye(3, dtype=np.float32))
    (session_dir / "annotation.hdf5").write_bytes(b"annotation")
    destination = tmp_path / "published.npz"
    write_persistent_kinematics(
        source,
        destination,
        session=SessionProbe("session_test", session_dir, 4, source_time[-1]),
        artifact_manifest_hash="artifact",
        dependency_manifest={"sha256": "dependency"},
        reference_preparation={"method": "g1_brainco_palm_refine+neighborhood_laplacian"},
        orientation_offset_json=None,
        contact_aware_candidates=right_candidate,
        joint_contact_wrist_candidate=right_candidate,
        joint_contact_hand=hand,
        contact_manifest=manifest,
        contact_json=contact_json,
        q0_mink=Q0MinkOptions(),
        reference_smoothing=ReferenceSmoothingOptions(),
        reference_smoothing_meta=None,
        hand_mode=HAND_MODE_BRAINCO,
        brainco_reference=brainco_reference,
        brainco_meta=brainco_meta,
    )
    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["left_contact_30hz"], [0, 1, 0, 0])
        np.testing.assert_array_equal(archive["right_contact_30hz"], [0, 0, 1, 0])
        np.testing.assert_array_equal(
            archive["contact_30hz"],
            np.maximum(archive["left_contact_30hz"], archive["right_contact_30hz"]),
        )
        np.testing.assert_array_equal(
            archive["contact_50hz"],
            np.maximum(archive["left_contact"], archive["right_contact"]),
        )
        assert int(archive["left_contact"].sum()) > 0
        assert int(archive["right_contact"].sum()) > 0
        assert archive["hand_qpos"].shape == (frames, 12)
        assert str(archive["joint_contact_bundle_sha256"]) == (
            joint_contact_bundle_sha256(right_candidate, hand, manifest)
        )
        assert str(archive["joint_contact_contract"]) == (
            "dual_wrist_yaw_link_SE3_plus_g1_brainco_6DoF_per_hand"
        )


def test_resolve_brainco_joint_contact_bundle(tmp_path: Path) -> None:
    session_id = "session_test"
    bundle_root = tmp_path / session_id / "contact_aware_auto"
    joint_root = bundle_root / "joint_contact"
    joint_root.mkdir(parents=True)
    timestamps = np.arange(3, dtype=np.float64) / 30.0
    np.savez_compressed(
        bundle_root / "contact_aware_wrist_candidates_native_left.npz",
        timestamps=timestamps,
    )
    np.savez_compressed(
        bundle_root / "contact_aware_wrist_candidates_native_right.npz",
        timestamps=timestamps,
    )
    np.savez_compressed(
        joint_root / "base_hand_brainco_50hz.npz",
        qpos=np.zeros((5, 12), dtype=np.float32),
    )
    np.savez_compressed(
        bundle_root / "hand_brainco_50hz_joint_contact.npz",
        qpos=np.ones((5, 12), dtype=np.float32),
    )
    (bundle_root / "contact_frames.json").write_text(
        json.dumps(
            {
                "contact_frames": [1],
                "left_contact_frames": [1],
                "right_contact_frames": [],
            }
        )
    )
    manifest = bundle_root / "contact_precompute_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "hand_mode": HAND_MODE_BRAINCO,
                "status": "contact_pointcloud_2finger",
                "joint_contact_optimization": {"status": "enabled"},
            }
        )
    )

    contact = resolve_joint_contact_bundle(
        tmp_path, session_id, hand_mode=HAND_MODE_BRAINCO
    )
    assert contact is not None
    assert contact.inject_wrist_residual
    assert contact.inject_sides == ("left",)
    assert contact.hand_kinematics.name == "hand_brainco_50hz_joint_contact.npz"
    assert contact.left_wrist_candidate is not None
    assert contact.right_wrist_candidate is not None

    manifest.write_text(
        json.dumps(
            {
                "hand_mode": HAND_MODE_BRAINCO,
                "status": "no_contact_detected",
                "joint_contact_optimization": {"status": "skipped_no_contact"},
            }
        )
    )
    (bundle_root / "contact_frames.json").write_text(
        json.dumps(
            {
                "contact_frames": [],
                "left_contact_frames": [],
                "right_contact_frames": [],
            }
        )
    )
    no_contact = resolve_joint_contact_bundle(
        tmp_path, session_id, hand_mode=HAND_MODE_BRAINCO
    )
    assert no_contact is not None
    assert not no_contact.inject_wrist_residual
    assert no_contact.hand_kinematics.name == "base_hand_brainco_50hz.npz"
    assert no_contact.inject_sides == ()






