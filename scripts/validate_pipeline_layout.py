#!/usr/bin/env python3
"""Validate the consolidated pipeline layout and retargeting contracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    required = [
        PROJECT_ROOT / "pipeline_smooth_motion.txt",
        PROJECT_ROOT / "pipeline_initandtracking_3point_batch.txt",
        PROJECT_ROOT / "run_batch_singleframe_view_change_30to50.sh",
        PROJECT_ROOT / "scripts/smooth_wrist_reference.py",
        PROJECT_ROOT / "scripts/smooth_three_point_reference.py",
        PROJECT_ROOT / "scripts/run_three_point_batch.py",
        PROJECT_ROOT / "scripts/view_change/extract_hand_archive.py",
        PROJECT_ROOT / "scripts/view_change/verify_tracking_render_kinematics.py",
        PROJECT_ROOT / "scripts/view_change/verify_target_camera_contract.py",
        PROJECT_ROOT / "scripts/view_change/configs/target_camera_672x376.json",
        PROJECT_ROOT / "scripts/view_change/retarget_template_palm_reference.py",
        PROJECT_ROOT / "scripts/view_change/run_singleframe_view_change_30to50.sh",
        PROJECT_ROOT / "scripts/view_change/run_contact_aware_6d_reference_tracking.sh",
        PROJECT_ROOT / "scripts/view_change/prepare_contact_template_palm_reference.sh",
        PROJECT_ROOT / "scripts/view_change/run_contact_aware_batch_view_change.sh",
        PROJECT_ROOT / "scripts/view_change/resident_gpu_workers.sh",
        PROJECT_ROOT / "src/koda/vendor/diffueraser/diffueraser_worker.py",
        PROJECT_ROOT / "src/koda/vendor/diffueraser/run_diffueraser.py",
        PROJECT_ROOT / "assets/manifest.yaml",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "scripts/view_change/brainco_hand_retarget.py",
        PROJECT_ROOT / "scripts/view_change/evaluate_hand_object_interaction.py",
        PROJECT_ROOT / "scripts/view_change/optimize_contact_aware_wrist_and_fingers.py",
        PROJECT_ROOT / "scripts/view_change/run_contact_precompute.sh",
        PROJECT_ROOT / "scripts/view_change/evaluate_hand_object_interaction.py",
        PROJECT_ROOT / "scripts/view_change/optimize_contact_aware_wrist_and_fingers.py",
        PROJECT_ROOT / "scripts/view_change/run_contact_precompute.sh",
        PROJECT_ROOT
        / "resources/brainco/robots/g1_brainco/g1_mocap_29dof_with_brainco_hands.xml",
        PROJECT_ROOT / "resources/brainco/robots/g1_brainco/meshes",
        PROJECT_ROOT
        / "resources/brainco/robots/hands/brainco_hand/brainco_hand_left.urdf",
        PROJECT_ROOT
        / "resources/brainco/robots/hands/brainco_hand/brainco_hand_right.urdf",
        PROJECT_ROOT / "resources/brainco/configs/retarget/brainco_hand_left.yml",
        PROJECT_ROOT / "resources/brainco/configs/retarget/brainco_hand_right.yml",
        PROJECT_ROOT / "resources/ScaleBFM",
        PROJECT_ROOT / "resources/hydrax",
        PROJECT_ROOT / "src/koda/retargeting/dex3.py",
        PROJECT_ROOT / "src/koda/retargeting/evaluation.py",
        PROJECT_ROOT / "src/koda/retargeting/configs/unitree_dex3_position.yml",
        PROJECT_ROOT / "src/koda/kinematics/compose.py",
        PROJECT_ROOT / "src/koda/kinematics/dex3.py",
        PROJECT_ROOT / "src/koda/ik/mink_wrist.py",
        PROJECT_ROOT / "src/koda/ik/wrist.py",
        PROJECT_ROOT / "src/koda/rendering/segmentation.py",
        PROJECT_ROOT / "resources/robots/g1_dex3/g1_with_visual_dex3_hands.xml",
        PROJECT_ROOT / "resources/robots/g1_dex3/g1_29dof_with_hand_st_collision.xml",
        PROJECT_ROOT / "resources/DiffuEraser",
    ]
    for path in required:
        require(path)

    driver = (PROJECT_ROOT / "scripts/run_three_point_batch.py").read_text()
    tracking_entry = (
        PROJECT_ROOT / "pipeline_initandtracking_3point_batch.txt"
    ).read_text()
    batch = (
        PROJECT_ROOT / "run_batch_singleframe_view_change_30to50.sh"
    ).read_text()
    single_view = (
        PROJECT_ROOT / "scripts/view_change/run_singleframe_view_change_30to50.sh"
    ).read_text()
    contact_tracking = (
        PROJECT_ROOT
        / "scripts/view_change/run_contact_aware_6d_reference_tracking.sh"
    ).read_text()
    contact_batch = (
        PROJECT_ROOT
        / "scripts/view_change/run_contact_aware_batch_view_change.sh"
    ).read_text()
    adapter = (
        PROJECT_ROOT
        / "scripts/view_change/retarget_template_palm_reference.py"
    ).read_text()
    brainco_retarget = (
        PROJECT_ROOT / "scripts/view_change/brainco_hand_retarget.py"
    ).read_text()
    contact_precompute = (
        PROJECT_ROOT / "scripts/view_change/run_contact_precompute.sh"
    ).read_text()
    contact_pointcloud = (
        PROJECT_ROOT / "scripts/view_change/contact_pointcloud_analysis.py"
    ).read_text()
    contact_precompute = (
        PROJECT_ROOT / "scripts/view_change/run_contact_precompute.sh"
    ).read_text()
    contact_pointcloud = (
        PROJECT_ROOT / "scripts/view_change/contact_pointcloud_analysis.py"
    ).read_text()
    brainco_mjcf = (
        PROJECT_ROOT
        / "resources/brainco/robots/g1_brainco/g1_mocap_29dof_with_brainco_hands.xml"
    ).read_text()
    compose_render = (
        PROJECT_ROOT / "src/koda/kinematics/compose.py"
    ).read_text()
    render_utils = (
        PROJECT_ROOT / "src/koda/kinematics/dex3.py"
    ).read_text()
    extract_hand = (
        PROJECT_ROOT / "scripts/view_change/extract_hand_archive.py"
    ).read_text()
    verify_render = (
        PROJECT_ROOT / "scripts/view_change/verify_tracking_render_kinematics.py"
    ).read_text()
    checks = {
        "wrist_reference_stage_present": (
            "retarget_template_palm_reference" in driver
            and "template_palm_kabsch" in driver
        ),
        "wrist_reference_uses_five_palm_anchors": (
            "MCP_INDICES = (5, 9, 13, 17)" in adapter
            and "ANCHOR_INDICES = (WRIST, *MCP_INDICES)" in adapter
        ),
        "three_point_reference_savgol_present": (
            "SMOOTH_REFERENCE" in driver
            and "smooth_three_point_reference" in driver
            and "reference = smoothed_reference" in driver
            and "targets = smoothed_targets" in driver
        ),
        "three_point_reference_savgol_default_enabled": (
            "enabled=True" in driver
            and "--no-reference-savgol" not in driver
            and "--reference-savgol\"" not in driver
            and "REFERENCE_SAVGOL_WINDOW=${REFERENCE_SAVGOL_WINDOW:-5}" in tracking_entry
            and "REFERENCE_SAVGOL_POLYORDER=${REFERENCE_SAVGOL_POLYORDER:-2}"
            in tracking_entry
        ),
        "formal_batch_uses_latest_tracking_contract": (
            "--joint-contact-root" in driver
            and 'TRACK_ARGS+=("--joint-contact-root"' in contact_tracking
            and "reference_smoothing_config_json" in contact_tracking
            and "hand_qpos" in contact_tracking
            and "contact_50hz" in contact_tracking
            and 'cp --reflink=auto "${TRACKING_KINEMATICS}" "${FINAL_KINEMATICS}"'
            in contact_tracking
            and "HAND_KINEMATICS_OVERRIDE" not in contact_tracking
            and "verify_final_contract" in contact_batch
            and "joint_contact_bundle_sha256" in contact_batch
            and "run_worker" in contact_batch
            and "QUEUE_INDEX_FILE" in contact_batch
            and "flock -x" in contact_batch
            and 'GPU="${gpu}"' in contact_batch
            and "prepare_contact_template_palm_reference.sh" in contact_batch
            and "KINEMATICS_ROOT=\"${reference_root}\"" in contact_batch
            and "raw_annotation_template_palm" in contact_batch
            and "prepare_contact_template_palm_reference.sh" in contact_tracking
            and "no tracking bootstrap" in contact_tracking
        ),
        "view_change_uses_tracking_hand_contract": (
            "hand_qpos" in batch
            and "hand_qvel" in batch
            and "hand_joint_names" in batch
            and "tracking_kinematics_29dof_body_plus_14dof_hand" in batch
            and "extract_hand_archive.py" in single_view
            and "verify_tracking_render_kinematics.py" in single_view
        ),
        "view_change_uses_fixed_target_camera_contract": (
            "TARGET_CAMERA_CALIBRATION" in batch
            and "target_camera_672x376.json" in batch
            and "SOURCE_RGB_K" in batch
            and "TARGET_CAMERA_CALIBRATION" in single_view
            and "--camera-calibration-json" in single_view
            and "verify_target_camera_contract.py" in single_view
            and "--rgb-k-npy \"${SOURCE_RGB_K}\"" not in single_view
        ),
        "view_change_has_no_hand_retarget_or_wrist_offset": (
            "RETARGET_TYPE" not in batch
            and "LEFT_WRIST_YAW_OFFSET_DEG" not in batch
            and "RIGHT_WRIST_YAW_OFFSET_DEG" not in batch
            and "koda.retargeting.dex3" not in single_view
            and "offset_kinematics_joint.py" not in single_view
        ),
        "no_external_pipeline_code_entry": (
            "egodex_text/_pipeline_scripts" not in batch
            and "ego_anywhere/run_three_point_batch.py" not in driver
        ),
        # BrainCo is an opt-in branch: every entry point still defaults to the
        # Dex3 contract, and the Dex3 widths stay literal wherever they used to
        # be hard-coded.
        "brainco_hand_mode_is_opt_in": (
            '"--hand-mode"' in driver
            and 'default=HAND_MODE_DEX3' in driver
            and 'HAND_MODE="${HAND_MODE:-dex3}"' in contact_batch
            and 'HAND_MODE="${HAND_MODE:-dex3}"' in contact_tracking
            and 'HAND_MODE="${HAND_MODE:-dex3}"' in single_view
            and '"dex3": {"nq": 50, "nv": 49, "hinges": 43, "bodies": 45}'
            in render_utils
            and "g1_with_visual_dex3_hands.xml" in single_view
            and 'default=EXPECTED_HAND_JOINTS' in compose_render
            and "EXPECTED_HAND_JOINTS = 14" in compose_render
            and '"--hand-dof",\n        type=int,\n        default=14,' in extract_hand
            and '"--render-dof",\n        type=int,\n        default=43,'
            in verify_render
        ),
        # BrainCo remains opt-in. Contact-on uses neighborhood Laplacian
        # wrist+finger refine; Dex3 still uses CupModel Powell. Neither
        # orchestrator forces the contact flag off.
        "brainco_optional_contact_anchors": (
            "BraincoJointRefiner" in brainco_retarget
            and "estimate_palm_local_correction" in brainco_retarget
            and "PALM_SIGN_ALIGN_MIN_ABS_DOT" in brainco_retarget
            and "_raw_palm_sign_confidence" in brainco_retarget
            and "brainco_reference" in driver
            and "HAND_DOF_BY_MODE" in driver
            and "optimize_brainco_contact_anchors" not in driver
            and "optimize_brainco_contact_anchors.py" in contact_precompute
            and "evaluate_hand_object_interaction.py" in contact_precompute
            and "optimize_contact_aware_wrist_and_fingers.py" in contact_precompute
            and "g1_brainco_palm_refine+neighborhood_laplacian" in contact_precompute
            and 'OBJECT_SOURCE="${OBJECT_SOURCE:-fingertip_neighborhood}"'
            in contact_precompute
            and 'CUP_PROMPTS="${CUP_PROMPTS:-a blue cup|a blue mug|a cup|a mug}"'
            in contact_precompute
            and "--object-source" in contact_pointcloud
            and "forcing JOINT_CONTACT_ENABLED=0" not in contact_batch
            and "forcing JOINT_CONTACT_ENABLED=0" not in contact_tracking
            and 'TRACK_ARGS+=("--hand-mode" "g1_brainco")' in contact_tracking
            and '"--contact-edge-hold"' in contact_tracking
            and "g1_brainco_has_no_contact_stage" not in contact_tracking
        ),
        # MuJoCo resolves mjEQ_JOINT in the solver, so the offline render path
        # has to expand the BrainCo distal joints from the XML's own polynomial.
        "brainco_render_expands_mimic_joints": (
            "<equality>" in brainco_mjcf
            and 'joint1="right_index_distal_joint"' in brainco_mjcf
            and "joint_mimic_rules" in compose_render
            and "evaluate_mimic_rule" in compose_render
            and "joint_mimic_rules" in render_utils
            and "mimic_joint_names" in verify_render
            and 'RENDER_HAND_DOF=22' in single_view
        ),
        # Wrist-yaw must keep the vendor base2 adapter and must not keep the
        # Dex3/Inspire palm leftover from the template MJCF.
        "brainco_mjcf_matches_urdf_hand_mount": (
            'name="left_base2_link"' in brainco_mjcf
            and 'name="right_base2_link"' in brainco_mjcf
            and "hand_palm_link" not in brainco_mjcf
            and "rubber_hand" not in brainco_mjcf
            and 'name="hand_metal"' in brainco_mjcf
            and 'class="hand_visual"' in brainco_mjcf
        ),
        "rerun_view_change_reuses_published_kinematics": (
            'RERUN_VIEW_CHANGE="${RERUN_VIEW_CHANGE:-0}"' in contact_batch
            and "reusable_tracking_root" in contact_batch
            and "skipped_missing_kinematics" in contact_batch
            and "FORCE=1 and RERUN_VIEW_CHANGE=1 cannot be set together"
            in contact_batch
            and 'VIEW_FORCE="${VIEW_FORCE:-${FORCE}}"' in contact_tracking
            and 'FORCE="${VIEW_FORCE}"' in contact_tracking
            and "SKIP_TRACKING=1 requires a reusable tracking cache" in contact_tracking
            and 'SKIP_TRACKING="$([ "${RERUN_VIEW_CHANGE}" == "1" ] && echo 1 || echo 0)"'
            in contact_batch
        ),
    }
    report = {
        "project_root": str(PROJECT_ROOT),
        "required_paths": [str(path) for path in required],
        "checks": checks,
        "passed": all(checks.values()),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
