#!/usr/bin/env python3
"""Render full G1+Dex3 post-IK keyframes with the observed cup model.

The cup is rendered as a diagnostic cylinder at the RGB-D fitted world pose.
Robot RGB, metric depth, robot IDs, all object IDs, and a comparison montage
are emitted for a small list of native source frames. Production rendering is
not modified.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from evaluate_hand_object_interaction import CupModel
from mujoco_robot_id_capture import robot_geom_ids, robot_id_from_segmentation
from render_torso_robot_rgb_depth import (
    CAMERA_BODY_NAME,
    CAMERA_MOUNT_POS,
    CAMERA_MOUNT_QUAT,
    CAMERA_NAME,
    CAMERA_OPTICAL_QUAT,
    CAMERA_PARENT,
    DEFAULT_DEX3_XML,
    DEFAULT_MESHDIR,
    HEADLIGHT_AMBIENT,
    HEADLIGHT_DIFFUSE,
    HEADLIGHT_SPECULAR,
    _vec,
)


CUP_BODY_NAME = "diagnostic_observed_cup"
CUP_GEOM_NAME = "diagnostic_observed_cup_geom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik-kinematics", type=Path, required=True)
    parser.add_argument("--hand-kinematics", type=Path, required=True)
    parser.add_argument("--post-ik-metrics", type=Path, required=True)
    parser.add_argument("--object-models", type=Path, required=True)
    parser.add_argument("--rgb-k-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-frames", default="331,341,360")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_DEX3_XML)
    parser.add_argument("--meshdir", type=Path, default=DEFAULT_MESHDIR)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_model(
    xml_path: Path,
    meshdir: Path,
    K: np.ndarray,
    width: int,
    height: int,
) -> mujoco.MjModel:
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(meshdir))
    compiler.set("boundmass", "1e-6")
    compiler.set("boundinertia", "1e-8")
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.append(visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(width))
    global_visual.set("offheight", str(height))

    torso = root.findall(f".//body[@name='{CAMERA_PARENT}']")
    if len(torso) != 1:
        raise ValueError(f"expected one {CAMERA_PARENT}, got {len(torso)}")
    mount = ET.SubElement(
        torso[0],
        "body",
        name=CAMERA_BODY_NAME,
        pos=_vec(CAMERA_MOUNT_POS),
        quat=_vec(CAMERA_MOUNT_QUAT),
    )
    ET.SubElement(
        mount,
        "camera",
        name=CAMERA_NAME,
        pos="0 0 0",
        quat=_vec(CAMERA_OPTICAL_QUAT),
        resolution=f"{width} {height}",
        focalpixel=_vec((float(K[0, 0]), float(K[1, 1]))),
        principalpixel=_vec(
            (float(K[0, 2] - width / 2.0), float(K[1, 2] - height / 2.0))
        ),
        sensorsize="1 1",
    )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    cup = ET.SubElement(
        worldbody,
        "body",
        name=CUP_BODY_NAME,
        mocap="true",
        pos="0 0 0",
    )
    ET.SubElement(
        cup,
        "geom",
        name=CUP_GEOM_NAME,
        type="cylinder",
        size="0.035 0.055",
        rgba="0.10 0.52 0.90 0.72",
        contype="0",
        conaffinity="0",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = HEADLIGHT_SPECULAR
    return model


def all_object_ids(segmentation: np.ndarray) -> np.ndarray:
    seg = np.asarray(segmentation)
    geom = np.asarray(seg[..., 0], dtype=np.int64)
    object_type = np.asarray(seg[..., 1], dtype=np.int64)
    valid = object_type == int(mujoco.mjtObj.mjOBJ_GEOM)
    output = np.zeros(geom.shape, dtype=np.uint16)
    output[valid] = (geom[valid] + 1).astype(np.uint16)
    return output


def write_png(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values).save(path)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(values[valid], [2.0, 98.0])
        scale = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        normalized[valid] = np.asarray(255.0 * (1.0 - scale[valid]), dtype=np.uint8)
    bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def colorize_ids(ids: np.ndarray) -> np.ndarray:
    values = np.asarray(ids, dtype=np.uint16)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    nonzero = values > 0
    output[..., 0][nonzero] = (53 * values[nonzero] % 251).astype(np.uint8)
    output[..., 1][nonzero] = (97 * values[nonzero] % 241).astype(np.uint8)
    output[..., 2][nonzero] = (193 * values[nonzero] % 239).astype(np.uint8)
    return output


def main() -> None:
    args = parse_args()
    paths = {
        "ik_kinematics": args.ik_kinematics.expanduser().resolve(),
        "hand_kinematics": args.hand_kinematics.expanduser().resolve(),
        "post_ik_metrics": args.post_ik_metrics.expanduser().resolve(),
        "object_models": args.object_models.expanduser().resolve(),
        "rgb_k_npy": args.rgb_k_npy.expanduser().resolve(),
        "robot_xml": args.robot_xml.expanduser().resolve(),
        "meshdir": args.meshdir.expanduser().resolve(),
    }
    for name, path in paths.items():
        if name == "meshdir":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    selected_frames = [int(value) for value in args.source_frames.split(",") if value.strip()]
    metrics = {int(record["frame"]): record for record in load_jsonl(paths["post_ik_metrics"])}
    cup_records = {int(record["frame"]): record for record in load_jsonl(paths["object_models"])}
    missing = [frame for frame in selected_frames if frame not in metrics or frame not in cup_records]
    if missing:
        raise KeyError(f"selected frames missing metrics or cup models: {missing}")

    ik = load_npz(paths["ik_kinematics"])
    hand = load_npz(paths["hand_kinematics"])
    K = np.asarray(np.load(paths["rgb_k_npy"]), dtype=np.float64).copy()
    K[0] *= float(args.width) / float(args.source_width)
    K[1] *= float(args.height) / float(args.source_height)
    model = build_model(paths["robot_xml"], paths["meshdir"], K, args.width, args.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    body_names = [str(value) for value in ik["body_names"].tolist()]
    joint_names = [str(value) for value in ik["joint_names"].tolist()]
    hand_names = [str(value) for value in hand["joint_names"].tolist()]
    all_names = joint_names + hand_names
    joint_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in all_names
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise KeyError("body or Dex3 joint missing from complete visual XML")
    free_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_ids) != 1:
        raise ValueError(f"expected one free joint, got {free_ids}")
    free_address = int(model.jnt_qposadr[free_ids[0]])
    pelvis_index = body_names.index("pelvis")
    cup_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUP_BODY_NAME))
    cup_mocap = int(model.body_mocapid[cup_body])
    cup_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CUP_GEOM_NAME))
    robot_geoms, _, _ = robot_geom_ids(model, root_body_hint="pelvis")

    frame_outputs = []
    try:
        for source_frame in selected_frames:
            metric = metrics[source_frame]
            ik_frame = int(metric["nearest_ik_frame"])
            cup_record = cup_records[source_frame]
            cup = CupModel(
                **{
                    name: cup_record[name]
                    for name in CupModel.__dataclass_fields__
                }
            )
            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.0
            data.qpos[free_address : free_address + 3] = ik["body_pos_w"][ik_frame, pelvis_index]
            root_quaternion = np.asarray(ik["body_quat_w"][ik_frame, pelvis_index], dtype=np.float64)
            data.qpos[free_address + 3 : free_address + 7] = root_quaternion / np.linalg.norm(root_quaternion)
            body_qpos = np.asarray(ik["joint_pos"][ik_frame], dtype=np.float64)
            hand_qpos = np.asarray(hand["qpos"][ik_frame], dtype=np.float64)
            for joint_id, value in zip(joint_ids, np.concatenate([body_qpos, hand_qpos])):
                data.qpos[int(model.jnt_qposadr[joint_id])] = value

            z_mid = 0.5 * (cup.z_min + cup.z_max)
            radius = float(cup.radius_at(z_mid))
            half_height = 0.5 * (cup.z_max - cup.z_min)
            model.geom_size[cup_geom, :2] = [radius, half_height]
            data.mocap_pos[cup_mocap] = [cup.center_xy[0], cup.center_xy[1], z_mid]
            data.mocap_quat[cup_mocap] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=CAMERA_NAME)
            rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
            renderer.enable_depth_rendering()
            try:
                renderer.update_scene(data, camera=CAMERA_NAME)
                depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            finally:
                renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            try:
                renderer.update_scene(data, camera=CAMERA_NAME)
                segmentation = np.asarray(renderer.render()).copy()
            finally:
                renderer.disable_segmentation_rendering()
            robot_id = robot_id_from_segmentation(segmentation, robot_geoms)
            object_id = all_object_ids(segmentation)
            stem = f"source_{source_frame:06d}_ik_{ik_frame:06d}"
            write_png(output / f"{stem}_rgb.png", rgb)
            np.save(output / f"{stem}_depth_m.npy", depth, allow_pickle=False)
            depth_rgb = colorize_depth(depth)
            write_png(output / f"{stem}_depth.png", depth_rgb)
            write_png(output / f"{stem}_robot_id.png", robot_id)
            write_png(output / f"{stem}_object_id.png", object_id)
            id_rgb = colorize_ids(object_id)
            montage = np.concatenate([rgb, depth_rgb, id_rgb], axis=1)
            write_png(output / f"{stem}_montage.png", montage)
            frame_outputs.append(
                {
                    "source_frame": source_frame,
                    "ik_frame": ik_frame,
                    "valid_grasp": bool(metric["valid_grasp"]),
                    "bilateral_contact": bool(metric["bilateral_contact"]),
                    "penetration_ratio": float(metric["hand_penetration_ratio"]),
                    "cup_geom_id": cup_geom,
                    "cup_object_id": cup_geom + 1,
                    "rgb": f"{stem}_rgb.png",
                    "depth_m": f"{stem}_depth_m.npy",
                    "robot_id": f"{stem}_robot_id.png",
                    "object_id": f"{stem}_object_id.png",
                    "montage": f"{stem}_montage.png",
                }
            )
    finally:
        renderer.close()

    manifest = {
        "schema_version": 1,
        "experiment_only": True,
        "production_pipeline_modified": False,
        **{name: str(path) for name, path in paths.items()},
        "image_size_hw": [args.height, args.width],
        "intrinsics": K.tolist(),
        "camera": CAMERA_NAME,
        "cup_visualization": (
            "RGB-D fitted upright tapered cup approximated by one diagnostic cylinder; "
            "contact metrics continue to use the tapered analytic model"
        ),
        "frames": frame_outputs,
    }
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
