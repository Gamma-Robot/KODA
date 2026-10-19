#!/usr/bin/env python3
"""Render Dex3 robot RGB, metric depth, geom IDs and camera poses.

The input is the composed 43-DoF archive produced from the new 3-point torso
tracking and DexPilot hand retarget.  The renderer deliberately loads the
visual Dex3 MJCF and assigns every hinge by name, so the rendered RGB/depth/ID
streams contain the retargeted hand geometry as well as the torso body.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from mujoco_robot_id_capture import robot_geom_ids, robot_id_from_segmentation

try:
    from frame_volume import create_volume
except ImportError:
    from .frame_volume import create_volume

try:
    from frame_volume import create_volume
except ImportError:
    from .frame_volume import create_volume


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
DEFAULT_DEX3_XML = (
    RESOURCE_ROOT / "SparseTrack/assets/g1_with_visual_dex3_hands.xml"
)
DEFAULT_MESHDIR = (
    RESOURCE_ROOT
    / "ScaleBFM/scalebridge/data/robot/g1_29dof/meshes"
)
CAMERA_BODY_NAME = "d435_link"
CAMERA_NAME = "front_cam"
CAMERA_PARENT = "torso_link"
CAMERA_MOUNT_POS = (0.0576235, 0.01753, 0.41987)
CAMERA_MOUNT_QUAT = (0.91496, 0.0, 0.403545, 0.0)
# D435 link -> OpenGL optical camera frame used by MuJoCo rendering.
CAMERA_OPTICAL_QUAT = (0.5, 0.5, -0.5, -0.5)
HEADLIGHT_AMBIENT = (0.30, 0.30, 0.30)
HEADLIGHT_DIFFUSE = (0.45, 0.45, 0.45)
HEADLIGHT_SPECULAR = (0.12, 0.12, 0.12)
SCENE_LIGHT_AMBIENT = (0.20, 0.20, 0.20)
SCENE_LIGHT_DIFFUSE = (0.55, 0.55, 0.55)
SCENE_LIGHT_SPECULAR = (0.10, 0.10, 0.10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _load_K(path: Path) -> np.ndarray:
    K = np.asarray(np.load(path), dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError(f"invalid RGB intrinsic matrix: {path} {K.shape}")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError("RGB focal lengths must be positive")
    return K


def _load_camera_calibration(path: Path) -> tuple[np.ndarray, dict]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "K" in metadata:
        K = np.asarray(metadata["K"], dtype=np.float64)
    else:
        K = np.array(
            [
                [float(metadata["fx"]), 0.0, float(metadata["cx"])],
                [0.0, float(metadata["fy"]), float(metadata["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError(f"invalid target camera intrinsic matrix: {path} {K.shape}")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError("target camera focal lengths must be positive")
    return K, metadata


def _inject_ground_plane(root: ET.Element) -> None:
    """Add a checker floor + skybox so third-person renders are not void-black."""
    if root.findall(".//geom[@name='floor']"):
        return
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if not root.findall(".//texture[@name='skybox']"):
        ET.SubElement(
            asset,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1="0.35 0.50 0.65",
            rgb2="0.05 0.08 0.12",
            width="512",
            height="3072",
        )
    if not root.findall(".//texture[@name='groundplane']"):
        ET.SubElement(
            asset,
            "texture",
            type="2d",
            name="groundplane",
            builtin="checker",
            mark="edge",
            rgb1="0.25 0.32 0.38",
            rgb2="0.55 0.58 0.60",
            markrgb="0.80 0.80 0.80",
            width="512",
            height="512",
        )
    if not root.findall(".//material[@name='groundplane']"):
        ET.SubElement(
            asset,
            "material",
            name="groundplane",
            texture="groundplane",
            texuniform="true",
            texrepeat="8 8",
            reflectance="0.15",
        )
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    # Visual-only plane at z=0; contact is disabled so playback poses stay unchanged.
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        material="groundplane",
        contype="0",
        conaffinity="0",
        group="0",
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0.15 0.25 0.35 1")
    if not root.findall(".//light[@name='tracking_key']"):
        ET.SubElement(
            worldbody,
            "light",
            name="tracking_key",
            pos="0 0 3.5",
            dir="0 0 -1",
            diffuse="0.55 0.55 0.55",
            ambient="0.18 0.18 0.18",
            specular="0.12 0.12 0.12",
            castshadow="false",
        )


def _build_model(
    xml_path: Path,
    meshdir: Path,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    add_ground: bool = False,
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
    # The Dex3 source XML has a smaller default framebuffer.  Match the
    # requested source-camera canvas before compiling the in-memory model.
    global_visual.set("offwidth", str(width))
    global_visual.set("offheight", str(height))
    if root.findall(f".//camera[@name='{CAMERA_NAME}']"):
        raise ValueError(f"camera {CAMERA_NAME!r} already exists in {xml_path}")
    if root.findall(f".//body[@name='{CAMERA_BODY_NAME}']"):
        raise ValueError(f"body {CAMERA_BODY_NAME!r} already exists in {xml_path}")
    torso = root.findall(f".//body[@name='{CAMERA_PARENT}']")
    if len(torso) != 1:
        raise ValueError(f"expected one {CAMERA_PARENT!r}, found {len(torso)}")
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
        principalpixel=_vec((float(K[0, 2] - width / 2.0), float(K[1, 2] - height / 2.0)),),
        sensorsize="1 1",
    )
    if add_ground:
        _inject_ground_plane(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = HEADLIGHT_SPECULAR
    if model.nlight:
        model.light_ambient[:] = SCENE_LIGHT_AMBIENT
        model.light_diffuse[:] = SCENE_LIGHT_DIFFUSE
        model.light_specular[:] = SCENE_LIGHT_SPECULAR
    return model


def _strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).reshape(-1).tolist()]


def _quat_from_mat(matrix: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(result, np.asarray(matrix, dtype=np.float64))
    if result[0] < 0.0:
        result *= -1.0
    return result


def _prepare_output(root: Path, force: bool) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    expected = {"rgb", "depth", "robot_id"}
    generated_files = {
        "kinematics.npz",
        "camera_K.json",
        "robot_instance.json",
    }
    unexpected = [
        path
        for path in root.iterdir()
        if path.name not in expected and path.name not in generated_files
    ]
    if unexpected:
        raise FileExistsError(f"unexpected entries in output: {unexpected}")
    packed_names = {"frames.npy", "volume_meta.json"}
    for name in expected:
        directory = root / name
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(directory)
        directory.mkdir(parents=True, exist_ok=True)
        existing = list(directory.iterdir())
        if existing and not force:
            raise FileExistsError(f"output directory is non-empty: {directory}")
        if force:
            for path in existing:
                allowed = (
                    path.is_file()
                    and (path.name[:6].isdigit() or path.name in packed_names)
                )
                if not allowed:
                    raise FileExistsError(f"refusing to remove unexpected entry {path}")
                path.unlink()
    return root / "rgb", root / "depth", root / "robot_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinematics-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument(
        "--camera-calibration-json",
        type=Path,
        help="Target camera K, image size and distortion metadata",
    )
    calibration.add_argument(
        "--rgb-k-npy",
        type=Path,
        help="Legacy target K-only input",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_DEX3_XML)
    parser.add_argument("--meshdir", type=Path, default=DEFAULT_MESHDIR)
    parser.add_argument(
        "--hand-dof",
        type=int,
        default=14,
        help=(
            "Hand width of the composed render archive: 14 for Dex3, 22 for "
            "the BrainCo hands once their mimic joints are expanded."
        ),
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.camera_calibration_json is not None:
        calibration_path = args.camera_calibration_json.expanduser().resolve()
        K, camera_calibration = _load_camera_calibration(calibration_path)
        configured_size = (
            int(camera_calibration["width"]),
            int(camera_calibration["height"]),
        )
        requested_size = (args.width, args.height)
        if configured_size != requested_size:
            raise ValueError(
                "target camera size must match its calibration exactly: "
                f"configured={configured_size}, requested={requested_size}"
            )
    else:
        calibration_path = args.rgb_k_npy.expanduser().resolve()
        K = _load_K(calibration_path)
        camera_calibration = {
            "width": args.width,
            "height": args.height,
            "K": K.tolist(),
            "resize_convention": "unspecified_legacy_k_npy",
            "projection_contract": "pinhole target projection; distortion is not applied",
        }
    source_path = args.kinematics_npz.expanduser().resolve()
    reference_path = args.reference_npz.expanduser().resolve()
    with np.load(source_path, allow_pickle=False) as source_archive:
        source = {name: np.asarray(source_archive[name]) for name in source_archive.files}
    with np.load(reference_path, allow_pickle=False) as reference_archive:
        reference = {name: np.asarray(reference_archive[name]) for name in reference_archive.files}
    required = ("joint_pos", "body_pos_w", "body_quat_w", "fps")
    missing = [name for name in required if name not in source]
    if missing:
        raise KeyError(f"kinematics missing {missing}")
    joint_names = _strings(source.get("joint_names", reference["joint_names"]))
    joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
    body_pos = np.asarray(source["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(source["body_quat_w"], dtype=np.float64)
    timestamps = np.asarray(
        source.get("timestamps", reference.get("timestamps")), dtype=np.float64
    )
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(joint_names):
        raise ValueError("joint_pos and joint_names shape mismatch")
    frames = int(joint_pos.shape[0])
    if args.max_frames is not None:
        frames = min(frames, int(args.max_frames))
    if frames <= 0:
        raise ValueError("no frames selected")
    if body_pos.shape[0] < frames or body_quat.shape[0] < frames:
        raise ValueError("body pose arrays are shorter than joint_pos")
    if timestamps.shape != (source["joint_pos"].shape[0],):
        raise ValueError(f"timestamps shape {timestamps.shape} does not match source")
    body_names = _strings(source.get("body_names", reference["body_names"]))
    if not body_names or body_names[0] != "pelvis":
        raise ValueError(f"expected pelvis as root body, got {body_names[:3]}")

    output = args.output_dir.expanduser().resolve()
    rgb_dir, depth_dir, robot_dir = _prepare_output(output, args.force)
    rgb_vol = create_volume(
        rgb_dir, frames, (args.height, args.width, 3), np.uint8, fill=0, color_space="rgb"
    )
    depth_vol = create_volume(depth_dir, frames, (args.height, args.width), np.float32, fill=0)
    robot_vol = create_volume(robot_dir, frames, (args.height, args.width), np.uint16, fill=0)
    rgb_vol = create_volume(
        rgb_dir, frames, (args.height, args.width, 3), np.uint8, fill=0, color_space="rgb"
    )
    depth_vol = create_volume(depth_dir, frames, (args.height, args.width), np.float32, fill=0)
    robot_vol = create_volume(robot_dir, frames, (args.height, args.width), np.uint16, fill=0)
    model = _build_model(
        args.robot_xml.expanduser().resolve(),
        args.meshdir.expanduser().resolve(),
        K,
        args.width,
        args.height,
    )
    model_joint_names = [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    if set(model_joint_names) != set(joint_names):
        raise ValueError(
            f"Dex3 joint mismatch; missing={sorted(set(model_joint_names)-set(joint_names))}, "
            f"extra={sorted(set(joint_names)-set(model_joint_names))}"
        )
    hand_joint_names = _strings(source.get("hand_joint_names", np.asarray([])))
    hand_dof = int(args.hand_dof)
    if len(hand_joint_names) != hand_dof or not set(hand_joint_names).issubset(
        set(joint_names)
    ):
        raise ValueError(
            f"composed render archive must contain all {hand_dof} hand_joint_names; "
            f"got {len(hand_joint_names)}"
        )
    hand_qpos = np.asarray(source.get("hand_qpos"), dtype=np.float64)
    if hand_qpos.shape != (source["joint_pos"].shape[0], hand_dof):
        raise ValueError(
            f"invalid hand_qpos shape {hand_qpos.shape}; expected (T, {hand_dof})"
        )
    if not np.all(np.isfinite(hand_qpos)):
        raise ValueError("hand_qpos contains non-finite values")
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    free_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_ids) != 1:
        raise ValueError(f"expected exactly one Dex3 free joint, found {free_ids}")
    free_id = int(free_ids[0])
    root_index = body_names.index("pelvis")
    body_ids_by_name = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names],
        dtype=np.int32,
    )
    missing_bodies = [
        name for name, body_id in zip(body_names, body_ids_by_name) if body_id < 0
    ]
    if missing_bodies:
        raise ValueError(f"kinematics body names are missing from Dex3 model: {missing_bodies}")
    camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME))
    geom_ids, body_ids, root_names = robot_geom_ids(model, root_body_hint="pelvis")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera_pos = np.empty((frames, 3), dtype=np.float32)
    camera_quat = np.empty((frames, 4), dtype=np.float32)
    rendered_body_pos = np.empty((frames, len(body_names), 3), dtype=np.float32)
    rendered_body_quat = np.empty((frames, len(body_names), 4), dtype=np.float32)
    robot_pixels: list[int] = []
    depth_min = math.inf
    depth_max = -math.inf
    started = time.perf_counter()
    try:
        for frame in range(frames):
            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.0
            free = int(model.jnt_qposadr[free_id])
            data.qpos[free : free + 3] = body_pos[frame, root_index]
            root_q = body_quat[frame, root_index]
            root_q = root_q / max(np.linalg.norm(root_q), 1.0e-12)
            data.qpos[free + 3 : free + 7] = root_q
            data.qpos[model.jnt_qposadr[joint_ids]] = joint_pos[frame]
            mujoco.mj_forward(model, data)

            rendered_body_pos[frame] = data.xpos[body_ids_by_name]
            rendered_body_quat[frame] = data.xquat[body_ids_by_name]
            camera_pos[frame] = data.cam_xpos[camera_id]
            camera_quat[frame] = _quat_from_mat(data.cam_xmat[camera_id]).astype(np.float32)
            renderer.update_scene(data, camera=CAMERA_NAME)
            rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
            if rgb.shape != (args.height, args.width, 3):
                raise RuntimeError(f"unexpected RGB shape {rgb.shape}")
            rgb_vol[frame] = rgb

            renderer.enable_depth_rendering()
            try:
                renderer.update_scene(data, camera=CAMERA_NAME)
                depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            finally:
                renderer.disable_depth_rendering()
            if depth.shape != (args.height, args.width):
                raise RuntimeError(f"unexpected depth shape {depth.shape}")
            depth_vol[frame] = depth
            finite = depth[np.isfinite(depth) & (depth > 0.0)]
            if finite.size:
                depth_min = min(depth_min, float(finite.min()))
                depth_max = max(depth_max, float(finite.max()))

            renderer.enable_segmentation_rendering()
            try:
                renderer.update_scene(data, camera=CAMERA_NAME)
                segmentation = np.asarray(renderer.render()).copy()
            finally:
                renderer.disable_segmentation_rendering()
            robot_id = robot_id_from_segmentation(segmentation, geom_ids)
            robot_vol[frame] = robot_id
            robot_pixels.append(int(np.count_nonzero(robot_id)))
            if frame == 0 or (frame + 1) % 50 == 0 or frame + 1 == frames:
                print(f"[torso-robot] frame {frame + 1}/{frames}", flush=True)
    finally:
        renderer.close()
        rgb_vol.flush()
        depth_vol.flush()
        robot_vol.flush()
    wall_s = time.perf_counter() - started

    camera_npz = {
        "joint_pos": np.asarray(source["joint_pos"][:frames]),
        "joint_vel": np.asarray(source.get("joint_vel", np.zeros_like(source["joint_pos"]))[:frames]),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        # Recompute all body poses from the assigned qpos. This keeps replay
        # outputs self-consistent when a diagnostic kinematics archive changes
        # a wrist joint while retaining the original root trajectory.
        "body_pos_w": rendered_body_pos,
        "body_quat_w": rendered_body_quat,
        "timestamps": timestamps[:frames],
        "fps": np.asarray(source["fps"]),
        "camera_pos_w": camera_pos,
        "camera_quat_wxyz": camera_quat,
        "camera_name": np.asarray(CAMERA_NAME),
        "hand_qpos": np.asarray(source["hand_qpos"][:frames]),
        "hand_joint_names": np.asarray(hand_joint_names),
    }
    for name in (
        "joint_offset_names",
        "joint_offset_degrees",
        "joint_offset_radians",
        "joint_offset_name",
        "joint_offset_deg",
        "joint_offset_rad",
        "joint_offset_source",
    ):
        if name in source:
            camera_npz[name] = np.asarray(source[name])
    np.savez_compressed(output / "kinematics.npz", **camera_npz)
    published_calibration = dict(camera_calibration)
    published_calibration.update(
        {
            "width": args.width,
            "height": args.height,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "K": K.tolist(),
            "unit": "pixel",
            "convention": "OpenGL camera; top-left RGB origin",
            "calibration_source": str(calibration_path),
        }
    )
    (output / "camera_K.json").write_text(
        json.dumps(
            published_calibration,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if "joint_offset_names" in source:
        offset_names = _strings(source["joint_offset_names"])
        offset_degrees = np.asarray(source["joint_offset_degrees"], dtype=np.float64).reshape(-1)
        offset_radians = np.asarray(source["joint_offset_radians"], dtype=np.float64).reshape(-1)
        if not (len(offset_names) == len(offset_degrees) == len(offset_radians)):
            raise ValueError("joint offset metadata arrays have inconsistent lengths")
        joint_offsets = [
            {"name": name, "degrees": float(degrees), "radians": float(radians)}
            for name, degrees, radians in zip(offset_names, offset_degrees, offset_radians)
        ]
    elif "joint_offset_name" in source:
        joint_offsets = [
            {
                "name": str(np.asarray(source["joint_offset_name"]).reshape(())),
                "degrees": float(np.asarray(source["joint_offset_deg"], dtype=np.float64).reshape(())),
                "radians": float(np.asarray(source["joint_offset_rad"], dtype=np.float64).reshape(())),
            }
        ]
    else:
        joint_offsets = []

    instance_payload = {
        "schema_version": 1,
        "mask_source": "mujoco_renderer_segmentation_replay_torso_umi3",
        "id_encoding": "mujoco_geom_id_plus_one",
        "background_id": 0,
        "mujoco_object_type": int(mujoco.mjtObj.mjOBJ_GEOM),
        "camera_name": CAMERA_NAME,
        "camera_pose_source": "run/kinematics.npz torso_link + fixed torso_to_d435 mount",
        "robot_root_body_names": root_names,
        "robot_body_ids": body_ids,
        "instances": [
            {
                "instance_id": int(geom_id) + 1,
                "geom_id": int(geom_id),
                "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)),
                "body_id": int(model.geom_bodyid[geom_id]),
                "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])),
            }
            for geom_id in geom_ids.tolist()
        ],
        "source_kinematics": str(source_path),
        "source_kinematics_sha256": _sha256(source_path),
        "reference_npz": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "robot_xml": str(args.robot_xml.expanduser().resolve()),
        "dex3_model_xml": str(args.robot_xml.expanduser().resolve()),
        "dex3_hinge_count": len(model_joint_names),
        "dex3_hand_joint_names": hand_joint_names,
        "hand_qpos_source": str(source.get("source_hand_kinematics", np.asarray(""))),
        "hand_retarget_method": "DexPilot qpos assigned by joint name",
        "joint_offsets": joint_offsets,
        "joint_offset": joint_offsets[0] if len(joint_offsets) == 1 else None,
        "frame_count": frames,
        "image_size_hw": [args.height, args.width],
        "intrinsics": K.tolist(),
        "rgb_path_pattern": "rgb/frames.npy",
        "depth_path_pattern": "depth/frames.npy",
        "robot_id_path_pattern": "robot_id/frames.npy",
        "depth_unit": "m",
        "robot_pixel_count": {
            "min": int(min(robot_pixels)),
            "max": int(max(robot_pixels)),
            "total": int(sum(robot_pixels)),
        },
        "depth_range_m": [
            None if not np.isfinite(depth_min) else depth_min,
            None if not np.isfinite(depth_max) else depth_max,
        ],
        "render_wall_s": wall_s,
        "formal_dataset_allowed": True,
        "legacy_robot_mask": False,
    }
    (output / "robot_instance.json").write_text(
        json.dumps(instance_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[torso-robot] wrote {frames} frames to {output} in {wall_s:.2f}s", flush=True)


if __name__ == "__main__":
    main()
