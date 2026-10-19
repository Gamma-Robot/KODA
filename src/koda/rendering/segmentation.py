"""Select robot geometry and decode MuJoCo segmentation into robot-ID masks.

The optional recorder hook also saves robot-ID masks alongside RGB and metric
depth for recorder classes exposing the FrontCamRecorder interface.
"""
# Migrated from KODA's SparseTrack adapter: scripts/mujoco_robot_id_capture.py.
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mujoco
import numpy as np
from PIL import Image


def _output_dir_from_argv(argv: list[str] | None = None) -> Path:
    args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(args):
        if value == "--output-dir":
            if index + 1 >= len(args):
                raise ValueError("--output-dir requires a path")
            return Path(args[index + 1]).expanduser().resolve()
        if value.startswith("--output-dir="):
            return Path(value.partition("=")[2]).expanduser().resolve()
    raise ValueError("robot-ID capture requires recorder --output-dir")


def _object_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str | None:
    return mujoco.mj_id2name(model, obj_type, int(obj_id))


def _matches_body_name(name: str | None, target: str) -> bool:
    if not name:
        return False
    return name == target or name.endswith(f"/{target}") or name.endswith(f":{target}")


def robot_geom_ids(
    model: mujoco.MjModel,
    *,
    root_body_hint: str | None = None,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Return geom IDs belonging to the robot body's kinematic subtree.

    G1 assets are rooted at pelvis. A namespaced recorder model is supported
    through suffix matching; KODA_ROBOT_ROOT_BODY can override that name for
    another robot asset. The previous SPARSETRACK_ROBOT_ROOT_BODY name remains
    a fallback for existing configurations.
    """
    hint = root_body_hint or os.environ.get(
        "KODA_ROBOT_ROOT_BODY", os.environ.get("SPARSETRACK_ROBOT_ROOT_BODY", "pelvis")
    )
    candidates = [part.strip() for part in hint.split(",") if part.strip()]
    if not candidates:
        raise ValueError("robot root-body hint is empty")

    roots: list[int] = []
    root_names: list[str] = []
    for body_id in range(int(model.nbody)):
        body_name = _object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if any(_matches_body_name(body_name, candidate) for candidate in candidates):
            roots.append(body_id)
            root_names.append(body_name or f"body_{body_id}")
    if not roots:
        available = [
            _object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(min(int(model.nbody), 16))
        ]
        raise ValueError(
            f"could not find robot root body {candidates!r}; first model bodies: {available!r}"
        )

    root_set = set(roots)
    robot_bodies: set[int] = set()
    for body_id in range(int(model.nbody)):
        current = body_id
        while True:
            if current in root_set:
                robot_bodies.add(body_id)
                break
            if current == 0:
                break
            current = int(model.body_parentid[current])

    geom_body = np.asarray(model.geom_bodyid, dtype=np.int32)
    geom_ids = np.asarray(
        [geom_id for geom_id, body_id in enumerate(geom_body) if int(body_id) in robot_bodies],
        dtype=np.int32,
    )
    if geom_ids.size == 0:
        raise ValueError(f"robot root bodies {root_names!r} own no visual geoms")
    return geom_ids, sorted(robot_bodies), root_names


def robot_id_from_segmentation(
    segmentation: np.ndarray,
    robot_geom_ids_: np.ndarray,
) -> np.ndarray:
    """Decode MuJoCo geom/object-type segmentation into uint16 IDs."""
    seg = np.asarray(segmentation)
    if seg.ndim != 3 or seg.shape[-1] != 2:
        raise ValueError(f"expected HxWx2 MuJoCo segmentation, got {seg.shape}")
    geom_id = np.asarray(seg[..., 0], dtype=np.int64)
    obj_type = np.asarray(seg[..., 1], dtype=np.int64)
    is_robot = (obj_type == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        geom_id, np.asarray(robot_geom_ids_, dtype=np.int64)
    )
    # Reserve zero for background; the stored value is MuJoCo geom_id + 1.
    out = np.zeros(geom_id.shape, dtype=np.uint16)
    out[is_robot] = (geom_id[is_robot] + 1).astype(np.uint16)
    return out


class _RobotIdCapture:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.robot_dir = output_dir / "robot_id"
        self.robot_dir.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0
        self.finalized = False
        self._geom_ids: np.ndarray | None = None
        self._body_ids: list[int] = []
        self._root_names: list[str] = []
        self._instances: list[dict[str, Any]] = []
        self._shape: tuple[int, int] | None = None

    def _attach(self, recorder: Any) -> None:
        if self._geom_ids is not None:
            return
        geom_ids, body_ids, root_names = robot_geom_ids(recorder._model)
        self._geom_ids = geom_ids
        self._body_ids = body_ids
        self._root_names = root_names
        self._instances = [
            {
                "instance_id": int(geom_id) + 1,
                "geom_id": int(geom_id),
                "geom_name": _object_name(
                    recorder._model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
                ),
                "body_id": int(recorder._model.geom_bodyid[geom_id]),
                "body_name": _object_name(
                    recorder._model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(recorder._model.geom_bodyid[geom_id]),
                ),
            }
            for geom_id in geom_ids.tolist()
        ]

    def capture(self, recorder: Any) -> None:
        self._attach(recorder)
        assert self._geom_ids is not None
        renderer = recorder._renderer
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(recorder._data, camera=recorder._camera_name)
            segmentation = np.asarray(renderer.render()).copy()
        finally:
            renderer.disable_segmentation_rendering()
        robot_id = robot_id_from_segmentation(segmentation, self._geom_ids)
        self._shape = tuple(int(value) for value in robot_id.shape)
        Image.fromarray(robot_id, mode="I;16").save(
            self.robot_dir / f"{self.frame_index:06d}.png"
        )
        self.frame_index += 1

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        rgb_count = len(list((self.output_dir / "rgb").glob("*.png")))
        depth_count = len(list((self.output_dir / "depth").glob("*.npy")))
        if rgb_count and rgb_count != self.frame_index:
            raise RuntimeError(
                f"robot-ID frame count {self.frame_index} != RGB frame count {rgb_count}"
            )
        if depth_count and depth_count != self.frame_index:
            raise RuntimeError(
                f"robot-ID frame count {self.frame_index} != depth frame count {depth_count}"
            )
        payload = {
            "schema_version": 1,
            "mask_source": "mujoco_renderer_segmentation",
            "id_encoding": "mujoco_geom_id_plus_one",
            "background_id": 0,
            "mujoco_object_type": int(mujoco.mjtObj.mjOBJ_GEOM),
            "robot_root_body_names": self._root_names,
            "robot_body_ids": self._body_ids,
            "instances": self._instances,
            "frame_count": self.frame_index,
            "image_size_hw": list(self._shape) if self._shape is not None else None,
            "formal_dataset_allowed": True,
            "legacy_robot_mask": False,
        }
        (self.output_dir / "robot_instance.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def install_robot_id_capture(module: ModuleType) -> bool:
    """Install the hook on a bytecode recorder module exactly once."""
    recorder_cls = getattr(module, "FrontCamRecorder", None)
    if recorder_cls is None:
        return False
    if getattr(recorder_cls, "_robot_id_capture_hook_installed", False):
        return True

    original_init = recorder_cls.__init__
    original_render = recorder_cls.render
    original_close = recorder_cls.close
    holder: dict[str, _RobotIdCapture | None] = {"capture": None}

    def capture() -> _RobotIdCapture:
        active = holder["capture"]
        if active is None or active.finalized:
            active = _RobotIdCapture(_output_dir_from_argv())
            holder["capture"] = active
        return active

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        capture()._attach(self)

    def patched_render(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_render(self, *args, **kwargs)
        capture().capture(self)
        return result

    def patched_close(self: Any) -> None:
        try:
            original_close(self)
        finally:
            capture().finalize()

    recorder_cls.__init__ = patched_init
    recorder_cls.render = patched_render
    recorder_cls.close = patched_close
    recorder_cls._robot_id_capture_hook_installed = True
    return True
