"""Use HaWoR's temporal network directly, without SLAM or motion infilling.

The network consumes 16-frame windows and pads its last window itself. We split
each hand into contiguous detected segments so missing frames never disappear
from the result or join nonadjacent observations. Single-frame inference is a
one-frame sequence padded by the same upstream API.

Joints are in the RGB optical camera frame, in metres. Exported MANO vertices
are local, with handedness applied; add ``_cam_t`` to obtain camera coordinates.
This is monocular model inference, not a metric depth measurement.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from typing import Optional

import cv2
import numpy as np

from stera.annotations.hands import FINGER_JOINTS, FINGER_NAMES, HandPose
from stera.core.types import Keypoint
from .config import HaWoRConfig

logger = logging.getLogger(__name__)
_LOAD_LOCK = threading.Lock()
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _natural_key(path):
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", Path(path).name)
    )


def _validate_intrinsics(intrinsics):
    if intrinsics is None:
        raise ValueError("HaWoR requires RGB camera intrinsics (a 3x3 rgb_K matrix).")
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("HaWoR intrinsics must be a finite 3x3 RGB camera matrix.")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("HaWoR RGB focal lengths fx and fy must be positive.")
    if not np.allclose(matrix[2], [0, 0, 1]) or not np.allclose(
        [matrix[0, 1], matrix[1, 0]], 0
    ):
        raise ValueError("HaWoR requires zero-skew pinhole RGB intrinsics.")
    return matrix.copy()


def _camera_translation(pred_cam, boxes, intrinsics):
    """Convert weak-perspective cameras using original, unflipped RGB boxes.

    HaWoR uses scalar focal length and a fixed 1.2 box dilation. The scalar is
    the mean of fx/fy; backprojection uses both calibrated focal lengths. We
    recompute translation because upstream's left-hand unflip assumes cx=W/2.
    """
    matrix = _validate_intrinsics(intrinsics)
    camera = np.asarray(pred_cam, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if camera.ndim != 2 or camera.shape[1] != 3 or boxes.shape != (len(camera), 4):
        raise ValueError("Expected camera shape (N,3) and boxes shape (N,4).")
    extent = boxes[:, 2:] - boxes[:, :2]
    if not np.isfinite(camera).all() or not np.isfinite(boxes).all():
        raise ValueError("HaWoR returned non-finite camera parameters or boxes.")
    if np.any(extent <= 0) or np.any(camera[:, 0] <= 0):
        raise ValueError("HaWoR needs positive bounding boxes and camera scales.")
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    denominator = extent.max(axis=1) * 1.2 * camera[:, 0]
    z = 2 * np.mean([matrix[0, 0], matrix[1, 1]]) / denominator
    translation = np.empty_like(camera)
    translation[:, 0] = camera[:, 1] + (center[:, 0] - matrix[0, 2]) * z / matrix[0, 0]
    translation[:, 1] = camera[:, 2] + (center[:, 1] - matrix[1, 2]) * z / matrix[1, 1]
    translation[:, 2] = z
    return translation.astype(np.float32)


def _to_hand_pose(
    joints, vertices, rotmat, betas, pred_cam, cam_t, intrinsics,
    confidence, side, frame_idx,
):
    joints = np.asarray(joints, dtype=np.float32).reshape(21, 3)
    if not np.isfinite(joints).all() or np.any(joints[:, 2] <= 0):
        raise ValueError("HaWoR returned invalid joints behind the RGB camera.")
    matrix = _validate_intrinsics(intrinsics)
    keypoints = [
        Keypoint(*map(float, xyz), confidence=float(confidence), name="wrist")
        for xyz in joints
    ]
    fingers = {}
    for i, finger in enumerate(FINGER_NAMES):
        fingers[finger] = keypoints[1 + 4 * i:5 + 4 * i]
        for kp, joint_name in zip(fingers[finger], FINGER_JOINTS):
            kp.name = f"{finger}_{joint_name}"
    hand = HandPose(
        wrist=keypoints[0], fingers=fingers, hand_side=side,
        frame_idx=frame_idx, confidence=float(confidence),
    )
    projected = joints @ matrix.T
    hand._kpts_2d_rgb = (projected[:, :2] / projected[:, 2:]).astype(np.float32)
    hand._mano_vertices = None if vertices is None else np.asarray(vertices, np.float32)
    hand._mano_global_orient = np.asarray(rotmat, np.float32)[:1].copy()
    hand._mano_hand_pose = np.asarray(rotmat, np.float32)[1:].copy()
    hand._mano_betas = np.asarray(betas, np.float32).reshape(10).copy()
    hand._pred_cam = np.asarray(pred_cam, np.float32).reshape(3).copy()
    # HaWoR returns a full-image translation, not WiLoR's crop-space camera.
    hand._pred_cam_t = np.asarray(cam_t, np.float32).reshape(3).copy()
    hand._cam_t = hand._pred_cam_t.copy()
    hand._focal_length = np.asarray([matrix[0, 0], matrix[1, 1]], np.float32)
    hand._backend = "hawor"
    return hand


class HaWoRHandTracker:
    """Temporal hand tracker backed by a local ThunderVVV/HaWoR checkout."""

    def __init__(self, config: Optional[HaWoRConfig] = None):
        self.config = config or HaWoRConfig()
        self._loaded = False
        self._model = None
        self._detector = None
        self._torch = None
        self._device = self.config.device
        self.last_stats = {}

    def load(self):
        if self._loaded:
            return
        cfg = self.config
        if not cfg.hawor_dir:
            raise ValueError(
                "Set HaWoRConfig.hawor_dir or HandTracker(model='hawor', "
                "model_path='/path/to/HaWoR')."
            )
        root = Path(cfg.hawor_dir).expanduser().resolve()
        if not (root / "lib/models/hawor.py").is_file():
            raise FileNotFoundError(
                f"HaWoR source missing at {root}; clone https://github.com/ThunderVVV/HaWoR."
            )

        def path_for(value, default):
            path = Path(value or default).expanduser()
            return path if path.is_absolute() else root / path

        checkpoint = path_for(cfg.checkpoint, "weights/hawor/checkpoints/hawor.ckpt")
        model_config = path_for(cfg.model_config, "weights/hawor/model_config.yaml")
        detector = path_for(cfg.hand_detector_path, "weights/external/detector.pt")
        for path in (checkpoint, model_config, detector):
            if not path.is_file():
                raise FileNotFoundError(
                    f"HaWoR asset missing: {path}. Follow HaWoR's model weights setup "
                    "or set the corresponding HaWoRConfig path."
                )
        if not 0 < cfg.yolo_conf <= 1:
            raise ValueError("HaWoR yolo_conf must be in (0, 1].")

        with _LOAD_LOCK:
            if self._loaded:
                return
            # HaWoR uses generic top-level 'lib' modules. Do not silently reuse
            # another checkout's modules in a long-lived Python process.
            for name in ("lib", "lib.models", "lib.models.hawor", "hawor", "hawor.configs"):
                module = sys.modules.get(name)
                origin = getattr(module, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(root):
                    raise RuntimeError(
                        f"Module {name!r} is already imported from {origin}; load "
                        "HaWoR in a fresh worker to avoid source-path conflicts."
                    )
            sys.path.insert(0, str(root))
            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                import torch
                from ultralytics import YOLO

                get_config = importlib.import_module("hawor.configs").get_config
                model_cfg = get_config(str(model_config), update_cachedir=True)
                model_cfg.defrost()
                # Keep all model assets independent of the caller's cwd.
                for key in ("MODEL_PATH", "MEAN_PARAMS", "DATA_DIR", "JOINT_REGRESSOR_EXTRA"):
                    value = model_cfg.MANO.get(key)
                    if value:
                        model_cfg.MANO[key] = str(path_for(value, value))
                mano_path = Path(model_cfg.MANO.MODEL_PATH)
                if mano_path.is_dir():
                    mano_path = mano_path / "MANO_RIGHT.pkl"
                for path in (mano_path, Path(model_cfg.MANO.MEAN_PARAMS)):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"HaWoR MANO asset missing: {path}. Install the licensed "
                            "MANO_RIGHT.pkl and bundled mano_mean_params.npz per HaWoR README."
                        )
                model_cfg.MODEL.LOAD_WEIGHTS = None
                model_cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
                if model_cfg.MODEL.BACKBONE.TYPE == "vit":
                    if model_cfg.MODEL.IMAGE_SIZE != 256:
                        raise ValueError("HaWoR ViT checkpoint requires MODEL.IMAGE_SIZE=256.")
                    model_cfg.MODEL.BBOX_SHAPE = [192, 256]
                model_cfg.freeze()
                device = torch.device(cfg.device)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError(
                        "HaWoR CUDA is unavailable. Use a CUDA PyTorch environment, "
                        "or explicitly choose device='cpu' for slow inference."
                    )
                model_type = importlib.import_module("lib.models.hawor").HAWOR
                model = model_type.load_from_checkpoint(
                    str(checkpoint), map_location="cpu", strict=False, weights_only=False, cfg=model_cfg,
                )
                self._model = model.to(device).eval()
                self._detector = YOLO(str(detector))
                self._torch = torch
                self._device = str(device)
                self._model_cfg = model_cfg
                self._loaded = True
            except ImportError as exc:
                raise RuntimeError(
                    f"HaWoR dependency could not be imported ({exc}). Install the "
                    "dependencies from the selected HaWoR checkout in this Python "
                    "environment; installing stera-sdk alone does not install model code."
                ) from exc
            finally:
                os.chdir(old_cwd)

    @contextmanager
    def _image_files(self, source):
        source = Path(source).expanduser().resolve()
        if source.is_dir():
            files = sorted(
                (p for p in source.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS),
                key=_natural_key,
            )
            if not files:
                raise ValueError(f"HaWoR input directory contains no JPEG/PNG frames: {source}")
            yield [str(p) for p in files]
            return
        if not source.is_file():
            raise FileNotFoundError(f"HaWoR RGB source not found: {source}")
        with tempfile.TemporaryDirectory(prefix="stera-hawor-", dir=self.config.work_dir) as temp:
            cap = cv2.VideoCapture(str(source))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Cannot open HaWoR RGB video: {source}")
                expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                files = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    path = Path(temp) / f"{len(files):09d}.png"
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        raise OSError(f"Cannot write temporary HaWoR frame: {path}")
                    files.append(str(path))
                if not files:
                    raise ValueError(f"HaWoR RGB video contains no decodable frames: {source}")
                if expected > 0 and len(files) != expected:
                    raise ValueError(
                        f"HaWoR decoded {len(files)} of {expected} reported video frames; "
                        "refusing a potentially misaligned sequence. Supply a complete frame directory."
                    )
                yield files
            finally:
                cap.release()

    def _detect_boxes(self, imgfiles):
        detections = []
        for path in imgfiles:
            # Ultralytics numpy input is BGR, matching cv2.imread.
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode HaWoR RGB frame: {path}")
            result = self._detector.predict(
                image, conf=self.config.yolo_conf,
                # Ultralytics rewrites CUDA_VISIBLE_DEVICES for string device
                # arguments. A torch.device retains this worker's allocation.
                device=self._torch.device(self._device), verbose=False,
            )[0]
            best = {}
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                for box, cls, score in zip(boxes, classes, scores):
                    if cls not in (0, 1) or not np.isfinite(box).all() or not np.isfinite(score):
                        continue
                    if np.any(box[2:] <= box[:2]) or score < self.config.yolo_conf:
                        continue
                    side = "right" if cls == 1 else "left"
                    if side not in best or score > best[side][1]:
                        best[side] = (box.copy(), float(score))
            detections.append(best)
        return detections

    def _infer_segment(self, imgfiles, boxes, side, intrinsics, image_size):
        torch = self._torch
        focal = float(np.mean([intrinsics[0, 0], intrinsics[1, 1]]))
        center = [float(intrinsics[0, 2]), float(intrinsics[1, 2])]
        is_left = side == "left"
        if is_left:
            center[0] = image_size[0] - center[0] - 1
        with torch.no_grad():
            result = self._model.inference(
                imgfiles, boxes.copy(), img_focal=focal, img_center=center,
                device=self._device, do_flip=is_left,
            )
            count = len(imgfiles)
            rotmat = result["pred_rotmat"].reshape(count, 16, 3, 3)
            betas = result["pred_shape"].reshape(count, 10)
            pred_cam = result["pred_cam"].reshape(count, 3).cpu().numpy()
            translations = _camera_translation(pred_cam, boxes, intrinsics)
            predictions = []
            # MANO decoding in small batches prevents memory growing with a
            # long uninterrupted track on a worker sharing a GPU.
            for start in range(0, count, 16):
                stop = min(start + 16, count)
                output = self._model.mano.query({
                    "pred_rotmat": rotmat[start:stop].to(self._device),
                    "pred_shape": betas[start:stop].to(self._device),
                })
                joints = output.joints.detach().cpu().numpy()[:, :21].copy()
                vertices = output.vertices.detach().cpu().numpy().copy()
                rotations = rotmat[start:stop].cpu().numpy().copy()
                shapes = betas[start:stop].cpu().numpy()
                if is_left:
                    joints[..., 0] *= -1
                    vertices[..., 0] *= -1
                    # Equivalent to upstream's (axis_x, -axis_y, -axis_z).
                    reflection = np.diag([-1, 1, 1]).astype(np.float32)
                    rotations = reflection @ rotations @ reflection
                for j, idx in enumerate(range(start, stop)):
                    predictions.append({
                        "joints": joints[j] + translations[idx],
                        "vertices": vertices[j] if self.config.save_mano_vertices else None,
                        "rotmat": rotations[j], "betas": shapes[j],
                        "pred_cam": pred_cam[idx], "cam_t": translations[idx],
                    })
        return predictions

    def detect_sequence(
        self, rgb_video, depth_dir=None, intrinsics=None, timestamps=None,
        image_is_rotated=None,
    ):
        """Return one entry per source frame, keeping undetected frames empty.

        Input is a video or a directory of JPEG/PNG frames in natural filename
        order. ``intrinsics`` must describe the supplied RGB images. Depth is
        ignored; a true ``image_is_rotated`` is rejected instead of silently
        applying the RGB/depth rotation convention of other backends.
        """
        matrix = _validate_intrinsics(intrinsics)
        if image_is_rotated:
            raise ValueError("HaWoR needs native RGB images and matching rgb_K; image_is_rotated must be False.")
        started = time.monotonic()
        self.last_stats = {}
        with self._image_files(rgb_video) as files:
            if timestamps is not None and len(timestamps) != len(files):
                raise ValueError("HaWoR timestamps must have exactly one entry per RGB frame.")
            size = None
            for path in files:
                frame = cv2.imread(path)
                if frame is None:
                    raise ValueError(f"Cannot decode HaWoR RGB frame: {path}")
                current_size = (frame.shape[1], frame.shape[0])
                if size is not None and current_size != size:
                    raise ValueError("HaWoR requires a constant RGB frame size matching rgb_K.")
                size = current_size
            self.load()
            detections = self._detect_boxes(files)
            if len(detections) != len(files):
                raise RuntimeError("HaWoR detector returned a misaligned frame sequence.")
            sequence = [[] for _ in files]
            segments = 0
            for side in ("left", "right"):
                indices = [i for i, det in enumerate(detections) if side in det]
                breaks = np.where(np.diff(indices) != 1)[0] + 1
                for segment in np.split(np.asarray(indices, dtype=int), breaks):
                    if not len(segment):
                        continue
                    boxes = np.asarray([detections[i][side][0] for i in segment], np.float32)
                    predictions = self._infer_segment(
                        [files[i] for i in segment], boxes, side, matrix, size,
                    )
                    if len(predictions) != len(segment):
                        raise RuntimeError("HaWoR inference returned a misaligned temporal segment.")
                    for index, prediction in zip(segment, predictions):
                        sequence[index].append(_to_hand_pose(
                            **prediction, intrinsics=matrix,
                            confidence=detections[index][side][1],
                            side=side, frame_idx=int(index),
                        ))
                    segments += 1
            self.last_stats = {
                "backend": "hawor", "coord_frame": "camera_3d",
                "n_frames": len(sequence), "n_segments": segments,
                "n_left": sum(any(h.hand_side == "left" for h in hs) for hs in sequence),
                "n_right": sum(any(h.hand_side == "right" for h in hs) for hs in sequence),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "device": str(self._device), "depth_used": False,
            }
            return sequence

    def detect_hands(self, rgb, depth=None, intrinsics=None, image_is_rotated=None):
        """Run a one-frame temporal segment; depth is accepted but unused."""
        _validate_intrinsics(intrinsics)
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("HaWoR detect_hands expects a uint8 RGB array of shape (H,W,3).")
        with tempfile.TemporaryDirectory(prefix="stera-hawor-", dir=self.config.work_dir) as temp:
            path = Path(temp) / "000000000.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Cannot write temporary HaWoR RGB frame: {path}")
            return self.detect_sequence(
                temp, intrinsics=intrinsics, image_is_rotated=image_is_rotated,
            )[0]
