"""Render one trajectory or an explicit left/right comparison as MP4.

This module is an offline presentation adapter.  CPU MuJoCo is used only to
assign recorded ``qpos``, run kinematics with ``mj_forward``, and render
pixels; it never advances dynamics or computes experiment metrics.  The
per-frame overlay values (label, root error, time) are display-only
quantities derived from the persisted traces; ``report.json`` remains the
sole experiment evidence.

Presentation composition per panel: the achieved robot keeps its native MJCF
materials while the nominal target pose is injected into the same 3D scene as
a translucent cyan ghost.  A checkered floor and world-fixed lighting are
added to the render model only; the physics model is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterator

import numpy as np


_CONTRACT_VERSION = "sil-tts-render-v2"

CAMERA_MODES = ("follow", "nominal-follow", "fixed-world")

_GHOST_RGBA = (0.0, 0.85, 1.0, 0.23)
_LIGHT_MODE_FIXED = 0
_LOOKAT_ROOT_LIFT_M = 0.15

_SCENE_VISUAL_XML = """\
  <visual>
    <rgba haze="0.12 0.18 0.28 1"/>
  </visual>
"""

_SCENE_ASSET_XML = """\
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.60" rgb2="0.08 0.12 0.20" width="512" height="3072"/>
    <texture type="2d" name="sil_tts_groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="sil_tts_groundplane" texture="sil_tts_groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""

_SCENE_WORLDBODY_XML = """\
    <geom name="sil_tts_floor" size="0 0 0.05" type="plane" material="sil_tts_groundplane" contype="0" conaffinity="0"/>
"""


@dataclass(frozen=True)
class RenderConfig:
    """Presentation settings for the offline comparison MP4.

    ``width`` is the total output width. A comparison gives half to each
    panel. ``camera_mode`` selects the per-frame look-at policy:

    - ``follow``: midpoint of the nominal and achieved root positions;
    - ``nominal-follow``: the nominal root position (shared across panels);
    - ``fixed-world``: one static look-at at the center of the rendered
      nominal root trajectory.

    ``nominal_lookat_height`` is the minimum world-z camera look-at value in
    the follow modes and the exact look-at z in ``fixed-world``.
    """

    enabled: bool = True
    fps: int = 30
    width: int = 1280
    height: int = 720
    distance: float = 4.0
    azimuth: float = 135.0
    elevation: float = -18.0
    nominal_lookat_height: float = 0.9
    camera_mode: str = "follow"
    show_root_error: bool = True
    show_time: bool = True

    def __post_init__(self) -> None:
        for name in ("enabled", "show_root_error", "show_time"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in ("fps", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        if self.width % 2:
            raise ValueError("width must be even for H.264 yuv420 output")
        if self.height % 2:
            raise ValueError("height must be even for H.264 yuv420 output")
        for name in (
            "distance",
            "azimuth",
            "elevation",
            "nominal_lookat_height",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.distance <= 0.0:
            raise ValueError("distance must be positive")
        if self.camera_mode not in CAMERA_MODES:
            raise ValueError(
                f"camera_mode must be one of {', '.join(CAMERA_MODES)}; "
                f"got {self.camera_mode!r}"
            )


@dataclass(frozen=True)
class RenderBindings:
    """Immutable experiment identities bound into render provenance."""

    source_commit: str
    target_sha256: str
    trace_sha256: str
    report_sha256: str
    mjcf_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_commit, str) or not self.source_commit:
            raise ValueError("source_commit must be a non-empty string")
        for name in (
            "target_sha256",
            "trace_sha256",
            "report_sha256",
            "mjcf_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class RenderTrack:
    """One named achieved qpos trajectory."""

    label: str
    qpos: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("render track label must be a non-empty string")


@dataclass(frozen=True)
class SingleView:
    """Render one achieved trajectory at full output width."""

    track: RenderTrack

    def __post_init__(self) -> None:
        if not isinstance(self.track, RenderTrack):
            raise TypeError("track must be RenderTrack")


@dataclass(frozen=True)
class ComparisonView:
    """Render an explicit left/right trajectory comparison."""

    left: RenderTrack
    right: RenderTrack

    def __post_init__(self) -> None:
        if not isinstance(self.left, RenderTrack) or not isinstance(
            self.right, RenderTrack
        ):
            raise TypeError("left and right must be RenderTrack")
        if self.left.label == self.right.label:
            raise ValueError("comparison track labels must be distinct")


RenderView = SingleView | ComparisonView


@dataclass(frozen=True)
class RenderProvenance:
    """Evidence returned to the composition root for sidecar persistence."""

    renderer_contract_version: str
    config: RenderConfig
    view: str
    track_labels: tuple[str, ...]
    source_commit: str
    target_sha256: str
    trace_sha256: str
    report_sha256: str
    mjcf_sha256: str
    frame_indices: tuple[int, ...]
    output_sha256: str


def render_mp4(
    *,
    mjcf_path: str | Path,
    timeline: np.ndarray,
    nominal_qpos: np.ndarray,
    view: RenderView,
    output_path: str | Path,
    config: RenderConfig,
    bindings: RenderBindings,
) -> RenderProvenance:
    """Render an explicit single or comparison view with one nominal ghost.

    ``nominal_qpos`` spans the full persisted timeline. Achieved arrays may be
    shorter after an early failure; presentation holds each last real pose to
    keep frames synchronized.
    """

    if not isinstance(config, RenderConfig):
        raise TypeError("config must be RenderConfig")
    if not config.enabled:
        raise ValueError("rendering is disabled by the persisted configuration")
    if not isinstance(bindings, RenderBindings):
        raise TypeError("bindings must be RenderBindings")

    model_path = Path(mjcf_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"MJCF does not exist: {model_path}")
    destination = Path(output_path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("output_path must end in .mp4")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing render: {destination}")

    tracks = _view_tracks(view)
    times, nominal, achieved = _validate_traces(timeline, nominal_qpos, tracks)
    indices = _frame_indices(times, config.fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = _render_frames(
        model_path,
        times,
        nominal,
        achieved,
        tuple(track.label for track in tracks),
        indices,
        config,
    )
    _encode_mp4(
        frames,
        destination,
        fps=config.fps,
        width=config.width,
        height=config.height,
    )
    if not destination.is_file():
        raise RuntimeError("MP4 encoder completed without producing output")

    return RenderProvenance(
        renderer_contract_version=_CONTRACT_VERSION,
        config=config,
        view="single" if isinstance(view, SingleView) else "comparison",
        track_labels=tuple(track.label for track in tracks),
        source_commit=bindings.source_commit,
        target_sha256=bindings.target_sha256,
        trace_sha256=bindings.trace_sha256,
        report_sha256=bindings.report_sha256,
        mjcf_sha256=bindings.mjcf_sha256,
        frame_indices=indices,
        output_sha256=_sha256_file(destination),
    )


def _view_tracks(view: RenderView) -> tuple[RenderTrack, ...]:
    if isinstance(view, SingleView):
        return (view.track,)
    if isinstance(view, ComparisonView):
        return (view.left, view.right)
    raise TypeError("view must be SingleView or ComparisonView")


def _validate_traces(
    timeline: np.ndarray,
    nominal_qpos: np.ndarray,
    tracks: tuple[RenderTrack, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    arrays = {
        "timeline": np.asarray(timeline),
        "nominal_qpos": np.asarray(nominal_qpos),
    }
    times = arrays["timeline"]
    if times.ndim != 1 or times.size < 1:
        raise ValueError("timeline must be a non-empty one-dimensional array")
    if not np.issubdtype(times.dtype, np.floating) or not np.isfinite(times).all():
        raise ValueError("timeline must be finite floating-point values")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("timeline must be strictly increasing")

    achieved = tuple(np.asarray(track.qpos) for track in tracks)
    for name, array in (
        ("nominal_qpos", arrays["nominal_qpos"]),
        *((track.label, array) for track, array in zip(tracks, achieved)),
    ):
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(f"{name} must have shape (frames, nq) with frames,nq > 0")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite floating-point values")
    nominal = arrays["nominal_qpos"]
    if nominal.shape[0] != times.size:
        raise ValueError("nominal_qpos length must equal timeline length")
    if any(array.shape[0] > times.size for array in achieved):
        raise ValueError("achieved qpos cannot extend beyond the nominal timeline")
    if any(array.shape[1] != nominal.shape[1] for array in achieved):
        raise ValueError("all qpos arrays must share the same nq")
    return (
        np.asarray(times, dtype=np.float64, order="C"),
        np.asarray(nominal, dtype=np.float64, order="C"),
        tuple(np.asarray(array, dtype=np.float64, order="C") for array in achieved),
    )


def _frame_indices(timeline: np.ndarray, fps: int) -> tuple[int, ...]:
    if timeline.size == 1:
        return (0,)
    sample_times = np.arange(
        timeline[0], timeline[-1] + np.finfo(np.float64).eps, 1.0 / fps
    )
    right = np.searchsorted(timeline, sample_times, side="left")
    right = np.clip(right, 0, timeline.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(sample_times - timeline[left]) <= np.abs(
        timeline[right] - sample_times
    )
    indices = np.where(choose_left, left, right).astype(np.int64).tolist()
    if indices[-1] != timeline.size - 1:
        indices.append(timeline.size - 1)
    return tuple(indices)


_INCLUDE_RE = re.compile(r'<include\b[^>]*\bfile="([^"]*)"[^>]*/>')


def _rewrite_resource_paths(xml_text: str, root: Path) -> str:
    def _absolute(match: re.Match[str]) -> str:
        attribute, value = match.group(1), match.group(2)
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return f'{attribute}="{path}"'

    xml_text = re.sub(r'(meshdir|texturedir)="([^"]*)"', _absolute, xml_text)
    if "<compiler" in xml_text and 'meshdir="' not in xml_text:
        xml_text = xml_text.replace(
            "<compiler", f'<compiler meshdir="{root}"', 1
        )
    return xml_text


def _mujoco_children(xml_text: str, source: Path) -> str:
    start = re.search(r"<mujoco\b[^>]*>", xml_text)
    end = xml_text.rfind("</mujoco>")
    if start is None or end < 0:
        raise ValueError(f"included MJCF is missing a <mujoco> root: {source}")
    return xml_text[start.end() : end]


def _expand_mjcf(mjcf_path: Path) -> str:
    mjcf_path = mjcf_path.resolve()
    xml_text = _rewrite_resource_paths(
        mjcf_path.read_text(encoding="utf-8"), mjcf_path.parent
    )

    def _inline_include(match: re.Match[str]) -> str:
        included = Path(match.group(1))
        if not included.is_absolute():
            included = mjcf_path.parent / included
        included = included.resolve()
        if not included.is_file():
            raise ValueError(f"MJCF include is missing: {included}")
        return _mujoco_children(_expand_mjcf(included), included)

    return _INCLUDE_RE.sub(_inline_include, xml_text)


def _scene_xml(mjcf_path: Path) -> str:
    """Return the robot MJCF with the presentation floor and skybox injected.

    Relative ``<include file>`` documents are inlined and
    ``meshdir``/``texturedir`` are rewritten to absolute paths so the
    composed XML can be compiled from a string without relocating assets.
    """

    xml_text = _expand_mjcf(mjcf_path)

    asset_open = xml_text.find("<asset>")
    if asset_open < 0:
        raise ValueError(f"MJCF has no <asset> block to extend: {mjcf_path}")
    if "<visual>" not in xml_text:
        xml_text = (
            xml_text[:asset_open] + _SCENE_VISUAL_XML + xml_text[asset_open:]
        )

    asset_close = xml_text.find("</asset>")
    xml_text = xml_text[:asset_close] + _SCENE_ASSET_XML + xml_text[asset_close:]

    worldbody_close = xml_text.find("</worldbody>")
    if worldbody_close < 0:
        raise ValueError(f"MJCF has no <worldbody> block to extend: {mjcf_path}")
    return (
        xml_text[:worldbody_close]
        + _SCENE_WORLDBODY_XML
        + xml_text[worldbody_close:]
    )


def _configure_global_lighting(model: object) -> None:
    """Use world-fixed lighting so brightness does not track the robot."""

    model.vis.headlight.ambient[:] = (0.28, 0.28, 0.28)
    model.vis.headlight.diffuse[:] = (0.0, 0.0, 0.0)
    model.vis.headlight.specular[:] = (0.0, 0.0, 0.0)
    fixed_lights = (
        ((-3.0, -4.0, 6.0), (3.0, 4.0, -6.0), (0.55, 0.55, 0.55)),
        ((4.0, 2.0, 5.0), (-4.0, -2.0, -5.0), (0.35, 0.35, 0.35)),
        ((0.0, 5.0, 4.0), (0.0, -5.0, -4.0), (0.25, 0.25, 0.25)),
    )
    for light_id in range(model.nlight):
        pos, direction, diffuse = fixed_lights[min(light_id, len(fixed_lights) - 1)]
        model.light_mode[light_id] = _LIGHT_MODE_FIXED
        model.light_pos[light_id] = pos
        model.light_dir[light_id] = direction
        model.light_diffuse[light_id] = diffuse
        model.light_ambient[light_id] = (0.12, 0.12, 0.12)
        model.light_specular[light_id] = (0.05, 0.05, 0.05)


def _lookat(
    config: RenderConfig,
    nominal_root: np.ndarray,
    achieved_root: np.ndarray,
    fixed_lookat: np.ndarray,
) -> np.ndarray:
    if config.camera_mode == "fixed-world":
        return fixed_lookat
    if config.camera_mode == "follow":
        lookat = 0.5 * (nominal_root + achieved_root)
    else:  # nominal-follow
        lookat = nominal_root.copy()
    lookat[2] = max(
        float(lookat[2]) + _LOOKAT_ROOT_LIFT_M, config.nominal_lookat_height
    )
    return lookat


def _annotate_panel(
    cv2: object,
    image: np.ndarray,
    *,
    label: str,
    root_error: float | None,
    time_s: float | None,
) -> None:
    height = image.shape[0]
    scale = max(0.35, height / 540.0)
    thickness = max(1, round(2.0 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines: list[tuple[str, float, tuple[int, int, int]]] = [
        (label, 0.7 * scale, (220, 220, 220)),
    ]
    if root_error is not None:
        lines.append((f"root err: {root_error:.2f} m", 0.55 * scale, (235, 235, 210)))
    if time_s is not None:
        lines.append((f"t = {time_s:.2f} s", 0.55 * scale, (235, 235, 210)))
    y = round(30.0 * scale)
    for text, font_scale, color in lines:
        cv2.putText(image, text, (10, y), font, font_scale, color, thickness)
        y += round(28.0 * scale)


def _render_frames(
    mjcf_path: Path,
    timeline: np.ndarray,
    nominal_qpos: np.ndarray,
    achieved_qpos: tuple[np.ndarray, ...],
    track_labels: tuple[str, ...],
    frame_indices: tuple[int, ...],
    config: RenderConfig,
) -> Iterator[np.ndarray]:
    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise RuntimeError("rendering requires the mujoco package") from error
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise RuntimeError("rendering requires opencv-python-headless") from error

    model = mujoco.MjModel.from_xml_string(_scene_xml(mjcf_path))
    if model.nq != nominal_qpos.shape[1]:
        raise ValueError(
            f"persisted qpos has nq={nominal_qpos.shape[1]}, MJCF expects {model.nq}"
        )
    _configure_global_lighting(model)
    panel_width = config.width // len(achieved_qpos)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, panel_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
    try:
        renderer = mujoco.Renderer(model, height=config.height, width=panel_width)
    except Exception as error:  # presentation dependency failure
        raise RuntimeError("MuJoCo could not create an offscreen renderer") from error
    data_sim = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = config.distance
    camera.azimuth = config.azimuth
    camera.elevation = config.elevation
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    perturb = mujoco.MjvPerturb()
    robot_geoms = np.flatnonzero(model.geom_bodyid != 0)
    robot_rgba = model.geom_rgba[robot_geoms].copy()
    ghost_rgba = np.asarray(_GHOST_RGBA, dtype=model.geom_rgba.dtype)

    rendered_roots = nominal_qpos[np.asarray(frame_indices, dtype=np.int64), :3]
    fixed_lookat = np.array(
        (
            0.5 * (rendered_roots[:, 0].min() + rendered_roots[:, 0].max()),
            0.5 * (rendered_roots[:, 1].min() + rendered_roots[:, 1].max()),
            config.nominal_lookat_height,
        ),
        dtype=np.float64,
    )

    try:
        for index in frame_indices:
            nominal = nominal_qpos[index]
            data_ref.qpos[:] = nominal
            mujoco.mj_forward(model, data_ref)
            time_s = float(timeline[index]) if config.show_time else None
            panels = []
            for qpos, label in zip(achieved_qpos, track_labels):
                achieved = qpos[min(index, qpos.shape[0] - 1)]
                data_sim.qpos[:] = achieved
                mujoco.mj_forward(model, data_sim)
                camera.lookat[:] = _lookat(
                    config, nominal[:3].copy(), achieved[:3].copy(), fixed_lookat
                )
                renderer.update_scene(data_sim, camera=camera)
                model.geom_rgba[robot_geoms] = ghost_rgba
                try:
                    mujoco.mjv_addGeoms(
                        model,
                        data_ref,
                        scene_option,
                        perturb,
                        mujoco.mjtCatBit.mjCAT_DYNAMIC,
                        renderer.scene,
                    )
                finally:
                    model.geom_rgba[robot_geoms] = robot_rgba
                image = np.array(renderer.render(), copy=True)
                root_error = (
                    float(np.linalg.norm(achieved[:3] - nominal[:3]))
                    if config.show_root_error
                    else None
                )
                _annotate_panel(
                    cv2,
                    image,
                    label=label,
                    root_error=root_error,
                    time_s=time_s,
                )
                panels.append(image)
            yield np.concatenate(panels, axis=1).astype(np.uint8)
    finally:
        renderer.close()


def _encode_mp4(
    frames: Iterator[np.ndarray],
    output_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("rendering requires ffmpeg on PATH")
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    assert process.stderr is not None
    try:
        count = 0
        for frame in frames:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError(
                    f"renderer produced {frame.shape}/{frame.dtype}; expected "
                    f"{(height, width, 3)}/uint8"
                )
            process.stdin.write(frame.tobytes(order="C"))
            count += 1
        process.stdin.close()
        stderr = process.stderr.read()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        output_path.unlink(missing_ok=True)
        raise
    if count == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("renderer produced no frames")
    if return_code:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace').strip()}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
