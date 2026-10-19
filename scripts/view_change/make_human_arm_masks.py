#!/usr/bin/env python3
"""Create per-frame full-body masks for the offline single-frame renderer.

Pixels removed from the measured RGB-D stream are the visible human body:
hands, arms, torso, and legs.  EfficientSAM3 is prompted for the arms, the
person, and the legs.  WiLoR/Stera 2-D keypoints still constrain the arm
prompt, so a separately segmented held object is not removed.  Torso and leg
components are kept without that keypoint test; only small components are
dropped.  When the model or its dependencies are unavailable, a conservative
keypoint hull plus a short forearm corridor is written and the metadata
records the low-confidence fallback.

The output intentionally uses ``human_mask`` rather than a generic
``dynamic_mask``.  Moving objects other than the human are not inferred or
removed by this stage.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from frame_volume import as_mask, create_volume, hardlink_packed, is_packed_stream, open_volume
except ImportError:
    from .frame_volume import as_mask, create_volume, hardlink_packed, is_packed_stream, open_volume


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
ASSET_ROOT = Path(os.environ.get("KODA_ASSETS", str(PROJECT_ROOT / "assets"))).expanduser().resolve()


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb-mp4", type=Path, required=True)
    p.add_argument(
        "--source-rgb-dir",
        type=Path,
        default=None,
        help="Optional packed OpenCV RGB volume; skips mp4 decode for size and SAM.",
    )
    p.add_argument("--annotation-h5", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--backend", choices=("auto", "efficientsam3", "keypoints"), default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--eff-root",
        type=Path,
        default=RESOURCE_ROOT / "efficientsam3/sam3",
    )
    p.add_argument(
        "--eff-ckpt",
        type=Path,
        default=ASSET_ROOT / "models/efficientsam3/efficientsam3_tinyvit.pt",
    )
    p.add_argument("--prompt", default="human arms and hands")
    p.add_argument("--body-prompt", default="person", help="Full visible body. Empty disables this prompt.")
    p.add_argument("--leg-prompt", default="human legs", help="Lower body. Empty disables this prompt.")
    p.add_argument("--body-min-area-fraction", type=float, default=0.002)
    p.add_argument("--object-protect-px", type=int, default=24)
    p.add_argument("--body-dilation-px", type=int, default=8)
    p.add_argument("--confidence-threshold", type=float, default=0.35)
    p.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="Hard-link packed human_mask volumes from a prior contact precompute instead of reloading EfficientSAM3",
    )
    p.add_argument("--support-dilation-px", type=int, default=14)
    p.add_argument("--corridor-width-px", type=int, default=42)
    p.add_argument("--corridor-length-scale", type=float, default=2.4)
    return p.parse_args()


def _hull(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    out = np.zeros((h, w), dtype=np.uint8)
    if len(pts) < 3:
        return out
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillConvexPoly(out, cv2.convexHull(pts.astype(np.int32)), 255)
    return out


def _support_mask(points: np.ndarray, valid: bool, shape: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    """Keypoint hull plus a bounded corridor towards the forearm."""
    h, w = shape
    if not valid:
        return np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if len(pts) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    hull = _hull(pts, shape)
    # MediaPipe/WiLoR convention: wrist is joint 0.  Extrapolate from the
    # palm centre through the wrist to obtain a short proximal arm corridor.
    wrist = pts[0]
    palm = np.mean(pts[: min(21, len(pts))], axis=0)
    direction = wrist - palm
    norm = float(np.linalg.norm(direction))
    span = float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 1])) * 0.5
    if norm < 1e-3:
        direction = np.array([0.0, 1.0], dtype=np.float32)
        norm = 1.0
    direction = direction / norm
    length = max(30.0, float(args.corridor_length_scale) * max(span, 20.0))
    end = wrist + direction * length
    corridor = np.zeros_like(hull)
    cv2.line(
        corridor,
        tuple(np.rint(wrist).astype(int)),
        tuple(np.rint(end).astype(int)),
        255,
        max(2, int(args.corridor_width_px)),
        cv2.LINE_AA,
    )
    out = np.maximum(hull, corridor)
    d = max(0, int(args.support_dilation_px))
    if d:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        out = cv2.dilate(out, k)
    return out


def _filtered_components(raw: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Keep only segmentation components supported by hand/arm keypoints."""
    raw = (np.asarray(raw) > 127).astype(np.uint8)
    if not raw.any():
        # EfficientSAM3 can legitimately return no text-grounded instance on
        # a difficult blur/occlusion frame.  The keypoint support is a
        # constrained rescue for that frame; silently returning an empty mask
        # would put the hand depth back into the geometry stream.
        return support.astype(np.uint8)
    if not np.any(support):
        # Missing/invalid WiLoR keypoints must not erase a valid text-grounded
        # EfficientSAM3 result.  In that case the model mask is the only
        # available evidence and is kept explicitly as a SAM-only frame.
        return raw * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    # A slightly enlarged support avoids dropping thin mask boundaries while
    # retaining the keypoint constraint at component level.
    support_d = cv2.dilate(support, np.ones((9, 9), np.uint8), iterations=1) > 0
    keep = np.zeros_like(raw, dtype=np.uint8)
    for label in range(1, n):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area and np.any(component & support_d):
            keep[component] = 255
    # If EfficientSAM3 returned a fragmented mask with no exact overlap, the
    # keypoint support is safer than silently producing an empty depth mask.
    return keep if keep.any() else support


def _body_components(raw: np.ndarray, min_area_fraction: float) -> np.ndarray:
    """Keep torso and leg components. Drop specks so a small prompt leak is not a mask."""
    raw_bin = (np.asarray(raw) > 127).astype(np.uint8)
    if not raw_bin.any():
        return raw_bin
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_bin, connectivity=8)
    height, width = raw_bin.shape
    min_area = max(64, int(float(min_area_fraction) * height * width))
    keep = np.zeros_like(raw_bin)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            keep[labels == label] = 255
    return keep


def _subtract_rejected_object(
    body: np.ndarray,
    arm_raw: np.ndarray,
    arm_kept: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """Drop a held object that the arm prompt segmented apart from the hand."""
    rejected = ((np.asarray(arm_raw) > 127) & (np.asarray(arm_kept) == 0)).astype(np.uint8)
    if radius_px <= 0 or not rejected.any():
        return body
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(radius_px) + 1, 2 * int(radius_px) + 1)
    )
    rejected = cv2.dilate(rejected, kernel)
    out = np.array(body, copy=True)
    out[rejected > 0] = 0
    return out


def compose_human_mask(
    arm_raw: np.ndarray,
    body_raws: list[np.ndarray],
    support: np.ndarray,
    *,
    min_area_fraction: float,
    object_protect_px: int,
    body_dilation_px: int,
) -> np.ndarray:
    """Union keypoint-constrained arms with torso and leg components."""
    arm = _filtered_components(arm_raw, support)
    body = np.zeros_like(arm)
    for raw in body_raws:
        body = np.maximum(body, _body_components(raw, min_area_fraction))
    body = _subtract_rejected_object(body, arm_raw, arm, object_protect_px)
    if body_dilation_px > 0 and np.any(body):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(body_dilation_px) + 1, 2 * int(body_dilation_px) + 1),
        )
        body = cv2.dilate(body, kernel)
        body = _subtract_rejected_object(body, arm_raw, arm, object_protect_px)
    return np.maximum(arm, body)


def _active_prompts(args: argparse.Namespace) -> tuple[str, list[str]]:
    arm = str(args.prompt).strip()
    if not arm:
        raise ValueError("--prompt must name the arm/hand text prompt")
    body: list[str] = []
    for prompt in (args.body_prompt, args.leg_prompt):
        text = str(prompt).strip()
        if text and text not in body and text != arm:
            body.append(text)
    return arm, body


def _read_keypoints(path: Path, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        left = np.asarray(f["hand-pose/left_kpts_2d_rgb"][start:end], dtype=np.float32)
        right = np.asarray(f["hand-pose/right_kpts_2d_rgb"][start:end], dtype=np.float32)
        left_valid = np.asarray(f["hand-pose/left_valid"][start:end], dtype=bool)
        right_valid = np.asarray(f["hand-pose/right_valid"][start:end], dtype=bool)
    return left, right, left_valid, right_valid


def _try_efficientsam3(
    args: argparse.Namespace,
    prompts: list[str],
    start: int,
    n: int,
    shape: tuple[int, int],
) -> tuple[dict[str, list[np.ndarray]] | None, dict[str, object]]:
    """Run the existing batched EfficientSAM3 adapter in an isolated temp dir."""
    if args.backend == "keypoints":
        return None, {"backend": "keypoints", "fallback": True, "reason": "requested"}
    try:
        from sam3_efficientsam3_hand_masks import (
            load_efficientsam3_image_model,
            run_episode_efficientsam3,
        )
        if not args.eff_ckpt.is_file():
            raise FileNotFoundError(args.eff_ckpt)
        import torch

        device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        sock = str(os.environ.get("EFFICIENTSAM3_WORKER_SOCK", "")).strip()
        packed_rgb = (
            args.source_rgb_dir.expanduser().resolve()
            if args.source_rgb_dir is not None
            else None
        )
        with tempfile.TemporaryDirectory(prefix="human-mask-") as td:
            root = Path(td)
            jobs = [(prompt, root / f"prompt_{index:02d}") for index, prompt in enumerate(prompts)]
            if sock:
                from efficientsam3_client import segment_jobs

                segment_jobs(
                    video=args.rgb_mp4,
                    jobs=jobs,
                    start_frame=start,
                    max_frames=n,
                    conf_threshold=args.confidence_threshold,
                    temporal_radius=0,
                    batch_size=8,
                    force=True,
                    packed_rgb=packed_rgb,
                    socket_path=sock,
                )
            else:
                model = load_efficientsam3_image_model(
                    eff_root=args.eff_root, eff_ckpt=args.eff_ckpt, device=device,
                    conf_threshold=args.confidence_threshold,
                )
                run_episode_efficientsam3(
                    model,
                    video=args.rgb_mp4,
                    jobs=jobs,
                    start_frame=start,
                    max_frames=n,
                    packed_rgb=packed_rgb,
                    temporal_radius=0,
                    batch_size=8,
                    eff_ckpt=args.eff_ckpt,
                )
            masks = {
                prompt: _read_sam_masks(directory / "masks", n, shape)
                for prompt, directory in jobs
            }
        return masks, {
            "backend": "EfficientSAM3",
            "fallback": False,
            "prompts": prompts,
        }
    except Exception as exc:  # model failure is an explicit low-confidence fallback
        if args.backend == "efficientsam3":
            raise RuntimeError(
                "EfficientSAM3 was explicitly requested but failed; refusing "
                "to generate a fallback human mask"
            ) from exc
        return None, {
            "backend": "keypoint_hull_forearm_corridor",
            "fallback": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "requested_backend": "EfficientSAM3",
        }


def _read_sam_masks(masks_dir: Path, n: int, shape: tuple[int, int]) -> list[np.ndarray]:
    if is_packed_stream(masks_dir):
        volume = open_volume(masks_dir)
        if len(volume) != n:
            raise RuntimeError(f"EfficientSAM3 returned {len(volume)} packed masks, expected {n}")
        masks = []
        for index in range(n):
            mask = (as_mask(volume[index]).astype(np.uint8) * 255)
            if mask.shape != shape:
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        return masks
    paths = sorted(masks_dir.glob("*.png"), key=lambda path: int(path.stem))
    if len(paths) != n:
        raise RuntimeError(f"EfficientSAM3 returned {len(paths)} masks, expected {n}")
    masks = []
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot read EfficientSAM3 mask {path}")
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(mask)
    return masks


def _probe_rgb_size(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.source_rgb_dir is not None and is_packed_stream(args.source_rgb_dir):
        volume = open_volume(args.source_rgb_dir, color_space="rgb")
        height, width = (int(value) for value in volume.frame_shape[:2])
        return len(volume), height, width
    cap = cv2.VideoCapture(str(args.rgb_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {args.rgb_mp4}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return total, height, width


def _reuse_packed_masks(src: Path, dst: Path) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    meta = src / "human_mask_metadata.json"
    if not meta.is_file():
        raise FileNotFoundError(meta)
    human = src / "human_mask"
    conf = src / "mask_confidence"
    if not is_packed_stream(human) or not is_packed_stream(conf):
        raise FileNotFoundError(f"packed human masks missing under {src}")
    dst.mkdir(parents=True, exist_ok=True)
    hardlink_packed(human, dst / "human_mask", force=True)
    hardlink_packed(conf, dst / "mask_confidence", force=True)
    (dst / "human_mask_metadata.json").write_text(meta.read_text(encoding="utf-8"))
    print(json.dumps({"status": "reused", "from": str(src), "to": str(dst)}), flush=True)


def main() -> None:
    args = _args()
    if args.reuse_from is not None:
        _reuse_packed_masks(args.reuse_from, args.out_dir)
        return
    total, height, width = _probe_rgb_size(args)
    if width <= 0 or height <= 0:
        raise RuntimeError("RGB video has invalid dimensions")
    start = max(0, int(args.start_frame))
    end = total if args.max_frames is None else min(total, start + int(args.max_frames))
    if end <= start:
        raise RuntimeError(f"empty frame range [{start}, {end})")
    n = end - start
    left, right, lv, rv = _read_keypoints(args.annotation_h5, start, end)
    if len(left) != n:
        raise RuntimeError(f"keypoint count {len(left)} != requested {n}")

    arm_prompt, body_prompts = _active_prompts(args)
    masks, backend_meta = _try_efficientsam3(
        args, [arm_prompt, *body_prompts], start, n, (height, width)
    )
    out = args.out_dir
    human_dir = out / "human_mask"
    conf_dir = out / "mask_confidence"
    human_vol = create_volume(human_dir, n, (height, width), np.uint8, fill=0)
    conf_vol = create_volume(conf_dir, n, (height, width), np.float32, fill=0)
    fractions: list[float] = []
    fallback_frames: list[int] = []
    keypoint_rescue_frames: list[int] = []
    sam_only_frames: list[int] = []
    empty_final_frames: list[int] = []
    empty_body_frames: list[int] = []
    for local in range(n):
        support = np.maximum(
            _support_mask(left[local], bool(lv[local]), (height, width), args),
            _support_mask(right[local], bool(rv[local]), (height, width), args),
        )
        if masks is None:
            mask = support
            confidence = 0.35 if mask.any() else 0.0
            fallback_frames.append(start + local)
        else:
            arm_raw = masks[arm_prompt][local]
            body_raws = [masks[prompt][local] for prompt in body_prompts]
            mask = compose_human_mask(
                arm_raw,
                body_raws,
                support,
                min_area_fraction=float(args.body_min_area_fraction),
                object_protect_px=int(args.object_protect_px),
                body_dilation_px=int(args.body_dilation_px),
            )
            if not np.any(arm_raw) and np.any(mask):
                keypoint_rescue_frames.append(start + local)
            if np.any(arm_raw) and not np.any(support):
                sam_only_frames.append(start + local)
            if body_prompts and not any(np.any(raw) for raw in body_raws):
                empty_body_frames.append(start + local)
            confidence = 0.85 if np.any(arm_raw) or any(np.any(raw) for raw in body_raws) else (
                0.35 if mask.any() else 0.0
            )
        if not np.any(mask):
            empty_final_frames.append(start + local)
        human_vol[local] = (mask > 127).astype(np.uint8) * 255
        conf_vol[local] = np.full((height, width), confidence, dtype=np.float32)
        fractions.append(float((mask > 127).mean()))
    human_vol.flush()
    conf_vol.flush()
    meta = {
        "schema_version": 1,
        "mask_kind": "human_body" if body_prompts else "human_hands_and_arms_only",
        "source_video": str(args.rgb_mp4.resolve()),
        "annotation_h5": str(args.annotation_h5.resolve()),
        "start_frame": start,
        "num_frames": n,
        "size": [width, height],
        "prompt": arm_prompt,
        "body_prompts": body_prompts,
        "backend": backend_meta,
        "fallback_frames": fallback_frames,
        "keypoint_rescue_frames": keypoint_rescue_frames,
        "sam_only_frames": sam_only_frames,
        "empty_final_frames": empty_final_frames,
        "empty_body_frames": empty_body_frames,
        "fallback_fraction": float(len(fallback_frames) / max(1, n)),
        "mask_fraction_mean": float(np.mean(fractions)) if fractions else 0.0,
        "moving_objects_policy": "preserve_nonhuman_objects",
        "keypoint_constraint": "WiLoR_2d_component_intersection",
        "confidence_source": "model_component_or_keypoint_fallback",
    }
    (out / "human_mask_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
