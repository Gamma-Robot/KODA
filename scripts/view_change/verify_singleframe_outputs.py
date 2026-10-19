#!/usr/bin/env python3
"""Strict structural/pixel checks for the offline single-frame experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from frame_volume import as_mask, as_rgb, is_packed_stream, open_volume
except ImportError:  # pragma: no cover - package-style import
    from .frame_volume import as_mask, as_rgb, is_packed_stream, open_volume


STREAMS = (
    "rgb_geometry", "depth", "valid", "hole", "confidence", "source_count",
    "robot_mask", "paste_mask", "rgb_final", "synthetic", "provenance",
)


def _open_stream(directory: Path, *, color_space: str | None = None):
    return open_volume(directory, color_space=color_space)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-30hz", type=Path, required=True)
    p.add_argument("--final-50hz", type=Path, required=True)
    p.add_argument("--timeline", type=Path, required=True)
    p.add_argument("--warp-status", type=Path, required=True)
    p.add_argument("--diffueraser-status", type=Path, required=True)
    p.add_argument("--expected-source", type=int, required=True)
    p.add_argument("--expected-target", type=int, required=True)
    p.add_argument("--target-calibration", type=Path, required=True)
    p.add_argument("--actual-target-calibration", type=Path, required=True)
    p.add_argument("--warp-meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    expected_camera = json.loads(args.target_calibration.read_text(encoding="utf-8"))
    actual_camera = json.loads(
        args.actual_target_calibration.read_text(encoding="utf-8")
    )
    warp_meta = json.loads(args.warp_meta.read_text(encoding="utf-8"))
    target_width = int(expected_camera["width"])
    target_height = int(expected_camera["height"])
    expected_k = np.asarray(expected_camera["K"], dtype=np.float64)
    actual_k = np.asarray(actual_camera["K"], dtype=np.float64)
    warp_k = np.asarray(warp_meta["target_K"], dtype=np.float64)
    if (int(actual_camera["width"]), int(actual_camera["height"])) != (
        target_width,
        target_height,
    ):
        raise ValueError("rendered target dimensions differ from target calibration")
    if tuple(map(int, warp_meta["target_size"])) != (target_width, target_height):
        raise ValueError("warp dimensions differ from target calibration")
    if not np.allclose(actual_k, expected_k, rtol=0.0, atol=1e-9):
        raise ValueError("rendered target K differs from target calibration")
    if not np.allclose(warp_k, expected_k, rtol=0.0, atol=1e-9):
        raise ValueError("warp target K differs from target calibration")

    timeline = json.loads(args.timeline.read_text())
    source_to_target = np.asarray(timeline["source_to_target"]["target_indices"], dtype=np.int64)
    output_map = np.asarray(timeline["output_50hz"]["source_indices"], dtype=np.int64)
    if len(source_to_target) != args.expected_source or len(output_map) != args.expected_target:
        raise ValueError("timeline frame counts do not match requested counts")
    if np.any(np.diff(output_map) < 0) or np.any(
        np.asarray(timeline["source"]["timestamps_relative_s"])[output_map]
        > np.asarray(timeline["target"]["timestamps_relative_s"]) + 1e-9
    ):
        raise ValueError("50 Hz mapping violates monotonic/past-only contract")

    processed_vols = {
        stream: _open_stream(
            args.processed_30hz / stream,
            color_space="rgb" if stream in ("rgb_geometry", "rgb_final") else None,
        )
        for stream in STREAMS
    }
    processed_counts = {stream: len(vol) for stream, vol in processed_vols.items()}
    if set(processed_counts.values()) != {args.expected_source}:
        raise ValueError(f"processed stream counts mismatch: {processed_counts}")

    final_mp4 = args.final_50hz / "rgb_final_50hz.mp4"
    if not final_mp4.is_file() or final_mp4.stat().st_size <= 0:
        raise ValueError("missing rgb_final_50hz.mp4")
    final_counts: dict[str, int | str] = {"rgb_final_50hz.mp4": args.expected_target}
    expected_shape = (target_height, target_width)
    if any(is_packed_stream(args.final_50hz / stream) for stream in STREAMS):
        final_vols = {
            stream: _open_stream(
                args.final_50hz / stream,
                color_space="rgb" if stream in ("rgb_geometry", "rgb_final") else None,
            )
            for stream in STREAMS
            if is_packed_stream(args.final_50hz / stream)
        }
        final_counts = {stream: len(vol) for stream, vol in final_vols.items()}
        if set(final_counts.values()) != {args.expected_target}:
            raise ValueError(f"final stream counts mismatch: {final_counts}")
        for stream, vol in final_vols.items():
            shape = tuple(int(v) for v in vol.frame_shape[:2])
            if shape != expected_shape:
                raise ValueError(
                    f"final {stream} frame has shape {shape}, expected {expected_shape}"
                )

    geom_vol = processed_vols["rgb_geometry"]
    final_vol = processed_vols["rgb_final"]
    synth_vol = processed_vols["synthetic"]
    prov_vol = processed_vols["provenance"]
    if geom_vol.packed and final_vol.packed and synth_vol.packed and prov_vol.packed:
        geom = as_rgb(geom_vol.array)
        final = as_rgb(final_vol.array)
        synth = as_mask(synth_vol.array)
        prov = np.asarray(prov_vol.array)
        if prov.ndim == 4:
            prov = prov[..., 0]
        if geom.shape[1:3] != expected_shape or final.shape[1:3] != expected_shape:
            raise ValueError("processed RGB spatial shape mismatch")
        fused = int(np.count_nonzero(prov == 2))
        changed_outside_synthetic = int(
            np.count_nonzero(np.any(geom != final, axis=-1) & ~synth)
        )
        if np.any(~np.any(final.reshape(final.shape[0], -1), axis=1)):
            blank = int(np.flatnonzero(~np.any(final.reshape(final.shape[0], -1), axis=1))[0])
            raise ValueError(f"blank final RGB frame {blank}")
    else:
        fused = 0
        changed_outside_synthetic = 0
        for i in range(args.expected_source):
            geom = as_rgb(geom_vol[i])
            final = as_rgb(final_vol[i])
            synth = as_mask(synth_vol[i])
            prov = np.asarray(prov_vol[i])
            if prov.ndim == 3:
                prov = prov[..., 0]
            if geom.shape[:2] != expected_shape or final.shape[:2] != expected_shape:
                raise ValueError(
                    f"processed RGB frame {i} has shape {geom.shape[:2]}/{final.shape[:2]}, "
                    f"expected {expected_shape}"
                )
            fused += int(np.count_nonzero(prov == 2))
            changed_outside_synthetic += int(
                np.count_nonzero(np.any(geom != final, axis=-1) & ~synth)
            )
            if not np.any(final):
                raise ValueError(f"blank final RGB frame {i}")
    if fused:
        raise ValueError(f"single-frame result contains {fused} fused pixels")
    if changed_outside_synthetic:
        raise ValueError(f"DiffuEraser changed {changed_outside_synthetic} observed/robot pixels")

    status_rows = [json.loads(line) for line in args.warp_status.read_text().splitlines() if line.strip()]
    if len(status_rows) != args.expected_source:
        raise ValueError("warp status frame count mismatch")
    status_target = [int(row["target_index"]) for row in status_rows]
    if status_target != source_to_target.tolist():
        raise ValueError("warp target indices differ from timeline source_to_target")
    diffu = json.loads(args.diffueraser_status.read_text())
    if not bool(diffu.get("success")) or diffu.get("backend") != "DiffuEraser":
        raise ValueError(f"DiffuEraser was not verified: {diffu}")
    native_run = diffu.get("native_run", {})
    prior_diffusion = diffu.get("priori_diffusion", {})
    if (
        "Normal CFG" not in str(native_run.get("checkpoint", ""))
        or int(native_run.get("num_inference_steps", 0)) <= 0
        or float(native_run.get("guidance_scale", 0.0)) <= 0.0
    ):
        raise ValueError(f"DiffuEraser diffusion contract was not verified: {native_run}")
    if (
        int(prior_diffusion.get("masked_pixels", 0)) > 0
        and float(prior_diffusion.get("changed_ratio", 0.0)) <= 0.0
    ):
        raise ValueError("DiffuEraser result is pixel-identical to ProPainter inside holes")

    report = {
        "status": "pass",
        "source_frames": args.expected_source,
        "target_frames": args.expected_target,
        "source_to_target_match": True,
        "past_only_mapping": True,
        "fused_pixels": fused,
        "changed_outside_synthetic": changed_outside_synthetic,
        "target_camera": {
            "width": target_width,
            "height": target_height,
            "K": expected_k.tolist(),
            "projection_contract": expected_camera.get("projection_contract"),
        },
        "diffueraser": diffu,
        "processed_counts": processed_counts,
        "final_counts": final_counts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
