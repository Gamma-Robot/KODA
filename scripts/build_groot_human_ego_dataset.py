#!/usr/bin/env python3
"""Build a GR00T dataset that keeps groot_human parquet/meta and swaps in 50 Hz ego RGB."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

FPS = 50
VIDEO_WIDTH = 672
VIDEO_HEIGHT = 376
VIDEO_KEY = "observation.images.ego_view"


def past_only_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Zero-order hold: last source frame with t_src <= t_tgt."""
    src = np.asarray(source_times, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target_times, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(src, tgt, side="right") - 1
    return np.clip(idx, 0, src.shape[0] - 1).astype(np.int64)


def encode_episode(job: dict[str, str | int]) -> dict[str, object]:
    ego_video = Path(str(job["ego_video"]))
    kinematics_path = Path(str(job["kinematics"]))
    destination = Path(str(job["output_video"]))
    rows = int(job["length"])
    session_id = str(job["session_id"])
    episode_index = int(job["episode_index"])

    source = np.load(kinematics_path)
    source_times = np.asarray(source["contact_30hz_timestamps"], dtype=np.float64)
    target_times = np.asarray(source["timestamps"], dtype=np.float64)[:rows]
    mapping = past_only_indices(source_times, target_times)

    capture = cv2.VideoCapture(str(ego_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {ego_video}")
    native: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            native.append(
                cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)
            )
    finally:
        capture.release()
    if len(native) != int(source_times.shape[0]):
        raise ValueError(
            f"{session_id}: ego frames {len(native)} != 30 Hz timestamps {source_times.shape[0]}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (VIDEO_WIDTH, VIDEO_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {tmp}")
    try:
        for source_index in mapping.tolist():
            writer.write(native[int(source_index)])
    finally:
        writer.release()
    tmp.replace(destination)
    return {
        "episode_index": episode_index,
        "session_id": session_id,
        "frames": rows,
        "unique_source_frames": int(np.unique(mapping).size),
        "output_video": str(destination),
    }


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groot-root", type=Path, default=Path("outputs/groot_base"))
    parser.add_argument("--ego-root", type=Path, default=Path("outputs/converted"))
    parser.add_argument("--source-root", type=Path, default=Path("outputs/view_change"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/groot"))
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groot = args.groot_root.expanduser().resolve()
    ego = args.ego_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output == groot:
        raise ValueError("refuse to overwrite the source groot dataset")

    episodes = [
        json.loads(line)
        for line in (groot / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (output / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "modality.json", "tasks.jsonl", "episodes.jsonl"):
        shutil.copy2(groot / "meta" / name, output / "meta" / name)

    parquet_src = groot / "data" / "chunk-000"
    parquet_dst = output / "data" / "chunk-000"
    parquet_dst.mkdir(parents=True, exist_ok=True)
    for parquet in sorted(parquet_src.glob("episode_*.parquet")):
        hardlink_or_copy(parquet, parquet_dst / parquet.name)

    jobs = []
    for episode in episodes:
        session_id = str(episode["session_id"])
        episode_index = int(episode["episode_index"])
        jobs.append(
            {
                "session_id": session_id,
                "episode_index": episode_index,
                "length": int(episode["length"]),
                "ego_video": str(ego / session_id / "rgb.mp4"),
                "kinematics": str(
                    source_root / session_id / "final_50hz" / "kinematics.npz"
                ),
                "output_video": str(
                    output
                    / "videos"
                    / "chunk-000"
                    / VIDEO_KEY
                    / f"episode_{episode_index:06d}.mp4"
                ),
            }
        )

    reports = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(encode_episode, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            job = futures[future]
            report = future.result()
            reports[int(report["episode_index"])] = report
            done += 1
            print(
                f"[{done}/{len(jobs)}] episode {report['episode_index']:03d} "
                f"{report['session_id']} frames={report['frames']} "
                f"unique_src={report['unique_source_frames']}",
                flush=True,
            )

    summary_src = json.loads((groot / "export_summary.json").read_text(encoding="utf-8"))
    assignments = []
    for old, job, report in zip(summary_src["assignments"], jobs, reports, strict=True):
        row = dict(old)
        row["source_video"] = job["ego_video"]
        row["output_parquet"] = str(parquet_dst / f"episode_{int(job['episode_index']):06d}.parquet")
        row["output_video"] = str(job["output_video"])
        row["video_fps"] = FPS
        row["video_size"] = [VIDEO_HEIGHT, VIDEO_WIDTH, 3]
        row["unique_source_frames"] = report["unique_source_frames"]
        assignments.append(row)
    summary = dict(summary_src)
    summary["output_root"] = str(output)
    summary["ego_root"] = str(ego)
    summary["video_policy"] = "egocentric rgb.mp4 past-only ZOH resampled to 50 Hz, 376x672"
    summary["assignments"] = assignments
    (output / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
