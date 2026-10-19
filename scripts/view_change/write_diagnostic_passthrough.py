#!/usr/bin/env python3
"""Write an explicitly invalid source-RGB diagnostic when geometry is gated out."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--reason", required=True)
    args = p.parse_args()
    out = args.out_dir / "diagnostic_passthrough"
    rgb_dir = out / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(args.video))
    writer = imageio.get_writer(str(out / "diagnostic.mp4"), fps=30.0, codec="libx264", macro_block_size=1)
    try:
        for i in range(args.count):
            frame = np.asarray(reader.get_data(i))[:, :, :3]
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            writer.append_data(frame)
            cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        reader.close()
        writer.close()
    (out / "diagnostic_manifest.json").write_text(
        json.dumps(
            {
                "status": "diagnostic_passthrough_invalid",
                "formal_view_change": False,
                "reason": args.reason,
                "frame_count": args.count,
                "size": [args.width, args.height],
                "source_video": str(args.video.resolve()),
                "note": "source RGB is resized for diagnosis only; no target geometry is claimed",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
