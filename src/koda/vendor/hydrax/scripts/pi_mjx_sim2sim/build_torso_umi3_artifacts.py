#!/usr/bin/env python3
"""Build isolated Torso-UMI-3 policy artifacts from a ScaleBFM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("KODA_HOME", str(SCRIPT_DIR.parents[6])))
sys.path.insert(0, str(SCRIPT_DIR))

from load_eager_policy import load_eager_policy  # noqa: E402


ASSET_ROOT = Path(os.environ.get("KODA_ASSETS", str(WORKSPACE / "assets"))).expanduser().resolve()
CHECKPOINT = Path(os.environ.get("KODA_SCALEBFM_CHECKPOINT", str(ASSET_ROOT / "models/scalebfm/model_22200.pt")))
TEMPLATE_METADATA = SCRIPT_DIR / "artifacts/model_22200_torchscript_metadata.json"
OUTPUT_DIR = Path(os.environ.get("KODA_ARTIFACT_DIR", str(WORKSPACE / ".cache/artifacts_torso_umi3")))
MODE_NAME = "Torso-UMI-3"
MODE_INDEX = 0
TRACKED_NAMES = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
TRACKED_SLOTS = (7, 10, 13)


class ScriptableTaskEmbedder(torch.nn.Module):
    """TorchScript form of TaskEmbedder for this non-reduced checkpoint."""

    def __init__(self, task_projection: torch.nn.Module):
        super().__init__()
        self.task_projection = task_projection

    def forward(self, task_obs: torch.Tensor) -> torch.Tensor:
        return self.task_projection(task_obs)


class ScriptableRMSNorm(torch.nn.Module):
    """Numerically equivalent RMSNorm with an unambiguous scripted norm call."""

    def __init__(self, original: torch.nn.Module):
        super().__init__()
        self.scale = original.scale
        self.eps = float(original.eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(values, p=2.0, dim=-1, keepdim=True) / (values.size(-1) ** 0.5)
        return self.scale * values / (norm + self.eps)


def make_scriptable(policy: torch.nn.Module) -> torch.nn.Module:
    policy.task_embedder = ScriptableTaskEmbedder(policy.task_embedder.task_projection)
    policy.final_norm = ScriptableRMSNorm(policy.final_norm)
    for block in policy.transformer_blocks:
        block.rmsnorm1 = ScriptableRMSNorm(block.rmsnorm1)
        block.rmsnorm2 = ScriptableRMSNorm(block.rmsnorm2)
        block.rmsnorm3 = ScriptableRMSNorm(block.rmsnorm3)
        block.cond_norm = ScriptableRMSNorm(block.cond_norm)
    return policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_inputs(meta: dict, batch: int) -> tuple[torch.Tensor, ...]:
    context = int(meta["history_buffer_size"])
    future = len(meta["future_idx"])
    selected = len(meta["selected_body_names"])
    root_quat = torch.zeros(batch, context, 4)
    root_quat[..., 0] = 1.0
    target_rot = torch.zeros(batch, future, selected, 4)
    target_rot[..., 0] = 1.0
    return (
        root_quat,
        torch.zeros(batch, context, 3),
        torch.zeros(batch, context, 29),
        torch.zeros(batch, context, 29),
        torch.zeros(batch, context, 29),
        torch.zeros(batch, future, selected, 3),
        target_rot,
        torch.full((batch,), MODE_INDEX, dtype=torch.long),
        torch.as_tensor(meta["future_idx"], dtype=torch.long)[None, :, None].expand(batch, -1, -1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--template-metadata", type=Path, default=TEMPLATE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_name = checkpoint.stem
    template_path = args.template_metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(template_path.read_text())
    selected = list(metadata["selected_body_names"])
    actual_slots = tuple(selected.index(name) for name in TRACKED_NAMES)
    if actual_slots != TRACKED_SLOTS:
        raise ValueError(f"Tracked body slot mismatch: expected {TRACKED_SLOTS}, got {actual_slots}")
    metadata["mode_names"] = [MODE_NAME]
    metadata_path = output_dir / f"{checkpoint_name}_torchscript_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    mode_table = torch.zeros((1, len(selected)), dtype=torch.float32)
    mode_table[0, list(TRACKED_SLOTS)] = 1.0
    mode_table_path = output_dir / "mode_table.pt"
    torch.save(mode_table, mode_table_path)

    checkpoint_data = torch.load(checkpoint, weights_only=False, map_location="cpu")
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_id": f"torso_umi3_{checkpoint_name.replace('_', '')}",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "iteration": int(checkpoint_data.get("iter", -1)),
        },
        "template_metadata": {
            "path": str(template_path),
            "sha256": sha256(template_path),
        },
        "files": {
            "metadata": metadata_path.name,
            "mode_table": mode_table_path.name,
            "torchscript": f"{checkpoint_name}_torchscript.pt",
        },
        "policy": {
            "mode_name": MODE_NAME,
            "mode_index": MODE_INDEX,
            "selected_body_slots": list(TRACKED_SLOTS),
            "selected_body_names": list(TRACKED_NAMES),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    policy, _ = load_eager_policy(
        output_dir,
        device="cpu",
        artifact_manifest=manifest_path,
    )
    policy.eval()
    if metadata["policy_architecture"]["reduced_task_dim"] is not None:
        raise ValueError("ScriptableTaskEmbedder only supports the non-reduced checkpoint")
    policy = make_scriptable(policy)
    with torch.inference_mode():
        for batch in (1, 8):
            target, action = policy(*smoke_inputs(metadata, batch))
            if target.shape != (batch, 29) or action.shape != (batch, 29):
                raise ValueError(
                    f"Unexpected eager output at batch {batch}: {target.shape}, {action.shape}"
                )

    scripted = torch.jit.freeze(torch.jit.script(policy))
    with torch.inference_mode():
        target, action = scripted(*smoke_inputs(metadata, 4))
    if target.shape != (4, 29) or action.shape != (4, 29):
        raise ValueError(f"Unexpected TorchScript output: {target.shape}, {action.shape}")
    torchscript_path = output_dir / manifest["files"]["torchscript"]
    scripted.save(str(torchscript_path))

    manifest["artifact_sha256"] = {
        "metadata": sha256(metadata_path),
        "mode_table": sha256(mode_table_path),
        "torchscript": sha256(torchscript_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
