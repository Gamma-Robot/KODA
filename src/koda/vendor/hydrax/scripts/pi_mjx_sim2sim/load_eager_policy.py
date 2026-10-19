#!/usr/bin/env python3
"""Load / rebuild eager batched π (TorchScript export optional; CEM uses eager)."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from _onboard_policy_lib import (
    HumanoidTransformer,
    HumanoidTransformerPolicyWrapperWithMode,
    TaskEmbedder,
    build_mode_mappings,
    parse_xml,
)


_RESOURCE_ROOT = Path(__file__).resolve().parents[3]
_SCALEBFM_ROOT = Path(os.environ.get("KODA_SCALEBFM_ROOT", str(_RESOURCE_ROOT / "scalebfm"))).expanduser()
_DEFAULT_CHECKPOINT = Path(os.environ.get(
    "KODA_SCALEBFM_CHECKPOINT",
    str(_SCALEBFM_ROOT / "ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt"),
)).expanduser()
_DEFAULT_ROBOT_XML = Path(os.environ.get(
    "KODA_SCALEBFM_ROBOT_XML",
    str(_SCALEBFM_ROOT / "ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"),
)).expanduser()
if not _DEFAULT_ROBOT_XML.is_file():
    _DEFAULT_ROBOT_XML = _SCALEBFM_ROOT / "scalebridge/data/robot/g1_29dof/g1_29dof.xml"


@dataclass(frozen=True)
class PolicyArtifactSpec:
    """Resolved files for one policy export.

    The legacy artifact directory has no manifest.  Keeping its resolution
    here makes the optional manifest path additive and preserves the existing
    model_22200 behavior.
    """

    root: Path
    metadata_path: Path
    mode_table_path: Path
    checkpoint_path: Path
    torchscript_path: Path
    mode_index: int
    manifest_path: Path | None = None
    checkpoint_sha256: str | None = None


def _resolve_relative(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else base / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_policy_artifacts(
    artifacts: str | Path,
    artifact_manifest: str | Path | None = None,
    *,
    verify_checkpoint: bool = True,
) -> PolicyArtifactSpec:
    """Resolve legacy files or an explicit versioned artifact manifest."""

    root = Path(artifacts).expanduser()
    if artifact_manifest is None:
        return PolicyArtifactSpec(
            root=root,
            metadata_path=root / "model_22200_torchscript_metadata.json",
            mode_table_path=root / "mode_table.pt",
            checkpoint_path=_DEFAULT_CHECKPOINT,
            torchscript_path=root / "model_22200_torchscript.pt",
            mode_index=1,
        )

    manifest_path = Path(artifact_manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported policy artifact manifest schema: {manifest.get('schema_version')}"
        )

    files = manifest["files"]
    checkpoint = manifest["checkpoint"]
    metadata_path = _resolve_relative(files["metadata"], root)
    mode_table_path = _resolve_relative(files["mode_table"], root)
    torchscript_path = _resolve_relative(files["torchscript"], root)
    checkpoint_path = _resolve_relative(checkpoint["path"], manifest_path.parent)
    expected_sha256 = str(checkpoint["sha256"]).lower()
    if verify_checkpoint:
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Checkpoint SHA256 mismatch for "
                f"{checkpoint_path}: expected {expected_sha256}, got {actual_sha256}"
            )

    return PolicyArtifactSpec(
        root=root,
        metadata_path=metadata_path,
        mode_table_path=mode_table_path,
        checkpoint_path=checkpoint_path,
        torchscript_path=torchscript_path,
        mode_index=int(manifest["policy"]["mode_index"]),
        manifest_path=manifest_path,
        checkpoint_sha256=expected_sha256,
    )


def load_eager_policy(
    artifacts: str | Path,
    device: str | torch.device = "cpu",
    *,
    artifact_manifest: str | Path | None = None,
    verify_checkpoint: bool = True,
):
    spec = resolve_policy_artifacts(
        artifacts,
        artifact_manifest,
        verify_checkpoint=verify_checkpoint,
    )
    device = torch.device(device)
    meta = json.loads(spec.metadata_path.read_text())
    architecture = meta["policy_architecture"]
    ckpt = spec.checkpoint_path
    xml_path = str(_DEFAULT_ROBOT_XML)

    actor = HumanoidTransformer(
        prop_obs_dim=architecture["prop_obs_dim"],
        action_dim=architecture["action_dim"],
        output_dim=architecture["output_dim"],
        embed_dim=architecture["embedding_dim"],
        num_heads=architecture["num_heads"],
        ff_dim=architecture["ff_dim"],
        num_layers=architecture["num_layers"],
    )
    actor_task_embedder = TaskEmbedder(
        task_obs_dim=architecture["task_obs_dim"],
        embedding_dim=architecture["embedding_dim"],
        reduced_task_dim=architecture["reduced_task_dim"],
        hidden_dims=architecture["task_embedder_hidden_dims"],
    )
    loaded = torch.load(ckpt, weights_only=False, map_location="cpu")
    model_dict = loaded["model_state_dict"]
    actor.load_state_dict({k[len("actor.") :]: v for k, v in model_dict.items() if k.startswith("actor.")})
    actor_task_embedder.load_state_dict(
        {
            k[len("actor_task_embedder.") :]: v
            for k, v in model_dict.items()
            if k.startswith("actor_task_embedder.")
        }
    )

    mode_vector = torch.load(spec.mode_table_path, map_location="cpu")
    mode_mappings = build_mode_mappings(
        mode_vector,
        meta["mode_feature_dims"],
        with_time=meta["mode_mapping_with_time"],
    )
    default_dof_pos = torch.tensor([meta["default_dof_pos"]], dtype=torch.float)
    action_scale = torch.tensor([meta["action_scale"]], dtype=torch.float)
    context_len = int(meta["history_buffer_size"])
    future_len = len(meta["future_idx"])

    body_names, joint_names, parent_indices, joint_axis, local_translation, local_rotation = parse_xml(
        xml_path, "cpu"
    )
    local_rotation = local_rotation / local_rotation.norm(dim=-1, keepdim=True)
    selected = list(meta["selected_body_names"])
    selected_body_indices = torch.tensor([body_names.index(n) for n in selected], dtype=torch.long)
    robot_joint_names = list(meta["joint_names"])
    lab_to_xml = torch.tensor([robot_joint_names.index(n) for n in joint_names], dtype=torch.long)

    wrapper = HumanoidTransformerPolicyWrapperWithMode(
        actor,
        actor_task_embedder,
        mode_mappings,
        mode_vector,
        default_dof_pos,
        action_scale,
        context_len,
        future_len,
        local_translation,
        local_rotation,
        parent_indices,
        joint_axis,
        selected_body_indices,
        lab_to_xml,
    )
    wrapper.eval()
    wrapper.to(device)
    return wrapper, meta
