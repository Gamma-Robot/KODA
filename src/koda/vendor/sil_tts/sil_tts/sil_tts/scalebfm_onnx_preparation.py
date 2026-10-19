"""Export Torso-UMI-3 ScaleBFM to a batched float32 ONNX policy.

The graph keeps only float observations and raw ``action`` used to form PD
targets. Mode index and time offsets are baked in. SIL persists kinematics,
not action or motion-token traces.

TorchScript control-flow cannot be exported, so this rebuilds the eager
module from hydrax artifacts + checkpoint, replaces MultiheadAttention /
RoPE scatter with matmul kernels, then traces a dynamic-batch graph.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from sil_tts.artifacts import default_repo_root, load_lock, resolve, sha256_file, verify

SOURCE_ID = "scalebfm-torso-umi3-policy.torchscript"
POLICY_ID = "scalebfm-torso-umi3-policy.batched"
FLOAT_INPUT_NAMES = (
    "root_quat_buffer",
    "base_ang_vel_buffer",
    "dof_pos_buffer",
    "dof_vel_buffer",
    "last_action_buffer",
    "target_body_pos_future_to_robot_base",
    "target_body_rot_future_to_robot_base",
)
OUTPUT_NAME = "action"
_DUMMY_SHAPES = {
    "root_quat_buffer": (2, 3, 4),
    "base_ang_vel_buffer": (2, 3, 3),
    "dof_pos_buffer": (2, 3, 29),
    "dof_vel_buffer": (2, 3, 29),
    "last_action_buffer": (2, 3, 29),
    "target_body_pos_future_to_robot_base": (2, 6, 14, 3),
    "target_body_rot_future_to_robot_base": (2, 6, 14, 4),
}
_WORKSPACE = Path(os.environ.get("KODA_HOME", Path(__file__).resolve().parents[6]))
_HYDRAX_SCRIPTS = (
    default_repo_root().parent / "hydrax" / "scripts" / "pi_mjx_sim2sim",
    default_repo_root().parent
    / "ego2humanoid" / "resources" / "hydrax" / "scripts" / "pi_mjx_sim2sim",
    _WORKSPACE / "hydrax" / "scripts" / "pi_mjx_sim2sim",
    _WORKSPACE / "ego2humanoid" / "resources" / "hydrax" / "scripts" / "pi_mjx_sim2sim",
)


def _set_batch(value_info) -> None:
    dims = value_info.type.tensor_type.shape.dim
    if not dims:
        raise ValueError(f"ONNX value {value_info.name!r} has no batch dimension")
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"


def _hydrax_scripts() -> Path:
    for path in _HYDRAX_SCRIPTS:
        if (path / "load_eager_policy.py").is_file():
            return path
    raise FileNotFoundError("hydrax ScaleBFM eager loader is missing")


def _dummy_inputs(torch):
    tensors = []
    for name in FLOAT_INPUT_NAMES:
        tensor = torch.zeros(_DUMMY_SHAPES[name], dtype=torch.float32)
        if name in {"root_quat_buffer", "target_body_rot_future_to_robot_base"}:
            tensor[..., 0] = 1.0
        tensors.append(tensor)
    return tuple(tensors)


def _quat_apply(quat, vec):
    xyz = quat[..., 1:]
    tangent = xyz.cross(vec, dim=-1) * 2
    return vec + quat[..., :1] * tangent + xyz.cross(tangent, dim=-1)


def _quat_apply_inverse(quat, vec):
    xyz = quat[..., 1:]
    tangent = xyz.cross(vec, dim=-1) * 2
    return vec - quat[..., :1] * tangent + xyz.cross(tangent, dim=-1)


def _quat_mul(first, second):
    import torch

    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_mul_inverse_right(first, second):
    import torch

    w2, x2, y2, z2 = second.unbind(-1)
    inverse = torch.stack((w2, -x2, -y2, -z2), dim=-1)
    return _quat_mul(first, inverse)


def _replace_multihead_attention(policy) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _OrtFriendlyMultiheadAttention(nn.Module):
        """Matmul attention with the same weights as ``nn.MultiheadAttention``."""

        def __init__(self, module: nn.MultiheadAttention) -> None:
            super().__init__()
            if not module._qkv_same_embed_dim or module.bias_k is not None:
                raise ValueError("ScaleBFM attention layout is not the packed QKV form")
            self.num_heads = int(module.num_heads)
            self.head_dim = int(module.head_dim)
            self.embed_dim = int(module.embed_dim)
            self.in_proj_weight = nn.Parameter(module.in_proj_weight.detach().clone())
            self.in_proj_bias = nn.Parameter(module.in_proj_bias.detach().clone())
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            with torch.no_grad():
                self.out_proj.weight.copy_(module.out_proj.weight)
                self.out_proj.bias.copy_(module.out_proj.bias)
            self.scale = self.head_dim ** -0.5

        def _heads(self, projected, batch, length):
            return projected.view(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        ):
            del key_padding_mask, need_weights, average_attn_weights, is_causal
            batch, query_len, _ = query.shape
            key_len = key.shape[1]
            weight_q, weight_k, weight_v = self.in_proj_weight.chunk(3)
            bias_q, bias_k, bias_v = self.in_proj_bias.chunk(3)
            query_heads = self._heads(
                F.linear(query, weight_q, bias_q), batch, query_len
            )
            key_heads = self._heads(F.linear(key, weight_k, bias_k), batch, key_len)
            value_heads = self._heads(F.linear(value, weight_v, bias_v), batch, key_len)
            scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                mask = attn_mask
                while mask.ndim < scores.ndim:
                    mask = mask.unsqueeze(0)
                scores = scores.masked_fill(mask, -1.0e9)
            context = torch.matmul(torch.softmax(scores, dim=-1), value_heads)
            merged = context.transpose(1, 2).contiguous().view(
                batch, query_len, self.embed_dim
            )
            return self.out_proj(merged), None

    for block in policy.transformer_blocks:
        if isinstance(block.self_attention, nn.MultiheadAttention):
            block.self_attention = _OrtFriendlyMultiheadAttention(block.self_attention)
        if isinstance(block.cross_attention, nn.MultiheadAttention):
            block.cross_attention = _OrtFriendlyMultiheadAttention(block.cross_attention)
    _patch_rope(policy)


def _patch_rope(policy) -> None:
    import torch

    seq_len = 6
    for block in policy.transformer_blocks:
        rope = block.rope
        position = torch.arange(
            seq_len, dtype=torch.float32, device=rope.inv_freq.device
        )
        freqs = torch.outer(position, rope.inv_freq.detach())
        rope.register_buffer(
            "_cos_freqs", torch.cos(freqs).unsqueeze(0), persistent=False
        )
        rope.register_buffer(
            "_sin_freqs", torch.sin(freqs).unsqueeze(0), persistent=False
        )

        def forward(self, x):
            even = x[..., 0::2]
            odd = x[..., 1::2]
            rotated_even = even * self._cos_freqs - odd * self._sin_freqs
            rotated_odd = even * self._sin_freqs + odd * self._cos_freqs
            return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

        rope.forward = forward.__get__(rope, type(rope))


def _bind_dynamic_batch_forward(policy) -> None:
    """Keep batch size symbolic and drop int64 / control-flow from the graph."""

    import torch

    policy.register_buffer(
        "_baked_time_offsets",
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        persistent=False,
    )

    def forward(
        self,
        root_quat_buffer,
        base_ang_vel_buffer,
        dof_pos_buffer,
        dof_vel_buffer,
        last_action_buffer,
        target_body_pos_future_to_robot_base,
        target_body_rot_future_to_robot_base,
    ):
        batch = root_quat_buffer.shape[0]
        grav = self.gravity_vec.expand(batch, -1, -1)
        projected_gravity_buffer = _quat_apply_inverse(root_quat_buffer, grav)
        dof_pos_rel_buffer = dof_pos_buffer - self.default_dof_pos
        prop_obs = torch.cat(
            [
                projected_gravity_buffer,
                base_ang_vel_buffer,
                dof_pos_rel_buffer,
                dof_vel_buffer * 0.05,
            ],
            dim=-1,
        )
        prop_token = self.prop_embedder(prop_obs)
        action_token = self.action_embedder(last_action_buffer)
        empty = self.empty_embedding[0, 0].expand(batch, -1)
        x = torch.stack(
            (
                prop_token[:, 0],
                action_token[:, 1],
                prop_token[:, 1],
                action_token[:, 2],
                prop_token[:, 2],
                empty,
            ),
            dim=1,
        )

        dof_pos = dof_pos_buffer[:, -1]
        half_angles = dof_pos[:, self.lab_to_xml_joint_indices].unsqueeze(-1) / 2
        joint_rot = torch.cat(
            (
                torch.cos(half_angles),
                self.joint_axis.unsqueeze(0) * torch.sin(half_angles),
            ),
            dim=-1,
        )
        nbody = int(self.parent_indices.shape[0])
        positions = [joint_rot.new_zeros((batch, 3))]
        orientations = [
            torch.cat(
                (joint_rot.new_ones((batch, 1)), joint_rot.new_zeros((batch, 3))),
                dim=-1,
            )
        ]
        for index in range(1, nbody):
            j_rot = joint_rot[:, index - 1]
            local_trans = self.local_translation[index - 1 : index].expand(batch, -1)
            local_rot = self.local_rotation[index - 1 : index].expand(batch, -1)
            parent = int(self.parent_indices[index].item())
            parent_pos = positions[parent]
            parent_rot = orientations[parent]
            positions.append(parent_pos + _quat_apply(parent_rot, local_trans))
            orientations.append(_quat_mul(parent_rot, _quat_mul(local_rot, j_rot)))
        body_pos = torch.stack(positions, dim=1)
        body_quat = torch.stack(orientations, dim=1)

        body_pos_to_robot_base = body_pos[:, self.selected_link_indices]
        body_quat_to_robot_base = body_quat[:, self.selected_link_indices]
        target_body_pos_future_rel_to_robot_base = (
            target_body_pos_future_to_robot_base - body_pos_to_robot_base[:, None, :, :]
        )
        tan_vec = self.tan_vec.expand(batch, -1, -1, -1)
        norm_vec = self.norm_vec.expand(batch, -1, -1, -1)
        target_body_rot_future_to_robot_base_tan_norm = torch.cat(
            [
                _quat_apply(target_body_rot_future_to_robot_base, tan_vec),
                _quat_apply(target_body_rot_future_to_robot_base, norm_vec),
            ],
            dim=-1,
        )
        target_body_rot_future_rel_to_robot_base = _quat_mul_inverse_right(
            target_body_rot_future_to_robot_base,
            body_quat_to_robot_base[:, None].expand(
                -1, target_body_rot_future_to_robot_base.shape[1], -1, -1
            ),
        )
        target_body_rot_future_rel_to_robot_base_tan_norm = torch.cat(
            [
                _quat_apply(target_body_rot_future_rel_to_robot_base, tan_vec),
                _quat_apply(target_body_rot_future_rel_to_robot_base, norm_vec),
            ],
            dim=-1,
        )
        time_offsets = self._baked_time_offsets.expand(batch, -1, -1)
        task_obs = torch.cat(
            [
                target_body_pos_future_to_robot_base.flatten(2, 3),
                target_body_pos_future_rel_to_robot_base.flatten(2, 3),
                target_body_rot_future_to_robot_base_tan_norm.flatten(2, 3),
                target_body_rot_future_rel_to_robot_base_tan_norm.flatten(2, 3),
                time_offsets,
            ],
            dim=-1,
        )
        mapping = self.mode_mappings[:1].expand(batch, -1)
        task_obs_masked = task_obs * mapping.unsqueeze(1)
        mode_vec = self.mode_vectors[:1].expand(batch, -1)
        task_input = torch.cat(
            [
                task_obs_masked,
                mode_vec.unsqueeze(1).expand(-1, task_obs_masked.shape[1], -1),
            ],
            dim=-1,
        )
        task_tokens = self.task_embedder(task_input)
        hidden = x
        for transformer_block in self.transformer_blocks:
            hidden = transformer_block(
                hidden, task_tokens, self_attn_mask=self.self_attn_mask
            )
        action = self.projection_head(self.final_norm(hidden)[:, -1, :])
        return action * self.action_scale + self.default_dof_pos, action

    policy.forward = forward.__get__(policy, type(policy))


def _load_eager_policy():
    scripts = _hydrax_scripts()
    artifacts = scripts / "artifacts_torso_umi3"
    manifest = artifacts / "artifact_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"ScaleBFM eager artifacts missing: {manifest}")
    inserted = str(scripts) not in sys.path
    if inserted:
        sys.path.insert(0, str(scripts))
    try:
        from load_eager_policy import load_eager_policy
    finally:
        if inserted:
            sys.path.remove(str(scripts))
            sys.path.insert(0, str(scripts))
    policy, _meta = load_eager_policy(
        artifacts, device="cpu", artifact_manifest=manifest
    )
    policy.eval()
    _replace_multihead_attention(policy)
    _bind_dynamic_batch_forward(policy)
    return policy


def export_scalebfm_batched_onnx(
    torchscript_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Trace the eager Torso-UMI-3 policy into a batched float32 ONNX graph."""

    import onnx
    import torch

    del torchscript_path
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else default_repo_root()
        / "artifacts"
        / "store"
        / "scalebfm"
        / "torso_umi3"
        / "policy.batched.onnx"
    )
    policy = _load_eager_policy()

    class _BakedScaleBfmPolicy(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(
            self,
            root_quat_buffer,
            base_ang_vel_buffer,
            dof_pos_buffer,
            dof_vel_buffer,
            last_action_buffer,
            target_body_pos_future_to_robot_base,
            target_body_rot_future_to_robot_base,
        ):
            _pd_target, action = self.inner(
                root_quat_buffer,
                base_ang_vel_buffer,
                dof_pos_buffer,
                dof_vel_buffer,
                last_action_buffer,
                target_body_pos_future_to_robot_base,
                target_body_rot_future_to_robot_base,
            )
            return action

    wrapper = _BakedScaleBfmPolicy(policy).eval()
    dummy = _dummy_inputs(torch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        candidate = Path(temporary) / destination.name
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                dummy,
                str(candidate),
                input_names=list(FLOAT_INPUT_NAMES),
                output_names=[OUTPUT_NAME],
                dynamic_axes={
                    name: {0: "batch"}
                    for name in (*FLOAT_INPUT_NAMES, OUTPUT_NAME)
                },
                opset_version=17,
                dynamo=False,
            )
        model = onnx.load(str(candidate))
        input_names = [value.name for value in model.graph.input]
        output_names = [value.name for value in model.graph.output]
        if tuple(input_names) != FLOAT_INPUT_NAMES:
            raise ValueError(f"ONNX inputs {input_names} != {list(FLOAT_INPUT_NAMES)}")
        if output_names != [OUTPUT_NAME]:
            raise ValueError(f"ONNX outputs {output_names} != {[OUTPUT_NAME]}")
        if "motion_token" in output_names:
            raise ValueError("ScaleBFM ONNX must not expose motion_token")
        for value in (*model.graph.input, *model.graph.output):
            declared = value.type.tensor_type.elem_type
            if declared != onnx.TensorProto.FLOAT:
                raise TypeError(
                    f"ONNX value {value.name!r} type {declared} is not float32"
                )
            _set_batch(value)
        onnx.checker.check_model(model)
        onnx.save(model, str(candidate))
        os.replace(candidate, destination)
    return destination


def prepare_scalebfm_torso_umi3(*, root: str | Path | None = None) -> Path:
    """Install the locked batched ONNX, exporting only when it is missing."""

    repo = default_repo_root() if root is None else Path(root).resolve()
    entries = load_lock(repo / "artifacts/lock.v1.json")
    verify(resolve(SOURCE_ID, lock=entries), root=repo)
    policy_entry = resolve(POLICY_ID, lock=entries)
    try:
        return verify(policy_entry, root=repo)
    except (FileNotFoundError, ValueError):
        pass
    output = (repo / str(policy_entry.path)).resolve()
    export_scalebfm_batched_onnx(output_path=output)
    if output.stat().st_size != policy_entry.bytes:
        raise ValueError(
            "derived ScaleBFM policy size mismatch: "
            f"{output.stat().st_size} != {policy_entry.bytes}"
        )
    if sha256_file(output) != policy_entry.sha256:
        raise ValueError("derived ScaleBFM policy sha256 mismatch")
    return verify(policy_entry, root=repo)


__all__ = [
    "FLOAT_INPUT_NAMES",
    "POLICY_ID",
    "SOURCE_ID",
    "export_scalebfm_batched_onnx",
    "prepare_scalebfm_torso_umi3",
]
