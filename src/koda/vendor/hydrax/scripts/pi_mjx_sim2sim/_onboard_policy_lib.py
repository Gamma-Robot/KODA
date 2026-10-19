"""Policy reconstruction helpers copied from play_export_check_humanoid_transformer_onboard.py (side-track)."""
import time
import math
import os
import re
import json
import pathlib
import torch
import numpy as np
import torch.nn as nn
import xml.etree.ElementTree as ET


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    # Broadcast (3,) or (1,3) gravity-like vec across quat batch dims.
    v = vec.reshape(-1)
    if v.numel() == 3 and quat.numel() > 4:
        out_shape = quat.shape[:-1] + (3,)
        n = 1
        for s in quat.shape[:-1]:
            n = n * int(s)
        quat_f = quat.reshape(n, 4)
        vec_f = v.reshape(1, 3).expand(n, 3).contiguous()
        xyz = quat_f[:, 1:]
        t = xyz.cross(vec_f, dim=-1) * 2
        return (vec_f + quat_f[:, 0:1] * t + xyz.cross(t, dim=-1)).view(out_shape)
    shape = vec.shape
    quat_f = quat.reshape(-1, 4)
    vec_f = vec.reshape(-1, 3)
    xyz = quat_f[:, 1:]
    t = xyz.cross(vec_f, dim=-1) * 2
    return (vec_f + quat_f[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    # Broadcast (3,) or (1,3) gravity-like vec across quat batch dims.
    v = vec.reshape(-1)
    if v.numel() == 3 and quat.numel() > 4:
        out_shape = quat.shape[:-1] + (3,)
        n = 1
        for s in quat.shape[:-1]:
            n = n * int(s)
        quat_f = quat.reshape(n, 4)
        vec_f = v.reshape(1, 3).expand(n, 3).contiguous()
        xyz = quat_f[:, 1:]
        t = xyz.cross(vec_f, dim=-1) * 2
        return (vec_f - quat_f[:, 0:1] * t + xyz.cross(t, dim=-1)).view(out_shape)
    shape = vec.shape
    quat_f = quat.reshape(-1, 4)
    vec_f = vec.reshape(-1, 3)
    xyz = quat_f[:, 1:]
    t = xyz.cross(vec_f, dim=-1) * 2
    return (vec_f - quat_f[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_mul_inverse_left(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w1_inv = w1
    x1_inv = -x1
    y1_inv = -y1
    z1_inv = -z1
    ww = (z1_inv + x1_inv) * (x2 + y2)
    yy = (w1_inv - y1_inv) * (w2 + z2)
    zz = (w1_inv + y1_inv) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1_inv - x1_inv) * (x2 - y2))
    w = qq - ww + (z1_inv - y1_inv) * (y2 - z2)
    x = qq - xx + (x1_inv + w1_inv) * (x2 + w2)
    y = qq - yy + (w1_inv - x1_inv) * (y2 + z2)
    z = qq - zz + (z1_inv + y1_inv) * (w2 - x2)
    
    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_mul_inverse_right(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w2_inv = w2
    x2_inv = -x2
    y2_inv = -y2
    z2_inv = -z2
    ww = (z1 + x1) * (x2_inv + y2_inv)
    yy = (w1 - y1) * (w2_inv + z2_inv)
    zz = (w1 + y1) * (w2_inv - z2_inv)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2_inv - y2_inv))
    w = qq - ww + (z1 - y1) * (y2_inv - z2_inv)
    x = qq - xx + (x1 + w1) * (x2_inv + w2_inv)
    y = qq - yy + (w1 - x1) * (y2_inv + z2_inv)
    z = qq - zz + (z1 + y1) * (w2_inv - x2_inv)
    
    return torch.stack([w, x, y, z], dim=-1).view(shape)

class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1000, base=10000):
        super(RoPEPositionalEncoding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # Create position indices
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Calculate frequencies
        freqs = torch.outer(position, self.inv_freq)  # (seq_len, dim//2)
        
        # Create rotation matrices
        cos_freqs = torch.cos(freqs)  # (seq_len, dim//2)
        sin_freqs = torch.sin(freqs)  # (seq_len, dim//2)
        
        # Apply RoPE rotation
        x_rotated = self.apply_rope(x, cos_freqs, sin_freqs)

        return x_rotated
    
    def apply_rope(self, x, cos_freqs, sin_freqs):
        # x shape: (batch_size, seq_len, dim)
        # cos_freqs, sin_freqs shape: (seq_len, dim//2)
        
        batch_size, seq_len, dim = x.shape
        
        # Reshape x to separate even and odd dimensions
        x_even = x[:, :, 0::2]  # (batch_size, seq_len, dim//2)
        x_odd = x[:, :, 1::2]   # (batch_size, seq_len, dim//2)
        
        # Expand cos_freqs and sin_freqs to match batch dimension
        cos_freqs = cos_freqs.unsqueeze(0)  # (1, seq_len, dim//2)
        sin_freqs = sin_freqs.unsqueeze(0)  # (1, seq_len, dim//2)
        
        # Apply rotation
        x_even_rotated = x_even * cos_freqs - x_odd * sin_freqs
        x_odd_rotated = x_even * sin_freqs + x_odd * cos_freqs
        
        # Interleave the rotated dimensions
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (..., dim)
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU activation function: SwiGLU(x) = SiLU(xW) ⊗ (xV)
    where SiLU(x) = x * sigmoid(x) (also known as Swish)
    """
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        # x shape: (..., input_dim)
        # SiLU activation: x * sigmoid(x)
        swish_gate = self.silu(self.w(x))
        # Element-wise multiplication with the value branch
        gated = swish_gate * self.v(x)
        # Project back to input dimension
        return self.output(gated)

class HumanoidTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.feed_forward = SwiGLU(embed_dim, ff_dim)

        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.rmsnorm3 = RMSNorm(embed_dim)
        self.cond_norm = RMSNorm(embed_dim)
        
        self.rope = RoPEPositionalEncoding(embed_dim)

    def forward(self, x, c, self_attn_mask = None):
        
        x_norm = self.rmsnorm1(x)
        x_rope = self.rope(x_norm)
        attn_output, _ = self.self_attention(x_rope, x_rope, x_norm, attn_mask=self_attn_mask)
        x = x + attn_output

        x_norm2 = self.rmsnorm2(x)
        c_norm = self.cond_norm(c)
        cross_output, _ = self.cross_attention(
            query=x_norm2,
            key=c_norm,
            value=c_norm
        )
        x = x + cross_output

        x_norm3 = self.rmsnorm3(x)
        ff_output = self.feed_forward(x_norm3)
        x = x + ff_output

        return x

class TaskEmbedder(nn.Module):
    def __init__(self,
                 task_obs_dim,
                 embedding_dim,
                 reduced_task_dim = None,
                 hidden_dims = None):
        super().__init__()

        if reduced_task_dim is not None:
            self.task_projection = self._build_task_projection(task_obs_dim, reduced_task_dim, hidden_dims)
            A = torch.randn(embedding_dim, reduced_task_dim, dtype=torch.float)
            Q, R = torch.linalg.qr(A, mode="reduced")
            diag = torch.sign(torch.diag(R))
            diag[diag==0] = 1.0
            self.register_buffer("W", Q * diag)
            self._forward_method = self._reduced_task_projection
        else:
            self.task_projection = self._build_task_projection(task_obs_dim, embedding_dim, hidden_dims)
            self._forward_method = self._normal_task_projection

    def _build_task_projection(self, input_dim, output_dim, hidden_dims = None):
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Linear(input_dim, output_dim)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ELU())

        for layer_index in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index+1]))
            layers.append(nn.ELU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def _reduced_task_projection(self, task_obs):
        task_embedding = self.task_projection(task_obs)
        task_embedding = task_embedding / (task_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        task_tokens = torch.matmul(task_embedding, self.W.T)
        return task_tokens
    
    def _normal_task_projection(self, task_obs):
        task_tokens = self.task_projection(task_obs)
        return task_tokens
    
    def forward(self, task_obs):
        return self._forward_method(task_obs)
    
    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                in_dim = m.weight.size(1)
                std = 1.0 / math.sqrt(in_dim)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class HumanoidTransformer(nn.Module):
    def __init__(self, 
                 prop_obs_dim, 
                 action_dim, 
                 output_dim,
                 embed_dim = 256, 
                 num_heads = 4, 
                 ff_dim = 256, 
                 num_layers = 4,
        ):
        super(HumanoidTransformer, self).__init__()
        self.prop_obs_dim = prop_obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.prop_projection = nn.Linear(prop_obs_dim, embed_dim)
        self.action_projection = nn.Linear(action_dim, embed_dim)
        
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            HumanoidTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(embed_dim)
        self.projection_head = nn.Linear(embed_dim, output_dim)

    def forward(self, prop_obs, action_obs, task_tokens):
        batch_size = prop_obs.shape[0]
        context_size = prop_obs.shape[1]

        prop_obs = self.prop_projection(prop_obs) # (bs, cl, dim)
        action_obs = self.action_projection(action_obs) # (bs, cl, dim)

        context = prop_obs.new_empty(batch_size, 2*context_size-1, self.embed_dim)
        context[:, 0::2] = prop_obs
        context[:, 1::2] = action_obs[:, 1:]

        empty_embedding = self.empty_embedding.expand(batch_size, -1, -1)
        x = torch.cat([context, empty_embedding], dim=1)
        
        self_attn_mask = torch.zeros(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        # self_attn_mask[:-1, -1] = True # weishuai: tensorrt does not support dynamic index
        row_idx = torch.arange(x.shape[1] - 1, device=x.device)
        col_idx = torch.full((x.shape[1] - 1,), x.shape[1] - 1, device=x.device)
        self_attn_mask[row_idx, col_idx] = True
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, task_tokens, self_attn_mask=self_attn_mask)
        x = self.final_norm(x)

        empty_embedding = x[:, -1, :]
        output = self.projection_head(empty_embedding)
        
        return output

    @torch.no_grad()
    def init_weights(self, head_init_val = None):
        L = len(self.transformer_blocks)
        res_scale = 1.0 / math.sqrt(2.0 * L)
        
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                continue
            
            if isinstance(m, RMSNorm):
                nn.init.ones_(m.scale)
            elif isinstance(m, nn.Linear):
                in_dim = m.weight.size(1)
                std = 1.0 / math.sqrt(in_dim)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                embed_dim = m.embed_dim
                std = 1.0 / math.sqrt(embed_dim)

                nn.init.normal_(m.in_proj_weight, mean=0.0, std=std)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                
                out_std = std / math.sqrt(2 * len(self.transformer_blocks))
                nn.init.normal_(m.out_proj.weight, mean=0.0, std=out_std)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

        for blk in self.transformer_blocks:
            if hasattr(blk, "feed_forward") and hasattr(blk.feed_forward, "output"):
                blk.feed_forward.output.weight.mul_(res_scale)

        nn.init.trunc_normal_(self.empty_embedding, std=0.02)

        if head_init_val is not None:
            nn.init.zeros_(self.projection_head.weight)
            self.projection_head.bias.copy_(head_init_val)
        else:
            nn.init.zeros_(self.projection_head.weight)
            nn.init.zeros_(self.projection_head.bias)

class HumanoidTransformerPolicyWrapperWithMode(nn.Module):

    def __init__(
        self,
        actor,
        actor_task_embedder,
        mode_mappings: torch.Tensor,
        mode_vectors: torch.Tensor,
        default_dof_pos: torch.Tensor,
        action_scale: torch.Tensor,
        context_len: int,
        future_len: int, 
        local_translation: torch.Tensor,
        local_rotation: torch.Tensor,
        parent_indices: torch.Tensor,
        joint_axis: torch.Tensor,
        selected_link_indices: torch.Tensor,
        lab_to_xml_joint_indices: torch.Tensor,
    ) -> None:
        super().__init__()

        # Modules and Components
        self.task_embedder = actor_task_embedder
        self.prop_embedder = actor.prop_projection
        self.action_embedder = actor.action_projection
        self.transformer_blocks = actor.transformer_blocks
        self.final_norm = actor.final_norm
        self.projection_head = actor.projection_head
        self.register_buffer("empty_embedding", actor.empty_embedding)
        attn_mask = torch.zeros(2*context_len, 2*context_len, dtype=torch.bool, device=mode_vectors.device)
        row_idx = torch.arange(2*context_len - 1, device=mode_vectors.device)
        col_idx = torch.full((2*context_len - 1,), 2*context_len - 1, device=mode_vectors.device)
        attn_mask[row_idx, col_idx] = True
        self.register_buffer("self_attn_mask", attn_mask)

        # Observation and Action
        self.register_buffer("mode_mappings", mode_mappings)
        self.register_buffer("mode_vectors", mode_vectors)
        self.register_buffer("gravity_vec", torch.tensor([[0,0,-1]], dtype=torch.float, device=mode_vectors.device).unsqueeze(1).expand(-1, context_len, -1))
        self.register_buffer("default_dof_pos", default_dof_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("selected_link_indices", selected_link_indices)
        tan_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float,device=mode_vectors.device)
        norm_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float,device=mode_vectors.device)
        tan_vec[..., 0] = 1
        norm_vec[..., -1] = 1
        self.register_buffer("tan_vec", tan_vec)
        self.register_buffer("norm_vec", norm_vec)

        # FK Modules
        self.register_buffer("local_translation", local_translation) # (num_joints, 3)
        self.register_buffer("local_rotation", local_rotation) # (num_joints, 4)
        self.register_buffer("parent_indices", parent_indices) # (num_joints + 1)
        self.register_buffer("joint_axis", joint_axis) # (num_joints, 3)
        self.register_buffer("lab_to_xml_joint_indices", lab_to_xml_joint_indices)

    def forward(
        self,
        root_quat_buffer: torch.Tensor, # wxyz
        base_ang_vel_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor, # in Isaaclab order
        dof_vel_buffer: torch.Tensor, # in Isaaclab order
        last_action_buffer: torch.Tensor,
        target_body_pos_future_to_robot_base: torch.Tensor, # (bs, num_future, num_link, 3)
        target_body_rot_future_to_robot_base: torch.Tensor,
        mode_index: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Build Context
        grav = self.gravity_vec.expand(root_quat_buffer.shape[0], -1, -1)
        projected_gravity_buffer = quat_apply_inverse(root_quat_buffer, grav)  # (bs, num_context, 3)
        dof_pos_rel_buffer = dof_pos_buffer - self.default_dof_pos
        prop_obs = torch.cat([
            projected_gravity_buffer,
            base_ang_vel_buffer,
            dof_pos_rel_buffer,
            dof_vel_buffer * 0.05
        ], dim=-1)
        prop_token = self.prop_embedder(prop_obs)
        action_token = self.action_embedder(last_action_buffer)

        x = torch.empty(prop_token.shape[0], 2*prop_token.shape[1], prop_token.shape[2], dtype=prop_token.dtype, device=prop_token.device)
        x[:, ::2] = prop_token
        x[:, 1:-1:2] = action_token[:, 1:]
        x[:, 2*prop_token.shape[1]-1] = self.empty_embedding[0, 0]

        # Extract current state from buffer
        root_quat = root_quat_buffer[:, -1]
        dof_pos = dof_pos_buffer[:, -1]

        # FK Module (supports batch size bs >= 1)
        bs = int(prop_token.shape[0])
        half_angles = dof_pos[:, self.lab_to_xml_joint_indices].unsqueeze(-1) / 2
        sin_half = torch.sin(half_angles)
        cos_half = torch.cos(half_angles)
        joint_rot = torch.cat([cos_half, self.joint_axis.unsqueeze(0) * sin_half], dim=-1)

        nbody = int(self.parent_indices.shape[0])
        body_pos = torch.zeros(bs, nbody, 3, dtype=torch.float, device=joint_rot.device)
        body_quat = torch.zeros(bs, nbody, 4, dtype=torch.float, device=joint_rot.device)
        body_quat[..., 0] = 1

        for j in range(1, nbody):
            j_rot = joint_rot[:, j - 1]
            local_trans = self.local_translation[j - 1 : j].expand(bs, -1)
            local_rot = self.local_rotation[j - 1 : j].expand(bs, -1)
            parent_idx = self.parent_indices[j : j + 1]
            parent_pos = body_pos[:, parent_idx].squeeze(1)
            parent_rot = body_quat[:, parent_idx].squeeze(1)
            world_trans = quat_apply(parent_rot, local_trans)
            curr_pos = parent_pos + world_trans
            curr_rot = quat_mul(local_rot, j_rot)
            curr_rot = quat_mul(parent_rot, curr_rot)
            body_pos[:, j] = curr_pos
            body_quat[:, j] = curr_rot

        body_pos_to_robot_base = body_pos[:, self.selected_link_indices]
        body_quat_to_robot_base = body_quat[:, self.selected_link_indices]

        # Build Task Observation
        target_body_pos_future_rel_to_robot_base = (
            target_body_pos_future_to_robot_base - body_pos_to_robot_base[:, None, :, :]
        )

        tan_vec = self.tan_vec.expand(bs, -1, -1, -1)
        norm_vec = self.norm_vec.expand(bs, -1, -1, -1)
        target_body_rot_future_to_robot_base_tan_norm = torch.cat(
            [
                quat_apply(target_body_rot_future_to_robot_base, tan_vec),
                quat_apply(target_body_rot_future_to_robot_base, norm_vec),
            ],
            dim=-1,
        )
        target_body_rot_future_rel_to_robot_base = quat_mul_inverse_right(
            target_body_rot_future_to_robot_base,
            body_quat_to_robot_base[:, None].expand(
                -1, target_body_rot_future_to_robot_base.shape[1], -1, -1
            ),
        )
        target_body_rot_future_rel_to_robot_base_tan_norm = torch.cat(
            [
                quat_apply(target_body_rot_future_rel_to_robot_base, tan_vec),
                quat_apply(target_body_rot_future_rel_to_robot_base, norm_vec),
            ],
            dim=-1,
        )
        
        task_obs = torch.cat([
            target_body_pos_future_to_robot_base.flatten(2,3),
            target_body_pos_future_rel_to_robot_base.flatten(2,3),
            target_body_rot_future_to_robot_base_tan_norm.flatten(2,3),
            target_body_rot_future_rel_to_robot_base_tan_norm.flatten(2,3),
            time_offsets
        ], dim=-1) # (bs, nf, ndim)

        # Apply control mode
        mapping = self.mode_mappings[mode_index]
        task_obs_masked = task_obs * mapping.unsqueeze(1)
        mode_vec = self.mode_vectors[mode_index]
        task_input = torch.cat([task_obs_masked, mode_vec.unsqueeze(1).expand(-1, task_obs_masked.shape[1], -1)], dim=-1)
        task_tokens = self.task_embedder(task_input)

        # Forward
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, task_tokens, self_attn_mask=self.self_attn_mask)
        x = self.final_norm(x)

        action = self.projection_head(x[:, -1, :])
        
        # Return Direct PD target and action for buffer
        return action * self.action_scale + self.default_dof_pos, action

def build_mode_mappings(
    mode_table: torch.Tensor,
    feature_dims_per_link,
    with_time: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build (num_modes, task_obs_dim) mapping from mode table and feature dims.

    Replicates the logic of mdp.mode_mapping for each row of mode_table, so the
    exported ONNX can apply the correct mask per mode index without env.
    task_obs_dim = sum(feature_dims_per_link) + (1 if with_time else 0).
    """
    device = device or mode_table.device
    mode_table = mode_table.to(device)
    assert mode_table.dim() == 2, "mode_table must be (num_modes, num_links)"
    num_modes, num_links = mode_table.shape[0], mode_table.shape[1]
    mappings = []
    for dim in feature_dims_per_link:
        # (num_modes, num_links, dim) * (num_modes, num_links, 1) -> view (num_modes, -1)
        part = (
            torch.ones(num_modes, mode_table.shape[1], dim, device=device, dtype=torch.float32)
            * mode_table.unsqueeze(-1)
        ).view(num_modes, -1)
        mappings.append(part)
    if with_time:
        mappings.append(torch.ones(num_modes, 1, device=device, dtype=torch.float32))
    return torch.cat(mappings, dim=-1)

def parse_xml(xml_file, device = "cpu"):
    body_names = []
    parent_indices = []
    local_translation = []
    local_rotation = []
    joint_axis = []
    joint_names = []
   
    tree = ET.parse(xml_file)
    xml_doc_root = tree.getroot()
    xml_world_body = xml_doc_root.find("worldbody")
    assert xml_world_body is not None, "worldbody not found"
    
    xml_body_root = xml_world_body.find("body")
    assert xml_body_root is not None, "body not found"
    
    compiler_data = xml_doc_root.find("compiler")
    rot_unit = compiler_data.attrib.get("angle", "degree")
    assert rot_unit in ["degree", "radian"], f"Invalid rotation unit: {rot_unit}"
    
    def _add_xml_body(xml_node, parent_index, body_index):
        body_name = xml_node.attrib.get("name")
        pos_data = xml_node.attrib.get("pos", "0 0 0")
        pos = np.fromstring(pos_data, dtype=float, sep=" ")
        
        rot_data = xml_node.attrib.get("quat", "1 0 0 0")
        rot = np.fromstring(rot_data, dtype=float, sep=" ")
        
        if body_index == 0:
            pass
        else:
            curr_joints = xml_node.findall("joint")
            num_joints = len(curr_joints)
            assert num_joints == 1
            _axis = np.fromstring(curr_joints[0].attrib.get("axis"), dtype=float, sep=" ")
            axis = torch.from_numpy(_axis)
            local_rotation.append(rot)
            local_translation.append(pos)
            joint_axis.append(axis)
            joint_names.append(curr_joints[0].attrib.get("name"))
        
        body_names.append(body_name)
        parent_indices.append(parent_index)
        
        curr_index = body_index
        body_index += 1
        for child in xml_node.findall("body"):
            body_index = _add_xml_body(child, curr_index, body_index)
            
        return body_index
    
    _add_xml_body(xml_body_root, -1, 0)
    
    parent_indices = torch.tensor(parent_indices, dtype=torch.long, device=device)
    local_translation = torch.tensor(np.array(local_translation), dtype=torch.float, device=device)
    local_rotation = torch.tensor(np.array(local_rotation), dtype=torch.float, device=device)
    joint_axis = torch.stack(joint_axis, dim=0).float().to(device) # weishuai: The original variable is float64

    return body_names, joint_names, parent_indices, joint_axis, local_translation, local_rotation
