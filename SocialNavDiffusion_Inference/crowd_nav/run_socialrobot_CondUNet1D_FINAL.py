#!/usr/bin/env python3
"""
run_robot_conditional.py

Now, we will use cross-attention to determine the most important parts of the conditioning vector at each timestep
------------------------------------------------------------------------------------------------------
We replaced flat conditioning vectors with tokenized conditioning (start, goal, and per-obstacle tokens)
and integrated them via cross-attention, allowing each trajectory timestep to selectively attend to relevant
scene elements. We also added attention masking for padded obstacles and timestep embeddings, making the diffusion
model robust to variable numbers of obstacles and improving conditioning fidelity and stability.
------------------------------------------------------------------------------------------------------

Conditional diffusion trainer for short-horizon robot trajectories.
Conditions: start (x, y, theta, v), goal (x, y), up to K_max dynamic obstacles (x,y,vx,vy).
Output: trajectory (x,y) over 16 samples (take each to be 0.1 s long so 1.6 s horizon)

TO START A COMPUTE NODE, DO IN TERMINAL: srun --mem=8G --time=1:00:00 --pty bash
THEN TO EXIT, CTRL+D
THIS WAY, YOU AREN'T IN LOGIN, CORRECTLY USE A DEDICATED NODE FOR COMPS

run_robot_conditional_socialCFG.py
 
Multi-axis CFG-controllable diffusion social navigation planner.
 
Conditioning is split into two groups:
  (A) SCENE conditioning  — start, goal, obstacles, optional static occupancy map.
                            Dropped jointly with probability `scene_cfg_dropout`
                            so the model learns a true unconditional baseline.
  (B) STYLE conditioning  — four scalar social-style axes:
                              s_prox       : proxemic conservativeness
                              s_pass       : pass-side preference (left vs right)
                              s_yield      : yielding aggressiveness
                              s_group      : group respect (no explicit group labels;
                                             relies on self-attention to discover
                                             group structure from obstacle positions
                                             and velocities)
                            Each axis is INDEPENDENTLY droppable during training
                            (probability `style_cfg_dropout` per axis), enabling
                            compositional classifier-free guidance at inference.
 
At inference we run 6 passes per denoising step in a single batched forward call:
    eps_uncond                — full null (scene & all style)
    eps_scene                 — scene on, style nulled
    eps_style_prox            — style axis 0 on, all others nulled
    eps_style_pass            — style axis 1 on, all others nulled
    eps_style_yield           — style axis 2 on, all others nulled
    eps_style_group           — style axis 3 on, all others nulled
 
and combine them via compositional CFG:
    eps = eps_uncond
          + w_scene * (eps_scene - eps_uncond)
          + sum_i w_i * (eps_style_i - eps_uncond)
"""

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet1DModel
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy

import sys
sys.path.append("/home/cschaibl/projects/aip-sl2smith/cschaibl/diffusion_planning")  # add the parent folder of somepackage

from diffusers_unet_1d_condition import UNet1DConditionModel

# =============================================================================
# Social style axes — names and ordering are GLOBAL CONSTANTS.
# The ordering here is the canonical ordering used for the tensor shape [B, n_axes].
# =============================================================================
STYLE_AXES = ["prox", "pass", "yield", "group"]
N_STYLE_AXES = len(STYLE_AXES)

STYLE_AXIS_LABELS_VIZ = [
    ("prox",  "close",        "conservative"),
    ("pass",  "left-hand",    "right-hand"),
    ("yield", "assertive",    "yielding"),
    ("group", "indifferent",  "group-aware"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Social cost validation
# Mirrors the cost formulation in expert_traj_generation so val metrics
# are directly comparable to expert trajectory quality.
# ─────────────────────────────────────────────────────────────────────────────

_VAL_DT             = 0.05
_VAL_V_MAX          = 1.0
_VAL_A_MAX          = 1.5
_VAL_W_MAX          = np.pi
_VAL_COLLISION_R    = 0.65
_VAL_PROX_SIGMA     = 2.0
_VAL_D_HEAD_ON      = 3.0
_VAL_D_SNEAK        = 4.0
_VAL_TTC_TAU        = 2.0
_VAL_TTC_MAX        = 5.0
_VAL_GROUP_DIST     = 1.5
_VAL_JERK_MAX       = 2 * _VAL_A_MAX / _VAL_DT
_VAL_SOCIAL_SCALE   = 1.5
_VAL_W_HEAD_ON      = 0.6
_VAL_W_SNEAK_UP     = 0.6
_VAL_W_SLOW_NEAR    = 0.3
_VAL_W_CUTOFF       = 1.2
_VAL_W_SPLIT_GROUP  = 1.0
_VAL_W_SMOOTH       = 0.05


from diffusers.models.attention_processor import Attention

class CrossAttnCaptureProcessor:
    """
    Drop-in replacement for AttnProcessor that captures real attention
    weights during the actual forward pass. Works with the Transformer1DModel
    -> BasicTransformerBlock -> Attention chain in your UNet.
    
    Stores last_attn as [B, n_heads, Q, K] for the most recent forward call
    where encoder_hidden_states is not None (i.e. cross-attention only).
    """
    def __init__(self):
        self.last_attn = None   # [B, H, Q, K] CPU tensor, set after each cross-attn call
        self.is_cross_attn = False

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        is_cross = encoder_hidden_states is not None
        self.is_cross_attn = is_cross

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)   # [B, H, Q, head_dim]
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)     # [B, H, K, head_dim]
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)   # [B, H, K, head_dim]

        # --- Compute attention scores manually so we can capture the weights ---
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B, H, Q, K]

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)  # [B, H, Q, K]

        # --- Capture only for cross-attention ---
        if is_cross:
            self.last_attn = attn_weights.detach().cpu()   # [B, H, Q, K]

        hidden_states = torch.matmul(attn_weights, value)  # [B, H, Q, head_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def attach_cross_attn_capture_processors(unet, block_filter=None):
    """
    Replace attention processors with CrossAttnCaptureProcessor.
    
    block_filter: optional callable(name) -> bool to select which attention
    layers to instrument. Defaults to all attn2 (cross-attention) layers.
    
    Returns a dict {name: processor} for later access.
    """
    if block_filter is None:
        # attn1 = self-attention, attn2 = cross-attention in BasicTransformerBlock
        block_filter = lambda name: "attn2" in name

    new_processors = {}
    for name, proc in unet.attn_processors.items():
        if block_filter(name):
            new_processors[name] = CrossAttnCaptureProcessor()
        else:
            new_processors[name] = proc

    unet.set_attn_processor(new_processors)
    return {k: v for k, v in new_processors.items() if isinstance(v, CrossAttnCaptureProcessor)}


def _get_mean_cross_attn(proc, batch_idx=0):
    if proc.last_attn is None:
        return None
    if batch_idx >= proc.last_attn.shape[0]:
        # Fallback to batch item 0 rather than crashing — happens if
        # last_attn was written by a training step (small batch) rather
        # than the logging forward pass (n_variants * K batch)
        return None
    return proc.last_attn[batch_idx].float().mean(dim=0).numpy()   # [Q, K]


def log_attention_heatmaps(unet, step, token_labels=None, section_boundaries=None, batch_idx=0):
    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, CrossAttnCaptureProcessor) or proc.last_attn is None:
            continue

        attn = _get_mean_cross_attn(proc, batch_idx=batch_idx)   # [Q, K]
        if attn is None:
            continue
        n_tokens = attn.shape[1]

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(attn, aspect="auto", cmap="viridis")
        ax.set_title(f"Cross-attention (mean over heads)\n{name}", fontsize=10)
        ax.set_xlabel("Scene tokens (K)")
        ax.set_ylabel("Trajectory queries (Q)")

        if token_labels is not None and len(token_labels) == n_tokens:
            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=6)

        if section_boundaries is not None:
            bounds_sorted = sorted(section_boundaries, key=lambda x: x[0])
            bounds_sorted.append((n_tokens, None))
            for boundary_idx, _ in bounds_sorted[:-1]:
                if 0 < boundary_idx < n_tokens:
                    ax.axvline(boundary_idx - 0.5, color="white", linewidth=1.0, alpha=0.85, zorder=4)
            for i in range(len(bounds_sorted) - 1):
                start_i, label = bounds_sorted[i]
                end_i = bounds_sorted[i + 1][0]
                if label and end_i > start_i:
                    ax.text(0.5 * (start_i + end_i - 1), -0.8, label,
                            ha="center", va="bottom", fontsize=9,
                            fontweight="bold", color="black")

        fig.colorbar(im, ax=ax)
        wandb.log({f"attn_heatmap/{name}": wandb.Image(fig)}, step=step)
        plt.close(fig)


def log_map_attention_2d(unet, step, scene_embedder, occ_map=None,
                          map_extent=None, per_timestep=False, batch_idx=0):
    n_map = scene_embedder.n_map_tokens
    if n_map == 0:
        return
    h_out = scene_embedder.map_encoder.h_out
    w_out = scene_embedder.map_encoder.w_out

    map_start = 2 + scene_embedder.k_max
    map_end   = map_start + n_map

    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, CrossAttnCaptureProcessor) or proc.last_attn is None:
            continue

        attn = _get_mean_cross_attn(proc, batch_idx=batch_idx)    # [Q, K_total]
        if attn is None:
            continue
        if attn.shape[1] < map_end:
            continue

        map_cols = attn[:, map_start:map_end]   # [Q, n_map]
        agg = map_cols.mean(axis=0).reshape(h_out, w_out)

        fig, ax = plt.subplots(figsize=(5, 5))
        if occ_map is not None and map_extent is not None:
            half = map_extent / 2.0
            ax.imshow(occ_map, extent=[-half, half, -half, half],
                      origin="lower", cmap="gray_r", alpha=0.4,
                      zorder=0, interpolation="nearest", vmin=0, vmax=1)
            im = ax.imshow(agg, extent=[-half, half, -half, half],
                           origin="lower", cmap="viridis", alpha=0.65,
                           zorder=1, interpolation="nearest")
        else:
            im = ax.imshow(agg, origin="lower", cmap="viridis", alpha=0.85)

        ax.set_title(f"Map attention (mean Q, mean heads)\n{name}", fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        wandb.log({f"map_attn_2d/{name}": wandb.Image(fig)}, step=step)
        plt.close(fig)


@torch.no_grad()
def _val_social_cost_batch(traj_xy_ego, obs_physical, obs_valid_mask, style_values=None):
    """
    Parameters
    ----------
    style_values : (B, N_STYLE_AXES) float tensor in [-1,1], or None for neutral.
                   Applied as multiplicative weight modifiers matching
                   style_to_effective_weights() in expert_traj_v65_fast.py.
    """
    B, T, _ = traj_xy_ego.shape
    device = traj_xy_ego.device

    # Compute per-sample effective weights from style vector.
    # Mirrors style_to_effective_weights(): scale(s) = exp(3 * log(2) * s)
    # so s=0 → 1x, s=+1 → 8x, s=-1 → 0.125x
    if style_values is not None:
        s = style_values.clamp(-1, 1)          # (B, 4): prox, pass, yield, group
        def _scale(s_val):
            return torch.exp(3.0 * math.log(2.0) * s_val)  # (B,)
        s_prox  = s[:, 0]; s_pass = s[:, 1]
        s_yield = s[:, 2]; s_group = s[:, 3]

        # Per-sample effective weights — shape (B,) each
        w_head_on     = _VAL_W_HEAD_ON     * _scale(s_yield)
        w_sneak_up    = _VAL_W_SNEAK_UP    * _scale(s_prox)
        w_slow_near   = _VAL_W_SLOW_NEAR   * _scale(s_prox)
        w_cutoff      = _VAL_W_CUTOFF      * _scale(s_yield)
        w_split_group = _VAL_W_SPLIT_GROUP * (1.0 + 1.5 * s_group).clamp(min=0.25)
        # prox sigma modulation (used in distance kernels)
        sigma_prox    = _VAL_PROX_SIGMA * (1.0 + 0.5 * s_prox)       # (B,)
    else:
        w_head_on     = torch.full((B,), _VAL_W_HEAD_ON,     device=device)
        w_sneak_up    = torch.full((B,), _VAL_W_SNEAK_UP,    device=device)
        w_slow_near   = torch.full((B,), _VAL_W_SLOW_NEAR,   device=device)
        w_cutoff      = torch.full((B,), _VAL_W_CUTOFF,      device=device)
        w_split_group = torch.full((B,), _VAL_W_SPLIT_GROUP, device=device)
        sigma_prox    = torch.full((B,), _VAL_PROX_SIGMA,    device=device)

    # Unsqueeze weights for broadcasting against (B, T-1, k_max) tensors
    def _b(w): return w.view(B, 1, 1)   # (B,1,1)
    def _b1(w): return w.view(B, 1)     # (B,1)
    
    dt = _VAL_DT
    mask = obs_valid_mask.bool()  # (B, k_max)
    M = obs_physical.shape[1]

    # ── smoothness cost (c8) ────────────────────────────────────────────────
    # Approximate v from positional finite differences; no v/omega state here.
    vel     = (traj_xy_ego[:, 1:] - traj_xy_ego[:, :-1]) / dt        # (B,T-1,2)
    speed   = torch.norm(vel, dim=-1)                                  # (B,T-1)
    acc     = (speed[:, 1:] - speed[:, :-1]) / dt                     # (B,T-2)
    jerk    = (acc[:, 1:]   - acc[:, :-1])   / dt                     # (B,T-3)
    jerk_n  = (jerk.abs() / _VAL_JERK_MAX).clamp(0, 1).mean(dim=1)   # (B,)
    # Turn-rate proxy: angle between successive velocity vectors
    vel_hat = F.normalize(vel, dim=-1, eps=1e-6)                      # (B,T-1,2)
    cos_dh  = (vel_hat[:, :-1] * vel_hat[:, 1:]).sum(dim=-1)          # (B,T-2)
    dh      = torch.acos(cos_dh.clamp(-1, 1))                         # (B,T-2)
    omega   = dh / dt
    alpha_n = (omega / _VAL_W_MAX).clamp(0, 1).mean(dim=1)            # (B,)
    c8      = 0.5 * jerk_n + 0.5 * alpha_n                            # (B,)

    # If no obstacles anywhere in the batch, return smoothness-only cost
    if not mask.any():
        social = _VAL_W_SMOOTH * c8
        bd = {k: torch.zeros(B, device=device)
              for k in ["c2_head_on", "c3_sneak_up", "c4_slow_near",
                        "c5_cutoff", "c7_split_group"]}
        bd["c8_smooth"] = _VAL_W_SMOOTH * c8
        return social.cpu().numpy(), {k: v.cpu().numpy() for k, v in bd.items()}

    # ── propagate obstacles forward ─────────────────────────────────────────
    # obs_future: (B, T-1, k_max, 2)  — positions at timesteps 1..T-1
    t_vec    = torch.arange(1, T, device=device).float() * dt           # (T-1,)
    obs_pos  = obs_physical[:, :, 0:2]                                   # (B,k_max,2)
    obs_vel  = obs_physical[:, :, 2:4]                                   # (B,k_max,2)
    obs_fut  = obs_pos.unsqueeze(1) + obs_vel.unsqueeze(1) * t_vec.view(1, T-1, 1, 1)
    # (B, T-1, k_max, 2)

    rob_xy   = traj_xy_ego[:, 1:]                                        # (B,T-1,2)
    # (B,T-1,k_max,2)
    delta    = rob_xy.unsqueeze(2) - obs_fut                             # rob - obs
    dist     = delta.norm(dim=-1).clamp(min=1e-4)                        # (B,T-1,k_max)

    # Mask padding
    m_exp    = mask.unsqueeze(1).float()                                  # (B,1,k_max)

    # Obstacle velocity direction (heading)
    obs_spd  = obs_vel.norm(dim=-1).clamp(min=1e-4)                      # (B,k_max)
    obs_vhat = obs_vel / obs_spd.unsqueeze(-1)                            # (B,k_max,2)

    # Robot velocity direction
    rob_vhat = F.normalize(vel, dim=-1, eps=1e-6)                        # (B,T-1,2)

    # Proximity kernel
    prox_w = torch.exp(-dist**2 / (2 * sigma_prox.view(B, 1, 1)**2))           # (B,T-1,k_max)

    # ── c2: head-on ─────────────────────────────────────────────────────────
    # cos of angle between robot heading and obstacle heading (negative = head-on)
    cos_hh   = (rob_vhat.unsqueeze(2) * obs_vhat.unsqueeze(1)).sum(-1)   # (B,T-1,k_max)
    head_on  = (-cos_hh).clamp(min=0)                                    # (B,T-1,k_max)
    spd_n    = (speed / _VAL_V_MAX).clamp(0, 1)                          # (B,T-1)
    active   = (dist < _VAL_D_HEAD_ON).float()
    c2_raw   = (head_on * spd_n.unsqueeze(2) * active * m_exp).max(dim=2).values
    c2 = w_head_on * c2_raw.clamp(0, 1).mean(dim=1)            # (B,)

    # ── c3: sneak-up ────────────────────────────────────────────────────────
    # from_hum_hat points FROM obstacle TO robot
    from_hum = (delta / dist.unsqueeze(-1))                              # (B,T-1,k_max,2)
    cos_phi  = (from_hum * obs_vhat.unsqueeze(1)).sum(-1)                # (B,T-1,k_max)
    behind   = (-cos_phi).clamp(min=0)
    # closing speed: relative velocity projected onto to-obstacle direction
    to_hum   = -from_hum                                                 # (B,T-1,k_max,2)
    rel_vel  = vel.unsqueeze(2) - obs_vel.unsqueeze(1)                   # (B,T-1,k_max,2)
    closing  = (rel_vel * to_hum).sum(-1).clamp(min=0) / _VAL_V_MAX     # (B,T-1,k_max)
    active3  = (dist < _VAL_D_SNEAK).float()
    c3_raw   = (behind * closing * prox_w * active3 * m_exp).max(dim=2).values
    c3 = w_sneak_up * c3_raw.clamp(0, 1).mean(dim=1)          # (B,)

    # ── c4: slow-near (speed in personal space) ─────────────────────────────
    personal = torch.exp(-dist**2 / (2 * 0.8**2))                       # (B,T-1,k_max)
    min_pers = (personal * m_exp).max(dim=2).values                      # (B,T-1)
    c4_raw   = 0.5 * (spd_n * min_pers).max(dim=1).values \
          + 0.5 * (spd_n * min_pers).mean(dim=1)          # (B,)
    c4       = w_slow_near * c4_raw.clamp(0, 1)               # (B,)

    # ── c5: cut-off ──────────────────────────────────────────────────────────
    # Project robot position onto obstacle forward direction
    ped_to_rob  = delta                                                   # (B,T-1,k_max,2)
    rob_fwd     = (ped_to_rob * obs_vhat.unsqueeze(1)).sum(-1)           # (B,T-1,k_max)
    ped_spd     = obs_spd.unsqueeze(1).clamp(min=1e-4)                   # (B,1,k_max)
    ped_ttc     = rob_fwd.clamp(min=0) / ped_spd                         # (B,T-1,k_max)
    ttc_w       = torch.exp(-ped_ttc / _VAL_TTC_TAU)
    ttc_active  = ((ped_ttc < _VAL_TTC_MAX) & (obs_spd.unsqueeze(1) > 0.05)).float() * m_exp
    c5_raw      = (ttc_w * ttc_active).max(dim=2).values
    c5 = w_cutoff * (0.5 * c5_raw.max(dim=1).values
                   + 0.5 * c5_raw.mean(dim=1)).clamp(0, 1)  # (B,)

    # ── c7: group splitting ──────────────────────────────────────────────────
    c7 = torch.zeros(B, device=device)
    if M >= 2:
        # pairwise obstacle distances and midpoints (upper triangle only)
        pos0   = obs_pos                                                   # (B,k_max,2)
        sep_   = pos0.unsqueeze(2) - pos0.unsqueeze(1)                    # (B,k_max,k_max,2)
        sep_d  = sep_.norm(dim=-1)                                        # (B,k_max,k_max)
        mid    = 0.5 * (pos0.unsqueeze(2) + pos0.unsqueeze(1))            # (B,k_max,k_max,2)
        # heading similarity
        cos_hum = (obs_vhat.unsqueeze(2) * obs_vhat.unsqueeze(1)).sum(-1) # (B,k_max,k_max)
        triu    = torch.triu(torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1)
        both_valid = mask.unsqueeze(2) & mask.unsqueeze(1)                # (B,k_max,k_max)
        grp_mask = ((sep_d < _VAL_GROUP_DIST) & triu & both_valid
                    & (cos_hum > 0.5))                                    # (B,k_max,k_max)
        if grp_mask.any():
            rob_xy_t = rob_xy                                              # (B,T-1,2)
            rob_to_mid = rob_xy_t.unsqueeze(2).unsqueeze(2) - mid.unsqueeze(1)
            # (B,T-1,k_max,k_max,2)
            d_mid    = rob_to_mid.norm(dim=-1).clamp(min=1e-4)            # (B,T-1,k_max,k_max)
            sigma_g  = sep_d.clamp(min=0.5).unsqueeze(1) * 1.5           # (B,1,k_max,k_max)
            intr     = torch.exp(-d_mid**2 / sigma_g**2)
            intr     = intr * grp_mask.unsqueeze(1).float()
            n_active = grp_mask.float().sum(dim=(1,2)).clamp(min=1)       # (B,)
            c7 = w_split_group * (intr.sum(dim=(2,3)) / n_active.view(B, 1)
                ).max(dim=1).values.clamp(0, 1)

    # ── combine ──────────────────────────────────────────────────────────────
    H = T - 1
    social_raw = c2 + c3 + c4 + c5 + c7 + _VAL_W_SMOOTH * c8
    social_scaled = social_raw * H * _VAL_SOCIAL_SCALE

    bd = {
        "c2_head_on":     (c2 * H * _VAL_SOCIAL_SCALE).cpu().numpy(),
        "c3_sneak_up":    (c3 * H * _VAL_SOCIAL_SCALE).cpu().numpy(),
        "c4_slow_near":   (c4 * H * _VAL_SOCIAL_SCALE).cpu().numpy(),
        "c5_cutoff":      (c5 * H * _VAL_SOCIAL_SCALE).cpu().numpy(),
        "c7_split_group": (c7 * H * _VAL_SOCIAL_SCALE).cpu().numpy(),
        "c8_smooth":      (_VAL_W_SMOOTH * c8 * H * _VAL_SOCIAL_SCALE).cpu().numpy(),
    }

    return social_scaled.cpu().numpy(), bd


@torch.no_grad()
def compute_social_validation_metrics(
    ema_model, ema_scene_embedder, ema_style_embedder,
    val_loader, noise_scheduler, norm_stats, args, device,
    n_batches=10,
):
    ema_model.eval()
    ema_scene_embedder.eval()
    ema_style_embedder.eval()

    ddim = DDIMScheduler(
        num_train_timesteps=noise_scheduler.config.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )
    ddim.set_timesteps(20, device=device)

    n_variants = 1 + 1 + N_STYLE_AXES
    K = max(1, args.K_samples)

    # Accumulators — neutral-weight cost is the primary comparable signal;
    # style-conditioned cost tracks style fidelity separately.
    all_cost_neutral, all_cost_styled = [], []
    all_c2, all_c3, all_c4, all_c5, all_c7, all_c8 = [], [], [], [], [], []
    all_collision, all_progress = [], []
    all_cost_neutral_styled, all_cost_neutral_neutral = [], []  # style-stratified
    all_cost_gt_styled, all_cost_gt_neutral = [], []   
    all_l2_vs_gt = []                                  

    traj_std_t  = torch.tensor(norm_stats["traj_std"],  dtype=torch.float32, device=device)
    traj_mean_t = torch.tensor(norm_stats["traj_mean"], dtype=torch.float32, device=device)
    obs_std_t   = torch.tensor(norm_stats["obs_std"],   dtype=torch.float32, device=device)
    obs_mean_t  = torch.tensor(norm_stats["obs_mean"],  dtype=torch.float32, device=device)
    goal_std_t  = torch.tensor(norm_stats["goal_std"],  dtype=torch.float32, device=device)
    goal_mean_t = torch.tensor(norm_stats["goal_mean"], dtype=torch.float32, device=device)

    batch_count = 0
    for batch in val_loader:
        if batch_count >= n_batches:
            break
        batch_count += 1

        traj_norm  = batch["trajectory"].to(device)
        start      = batch["start_state"].to(device)
        goal_norm  = batch["goal"].to(device)
        obs_norm   = batch["obstacles"].to(device)
        obs_mask   = batch["obs_mask"].to(device)
        occ_map    = batch["occ_map"].to(device)
        has_map    = batch["has_map"].to(device)
        style_vals = batch["style_values"].to(device)   # (B, N_STYLE_AXES)

        B_scene, T, D = traj_norm.shape

        # ── CFG variant layout (identical to predict() and _sample_and_log) ──
        scene_drop_v = torch.zeros(n_variants, device=device)
        scene_drop_v[0] = 1.0
        style_drop_v = torch.ones(n_variants, N_STYLE_AXES, device=device)
        for i in range(N_STYLE_AXES):
            style_drop_v[2 + i, i] = 0.0

        def _exp(x):
            return x.unsqueeze(1).expand(-1, n_variants, *x.shape[1:]).reshape(
                B_scene * n_variants, *x.shape[1:])

        sd_b         = scene_drop_v.unsqueeze(0).expand(B_scene, -1).reshape(B_scene * n_variants)
        style_drop_b = style_drop_v.unsqueeze(0).expand(B_scene, -1, -1).reshape(
                           B_scene * n_variants, N_STYLE_AXES)
        # Each scene's own style replicated across its n_variants CFG variants
        sv_b         = style_vals.unsqueeze(1).expand(-1, n_variants, -1).reshape(
                           B_scene * n_variants, N_STYLE_AXES)

        style_tok = ema_style_embedder(sv_b, style_drop_b)
        tokens_v, _ = ema_scene_embedder(
            _exp(start), _exp(goal_norm), _exp(obs_norm), _exp(obs_mask),
            occ_map=_exp(occ_map) if args.use_map else None,
            has_map=_exp(has_map.unsqueeze(-1)).squeeze(-1) if args.use_map else None,
            style_tokens=style_tok,
            scene_drop=sd_b,
        )
        attn_mask_v = build_attn_mask(
            _exp(obs_mask),
            n_map_tokens=ema_scene_embedder.n_map_tokens,
            n_style_tokens=N_STYLE_AXES,
        )
        sdv = sd_b.bool().unsqueeze(1)
        attn_mask_v = attn_mask_v.clone()
        attn_mask_v[:, 2:2 + ema_scene_embedder.k_max] = torch.where(
            sdv.expand(-1, ema_scene_embedder.k_max),
            torch.ones_like(attn_mask_v[:, 2:2 + ema_scene_embedder.k_max]),
            attn_mask_v[:, 2:2 + ema_scene_embedder.k_max],
        )

        tokens_bk = (tokens_v.view(B_scene, n_variants, -1, tokens_v.shape[-1])
                              .unsqueeze(2).expand(-1, -1, K, -1, -1)
                              .reshape(B_scene * n_variants * K, -1, tokens_v.shape[-1]))
        attn_bk   = (attn_mask_v.view(B_scene, n_variants, -1)
                                 .unsqueeze(2).expand(-1, -1, K, -1)
                                 .reshape(B_scene * n_variants * K, -1))

        # ── DDIM rollout ──────────────────────────────────────────────────────
        x_in      = torch.randn(B_scene * K, D, T, device=device)
        w_scene   = args.cfg_w_scene
        w_style_t = torch.tensor(args.cfg_w_style, device=device)

        for t_ddim in ddim.timesteps:
            t_tensor = torch.full((B_scene * n_variants * K,), t_ddim,
                                  device=device, dtype=torch.long)
            x_bk = (x_in.view(B_scene, K, D, T)
                        .unsqueeze(1).expand(-1, n_variants, -1, -1, -1)
                        .reshape(B_scene * n_variants * K, D, T))

            noise_bk = ema_model(
                sample=x_bk, timestep=t_tensor,
                encoder_hidden_states=tokens_bk,
                encoder_attention_mask=attn_bk.bool(),
            ).sample

            noise_v    = noise_bk.view(B_scene, n_variants, K, D, T)
            eps_uncond = noise_v[:, 0]
            eps_scene  = noise_v[:, 1]
            eps_axes   = noise_v[:, 2:]

            noise_pred = eps_uncond + w_scene * (eps_scene - eps_uncond)
            for i in range(N_STYLE_AXES):
                noise_pred = noise_pred + w_style_t[i] * (eps_axes[:, i] - eps_scene)

            x_in = ddim.step(noise_pred.reshape(B_scene * K, D, T), t_ddim, x_in).prev_sample

        # ── de-normalise ──────────────────────────────────────────────────────
        x_phys    = (x_in.view(B_scene, K, D, T).permute(0, 1, 3, 2)
                     * traj_std_t + traj_mean_t)              # (B_scene, K, T, 2)
        obs_phys  = obs_norm  * obs_std_t  + obs_mean_t       # (B_scene, k_max, 4)
        goal_phys = goal_norm * goal_std_t + goal_mean_t      # (B_scene, 2)

        # ── best-of-K selection using style-conditioned cost ──────────────────
        # Selection uses the style-conditioned cost so we pick the sample that
        # best satisfies the requested style, matching how the expert was generated.
        scene_costs = np.full((B_scene, K), np.inf)
        for k in range(K):
            cost_k, _ = _val_social_cost_batch(
                x_phys[:, k], obs_phys, obs_mask.float(),
                style_values=style_vals,
            )
            scene_costs[:, k] = cost_k

        best_k     = np.argmin(scene_costs, axis=1)           # (B_scene,)
        best_trajs = x_phys[torch.arange(B_scene),
                             torch.from_numpy(best_k).to(device)]   # (B_scene, T, 2)

        # ── score best trajectory two ways ────────────────────────────────────
        # 1. Style-conditioned: weighted by the sample's own style vector.
        #    Answers "did we produce style-appropriate behavior?"
        cost_styled, bd_styled = _val_social_cost_batch(
            best_trajs, obs_phys, obs_mask.float(),
            style_values=style_vals,
        )

        # 2. Neutral weights: fixed weights for all samples.
        #    Answers "is the trajectory physically reasonable?"
        #    This is the comparable number to use for checkpoint selection.
        cost_neutral, bd_neutral = _val_social_cost_batch(
            best_trajs, obs_phys, obs_mask.float(),
            style_values=None,
        )

        # Score the ground-truth trajectory at the same style weights
        traj_gt_phys = (traj_norm * traj_std_t + traj_mean_t)  # (B_scene, T, 2)
        cost_gt_styled, _  = _val_social_cost_batch(
            traj_gt_phys, obs_phys, obs_mask.float(),
            style_values=style_vals,
        )
        cost_gt_neutral, _ = _val_social_cost_batch(
            traj_gt_phys, obs_phys, obs_mask.float(),
            style_values=None,
        )

        # L2 deviation from GT (physical metres, mean over time)
        l2_vs_gt = (best_trajs - traj_gt_phys).norm(dim=-1).mean(dim=-1).cpu().numpy()  # (B,)

        # ── collision rate ────────────────────────────────────────────────────
        rob_xy  = best_trajs[:, 1:]
        t_vec   = torch.arange(1, T, device=device).float() * _VAL_DT
        obs_fut = (obs_phys[:, :, :2].unsqueeze(1)
                   + obs_phys[:, :, 2:].unsqueeze(1) * t_vec.view(1, T-1, 1, 1))
        dists   = (rob_xy.unsqueeze(2) - obs_fut).norm(dim=-1)
        # Zero out padding so it never triggers a false collision
        dists   = (dists * obs_mask.float().unsqueeze(1)
                   + (1 - obs_mask.float()).unsqueeze(1) * 1e6)
        collided = (dists.min(dim=-1).values.min(dim=-1).values
                    < _VAL_COLLISION_R).float().cpu().numpy()

        # ── goal progress ─────────────────────────────────────────────────────
        start_dist = goal_phys.norm(dim=-1).cpu().numpy()
        end_dist   = (best_trajs[:, -1] - goal_phys).norm(dim=-1).cpu().numpy()
        progress   = start_dist - end_dist

        # ── style-stratified split ────────────────────────────────────────────
        style_mag  = style_vals.norm(dim=-1).cpu().numpy()   # (B_scene,)
        is_styled  = style_mag >= 0.1                        # bool array

        # ── accumulate ────────────────────────────────────────────────────────
        all_cost_neutral.extend(cost_neutral.tolist())
        all_cost_styled.extend(cost_styled.tolist())
        # Use neutral-weight breakdown for the per-term logs (comparable across samples)
        all_c2.extend(bd_neutral["c2_head_on"].tolist())
        all_c3.extend(bd_neutral["c3_sneak_up"].tolist())
        all_c4.extend(bd_neutral["c4_slow_near"].tolist())
        all_c5.extend(bd_neutral["c5_cutoff"].tolist())
        all_c7.extend(bd_neutral["c7_split_group"].tolist())
        all_c8.extend(bd_neutral["c8_smooth"].tolist())
        all_collision.extend(collided.tolist())
        all_progress.extend(progress.tolist())
        all_cost_gt_styled.extend(cost_gt_styled.tolist())
        all_cost_gt_neutral.extend(cost_gt_neutral.tolist())
        all_l2_vs_gt.extend(l2_vs_gt.tolist())

        if is_styled.any():
            all_cost_neutral_styled.extend(cost_neutral[is_styled].tolist())
        if (~is_styled).any():
            all_cost_neutral_neutral.extend(cost_neutral[~is_styled].tolist())

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")

    return {
        # Primary checkpoint selection metric — comparable across all samples
        "val_social/cost_neutral_weights":        _m(all_cost_neutral),
        # Style fidelity metric — higher gap vs neutral means style is being expressed
        "val_social/cost_style_conditioned":      _m(all_cost_styled),
        # Per-term breakdown (neutral weights, so terms are on a fixed scale)
        "val_social/c2_head_on":                  _m(all_c2),
        "val_social/c3_sneak_up":                 _m(all_c3),
        "val_social/c4_slow_near":                _m(all_c4),
        "val_social/c5_cutoff":                   _m(all_c5),
        "val_social/c7_split_group":              _m(all_c7),
        "val_social/c8_smooth":                   _m(all_c8),
        # Physical safety metrics
        "val_social/collision_rate":              _m(all_collision),
        "val_social/goal_progress_m":             _m(all_progress),
        # Style-stratified neutral cost — watch for collapse if these converge
        "val_social/cost_neutral_on_styled_scenes":  _m(all_cost_neutral_styled),
        "val_social/cost_neutral_on_neutral_scenes": _m(all_cost_neutral_neutral),
        "val_social/n_scenes":                    len(all_cost_neutral),
        # GT reference costs — what the expert actually achieved
        "val_social/cost_gt_style_conditioned":  _m(all_cost_gt_styled),
        "val_social/cost_gt_neutral_weights":    _m(all_cost_gt_neutral),
        # Gap: generated - GT. Positive = worse than expert, negative = somehow better
        # This is your primary "are we imitating well" signal
        "val_social/cost_gap_styled":   _m([g - e for g, e in zip(all_cost_styled,  all_cost_gt_styled)]),
        "val_social/cost_gap_neutral":  _m([g - e for g, e in zip(all_cost_neutral, all_cost_gt_neutral)]),
        "val_social/l2_vs_gt_m":        _m(all_l2_vs_gt),
    }

def collision_cost(
    x_trajs,
    obstacles,
    obs_mask,
    dt=0.05,
    safety_radius=0.5,
    sharpness=20.0
):
    """
    x_trajs:   [B, K, D, T]
    obstacles: [B, k_max, 4]  (x,y,vx,vy)
    obs_mask:  [B, k_max]

    returns:
        cost: [B, K]
    """

    B, K, D, T = x_trajs.shape
    k_max = obstacles.shape[1]

    # Robot trajectory positions
    traj_xy = x_trajs[:, :, 0:2, :]         # [B,K,2,T]
    traj_xy = traj_xy.permute(0,1,3,2)      # [B,K,T,2]

    # Obstacle state
    obs_pos = obstacles[:, :, 0:2]          # [B,k_max,2]
    obs_vel = obstacles[:, :, 2:4]          # [B,k_max,2]

    # Predict obstacle future
    t_vec = torch.arange(T, device=x_trajs.device).float() * dt
    t_vec = t_vec.view(1,1,T,1)              # [1,1,T,1]

    obs_pos = obs_pos.unsqueeze(2)           # [B,k_max,1,2]
    obs_vel = obs_vel.unsqueeze(2)           # [B,k_max,1,2]

    obs_future = obs_pos + obs_vel * t_vec   # [B,k_max,T,2]

    # Broadcast for pairwise distance
    traj_xy = traj_xy.unsqueeze(2)           # [B,K,1,T,2]
    obs_future = obs_future.unsqueeze(1)     # [B,1,k_max,T,2]

    diff = traj_xy - obs_future              # [B,K,k_max,T,2]
    dist = torch.norm(diff, dim=-1)          # [B,K,k_max,T]

    # Smooth collision barrier
    # penetration = safety_radius - dist
    # penalty = F.softplus(sharpness * penetration) / sharpness

    # Only penalize actual penetration, zero gradient when safe
    penetration = F.relu(safety_radius - dist)  # exactly 0 when dist >= safety_radius
    penalty = penetration ** 4                   # quartic inside collision zone only

    # Apply obstacle mask
    obs_mask_exp = obs_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,k_max,1]
    penalty = penalty * obs_mask_exp

    # Sum across obstacles + time
    cost = penalty.sum(dim=[2,3])   # [B,K]
    return cost

def start_position_penalty(x_trajs_denorm):
    """
    x_trajs_denorm: [B, K, D, T] in real-world coordinates
    Start is always (0, 0) in world frame, so penalty is just
    the squared distance of the first waypoint from the origin.
    """
    first_wp = x_trajs_denorm[:, :, 0:2, 0]  # [B, K, 2]
    dist = torch.norm(first_wp, dim=-1)        # [B, K] — distance from origin
    # Combine L1 (always has gradient) + L2 (penalizes large drift harder)
    with torch.no_grad():
        print({"START_PEN:":    f"{dist.mean().item() + dist.pow(2).mean().item():.4f}",
        }, flush=True)
    return dist + dist.pow(2)

def goal_progress_reward(x_trajs_denorm, goal, goal_mean, goal_std):
    device = x_trajs_denorm.device

    goal_mean_t = torch.tensor(goal_mean[0:2], dtype=torch.float32, device=device)
    goal_std_t  = torch.tensor(goal_std[0:2],  dtype=torch.float32, device=device)

    goal_xy = goal[:, 0:2] * goal_std_t + goal_mean_t        # [B, 2]
    goal_xy = goal_xy.unsqueeze(1).unsqueeze(3)               # [B, 1, 2, 1]

    traj_xy = x_trajs_denorm[:, :, 0:2, :]                   # [B, K, 2, T]

    dist_to_goal = torch.norm(traj_xy - goal_xy, dim=2)       # [B, K, T]

    # Distance at start and end of trajectory
    start_dist = dist_to_goal[:, :, 0]                        # [B, K]
    final_dist = dist_to_goal[:, :, -1]                       # [B, K]

    # Progress: how much closer did we get? Positive = moved toward goal
    progress = start_dist - final_dist                         # [B, K]

    # Per-step progress: each step should reduce distance (penalize backtracking)
    step_progress = dist_to_goal[:, :, :-1] - dist_to_goal[:, :, 1:]  # [B, K, T-1]
    avg_step_progress = step_progress.mean(dim=2)              # [B, K]

    # Combine: reward net progress + reward consistent per-step progress
    return progress + 0.5 * avg_step_progress                  # [B, K]

def unicycle_feasibility_penalty(
    x_trajs_denorm,
    fixed_heading,
    v0,
    dt=0.05,
    max_speed=2.0,
    max_accel=1.5,
    max_omega=np.pi,   
    max_alpha=np.pi, 
    speed_gate_thresh=0.15, # m/s — below this, heading-derived terms fade out
):
    """ 
    x_trajs_denorm: [B, K, D, T] — ego-frame displacement coordinates
                    x-axis = forward (heading), y-axis = left
    
    Enforces unicycle kinematic feasibility:
      - Speed within [0, max_speed]
      - Linear acceleration within [-max_accel, max_accel]
      - Turn rate (omega) within [-max_omega, max_omega]
      - Angular acceleration within [-max_alpha, max_alpha]
      - No backward motion (negative forward speed) — optional
    
    All penalties are quadratic in violation magnitude → zero when feasible.
    Returns: [B, K]
    """

    """
    Returns [B, K] — all terms normalized to O(1) at limit, 0 when feasible.
    Stopping is allowed. Reverse is penalized via heading consistency, not vx sign.
    Alpha included but noise-gated by speed to prevent dt^2 amplification of noise.
    """
    pos   = x_trajs_denorm[:, :, 0:2, :]                         # [B, K, 2, T]
    vel   = (pos[:, :, :, 1:] - pos[:, :, :, :-1]) / dt          # [B, K, 2, T-1]
    vx    = vel[:, :, 0, :]                                       # [B, K, T-1]
    vy    = vel[:, :, 1, :]
    speed = torch.norm(vel, dim=2)                                # [B, K, T-1]

    # ── 1. Speed limit ────────────────────────────────────────────────────────
    speed_pen = (F.relu(speed - max_speed) / max_speed).pow(2).mean(dim=2)

    # ── 2. Longitudinal acceleration via speed differences ───────────────────
    # Avoids heading projection noise entirely. For a unicycle this is the
    # scalar control input — the signed speed change along direction of motion.
    # Works correctly whether the robot is turning or going straight.
    speed_diff  = speed[:, :, 1:] - speed[:, :, :-1]             # [B, K, T-2]
    a_long      = speed_diff / dt                                  # [B, K, T-2] m/s^2
    accel_pen   = (F.relu(a_long.abs() - max_accel) / max_accel).pow(2).mean(dim=2)

    # ── 3. Reverse motion ────────────────────────────────────────────────────
    # A unicycle moving in reverse has velocity pointing opposite to its heading.
    # heading[t] = atan2(vy[t], vx[t]) is the direction of velocity.
    # At t=0 robot faces +x, so large |heading[0]| means moving sideways/backward.
    # Over the trajectory: penalize the speed-weighted heading magnitude,
    # which captures "moving in a direction far from forward" generically.
    # This gracefully handles turning since a turning robot has a smoothly
    # changing heading, not a sudden large heading.
    heading = torch.atan2(vy, vx)                                 # [B, K, T-1]

    # Soft speed gate — suppresses heading-based terms near zero speed
    # where atan2 is undefined. Smooth sigmoid, no gradient discontinuity.
    gate_t = torch.sigmoid(
        (speed - speed_gate_thresh) / (speed_gate_thresh * 0.3)
    )                                                              # [B, K, T-1], in [0,1]

    # ── 4. Omega — turn rate, speed-gated ────────────────────────────────────
    d_heading   = heading[:, :, 1:] - heading[:, :, :-1]          # [B, K, T-2]
    d_heading   = (d_heading + math.pi) % (2 * math.pi) - math.pi # wrap to [-π, π]
    omega       = d_heading / dt                                   # [B, K, T-2] rad/s

    # Gate uses minimum speed of the two adjacent timesteps
    gate_omega  = torch.min(gate_t[:, :, :-1], gate_t[:, :, 1:]) # [B, K, T-2]

    omega_violation = F.relu(omega.abs() - max_omega) / max_omega  # [B, K, T-2]
    omega_pen   = (omega_violation.pow(2) * gate_omega).mean(dim=2)

    # ── 5. Alpha — angular acceleration, doubly speed-gated ──────────────────
    # alpha = d_omega/dt. Normalize by (max_omega/dt) rather than max_alpha
    # so it stays on the same O(1) scale as omega_pen.
    # Physically: if omega can change by at most max_omega in one step,
    # the worst-case alpha is max_omega/dt. Normalizing by this keeps the
    # term in [0, ~1] for realistic violations.
    d_omega      = omega[:, :, 1:] - omega[:, :, :-1]              # [B, K, T-3]
    alpha        = d_omega / dt                                     # [B, K, T-3] rad/s^2

    # Alpha normalization: use max_omega/dt as the natural scale
    # This is the maximum omega change achievable in one step, so violations
    # beyond this are truly extreme and the term stays O(1)
    alpha_scale  = max_omega / dt                                   # e.g. π/2 / 0.05 = 31 rad/s^2
    # Additionally gate by speed at all three involved timesteps
    gate_alpha   = torch.min(
        torch.min(gate_t[:, :, :-2], gate_t[:, :, 1:-1]),
        gate_t[:, :, 2:]
    )                                                               # [B, K, T-3]

    alpha_violation = F.relu(alpha.abs() - max_alpha) / alpha_scale # [B, K, T-3]
    alpha_pen    = (alpha_violation.pow(2) * gate_alpha).mean(dim=2)

    # ── 6. Initial velocity matching ──────────────────────────────────────────
    speed_init   = speed[:, :, 0]                                  # [B, K]
    if v0.dim() == 1:
        v0_exp = v0.unsqueeze(1).expand(-1, speed_init.shape[1])
    else:
        v0_exp = v0.expand(-1, speed_init.shape[1])
    init_vel_pen = ((speed_init - v0_exp) / max_speed).pow(2)

    # ── 7. Initial heading (fixed_heading mode only) ──────────────────────────
    # In fixed_heading mode, robot starts facing +x, so heading[0] should be ~0.
    # We use the soft-gated heading directly rather than checking vx sign.
    if fixed_heading:
        heading_init     = heading[:, :, 0]                        # [B, K]
        gate_init        = gate_t[:, :, 0]                         # [B, K]
        # Normalize by pi: heading of pi means completely backward
        heading_init_pen = (heading_init / math.pi*8).pow(2) * gate_init
    else:
        heading_init_pen = torch.zeros_like(speed_pen)

    total = (
          speed_pen          # O(1)
        + accel_pen          # O(1)
        + omega_pen          # O(1), speed-gated
        + alpha_pen          # O(1) via alpha_scale normalization, doubly speed-gated
        + init_vel_pen       # O(1)
        + heading_init_pen   # O(1), speed-gated
    )

    # Per-term diagnostics — print means to verify balance
    with torch.no_grad():
        print({
            "speed":    f"{speed_pen.mean().item():.4f}",
            "accel":    f"{accel_pen.mean().item():.4f}",
            "omega":    f"{omega_pen.mean().item():.4f}",
            "alpha":    f"{alpha_pen.mean().item():.4f}",
            "init_vel": f"{init_vel_pen.mean().item():.4f}",
            "hdg_init": f"{heading_init_pen.mean().item():.4f}",
        }, flush=True)

    return total



def _draw_style_panel(ax_s, style_values):
    """
    Render four signed horizontal gauges, one per style axis, into ax_s.
    style_values: array-like of length 4, in [-1, 1], OR None.
    """
    n_axes = len(STYLE_AXIS_LABELS_VIZ)
 
    ax_s.clear()
    ax_s.set_xlim(-1.2, 1.2)
    ax_s.set_ylim(-0.5, n_axes - 0.5)
    ax_s.invert_yaxis()
    ax_s.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_s.set_xticklabels(["-1", "-.5", "0", ".5", "1"], fontsize=8)
    ax_s.set_yticks(range(n_axes))
    ax_s.set_yticklabels([row[0] for row in STYLE_AXIS_LABELS_VIZ],
                         fontsize=9, fontweight="bold")
    ax_s.axvline(0, color="black", linewidth=0.8, alpha=0.5, zorder=1)
    ax_s.set_title("Social style", fontsize=10, pad=6)
    for i in range(n_axes):
        ax_s.axhline(i + 0.5, color="gray", linewidth=0.3, alpha=0.4)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
 
    for i, (_, neg_lbl, pos_lbl) in enumerate(STYLE_AXIS_LABELS_VIZ):
        ax_s.text(-1.18, i - 0.32, neg_lbl, fontsize=6, color="tab:blue",
                  ha="left", va="center", style="italic")
        ax_s.text( 1.18, i - 0.32, pos_lbl, fontsize=6, color="tab:red",
                  ha="right", va="center", style="italic")
 
    if style_values is None:
        ax_s.text(0.0, (n_axes - 1) / 2.0, "no style", fontsize=10,
                  ha="center", va="center", style="italic", color="gray")
        return
 
    vals = np.asarray(style_values, dtype=np.float32).reshape(-1)
    out  = np.zeros(n_axes, dtype=np.float32)
    n_copy = min(vals.shape[0], n_axes)
    out[:n_copy] = vals[:n_copy]
    out = np.clip(out, -1.0, 1.0)
 
    for i, v in enumerate(out):
        height = 0.55
        color  = "tab:red" if v >= 0 else "tab:blue"
        if v >= 0:
            rect = plt.Rectangle((0, i - height / 2), v, height,
                                 color=color, alpha=0.75, zorder=2)
        else:
            rect = plt.Rectangle((v, i - height / 2), -v, height,
                                 color=color, alpha=0.75, zorder=2)
        ax_s.add_patch(rect)
 
        if abs(v) < 0.05:
            ax_s.text(0.0, i + 0.30, f"{v:+.2f}", fontsize=7, color="black",
                      ha="center", va="center")
        else:
            x_label = v + (0.06 if v > 0 else -0.06)
            ha = "left" if v > 0 else "right"
            ax_s.text(x_label, i, f"{v:+.2f}", fontsize=8, color="black",
                      ha=ha, va="center", fontweight="bold")


def log_trajectory_scene(
    gt,
    pred,
    start=None,
    goal=None,
    obs_pos=None,
    obs_vel=None,
    occ_map=None,       # NEW: [H, W] numpy array, binary
    map_extent=None,    # NEW: float, side length in meters (e.g. 10)
    step=0,
    prefix="sample",
    style_values=None,
):
    """
    gt, pred: [T, 2]
    start, goal: [2]
    obs_pos: [N, 2]
    obs_vel: [N, 2]
    occ_map: [H, W] ego-centered binary occupancy map (1 = occupied)
    map_extent: side length in meters; map covers [-extent/2, +extent/2] in both axes
    """

    from matplotlib.gridspec import GridSpec
 
    if style_values is not None:
        fig = plt.figure(figsize=(7, 5))
        gs  = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.30, figure=fig)
        ax       = fig.add_subplot(gs[0, 0])
        ax_style = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax_style = None

    # Occupancy map underlay
    if occ_map is not None and map_extent is not None:
        half = map_extent / 2.0
        ax.imshow(
            occ_map,
            extent=[-half, half, -half, half],
            origin="lower",         # row 0 = -y, NOT top
            cmap="gray_r",
            alpha=0.4,
            zorder=0,
            interpolation="nearest",
        )

    # GT and predicted trajectories
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], "o-", label="GT", alpha=0.8, zorder=3)
    ax.plot(pred[:, 0], pred[:, 1], "x--", label="Pred", alpha=0.8, zorder=3)

    # Start / Goal
    if start is not None:
        if len(start) >= 3:
            ax.scatter(start[0], start[1], c="green", s=100, marker="o", label="Start", zorder=4)
        else:
            ax.scatter(0, 0, c="green", s=100, marker="o", label="Start", zorder=4)
    if goal is not None:
        ax.scatter(goal[0], goal[1], c="red", s=100, marker="*", label="Goal", zorder=4)

    # Obstacles
    if obs_pos is not None:
        ax.scatter(obs_pos[:, 0], obs_pos[:, 1], c="black", s=60, marker="s",
                   label="Obstacle", zorder=4)
        if obs_vel is not None:
            ax.quiver(obs_pos[:, 0], obs_pos[:, 1], obs_vel[:, 0], obs_vel[:, 1],
                      angles="xy", scale_units="xy", scale=1.0, width=0.003,
                      color="black", alpha=0.8, zorder=4)

    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Trajectory @ step {step}")
    ax.grid(True, alpha=0.3)

    # Lock axis to map extent if provided, so we always see the same frame
    # if map_extent is not None:
    #     half = map_extent / 2.0
    #     ax.set_xlim(-half, half)
    #     ax.set_ylim(-half, half)
    if ax_style is not None:
        _draw_style_panel(ax_style, style_values)

    wandb.log({f"scene/{prefix}": wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_trajectory_scene_K(
    gt,
    pred,
    start=None,
    goal=None,
    obs_pos=None,
    obs_vel=None,
    occ_map=None,       
    map_extent=None,    
    step=0,
    prefix="sample",
    style_values=None,
):
    """
    pred: [K, T, 2] tensor
    occ_map: [H, W] numpy ego-centered binary occupancy
    map_extent: float, side length in meters
    """

    from matplotlib.gridspec import GridSpec
 
    K = pred.shape[0]
    if style_values is not None:
        fig = plt.figure(figsize=(7, 5))
        gs  = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.30, figure=fig)
        ax       = fig.add_subplot(gs[0, 0])
        ax_style = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax_style = None

    # Occupancy map underlay
    if occ_map is not None and map_extent is not None:
        half = map_extent / 2.0
        ax.imshow(
            occ_map,
            extent=[-half, half, -half, half],
            origin="lower",
            cmap="gray_r",
            alpha=0.4,
            zorder=0,
            interpolation="nearest",
        )

    traj_np = pred.cpu().numpy()
    for k in range(K):
        t = traj_np[k]
        ax.plot(t[:, 0], t[:, 1], linewidth=2, alpha=0.9, zorder=3)

    if start is not None:
        if len(start) >= 3:
            ax.scatter(start[0], start[1], c="green", s=100, marker="o", label="Start", zorder=4)
        else:
            ax.scatter(0, 0, c="green", s=100, marker="o", label="Start", zorder=4)
    if goal is not None:
        ax.scatter(goal[0], goal[1], c="red", s=120, marker="*", label="Goal", zorder=4)

    if obs_pos is not None:
        ax.scatter(obs_pos[:, 0], obs_pos[:, 1], c="black", s=60, marker="s",
                   label="Obstacle", zorder=4)
        if obs_vel is not None:
            ax.quiver(obs_pos[:, 0], obs_pos[:, 1], obs_vel[:, 0], obs_vel[:, 1],
                      angles="xy", scale_units="xy", scale=1.0, width=0.003,
                      color="black", alpha=0.8, zorder=4)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"K Sampled Trajectories @ step {step}")

    # if map_extent is not None:
    #     half = map_extent / 2.0
    #     ax.set_xlim(-half, half)
    #     ax.set_ylim(-half, half)

    if ax_style is not None:
        _draw_style_panel(ax_style, style_values)

    wandb.log({f"scene/{prefix}": wandb.Image(fig)}, step=step)
    plt.close(fig)



# --- Warmup + cosine decay scheduler ---
def get_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, lr_min=1e-6):
    """
    Returns a LambdaLR scheduler that:
    - linearly warms up LR for `num_warmup_steps`
    - then decays LR following a cosine to `lr_min`
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # cosine decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(lr_min / optimizer.defaults['lr'], 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)





def log_scene_token_importance(unet, step, token_labels=None, batch_idx=0):
    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, CrossAttnCaptureProcessor) or proc.last_attn is None:
            continue
        attn = _get_mean_cross_attn(proc, batch_idx=batch_idx)        # [Q, K]
        importance = attn.mean(axis=0)            # [K] — mean over trajectory queries

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(importance)), importance)
        if token_labels is not None and len(token_labels) == importance.shape[0]:
            ax.set_xticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        ax.set_title(f"Token importance (mean attn over queries)\n{name}")
        ax.set_ylabel("Mean attention")
        wandb.log({f"scene_token_importance/{name}": wandb.Image(fig)}, step=step)
        plt.close(fig)


def log_traj_token_attention(unet, step, batch_idx=0):
    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, CrossAttnCaptureProcessor) or proc.last_attn is None:
            continue
        attn = _get_mean_cross_attn(proc, batch_idx=batch_idx)        # [Q, K]
        traj_importance = attn.mean(axis=1)       # [Q] — mean over scene tokens

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(traj_importance)
        ax.set_title(f"Trajectory-step attention\n{name}")
        ax.set_xlabel("Trajectory timestep (query)")
        ax.set_ylabel("Mean attention over tokens")
        wandb.log({f"traj_token_attention/{name}": wandb.Image(fig)}, step=step)
        plt.close(fig)


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """In-place EMA update of ema_model parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1 - decay)
    # Also update buffers (BN running stats etc.) — copy directly, no decay
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)

# ------------------------
# Dataset - supports variable obstacles via padding and mask
# ------------------------
class RobotTrajectoryDataset(Dataset):
    def __init__(self, dataset_dir, horizon=32, k_max=10, norm_stats = None, files=None, map_size=50, n_style_axes=N_STYLE_AXES):
        """
        dataset_dir: directory with .npz or .npy files
        horizon: number of timesteps per training segment
        k_max: maximum number of obstacles to encode (pad/truncate)
        map_size: size of the occupancy map
        """
        if files is None:
            files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz"))
                           + glob.glob(os.path.join(dataset_dir, "*.npy")))
        self.files = files
        if len(files) == 0:
            raise ValueError(f"No .npz/.npy files found in {dataset_dir}")
        self.horizon = horizon
        self.k_max = k_max
        self.norm_stats = norm_stats
        self.map_size = map_size
        self.n_style_axes = n_style_axes

    def __len__(self):
        return len(self.files)

    def _load_file(self, path):
        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            traj = data["trajectory"] if "trajectory" in data else data["traj_xy"]
            start = data["start_state"] if "start_state" in data else traj[0]
            goal = data["goal"] if "goal" in data else traj[-1]
            obstacles = data["obstacles"] if "obstacles" in data else np.zeros((0,4), dtype=np.float32)
            # NEW: static occupancy map (ego-centered, robot at center cell, +x forward)
            if "has_map" in data.files:
                has_map = np.float32(float(data["has_map"]))
            elif "occupancy_map" in data.files:
                has_map = np.float32(1.0)
            else:
                has_map = np.float32(0.0)

            if "occupancy_map" in data.files:
                occ_map = data["occupancy_map"].astype(np.float32)
                # If the file says no-map, zero the grid as a defensive measure.
                # The model gates on has_map anyway, but this avoids any chance
                # of stale data leaking through if has_map flag fails downstream.
                if has_map < 0.5:
                    occ_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
            else:
                occ_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
                
            # NEW: style values, one scalar per axis, in approximately [-1, 1].
            # When the dataset is generated with mixture sampling, these are the
            # cost-weight settings that produced the trajectory.
            if "style_values" in data.files:
                style_values = data["style_values"].astype(np.float32)
                # Defensive: clip and pad / truncate to n_style_axes if needed
                if style_values.shape[0] != self.n_style_axes:
                    out = np.zeros((self.n_style_axes,), dtype=np.float32)
                    n = min(style_values.shape[0], self.n_style_axes)
                    out[:n] = style_values[:n]
                    style_values = out
            else:
                # Old data without style labels — treat as neutral (zero) style.
                # During training these samples get full style dropout via the
                # joint-null path so they don't contaminate the per-axis signal.
                style_values = np.zeros((self.n_style_axes,), dtype=np.float32)

            style_valid = np.float32(float("style_values" in data.files))
            
        else:
            traj = np.load(path)
            start = traj[0]; goal = traj[-1]
            obstacles = np.zeros((0,4), dtype=np.float32)
            occ_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
            has_map = np.float32(0.0)
            style_values = np.zeros((self.n_style_axes,), dtype=np.float32)
            style_valid = np.float32(0.0)

        return (traj.astype(np.float32), start.astype(np.float32),
                goal.astype(np.float32), obstacles.astype(np.float32),
                occ_map, has_map, style_values, style_valid)
                
    def __getitem__(self, idx):
        traj, start, goal, obstacles, occ_map, has_map, style_values, style_valid = self._load_file(self.files[idx])
        T_full = traj.shape[0]

        # choose a segment of length horizon
        if T_full >= self.horizon:
            # pick random segment
            i = np.random.randint(0, T_full - self.horizon + 1)
            seg = traj[i:i+self.horizon]
        else:
            # pad by repeating last state
            pad_len = self.horizon - T_full
            pad = np.repeat(traj[-1:,:], pad_len, axis=0)
            seg = np.concatenate([traj, pad], axis=0)

        # process obstacles: pad/truncate to k_max, and mask
        N = obstacles.shape[0]
        obs_trunc = np.zeros((self.k_max, 4), dtype=np.float32)
        obs_mask = np.zeros((self.k_max,), dtype=np.float32)

        if N > 0:
            n_to_copy = min(N, self.k_max)
            obs_trunc[:n_to_copy, :] = obstacles[:n_to_copy, :4]
            obs_mask[:n_to_copy] = 1.0

        # Apply normalization if stats provided
        if self.norm_stats is not None:
            start = (start - self.norm_stats["start_mean"]) / self.norm_stats["start_std"]
            goal  = (goal  - self.norm_stats["goal_mean"])  / self.norm_stats["goal_std"]
            seg  = (seg  - self.norm_stats["traj_mean"])  / self.norm_stats["traj_std"]
        
            valid = obs_mask.astype(bool)
            obs_trunc[valid] = (obs_trunc[valid] - self.norm_stats["obs_mean"]) / self.norm_stats["obs_std"]
            # obs_trunc = (obs_trunc - self.norm_stats["obs_mean"]) / self.norm_stats["obs_std"]

        # NOTE: style_values are NOT normalised by norm_stats. They are
        # already in [-1, 1] by construction; preserving that range is
        # important for clean CFG behaviour.

        sample = {
            "trajectory": torch.from_numpy(seg),
            "start_state": torch.from_numpy(start),
            "goal": torch.from_numpy(goal),
            "obstacles": torch.from_numpy(obs_trunc),
            "obs_mask": torch.from_numpy(obs_mask),
            "occ_map": torch.from_numpy(occ_map).unsqueeze(0),  # [1, H, W]
            "has_map": torch.tensor(has_map),                    # scalar
            "style_values": torch.from_numpy(style_values),     # [n_axes]
            "style_valid":  torch.tensor(style_valid),          # scalar in {0, 1}
        }
        return sample


class MapEncoder(nn.Module):
    """
    Encodes a [B, 1, H, W] ego-centered occupancy map into a sequence of
    spatially-aware tokens with 2D positional embeddings.

    Default: 50x50 input → 7x7 feature map → 49 tokens of dim `token_dim`.
    """
    def __init__(self, token_dim, in_size=50, base_channels=32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        # 3 stride-2 stages: 50 -> 25 -> 13 -> 7 (with padding=1)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  c1, kernel_size=3, stride=2, padding=1),  # 50 -> 25
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),  # 25 -> 13
            nn.GroupNorm(8, c2),
            nn.SiLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),  # 13 -> 7
            nn.GroupNorm(8, c3),
            nn.SiLU(),
        )
        # Compute output spatial size with a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_size, in_size)
            feat = self.cnn(dummy)
            _, _, self.h_out, self.w_out = feat.shape
        self.n_tokens = self.h_out * self.w_out

        # Project CNN channels to token_dim
        self.proj = nn.Linear(c3, token_dim)

        # 2D positional embedding (learned, one per cell)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_tokens, token_dim) * 0.02
        )
        # Distinct type embedding for map tokens
        self.type_embed = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # Learned "null map" tokens for when no map is provided
        self.null_map_tokens = nn.Parameter(
            torch.randn(1, self.n_tokens, token_dim) * 0.02
        )

        self.norm = nn.LayerNorm(token_dim)

    def forward(self, occ_map, has_map):
        """
        occ_map: [B, 1, H, W]   binary
        has_map: [B]            1.0 if map present, 0.0 otherwise
        returns: [B, n_tokens, token_dim]
        """
        B = occ_map.shape[0]
        feat = self.cnn(occ_map)                  # [B, C, h, w]
        feat = feat.flatten(2).transpose(1, 2)    # [B, h*w, C]
        tokens = self.proj(feat)                  # [B, n_tokens, token_dim]
        tokens = tokens + self.pos_embed + self.type_embed

        # Replace with null tokens where has_map == 0
        gate = has_map.view(B, 1, 1)              # [B, 1, 1]
        tokens = gate * tokens + (1.0 - gate) * self.null_map_tokens.expand(B, -1, -1)

        return self.norm(tokens)


# =============================================================================
# NEW: style token embedder.
#   - One MLP per axis (scalar -> token_dim).
#   - One learned null token per axis (used when an axis is CFG-dropped).
#   - One type embedding per axis (helps cross-attention distinguish them).
#   - Plain scalar -> MLP path (NO sinusoidal expansion); style values live
#     in [-1, 1] and we want smooth interpolation / smooth extrapolation
#     under CFG weight, not high-frequency discrimination of nearby values.
# =============================================================================
class StyleTokenEmbedder(nn.Module):
    def __init__(self, axis_names, token_dim):
        super().__init__()
        self.axis_names = list(axis_names)
        self.n_axes = len(self.axis_names)
        self.token_dim = token_dim
        hidden_dim = token_dim
        # Per-axis MLP from scalar to token_dim.
        self.axis_mlps = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, token_dim),
            )
            for name in self.axis_names
        })
        # One type embedding per axis (acts like a positional code for axes).
        self.type_embed = nn.Parameter(torch.randn(self.n_axes, token_dim) * 0.02)
        # One learned null token per axis.
        self.null_tokens = nn.Parameter(torch.randn(self.n_axes, token_dim) * 0.02)
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, style_values, style_drop_mask):
        """
        style_values:    [B, n_axes]   floats roughly in [-1, 1]
        style_drop_mask: [B, n_axes]   1.0 = use null token, 0.0 = use real token
        returns:         [B, n_axes, token_dim]
        """
        B = style_values.shape[0]
        out = []
        for i, name in enumerate(self.axis_names):
            scalar = style_values[:, i:i + 1]                              # [B, 1]
            tok = self.axis_mlps[name](scalar) + self.type_embed[i]        # [B, token_dim]
            null = self.null_tokens[i].unsqueeze(0).expand(B, -1)          # [B, token_dim]
            drop = style_drop_mask[:, i:i + 1]                              # [B, 1]
            tok = drop * null + (1.0 - drop) * tok
            out.append(tok.unsqueeze(1))                                    # [B, 1, token_dim]
        out = torch.cat(out, dim=1)                                         # [B, n_axes, D]
        return self.final_norm(out)


# =============================================================================
# Scene token embedder — extended to:
#   - Apply scene-wide dropout via per-section null tokens (cleaner than the
#     old monolithic null_token Parameter that mixed scene + map).
#   - Accept style tokens from StyleTokenEmbedder and concatenate them into
#     the shared self-attention encoder so style tokens can attend to scene
#     tokens (and vice versa).
#   - Style nullification is handled inside StyleTokenEmbedder, so this class
#     only needs to know about scene-side dropout.
# =============================================================================

class SceneTokenEmbedder(nn.Module):
    def __init__(self, start_dim, goal_dim, obs_dim, token_dim, k_max,
                 n_self_attn_layers=2, map_size=50, use_map=True):
        super().__init__()
        hidden_dim = token_dim
        self.use_map = use_map
        self.k_max = k_max 

        def make_mlp(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, token_dim),
            )

        self.start_proj = make_mlp(start_dim)
        self.goal_proj  = make_mlp(goal_dim)
        self.obs_proj   = make_mlp(obs_dim)
        self.type_embed = nn.Parameter(torch.randn(3, token_dim) * 0.02)

        if use_map:
            self.map_encoder = MapEncoder(token_dim=token_dim, in_size=map_size)
            n_map_tokens = self.map_encoder.n_tokens
        else:
            self.map_encoder = None
            n_map_tokens = 0
        self.n_map_tokens = n_map_tokens

        # Per-section null tokens. Each is used when the corresponding scene
        # section is CFG-dropped. (Map has its own null inside MapEncoder via
        # `has_map`, so we don't duplicate it here — see forward.)
        self.null_start_token = nn.Parameter(torch.randn(1, 1,     token_dim) * 0.02)
        self.null_goal_token  = nn.Parameter(torch.randn(1, 1,     token_dim) * 0.02)
        self.null_obs_tokens  = nn.Parameter(torch.randn(1, self.k_max, token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=4,
            dim_feedforward=token_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.scene_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_self_attn_layers
        )
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, start, goal, obstacles, obs_mask,
                occ_map=None, has_map=None,
                style_tokens=None,            # [B, n_axes, D] from StyleTokenEmbedder
                scene_drop=None,              # [B] in {0., 1.} — 1.0 = drop all scene
                ):
        """
        Builds the full token sequence:
            [ start | goal | obstacles | (map) | (style) ]

        scene_drop is the SCENE-wide CFG dropout indicator: when 1.0 we replace
        start, goal, obstacles, and map (via has_map = 0) with their learned null
        tokens. Style tokens are independent and are handled by StyleTokenEmbedder.
        """
        B = start.shape[0]
        device = start.device

        if scene_drop is None:
            scene_drop = torch.zeros(B, device=device)

        sd = scene_drop.view(B, 1, 1)  # broadcast helper

        # Start / goal / obs tokens with per-section dropout.
        start_tok_real = self.start_proj(start).unsqueeze(1)
        start_tok = sd * self.null_start_token + (1.0 - sd) * start_tok_real
        start_tok = start_tok + self.type_embed[0]

        goal_tok_real = self.goal_proj(goal).unsqueeze(1)
        goal_tok = sd * self.null_goal_token + (1.0 - sd) * goal_tok_real
        goal_tok = goal_tok + self.type_embed[1]

        obs_tok_real = self.obs_proj(obstacles)
        obs_tok = sd * self.null_obs_tokens + (1.0 - sd) * obs_tok_real
        obs_tok = obs_tok + self.type_embed[2]
        
        tokens = torch.cat([start_tok, goal_tok, obs_tok], dim=1)  # [B, 2+K, D]

        # Validity mask for self-attention encoder.
        # When scene is dropped, every position is still valid (we want the
        # nulls to participate in attention), so we OR the obs_mask with the
        # scene_drop flag.
        scene_drop_bool = scene_drop.bool().unsqueeze(1)  # [B, 1]
        obs_valid = obs_mask.bool() | scene_drop_bool      # [B, k_max]
        valid = torch.cat(
            [torch.ones(B, 2, device=device, dtype=torch.bool),
             obs_valid], dim=1
        )

        # Map tokens — gated by has_map; scene-drop forces has_map -> 0.

        # Append map tokens (always valid — null embedding handles absent maps)
        if self.use_map:
            assert occ_map is not None and has_map is not None
            effective_has_map = has_map * (1.0 - scene_drop)
            map_tokens = self.map_encoder(occ_map, effective_has_map)  # [B, n_map, D]
            tokens = torch.cat([tokens, map_tokens], dim=1)
            valid = torch.cat(
                [valid, torch.ones(B, self.n_map_tokens, device=device, dtype=torch.bool)],
                dim=1,
            )

        # Style tokens — already-nullified per-axis by StyleTokenEmbedder.
        if style_tokens is not None:
            n_style = style_tokens.shape[1]
            tokens = torch.cat([tokens, style_tokens], dim=1)
            valid = torch.cat(
                [valid, torch.ones(B, n_style, device=device, dtype=torch.bool)],
                dim=1,
            )

        key_padding_mask = ~valid  # True = ignore in self-attention
        tokens = self.scene_encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.final_norm(tokens), valid.float()




# =============================================================================
# Build the cross-attention mask that the UNet sees.
# Includes style tokens (always valid; nullification is internal to the
# embedder, the cross-attention mask is just "is there a token here at all").
# =============================================================================
def build_attn_mask(obs_mask, n_map_tokens=0, n_style_tokens=0):
    B, K = obs_mask.shape
    device = obs_mask.device
    base = torch.ones(B, 2, device=device)
    pieces = [base, obs_mask]
    if n_map_tokens > 0:
        pieces.append(torch.ones(B, n_map_tokens, device=device))
    if n_style_tokens > 0:
        pieces.append(torch.ones(B, n_style_tokens, device=device))
    return torch.cat(pieces, dim=1)



@torch.no_grad()
def compute_validation_loss(model, scene_embedder, style_embedder, dataloader, noise_scheduler, args, device):
    """
    Returns a dict with three loss values:
        'val/loss'         — overall mean (matches previous behavior)
        'val/loss_with_map'    — mean over samples where has_map == 1
        'val/loss_without_map' — mean over samples where has_map == 0

    Per-sample losses (rather than per-batch) so the with/without split is
    correct regardless of how the batch's has_map mix changes.
    """
    model.eval()
    sum_loss_all     = 0.0
    sum_loss_map     = 0.0
    sum_loss_nomap   = 0.0
    n_all            = 0
    n_map            = 0
    n_nomap          = 0

    with torch.random.fork_rng(enabled=True):
        torch.manual_seed(42)
        for batch in dataloader:
            traj      = batch["trajectory"].to(device)
            start     = batch["start_state"].to(device)
            goal      = batch["goal"].to(device)
            obstacles = batch["obstacles"].to(device)
            obs_mask  = batch["obs_mask"].to(device)
            occ_map   = batch["occ_map"].to(device)
            has_map   = batch["has_map"].to(device)   # [B]
            style_values = batch["style_values"].to(device)

            B, T, D = traj.shape

            # No dropout at validation time — we measure standard conditional loss.
            scene_drop = torch.zeros(B, device=device)
            style_drop = torch.zeros(B, N_STYLE_AXES, device=device)

            style_tokens = style_embedder(style_values, style_drop)
            tokens, _ = scene_embedder(
                start, goal, obstacles, obs_mask,
                occ_map=occ_map, has_map=has_map,
                style_tokens=style_tokens,
                scene_drop=scene_drop,
            )
            attn_mask = build_attn_mask(
                obs_mask,
                n_map_tokens=scene_embedder.n_map_tokens,
                n_style_tokens=N_STYLE_AXES,
            )

            t_batch = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                    (B,), device=device)
            noise = torch.randn_like(traj)
            x_noisy = noise_scheduler.add_noise(
                traj.permute(0, 2, 1), noise.permute(0, 2, 1), t_batch
            )

            pred_noise = model(
                sample=x_noisy, timestep=t_batch,
                encoder_hidden_states=tokens,
                encoder_attention_mask=attn_mask.bool(),
            ).sample

            target = noise.permute(0, 2, 1)
            per_sample_loss = F.mse_loss(pred_noise, target, reduction="none").mean(dim=(1, 2))

            sum_loss_all += per_sample_loss.sum().item()
            n_all += B

            map_mask = has_map > 0.5
            nomap_mask = ~map_mask
            if map_mask.any():
                sum_loss_map += per_sample_loss[map_mask].sum().item()
                n_map += int(map_mask.sum().item())
            if nomap_mask.any():
                sum_loss_nomap += per_sample_loss[nomap_mask].sum().item()
                n_nomap += int(nomap_mask.sum().item())

    model.train()
    return {
        "val/loss":              sum_loss_all / max(1, n_all),
        "val/loss_with_map":     sum_loss_map / max(1, n_map) if n_map > 0 else float("nan"),
        "val/loss_without_map":  sum_loss_nomap / max(1, n_nomap) if n_nomap > 0 else float("nan"),
        "val/n_with_map":        n_map,
        "val/n_without_map":     n_nomap,
    }


# ------------------------
# Training loop with checkpointing & resume
# ------------------------
def train_loop(model, scene_embedder, style_embedder, ema_model, ema_scene_embedder, ema_style_embedder, dataloader, val_loader, norm_stats, optimizer, lr_scheduler, noise_scheduler, args, device):
    train_losses = []
    val_losses = []

    # resume logic
    ckpt_dir = os.path.join("checkpoints", args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_step") and f.endswith(f"_{args.exp_name}.pt")],
        key=lambda x: int(x.split("step")[1].split("_")[0])
    )
    start_step = 0
    if len(ckpt_files) > 0:
        latest = ckpt_files[-1]
        ckpt = torch.load(os.path.join(ckpt_dir, latest), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        scene_embedder.load_state_dict(ckpt["scene_embedder_state_dict"])
        style_embedder.load_state_dict(ckpt["style_embedder_state_dict"])
        if "ema_model_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model_state_dict"])
            ema_scene_embedder.load_state_dict(ckpt["ema_scene_embedder_state_dict"])
            ema_style_embedder.load_state_dict(ckpt["ema_style_embedder_state_dict"])
        else:
            # backward compatibility: initialize EMA from model
            ema_model.load_state_dict(ckpt["model_state_dict"])
            ema_scene_embedder.load_state_dict(ckpt["scene_embedder_state_dict"])
            ema_style_embedder.load_state_dict(ckpt["style_embedder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        print(f"Resuming from checkpoint {latest} at step {start_step}")

    model.train()
    step = start_step
    save_every = args.save_every

    while step < args.num_steps:
        for batch in dataloader:
            traj = batch["trajectory"].to(device)        # [B, T, D]
            start = batch["start_state"].to(device)           # [B, start_dim]
            goal = batch["goal"].to(device)             # [B, goal_dim]
            obstacles = batch["obstacles"].to(device)   # [B, k_max, 4]
            obs_mask = batch["obs_mask"].to(device)     # [B, k_max]
            occ_map   = batch["occ_map"].to(device)
            has_map   = batch["has_map"].to(device)
            style_values = batch["style_values"].to(device)
            style_valid  = batch["style_valid"].to(device)         # [B] in {0., 1.}

            B, T, D = traj.shape

            # ----------------------------------------------------------------
            # CFG dropout policy
            # ----------------------------------------------------------------
            # (a) Scene-wide dropout, independent per sample.
            scene_drop = (torch.rand(B, device=device) < args.scene_cfg_dropout).float()

            # (b) Per-axis style dropout, independent per sample and per axis.
            style_drop = (torch.rand(B, N_STYLE_AXES, device=device)
                          < args.style_cfg_dropout).float()

            # (c) Joint full-null draw on top: with a small probability, force
            #     EVERYTHING to be dropped. This anchors the unconditional path
            #     and ensures the model sees the full-null configuration often
            #     enough to learn a clean prior.
            joint_null = (torch.rand(B, device=device) < args.joint_null_rate).float()
            scene_drop = torch.clamp(scene_drop + joint_null, max=1.0)
            style_drop = torch.clamp(style_drop + joint_null.unsqueeze(1), max=1.0)

            # (d) Defensive: samples that don't have style labels (style_valid=0)
            #     get all-style-dropped. This means old / unlabelled data falls
            #     onto the unconditional path along the style axes; it still
            #     supplies useful scene-conditional supervision.
            unlabelled = (1.0 - style_valid).unsqueeze(1)          # [B, 1]
            style_drop = torch.clamp(style_drop + unlabelled, max=1.0)

            # ----------------------------------------------------------------
            # Build tokens.
            # ----------------------------------------------------------------
            style_tokens = style_embedder(style_values, style_drop)
            tokens, _ = scene_embedder(
                start, goal, obstacles, obs_mask,
                occ_map=occ_map, has_map=has_map,
                style_tokens=style_tokens,
                scene_drop=scene_drop,
            )
            attn_mask = build_attn_mask(
                obs_mask,
                n_map_tokens=scene_embedder.n_map_tokens,
                n_style_tokens=N_STYLE_AXES,
            )
            # When scene is dropped, all token positions are valid (model
            # attends to the null tokens we inserted). When obstacles are
            # padded, the standard padding mask still applies — except when
            # scene_drop=1, in which case we force them valid here too.
            scene_drop_bool = scene_drop.bool().unsqueeze(1)
            # obs occupies positions [2:2+k_max]
            attn_mask = attn_mask.clone()
            attn_mask[:, 2:2 + scene_embedder.k_max] = torch.where(
                scene_drop_bool.expand(-1, scene_embedder.k_max),
                torch.ones_like(attn_mask[:, 2:2 + scene_embedder.k_max]),
                attn_mask[:, 2:2 + scene_embedder.k_max],
            )

            # ----------------------------------------------------------------
            # Diffusion forward + loss.
            # ----------------------------------------------------------------
            t_batch = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                    (B,), device=device)
            noise = torch.randn_like(traj)
            x_noisy = noise_scheduler.add_noise(
                traj.permute(0,2,1),
                noise.permute(0,2,1),
                t_batch
            )

            # pred_x0 = model(
            #     sample=x_noisy,
            #     timestep=t_batch,
            #     encoder_hidden_states=tokens,
            #     encoder_attention_mask=attn_mask.bool()
            # ).sample

            # loss = F.mse_loss(pred_x0, traj.permute(0, 2, 1))   # ← target is clean traj, not noise

            pred_noise = model(
                sample=x_noisy,
                timestep=t_batch,
                encoder_hidden_states=tokens,
                encoder_attention_mask=attn_mask.bool()
            ).sample

            loss = F.mse_loss(pred_noise, noise.permute(0,2,1))

            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(scene_embedder.parameters())
                + list(style_embedder.parameters()),
                max_norm=args.grad_clip,
            )
            optimizer.step()
            lr_scheduler.step()  # update LR

            # EMA update — start after warmup
            if step >= args.ema_warmup_steps:
                ema_update(ema_model, model, args.ema_decay)
                ema_update(ema_scene_embedder, scene_embedder, args.ema_decay)
                ema_update(ema_style_embedder, style_embedder, args.ema_decay)
            else:
                # Before warmup, keep EMA in sync with model
                ema_update(ema_model, model, decay=0.0)
                ema_update(ema_scene_embedder, scene_embedder, decay=0.0)
                ema_update(ema_style_embedder, style_embedder, decay=0.0)

            wandb.log({
                "train/loss":               loss.item(),
                "diffusion/timestep_mean":  t_batch.float().mean().item(),
                "diffusion/noise_std":      noise.std().item(),
                "train/lr":                 lr_scheduler.get_last_lr()[0],
                "cfg/scene_effective_drop":     scene_drop.mean().item(),
                "cfg/style_effective_drop":     style_drop.mean().item(),
                "cfg/joint_null_rate":      joint_null.mean().item(),
                "cfg/style_unlabelled_frac": (1.0 - style_valid).mean().item(),
            }, step=step)

            if step > 0 and step % save_every == 0:

                val_metrics = compute_validation_loss(
                    ema_model, ema_scene_embedder, ema_style_embedder,
                    val_loader, noise_scheduler, args, device
                )
                val_losses.append(val_metrics["val/loss"])

                wandb.log(val_metrics, step=step)

                # ── NEW: social cost validation (rollout-based) ───────────────
                # Run every save_every steps but only over 10 batches to keep
                # wall-time reasonable. Increase n_batches at end of training
                # for a full sweep.
                social_metrics = compute_social_validation_metrics(
                    ema_model, ema_scene_embedder, ema_style_embedder,
                    val_loader, noise_scheduler, norm_stats, args, device,
                    n_batches=10,
                )
                wandb.log(social_metrics, step=step)
                # ─────────────────────────────────────────────────────────────

                print(
                    f"[Step {step}] train={loss.item():.6f}  "
                    f"val={val_metrics['val/loss']:.6f}  "
                    f"social_cost={social_metrics['val_social/cost_neutral_weights']:.3f}  "
                    f"collision={social_metrics['val_social/collision_rate']:.3f}  "
                    f"progress={social_metrics['val_social/goal_progress_m']:.2f}m  "
                    f"val_map={val_metrics['val/loss_with_map']:.6f} (n={val_metrics['val/n_with_map']})  "
                    f"val_nomap={val_metrics['val/loss_without_map']:.6f} (n={val_metrics['val/n_without_map']})"
                )

                # Save checkpoint — use social cost as the primary quality signal
                # rather than MSE val loss. Lower = better.
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step}_{args.exp_name}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict":               model.state_dict(),
                    "ema_model_state_dict":           ema_model.state_dict(),
                    "scene_embedder_state_dict":      scene_embedder.state_dict(),
                    "ema_scene_embedder_state_dict":  ema_scene_embedder.state_dict(),
                    "style_embedder_state_dict":      style_embedder.state_dict(),
                    "ema_style_embedder_state_dict":  ema_style_embedder.state_dict(),
                    "optimizer_state_dict":           optimizer.state_dict(),
                    "lr_scheduler_state_dict":        lr_scheduler.state_dict(),
                    "train_losses":  train_losses,
                    "val_losses":    val_losses,
                    "style_axes":    STYLE_AXES,
                    # Primary checkpoint selection metric:
                    "val_social_cost": social_metrics["val_social/cost_neutral_weights"],
                    "val_mse_loss":    val_metrics["val/loss"],
                }, ckpt_path)
                # print(f"[Step {step}] train={loss.item():.6f}, val={val_loss:.6f}")

                _sample_and_log(
                    ema_model, ema_scene_embedder, ema_style_embedder,
                    val_loader, noise_scheduler, norm_stats, args, device, step
                )

                model.train()
 
            step += 1
            if step >= args.num_steps:
                break



@torch.no_grad()
def _log_style_sweep_grid(ema_model, ema_scene_embedder, ema_style_embedder,
                          val_sample, noise_scheduler, norm_stats, args, device, step):
    """
    Sample one trajectory per (axis, axis_value) for axis_value in {-1, 0, +1}
    on a fixed scene, using compositional CFG. Plot the four axes as a 2x2
    grid where each panel overlays the three trajectories for that axis.
 
    Compute cost: this batches all 12 style settings together so the model
    runs once per denoising step with batch size (n_variants * 12). At 100
    DDPM steps that's a few hundred ms on a modern GPU.
    """
    import matplotlib.pyplot as plt
 
    # ---- Unpack the scene (same as in _sample_and_log) ----
    start     = val_sample["start_state"].unsqueeze(0).to(device)
    goal      = val_sample["goal"].unsqueeze(0).to(device)
    obstacles = val_sample["obstacles"].unsqueeze(0).to(device)
    obs_mask  = val_sample["obs_mask"].unsqueeze(0).to(device)
    occ_map   = val_sample["occ_map"].unsqueeze(0).to(device)
    has_map   = val_sample["has_map"].unsqueeze(0).to(device)
    D = val_sample["trajectory"].shape[1]
    T = args.horizon
 
    # ---- Define the 12 style configurations: 4 axes x {-1, 0, +1} ----
    # Sweep values; -1 and +1 are the extremes, 0 is neutral and is the
    # SAME trajectory for every axis (so the neutral curve will be
    # identical across panels — that's expected and is a useful sanity
    # check that the per-axis MLPs really do gate cleanly when the axis
    # is at its identity setting).
    sweep_values = [-1.0, 0.0, +1.0]
    n_per_axis   = len(sweep_values)
    n_styles     = N_STYLE_AXES * n_per_axis           # 4 x 3 = 12
 
    style_settings = torch.zeros(n_styles, N_STYLE_AXES, device=device)
    style_meta     = []                                # (axis_idx, sweep_val) per setting
    for a in range(N_STYLE_AXES):
        for j, v in enumerate(sweep_values):
            row = a * n_per_axis + j
            style_settings[row, a] = float(v)
            style_meta.append((a, v))
 
    # ---- Build the 6 CFG variants for each of the 12 styles -----------------
    # For each style setting we need: uncond, scene-only, and one axis-only
    # variant per style axis. Total = n_styles * (2 + N_STYLE_AXES) = 12 * 6 = 72
    # conditioning variants per denoising step. We replicate-and-flatten so the
    # model sees a single batch.
    n_variants_per = 1 + 1 + N_STYLE_AXES     # 6
    n_total        = n_styles * n_variants_per
 
    # scene_drop_v has shape [n_variants_per] — same pattern as _sample_and_log
    scene_drop_template = torch.zeros(n_variants_per, device=device)
    scene_drop_template[0]   = 1.0                       # uncond
 
    style_drop_template = torch.ones(n_variants_per, N_STYLE_AXES, device=device)
    for i in range(N_STYLE_AXES):
        style_drop_template[2 + i, i] = 0.0              # axis i kept

 
    # Repeat per style setting: shape [n_styles, n_variants_per, ...] -> flatten
    scene_drop_full = scene_drop_template.unsqueeze(0).expand(n_styles, -1)
    scene_drop_full = scene_drop_full.reshape(n_total)
 
    style_drop_full = style_drop_template.unsqueeze(0).expand(n_styles, -1, -1)
    style_drop_full = style_drop_full.reshape(n_total, N_STYLE_AXES)
 
    # style_values per row: same across all 6 variants of the same setting
    style_values_full = style_settings.unsqueeze(1).expand(-1, n_variants_per, -1)
    style_values_full = style_values_full.reshape(n_total, N_STYLE_AXES)
 
    # ---- Replicate scene inputs across all 72 variants ----
    start_v     = start.expand(n_total, -1).contiguous()
    goal_v      = goal.expand(n_total, -1).contiguous()
    obstacles_v = obstacles.expand(n_total, -1, -1).contiguous()
    obs_mask_v  = obs_mask.expand(n_total, -1).contiguous()
    occ_map_v   = occ_map.expand(n_total, -1, -1, -1).contiguous()
    has_map_v   = has_map.expand(n_total).contiguous()
 
    style_tok_v = ema_style_embedder(style_values_full, style_drop_full)
    tokens_v, _ = ema_scene_embedder(
        start_v, goal_v, obstacles_v, obs_mask_v,
        occ_map=occ_map_v, has_map=has_map_v,
        style_tokens=style_tok_v,
        scene_drop=scene_drop_full,
    )
    attn_mask_v = build_attn_mask(
        obs_mask_v,
        n_map_tokens=ema_scene_embedder.n_map_tokens,
        n_style_tokens=N_STYLE_AXES,
    )
    # When scene is dropped, force the obstacle slots to be valid so the
    # transformer attends to the null obstacle tokens we inserted.
    sdv = scene_drop_full.bool().unsqueeze(1)
    attn_mask_v = attn_mask_v.clone()
    attn_mask_v[:, 2:2 + ema_scene_embedder.k_max] = torch.where(
        sdv.expand(-1, ema_scene_embedder.k_max),
        torch.ones_like(attn_mask_v[:, 2:2 + ema_scene_embedder.k_max]),
        attn_mask_v[:, 2:2 + ema_scene_embedder.k_max],
    )
 
    # ---- Denoise ----
    x_in = torch.randn(n_styles, D, T, device=device)         # one noise init per style setting
    # Per denoising step we need [n_total] copies of x_in (each style setting
    # uses the SAME noise init across its 6 CFG variants).
    w_scene = args.cfg_w_scene
    w_style = torch.tensor(args.cfg_w_style, device=device)   # [N_STYLE_AXES]
 
    for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
        t_tensor = torch.full((n_total,), t, device=device, dtype=torch.long)
        # Replicate x_in across the 6 variants per style:
        x_in_b = x_in.unsqueeze(1).expand(-1, n_variants_per, -1, -1).contiguous()
        x_in_b = x_in_b.reshape(n_total, D, T)
 
        noise_b = ema_model(
            sample=x_in_b, timestep=t_tensor,
            encoder_hidden_states=tokens_v,
            encoder_attention_mask=attn_mask_v.bool(),
        ).sample                                              # [n_total, D, T]
 
        # Reshape to [n_styles, n_variants_per, D, T] so we can combine.
        noise_b = noise_b.view(n_styles, n_variants_per, D, T)
 
        eps_uncond = noise_b[:, 0]                            # [n_styles, D, T]
        eps_scene  = noise_b[:, 1]
        eps_axes   = noise_b[:, 2:]                           # [n_styles, N_AXES, D, T]
 
        # Compositional CFG, matching _sample_and_log and the inference policy.
        noise_pred = eps_uncond + w_scene * (eps_scene - eps_uncond)
        for i in range(N_STYLE_AXES):
            # eps_axes[:, i] picks the i-th axis variant for ALL n_styles
            # rows simultaneously — shape [n_styles, D, T].
            noise_pred = noise_pred + w_style[i] * (eps_axes[:, i] - eps_scene)
 
        step_out = noise_scheduler.step(noise_pred, t, x_in)
        x_in = step_out.prev_sample
 
    # ---- Denormalise trajectories ----
    # x_in is [n_styles, D, T] → permute to [n_styles, T, D] for plotting.
    trajs = x_in.permute(0, 2, 1).cpu().numpy()               # [n_styles, T, D]
    traj_std  = norm_stats["traj_std"]
    traj_mean = norm_stats["traj_mean"]
    trajs_denorm = trajs * traj_std[None, None, :] + traj_mean[None, None, :]
 
    # Scene context — denorm for plotting backdrop
    obs       = obstacles[0]
    mask_bool = obs_mask[0].bool()
    obs_valid = obs[mask_bool]
    if obs_valid.shape[0] > 0:
        obs_pos = obs_valid[:, 0:2].cpu().numpy() * norm_stats["obs_std"][0:2] + norm_stats["obs_mean"][0:2]
        obs_vel = obs_valid[:, 2:4].cpu().numpy() * norm_stats["obs_std"][2:4] + norm_stats["obs_mean"][2:4]
    else:
        obs_pos = None
        obs_vel = None
    goal_np = goal[0].cpu().numpy() * norm_stats["goal_std"] + norm_stats["goal_mean"]
    if has_map[0].item() > 0.5:
        occ_map_np = occ_map[0, 0].cpu().numpy()
        map_extent = args.map_extent
    else:
        occ_map_np = None
        map_extent = None
 
    # ---- Build the 2x2 grid figure ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.flatten()
 
    sweep_colors = {-1.0: "tab:blue", 0.0: "gray", +1.0: "tab:red"}
    sweep_labels = {-1.0: "s=-1",     0.0: "s=0",  +1.0: "s=+1"}
    axis_titles = [
        "prox  (close ←→ conservative)",
        "pass  (left-hand ←→ right-hand)",
        "yield (assertive ←→ yielding)",
        "group (indifferent ←→ group-aware)",
    ]
 
    # Compute a global view box so the four panels share the same axes
    # (otherwise each panel autoscales and you can't compare across them).
    all_x = trajs_denorm[..., 0].flatten()
    all_y = trajs_denorm[..., 1].flatten()
    pad = 0.8
    xlim = (min(0.0, all_x.min(), goal_np[0]) - pad,
            max(0.0, all_x.max(), goal_np[0]) + pad)
    ylim = (min(0.0, all_y.min(), goal_np[1]) - pad,
            max(0.0, all_y.max(), goal_np[1]) + pad)
    if obs_pos is not None:
        xlim = (min(xlim[0], obs_pos[:, 0].min() - pad),
                max(xlim[1], obs_pos[:, 0].max() + pad))
        ylim = (min(ylim[0], obs_pos[:, 1].min() - pad),
                max(ylim[1], obs_pos[:, 1].max() + pad))
 
    for a in range(N_STYLE_AXES):
        ax = axes[a]
 
        # Map underlay
        if occ_map_np is not None and map_extent is not None:
            half = map_extent / 2.0
            ax.imshow(occ_map_np, extent=[-half, half, -half, half],
                      origin="lower", cmap="gray_r", alpha=0.3,
                      zorder=0, interpolation="nearest", vmin=0, vmax=1)
 
        # Scene backdrop (light, so it doesn't crowd the trajectories)
        ax.scatter(0, 0, c="green", s=80, marker="o", zorder=3,
                   alpha=0.7, label="Start")
        ax.scatter(goal_np[0], goal_np[1], c="red", s=120, marker="*",
                   zorder=3, alpha=0.7, label="Goal")
        if obs_pos is not None:
            ax.scatter(obs_pos[:, 0], obs_pos[:, 1], c="black", s=50,
                       marker="s", alpha=0.6, zorder=3, label="Obstacle")
            if obs_vel is not None:
                ax.quiver(obs_pos[:, 0], obs_pos[:, 1],
                          obs_vel[:, 0], obs_vel[:, 1],
                          angles="xy", scale_units="xy", scale=1.0,
                          width=0.003, color="black", alpha=0.5, zorder=3)
 
        # The three trajectories for this axis
        for j, v in enumerate(sweep_values):
            row = a * n_per_axis + j
            traj = trajs_denorm[row]                       # [T, D]
            ax.plot(traj[:, 0], traj[:, 1], "-",
                    color=sweep_colors[v], linewidth=2.0, alpha=0.95,
                    zorder=4, label=sweep_labels[v])
            # Endpoint marker
            ax.scatter(traj[-1, 0], traj[-1, 1], color=sweep_colors[v],
                       s=40, marker="o", zorder=5, edgecolor="white",
                       linewidth=0.8)
 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(axis_titles[a], fontsize=10)
        ax.legend(loc="best", fontsize=7, ncol=2, framealpha=0.85)
 
    fig.suptitle(f"Style sweep @ step {step}  (other axes held at 0)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    wandb.log({"style_sweep/2x2_grid": wandb.Image(fig)}, step=step)
    plt.close(fig)

# =============================================================================
# Sampling at checkpoint time.
# Runs compositional CFG with one 6-pass batched forward call per denoising step.
# At checkpoint time we sample at NEUTRAL style (zeros) with all-axis CFG ON
# to verify the model is healthy in the most common operating regime.
# (CFG sweeps for the headline experiments are run by a separate eval script.)
# =============================================================================
@torch.no_grad()
def _sample_and_log(ema_model, ema_scene_embedder, ema_style_embedder,
                    val_loader, noise_scheduler, norm_stats, args, device, step):
 
    ema_model.eval()
    ema_scene_embedder.eval()
    ema_style_embedder.eval()
 
    idx = np.random.randint(len(val_loader.dataset))
    val_sample = val_loader.dataset[idx]
 
    traj      = val_sample["trajectory"].unsqueeze(0).to(device)
    start     = val_sample["start_state"].unsqueeze(0).to(device)
    goal      = val_sample["goal"].unsqueeze(0).to(device)
    obstacles = val_sample["obstacles"].unsqueeze(0).to(device)
    obs_mask  = val_sample["obs_mask"].unsqueeze(0).to(device)
    occ_map   = val_sample["occ_map"].unsqueeze(0).to(device)
    has_map   = val_sample["has_map"].unsqueeze(0).to(device)
 
    B, T, D = traj.shape
    K = args.K_samples
 
    # Neutral style for the checkpoint sample.
    style_values = torch.zeros(1, N_STYLE_AXES, device=device)
 
    # ---- Build the 6 conditioning variants and stack into one batched call ----
    # Variant 0: full null            (scene_drop = 1, all style dropped)
    # Variant 1: scene-only           (scene_drop = 0, all style dropped)
    # Variant 2..5: style-axis-i only (scene_drop = 1, only axis i kept)
    # The resulting batch dimension is V = 6.
    n_variants = 1 + 1 + N_STYLE_AXES   # = 6 for 4 axes
 
    # Variant layout:
    #   V0:           scene_drop=1, all style dropped  -> eps_uncond
    #   V1:           scene_drop=0, all style dropped  -> eps_scene
    #   V2..V_{n+1}:  scene_drop=0, only axis i kept   -> eps_scene_plus_axis_i
    scene_drop_v = torch.zeros(n_variants, device=device)
    scene_drop_v[0] = 1.0   # V0: full null
 
    style_drop_v = torch.ones(n_variants, N_STYLE_AXES, device=device)
    # V0 and V1 already have all style dropped (ones init)
    for i in range(N_STYLE_AXES):
        style_drop_v[2 + i, i] = 0.0
 
    # Replicate inputs across the variant dimension.
    start_v     = start.expand(n_variants, -1)
    goal_v      = goal.expand(n_variants, -1)
    obstacles_v = obstacles.expand(n_variants, -1, -1)
    obs_mask_v  = obs_mask.expand(n_variants, -1)
    occ_map_v   = occ_map.expand(n_variants, -1, -1, -1)
    has_map_v   = has_map.expand(n_variants)
    style_v     = style_values.expand(n_variants, -1)
 
    style_tok_v = ema_style_embedder(style_v, style_drop_v)
    tokens_v, _ = ema_scene_embedder(
        start_v, goal_v, obstacles_v, obs_mask_v,
        occ_map=occ_map_v, has_map=has_map_v,
        style_tokens=style_tok_v,
        scene_drop=scene_drop_v,
    )
    attn_mask_v = build_attn_mask(
        obs_mask_v,
        n_map_tokens=ema_scene_embedder.n_map_tokens,
        n_style_tokens=N_STYLE_AXES,
    )
    # Force obstacle positions valid where scene_drop=1 so attention sees the
    # null tokens we inserted.
    sdv = scene_drop_v.bool().unsqueeze(1)
    attn_mask_v = attn_mask_v.clone()
    attn_mask_v[:, 2:2 + ema_scene_embedder.k_max] = torch.where(
        sdv.expand(-1, ema_scene_embedder.k_max),
        torch.ones_like(attn_mask_v[:, 2:2 + ema_scene_embedder.k_max]),
        attn_mask_v[:, 2:2 + ema_scene_embedder.k_max],
    )
 
    # Replicate variants across K parallel samples: final batch = V * K.
    tokens_b = tokens_v.unsqueeze(1).repeat(1, K, 1, 1).reshape(n_variants * K, -1, tokens_v.shape[-1])
    attn_mask_b = attn_mask_v.unsqueeze(1).repeat(1, K, 1).reshape(n_variants * K, -1)
 
    x_in = torch.randn(K, D, T, device=device)              # K parallel noise inits
 
    # CFG weights (read from args; one per axis plus one for scene)
    w_scene = args.cfg_w_scene
    w_style = torch.tensor(args.cfg_w_style, device=device)     # [N_STYLE_AXES]
 
    num_snapshots = 5
    snap_indices = np.linspace(0, noise_scheduler.config.num_train_timesteps - 1,
                               num_snapshots + 1, dtype=int)[1:]
    traj_snapshots = []
 
    for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
        t_tensor_v = torch.full((n_variants * K,), t, device=device, dtype=torch.long)
        x_in_b = x_in.unsqueeze(0).expand(n_variants, -1, -1, -1).contiguous().reshape(n_variants * K, D, T)
 
        noise_b = ema_model(
            sample=x_in_b, timestep=t_tensor_v,
            encoder_hidden_states=tokens_b,
            encoder_attention_mask=attn_mask_b.bool(),
        ).sample                                            # [V*K, D, T]
        noise_b = noise_b.view(n_variants, K, D, T)
 
        eps_uncond = noise_b[0]                              # [K, D, T]
        eps_scene  = noise_b[1]
        eps_axes   = noise_b[2:]                             # [N_STYLE_AXES, K, D, T]
 
        noise_pred = eps_uncond + w_scene * (eps_scene - eps_uncond)
        for i in range(N_STYLE_AXES):
            noise_pred = noise_pred + w_style[i] * (eps_axes[i] - eps_scene)
 
        step_out = noise_scheduler.step(noise_pred, t, x_in)
        x_in = step_out.prev_sample
 
        if t in snap_indices:
            clean_est = step_out.pred_original_sample
            traj_snapshots.append(clean_est.permute(0, 2, 1).detach().cpu())
 
    sample = x_in.permute(0, 2, 1)                          # [K, T, D]
 
    # ---------- Logging (unchanged shape; some helper denorm) ----------
    b = 0
    obs = obstacles[b]
    mask = obs_mask[b].bool()
    obs_valid = obs[mask]
    obs_pos = obs_valid[:, 0:2].cpu().numpy() * norm_stats["obs_std"][0:2] + norm_stats["obs_mean"][0:2]
    obs_vel = obs_valid[:, 2:4].cpu().numpy() * norm_stats["obs_std"][2:4] + norm_stats["obs_mean"][2:4]
    if obs_pos.shape[0] == 0:
        obs_pos = obs_vel = None
 
    start_np = start[b].cpu().numpy() * norm_stats["start_std"] + norm_stats["start_mean"]
    goal_np  = goal[b].cpu().numpy() * norm_stats["goal_std"]  + norm_stats["goal_mean"]
 
    if has_map[0].item() > 0.5:
        occ_map_np = occ_map[0, 0].cpu().numpy()
        map_extent_m = args.map_extent
    else:
        occ_map_np = None
        map_extent_m = None
 
    gt_traj_denorm = traj[0].cpu().numpy() * norm_stats["traj_std"][None, :] + norm_stats["traj_mean"][None, :]
    pred_denorm    = sample[0].cpu().numpy() * norm_stats["traj_std"][None, :] + norm_stats["traj_mean"][None, :]
 
    # Style label used for sampling at checkpoint time (always neutral here).
    sampled_style_np = style_values[0].cpu().numpy()
 
    # Style label of the ground-truth sample we drew (for comparison).
    if "style_values" in val_sample:
        gt_style_np = val_sample["style_values"].cpu().numpy()
    else:
        gt_style_np = None
 
    log_trajectory_scene(
        gt=gt_traj_denorm, pred=pred_denorm,
        start=start_np, goal=goal_np, obs_pos=obs_pos, obs_vel=obs_vel,
        occ_map=occ_map_np, map_extent=map_extent_m,
        step=step, prefix="samples",
        style_values=sampled_style_np,                # <-- sampled style on the rendered traj
    )
    log_trajectory_scene_K(
        gt=gt_traj_denorm,
        pred=sample * torch.from_numpy(norm_stats["traj_std"]).to(sample.device)
             + torch.from_numpy(norm_stats["traj_mean"]).to(sample.device),
        start=start_np, goal=goal_np, obs_pos=obs_pos, obs_vel=obs_vel,
        occ_map=occ_map_np, map_extent=map_extent_m,
        step=step, prefix="samples_K",
        style_values=sampled_style_np,
    )
 
    # Also log a tiny "GT scene's style label" plot so you can verify the
    # validation sample we're comparing against is itself labelled, and at
    # what style. This is independent of the model output.
    if gt_style_np is not None:
        import matplotlib.pyplot as plt
        fig_gt, ax_gt = plt.subplots(figsize=(3.2, 2.4))
        _draw_style_panel(ax_gt, gt_style_np)
        ax_gt.set_title("GT sample's style label", fontsize=9, pad=6)
        wandb.log({"scene/gt_style_label": wandb.Image(fig_gt)}, step=step)
        plt.close(fig_gt)
    
    # Per-axis style sweep: visualise s_i ∈ {-1, 0, +1} for each of the four
    # axes on the same scene, so we can watch the axes diverge as training
    # progresses. Other axes are held at 0 during each sweep.
    _log_style_sweep_grid(
        ema_model, ema_scene_embedder, ema_style_embedder,
        val_sample, noise_scheduler, norm_stats, args, device, step,
    )
    
    # Token labels for the cross-attention heatmap (so style tokens are
    # visually identifiable).
    n_obs = ema_scene_embedder.k_max
    n_map = ema_scene_embedder.n_map_tokens
    token_labels = (
        ["start", "goal"]
        + [f"obs{i}" for i in range(n_obs)]
        + [f"map{i}" for i in range(n_map)]
        + [f"style/{name}" for name in STYLE_AXES]
    )
    section_boundaries = [
        (0,                   "scene"),
        (2,                   "obs"),
        (2 + n_obs,           "map" if n_map > 0 else None),
        (2 + n_obs + n_map,   "style"),
    ]
    # Filter out any sections with None label (e.g. no map).
    section_boundaries = [(b, lbl) for b, lbl in section_boundaries if lbl is not None]
    
    # The captured attention has batch layout [variant_0_sample_0..K-1,
    # variant_1_sample_0..K-1, ...]. Variant 1 is the SCENE-ONLY variant
    # (scene_drop=0, all style nulled), which is the clearest variant to
    # show attention patterns for in the paper — it isolates how the
    # model attends to scene tokens without style interference.
    # So we visualize batch_idx = K = variant_1, sample 0.
    scene_only_batch_idx = K
    log_attention_heatmaps(
        ema_model, step,
        token_labels=token_labels,
        section_boundaries=section_boundaries,
        batch_idx=scene_only_batch_idx,
    )
    log_scene_token_importance(
        ema_model, step, token_labels=token_labels,
        batch_idx=scene_only_batch_idx,
    )
    log_traj_token_attention(ema_model, step, batch_idx=scene_only_batch_idx)
 
    # NEW: 2D occupancy-map attention overlay. Uses the same occ_map / extent
    # we already prepared earlier in _sample_and_log.
    if ema_scene_embedder.n_map_tokens > 0:
        log_map_attention_2d(
            ema_model, step,
            scene_embedder=ema_scene_embedder,
            occ_map=occ_map_np,        # set earlier in this function
            map_extent=map_extent_m,   # set earlier in this function
            per_timestep=False,        # toggle to True for the per-query strip
            batch_idx=scene_only_batch_idx,
        )
 


def compute_normalization_stats(dataset):
    """
    Compute mean and std for start, goal, trajectory, and obstacles.
    """
    start_list = []
    goal_list = []
    traj_list = []
    obs_list = []

    for i in range(len(dataset)):
        sample = dataset[i]
        start_list.append(sample["start_state"].numpy())
        goal_list.append(sample["goal"].numpy())
        traj_list.append(sample["trajectory"].numpy())

        obs = sample["obstacles"].numpy()
        mask = sample["obs_mask"].numpy().astype(bool)

        if mask.any():
            obs_list.append(obs[mask])   # ONLY real obstacles

    start_array = np.stack(start_list, axis=0)
    goal_array = np.stack(goal_list, axis=0)
    traj_array = np.concatenate(traj_list, axis=0)
    
    if len(obs_list) > 0:
        obs_array = np.concatenate(obs_list, axis=0)
        obs_mean = obs_array.mean(axis=0)
        obs_std  = obs_array.std(axis=0) + 1e-8
    else:
        obs_mean = np.zeros(4)
        obs_std  = np.ones(4)

    stats = {
        "start_mean": start_array.mean(axis=0),
        "start_std":  start_array.std(axis=0) + 1e-8,
        "goal_mean":  goal_array.mean(axis=0),
        "goal_std":   goal_array.std(axis=0) + 1e-8,
        "traj_mean":  traj_array.mean(axis=0),
        "traj_std":   traj_array.std(axis=0) + 1e-8,
        "obs_mean":   obs_mean,
        "obs_std":    obs_std,
    }
    return stats


def _parse_csv_floats(s, expected_len):
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated floats, got {len(vals)}: {s!r}")
    return vals


# ------------------------
# Main: argparsing and setup
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="diffusion_socialCFG")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=32, help="trajectory length per sample")
    parser.add_argument("--k_max", type=int, default=10, help="max number of dynamic obstacles")
    # ---- CFG dropout policy ----
    parser.add_argument("--scene_cfg_dropout", type=float, default=0.1,
                        help="Probability of dropping ALL scene conditioning per sample.")
    parser.add_argument("--style_cfg_dropout", type=float, default=0.1,
                        help="INDEPENDENT per-axis dropout probability for each style axis.")
    parser.add_argument("--joint_null_rate", type=float, default=0.05,
                        help="Probability of forcing FULL null (scene + every style axis) per sample.")

    # ---- Inference-time CFG weights ----
    parser.add_argument("--cfg_w_scene", type=float, default=1.0,
                        help="CFG weight on the scene-only path at inference / checkpoint sampling.")
    parser.add_argument("--cfg_w_style", type=str, default="1.0,1.0,1.0,1.0",
                        help="Comma-separated CFG weights, one per style axis (prox, pass, yield, group).")

    parser.add_argument("--K_samples", type=int, default=5,
                        help="Number of parallel samples drawn at checkpoint logging.")
    parser.add_argument("--save_every", type=int, default=1000)
    # parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--token_dim", type=int, default=128, help="Dimension of each conditioning token CrossAttn latent space")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA decay rate")
    parser.add_argument("--ema_warmup_steps", type=int, default=1000, help="Warmup steps for EMA (no EMA updates during warmup)")
    parser.add_argument("--use_map", action="store_true",
                    help="Enable static occupancy map conditioning")
    parser.add_argument("--map_size", type=int, default=50,
                        help="Side length in cells of ego-centered occupancy map")
    parser.add_argument("--map_extent", type=float, default=10,
                    help="Physical side length of ego-centered map in meters")
                        
    args = parser.parse_args()

    # Parse CFG style weights
    args.cfg_w_style = _parse_csv_floats(args.cfg_w_style, N_STYLE_AXES)

    wandb.init(
        project="Social Nav Diffusion",
        name=args.exp_name,
        config=vars(args),   # logs all hyperparameters
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "Style axes:", STYLE_AXES)

    all_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.npz"))
                   + glob.glob(os.path.join(args.dataset_dir, "*.npy")))

    # ---- train/val split (80/20, with fixed shuffle for reproducibility) ----
    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_files)

    n_total = len(all_files)
    n_val = max(1, int(0.2 * n_total))

    train_files = all_files[:n_total - n_val]
    val_files   = all_files[n_total - n_val:]

    # temp dataset to compute normalization
    train_ds_tmp = RobotTrajectoryDataset(
        args.dataset_dir, args.horizon, args.k_max, files=train_files, map_size=args.map_size
    )

    norm_stats = compute_normalization_stats(train_ds_tmp)
    print("Normalization stats:", norm_stats)
    
    ckpt_dir = f"checkpoints/{args.exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    np.save(os.path.join(ckpt_dir, f"norm_stats_{args.exp_name}.npy"), norm_stats)

    train_ds = RobotTrajectoryDataset(
        args.dataset_dir, args.horizon, args.k_max,
        norm_stats=norm_stats, files=train_files, map_size=args.map_size
    )
    val_ds = RobotTrajectoryDataset(
        args.dataset_dir, args.horizon, args.k_max,
        norm_stats=norm_stats, files=val_files, map_size=args.map_size
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True,
    )

    # infer dims from dataset first file
    sample = train_ds_tmp[0]
    traj_dim = sample["trajectory"].shape[1]
    start_dim = sample["start_state"].shape[0]
    goal_dim = sample["goal"].shape[0]

    noise_scheduler = DDPMScheduler( #Can use DDIM later for inference
        num_train_timesteps=args.num_diffusion_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )

    scene_embedder = SceneTokenEmbedder(
        start_dim=start_dim, goal_dim=goal_dim, obs_dim=4,
        token_dim=args.token_dim, k_max=args.k_max,
        map_size=args.map_size, use_map=args.use_map,
    ).to(device)

    style_embedder = StyleTokenEmbedder(
        axis_names=STYLE_AXES, token_dim=args.token_dim,
    ).to(device)


    model = UNet1DConditionModel(
        sample_size=args.horizon,          # T = 32
        in_channels=traj_dim,   # trajectory + conditioning channels + time embedding
        out_channels=traj_dim,             # predict noise on trajectory only
        layers_per_block=3,
        block_out_channels=(64, 128, 256),
        cross_attention_dim=args.token_dim,  # conditioning vector dimension
        down_block_types=(
            "CrossAttnDownBlock1D",  # cross-attention uses encoder_hidden_states
            "CrossAttnDownBlock1D",
            "CrossAttnDownBlock1D",
        ),
        up_block_types=(
            "CrossAttnUpBlock1D",
            "CrossAttnUpBlock1D",
            "CrossAttnUpBlock1D",              # optional: just convolution up
        ),
        mid_block_type="UNetMidBlock1DCrossAttn",  # middle bottleneck block with cross-attention
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + list(scene_embedder.parameters())
        + list(style_embedder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    ema_model = deepcopy(model)
    ema_scene_embedder = deepcopy(scene_embedder)
    ema_style_embedder = deepcopy(style_embedder)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    for p in ema_scene_embedder.parameters():
        p.requires_grad_(False)
    for p in ema_style_embedder.parameters():
        p.requires_grad_(False)

    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_steps,
        lr_min=1e-6,
    )

    attach_cross_attn_capture_processors(model)
    attach_cross_attn_capture_processors(ema_model)
    wandb.watch([model, scene_embedder, style_embedder], log="all", log_freq=500)

    # ---- Dummy forward to verify shapes and report parameter counts ----
    B = 1
    dummy_x = torch.randn(B, traj_dim, args.horizon, device=device)
    dummy_t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device)
    dummy_start = torch.randn(B, start_dim, device=device)
    dummy_goal  = torch.randn(B, goal_dim, device=device)
    dummy_obs   = torch.randn(B, args.k_max, 4, device=device)
    dummy_mask  = torch.ones(B, args.k_max, device=device)
    dummy_occ   = torch.zeros(B, 1, args.map_size, args.map_size, device=device)
    dummy_hasmap = torch.zeros(B, device=device)
    dummy_style  = torch.zeros(B, N_STYLE_AXES, device=device)
    dummy_style_drop = torch.zeros(B, N_STYLE_AXES, device=device)
    dummy_scene_drop = torch.zeros(B, device=device)

    style_tok = style_embedder(dummy_style, dummy_style_drop)
    tokens, _ = scene_embedder(
        dummy_start, dummy_goal, dummy_obs, dummy_mask,
        occ_map=dummy_occ if args.use_map else None,
        has_map=dummy_hasmap if args.use_map else None,
        style_tokens=style_tok,
        scene_drop=dummy_scene_drop,
    )
    attn_mask = build_attn_mask(dummy_mask, n_map_tokens=scene_embedder.n_map_tokens,
                                n_style_tokens=N_STYLE_AXES)
    _ = model(sample=dummy_x, timestep=dummy_t,
              encoder_hidden_states=tokens,
              encoder_attention_mask=attn_mask.bool())

    n_unet  = sum(p.numel() for p in model.parameters()) / 1e6
    n_scene = sum(p.numel() for p in scene_embedder.parameters()) / 1e6
    n_style = sum(p.numel() for p in style_embedder.parameters()) / 1e6
    print(f"UNet {n_unet:.2f}M | SceneEmb {n_scene:.2f}M | StyleEmb {n_style:.2f}M  "
          f"| total {(n_unet + n_scene + n_style):.2f}M parameters")

    train_loop(
        model, scene_embedder, style_embedder,
        ema_model, ema_scene_embedder, ema_style_embedder,
        train_loader, val_loader, norm_stats,
        optimizer, lr_scheduler, noise_scheduler, args, device,
    )

    
if __name__ == "__main__":
    main()
