#
# diffusionCondNetCFG.py  (with feasibility projection layer)
#
# Modifications vs the original:
#   1. __init__ now accepts (and configure() reads) feasibility-projection
#      hyperparameters and builds the acados projection solver once.
#   2. predict() runs the full diffusion + scoring pipeline as before,
#      then PROJECTS the selected best trajectory onto the feasible,
#      collision-free set before extracting the action.
#   3. A safety fallback (controlled stop) triggers if the projection
#      fails or returns a trajectory with large slack violation /
#      excessive deviation from the reference.
#   4. _v0_from_last_step / _w0_from_last_step are now updated from the
#      PROJECTED trajectory (not the raw diffusion output), so warm-state
#      conditioning stays consistent with what the robot actually executes.
#
# Sections marked  ── PROJECTION ──  are the new additions.
#

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import torch
import numpy as np
import math
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet1DModel
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F

import sys
import os
import contextlib
sys.path.append("/home/cschaibl/projects/aip-sl2smith/cschaibl/diffusion_planning")

from diffusers_unet_1d_condition import UNet1DConditionModel

# ── PROJECTION ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from projection_solver import (
    build_projection_solver,
    pack_obstacle_params,
    solve_projection,
    kinematic_warm_start_from_xy,
)
# ───────────────────────────────────────────────────────────────────────

# ── SOCIAL STYLE ───────────────────────────────────────────────────────
# Canonical axis ordering must match the training script exactly.
STYLE_AXES    = ["prox", "pass", "yield", "group"]
N_STYLE_AXES  = len(STYLE_AXES)
# ───────────────────────────────────────────────────────────────────────


# =====================================================================
# (Token embedder, helper losses — unchanged)
# =====================================================================

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
# StyleTokenEmbedder — one MLP per social-style axis.
# Matches the training script exactly so checkpoint weights load correctly.
# =============================================================================
class StyleTokenEmbedder(nn.Module):
    def __init__(self, axis_names, token_dim):
        super().__init__()
        self.axis_names = list(axis_names)
        self.n_axes = len(self.axis_names)
        self.token_dim = token_dim
        hidden_dim = token_dim
        self.axis_mlps = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, token_dim),
            )
            for name in self.axis_names
        })
        self.type_embed  = nn.Parameter(torch.randn(self.n_axes, token_dim) * 0.02)
        self.null_tokens = nn.Parameter(torch.randn(self.n_axes, token_dim) * 0.02)
        self.final_norm  = nn.LayerNorm(token_dim)

    def forward(self, style_values, style_drop_mask):
        """
        style_values:    [B, n_axes]  floats in ~[-1, 1]
        style_drop_mask: [B, n_axes]  1.0 = replace with null token, 0.0 = use real
        returns:         [B, n_axes, token_dim]
        """
        B = style_values.shape[0]
        out = []
        for i, name in enumerate(self.axis_names):
            scalar = style_values[:, i:i + 1]
            tok    = self.axis_mlps[name](scalar) + self.type_embed[i]
            null   = self.null_tokens[i].unsqueeze(0).expand(B, -1)
            drop   = style_drop_mask[:, i:i + 1]
            tok    = drop * null + (1.0 - drop) * tok
            out.append(tok.unsqueeze(1))
        out = torch.cat(out, dim=1)   # [B, n_axes, D]
        return self.final_norm(out)


# =============================================================================
# SceneTokenEmbedder — updated to match the training script:
#   • Per-section null tokens (start, goal, obs) instead of a monolithic null_token.
#   • Accepts style_tokens from StyleTokenEmbedder and appends them.
#   • Accepts scene_drop (per-sample {0,1} flag) for CFG unconditional path.
#   • Returns (tokens, valid.float()) so callers can see the validity mask.
# =============================================================================
class SceneTokenEmbedder(nn.Module):
    def __init__(self, start_dim, goal_dim, obs_dim, token_dim, k_max,
                 n_self_attn_layers=2, map_size=50, use_map=True):
        super().__init__()
        hidden_dim = token_dim
        self.use_map = use_map
        self.k_max   = k_max

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

        # Per-section null tokens for CFG scene dropout.
        self.null_start_token = nn.Parameter(torch.randn(1, 1,     token_dim) * 0.02)
        self.null_goal_token  = nn.Parameter(torch.randn(1, 1,     token_dim) * 0.02)
        self.null_obs_tokens  = nn.Parameter(torch.randn(1, k_max, token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=4,
            dim_feedforward=token_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.scene_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_self_attn_layers
        )
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, start, goal, obstacles, obs_mask=None,
                occ_map=None, has_map=None,
                style_tokens=None,   # [B, n_axes, D] from StyleTokenEmbedder
                scene_drop=None,     # [B] float in {0., 1.} — 1.0 = drop all scene
                ):
        """
        Returns (tokens [B, N_tok, D], valid [B, N_tok] float).
        Token layout: [ start | goal | obstacles | (map) | (style) ]
        """
        B      = start.shape[0]
        device = start.device

        if scene_drop is None:
            scene_drop = torch.zeros(B, device=device)
        sd = scene_drop.view(B, 1, 1)

        start_tok = sd * self.null_start_token + (1.0 - sd) * self.start_proj(start).unsqueeze(1)
        start_tok = start_tok + self.type_embed[0]

        goal_tok  = sd * self.null_goal_token  + (1.0 - sd) * self.goal_proj(goal).unsqueeze(1)
        goal_tok  = goal_tok  + self.type_embed[1]

        obs_tok   = sd * self.null_obs_tokens  + (1.0 - sd) * self.obs_proj(obstacles)
        obs_tok   = obs_tok   + self.type_embed[2]

        tokens = torch.cat([start_tok, goal_tok, obs_tok], dim=1)   # [B, 2+K, D]

        # Validity mask — when scene is dropped, null obs tokens are still valid.
        scene_drop_bool = scene_drop.bool().unsqueeze(1)   # [B, 1]
        if obs_mask is not None:
            obs_valid = obs_mask.bool() | scene_drop_bool  # [B, k_max]
        else:
            obs_valid = torch.ones(B, self.k_max, device=device, dtype=torch.bool)
        valid = torch.cat(
            [torch.ones(B, 2, device=device, dtype=torch.bool), obs_valid], dim=1
        )

        if self.use_map:
            assert occ_map is not None and has_map is not None, \
                "use_map=True requires occ_map and has_map"
            effective_has_map = has_map * (1.0 - scene_drop)
            map_tokens = self.map_encoder(occ_map, effective_has_map)
            tokens = torch.cat([tokens, map_tokens], dim=1)
            valid  = torch.cat(
                [valid, torch.ones(B, self.n_map_tokens, device=device, dtype=torch.bool)],
                dim=1,
            )

        if style_tokens is not None:
            n_style = style_tokens.shape[1]
            tokens = torch.cat([tokens, style_tokens], dim=1)
            valid  = torch.cat(
                [valid, torch.ones(B, n_style, device=device, dtype=torch.bool)],
                dim=1,
            )

        key_padding_mask = ~valid
        tokens = self.scene_encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.final_norm(tokens), valid.float()


def build_attn_mask(obs_mask, n_map_tokens=0, n_style_tokens=0):
    """
    obs_mask:       [B, K]
    n_map_tokens:   int — map tokens after obstacles
    n_style_tokens: int — style tokens after map tokens
    """
    B, K = obs_mask.shape
    device = obs_mask.device
    base = torch.ones(B, 2, device=device)   # start + goal always valid
    pieces = [base, obs_mask]
    if n_map_tokens > 0:
        pieces.append(torch.ones(B, n_map_tokens, device=device))
    if n_style_tokens > 0:
        pieces.append(torch.ones(B, n_style_tokens, device=device))
    return torch.cat(pieces, dim=1)


def sinusoidal_pos_emb(T, dim, device):
    pos = torch.arange(T, device=device)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim)
    )
    emb = torch.zeros(T, dim, device=device)
    emb[:, 0::2] = torch.sin(pos[:, None] * div)
    emb[:, 1::2] = torch.cos(pos[:, None] * div)
    return emb


def move_scheduler_to_device(scheduler, device):
    for name, value in scheduler.__dict__.items():
        if torch.is_tensor(value):
            scheduler.__dict__[name] = value.to(device)
    return scheduler


def control_effort_penalty(x_trajs_denorm):
    pos = x_trajs_denorm[:, :, 0:2, :]
    vel = pos[:, :, :, 1:] - pos[:, :, :, :-1]
    forward_effort = vel[:, :, 0, :].pow(2)
    lateral_effort = vel[:, :, 1, :].pow(2)
    effort = forward_effort + 3.0 * lateral_effort
    return effort.mean(dim=2)


def collision_cost(x_trajs, obstacles, obs_mask, dt=0.05, safety_radius=0.5):
    B, K, D, T = x_trajs.shape
    traj_xy = x_trajs[:, :, 0:2, :].permute(0, 1, 3, 2)
    obs_pos = obstacles[:, :, 0:2].unsqueeze(2)
    obs_vel = obstacles[:, :, 2:4].unsqueeze(2)
    t_vec   = torch.arange(T, device=x_trajs.device).float() * dt
    t_vec   = t_vec.view(1, 1, T, 1)
    obs_future = obs_pos + obs_vel * t_vec
    traj_xy    = traj_xy.unsqueeze(2)
    obs_future = obs_future.unsqueeze(1)
    dist       = torch.norm(traj_xy - obs_future, dim=-1)
    penetration = F.relu(safety_radius - dist)
    penalty     = penetration ** 2
    obs_mask_exp = obs_mask.unsqueeze(1).unsqueeze(-1)
    cost = (penalty * obs_mask_exp).sum(dim=[2, 3])
    return cost


def smoothness_penalty(x_trajs_denorm):
    pos     = x_trajs_denorm[:, :, 0:2, :]
    vel     = pos[:, :, :, 1:] - pos[:, :, :, :-1]
    acc     = vel[:, :, :, 1:] - vel[:, :, :, :-1]
    acc_mag = acc.pow(2).sum(dim=2)
    return acc_mag.pow(1.5).mean(dim=2)


def goal_progress_reward(x_trajs_denorm, goal, goal_mean, goal_std):
    device = x_trajs_denorm.device
    goal_mean_t = torch.as_tensor(goal_mean[0:2], dtype=torch.float32, device=device)
    goal_std_t  = torch.as_tensor(goal_std[0:2],  dtype=torch.float32, device=device)
    goal_xy = (goal[:, 0:2] * goal_std_t + goal_mean_t)
    goal_xy = goal_xy.unsqueeze(1).unsqueeze(3)
    traj_xy       = x_trajs_denorm[:, :, 0:2, :]
    dist_to_goal  = torch.norm(traj_xy - goal_xy, dim=2)
    progress      = dist_to_goal[:, :, 0] - dist_to_goal[:, :, -1]
    step_progress = (dist_to_goal[:, :, :-1] - dist_to_goal[:, :, 1:]).mean(dim=2)
    return progress + 0.5 * step_progress


def score_trajectories(x_trajs, obstacles, obs_mask, goal, norm,
                       collision_coef, smooth_pen_coef, goal_reward_coef, control_effort_coef,
                       safety_radius=0.5, dt=0.05,
                       static_obs_xy=None):
    """
    Score K diffusion trajectory samples.

    Parameters
    ----------
    x_trajs       : (B, K, traj_dim, T) normalised samples from the diffusion
    obstacles     : (B, k_max, 4)  normalised dynamic human states (px,py,vx,vy)
    obs_mask      : (B, k_max)     1 = valid human slot, 0 = padding
    static_obs_xy : (B, M, 2) physical ego-frame positions of static map cells,
                    or None.  Zero velocity is assumed — positions stay fixed
                    across the horizon.  Penalised with the same collision_coef.

    Returns
    -------
    rewards  : (B, K)
    best_idx : (B,)
    """
    traj_std  = torch.as_tensor(norm["traj_std"], dtype=torch.float32, device=x_trajs.device)
    traj_mean = torch.as_tensor(norm["traj_mean"], dtype=torch.float32, device=x_trajs.device)
    obs_std   = torch.as_tensor(norm["obs_std"],  dtype=torch.float32, device=x_trajs.device)
    obs_mean  = torch.as_tensor(norm["obs_mean"], dtype=torch.float32, device=x_trajs.device)
    obs_denorm = obstacles * obs_std + obs_mean

    x_denorm = x_trajs * traj_std.view(1, 1, -1, 1) + traj_mean.view(1, 1, -1, 1)

    # ── Dynamic human collision cost ────────────────────────────────────
    C_dyn = collision_cost(x_denorm, obs_denorm, obs_mask, dt=dt, safety_radius=safety_radius)

    # ── Static map cell collision cost ──────────────────────────────────
    # static_obs_xy is in physical ego-frame metres; zero velocity means
    # obs_future = obs_pos for every horizon node — collision_cost handles
    # this naturally when obs_vel = 0.
    C_static = torch.zeros_like(C_dyn)
    if static_obs_xy is not None and static_obs_xy.shape[1] > 0:
        B_s, M_s, _ = static_obs_xy.shape
        zeros_vel    = torch.zeros(B_s, M_s, 2,
                                   device=static_obs_xy.device,
                                   dtype=static_obs_xy.dtype)
        static_obs_4d = torch.cat([static_obs_xy, zeros_vel], dim=-1)   # (B, M, 4)
        static_mask   = torch.ones(B_s, M_s, device=static_obs_xy.device)
        C_static = collision_cost(x_denorm, static_obs_4d, static_mask,
                                  dt=dt, safety_radius=safety_radius)

    smooth_pen = smoothness_penalty(x_denorm)
    goal_rew   = goal_progress_reward(x_denorm, goal, norm["goal_mean"], norm["goal_std"])
    effort_pen = control_effort_penalty(x_denorm)

    rewards = (
        - collision_coef      * (C_dyn + C_static)   # same weight for both types
        - smooth_pen_coef     * smooth_pen
        - control_effort_coef * effort_pen
        + goal_reward_coef    * goal_rew
    )

    # A trajectory is "safe" only if it avoids BOTH dynamic agents AND walls.
    EPS = 1e-6
    is_safe  = (C_dyn < EPS) & (C_static < EPS)
    has_safe = is_safe.any(dim=1)
    rewards_masked = rewards.masked_fill(~is_safe, float('-inf'))
    best_idx_safe = rewards_masked.argmax(dim=1)
    best_idx_all  = rewards.argmax(dim=1)
    best_idx = torch.where(has_safe, best_idx_safe, best_idx_all)

    return rewards, best_idx


# =====================================================================
# Policy
# =====================================================================

class DiffusionConditionalUNet1DCFG(Policy):
    def __init__(self):
        super().__init__()
        self.model = None
        self.token_embedder = None
        self.diffusion = None
        self.norm = None
        self.device = None
        self.horizon = None
        self.k_max = None
        self.start_dim = None
        self.goal_dim = None
        self.kinematics = None
        self.enforce_lims = False
        self.max_vel = float('inf')
        self.max_wrot = float('inf')
        self.max_accel = float('inf')
        self.max_w_accel = float('inf')
        self.multiagent_training = None
        self.predicted_traj = None
        self.name = "DiffusionConditionalUNet1DCFG"
        self.prev_v = 0.0
        self.prev_omega = 0.0

        # ── PROJECTION ──────────────────────────────────────────────────
        self.use_projection      = True   # set False to disable layer entirely
        self.proj_solver         = None
        self.proj_model          = None
        self.proj_constraint     = None
        self.proj_dt             = 0.05   # diffusion sample period
        self.proj_N              = None   # set in configure() = horizon - 1
        self.proj_Tf             = None   # set in configure() = (horizon-1) * dt
        self.proj_max_dev_thresh = 1.5    # [m] fallback if deviation exceeds this
        self.proj_max_slack_thresh = 0.5  # fallback if obstacle slack exceeds this
        self.proj_solver_mode    = "RTI"  # "RTI" or "SQP"
        # Obstacle budget split:
        #   k_max    → dynamic agents only.  Set in configure() from config.
        #   k_static → dedicated static map cell slots.  NEVER shared with
        #              dynamic agents, so dense walls always get MPC coverage
        #              regardless of crowd size.
        #   proj_n_obs = k_max + k_static  → total n_obs_max passed to acados.
        self.k_static   = 20    # overridden in configure()
        self.proj_n_obs = None  # set in configure() = k_max + k_static
        # Clearance philosophy:
        #   The projection uses the SAME safety_radius the diffusion's
        #   collision_cost / scoring uses. This is a SINGLE GLOBAL
        #   "robot-center to obstacle-center" distance — it bundles the
        #   robot radius, obstacle radius, and any margin into one number.
        #   We do NOT add r_robot + r_human separately, since safety_radius
        #   is already meant to encode that combined clearance.
        #   self.safety_radius is populated in configure() from
        #   [diffusion_conditional_unet1dcfg] safety_radius (× 1.3).
        # Warm-start policy: ALWAYS from the current diffusion sample.
        # We do NOT cache the previous solve, because the diffusion may
        # produce a different homotopy class each tick (different side
        # of an obstacle, different timing through gates), making the
        # previous solve a misleading anchor for SQP linearization.
        # The diffusion path itself is by construction in the right
        # homotopy class — that's the strongest warm start we have.
        # ────────────────────────────────────────────────────────────────

        self.use_map           = False
        self.map_size          = 50
        self.map_extent        = 10
        self._cur_occ_world_xy = None
        self._cur_has_map      = 0.0          # float
        self._cur_map_extent   = float(self.map_extent)  # valid default until set_static_map()

        # ── SOCIAL STYLE ─────────────────────────────────────────────────
        # style_vector: numpy array of shape (N_STYLE_AXES,), values in [-1,1].
        # Set in configure() from config or left as neutral zeros.
        # Kept fixed for the lifetime of this policy instance — no per-step changes.
        self.style_vector  = np.zeros(N_STYLE_AXES, dtype=np.float32)
        self.n_style_axes  = N_STYLE_AXES
        self.style_embedder = None   # StyleTokenEmbedder, built in configure()

        # 3-pass compositional CFG weights.
        #   cfg_w_scene : how hard to push scene conditioning (goal, obstacles, map)
        #   cfg_w_style : how hard to push social-style conditioning independently
        # Set in configure() from policy.config; fall back to legacy cfg_weight if
        # the new keys are absent (preserves backward compatibility).
        self.cfg_w_scene = 1.0
        self.cfg_w_style = 1.0
        # ─────────────────────────────────────────────────────────────────

        # ── AMP ──────────────────────────────────────────────────────────
        # Optional bf16 autocast around the UNet forward pass in the
        # denoising loop. Off by default; set use_amp = true under
        # [diffusion_conditional_unet1dcfg] to enable. Kept as a flag
        # (rather than always-on) so it can be A/B compared for timing
        # impact without touching the eager/fp32 numerics otherwise.
        self.use_amp = False
        self.amp_dtype = torch.bfloat16
        # ─────────────────────────────────────────────────────────────────

    def configure(self, config):
        import torch
        import numpy as np

        self.start_dim = config.getint('diffusion_conditional_unet1dcfg', 'start_dim')
        self.goal_dim = config.getint('diffusion_conditional_unet1dcfg', 'goal_dim')
        self.token_dim = config.getint('diffusion_conditional_unet1dcfg', 'token_dim')
        self.horizon = config.getint('diffusion_conditional_unet1dcfg', 'horizon')
        self.k_max = config.getint('diffusion_conditional_unet1dcfg', 'k_max')
        self.kinematics = config.get('action_space', 'kinematics')
        self.enforce_lims = config.getboolean('action_space', 'enforce_lims')
        self.max_vel = config.getfloat('action_space', 'max_vel')
        self.max_wrot = config.getfloat('action_space', 'max_wrot')
        self.max_accel = config.getfloat('action_space', 'max_accel')
        self.max_w_accel = config.getfloat('action_space', 'max_w_accel')
        self.multiagent_training = config.getboolean('diffusion_conditional_unet1dcfg', 'multiagent_training')
        self.num_diffusion_timesteps = config.getint('diffusion_conditional_unet1dcfg', 'num_diffusion_timesteps')
        self.num_inference_steps = config.getint('diffusion_conditional_unet1dcfg', 'num_inference_steps')
        self.ddim_inference = config.getboolean('diffusion_conditional_unet1dcfg', 'ddim_inference')
        self.cfg_weight = config.getfloat('diffusion_conditional_unet1dcfg', 'cfg_weight')
        self.guidance_scale = config.getfloat('diffusion_conditional_unet1dcfg', 'guidance_scale')

        # 3-pass compositional CFG weights (optional; fall back to cfg_weight).
        # cfg_w_scene controls scene guidance strength independently of style.
        # cfg_w_style controls how hard the style axis is pushed on top of scene.
        _sec = 'diffusion_conditional_unet1dcfg'
        self.cfg_w_scene = (config.getfloat(_sec, 'cfg_w_scene')
                            if config.has_option(_sec, 'cfg_w_scene')
                            else self.cfg_weight)
        self.cfg_w_style = (config.getfloat(_sec, 'cfg_w_style')
                            if config.has_option(_sec, 'cfg_w_style')
                            else self.cfg_weight)
        self.classifierguide = config.getboolean('diffusion_conditional_unet1dcfg', 'classifierguide')
        self.num_samples = config.getint('diffusion_conditional_unet1dcfg', 'num_samples')
        self.collision_coef = config.getfloat('diffusion_conditional_unet1dcfg', 'collision_coef')
        self.smooth_pen_coef = config.getfloat('diffusion_conditional_unet1dcfg', 'smooth_pen_coef')
        self.goal_reward_coef = config.getfloat('diffusion_conditional_unet1dcfg', 'goal_reward_coef')
        self.safety_radius = config.getfloat('diffusion_conditional_unet1dcfg', 'safety_radius') * 1.3
        self.control_effort_coef = config.getfloat('diffusion_conditional_unet1dcfg', 'control_effort_coef')

        self.use_map = (config.getboolean('diffusion_conditional_unet1dcfg', 'use_map')
                if config.has_option('diffusion_conditional_unet1dcfg', 'use_map')
                else False)
        self.map_size = (config.getint('diffusion_conditional_unet1dcfg', 'map_size')
                        if config.has_option('diffusion_conditional_unet1dcfg', 'map_size')
                        else 50)
        self.map_extent = (config.getfloat('diffusion_conditional_unet1dcfg', 'map_extent')
                        if config.has_option('diffusion_conditional_unet1dcfg', 'map_extent')
                        else 10)

        # ── AMP ──────────────────────────────────────────────────────────
        self.use_amp = (config.getboolean('diffusion_conditional_unet1dcfg', 'use_amp')
                        if config.has_option('diffusion_conditional_unet1dcfg', 'use_amp')
                        else False)
        amp_dtype_str = (config.get('diffusion_conditional_unet1dcfg', 'amp_dtype')
                        if config.has_option('diffusion_conditional_unet1dcfg', 'amp_dtype')
                        else 'bfloat16')
        _amp_dtype_map = {'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
                          'float16': torch.float16, 'fp16': torch.float16}
        assert amp_dtype_str in _amp_dtype_map, (
            f"amp_dtype must be one of {list(_amp_dtype_map)}, got {amp_dtype_str!r}"
        )
        self.amp_dtype = _amp_dtype_map[amp_dtype_str]
        # ─────────────────────────────────────────────────────────────────

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print("Device:", device)

        # Fixed-shape 1D convs run every tick with identical sizes — let
        # cuDNN autotune the fastest algorithm once instead of using the
        # default heuristic every call. Same math, same precision, just
        # picks a faster kernel implementation.
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        norm_file = config.get('diffusion_conditional_unet1dcfg', 'norm_file')
        self.norm = np.load(norm_file, allow_pickle=True).item()

        # Cached GPU tensor copies of self.norm for the hot (per-tick)
        # math path, so predict()/score_trajectories() don't rebuild +
        # re-transfer these from numpy on every call. self.norm itself is
        # left untouched (still numpy) since build_condition_from_state()
        # relies on plain numpy arithmetic.
        self.norm_t = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in self.norm.items()
        }

        self.traj_dim = 2

        self.token_embedder = SceneTokenEmbedder(
            start_dim=self.start_dim,
            goal_dim=self.goal_dim,
            obs_dim=4,
            token_dim=self.token_dim,
            k_max=self.k_max,
            map_size=self.map_size,
            use_map=self.use_map,
        ).to(self.device)

        self.model = UNet1DConditionModel(
            sample_size=self.horizon,
            in_channels=self.traj_dim,
            out_channels=self.traj_dim,
            layers_per_block=3,
            block_out_channels=(64, 128, 256),
            cross_attention_dim=self.token_dim,
            down_block_types=(
                "CrossAttnDownBlock1D",
                "CrossAttnDownBlock1D",
                "CrossAttnDownBlock1D",
            ),
            up_block_types=(
                "CrossAttnUpBlock1D",
                "CrossAttnUpBlock1D",
                "CrossAttnUpBlock1D",
            ),
            mid_block_type="UNetMidBlock1DCrossAttn",
        ).to(self.device)

        if self.ddim_inference:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=False,
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=False,
            )
        self.noise_scheduler = move_scheduler_to_device(self.noise_scheduler, self.device)

        # ── SOCIAL STYLE ─────────────────────────────────────────────────
        # Optional: fixed style vector to use for inference.
        # Format in config: style_vector = 0.0,0.0,0.0,0.0  (prox,pass,yield,group)
        if config.has_option('diffusion_conditional_unet1dcfg', 'style_vector'):
            raw = config.get(
                'diffusion_conditional_unet1dcfg', 'style_vector'
            ).strip()
            if raw:
                vals = [float(v.strip()) for v in raw.split(',') if v.strip()]
            else:
                vals = []
            sv = np.zeros(self.n_style_axes, dtype=np.float32)
            n = min(len(vals), self.n_style_axes)
            if n > 0:
                sv[:n] = np.array(vals[:n], dtype=np.float32)
            self.style_vector = np.clip(sv, -1.0, 1.0)
        else:
            self.style_vector = np.zeros(self.n_style_axes, dtype=np.float32)
        print(f"Social style vector: {dict(zip(STYLE_AXES, self.style_vector))}")

        self.style_embedder = StyleTokenEmbedder(
            axis_names=STYLE_AXES,
            token_dim=self.token_dim,
        ).to(self.device)
        # ─────────────────────────────────────────────────────────────────

        ckpt_path = config.get('diffusion_conditional_unet1dcfg', 'ckpt_path')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        if "ema_model_state_dict" in ckpt:
            print("Loading EMA weights for inference")
            self.model.load_state_dict(ckpt["ema_model_state_dict"])
            # Support both old key name and new key name for scene embedder.
            scene_key = ("ema_scene_embedder_state_dict"
                         if "ema_scene_embedder_state_dict" in ckpt
                         else "ema_token_embedder_state_dict")
            self.token_embedder.load_state_dict(ckpt[scene_key])
            if "ema_style_embedder_state_dict" in ckpt:
                self.style_embedder.load_state_dict(ckpt["ema_style_embedder_state_dict"])
                print("Loaded EMA style embedder weights")
            else:
                print("WARNING: no style embedder weights in checkpoint — "
                      "style_embedder will use random initialisation")
        else:
            print("EMA weights not found — falling back to bare model weights")
            self.model.load_state_dict(ckpt["model_state_dict"])
            scene_key = ("scene_embedder_state_dict"
                         if "scene_embedder_state_dict" in ckpt
                         else "token_embedder_state_dict")
            self.token_embedder.load_state_dict(ckpt[scene_key])
            if "style_embedder_state_dict" in ckpt:
                self.style_embedder.load_state_dict(ckpt["style_embedder_state_dict"])
            else:
                print("WARNING: no style embedder weights in checkpoint — "
                      "style_embedder will use random initialisation")

        self.model.eval()
        self.token_embedder.eval()
        self.style_embedder.eval()

        print(f"AMP autocast for UNet forward: "
              f"{'enabled (' + str(self.amp_dtype) + ')' if self.use_amp else 'disabled'}")

        # Fuse Q/K/V projection weights for attention layers (self-attn:
        # fuse all 3; cross-attn: fuse K+V since both read encoder_hidden_states).
        # Concatenates the already-trained weight matrices into one combined
        # matmul — identical outputs, fewer kernel launches. This model's
        # cross-attn blocks (CrossAttnDownBlock1D/UpBlock1D, the mid block)
        # dominate its structure, so this directly targets the per-step cost.
        # HuggingFace marks this API "experimental"; it self-guards by raising
        # ValueError for architectures it can't safely fuse (e.g. added-KV
        # projections), so a failure here is loud, not silent.
        try:
            self.model.fuse_qkv_projections()
            print("Fused QKV projections enabled for UNet attention layers")
        except Exception as e:
            print(f"fuse_qkv_projections unavailable, leaving attention unfused: {e}")

        # Compile the UNet forward pass — same graph, fused CUDA kernels.
        # predict() calls self.model with identical tensor shapes every
        # step (N_V*K, traj_dim, horizon), so this specializes once and
        # is reused for the rest of the run. Falls back to eager mode if
        # compilation isn't supported on this torch/CUDA build, so a
        # compile failure can never break inference.
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("torch.compile enabled for UNet (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile unavailable, falling back to eager UNet: {e}")

        # ── PROJECTION ──────────────────────────────────────────────────
        # Optional config knobs (with defaults) — add these to your config
        # under [diffusion_conditional_unet1dcfg] if you want non-default values.
        if config.has_option('diffusion_conditional_unet1dcfg', 'use_projection'):
            self.use_projection = config.getboolean(
                'diffusion_conditional_unet1dcfg', 'use_projection')
        if config.has_option('diffusion_conditional_unet1dcfg', 'proj_max_dev_thresh'):
            self.proj_max_dev_thresh = config.getfloat(
                'diffusion_conditional_unet1dcfg', 'proj_max_dev_thresh')
        if config.has_option('diffusion_conditional_unet1dcfg', 'proj_max_slack_thresh'):
            self.proj_max_slack_thresh = config.getfloat(
                'diffusion_conditional_unet1dcfg', 'proj_max_slack_thresh')
        if config.has_option('diffusion_conditional_unet1dcfg', 'proj_solver_mode'):
            self.proj_solver_mode = config.get(
                'diffusion_conditional_unet1dcfg', 'proj_solver_mode')
        
        assert self.proj_solver_mode in ("RTI", "SQP"), (
            f"proj_solver_mode must be 'RTI' or 'SQP', got "
            f"{self.proj_solver_mode!r}"
        )

        # Dedicated static map obstacle budget — independent of k_max so
        # dense walls are never crowded out by dynamic agents.
        if config.has_option('diffusion_conditional_unet1dcfg', 'k_static'):
            self.k_static = config.getint('diffusion_conditional_unet1dcfg', 'k_static')

        # Total slots for the acados solver = dynamic + static.
        self.proj_n_obs = self.k_max + self.k_static

        # Note: discomfort_dist and any analogous training-side safety
        # buffer are RL training/reward concerns and are intentionally NOT
        # used here. The projection enforces clearance purely via
        # self.safety_radius, which is the same number the diffusion's
        # scoring uses, ensuring the projector and the sampler agree on
        # what "safe" means.

        if self.use_projection:
            # Horizon convention (matches the diffusion exactly):
            #   The diffusion outputs T = self.horizon points at times
            #     0, dt, 2*dt, ..., (T-1)*dt
            #   where INDEX 0 IS THE CURRENT ROBOT POSITION (i.e. always
            #   (0,0) in ego frame, modulo denoising noise) and indices
            #   1..T-1 are the future positions.
            #
            #   The OCP therefore uses:
            #     N  = T - 1   shooting intervals  (so T nodes total)
            #     Tf = (T - 1) * proj_dt
            #
            #   Node 0 is the initial-condition node (x0 = current state),
            #   nodes 1..N track diffusion[1..T-1] as references.
            #   We deliberately do NOT track diffusion[0] — the model
            #   outputs only an approximation of (0,0) there, but we
            #   already know the true value exactly via x0.
            self.proj_N  = self.horizon - 1
            self.proj_Tf = self.proj_N * self.proj_dt

            print(f"Building acados projection solver: "
                  f"N={self.proj_N}, Tf={self.proj_Tf:.3f}s, "
                  f"n_obs={self.proj_n_obs} (k_max={self.k_max} dynamic "
                  f"+ k_static={self.k_static} static), "
                  f"mode={self.proj_solver_mode}")
            t0 = datetime.now()
            self.proj_solver, self.proj_model, self.proj_constraint = \
                build_projection_solver(
                    N=self.proj_N,
                    Tf=self.proj_Tf,
                    n_obs_max=self.proj_n_obs,
                    solver_mode=self.proj_solver_mode,
                )
            print(f"  solver built in {datetime.now() - t0}")
        # ────────────────────────────────────────────────────────────────

    def set_static_map(self, occ_world_xy, has_map, map_extent):
        """
        Called by the env at episode start.
        occ_world_xy: (N, 2) numpy array of occupied-cell centers in WORLD frame,
                    or None.
        has_map:      1.0 if scenario has a map, 0.0 otherwise.
        map_extent:   physical side length of the map (meters).
        """
        self._cur_occ_world_xy = occ_world_xy
        self._cur_has_map      = float(has_map)
        # Guard: preserve the non-zero default when the map is absent so
        # reach computations using _cur_map_extent never divide by zero.
        self._cur_map_extent   = float(map_extent) if float(map_extent) > 0 else float(self.map_extent)

    def _get_static_cells_ego(self, state):
        """
        Return ego-frame (px, py) positions of the k_static nearest occupied
        map cells in physical metres.  Identical ego-frame transform as Pool B
        in _gather_obstacles_ego_physical and _rasterize_ego_map.

        Called before trajectory scoring so sample selection is also wall-aware.

        Returns numpy (M, 2) with M <= k_static, or None when no map / no cells.
        """
        if not (self.use_map
                and self._cur_has_map > 0.5
                and self._cur_occ_world_xy is not None
                and self._cur_occ_world_xy.shape[0] > 0
                and self.k_static > 0):
            return None

        px_r  = state.self_state.px
        py_r  = state.self_state.py
        theta = state.self_state.theta
        cos_t = np.cos(theta); sin_t = np.sin(theta)

        # World → ego: subtract robot position, rotate by -theta
        pts_w = self._cur_occ_world_xy
        ddx   = pts_w[:, 0] - px_r
        ddy   = pts_w[:, 1] - py_r
        x_e   =  cos_t * ddx + sin_t * ddy
        y_e   = -sin_t * ddx + cos_t * ddy
        dists = np.sqrt(x_e ** 2 + y_e ** 2)

        # Include all cells within the map half-extent (full coverage)
        reach = self._cur_map_extent / 2.0
        sel   = dists < reach
        if not sel.any():
            return None

        x_e_s, y_e_s, d_s = x_e[sel], y_e[sel], dists[sel]
        order = np.argsort(d_s)[:self.k_static]
        return np.stack([x_e_s[order], y_e_s[order]], axis=1).astype(np.float32)
        
    def _rasterize_ego_map(self, state):
        """
        Build a (map_size, map_size) ego-centered occupancy map from the
        world-frame occupied-cell positions, using the robot's CURRENT pose.

        Returns (occ_map_np, has_map_float). When use_map is off or no map
        is loaded, returns (zeros, 0.0).
        """
        H = W = self.map_size
        if (not self.use_map
                or self._cur_occ_world_xy is None
                or self._cur_has_map < 0.5
                or self._cur_occ_world_xy.shape[0] == 0):
            return np.zeros((H, W), dtype=np.float32), 0.0

        # Robot pose in world
        px = state.self_state.px
        py = state.self_state.py
        theta = state.self_state.theta

        # Transform world points into the ego frame: subtract robot position,
        # then rotate by -theta (same convention used in build_condition_from_state).
        pts_w = self._cur_occ_world_xy           # (N, 2)
        dx = pts_w[:, 0] - px
        dy = pts_w[:, 1] - py
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        x_ego =  cos_t * dx + sin_t * dy
        y_ego = -sin_t * dx + cos_t * dy

        # Map cell convention: row → y, col → x, ego-centered.
        res = self._cur_map_extent / W
        cols = np.round(x_ego / res + W / 2).astype(np.int32)
        rows = np.round(y_ego / res + H / 2).astype(np.int32)

        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows = rows[in_bounds]; cols = cols[in_bounds]

        occ = np.zeros((H, W), dtype=np.float32)
        if rows.size > 0:
            occ[rows, cols] = 1.0
        return occ, 1.0
        
    def get_model(self):
        return None

    def _reset_unicycle_state(self, v0=0.0, omega0=0.0):
        self.prev_v = v0
        self.prev_omega = omega0
        self._v0_from_last_step = v0
        self._w0_from_last_step = omega0
        # No previous-solve cache to clear — we always warm-start from the
        # current diffusion sample (see __init__ comment).

    def build_condition_from_state(self, state):
        v0 = getattr(self, '_v0_from_last_step', 0.0)
        if self.start_dim == 2:
            w0 = getattr(self, '_w0_from_last_step', 0.0)
            start = np.array([v0, w0], dtype=np.float32)
        else:
            start = np.array([v0], dtype=np.float32)

        theta = state.self_state.theta
        R = np.array([
            [ np.cos(theta),  np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)

        goal_world = np.array([state.self_state.gx, state.self_state.gy])
        pos_world  = np.array([state.self_state.px, state.self_state.py])
        goal_ego = R @ (goal_world - pos_world)
        goal = goal_ego.astype(np.float32)

        obs_list = []
        for h in state.human_states:
            pos_ego = R @ (np.array([h.px, h.py]) - pos_world)
            vel_ego = R @ np.array([h.vx, h.vy])
            obs_list.append([pos_ego[0], pos_ego[1], vel_ego[0], vel_ego[1]])
        obstacles = np.array(obs_list, dtype=np.float32)

        obs_padded = np.zeros((self.k_max, 4), dtype=np.float32)
        obs_mask   = np.zeros((self.k_max,), dtype=np.float32)
        n = min(len(obstacles), self.k_max)
        if n > 0:
            obs_padded[:n] = obstacles[:n]
            obs_mask[:n] = 1.0

        start = (start - self.norm["start_mean"]) / self.norm["start_std"]
        goal  = (goal  - self.norm["goal_mean"])  / self.norm["goal_std"]
        if n > 0:
            obs_padded[:n] = (obs_padded[:n] - self.norm["obs_mean"]) / self.norm["obs_std"]

        start_t = torch.from_numpy(start).unsqueeze(0).to(self.device)
        goal_t  = torch.from_numpy(goal).unsqueeze(0).to(self.device)
        obs_t   = torch.from_numpy(obs_padded).unsqueeze(0).to(self.device)
        mask_t  = torch.from_numpy(obs_mask).unsqueeze(0).to(self.device)

        return start_t, goal_t, obs_t, mask_t

    # =================================================================
    # ── PROJECTION ─ helper: gather raw (un-normalized, ego-frame)
    #                obstacle state directly from the simulator state.
    #                We need this because obstacles_t passed to the
    #                diffusion is normalized; we want PHYSICAL units
    #                for the OCP.
    # =================================================================
    def _gather_obstacles_ego_physical(self, state):
        """
        Fill TWO independent obstacle pools for the MPC projection:

          slots 0 .. k_max-1          → dynamic agents (humans)
          slots k_max .. proj_n_obs-1 → static map cells

        The two pools never compete for the same slots, so a dense crowd
        cannot crowd out wall cells and vice versa.  Each pool is sorted
        by distance and capped at its budget.  pack_obstacle_params
        propagates dynamic agents forward with their velocity and leaves
        static cells stationary (v = 0).

        Returns (obs_xy, obs_v, radii) of length <= proj_n_obs.
        """
        theta = state.self_state.theta
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        R = np.array([[ cos_t,  sin_t],
                      [-sin_t,  cos_t]], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        # ── Pool A: dynamic agents → slots 0..k_max-1 ──────────────────
        dyn_items = []   # (dist, xy_ego, v_ego)

        for h in state.human_states:
            p_ego = R @ (np.array([h.px, h.py]) - pos_world)
            v_ego = R @ np.array([h.vx, h.vy])
            dyn_items.append((float(np.linalg.norm(p_ego)),
                              p_ego.astype(np.float32),
                              v_ego.astype(np.float32)))

        # Legacy static_obstacles from state attribute (point obstacles
        # without velocity — treat as dynamic-pool obstacles so they don't
        # consume static-map budget).
        extra_static = getattr(state, 'static_obstacles', None) or \
                       getattr(state, 'obstacles', None)
        if extra_static is not None:
            for s in extra_static:
                p_ego = R @ (np.array([s.px, s.py]) - pos_world)
                dyn_items.append((float(np.linalg.norm(p_ego)),
                                  p_ego.astype(np.float32),
                                  np.zeros(2, dtype=np.float32)))

        dyn_items.sort(key=lambda t: t[0])
        dyn_items = dyn_items[:self.k_max]

        # ── Pool B: static map cells → slots k_max..proj_n_obs-1 ───────
        static_items = []   # (dist, xy_ego)

        if (self.use_map
                and self._cur_occ_world_xy is not None
                and self._cur_has_map > 0.5
                and self._cur_occ_world_xy.shape[0] > 0
                and self.k_static > 0):
            px_r = state.self_state.px
            py_r = state.self_state.py
            pts_w = self._cur_occ_world_xy          # (N, 2) world frame
            ddx = pts_w[:, 0] - px_r
            ddy = pts_w[:, 1] - py_r
            x_e =  cos_t * ddx + sin_t * ddy
            y_e = -sin_t * ddx + cos_t * ddy
            dists_map = np.sqrt(x_e ** 2 + y_e ** 2)

            # Use the full map half-extent as reach so no cell that lies
            # within the loaded map is missed. _get_static_cells_ego uses
            # the same value for scoring, keeping both consistent.
            reach = self._cur_map_extent / 2.0
            sel   = dists_map < reach
            if sel.any():
                x_e_s = x_e[sel]; y_e_s = y_e[sel]; d_s = dists_map[sel]
                # k_static closest — argsort is exact (no partial sort needed
                # since the sel array is already bounded to a local region).
                order = np.argsort(d_s)[:self.k_static]
                for idx in order:
                    static_items.append((float(d_s[idx]),
                                         np.array([x_e_s[idx], y_e_s[idx]],
                                                  dtype=np.float32)))

        # ── Combine pools ────────────────────────────────────────────────
        total = len(dyn_items) + len(static_items)
        if total == 0:
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0,),   dtype=np.float32))

        xy_list, v_list = [], []
        for _, p, v in dyn_items:
            xy_list.append(p); v_list.append(v)
        for _, p in static_items:
            xy_list.append(p)
            v_list.append(np.zeros(2, dtype=np.float32))

        obs_xy = np.stack(xy_list, axis=0)
        obs_v  = np.stack(v_list,  axis=0)
        radii  = np.full(len(xy_list), float(self.safety_radius), dtype=np.float32)
        return obs_xy, obs_v, radii

    # =================================================================
    # ── PROJECTION ─ run the OCP and return the projected trajectory
    #                or None if the projection failed.
    # =================================================================
    def _project_trajectory(self, diffusion_xy_ego, state):
        """
        Project a candidate (px, py) trajectory onto the feasible /
        collision-free set under unicycle dynamics + dynamic obstacles.

        Indexing convention (matches the ORIGINAL diffusion code):
          The diffusion outputs T = self.horizon points at times
              0, dt, 2*dt, ..., (T-1)*dt
          where INDEX 0 is the current robot position (i.e. (0,0) in ego
          frame, modulo denoising noise) and indices 1..T-1 are future
          positions.

          The OCP has N = T-1 shooting intervals (T nodes total):
              node 0    : initial-condition node, x0 = current state
              nodes 1..N: track diffusion[1..T-1] as references

          We use the actual current state (0,0,0,v0,w0) for x0 — NOT
          diffusion[0] — because diffusion[0] is only an approximation
          of (0,0) and we know the true value exactly.

        Parameters
        ----------
        diffusion_xy_ego : (T, 2) ego-frame positions in physical units
                          (meters), as output by the diffusion model
                          after de-normalization.
        state            : simulator state.

        Returns
        -------
        proj_x_traj : (T, 5) projected state trajectory at horizon ticks
                      0, dt, ..., (T-1)*dt  (ego frame), or None on failure.
        proj_u_traj : (T-1, 2) projected inputs, or None.
        info        : dict with status, slack, deviation, timing, fallback
                      reason.
        """
        T = diffusion_xy_ego.shape[0]
        N = T - 1
        dt = self.proj_dt
        assert N == self.proj_N, (
            f"Projection horizon mismatch: diffusion gave T={T} points, "
            f"solver was built with N={self.proj_N} (expected N == T-1)"
        )

        # --- Build (N+1, 2) reference ----------------------------------
        # Index 0 must be (0,0) by the indexing convention. We *force*
        # it to (0,0) regardless of what the noisy diffusion[0] says,
        # since this stage's residual against x0 = (0,0,...) will be
        # zero anyway (x0 is a hard constraint), but we want a clean
        # cost expression and zero residual contribution from stage 0.
        ref_xy = np.zeros((T, 2), dtype=np.float64)
        ref_xy[0]  = 0.0
        ref_xy[1:] = diffusion_xy_ego[1:]

        # --- Initial state in ego frame --------------------------------
        v0 = float(getattr(self, '_v0_from_last_step', 0.0))
        w0 = float(getattr(self, '_w0_from_last_step', 0.0))
        x0 = np.array([0.0, 0.0, 0.0, v0, w0], dtype=np.float64)

        # --- Obstacles (physical units, ego frame) ---------------------
        obs_xy, obs_v, radii = self._gather_obstacles_ego_physical(state)

        # --- Per-node parameters (constant-velocity propagation) -------
        # n_obs_max must match what the solver was built with (proj_n_obs =
        # k_max + k_static). Dynamic agents (slots 0..k_max-1) are propagated
        # forward with their velocity; static cells (slots k_max..proj_n_obs-1)
        # have v=0 and stay fixed across the horizon.
        params_per_node = pack_obstacle_params(
            obs_xy, obs_v, radii,
            n_obs_max=self.proj_n_obs,
            N=N, dt=dt,
        )

        # --- Warm start ALWAYS from the diffusion path -----------------
        # See __init__ docstring for rationale.
        ws_x, ws_u = kinematic_warm_start_from_xy(
            ref_xy, dt=dt, theta0=0.0, v0=v0, omega0=w0,
        )

        # --- Solve -----------------------------------------------------
        result = solve_projection(
            self.proj_solver,
            x0=x0,
            ref_xy=ref_xy,
            params_per_node=params_per_node,
            warm_start_x=ws_x,
            warm_start_u=ws_u,
            N=N,
        )

        info = dict(
            status          = result["status"],
            max_slack       = result["max_slack"],
            solve_ms        = result["solve_ms"],
            max_dev         = None,
            fallback_reason = None,
        )

        # acados status: 0 = success, 2 = max iter (still usable in RTI)
        if result["status"] not in (0, 2):
            info["fallback_reason"] = f"acados status {result['status']}"
            return None, None, info

        proj_x = result["x_traj"]   # shape (N+1, 5) = (T, 5)
        proj_u = result["u_traj"]   # shape (N, 2)   = (T-1, 2)
        proj_xy = proj_x[:, 0:2]

        # Deviation against the diffusion reference. Skip index 0 since
        # both projection and reference are pinned to (0,0) there.
        dev = np.linalg.norm(proj_xy[1:] - ref_xy[1:], axis=1)
        max_dev = float(dev.max()) if dev.size > 0 else 0.0
        info["max_dev"] = max_dev

        if result["max_slack"] > self.proj_max_slack_thresh:
            info["fallback_reason"] = (
                f"obstacle slack {result['max_slack']:.3f} "
                f"> {self.proj_max_slack_thresh}"
            )
            return None, None, info

        if max_dev > self.proj_max_dev_thresh:
            info["fallback_reason"] = (
                f"deviation {max_dev:.3f} m > {self.proj_max_dev_thresh} m "
                f"(likely local minimum, e.g. stop-in-front-of-obstacle)"
            )
            return None, None, info

        return proj_x, proj_u, info

    # =================================================================
    # ── PROJECTION ─ controlled-stop fallback action.
    # =================================================================
    def _emergency_brake_action(self):
        """
        Apply a controlled deceleration: command a speed that is one
        max-decel-step below the previous executed speed, omega ramps to
        zero. Returns an ActionXY or ActionRot depending on kinematics.
        """
        decel = self.max_accel if np.isfinite(self.max_accel) else 2.0
        wdecel = self.max_w_accel if np.isfinite(self.max_w_accel) else 3.0

        v_target = max(0.0, self.prev_v - decel * self.time_step)
        if self.prev_omega > 0:
            w_target = max(0.0, self.prev_omega - wdecel * self.time_step)
        else:
            w_target = min(0.0, self.prev_omega + wdecel * self.time_step)

        if self.kinematics == "holonomic":
            return ActionXY(0.0, 0.0)
        else:
            return ActionRot(v_target, w_target * self.time_step)

    # =================================================================
    # ── PROJECTION ─ record a fallback frame in predicted_traj so the
    #                 visualizer can distinguish brake events from normal
    #                 steps. Keeps diffusion candidates visible (for
    #                 debugging) but sets projection=None.
    # =================================================================
    def _record_predicted_traj_fallback(self, state, x_tensor):
        """
        Build the predicted_traj dict for a brake frame.
        x_tensor : (K, traj_dim, T) post-scoring (best at index 0).
        """
        theta_robot = state.self_state.theta
        R_world = np.array([
            [np.cos(theta_robot), -np.sin(theta_robot)],
            [np.sin(theta_robot),  np.cos(theta_robot)]
        ], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        traj_norm_np = x_tensor.cpu().numpy()
        K = traj_norm_np.shape[0]
        all_samples_world = []
        for k in range(K):
            traj = traj_norm_np[k].T * self.norm["traj_std"] + self.norm["traj_mean"]
            traj_world = np.array([pos_world + R_world @ np.array([dx, dy])
                                   for dx, dy in traj])
            all_samples_world.append(traj_world)

        self.predicted_traj = {
            'selected_sample': all_samples_world[0] if K > 0 else None,
            'projection'     : None,             # ← brake event
            'all_samples'    : all_samples_world,
            'used_projection': False,
        }

    # =================================================================
    # ── PROJECTION ─ interpolate the projected state at any time t in
    #                [0, N*proj_dt]. Linearly interpolates px, py, v,
    #                omega; SLERP-style interpolates theta to handle
    #                wrap-around. Used to support sim_timestep != proj_dt.
    # =================================================================
    def _interp_projected_state(self, proj_x_traj, t):
        """
        proj_x_traj : (N+1, 5) — state trajectory at horizon ticks 0,dt,...,N*dt
        t           : float — query time in [0, N*proj_dt]
        Returns
        -------
        px, py, theta, v, omega  (floats)
        """
        N = proj_x_traj.shape[0] - 1
        dt = self.proj_dt
        T_horizon = N * dt

        # Clamp to within the horizon.
        t = max(0.0, min(t, T_horizon - 1e-9))

        # Find bracketing indices.
        idx_lo = int(np.floor(t / dt))
        idx_hi = min(idx_lo + 1, N)
        frac   = (t - idx_lo * dt) / dt

        s_lo = proj_x_traj[idx_lo]
        s_hi = proj_x_traj[idx_hi]

        # Linear interp for px, py, v, omega
        px    = (1 - frac) * s_lo[0] + frac * s_hi[0]
        py    = (1 - frac) * s_lo[1] + frac * s_hi[1]
        v     = (1 - frac) * s_lo[3] + frac * s_hi[3]
        omega = (1 - frac) * s_lo[4] + frac * s_hi[4]

        # Heading: interpolate the unwrapped delta to handle 2π discontinuity.
        dtheta = s_hi[2] - s_lo[2]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        theta  = s_lo[2] + frac * dtheta

        return float(px), float(py), float(theta), float(v), float(omega)

    # =================================================================
    # predict()
    # =================================================================
    def predict(self, state):
        t_predict_start = datetime.now()

        if self.reach_destination(state):
            return ActionRot(0, 0)

        start, goal, obstacles, obs_mask = self.build_condition_from_state(state)

        # Prepare map tensors for this step (no-op when use_map=False)
        if self.use_map:
            occ_np, has_map_f = self._rasterize_ego_map(state)
            occ_map_t = torch.from_numpy(occ_np).unsqueeze(0).unsqueeze(0).to(self.device)
            has_map_t = torch.tensor([has_map_f], dtype=torch.float32, device=self.device)
        else:
            occ_map_t = None
            has_map_t = None

        K = self.num_samples

        x = torch.randn(K, self.traj_dim, self.horizon, device=self.device)

        with torch.no_grad():
            # ── 6-pass compositional CFG — matches the training script exactly ──
            #
            # Variant layout (N_V = 2 + N_STYLE_AXES = 6):
            #   V0: scene_drop=1, all style null          → eps_uncond
            #   V1: scene_drop=0, all style null          → eps_scene
            #   V2: scene_drop=0, only axis 0 kept        → eps_scene_plus_prox
            #   V3: scene_drop=0, only axis 1 kept        → eps_scene_plus_pass
            #   V4: scene_drop=0, only axis 2 kept        → eps_scene_plus_yield
            #   V5: scene_drop=0, only axis 3 kept        → eps_scene_plus_group
            #
            # Style variants keep scene ON because at deployment the policy
            # always sees real scene conditioning; the style guidance is a
            # delta from that scene-conditional baseline, not from null.
            #
            #   eps = eps_uncond
            #         + cfg_w_scene * (eps_scene             - eps_uncond)
            #         + Σ_i cfg_w_style * (eps_scene_plus_axis_i - eps_scene)
            #
            # At default weights (cfg_w_scene = cfg_w_style = 1) this
            # reduces exactly to the log-product-of-experts target for the
            # compositional generative model defined by independent
            # per-axis dropout in training.
            #
            # This is the compositional structure the model was trained with.
            # cfg_w_scene and cfg_w_style are independent knobs in policy.config.

            N_V = 2 + self.n_style_axes   # = 6 for 4 axes

            style_vals_t = torch.tensor(
                self.style_vector, dtype=torch.float32, device=self.device
            ).unsqueeze(0)                                # [1, n_axes]

            # Variant layout:
            #   V0:           scene_drop=1, all style dropped  -> eps_uncond
            #   V1:           scene_drop=0, all style dropped  -> eps_scene
            #   V2..V_{n+1}:  scene_drop=0, only axis i kept   -> eps_scene_plus_axis_i
            #
            # The compositional CFG formula then subtracts eps_scene (not
            # eps_uncond) from each axis variant, so the style guidance is
            # defined as a delta from the scene-conditional baseline. This
            # matches the regime the model is used in at deployment.
            scene_drop_v = torch.zeros(N_V, device=self.device)
            scene_drop_v[0] = 1.0   # V0: full null
 
            style_drop_v = torch.ones(N_V, self.n_style_axes, device=self.device)
            # V0: all dropped (already)
            # V1: all dropped (already)
            for i in range(self.n_style_axes):
                style_drop_v[2 + i, i] = 0.0   # keep axis i in variant 2+i

            # Expand scene inputs to N_V for one batched embedder call
            start_v     = start.expand(N_V, -1)
            goal_v      = goal.expand(N_V, -1)
            obstacles_v = obstacles.expand(N_V, -1, -1)
            obs_mask_v  = obs_mask.expand(N_V, -1)
            if occ_map_t is not None:
                occ_map_v = occ_map_t.expand(N_V, -1, -1, -1)
                has_map_v = has_map_t.expand(N_V)
            else:
                occ_map_v = None
                has_map_v = None
            style_vals_v = style_vals_t.expand(N_V, -1)   # [N_V, n_axes]

            # Build all N_V token sequences in a single batched embedder call
            style_toks_v = self.style_embedder(style_vals_v, style_drop_v)
            tokens_v, _  = self.token_embedder(
                start_v, goal_v, obstacles_v, obs_mask_v,
                occ_map=occ_map_v, has_map=has_map_v,
                style_tokens=style_toks_v,
                scene_drop=scene_drop_v,
            )   # [N_V, seq_len, D]

            n_map_tokens = self.token_embedder.n_map_tokens if self.use_map else 0
            attn_mask_v  = build_attn_mask(
                obs_mask_v,
                n_map_tokens=n_map_tokens,
                n_style_tokens=self.n_style_axes,
            )   # [N_V, seq_len]

            # Force obstacle slots valid where scene is dropped so the transformer
            # attends to the null tokens that occupy those positions.
            sdv = scene_drop_v.bool().unsqueeze(1)   # [N_V, 1]
            attn_mask_v = attn_mask_v.clone()
            attn_mask_v[:, 2:2 + self.k_max] = torch.where(
                sdv.expand(-1, self.k_max),
                torch.ones_like(attn_mask_v[:, 2:2 + self.k_max]),
                attn_mask_v[:, 2:2 + self.k_max],
            )

            # Expand to [N_V * K, seq_len, D] — replicate each variant across K
            # noise samples (the K parallel trajectory candidates share tokens).
            tokens_6k    = (tokens_v.unsqueeze(1)
                            .expand(-1, K, -1, -1)
                            .contiguous()
                            .reshape(N_V * K, tokens_v.shape[1], tokens_v.shape[2]))
            attn_mask_6k = (attn_mask_v.unsqueeze(1)
                            .expand(-1, K, -1)
                            .contiguous()
                            .reshape(N_V * K, attn_mask_v.shape[1]))
            # attn_mask_6k never changes across the denoising loop (it only
            # depends on the scene, not on x or t) — cast to bool once here
            # instead of every one of the 5 denoising steps.
            attn_mask_6k_bool = attn_mask_6k.bool()

        denoise_mode = "DDIM" if self.ddim_inference else "DDPM"
        time1 = datetime.now()

        traj_std  = self.norm_t["traj_std"]
        traj_mean = self.norm_t["traj_mean"]
        obs_std   = self.norm_t["obs_std"]
        obs_mean  = self.norm_t["obs_mean"]
        obs_denorm = obstacles * obs_std + obs_mean

        if self.ddim_inference:
            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.noise_scheduler.timesteps
        else:
            timesteps = reversed(range(self.noise_scheduler.config.num_train_timesteps))

        for t in timesteps:
            # Single (N_V*K,)-shaped fill, built directly instead of a
            # (K,) tensor + .repeat(N_V) (same broadcasted values, one
            # allocation instead of two).
            timestep_6k = torch.full((N_V * K,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                # Expand x across N_V variants: [N_V*K, D, T]
                x_6k = (x.unsqueeze(0)
                         .expand(N_V, -1, -1, -1)
                         .contiguous()
                         .reshape(N_V * K, self.traj_dim, self.horizon))

                # Optional bf16/fp16 autocast around the UNet forward only
                # (config flag `use_amp`). Everything before/after this
                # call — CFG combination, scheduler.step, scoring — stays
                # in fp32, so amp only affects the one op we expect to
                # actually benefit from Tensor Core throughput.
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                    if (self.use_amp and self.device.type == "cuda")
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    noise_6k = self.model(
                        sample                 = x_6k,
                        timestep               = timestep_6k,
                        encoder_hidden_states  = tokens_6k,
                        encoder_attention_mask = attn_mask_6k_bool,
                    ).sample                                          # [N_V*K, D, T]
                noise_6k = noise_6k.float()

                noise_v    = noise_6k.view(N_V, K, self.traj_dim, self.horizon)
                eps_uncond = noise_v[0]       # [K, D, T]  full null
                eps_scene  = noise_v[1]       # [K, D, T]  scene only
                eps_axes   = noise_v[2:]      # [n_axes, K, D, T]  per-axis style

                # Compositional CFG:
                #   eps = eps_uncond
                #       + w_scene * (eps_scene             - eps_uncond)
                #       + Σ_i w_i * (eps_scene_plus_axis_i - eps_scene)
                #
                # The style delta is computed relative to eps_scene rather
                # than eps_uncond, because each style variant carries the
                # scene context (scene_drop=0). This matches how the model
                # is conditioned in deployment.
                noise_pred = eps_uncond + self.cfg_w_scene * (eps_scene - eps_uncond)
                for i in range(self.n_style_axes):
                    noise_pred = noise_pred + self.cfg_w_style * (eps_axes[i] - eps_scene)

            if self.classifierguide and t < self.num_diffusion_timesteps // 2:
                x = x.detach().requires_grad_(True)
                alpha_bar = self.noise_scheduler.alphas_cumprod[t]
                x0_pred = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
                x0_denorm = x0_pred * traj_std.view(1, -1, 1) + traj_mean.view(1, -1, 1)
                cost = collision_cost(x0_denorm.unsqueeze(0), obs_denorm, obs_mask).sum()
                grad = torch.autograd.grad(cost, x)[0]
                with torch.no_grad():
                    print(f"t={t}  cost={cost.item():.4f}  grad_norm={grad.norm().item():.4f}", flush=True)
                    x = x - self.guidance_scale * grad
                    x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
            else:
                with torch.no_grad():
                    x = self.noise_scheduler.step(noise_pred, t, x).prev_sample

        with torch.no_grad():
            time2 = datetime.now()
            print(f"{denoise_mode} sampling ({K} trajs):", time2 - time1)

            x_scored = x.unsqueeze(0)
            obs_b    = obstacles
            mask_b   = obs_mask
            goal_b   = goal

            # Gather static map cells in ego frame (physical metres) for
            # scoring.  This makes sample selection wall-aware — not just
            # the MPC projection layer.  No-map episodes return None and
            # score_trajectories degrades to dynamic-only scoring.
            static_obs_xy_t = None
            if self.use_map:
                static_xy_np = self._get_static_cells_ego(state)
                if static_xy_np is not None:
                    static_obs_xy_t = (
                        torch.from_numpy(static_xy_np)
                        .unsqueeze(0)          # (1, M, 2)
                        .to(self.device)
                    )

            rewards, best_idx = score_trajectories(
                x_scored, obs_b, mask_b, goal_b, self.norm_t,
                collision_coef      = self.collision_coef,
                smooth_pen_coef     = self.smooth_pen_coef,
                goal_reward_coef    = self.goal_reward_coef,
                control_effort_coef = self.control_effort_coef,
                safety_radius       = self.safety_radius,
                dt                  = 0.05,
                static_obs_xy       = static_obs_xy_t,
            )
            best_k = best_idx[0].item()
            print(f"Rewards: {rewards[0].cpu().numpy()}  → selected traj {best_k}")

            idx_order = [best_k] + [i for i in range(K) if i != best_k]
            x = x[idx_order]
            main_traj = x[0]

        # =================================================================
        # ── PROJECTION ─ project the selected diffusion trajectory onto
        #               the feasible, collision-free set.
        # =================================================================
        # Denormalize the selected diffusion trajectory to physical
        # (ego-frame) units. Shape is (T, 2). Convention (matches the
        # ORIGINAL diffusion code):
        #   index 0     = current position (≈ (0,0), modulo denoise noise)
        #   indices 1.. = future positions at times dt, 2*dt, ...
        diffusion_xy_ego = (main_traj.cpu().numpy().T
                            * self.norm["traj_std"]
                            + self.norm["traj_mean"])     # shape (T, 2)

        used_proj   = False
        proj_info   = None
        proj_x_traj = None  # (T, 5) — same length as diffusion
        proj_u_traj = None  # (T-1, 2)

        if self.use_projection:
            proj_x_traj, proj_u_traj, proj_info = \
                self._project_trajectory(diffusion_xy_ego, state)

            if proj_x_traj is not None:
                used_proj = True
                exec_traj_ego = proj_x_traj[:, 0:2]   # (T, 2)
                print(f"  [proj] OK  status={proj_info['status']}  "
                      f"slack={proj_info['max_slack']:.4f}  "
                      f"max_dev={proj_info['max_dev']:.3f} m  "
                      f"solve={proj_info['solve_ms']:.2f} ms")
            else:
                print(f"  [proj] FALLBACK: {proj_info['fallback_reason']}")
                # Apply the same controlled deceleration we're about to
                # command, in self.prev_v / self.prev_omega, so that the
                # next tick's enforce_lims clipping anchors at the
                # post-brake state rather than stale pre-brake values.
                decel = self.max_accel if np.isfinite(self.max_accel) else 2.0
                wdecel = self.max_w_accel if np.isfinite(self.max_w_accel) else 3.0
                v_brake = max(0.0, self.prev_v - decel * self.time_step)
                if self.prev_omega > 0:
                    w_brake = max(0.0, self.prev_omega - wdecel * self.time_step)
                else:
                    w_brake = min(0.0, self.prev_omega + wdecel * self.time_step)
                self.prev_v, self.prev_omega = v_brake, w_brake

                # Same values for next-tick diffusion conditioning.
                self._v0_from_last_step = v_brake
                self._w0_from_last_step = w_brake

                # Record fallback frame for the visualizer: keep diffusion
                # candidates visible (so we can see WHAT was rejected) but
                # set projection=None and used_projection=False to flag the
                # brake event distinctly from a normal step.
                self._record_predicted_traj_fallback(state, x)
                print(f"  [predict] BRAKE total: "
                      f"{(datetime.now() - t_predict_start).total_seconds()*1e3:.1f} ms")

                return self._emergency_brake_action()
        else:
            # No projection: fall through to original action-extraction
            # logic on the raw diffusion path. exec_traj_ego shape is (T, 2)
            # with index 0 being the (denoised approximation of the) current
            # position.
            exec_traj_ego = diffusion_xy_ego

        # =================================================================
        # Visualization data: store all candidates, the selected diffusion
        # sample, and the projected trajectory as separate world-frame
        # entries so consumers can distinguish them.
        #
        # self.predicted_traj is a dict:
        #     'selected_sample' : (T, 2) world-frame — the K-of-many
        #                          diffusion sample chosen by scoring,
        #                          BEFORE projection.
        #     'projection'      : (T, 2) world-frame — the feasible/
        #                          collision-free projection of the
        #                          selected sample. None if projection
        #                          was disabled or fell back.
        #     'all_samples'     : list of K (T, 2) world-frame arrays —
        #                          every diffusion candidate. Index 0 is
        #                          the selected sample (matches
        #                          'selected_sample').
        #     'used_projection' : bool — True iff the executed action came
        #                          from the projection.
        # =================================================================
        theta_robot = state.self_state.theta
        R_world = np.array([
            [np.cos(theta_robot), -np.sin(theta_robot)],
            [np.sin(theta_robot),  np.cos(theta_robot)]
        ], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        def _ego_to_world(traj_ego):
            return np.array([pos_world + R_world @ np.array([dx, dy])
                             for dx, dy in traj_ego])

        # All K diffusion candidates (de-normalized, world frame).
        # x has been reordered so index 0 is the selected sample.
        traj_norm_np = x.cpu().numpy()
        all_samples_world = []
        for k in range(K):
            traj = traj_norm_np[k].T * self.norm["traj_std"] + self.norm["traj_mean"]
            all_samples_world.append(_ego_to_world(traj))

        # Selected diffusion sample (matches all_samples_world[0]).
        selected_sample_world = all_samples_world[0]

        # Projection (only if we actually used it).
        if used_proj and proj_x_traj is not None:
            projection_world = _ego_to_world(proj_x_traj[:, 0:2])
        else:
            projection_world = None

        self.predicted_traj = {
            'selected_sample': selected_sample_world,
            'projection'     : projection_world,
            'all_samples'    : all_samples_world,
            'used_projection': used_proj,
        }

        # =================================================================
        # Action extraction.
        #
        # General convention: simulator advances by self.time_step which
        # may be larger than self.proj_dt. We compute the action by
        # interpolating the executed trajectory at t = self.time_step
        # (= where the robot needs to be one sim-tick from now, and at
        # what velocity).
        #
        # Time axis convention (matches original code exactly):
        #   exec_traj_ego[0] is at time 0  (= current position, (0,0))
        #   exec_traj_ego[k] is at time k * proj_dt
        # So for self.time_step = k * proj_dt the displacement to reach
        # over one sim-tick is exec_traj_ego[k] - exec_traj_ego[0] which
        # for the matched-time case (time_step = proj_dt) reduces to
        # exec_traj_ego[1] - (0,0) = exec_traj_ego[1].
        #
        # If projection was used we have v(t) and omega(t) directly from
        # the OCP solution and use those rather than finite-differencing.
        # =================================================================
        dt_diff   = self.proj_dt
        T_horizon = (exec_traj_ego.shape[0] - 1) * dt_diff
        t_apply   = min(self.time_step, T_horizon - 1e-9)

        if used_proj and proj_x_traj is not None:
            # Read v, omega, theta directly from the projected state
            # interpolated at t = time_step.
            _, _, theta_apply, v_apply, omega_apply = \
                self._interp_projected_state(proj_x_traj, t_apply)

            v     = float(v_apply)
            omega = float(omega_apply)

            # Cache for next-tick diffusion conditioning. By the next
            # call, the robot will have advanced by self.time_step, so
            # the cached v0/w0 should be the values at t = time_step.
            self._v0_from_last_step = v
            self._w0_from_last_step = omega

            if self.kinematics == "holonomic":
                vx_ego = v * np.cos(theta_apply)
                vy_ego = v * np.sin(theta_apply)
                v_world = R_world @ np.array([vx_ego, vy_ego])
                action_world_xy = (float(v_world[0]), float(v_world[1]))
            else:
                action_world_xy = None

        else:
            # ----- Original action-extraction (use_projection=False) ----
            # Replicates the original code's logic exactly.
            traj = exec_traj_ego  # shape (T, 2), index 0 = current pos

            steps  = self.time_step / dt_diff
            idx_lo = int(np.floor(steps))
            idx_hi = min(idx_lo + 1, traj.shape[0] - 1)
            interp = steps - idx_lo

            # Position to reach at t = time_step  (linear interp in time)
            dx = traj[idx_lo, 0] * (1 - interp) + traj[idx_hi, 0] * interp
            dy = traj[idx_lo, 1] * (1 - interp) + traj[idx_hi, 1] * interp

            # IMPORTANT: this displacement is FROM CURRENT POSITION (0,0).
            # Since traj[0] = (0,0) by convention, dx, dy is the absolute
            # ego-frame target position, which equals the displacement.
            vx = dx / self.time_step
            vy = dy / self.time_step
            v  = float(np.sqrt(vx**2 + vy**2))

            theta_desired = float(np.arctan2(dy, dx)) \
                            if (abs(dx) + abs(dy)) > 1e-6 else 0.0
            omega = theta_desired / self.time_step

            self._v0_from_last_step = v
            self._w0_from_last_step = omega

            if self.kinematics == "holonomic":
                v_world = R_world @ np.array([vx, vy])
                action_world_xy = (float(v_world[0]), float(v_world[1]))
            else:
                action_world_xy = None

        # ------ enforce_lims clipping (identical policy to original) ----
        if self.enforce_lims:
            v_clipped = np.clip(v, -self.max_vel, self.max_vel)
            omega_clipped = np.clip(omega, -self.max_wrot, self.max_wrot)
            v_clipped = np.clip(
                v_clipped,
                self.prev_v - self.max_accel * self.time_step,
                self.prev_v + self.max_accel * self.time_step,
            )
            omega_clipped = np.clip(
                omega_clipped,
                self.prev_omega - self.max_w_accel * self.time_step,
                self.prev_omega + self.max_w_accel * self.time_step,
            )
            self.prev_v, self.prev_omega = v_clipped, omega_clipped
            # Sync diffusion-conditioning state to the value the robot will
            # actually be at next tick. Without this, the diffusion is fed
            # an unclipped target velocity the robot never reached.
            self._v0_from_last_step = float(v_clipped)
            self._w0_from_last_step = float(omega_clipped)
            v_out, omega_out = v_clipped, omega_clipped
        else:
            self.prev_v, self.prev_omega = v, omega
            self._v0_from_last_step = float(v)
            self._w0_from_last_step = float(omega)
            v_out, omega_out = v, omega

        if self.kinematics == "holonomic":
            print(f"  [predict] total: "
                  f"{(datetime.now() - t_predict_start).total_seconds()*1e3:.1f} ms")
            return ActionXY(*action_world_xy)
        else:
            print(f"  [predict] total: "
                  f"{(datetime.now() - t_predict_start).total_seconds()*1e3:.1f} ms")
            return ActionRot(v_out, omega_out * self.time_step)
