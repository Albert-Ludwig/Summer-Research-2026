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
sys.path.append("/home/cschaibl/projects/aip-sl2smith/cschaibl/diffusion_planning")  # add the parent folder of somepackage

from diffusers_unet_1d_condition import UNet1DConditionModel



class SceneTokenEmbedder(nn.Module):
    def __init__(self, start_dim, goal_dim, obs_dim, token_dim, k_max):
        super().__init__()
        # Using the hidden_dim pattern from your MLPLayer example
        hidden_dim = token_dim
        self.null_token = nn.Parameter(torch.randn(1, 2 + k_max, token_dim) * 0.02)

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

        # Learned "Identity" tokens to help Cross-Attention distinguish roles
        # We use a smaller initialization to ensure spatial data leads initially
        self.type_embed = nn.Parameter(torch.randn(3, token_dim) * 0.02)
        
        # Final norm to fuse spatial data + type embedding
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(self, start, goal, obstacles):
        """
        start: [B, Ds], goal: [B, Dg], obstacles: [B, K, Do]
        """
        # 1. Project raw states to the token dimension
        # The MLPLayer already includes LayerNorm and SiLU
        start_tok = self.start_proj(start).unsqueeze(1) # [B, 1, D]
        goal_tok  = self.goal_proj(goal).unsqueeze(1)   # [B, 1, D]
        obs_tok   = self.obs_proj(obstacles)            # [B, K, D]

        # 2. Add Type Embeddings (Broadcasting handles the Batch and K dimensions)
        # Type 0: Start, Type 1: Goal, Type 2: Obstacles
        start_tok = start_tok + self.type_embed[0]
        goal_tok  = goal_tok + self.type_embed[1]
        obs_tok   = obs_tok + self.type_embed[2]

        # 3. Concatenate all tokens
        tokens = torch.cat([start_tok, goal_tok, obs_tok], dim=1) # [B, 2+K, D]

        # 4. Final Norm: This prevents any one component (like type_embed) 
        # from dominating the magnitude of the vectors sent to the UNet.
        return self.final_norm(tokens)



def build_attn_mask(obs_mask):
    """
    obs_mask: [B, K]  (1 = valid, 0 = padded)
    """
    B, K = obs_mask.shape
    device = obs_mask.device

    base = torch.ones(B, 2, device=device)  # start + goal
    attn_mask = torch.cat([base, obs_mask], dim=1)
    return attn_mask


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
    """
    x_trajs_denorm: [B, K, D, T]
    Penalizes turning / lateral effort by weighting lateral velocity changes
    more heavily than forward ones. Keeps units consistent with smoothness_penalty.
    """
    pos = x_trajs_denorm[:, :, 0:2, :]              # [B, K, 2, T]
    vel = pos[:, :, :, 1:] - pos[:, :, :, :-1]      # [B, K, 2, T-1]

    # In ego frame: x=forward, y=lateral
    forward_effort = vel[:, :, 0, :].pow(2)          # [B, K, T-1]
    lateral_effort = vel[:, :, 1, :].pow(2)          # [B, K, T-1]

    # Weight lateral more heavily to discourage turning
    effort = forward_effort + 3.0 * lateral_effort   # [B, K, T-1]
    return effort.mean(dim=2)                         # [B, K]

def collision_cost(x_trajs, obstacles, obs_mask, dt=0.05, safety_radius=0.5):
    """
    x_trajs:   [B, K, D, T]  — normalized ego-frame trajectories
    obstacles: [B, k_max, 4] — (x, y, vx, vy) normalized ego-frame
    obs_mask:  [B, k_max]    — 1=valid, 0=pad
    returns:   [B, K]
    """
    B, K, D, T = x_trajs.shape
    traj_xy = x_trajs[:, :, 0:2, :].permute(0, 1, 3, 2)  # [B, K, T, 2]

    obs_pos = obstacles[:, :, 0:2].unsqueeze(2)            # [B, k_max, 1, 2]
    obs_vel = obstacles[:, :, 2:4].unsqueeze(2)            # [B, k_max, 1, 2]
    t_vec   = torch.arange(T, device=x_trajs.device).float() * dt
    t_vec   = t_vec.view(1, 1, T, 1)

    obs_future = obs_pos + obs_vel * t_vec                  # [B, k_max, T, 2]

    traj_xy    = traj_xy.unsqueeze(2)                       # [B, K, 1, T, 2]
    obs_future = obs_future.unsqueeze(1)                    # [B, 1, k_max, T, 2]

    dist       = torch.norm(traj_xy - obs_future, dim=-1)   # [B, K, k_max, T]
    penetration = F.relu(safety_radius - dist)
    penalty     = penetration ** 2

    obs_mask_exp = obs_mask.unsqueeze(1).unsqueeze(-1)      # [B, 1, k_max, 1]
    cost = (penalty * obs_mask_exp).sum(dim=[2, 3])         # [B, K]
    return cost


def smoothness_penalty(x_trajs_denorm):
    """
    x_trajs_denorm: [B, K, D, T]
    returns: [B, K]
    """
    pos     = x_trajs_denorm[:, :, 0:2, :]
    vel     = pos[:, :, :, 1:] - pos[:, :, :, :-1]
    acc     = vel[:, :, :, 1:] - vel[:, :, :, :-1]
    acc_mag = acc.pow(2).sum(dim=2)                         # [B, K, T-2]
    return acc_mag.pow(1.5).mean(dim=2)                     # [B, K]


def goal_progress_reward(x_trajs_denorm, goal, goal_mean, goal_std):
    """
    x_trajs_denorm: [B, K, D, T]
    goal:           [B, 2]  — normalized ego-frame goal
    returns:        [B, K]
    """
    device = x_trajs_denorm.device
    goal_mean_t = torch.tensor(goal_mean[0:2], dtype=torch.float32, device=device)
    goal_std_t  = torch.tensor(goal_std[0:2],  dtype=torch.float32, device=device)

    goal_xy = (goal[:, 0:2] * goal_std_t + goal_mean_t)    # [B, 2]
    goal_xy = goal_xy.unsqueeze(1).unsqueeze(3)             # [B, 1, 2, 1]

    traj_xy       = x_trajs_denorm[:, :, 0:2, :]           # [B, K, 2, T]
    dist_to_goal  = torch.norm(traj_xy - goal_xy, dim=2)   # [B, K, T]

    progress      = dist_to_goal[:, :, 0] - dist_to_goal[:, :, -1]
    step_progress = (dist_to_goal[:, :, :-1] - dist_to_goal[:, :, 1:]).mean(dim=2)
    return progress + 0.5 * step_progress                   # [B, K]


def score_trajectories(x_trajs, obstacles, obs_mask, goal, norm,
                       collision_coef, smooth_pen_coef, goal_reward_coef, control_effort_coef,
                       safety_radius=0.5, dt=0.05):
    """
    x_trajs:   [B, K, D, T]  — normalized
    obstacles: [B, k_max, 4] — normalized
    obs_mask:  [B, k_max]
    goal:      [B, goal_dim] — normalized
    returns:   rewards [B, K], best_idx [B]
    """
    traj_std  = torch.tensor(norm["traj_std"], dtype=torch.float32, device=x_trajs.device)
    traj_mean = torch.tensor(norm["traj_mean"], dtype=torch.float32, device=x_trajs.device)

    obs_std  = torch.tensor(norm["obs_std"],  dtype=torch.float32, device=x_trajs.device)
    obs_mean = torch.tensor(norm["obs_mean"], dtype=torch.float32, device=x_trajs.device)
    obs_denorm = obstacles * obs_std + obs_mean   # [B, k_max, 4]

    # Denormalize for physical-space cost functions: traj_std/mean are [2] or [D]
    x_denorm = x_trajs * traj_std.view(1, 1, -1, 1) + traj_mean.view(1, 1, -1, 1)

    C          = collision_cost(x_denorm, obs_denorm, obs_mask,
                                dt=dt, safety_radius=safety_radius)
    smooth_pen = smoothness_penalty(x_denorm)
    goal_rew   = goal_progress_reward(x_denorm, goal,
                                      norm["goal_mean"], norm["goal_std"])
    effort_pen   = control_effort_penalty(x_denorm)

    rewards = (
        - collision_coef    * C
        - smooth_pen_coef   * smooth_pen
        - control_effort_coef * effort_pen
        + goal_reward_coef  * goal_rew
    )  # [B, K]

    best_idx = rewards.argmax(dim=1)   # [B]
    return rewards, best_idx

class DiffusionConditionalUNet1DCFG_FEASIBLE(Policy):
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
        self.name = "DiffusionConditionalUNet1DCFG_FEASIBLE"

    def configure(self, config):
        import torch
        import numpy as np

        self.start_dim = config.getint('diffusion_conditional_unet1dcfg_feasible', 'start_dim')
        self.goal_dim = config.getint('diffusion_conditional_unet1dcfg_feasible', 'goal_dim')
        self.token_dim = config.getint('diffusion_conditional_unet1dcfg_feasible', 'token_dim')
        self.horizon = config.getint('diffusion_conditional_unet1dcfg_feasible', 'horizon')
        self.k_max = config.getint('diffusion_conditional_unet1dcfg_feasible', 'k_max')
        self.kinematics = config.get('action_space', 'kinematics')
        self.enforce_lims = config.getboolean('action_space', 'enforce_lims')
        self.max_vel = config.getfloat('action_space', 'max_vel')
        self.max_wrot = config.getfloat('action_space', 'max_wrot')
        self.max_accel = config.getfloat('action_space', 'max_accel')
        self.max_w_accel = config.getfloat('action_space', 'max_w_accel')
        self.multiagent_training = config.getboolean('diffusion_conditional_unet1dcfg_feasible', 'multiagent_training')
        self.num_diffusion_timesteps = config.getint('diffusion_conditional_unet1dcfg_feasible', 'num_diffusion_timesteps')
        self.num_inference_steps = config.getint('diffusion_conditional_unet1dcfg_feasible', 'num_inference_steps')
        self.ddim_inference = config.getboolean('diffusion_conditional_unet1dcfg_feasible', 'ddim_inference')
        self.cfg_weight = config.getfloat('diffusion_conditional_unet1dcfg_feasible', 'cfg_weight')
        self.num_samples = config.getint('diffusion_conditional_unet1dcfg_feasible', 'num_samples')
        self.collision_coef = config.getfloat('diffusion_conditional_unet1dcfg_feasible', 'collision_coef')
        self.smooth_pen_coef = config.getfloat('diffusion_conditional_unet1dcfg_feasible', 'smooth_pen_coef')
        self.goal_reward_coef = config.getfloat('diffusion_conditional_unet1dcfg_feasible', 'goal_reward_coef')
        self.safety_radius = config.getfloat('diffusion_conditional_unet1dcfg_feasible', 'safety_radius')
        self.control_effort_coef = config.getfloat('diffusion_conditional_unet1dcfg_feasible', 'control_effort_coef')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print("Device:", device)

        # Load normalization stats
        norm_file = config.get('diffusion_conditional_unet1dcfg_feasible', 'norm_file')
        self.norm = np.load(norm_file, allow_pickle=True).item()

        self.traj_dim=2

        self.token_embedder = SceneTokenEmbedder(
            start_dim=self.start_dim,
            goal_dim=self.goal_dim,
            obs_dim=4,
            token_dim=self.token_dim,
            k_max = self.k_max
        ).to(self.device)
        
        # Construct model + diffusion
        self.model = UNet1DConditionModel(
            sample_size=self.horizon,          # T = 32
            in_channels=self.traj_dim,   # trajectory + conditioning channels + time embedding
            out_channels=self.traj_dim,             # predict noise on trajectory only
            layers_per_block=2,
            block_out_channels=(128, 256),
            cross_attention_dim=self.token_dim,  # conditioning vector dimension
            down_block_types=(
                "CrossAttnDownBlock1D",  # cross-attention uses encoder_hidden_states
                "CrossAttnDownBlock1D",
            ),
            up_block_types=(
                "CrossAttnUpBlock1D",
                "UpBlock1D",              # optional: just convolution up
            ),
            mid_block_type="UNetMidBlock1DCrossAttn",  # middle bottleneck block with cross-attention
        )
        self.model = self.model.to(self.device)

        if self.ddim_inference:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="sample",   # matches DDPM setup
                clip_sample=False,           # keep consistent with training
            )
        else:
            self.noise_scheduler = DDPMScheduler( #Can use DDIM later for inference
                num_train_timesteps=self.num_diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="sample",  # matches current setup
                clip_sample=False,
            )

        self.noise_scheduler = move_scheduler_to_device(self.noise_scheduler, self.device)

        # Recompute alphas_cumprod with tighter floor
        betas = self.noise_scheduler.betas.clone()
        betas[0] = 1e-6
        alphas = 1.0 - betas
        self.noise_scheduler.betas = betas
        self.noise_scheduler.alphas = alphas
        self.noise_scheduler.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.noise_scheduler.alphas_cumprod_prev = F.pad(
            self.noise_scheduler.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.noise_scheduler.alphas_cumprod_prev = self.noise_scheduler.alphas_cumprod_prev.to(self.device)

        # self.model = ConditionalTemporalNet(traj_dim=self.horizon, start_dim=self.start_dim, goal_dim=self.goal_dim,k_max=self.k_max).to(device)
        # self.diffusion = GaussianDiffusionSimple(device=device)

        # Load checkpoint
        ckpt_path = config.get('diffusion_conditional_unet1dcfg_feasible', 'ckpt_path')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.token_embedder.load_state_dict(ckpt["token_embedder_state_dict"])
        self.model.eval()

    def get_model(self):
        return None


    # --------------------------
    # Build conditioning vector
    # --------------------------
    def build_condition_from_state(self, state):
        v0 = getattr(self, '_v0_from_last_step', 1.0)
        w0 = getattr(self, '_w0_from_last_step', 0.0)
        
        # --- start state ---
        start = np.array([
            v0,
            # 1.0,
            w0
        ], dtype=np.float32)

        # rotation to ego frame
        theta = state.self_state.theta
        R = np.array([
            [ np.cos(theta),  np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)

        # --- goal in ego frame ---
        goal_world = np.array([state.self_state.gx, state.self_state.gy])
        pos_world  = np.array([state.self_state.px, state.self_state.py])
        goal_ego = R @ (goal_world - pos_world)

        goal = goal_ego.astype(np.float32)

        # --- obstacles in ego frame ---
        obs_list = []
        for h in state.human_states:
            pos_ego = R @ (np.array([h.px, h.py]) - pos_world)
            vel_ego = R @ np.array([h.vx, h.vy])
            obs_list.append([pos_ego[0], pos_ego[1], vel_ego[0], vel_ego[1]])

        obstacles = np.array(obs_list, dtype=np.float32)

        # --- pad / truncate obstacles ---
        obs_padded = np.zeros((self.k_max, 4), dtype=np.float32)
        obs_mask   = np.zeros((self.k_max,), dtype=np.float32)

        n = min(len(obstacles), self.k_max)
        if n > 0:
            obs_padded[:n] = obstacles[:n]
            obs_mask[:n] = 1.0

        # --- normalize ---
        start = (start - self.norm["start_mean"]) / self.norm["start_std"]
        goal  = (goal  - self.norm["goal_mean"])  / self.norm["goal_std"]

        if n > 0:
            obs_padded[:n] = (obs_padded[:n] - self.norm["obs_mean"]) / self.norm["obs_std"]

        # --- convert to torch ---
        start_t = torch.from_numpy(start).unsqueeze(0).to(self.device)
        goal_t  = torch.from_numpy(goal).unsqueeze(0).to(self.device)
        obs_t   = torch.from_numpy(obs_padded).unsqueeze(0).to(self.device)
        mask_t  = torch.from_numpy(obs_mask).unsqueeze(0).to(self.device)

        return start_t, goal_t, obs_t, mask_t



    def predict(self, state):
        if self.reach_destination(state):
            return ActionRot(0, 0)

        start, goal, obstacles, obs_mask = self.build_condition_from_state(state)

        K = self.num_samples  # number of candidate trajectories — expose via config if needed

        with torch.no_grad():
            # ── 1. Batch K noise samples ──────────────────────────────────────────
            # x shape: [1, K, traj_dim, T] → reshaped to [K, traj_dim, T] for the UNet
            x = torch.randn(K, self.traj_dim, self.horizon, device=self.device)

            # ── 2. Build conditioning tokens (same for all K candidates) ─────────
            tokens = self.token_embedder(start, goal, obstacles)      # [1, N_tok, D]
            attn_mask = build_attn_mask(obs_mask)                     # [1, N_tok]

            null_tokens      = self.token_embedder.null_token.expand(1, -1, -1)  # [1, N_tok, D]
            attn_mask_uncond = torch.ones(1, tokens.shape[1],
                                        device=self.device, dtype=attn_mask.dtype)

            # Expand to [K, ...] so one forward pass handles all K candidates
            tokens_k      = tokens.expand(K, -1, -1)           # [K, N_tok, D]
            attn_mask_k   = attn_mask.expand(K, -1)            # [K, N_tok]
            null_tokens_k = null_tokens.expand(K, -1, -1)      # [K, N_tok, D]
            attn_mask_unc = attn_mask_uncond.expand(K, -1)     # [K, N_tok]

            # Stack cond + uncond: [2K, N_tok, D]
            tokens_double = torch.cat([tokens_k, null_tokens_k], dim=0)
            mask_double   = torch.cat([attn_mask_k, attn_mask_unc], dim=0).bool()

            denoise_mode = "DDIM" if self.ddim_inference else "DDPM"
            time1 = datetime.now()

            # ── 3. Batched denoising loop ─────────────────────────────────────────
            if self.ddim_inference:
                self.noise_scheduler.set_timesteps(self.num_inference_steps,
                                                device=self.device)
                timesteps = self.noise_scheduler.timesteps
            else:
                timesteps = reversed(range(
                    self.noise_scheduler.config.num_train_timesteps))

            for t in timesteps:
                t_tensor = torch.full((K,), t, device=self.device, dtype=torch.long)

                # Single forward pass for all K cond + K uncond samples
                x_double     = torch.cat([x, x], dim=0)         # [2K, D, T]
                noise_both   = self.model(
                    sample                = x_double,
                    timestep              = t_tensor.repeat(2),  # [2K]
                    encoder_hidden_states = tokens_double,
                    encoder_attention_mask= mask_double,
                ).sample                                         # [2K, D, T]

                noise_cond, noise_uncond = noise_both[:K], noise_both[K:]
                noise_pred = noise_uncond + self.cfg_weight * (noise_cond - noise_uncond)

                if self.ddim_inference:
                    x = self.noise_scheduler.step(
                        noise_pred, t_tensor[0], x).prev_sample  # scheduler takes scalar t
                else:
                    x = self.noise_scheduler.step(
                        noise_pred, t_tensor[0], x).prev_sample

            time2 = datetime.now()
            print(f"{denoise_mode} sampling ({K} trajs):", time2 - time1)

            # x: [K, traj_dim, T]  — add batch dim for scoring helpers
            x_scored = x.unsqueeze(0)          # [1, K, traj_dim, T]
            obs_b    = obstacles               # [1, k_max, 4]
            mask_b   = obs_mask                # [1, k_max]
            goal_b   = goal                    # [1, goal_dim]

            # ── 4. Score and select best trajectory ───────────────────────────────
            rewards, best_idx = score_trajectories(
                x_scored, obs_b, mask_b, goal_b, self.norm,
                collision_coef   = self.collision_coef,
                smooth_pen_coef  = self.smooth_pen_coef,
                goal_reward_coef = self.goal_reward_coef,
                control_effort_coef = self.control_effort_coef,
                safety_radius    = self.safety_radius,
                dt               = 0.05,
            )
            best_k = best_idx[0].item()        # scalar index into K
            print(f"Rewards: {rewards[0].cpu().numpy()}  → selected traj {best_k}")

            # Reorder x so best trajectory is at index 0
            idx_order = [best_k] + [i for i in range(K) if i != best_k]
            x = x[idx_order]  # [K, traj_dim, T] with best at [0]

            main_traj = x[0]  # always index 0 now

        # ── 5. Store all world-frame trajectories for visualization ───────────────
        theta   = state.self_state.theta
        R_world = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)
        pos_world = np.array([state.self_state.px, state.self_state.py])

        traj_norm_np = x.cpu().numpy()  # [K, traj_dim, T]
        all_trajs_world = []
        for k in range(K):
            traj = traj_norm_np[k].T  # [T, traj_dim]
            traj = traj * self.norm["traj_std"] + self.norm["traj_mean"]
            traj_world = np.array([pos_world + R_world @ np.array([dx, dy])
                                    for dx, dy in traj])
            all_trajs_world.append(traj_world)

        self.predicted_traj = all_trajs_world

        # ── 6. Extract action from best trajectory ────────────────────────────────
        traj = main_traj.cpu().numpy().T  # [T, traj_dim]
        traj = traj * self.norm["traj_std"] + self.norm["traj_mean"]

        dt_hz = 0.05  # horizon fidelity
        
        dt_sim = self.time_step / dt_hz  # convert sim time step to horizon time step units
        interp = dt_sim - int(dt_sim)
        dx, dy = (traj[int(np.floor(dt_sim))] * (1 - interp)
                + traj[int(np.floor(dt_sim)) + 1] * interp)
        vx, vy = dx / self.time_step, dy / self.time_step

        v            = np.sqrt(vx**2 + vy**2)
        theta_desired = np.arctan2(dy, dx)
        omega        = theta_desired / self.time_step

        exec_step = int(self.time_step / dt_hz)

        # Forward-difference heading rate at the executed step → w0 for next call
        dx_fwd = traj[exec_step + 1, 0] - traj[exec_step, 0]
        dy_fwd = traj[exec_step + 1, 1] - traj[exec_step, 1]
        heading_fwd  = np.arctan2(dy_fwd, dx_fwd) if (abs(dx_fwd) + abs(dy_fwd)) > 1e-6 else 0.0

        dx_cur = traj[exec_step, 0] - traj[exec_step - 1, 0]  # or traj[exec_step] if positions from origin
        dy_cur = traj[exec_step, 1] - traj[exec_step - 1, 1]
        heading_cur  = np.arctan2(dy_cur, dx_cur) if (abs(dx_cur) + abs(dy_cur)) > 1e-6 else 0.0

        dheading = heading_fwd - heading_cur
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi   # wrap
        self._w0_from_last_step = dheading / dt_hz

        # Forward-difference speed at exec_step
        self._v0_from_last_step = np.sqrt(dx_fwd**2 + dy_fwd**2) / dt_hz  # m/s

        if self.enforce_lims:
            v     = np.clip(v,     -self.max_vel,  self.max_vel)
            omega = np.clip(omega, -self.max_wrot, self.max_wrot)
            if not hasattr(self, "prev_v"):
                self.prev_v, self.prev_omega = 0.0, 0.0
            v     = np.clip(v,
                            self.prev_v     - self.max_accel   * self.time_step,
                            self.prev_v     + self.max_accel   * self.time_step)
            omega = np.clip(omega,
                            self.prev_omega - self.max_w_accel * self.time_step,
                            self.prev_omega + self.max_w_accel * self.time_step)
            self.prev_v, self.prev_omega = v, omega

        v_world = R_world @ np.array([vx, vy])

        print('Trajectory (ego, unnorm):', traj)
        print('Next pos (ego, unnorm):', dx, dy)
        print('Vel, rot (desired / actual):', np.sqrt(vx**2+vy**2), theta_desired/self.time_step, '/', v, omega)

        if self.kinematics == "holonomic":
            return ActionXY(v_world[0], v_world[1])
        else:
            return ActionRot(v, omega * self.time_step)

