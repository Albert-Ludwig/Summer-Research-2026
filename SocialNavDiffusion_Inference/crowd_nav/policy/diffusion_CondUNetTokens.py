from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import torch
import numpy as np
import math
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet1DModel
import torch.nn as nn
from datetime import datetime

import sys
sys.path.append("/home/cschaibl/projects/aip-sl2smith/cschaibl/diffusion_planning")  # add the parent folder of somepackage

from diffusers_unet_1d_condition import UNet1DConditionModel



class SceneTokenEmbedder(nn.Module):
    def __init__(self, start_dim, goal_dim, obs_dim, token_dim):
        super().__init__()
        # Using the hidden_dim pattern from your MLPLayer example
        hidden_dim = token_dim

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


class DiffusionConditionalUNet1DTokens(Policy):
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
        self.name = "DiffusionConditionalUNet1DTokens"

    def configure(self, config):
        import torch
        import numpy as np

        self.start_dim = config.getint('diffusion_conditional_unet1dtokens', 'start_dim')
        self.goal_dim = config.getint('diffusion_conditional_unet1dtokens', 'goal_dim')
        self.horizon = config.getint('diffusion_conditional_unet1dtokens', 'horizon')
        self.k_max = config.getint('diffusion_conditional_unet1dtokens', 'k_max')
        self.kinematics = config.get('action_space', 'kinematics')
        self.enforce_lims = config.getboolean('action_space', 'enforce_lims')
        self.max_vel = config.getfloat('action_space', 'max_vel')
        self.max_wrot = config.getfloat('action_space', 'max_wrot')
        self.max_accel = config.getfloat('action_space', 'max_accel')
        self.max_w_accel = config.getfloat('action_space', 'max_w_accel')
        self.multiagent_training = config.getboolean('diffusion_conditional_unet1dtokens', 'multiagent_training')
        self.num_inference_steps = config.getint('diffusion_conditional_unet1dtokens', 'num_inference_steps')
        self.ddim_inference = config.getboolean('diffusion_conditional_unet1dtokens', 'ddim_inference')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print("Device:", device)

        # Load normalization stats
        norm_file = config.get('diffusion_conditional_unet1dtokens', 'norm_file')
        self.norm = np.load(norm_file, allow_pickle=True).item()

        self.traj_dim=2
        self.time_emb_dim=32
        self.cond_dim=self.start_dim + self.goal_dim + self.k_max * 5
        self.num_diffusion_timesteps = 50 #Training # denoising steps
        self.token_dim=128

        self.token_embedder = SceneTokenEmbedder(
            start_dim=self.start_dim,
            goal_dim=self.goal_dim,
            obs_dim=4,
            token_dim=self.token_dim,
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
                prediction_type="epsilon",   # matches DDPM setup
                clip_sample=False,           # keep consistent with training
            )
        else:
            self.noise_scheduler = DDPMScheduler( #Can use DDIM later for inference
                num_train_timesteps=self.num_diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",  # matches current setup
                clip_sample=False,
            )

        self.noise_scheduler = move_scheduler_to_device(self.noise_scheduler, self.device)

        # self.model = ConditionalTemporalNet(traj_dim=self.horizon, start_dim=self.start_dim, goal_dim=self.goal_dim,k_max=self.k_max).to(device)
        # self.diffusion = GaussianDiffusionSimple(device=device)

        # Load checkpoint
        ckpt_path = config.get('diffusion_conditional_unet1dtokens', 'ckpt_path')
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
        # --- start state ---
        start = np.array([
            # np.sqrt(state.self_state.vx**2 + state.self_state.vy**2)
            1.0
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

        shape = (1, self.horizon, 2)

        num_modes = 4
        all_trajs_world = []
        main_traj=[]

        for mode_idx in range(num_modes):
            with torch.no_grad():
                # start from pure noise
                x = torch.randn(1, self.traj_dim, self.horizon, device=self.device)  # (B, traj_dim, T)
                traj_time_emb = sinusoidal_pos_emb(self.horizon, self.time_emb_dim, self.device)
                traj_time_emb = traj_time_emb.T.unsqueeze(0).repeat(1, 1, 1)
                tokens = self.token_embedder(start, goal, obstacles)

                attn_mask = build_attn_mask(obs_mask)
                
                x_in = torch.cat([x], dim=1)

                denoise_mode="DDIM" if self.ddim_inference else "DDPM"

                time1 = datetime.now()

                if self.ddim_inference: #DDIM
                    self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
                    for t in self.noise_scheduler.timesteps:
                        t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                        noise_pred = self.model(sample=x_in, timestep=t_tensor, encoder_hidden_states=tokens, encoder_attention_mask=attn_mask.bool()).sample
                        x = self.noise_scheduler.step(noise_pred, t, x_in[:, :self.traj_dim, :], eta=0.0).prev_sample # deterministic DDIM
                        x_in = torch.cat([x], dim=1)


                else: #DDPM
                    for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
                        t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                        noise_pred = self.model(sample=x_in, timestep=t_tensor, encoder_hidden_states=tokens, encoder_attention_mask=attn_mask.bool()).sample
                        x = self.noise_scheduler.step(noise_pred, t_tensor, x_in[:, :self.traj_dim, :]).prev_sample
                        x_in = torch.cat([x], dim=1)
                
                time2 = datetime.now()
                print(f"{denoise_mode} Time taken for sampling:", time2 - time1)

                traj = x
                traj = traj[0].transpose(1, 0)

                if mode_idx == 0:
                    main_traj = traj

                traj = traj.cpu().numpy()
                traj = traj * self.norm["traj_std"] + self.norm["traj_mean"]

                theta = state.self_state.theta
                R_world = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ], dtype=np.float32)

                pos_world = np.array([state.self_state.px, state.self_state.py])

                traj_world = []
                for dx, dy in traj:
                    traj_world.append(pos_world + R_world @ np.array([dx, dy]))

                all_trajs_world.append(np.array(traj_world))



        # with torch.no_grad():
        #     # traj = self.diffusion.p_sample_loop(
        #     #     self.model,
        #     #     shape=shape,
        #     #     start=start,
        #     #     goal=goal,
        #     #     obstacles=obstacles,
        #     #     obs_mask=obs_mask,
        #     #     device=self.device
        #     # )[0]
        #     traj = self.diffusion.ddim_sample(
        #         self.model,
        #         shape=shape,
        #         start=start,
        #         goal=goal,
        #         obstacles=obstacles,
        #         obs_mask=obs_mask,
        #         device=self.device
        #     )[0]
        
        # unnormalize trajectory
        traj = main_traj.cpu().numpy()
        traj = traj * self.norm["traj_std"] + self.norm["traj_mean"]

        # first step → velocity
        # dx, dy = traj[0]
        # dt = self.time_step #Note this will need to be fixed since our first traj is 0.05 s but the time step is 0.25 s (need to interpolate between 2nd and 3rd points)
        # vx, vy = dx / dt, dy / dt

        #Using real time step, given trained at 0.05 s samples
        dt = self.time_step*20
        interp = dt - int(dt)
        dx, dy = traj[int(np.floor(dt))] * (1 - interp) + traj[int(np.floor(dt))+1] * interp
        vx, vy = dx / self.time_step, dy / self.time_step #Ensure we are starting at 0,0 for this setup. If note, need to -1 indices

        #Use unicycle model now
        v = np.sqrt(vx**2 + vy**2)
        theta_desired = np.arctan2(dy, dx)
        omega = theta_desired / self.time_step

        v_des=v
        omega_des=omega

        if self.enforce_lims:
            # velocity limits
            v = np.clip(v, -self.max_vel, self.max_vel)
            omega = np.clip(omega, -self.max_wrot, self.max_wrot)

            # acceleration limits
            if not hasattr(self, "prev_v"):
                self.prev_v = 0.0
                self.prev_omega = 0.0

            v = np.clip(v,
                        self.prev_v - self.max_accel*self.time_step,
                        self.prev_v + self.max_accel*self.time_step)

            omega = np.clip(omega,
                            self.prev_omega - self.max_w_accel*self.time_step,
                            self.prev_omega + self.max_w_accel*self.time_step)

            self.prev_v = v
            self.prev_omega = omega


        theta = state.self_state.theta
        R_world = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)

        v_world = R_world @ np.array([vx, vy])
        print('State (Ego, Norm):',start,goal,obstacles,obs_mask)
        print('Trajectory (Ego, Unnorm):', traj)
        print('Next Pos (Ego, Unnorm):',dx,dy)
        print("Desired Vel, Rot:",v_des,omega_des)
        print("Actual Vel, Rot:",v,omega)
        # print('Next Vel (Ego, Unnorm):',vx,vy)

        pos_world = np.array([state.self_state.px, state.self_state.py])

        traj_world = []
        for dx, dy in traj:
            d_world = R_world @ np.array([dx, dy])
            traj_world.append(pos_world + d_world)

        self.predicted_traj = all_trajs_world

        if self.kinematics == "holonomic":
            return ActionXY(v_world[0], v_world[1])
        else:
            return ActionRot(v, omega*self.time_step)

