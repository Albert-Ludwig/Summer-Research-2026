from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import torch
import numpy as np
import math
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet1DModel

import sys
sys.path.append("/home/cschaibl/projects/aip-sl2smith/cschaibl/diffusion_planning")  # add the parent folder of somepackage

from diffusers_unet_1d_condition import UNet1DConditionModel


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


class DiffusionConditionalUNet1D(Policy):
    def __init__(self):
        super().__init__()
        self.model = None
        self.diffusion = None
        self.norm = None
        self.device = None
        self.horizon = None
        self.k_max = None
        self.start_dim = None
        self.goal_dim = None
        self.kinematics = None
        self.multiagent_training = None
        self.predicted_traj = None
        self.name = "DiffusionConditionalUNet1D"

    def configure(self, config):
        import torch
        import numpy as np

        self.start_dim = config.getint('diffusion_conditional_unet1d', 'start_dim')
        self.goal_dim = config.getint('diffusion_conditional_unet1d', 'goal_dim')
        self.horizon = config.getint('diffusion_conditional_unet1d', 'horizon')
        self.k_max = config.getint('diffusion_conditional_unet1d', 'k_max')
        self.kinematics = config.get('action_space', 'kinematics')
        self.multiagent_training = config.getboolean('diffusion_conditional_unet1d', 'multiagent_training')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print("Device:", device)

        # Load normalization stats
        norm_file = config.get('diffusion_conditional_unet1d', 'norm_file')
        self.norm = np.load(norm_file, allow_pickle=True).item()

        self.traj_dim=2
        self.time_emb_dim=32
        self.cond_dim=self.start_dim + self.goal_dim + self.k_max * 5
        self.num_diffusion_timesteps = 50
        
        # Construct model + diffusion
        self.model = UNet1DConditionModel(
            sample_size=self.horizon,          # T = 32
            in_channels=self.traj_dim + self.time_emb_dim,   # trajectory + conditioning channels + time embedding
            out_channels=self.traj_dim,             # predict noise on trajectory only
            layers_per_block=2,
            block_out_channels=(128, 256),
            cross_attention_dim=self.cond_dim,  # conditioning vector dimension
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

        self.noise_scheduler = DDPMScheduler( #Can use DDIM later for inference
            num_train_timesteps=self.num_diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",  # matches your current setup
        )
        self.noise_scheduler = move_scheduler_to_device(self.noise_scheduler, self.device)

        # self.model = ConditionalTemporalNet(traj_dim=self.horizon, start_dim=self.start_dim, goal_dim=self.goal_dim,k_max=self.k_max).to(device)
        # self.diffusion = GaussianDiffusionSimple(device=device)

        # Load checkpoint
        ckpt_path = config.get('diffusion_conditional_unet1d', 'ckpt_path')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def get_model(self):
        return None


    # --------------------------
    # Build conditioning vector
    # --------------------------
    def build_condition_from_state(self, state):
        # --- start state ---
        start = np.array([
            np.sqrt(state.self_state.vx**2 + state.self_state.vy**2)
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
            return ActionXY(0, 0)

        start, goal, obstacles, obs_mask = self.build_condition_from_state(state)

        shape = (1, self.horizon, 2)

        with torch.no_grad():
            # start from pure noise
            x = torch.randn(1, self.traj_dim, self.horizon, device=self.device)  # (B, traj_dim, T)
            cond = torch.cat([start, goal, obstacles.view(1,-1), obs_mask], dim=-1)[:, :, None].repeat(1, 1, self.horizon)
            traj_time_emb = sinusoidal_pos_emb(self.horizon, self.time_emb_dim, self.device)
            traj_time_emb = traj_time_emb.T.unsqueeze(0).repeat(1, 1, 1)
            
            x_in = torch.cat([x, traj_time_emb], dim=1)
            cond = cond.permute(0,2,1)  # (B, T, cond_dim)

            for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                noise_pred = self.model(sample=x_in, timestep=t_tensor, encoder_hidden_states=cond).sample
                x = self.noise_scheduler.step(noise_pred, t_tensor, x_in[:, :self.traj_dim, :]).prev_sample
                x_in = torch.cat([x, traj_time_emb], dim=1)

            traj = x

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
        traj = traj[0].transpose(1, 0)
        # unnormalize trajectory
        traj = traj.cpu().numpy()
        traj = traj * self.norm["traj_std"] + self.norm["traj_mean"]

        # first step → velocity
        # dx, dy = traj[0]
        # dt = self.time_step #Note this will need to be fixed since our first traj is 0.05 s but the time step is 0.25 s (need to interpolate between 2nd and 3rd points)
        # vx, vy = dx / dt, dy / dt

        #Using real time step, given trained at 0.05 s samples
        dt = self.time_step*20
        interp = dt - int(dt)
        dx, dy = traj[int(np.floor(dt))] * (1 - interp) + traj[int(np.floor(dt))+1] * interp
        vx, vy = dx / self.time_step, dy / self.time_step

        theta = state.self_state.theta
        R_world = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)

        v_world = R_world @ np.array([vx, vy])
        print('State (Ego, Norm):',start,goal,obstacles,obs_mask)
        print('Trajectory (Ego, Unnorm):', traj)
        print('Next Pos (Ego, Unnorm):',dx,dy)
        print('Next Vel (Ego, Unnorm):',vx,vy)

        pos_world = np.array([state.self_state.px, state.self_state.py])

        traj_world = []
        for dx, dy in traj:
            d_world = R_world @ np.array([dx, dy])
            traj_world.append(pos_world + d_world)

        self.predicted_traj = np.array(traj_world)

        return ActionXY(v_world[0], v_world[1])

