from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
import torch
import numpy as np

class DiffusionWF(Policy):
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
        self.name = "DiffusionWF"

    def configure(self, config):
        import torch
        from crowd_nav.policy.run_socialrobot import ConditionalTemporalNet, GaussianDiffusionSimple
        import numpy as np

        self.start_dim = config.getint('diffusion_wf', 'start_dim')
        self.goal_dim = config.getint('diffusion_wf', 'goal_dim')
        self.horizon = config.getint('diffusion_wf', 'horizon')
        self.k_max = config.getint('diffusion_wf', 'k_max')
        self.kinematics = config.get('action_space', 'kinematics')
        self.multiagent_training = config.getboolean('diffusion_wf', 'multiagent_training')

        self.cond_dim = (
            self.start_dim +
            self.goal_dim +
            self.k_max * 4 +
            self.k_max
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load normalization stats
        norm_file = config.get('diffusion_wf', 'norm_file')
        self.norm = np.load(norm_file, allow_pickle=True).item()
        
        # Construct model + diffusion
        self.model = ConditionalTemporalNet(
            traj_dim=2,
            cond_dim=self.cond_dim
        ).to(self.device)
        self.diffusion = GaussianDiffusionSimple(device=device)

        # Load checkpoint
        ckpt_path = config.get('diffusion_wf', 'ckpt_path')
        ckpt = torch.load(ckpt_path, map_location=device)
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
            state.self_state.px,
            state.self_state.py,
            state.self_state.theta,
            np.sqrt(state.self_state.vx**2 + state.self_state.vy**2)
        ], dtype=np.float32)

        # rotation to ego frame
        # theta = state.self_state.theta
        # R = np.array([
        #     [ np.cos(theta),  np.sin(theta)],
        #     [-np.sin(theta),  np.cos(theta)]
        # ], dtype=np.float32)

        # --- goal in ego frame ---
        goal_world = np.array([state.self_state.gx, state.self_state.gy])
        # pos_world  = np.array([state.self_state.px, state.self_state.py])
        # goal_ego = R @ (goal_world - pos_world)

        goal = goal_world.astype(np.float32)

        # --- obstacles in ego frame ---
        obs_list = []
        for h in state.human_states:
            pos_ego = np.array([h.px, h.py])
            vel_ego = np.array([h.vx, h.vy])
            obs_list.append([pos_ego[0], pos_ego[1], vel_ego[0], vel_ego[1]])

        obstacles = np.array(obs_list, dtype=np.float32)

        # --- pad / truncate obstacles ---
        obs_padded = np.zeros((self.k_max, 4), dtype=np.float32)
        obs_mask   = np.zeros((self.k_max,), dtype=np.float32)

        n = min(len(obstacles), self.k_max)
        if n > 0:
            obs_padded[:n] = obstacles[:n]
            obs_mask[:n] = 1.0

        print('State:',start,goal,obstacles,obs_mask)
        # --- normalize ---
        start = (start - self.norm["start_mean"]) / self.norm["start_std"]
        goal  = (goal  - self.norm["goal_mean"])  / self.norm["goal_std"]

        if n > 0:
            obs_padded[:n] = (obs_padded[:n] - self.norm["obs_mean"]) / self.norm["obs_std"]

        # ---- flatten & concat ----
        cond = np.concatenate([
            start,
            goal,
            obs_padded.reshape(-1),
            obs_mask
        ], axis=0)


        return torch.from_numpy(cond).unsqueeze(0).to(self.device)



    def predict(self, state):
        if self.reach_destination(state):
            return ActionXY(0, 0)

        cond = self.build_condition_from_state(state)

        shape = (1, self.horizon, 2)

        with torch.no_grad():
            traj = self.diffusion.p_sample_loop(
                self.model,
                shape,
                cond,
                self.device
            )[0]

        # unnormalize trajectory
        traj = traj.cpu().numpy()
        traj = traj * self.norm["traj_std"] + self.norm["traj_mean"]
        print('Trajectory:', traj)
        # first step → velocity
        # dx, dy = traj[0]
        # dt = self.time_step #Note this will need to be fixed since our first traj is 0.1 s but the time step is 0.25 s (need to interpolate between 2nd and 3rd points)
        # vx, vy = dx / dt, dy / dt

        #Using real time step, given trained at 0.1 s samples
        dt = self.time_step*10
        interp = dt - int(dt)
        dx, dy = traj[int(np.floor(dt))-1] * (1 - interp) + traj[int(np.floor(dt))] * interp
        vx, vy = (dx-state.self_state.px) / self.time_step, (dy-state.self_state.py) / self.time_step

        theta = state.self_state.theta
        # R_world = np.array([
        #     [np.cos(theta), -np.sin(theta)],
        #     [np.sin(theta),  np.cos(theta)]
        # ], dtype=np.float32)

        # v_world = R_world @ np.array([vx, vy])
        # print('State (Ego, Norm):',start,goal,obstacles,obs_mask)
        # print('Trajectory (Ego, Unnorm):', traj)
        # print('Next Pos (Ego, Unnorm):',dx,dy)
        # print('Next Vel (Ego, Unnorm):',vx,vy)

        # pos_world = np.array([state.self_state.px, state.self_state.py])

        # traj_world = []
        # for dx, dy in traj:
        #     d_world = R_world @ np.array([dx, dy])
        #     traj_world.append(pos_world + d_world)

        # self.predicted_traj = np.array(traj_world)
        self.predicted_traj = np.array(traj)

        # return ActionXY(v_world[0], v_world[1])
        return ActionXY(vx, vy)

