import torch
import torch.nn.functional as F
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


class SingleagentProxy(Policy):
    """
    CADRL-style policy that samples K action sequences of length H,
    rolls them out, evaluates a multi-term cost over the full trajectory,
    and executes the first action of the best sequence.

    Observation at each step: list of ObservableState (other agents).
    No learned model — pure model-predictive trajectory optimization.
    """

    def __init__(self):
        super().__init__()
        self.name = 'SingleagentProxy'
        self.trainable = False
        self.multiagent_training = False
        self.kinematics = None

        # ── Trajectory sampling ───────────────────────────────────────────────
        self.horizon        = 6       # H: steps to roll out
        self.speed_samples  = 10        # speeds to sample
        self.rotation_samples = 24     # directions to sample
        self.action_space   = None

        # ── Kinematic limits ──────────────────────────────────────────────────
        self.max_vel        = 2.0
        self.max_wrot       = np.pi
        self.max_accel      = 1.5
        self.max_w_accel    = np.pi
        self.max_omega      = np.pi

        # ── Cost coefficients ─────────────────────────────────────────────────
        self.goal_reward_coef  = 10.0

        # Lagrangian multipliers (fixed; no auto-tuning in greedy MPC)
        self.collision_coef   = 500.0

        # Safety
        self.safety_radius      = 1.2

        # Normalisation running stats (updated each call)
        self._norm_stats = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────────

    def configure(self, config):
        self.kinematics        = config.get('action_space', 'kinematics')
        self.max_vel           = config.getfloat('action_space', 'max_vel')
        self.max_wrot          = config.getfloat('action_space', 'max_wrot')
        self.max_accel         = config.getfloat('action_space', 'max_accel')
        self.max_w_accel       = config.getfloat('action_space', 'max_w_accel')
        self.speed_samples     = config.getint('action_space', 'speed_samples')
        self.rotation_samples  = config.getint('action_space', 'rotation_samples')
        self.horizon           = config.getint('single_agent_proxy', 'horizon', fallback=6)

        self.goal_reward_coef  = config.getfloat('single_agent_proxy', 'goal_reward_coef',  fallback=10.0)
        self.collision_coef  = config.getfloat('single_agent_proxy', 'collision_coef',  fallback=500.0)
        self.safety_radius     = config.getfloat('single_agent_proxy', 'safety_radius',     fallback=1.2)

        logging.info('Policy: SingleagentProxy (MPC, horizon=%d, K=%d)',
                     self.horizon,
                     1 + self.speed_samples * self.rotation_samples)

    def set_device(self, device):
        self.device = device

    # ─────────────────────────────────────────────────────────────────────────
    # Action space  (constant-action sequences)
    # ─────────────────────────────────────────────────────────────────────────

    def build_action_space(self, v_pref):
        """
        Each 'action' in the space is a primitive repeated for H steps —
        a constant-velocity straight line or constant-curvature arc.
        We store the primitives; trajectories are rolled out in predict().
        """
        holonomic = (self.kinematics == 'holonomic')
        speeds = [
            (np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.max_vel
            for i in range(self.speed_samples)
        ]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(
                -self.max_wrot * self.time_step,
                 self.max_wrot * self.time_step,
                self.rotation_samples
            )

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation),
                                             speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.action_space = action_space

    # ─────────────────────────────────────────────────────────────────────────
    # State propagation
    # ─────────────────────────────────────────────────────────────────────────

    def _propagate_ego(self, state: FullState, action) -> FullState:
        """One kinematic step for the ego agent."""
        if self.kinematics == 'holonomic':
            return FullState(
                state.px + action.vx * self.time_step,
                state.py + action.vy * self.time_step,
                action.vx, action.vy,
                state.radius, state.gx, state.gy, state.v_pref,
                np.arctan2(action.vy, action.vx)
            )
        else:
            next_theta = state.theta + action.r
            vx = action.v * np.cos(next_theta)
            vy = action.v * np.sin(next_theta)
            return FullState(
                state.px + vx * self.time_step,
                state.py + vy * self.time_step,
                vx, vy,
                state.radius, state.gx, state.gy, state.v_pref, next_theta
            )

    def _rollout_ego(self, init_state: FullState, action) -> np.ndarray:
        """
        Roll out constant action for H steps.
        Returns array of shape (2, H) — world-frame (x, y) at each step.
        """
        traj = []
        s = init_state
        for _ in range(self.horizon):
            s = self._propagate_ego(s, action)
            traj.append([s.px, s.py])
        return np.array(traj).T  # (2, H)

    def _rollout_ego_fine(self, init_state: FullState, action,
                        fine_dt: float = 0.05) -> np.ndarray:
        """
        Roll out at fine_dt for collision checking, keeping env time_step
        unchanged. Total duration = self.horizon * self.time_step.
        Returns (2, T_fine) world-frame positions.
        """
        total_duration = self.horizon * self.time_step
        n_steps = int(round(total_duration / fine_dt))

        # Build a scaled action for fine_dt
        if self.kinematics == 'holonomic':
            fine_action = ActionXY(action.vx, action.vy)  # velocity unchanged
        else:
            # angular rate is action.r / time_step rad/step → scale to fine_dt
            omega = action.r / self.time_step              # rad/s
            fine_action = ActionRot(action.v, omega * fine_dt)

        traj = []
        s = init_state
        # Temporarily swap time_step for propagation
        orig_ts = self.time_step
        self.time_step = fine_dt
        for _ in range(n_steps):
            s = self._propagate_ego(s, fine_action)
            traj.append([s.px, s.py])
        self.time_step = orig_ts
        return np.array(traj).T  # (2, T_fine)

    def _rollout_obstacles(self, ob: list) -> np.ndarray:
        """
        Constant-velocity prediction for each observable agent.
        ob: list of ObservableState  (N agents)
        Returns (N, 4) — (x, y, vx, vy) at t=0  (propagation done in cost fn).
        """
        if not ob:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(
            [[o.px, o.py, o.vx, o.vy] for o in ob],
            dtype=np.float32
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Cost helpers (tensor wrappers around your existing functions)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Normalise a [K] tensor to [0, 1] range across the K candidates."""
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + eps)

    def _goal_progress(self, ego_trajs: torch.Tensor,
                       goal: torch.Tensor) -> torch.Tensor:
        """
        ego_trajs: (K, 2, H)
        goal:      (2,)
        Returns:   (K,)  — normalised progress toward goal
        """
        goal_exp = goal.view(1, 2, 1)                    # (1, 2, 1)
        dist     = torch.norm(ego_trajs - goal_exp, dim=1)  # (K, H)
        start_d  = dist[:, 0]
        final_d  = dist[:, -1]
        progress = start_d - final_d                      # positive = closer
        step_progress = (dist[:, :-1] - dist[:, 1:]).mean(dim=1)
        return progress + 0.5 * step_progress             # (K,)

    def _collision_cost(self, ego_trajs: torch.Tensor,
                    obs_state: torch.Tensor,
                    obs_mask:  torch.Tensor,
                    fine_dt:   float = None) -> torch.Tensor:
        N = obs_state.shape[0]
        if N == 0:
            return torch.zeros(ego_trajs.shape[0], device=self.device)

        dt = fine_dt if fine_dt is not None else self.time_step
        H  = ego_trajs.shape[2]
        t  = torch.arange(H, device=self.device, dtype=torch.float32) * dt

        obs_pos = obs_state[:, :2].unsqueeze(-1)
        obs_vel = obs_state[:, 2:4].unsqueeze(-1)
        obs_fut = obs_pos + obs_vel * t.view(1, 1, H)   # (N, 2, H_fine)

        ego_exp = ego_trajs.unsqueeze(1)                # (K, 1, 2, H_fine)
        obs_exp = obs_fut.unsqueeze(0)                  # (1, N, 2, H_fine)
        dist    = torch.norm(ego_exp - obs_exp, dim=2)  # (K, N, H_fine)

        pen = F.relu(self.safety_radius - dist) ** 4
        pen = pen * obs_mask.view(1, N, 1)
        return pen.sum(dim=[1, 2])


    # ─────────────────────────────────────────────────────────────────────────
    # Main predict
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, state):
        """
        state.self_state:    FullState
        state.human_states:  list[ObservableState]

        Returns the first action of the best constant-action trajectory.
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase and device must be set.')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        FINE_DT = 0.05  # internal collision-check resolution — env time_step unchanged

        self.robot_v_pref = state.self_state.v_pref
        ss  = state.self_state
        ob  = state.human_states   # list[ObservableState]
        dev = self.device

        # ── 1. Roll out all K trajectories ────────────────────────────────────
        # ego_trajs: (K, 2, H) in world frame
        K = len(self.action_space)
        H = self.horizon
        ego_np = np.stack(
            [self._rollout_ego_fine(ss, a, fine_dt=FINE_DT) for a in self.action_space],
            axis=0
        )                                                     # (K, 2, T_fine)
        ego_trajs = torch.tensor(ego_np, dtype=torch.float32, device=dev)

        # ── 2. Obstacle state tensor ──────────────────────────────────────────
        obs_np   = self._rollout_obstacles(ob)
        obs_t    = torch.tensor(obs_np, dtype=torch.float32, device=dev)
        obs_mask = torch.ones(obs_np.shape[0], dtype=torch.float32, device=dev)

        goal    = torch.tensor([ss.gx, ss.gy], dtype=torch.float32, device=dev)
        ego_pos = torch.tensor([ss.px, ss.py], dtype=torch.float32, device=dev)

        near_goal = torch.norm(ego_pos - goal).item() < 0.5

        # ── 3. Compute raw costs ──────────────────────────────────────────────
        raw = {
            'goal':      self._goal_progress(ego_trajs, goal),
            'collision': self._collision_cost(ego_trajs, obs_t, obs_mask,
                                            fine_dt=FINE_DT) if not near_goal
                        else torch.zeros(K, device=dev),
        }

        # ── 4. Normalise across K candidates ─────────────────────────────────
        normed = {k: self._norm(v) for k, v in raw.items()}

        # ── 5. Composite score ────────────────────────────────────────────────
        scores = (
              self.goal_reward_coef   *  raw['goal']
            - self.collision_coef   *  raw['collision']
        )                                                         # (K,)

        # ── 6. Pick best and return its first action ──────────────────────────
        best_idx  = int(torch.argmax(scores).item())
        best_action = self.action_space[best_idx]

        print(raw['goal'][best_idx],raw['collision'][best_idx],flush=True)

        if self.phase == 'train':
            self.last_state = self.transform(state)

        # ── After computing scores, before returning ──────────────────────────────
        # Save for renderer: list of (endpoint_x, endpoint_y, score) per candidate
        # Use coarse traj (H steps) for display, not fine_dt version
        ego_coarse = np.stack(
            [self._rollout_ego(ss, a) for a in self.action_space],
            axis=0
        )  # (K, 2, H)

        scores_np = scores.cpu().numpy()
        # Downsample to 5 display candidates: best + 4 spread across score range
        sorted_idx = np.argsort(scores_np)[::-1]  # best first
        display_indices = [
            sorted_idx[0],                          # best
            sorted_idx[len(sorted_idx) // 4],       # 75th percentile
            sorted_idx[len(sorted_idx) // 2],       # median
            sorted_idx[3 * len(sorted_idx) // 4],   # 25th percentile
            sorted_idx[-1],                         # worst
        ]
        self.render_candidates = [
            {
                'traj': ego_coarse[idx],      # (2, H) absolute positions
                'score': float(scores_np[idx]),
                'is_best': (idx == best_idx),
            }
            for idx in display_indices
        ]

        return best_action

    def transform(self, state):
        """Minimal transform for API compatibility — not used in inference."""
        return state