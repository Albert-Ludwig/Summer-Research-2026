import torch
import torch.nn.functional as F
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


class MultiagentProxy(Policy):
    """
    CADRL-style policy that samples K action sequences of length H,
    rolls them out, evaluates a multi-term cost over the full trajectory,
    and executes the first action of the best sequence.

    Observation at each step: list of ObservableState (other agents).
    No learned model — pure model-predictive trajectory optimization.
    """

    def __init__(self):
        super().__init__()
        self.name = 'MultiagentProxy'
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
        self.goal_reward_coef   = 10.0
        self.social_coef        = 1.0
        self.headon_coef        = 5.0
        self.yield_coef        = 1.0
        self.cutoff_coef       = 1.0

        # Lagrangian multipliers (fixed; no auto-tuning in greedy MPC)
        self.collision_coef   = 20.0

        # Safety
        self.safety_radius      = 1.2

        # Normalisation running stats (updated each call)
        self._norm_stats = {}

        # in __init__:
        self._freeze_counter = 0
        self._freeze_threshold = 2   # steps before anti-freeze kicks in
        self._freeze_min_speed = 0.05

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
        self.horizon           = config.getint('multiagent_proxy', 'horizon', fallback=6)

        self.goal_reward_coef  = config.getfloat('multiagent_proxy', 'goal_reward_coef',  fallback=10.0)
        self.social_coef  = config.getfloat('multiagent_proxy', 'social_coef',  fallback=1.0)
        self.yield_coef        = config.getfloat('multiagent_proxy', 'yield_coef',        fallback=1.0)
        self.headon_coef        = config.getfloat('multiagent_proxy', 'headon_coef',        fallback=5.0)
        self.cutoff_coef       = config.getfloat('multiagent_proxy', 'cutoff_coef',       fallback=1.0)
        self.collision_coef  = config.getfloat('multiagent_proxy', 'collision_coef',  fallback=20.0)
        self.safety_radius     = config.getfloat('multiagent_proxy', 'safety_radius',     fallback=1.2)

        logging.info('Policy: MultiagentProxy (MPC, horizon=%d, K=%d)',
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
        """
        ego_trajs:  (K, 2, H)
        obs_state:  (N, 4)   — (x, y, vx, vy) at t=0
        obs_mask:   (N,)     — 1 = valid
        Returns:    (K,)
        """
        N = obs_state.shape[0]
        if N == 0:
            return torch.zeros(ego_trajs.shape[0], device=self.device)

        dt = fine_dt if fine_dt is not None else self.time_step
        H  = ego_trajs.shape[2]
        t  = torch.arange(H, device=self.device, dtype=torch.float32) * dt

        # Obstacle future positions: (N, 2, H)
        obs_pos  = obs_state[:, :2].unsqueeze(-1)        # (N, 2, 1)
        obs_vel  = obs_state[:, 2:4].unsqueeze(-1)       # (N, 2, 1)
        obs_fut  = obs_pos + obs_vel * t.view(1, 1, H)   # (N, 2, H)

        # Pairwise distances: (K, N, H)
        ego_exp  = ego_trajs.unsqueeze(1)                # (K, 1, 2, H)
        obs_exp  = obs_fut.unsqueeze(0)                  # (1, N, 2, H)
        dist     = torch.norm(ego_exp - obs_exp, dim=2)  # (K, N, H)

        pen      = F.relu(self.safety_radius - dist) ** 4  # (K, N, H)
        pen      = pen * obs_mask.view(1, N, 1)
        return pen.sum(dim=[1, 2])                        # (K,)

    def _time_to_conflict(self, ego_pos: torch.Tensor,
                        ego_vel: torch.Tensor,
                        obs_pos: torch.Tensor,
                        obs_vel: torch.Tensor,
                        conflict_r: float) -> torch.Tensor:
        """
        Estimate time-to-conflict-zone for ego and one obstacle.
        Used for right-of-way: whoever arrives LATER should yield.
        
        ego_pos: (K, 2, H)  obs_pos: (2, H)
        ego_vel: (K, 2, H-1)
        Returns: (K,) — mean estimated TTC across horizon
        """
        # Conflict point: midpoint between current positions
        conflict_pt = ((ego_pos[:, :, 0] + obs_pos[:, 0].unsqueeze(0)) / 2.0)  # (K, 2)

        # Ego distance to conflict
        ego_to_conf = torch.norm(ego_pos[:, :, 0] - conflict_pt, dim=1)         # (K,)
        ego_spd     = torch.norm(ego_vel[:, :, 0], dim=1).clamp(min=1e-6)       # (K,)
        ego_ttc     = ego_to_conf / ego_spd                                      # (K,)

        # Obs distance to conflict
        obs_to_conf = torch.norm(obs_pos[:, 0] - conflict_pt[0], dim=0)         # scalar
        obs_spd     = torch.norm(obs_vel).clamp(min=1e-6)
        obs_ttc     = obs_to_conf / obs_spd                                      # scalar

        return ego_ttc, obs_ttc.item()

    def _social_cost(self, ego_trajs: torch.Tensor,
                    obs_state: torch.Tensor,
                    obs_mask: torch.Tensor,
                    fine_dt: float = 0.05) -> torch.Tensor:
        """
        Social norm cost with three components:

        1. INTRUSION — ego enters personal space of agent who arrives
        at the conflict zone FIRST (they have right-of-way by proximity,
        not speed — prevents barging).

        2. CROSSING / CUT-IN — ego cuts in front of a moving agent.
        Penalizes crossing another's path when they are committed.
        Right-hand bias: penalize left-side crossing more than right.

        3. WAKE DISRUPTION — ego sits in another agent's projected forward
        path, forcing them to divert. Proximity weighted.
        """
        N = obs_state.shape[0]
        K = ego_trajs.shape[0]
        if N == 0:
            return torch.zeros(K, device=self.device)

        H = ego_trajs.shape[2]
        t = torch.arange(H, device=self.device, dtype=torch.float32) * fine_dt

        obs_pos = obs_state[:, :2].unsqueeze(-1)
        obs_vel = obs_state[:, 2:4].unsqueeze(-1)
        obs_fut = obs_pos + obs_vel * t.view(1, 1, H)     # (N, 2, H)
        obs_spd = torch.norm(obs_state[:, 2:4], dim=1)    # (N,)

        ego_vel_vec  = ego_trajs[:, :, 1:] - ego_trajs[:, :, :-1]  # (K, 2, H-1)
        ego_mean_spd = torch.norm(ego_vel_vec, dim=1).mean(dim=1)   # (K,)

        # Pairwise distances: (K, N, H)
        ego_exp = ego_trajs.unsqueeze(1)
        obs_exp = obs_fut.unsqueeze(0)
        dist    = torch.norm(ego_exp - obs_exp, dim=2)    # (K, N, H)

        cost_intrusion = torch.zeros(K, device=self.device)
        cost_crossing  = torch.zeros(K, device=self.device)
        cost_wake      = torch.zeros(K, device=self.device)

        personal_space = self.safety_radius * 2.0

        for i in range(N):
            if obs_mask[i] < 0.5:
                continue

            dist_i = dist[:, i, :]                        # (K, H)
            spd_i  = obs_spd[i]
            vel_i  = obs_state[i, 2:4]                    # (2,)

            # ── 1. INTRUSION — proximity-based right of way ───────────────────
            in_bubble = F.relu(personal_space - dist_i)   # (K, H)

            # Who arrives at conflict zone first?
            # Ego TTC vs obs TTC — whoever arrives LATER should yield
            ego_ttc, obs_ttc = self._time_to_conflict(
                ego_trajs, ego_vel_vec,
                obs_fut[i], obs_state[i, 2:4],
                conflict_r=personal_space
            )
            # If ego arrives later (higher TTC) → ego should yield → high cost for intruding
            # If ego arrives first (lower TTC) → ego has right of way → lower cost
            # row_weight > 0 when ego_ttc > obs_ttc (obs gets there first)
            row_weight = (ego_ttc - obs_ttc).clamp(min=0.0) / (self.time_step * H + 1e-6)
            row_weight = row_weight.clamp(max=1.0)          # (K,) in [0,1]

            intrusion_pen  = in_bubble.pow(2).mean(dim=1)   # (K,)
            cost_intrusion += intrusion_pen * (1.0 + row_weight)

            # ── 2. CROSSING — right-hand bias ────────────────────────────────
            if spd_i < 0.05:
                continue

            obs_dir = F.normalize(vel_i.unsqueeze(0), dim=-1)   # (1, 2)

            rel   = ego_trajs - obs_fut[i].unsqueeze(0)         # (K, 2, H)
            rel_t = rel.permute(0, 2, 1)                        # (K, H, 2)

            # Forward: positive = ego is ahead of obs
            fwd = (rel_t * obs_dir.unsqueeze(0)).sum(dim=-1)    # (K, H)

            # Signed lateral: positive = ego is to the LEFT of obs's heading
            # (right-hand rule: left is the dangerous side to cross from)
            lat_signed = (rel_t[..., 0] * obs_dir[0, 1]
                        - rel_t[..., 1] * obs_dir[0, 0])        # (K, H) signed

            lat_abs = lat_signed.abs()

            # Crossing from the LEFT of obs's path is worse (violates right-hand norm)
            # lat_signed > 0 means ego is to obs's left
            left_side_penalty = torch.sigmoid(lat_signed / 0.3)  # (K, H) ∈ (0,1), 0.5 when centred

            # Base cut-in: ahead + close laterally
            ahead_weight = torch.sigmoid(fwd / 0.3)
            lat_pen      = F.relu(personal_space - lat_abs)
            spd_weight   = min(spd_i.item() / self.max_vel, 1.0)

            # Left-side crossings penalized 2x, right-side 1x
            side_multiplier = 1.0 + left_side_penalty            # (K, H) ∈ (1, 2)
            crossing_pen    = (ahead_weight * lat_pen * side_multiplier).mean(dim=1) * spd_weight
            cost_crossing  += crossing_pen

            # ── 3. WAKE DISRUPTION ────────────────────────────────────────────
            forward_cone = F.relu(fwd)
            narrow_lat   = F.relu(self.safety_radius * 1.5 - lat_abs)
            in_path      = forward_cone * narrow_lat
            proximity    = torch.exp(-dist_i / 2.0)
            wake_pen     = (in_path * proximity).mean(dim=1)
            cost_wake   += wake_pen

        return cost_intrusion + cost_crossing + cost_wake


    def _headon_cost(self, ego_trajs: torch.Tensor,
                    obs_state: torch.Tensor,
                    obs_mask: torch.Tensor,
                    fine_dt: float = 0.05) -> torch.Tensor:
        """
        Penalizes trajectories that approach any agent head-on — including
        approaching a stationary or laterally moving agent from the front.

        Right-hand convention:
        - Approaching head-on AND from the LEFT of obs's path → maximum cost
        - Approaching head-on AND already offset to the RIGHT → cost fades out
        - Pure right-side passing (ego to obs's right) → zero cost

        Two sub-costs:
        A. EGO-CENTRIC: ego heading directly toward another agent
            regardless of the other agent's heading (covers the case of
            charging into someone's side or front).
        B. MUTUAL HEAD-ON: both headings are opposing — full bilateral penalty.
        """
        N = obs_state.shape[0]
        K = ego_trajs.shape[0]
        if N == 0:
            return torch.zeros(K, device=self.device)

        H = ego_trajs.shape[2]
        t = torch.arange(H, device=self.device, dtype=torch.float32) * fine_dt

        obs_fut = (obs_state[:, :2].unsqueeze(-1)
                + obs_state[:, 2:4].unsqueeze(-1) * t.view(1, 1, H))  # (N, 2, H)

        ego_vel = ego_trajs[:, :, 1:] - ego_trajs[:, :, :-1]             # (K, 2, H-1)
        ego_spd = torch.norm(ego_vel, dim=1, keepdim=True).clamp(min=1e-6)
        ego_dir = ego_vel / ego_spd                                        # (K, 2, H-1)

        total = torch.zeros(K, device=self.device)

        for i in range(N):
            if obs_mask[i] < 0.5:
                continue

            obs_vel_i = obs_state[i, 2:4]
            obs_spd_i = torch.norm(obs_vel_i).clamp(min=1e-6)
            obs_dir_i = obs_vel_i / obs_spd_i                             # (2,)

            ego_pos_t  = ego_trajs[:, :, :-1]                             # (K, 2, H-1)
            obs_pos_t  = obs_fut[i, :, :-1].unsqueeze(0)                  # (1, 2, H-1)
            to_obs     = obs_pos_t - ego_pos_t                            # (K, 2, H-1)
            dist_to_obs = torch.norm(to_obs, dim=1).clamp(min=1e-6)       # (K, H-1)
            to_obs_dir  = to_obs / dist_to_obs.unsqueeze(1)               # (K, 2, H-1)

            # ── Factor 1: Ego closing toward obs ─────────────────────────────
            closing = (ego_dir * to_obs_dir).sum(dim=1).clamp(min=0.0)    # (K, H-1)

            # ── Factor 2: Signed lateral offset from obs's heading line ──────
            # Positive = ego is to the LEFT of obs's heading (bad — wrong side)
            # Negative = ego is to the RIGHT (good — correct side)
            obs_dir_exp = obs_dir_i.view(1, 2, 1).expand(K, 2, H - 1)
            # cross product z-component: to_obs_dir × obs_dir
            lat_signed = (to_obs_dir[:, 0, :] * obs_dir_exp[:, 1, :]
                        - to_obs_dir[:, 1, :] * obs_dir_exp[:, 0, :])     # (K, H-1)
            # Right-side offset (ego to obs's right) → should be rewarded / not penalized
            # Left-side offset → penalized
            # Map: left (positive) → high cost, right (negative) → low cost
            side_cost = torch.sigmoid(lat_signed / 0.4)                   # (K, H-1) ∈ (0,1)
            # When perfectly to the right (lat_signed = -inf) → side_cost → 0
            # When perfectly to the left (lat_signed = +inf) → side_cost → 1

            # ── Sub-cost A: Ego-centric head-on ──────────────────────────────
            # Penalize ego charging toward anyone, weighted by wrong-side approach
            interaction_range = 4.0
            range_gate = torch.exp(-dist_to_obs / interaction_range)

            # Soft lateral threshold: cost fades when ego is clearly offset right
            on_course = side_cost                                          # (K, H-1)

            pen_a = closing * on_course * range_gate                      # (K, H-1)

            # ── Sub-cost B: Mutual head-on ────────────────────────────────────
            # Both agents facing each other — strongest violation
            headon_align = (ego_dir * (-obs_dir_exp)).sum(dim=1)          # (K, H-1)
            # Fire only when angle between headings > 120 deg (cos < -0.5 → opposing)
            # But we defined headon as ego toward obs AND obs toward ego
            # Use threshold: cos(angle) > 0.5 after negating = headings oppose
            headon_align = F.relu(headon_align - 0.5) / 0.5               # (K, H-1) ∈ [0,1]

            pen_b = closing * headon_align * on_course * range_gate       # (K, H-1)

            # B is weighted higher — mutual head-on is worse than unilateral approach
            total += (pen_a + 2.0 * pen_b).mean(dim=1)                   # (K,)

        return total


    def _yielding_cost(self, ego_trajs: torch.Tensor,
                    obs_state: torch.Tensor,
                    obs_mask: torch.Tensor,
                    fine_dt: float = 0.05) -> torch.Tensor:
        """
        Penalizes ego for failing to yield when the obstacle arrives at the
        shared conflict zone BEFORE ego (proximity-based right of way, not speed).

        Right-hand convention:
        - If obs approaches from ego's RIGHT → obs has right of way → ego yields
        - If obs approaches from ego's LEFT  → ego has right of way → no yield penalty
        This matches standard road/pedestrian rules (right-hand traffic).
        """
        N = obs_state.shape[0]
        K = ego_trajs.shape[0]
        if N == 0:
            return torch.zeros(K, device=self.device)

        H = ego_trajs.shape[2]
        t = torch.arange(H, device=self.device, dtype=torch.float32) * fine_dt

        obs_fut = (obs_state[:, :2].unsqueeze(-1)
                + obs_state[:, 2:4].unsqueeze(-1) * t.view(1, 1, H))  # (N, 2, H)

        ego_xy  = ego_trajs.permute(0, 2, 1)                              # (K, H, 2)
        ego_vel = ego_xy[:, 1:] - ego_xy[:, :-1]                         # (K, H-1, 2)

        # Ego heading direction at each step
        ego_spd = torch.norm(ego_vel, dim=-1, keepdim=True).clamp(min=1e-6)
        ego_dir = ego_vel / ego_spd                                        # (K, H-1, 2)

        total = torch.zeros(K, device=self.device)

        conflict_radius = self.safety_radius * 1.5

        for i in range(N):
            if obs_mask[i] < 0.5:
                continue

            ag_xy  = obs_fut[i].T                                         # (H, 2)
            ag_vel_vec = obs_state[i, 2:4]                                # (2,)
            ag_spd = torch.norm(ag_vel_vec).clamp(min=1e-6).item()

            dist = torch.norm(
                ego_xy[:, :-1] - ag_xy[:-1].unsqueeze(0), dim=-1
            )                                                             # (K, H-1)

            in_zone = (dist < conflict_radius).float()                    # (K, H-1)

            # ── Right-of-way from the RIGHT rule ─────────────────────────────
            # Compute which side of ego's heading the obstacle is on.
            # Positive cross product = obs is to ego's LEFT
            # Negative cross product = obs is to ego's RIGHT → obs has right of way
            to_obs = ag_xy[:-1].unsqueeze(0) - ego_xy[:, :-1]           # (K, H-1, 2)
            # 2D cross: ego_dir × to_obs
            cross = (ego_dir[:, :, 0] * to_obs[:, :, 1]
                - ego_dir[:, :, 1] * to_obs[:, :, 0])                # (K, H-1)
            # cross < 0 → obs is to ego's RIGHT → obs has right of way → ego should yield
            obs_on_right = torch.sigmoid(-cross / 0.3)                   # (K, H-1) ∈ (0,1)
            # 1.0 when obs is clearly to the right, 0.5 when directly ahead, ~0 when left

            # ── TTC-based right of way ────────────────────────────────────────
            # obs arrives at conflict zone before ego → obs has priority
            ego_dist_to_zone = (dist - conflict_radius).clamp(min=0.0)   # (K, H-1)
            ego_ttc_step     = ego_dist_to_zone / ego_spd[:, :, 0].clamp(min=1e-6)  # (K, H-1)

            obs_dist_to_zone = (dist[0] - conflict_radius).clamp(min=0.0)  # (H-1,) approx
            obs_ttc_step     = obs_dist_to_zone / ag_spd                    # (H-1,)

            # ego_ttc > obs_ttc → obs arrives first → ego should yield
            ttc_yield_weight = F.relu(
                ego_ttc_step - obs_ttc_step.unsqueeze(0)
            ) / (H * fine_dt + 1e-6)                                     # (K, H-1)
            ttc_yield_weight = ttc_yield_weight.clamp(max=1.0)

            # Combined right-of-way weight: right-side rule OR arrival-first rule
            row_weight = torch.max(obs_on_right, ttc_yield_weight)        # (K, H-1)

            # ── Yield penalty: closing toward obs who has right of way ────────
            to_obs_n  = F.normalize(to_obs, dim=-1)
            closing   = F.relu((ego_vel * to_obs_n).sum(dim=-1))          # (K, H-1)

            # Penalty: ego closing into conflict zone where obs has priority
            pen = closing * in_zone * row_weight                          # (K, H-1)
            total += pen.mean(dim=1)

        return total

    def _cutoff_cost(self, ego_trajs: torch.Tensor,
                     obs_state: torch.Tensor,
                     obs_mask:  torch.Tensor,
                     cut_horizon: int = 6,
                     lat_thresh:  float = 0.8) -> torch.Tensor:
        """ego_trajs: (K, 2, H)  → (K,)"""
        N = obs_state.shape[0]
        if N == 0:
            return torch.zeros(ego_trajs.shape[0], device=self.device)

        dt = self.time_step
        H  = ego_trajs.shape[2]
        t  = torch.arange(H, device=self.device, dtype=torch.float32) * dt

        obs_fut = (obs_state[:, :2].unsqueeze(-1)
                   + obs_state[:, 2:4].unsqueeze(-1) * t.view(1, 1, H))  # (N, 2, H)

        ego_xy  = ego_trajs.permute(0, 2, 1)                        # (K, H, 2)
        total   = torch.zeros(ego_trajs.shape[0], device=self.device)

        for i in range(N):
            if obs_mask[i] < 0.5:
                continue
            ag_xy  = obs_fut[i].T                                    # (H, 2)
            ag_vel = ag_xy[1:] - ag_xy[:-1]                         # (H-1, 2)
            ag_spd = torch.norm(ag_vel, dim=-1)                      # (H-1,)
            ag_dir = F.normalize(ag_vel, dim=-1)                     # (H-1, 2)

            pens = []
            for t_i in range(H - cut_horizon - 1):
                a_pos = ag_xy[t_i]                                   # (2,)
                a_dir = ag_dir[t_i]                                  # (2,)
                a_spd = ag_spd[t_i]

                ego_fut = ego_xy[:, t_i+1:t_i+cut_horizon+1]        # (K, H', 2)
                rel     = ego_fut - a_pos.view(1, 1, 2)
                fwd     = (rel * a_dir.view(1, 1, 2)).sum(dim=-1)    # (K, H')
                lat     = (rel[..., 0] * a_dir[1] - rel[..., 1] * a_dir[0]).abs()

                ahead_w = torch.sigmoid(fwd / 0.3)
                lat_pen = F.relu(lat_thresh - lat)
                spd_w   = torch.sigmoid(a_spd / 0.3)

                pens.append((ahead_w * lat_pen * spd_w).max(dim=-1).values)

            if pens:
                total += torch.stack(pens, dim=1).mean(dim=1)
        return total

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
        obs_np   = self._rollout_obstacles(ob)                   # (N, 4)
        N        = obs_np.shape[0]
        obs_t    = torch.tensor(obs_np,  dtype=torch.float32, device=dev)
        obs_mask = torch.ones(N, dtype=torch.float32, device=dev)

        goal     = torch.tensor([ss.gx, ss.gy], dtype=torch.float32, device=dev)
        ego_pos  = torch.tensor([ss.px, ss.py], dtype=torch.float32, device=dev)
        v0       = float(np.sqrt(ss.vx**2 + ss.vy**2))


        near_goal = torch.norm(ego_pos - goal).item() < 1.2

        # ── 3. Compute raw costs ──────────────────────────────────────────────
        raw = {
            'goal':        self._goal_progress(ego_trajs, goal),
            'headon':    self._headon_cost(ego_trajs, obs_t, obs_mask, fine_dt=FINE_DT),
            'cutoff':      self._cutoff_cost(ego_trajs, obs_t, obs_mask),
            'social': self._social_cost(ego_trajs, obs_t, obs_mask),
            'yield_cost':  self._yielding_cost(ego_trajs, obs_t, obs_mask),
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
            - self.social_coef   *  raw['social']
            - self.yield_coef         *  raw['yield_cost']
            - self.headon_coef         *  raw['headon']
            - self.cutoff_coef        *  raw['cutoff']
        )                                                         # (K,)
        
        # ── 6. Pick best and return its first action ──────────────────────────
        best_idx  = int(torch.argmax(scores).item())
        best_action = self.action_space[best_idx]

        print(raw['goal'][best_idx],raw['collision'][best_idx],raw['social'][best_idx],
        raw['yield_cost'][best_idx],raw['cutoff'][best_idx],raw['headon'][best_idx]
        ,flush=True)

        if self.phase == 'train':
            self.last_state = self.transform(state)

        speed_chosen = (np.sqrt(best_action.vx**2 + best_action.vy**2)
                if self.kinematics == 'holonomic'
                else best_action.v)

        if speed_chosen < self._freeze_min_speed:
            self._freeze_counter += 1
        else:
            self._freeze_counter = 0

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
                'traj': ego_coarse[idx],   # (2, H) — x/y positions at each horizon step
                'score': float(scores_np[idx]),
                'is_best': (idx == best_idx),
            }
            for idx in display_indices
        ]

        if self._freeze_counter >= self._freeze_threshold:
            # Head directly toward goal at v_pref, ignoring all costs
            dx = ss.gx - ss.px
            dy = ss.gy - ss.py
            dist = np.sqrt(dx**2 + dy**2)
            if dist > ss.radius:
                spd = min(ss.v_pref, self.max_vel)
                if self.kinematics == 'holonomic':
                    best_action = ActionXY(spd * dx / dist, spd * dy / dist)
                else:
                    desired_theta = np.arctan2(dy, dx)
                    rot = (desired_theta - ss.theta + np.pi) % (2 * np.pi) - np.pi
                    best_action = ActionRot(spd, rot)
            # self._freeze_counter = 0   # reset so it doesn't fire every step

        return best_action

    def transform(self, state):
        """Minimal transform for API compatibility — not used in inference."""
        return state