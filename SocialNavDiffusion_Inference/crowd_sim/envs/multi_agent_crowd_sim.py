import logging
import os
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as patches
from numpy.linalg import norm
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout, Danger, Nothing
from crowd_sim.envs.utils.utils import point_to_segment_dist
import math


class MultiAgentCrowdSim(CrowdSim):
    """
    All N agents run your policy (CADRL / custom).
    Each agent has its own goal, metric tracker, and trajectory log.
    Agents treat each other as observable obstacles.

    Single-agent behaviour is fully preserved — if num_agents == 1 the
    environment is functionally identical to CrowdSim with no humans.
    Switch between modes by setting env.num_agents before calling reset().
    """

    def __init__(self):
        super().__init__()
        self.num_agents = 1             # default: behaves like single-agent
        self.agents = []                # Robot instances, all running your policy
        self.agent_metrics = []         # per-agent metric dicts
        self.agent_done = []            # bool: has this agent finished?
        self.agent_policy_factory = None  # callable: () -> policy instance

    # ─────────────────────────────────────────────────────────────────────────
    # Configure — identical to parent, just passes through
    # ─────────────────────────────────────────────────────────────────────────

    def set_agent_policy_factory(self, factory):
        """
        Call once before reset().
        factory must be a zero-argument callable that returns a configured,
        model-loaded policy instance ready for inference.

        Example:
            def make_policy():
                p = policy_factory['cadrl']()
                p.configure(policy_config)
                p.get_model().load_state_dict(torch.load(weights))
                p.set_phase('test')
                p.set_device(device)
                return p
            env.set_agent_policy_factory(make_policy)
        """
        self.agent_policy_factory = factory

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """
        Extends parent reset().  For single-agent (num_agents == 1) the
        call is forwarded to the parent unchanged so nothing breaks.
        """
        # ── Shared bookkeeping ────────────────────────────────────────────────
        self.predicted_trajs = []
        self.minobsdist      = float('inf')
        self.pathlength      = 0
        self.avgobsdist      = []
        self.global_time     = 0
        self.states          = []
        # In reset(), initialise the per-frame store:
        self._render_candidates_history = []  # plain empty list

        # Reset unicycle dynamics state for the primary robot (single-agent path).
        # In multi-agent mode each agent's policy is reset individually after
        # placement in _place_agents() → _reset_agent_unicycle_state(), so this
        # block only matters for the single-agent delegation path.
        if hasattr(self.robot, 'policy'):
            if hasattr(self.robot.policy, '_reset_unicycle_state'):
                self.robot.policy._reset_unicycle_state(0.0, 0.0)
            else:
                if hasattr(self.robot.policy, 'prev_v'):
                    self.robot.policy.prev_v = 0.0
                if hasattr(self.robot.policy, 'prev_omega'):
                    self.robot.policy.prev_omega = 0.0

        phase     = kwargs.get('phase', 'test')
        test_case = kwargs.get('test_case', None)
        assert phase in ['train', 'val', 'test']

        # ── Single-agent: delegate entirely to parent ─────────────────────────
        if self.num_agents == 1:
            return super().reset(**kwargs)

        # ── Multi-agent ───────────────────────────────────────────────────────
        if self.agent_policy_factory is None:
            raise RuntimeError(
                'Call env.set_agent_policy_factory(fn) before reset() '
                'in multi-agent mode.'
            )

        # Build agents
        self.agents = []
        for _ in range(self.num_agents):
            agent = Robot(self.config, 'robot')
            agent.policy = self.agent_policy_factory()
            agent.policy.time_step = self.time_step
            if hasattr(agent.policy, 'set_env'):
                agent.policy.set_env(self)
            self.agents.append(agent)

        # Seed / counter logic (mirrors parent)
        counter_offset = {
            'train': self.case_capacity['val'] + self.case_capacity['test'],
            'val':   0,
            'test':  self.case_capacity['val'],
        }
        if test_case is not None:
            self.case_counter[phase] = test_case

        seed = counter_offset[phase] + self.case_counter[phase] + self.testoffset
        rng  = np.random.RandomState(seed)
        self.testoffset += 1

        # Place agents
        if phase == 'test' and self.test_sim == 'npz_hard':
            rule = 'npz_hard'
        elif phase == 'test':
            rule = self.test_sim
        else:
            rule = self.train_val_sim
        self._place_agents(rule, rng)
        if rule != 'npz_hard':
            self.case_counter[phase] = (
                (self.case_counter[phase] + 1) % self.case_size[phase]
            )

        # Agent state
        self.agent_done    = [False] * self.num_agents
        self.agent_metrics = []
        for i in range(self.num_agents):
            a = self.agents[i]
            self.agent_metrics.append({
                # ── existing ──────────────────────────────────────────────────
                'success':              False,
                'collision':            False,
                'timeout':              False,
                'time_to_goal':         None,
                'path_length':          0.0,
                'min_obs_dist':         float('inf'),
                'avg_obs_dist':         [],
                'trajectory':           [(a.px, a.py)],
                'goal':                 (a.gx, a.gy),
                # ── velocity / smoothness ──────────────────────────────────────
                'velocities':           [],        # (vx,vy) each step
                'speeds':               [],        # scalar speed each step
                'accelerations':        [],        # |Δv| / dt each step
                'jerks':                [],        # |Δa| / dt each step
                'angular_velocities':   [],        # |Δtheta| / dt each step
                # ── path quality ───────────────────────────────────────────────
                'path_irregularity':    0.0,       # cumulative heading deviation from straight line
                'straight_line_dist':   0.0,       # Euclidean start→goal
                # ── social / pairwise ──────────────────────────────────────────
                'yielding_events':      0,         # times agent decelerated toward another
                'cut_in_events':        0,         # times agent crossed another's projected path
                'near_miss_events':     0,         # times dist < 1.5 * discomfort_dist
                'time_in_danger':       0,         # steps inside discomfort_dist
                'space_violations':     0,         # steps inside personal space (0.5m)
                'side_pass_right':      0,         # right-side passes with another agent
                'side_pass_left':       0,         # left-side passes with another agent
                'head_on_events':       0,         # approaching head-on (angle > 150 deg)
                'disturbance_caused':   [],        # per-step sum of velocity change induced in others
                'freeze_events':        0,         # times agent nearly stopped due to others
                # ── control effort ─────────────────────────────────────────────
                'control_effort':       0.0,       # sum |a| * dt  (total impulse)
                'direction_changes':    0,         # heading reversals > 90 deg
                # ── previous values for derivative metrics ─────────────────────
                '_prev_vx':             0.0,
                '_prev_vy':             0.0,
                '_prev_ax':             0.0,
                '_prev_ay':             0.0,
                '_prev_theta':          a.theta,
                '_prev_speed':          0.0,
            })
        # store start positions for straight-line distance
        self._agent_starts = [(a.px, a.py) for a in self.agents]

        # Expose self.robot / self.humans for any parent code that expects them
        # (e.g. rendering helpers, explorer, etc.)
        self.robot  = self.agents[0]
        self.humans = self.agents[1:]

        # Per-agent time trackers (parent uses self.human_times)
        self.human_times = [0] * max(len(self.humans), 1)

        for agent in self.agents:
            agent.time_step        = self.time_step
            agent.policy.time_step = self.time_step

        return self._get_obs()

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        """One observation list per agent (all others as ObservableState)."""
        return [self._get_obs_for_agent(i) for i in range(self.num_agents)]

    def _get_obs_for_agent(self, i):
        return [
            a.get_observable_state()
            for j, a in enumerate(self.agents)
            if j != i
        ]

    def onestep_lookahead(self, action):
        if self.num_agents == 1:
            return super().onestep_lookahead(action)
        idx = getattr(self, '_current_acting_agent', 0)
        obs, rewards, done, infos = self.step(action, update=False)
        return obs[idx], rewards[idx], done, infos[idx]
        
    # ─────────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, actions=None, update=True):
        """
        Single-agent: forward to parent (actions is a single Action object).
        Multi-agent:  actions is a list[Action] or None (self-act).

        Returns
        -------
        Single-agent: ob, reward, done, info          (unchanged)
        Multi-agent:  obs, rewards, done_all, infos   (lists)
        """
        if self.num_agents == 1:
            return super().step(actions, update=update)



        # ── Lookahead detection ───────────────────────────────────────────────
        # CADRL (and any policy using onestep_lookahead) passes a single Action
        # object rather than a list.  Detect this and expand into a full list
        # with all other agents frozen (zero velocity).
        from crowd_sim.envs.utils.action import ActionXY, ActionRot
        _is_single_action = (
            actions is not None
            and not isinstance(actions, list)
            and isinstance(actions, (ActionXY, ActionRot))
        )
        if _is_single_action:
            idx    = getattr(self, '_current_acting_agent', 0)
            padded = [None] * self.num_agents
            padded[idx] = self._coerce_action(actions, self.agents[idx])
            for j, agent in enumerate(self.agents):
                if j == idx:
                    continue
                if self.agent_done[j]:
                    padded[j] = ActionXY(0.0, 0.0) if agent.kinematics == 'holonomic' \
                                else ActionRot(0.0, 0.0)
                elif agent.kinematics == 'holonomic':
                    padded[j] = ActionXY(agent.vx, agent.vy)   # constant velocity
                else:
                    padded[j] = ActionRot(np.sqrt(agent.vx**2 + agent.vy**2), 0.0)
            actions = padded
            
        # Guard: reset() must have run successfully
        if len(self.agent_done) != self.num_agents:
            raise RuntimeError(
                f'agent_done length {len(self.agent_done)} != '
                f'num_agents {self.num_agents}. Call reset() before step().'
            )

        # ── 1. Compute actions for active agents ──────────────────────────────
        if actions is None:
            actions = []
            for i, agent in enumerate(self.agents):
                if self.agent_done[i]:
                    zero = ActionXY(0.0, 0.0) if agent.kinematics == 'holonomic' \
                        else ActionRot(0.0, 0.0)
                    actions.append(zero)
                else:
                    ob  = self._get_obs_for_agent(i)
                    self._current_acting_agent = i   # ← tell lookahead who's planning
                    act = agent.act(ob)
                    act = self._coerce_action(act, agent)
                    actions.append(act)
            self._current_acting_agent = 0  # reset to safe default
        

        # ── 2. Pairwise collision detection ───────────────────────────────────
        collisions    = [False] * self.num_agents
        dmin_per_agent = [float('inf')] * self.num_agents

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                # Skip pairs where both are already done
                if self.agent_done[i] and self.agent_done[j]:
                    continue
                dist = self._agent_pair_closest_dist(i, j, actions[i], actions[j])
                if dist < 0:
                    if not self.agent_done[i]:
                        collisions[i] = True
                        dmin_per_agent[i] = 0
                    if not self.agent_done[j]:
                        collisions[j] = True
                        dmin_per_agent[j] = 0
                else:
                    dmin_per_agent[i] = min(dmin_per_agent[i], dist)
                    dmin_per_agent[j] = min(dmin_per_agent[j], dist)

        # ── 3. Per-agent reward / done / info ─────────────────────────────────
        rewards, dones, infos = [], [], []
        end_positions = []

        for i, agent in enumerate(self.agents):
            end_pos = np.array(agent.compute_position(actions[i], self.time_step))
            end_positions.append(end_pos)

            if self.agent_done[i]:
                # Already finished — pass through neutral values
                rewards.append(0)
                dones.append(True)
                # Preserve the terminal info from the step it finished
                infos.append(Nothing())
                continue

            reaching_goal = (
                norm(end_pos - np.array(agent.get_goal_position())) < agent.radius
            )

            if self.global_time >= self.time_limit - 1:
                rewards.append(0)
                dones.append(True)
                infos.append(Timeout(0))
            elif collisions[i]:
                rewards.append(self.collision_penalty)
                dones.append(True)
                infos.append(Collision(self.collision_penalty))
            elif reaching_goal:
                rewards.append(self.success_reward)
                dones.append(True)
                infos.append(ReachGoal(self.success_reward))
            elif dmin_per_agent[i] < self.discomfort_dist:
                r = (
                    (dmin_per_agent[i] - self.discomfort_dist)
                    * self.discomfort_penalty_factor
                    * self.time_step
                )
                rewards.append(r)
                dones.append(False)
                infos.append(Danger(dmin_per_agent[i]))
            else:
                rewards.append(0)
                dones.append(False)
                infos.append(Nothing())

        # ── 4. Update metrics and positions ───────────────────────────────────
        if update:
            for i, agent in enumerate(self.agents):
                if self.agent_done[i]:
                    continue
                m = self.agent_metrics[i]
                step_dist = norm(
                    end_positions[i] - np.array([agent.px, agent.py])
                )
                m['path_length']  += step_dist
                m['avg_obs_dist'].append(dmin_per_agent[i])
                m['min_obs_dist']  = min(m['min_obs_dist'], dmin_per_agent[i])
                m['trajectory'].append((end_positions[i][0], end_positions[i][1]))

                if isinstance(infos[i], ReachGoal):
                    m['success']      = True
                    m['time_to_goal'] = self.global_time + self.time_step
                    self.agent_done[i] = True
                elif isinstance(infos[i], Collision):
                    m['collision']     = True
                    self.agent_done[i] = True
                elif isinstance(infos[i], Timeout):
                    m['timeout']       = True
                    self.agent_done[i] = True

            # Move only agents still active
            for i, agent in enumerate(self.agents):
                if not self.agent_done[i] or isinstance(infos[i], ReachGoal):
                    # Step agents that just reached goal so position is at goal
                    agent.step(actions[i])

            self.global_time += self.time_step

            # Keep self.pathlength / self.minobsdist in sync with agent 0
            # so any parent-class code that reads these still works
            self.pathlength  += math.sqrt(
                (end_positions[0][0] - self.agents[0].px) ** 2
                + (end_positions[0][1] - self.agents[0].py) ** 2
            )
            self.minobsdist   = min(self.minobsdist, dmin_per_agent[0])
            self.avgobsdist.append(dmin_per_agent[0])

            self.states.append([a.get_full_state() for a in self.agents])

            # At the END of the `if update:` block, alongside self.states.append(...):
            frame_cands = []
            for agent in self.agents:
                cands = getattr(agent.policy, 'render_candidates', None)
                frame_cands.append(list(cands) if cands else [])
            self._render_candidates_history.append(frame_cands)

            # Social metrics update (uses positions AFTER stepping)
            self._update_social_metrics(actions, end_positions, dmin_per_agent, infos)

        obs      = self._get_obs()
        done_all = all(self.agent_done)
        return obs, rewards, done_all, infos

    # ─────────────────────────────────────────────────────────────────────────
    # Social metrics computation
    # ─────────────────────────────────────────────────────────────────────────

    def _update_social_metrics(self, actions, end_positions, dmin_per_agent, infos):
        """
        Called once per step (after positions are updated) to accumulate all
        social and control metrics.  Uses current agent states (post-step).
        """
        dt = self.time_step
        n  = self.num_agents

        # ── per-agent scalar metrics ──────────────────────────────────────────
        for i, agent in enumerate(self.agents):
            if self.agent_done[i] and not isinstance(infos[i], ReachGoal):
                continue  # skip agents that have already terminated

            m   = self.agent_metrics[i]
            act = actions[i]
            from crowd_sim.envs.utils.action import ActionXY, ActionRot

            # Current velocity
            if isinstance(act, ActionXY):
                vx, vy = act.vx, act.vy
            else:
                vx = act.v * np.cos(act.r + agent.theta)
                vy = act.v * np.sin(act.r + agent.theta)

            speed = np.sqrt(vx**2 + vy**2)
            m['velocities'].append((vx, vy))
            m['speeds'].append(speed)

            # Acceleration
            ax = (vx - m['_prev_vx']) / dt
            ay = (vy - m['_prev_vy']) / dt
            accel_mag = np.sqrt(ax**2 + ay**2)
            m['accelerations'].append(accel_mag)
            m['control_effort'] += accel_mag * dt

            # Jerk (rate of change of acceleration)
            jx = (ax - m['_prev_ax']) / dt
            jy = (ay - m['_prev_ay']) / dt
            m['jerks'].append(np.sqrt(jx**2 + jy**2))

            # Angular velocity (heading change rate)
            cur_theta  = np.arctan2(vy, vx) if speed > 0.05 else m['_prev_theta']
            delta_theta = cur_theta - m['_prev_theta']
            # normalise to (-pi, pi]
            delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
            omega = abs(delta_theta) / dt
            m['angular_velocities'].append(omega)

            # Direction reversal (heading change > 90 deg)
            if abs(delta_theta) > np.pi / 2:
                m['direction_changes'] += 1

            # Freeze: agent nearly stopped (< 5% of preferred speed)
            if speed < 0.05 * agent.v_pref and m['_prev_speed'] >= 0.05 * agent.v_pref:
                m['freeze_events'] += 1

            # Path irregularity: deviation of current heading from straight-line
            # direction to goal
            dx_goal = agent.gx - agent.px
            dy_goal = agent.gy - agent.py
            dist_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)
            if dist_to_goal > 0.1 and speed > 0.05:
                ideal_theta = np.arctan2(dy_goal, dx_goal)
                heading_err = abs((cur_theta - ideal_theta + np.pi) % (2 * np.pi) - np.pi)
                m['path_irregularity'] += heading_err * dt

            # Danger / personal space
            if dmin_per_agent[i] < self.discomfort_dist:
                m['time_in_danger'] += 1
            if dmin_per_agent[i] < 0.5:
                m['space_violations'] += 1

            # Near-miss
            if dmin_per_agent[i] < 1.5 * self.discomfort_dist:
                m['near_miss_events'] += 1

            # Update previous values
            m['_prev_vx']    = vx
            m['_prev_vy']    = vy
            m['_prev_ax']    = ax
            m['_prev_ay']    = ay
            m['_prev_theta'] = cur_theta
            m['_prev_speed'] = speed

        # ── pairwise social metrics ───────────────────────────────────────────
        for i in range(n):
            if self.agent_done[i] and not isinstance(infos[i], ReachGoal):
                continue
            mi  = self.agent_metrics[i]
            ai  = self.agents[i]
            act_i = actions[i]
            from crowd_sim.envs.utils.action import ActionXY, ActionRot
            if isinstance(act_i, ActionXY):
                vxi, vyi = act_i.vx, act_i.vy
            else:
                vxi = act_i.v * np.cos(act_i.r + ai.theta)
                vyi = act_i.v * np.sin(act_i.r + ai.theta)
            speed_i = np.sqrt(vxi**2 + vyi**2)

            disturbance_this_step = 0.0

            for j in range(n):
                if i == j:
                    continue
                if self.agent_done[j] and not isinstance(infos[j], ReachGoal):
                    continue
                aj    = self.agents[j]
                act_j = actions[j]
                if isinstance(act_j, ActionXY):
                    vxj, vyj = act_j.vx, act_j.vy
                else:
                    vxj = act_j.v * np.cos(act_j.r + aj.theta)
                    vyj = act_j.v * np.sin(act_j.r + aj.theta)

                # Vector from i to j
                dx = aj.px - ai.px
                dy = aj.py - ai.py
                dist_ij = np.sqrt(dx**2 + dy**2) + 1e-6

                # ── Head-on detection ─────────────────────────────────────────
                # Agents are head-on if velocity vectors point toward each other
                # and are approaching (dot product of relative position and
                # relative velocity is negative)
                rel_vx = vxi - vxj
                rel_vy = vyi - vyj
                approach_rate = (dx * rel_vx + dy * rel_vy) / dist_ij
                if speed_i > 0.05 and np.sqrt(vxj**2 + vyj**2) > 0.05:
                    cos_angle = (-(vxi * vxj + vyi * vyj) /
                                 (speed_i * np.sqrt(vxj**2 + vyj**2) + 1e-6))
                    if cos_angle > np.cos(np.radians(30)) and approach_rate < -0.1:
                        mi['head_on_events'] += 1

                # ── Yielding ─────────────────────────────────────────────────
                # Agent i yields if it decelerates when j is within 2x discomfort
                # and approaching
                if dist_ij < 2.0 * self.discomfort_dist and approach_rate < -0.05:
                    prev_speed_i = mi['_prev_speed']
                    if speed_i < prev_speed_i - 0.05:
                        mi['yielding_events'] += 1

                # ── Side-passing ──────────────────────────────────────────────
                # Determine which side of i's heading j is on
                # Positive cross product = j is to the left of i's direction
                if speed_i > 0.05 and dist_ij < 2.0 * self.discomfort_dist:
                    cross = vxi * dy - vyi * dx  # z-component of v_i × r_ij
                    if cross > 0:
                        mi['side_pass_left']  += 1
                    else:
                        mi['side_pass_right'] += 1

                # ── Cut-in detection ──────────────────────────────────────────
                # i cuts in front of j if i crosses j's projected path closely
                # Project i's position onto j's velocity ray
                speed_j = np.sqrt(vxj**2 + vyj**2)
                if speed_j > 0.05:
                    # vector from j to i
                    dxji = ai.px - aj.px
                    dyji = ai.py - aj.py
                    proj  = (dxji * vxj + dyji * vyj) / speed_j  # along j's path
                    perp  = abs(dxji * vyj - dyji * vxj) / speed_j  # lateral offset
                    # cut-in: i is just ahead of j, close laterally, and moving
                    # across j's path (high lateral relative velocity)
                    lat_rel_v = abs(vxi * vyj - vyi * vxj) / speed_j
                    if (0.0 < proj < 2.0 * self.discomfort_dist
                            and perp < self.discomfort_dist
                            and lat_rel_v > 0.2):
                        mi['cut_in_events'] += 1

                # ── Disturbance caused by i to j ──────────────────────────────
                # Approximate as how much j had to deviate from its preferred
                # velocity due to i's presence — use delta speed as proxy
                mj = self.agent_metrics[j]
                speed_j_now  = np.sqrt(vxj**2 + vyj**2)
                speed_j_pref = aj.v_pref
                if dist_ij < 3.0 * self.discomfort_dist:
                    disturbance_this_step += abs(speed_j_now - speed_j_pref)

            mi['disturbance_caused'].append(disturbance_this_step)

    # ─────────────────────────────────────────────────────────────────────────
    # Action type coercion
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_action(action, agent):
        """
        Ensure the action type matches what the agent's kinematics expects.
        Policies trained holonomic output ActionXY; unicycle agents need ActionRot.
        Policies trained unicycle output ActionRot; holonomic agents need ActionXY.
        """
        from crowd_sim.envs.utils.action import ActionXY, ActionRot

        if agent.kinematics == 'holonomic' and isinstance(action, ActionRot):
            # Convert ActionRot → ActionXY
            vx = action.v * np.cos(action.r + agent.theta)
            vy = action.v * np.sin(action.r + agent.theta)
            return ActionXY(vx, vy)

        elif agent.kinematics != 'holonomic' and isinstance(action, ActionXY):
            # Convert ActionXY → ActionRot
            speed = np.sqrt(action.vx ** 2 + action.vy ** 2)
            desired_theta = np.arctan2(action.vy, action.vx)
            # Rotation is the delta from current heading
            rotation = desired_theta - agent.theta
            # Normalise to (-pi, pi]
            rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
            return ActionRot(speed, rotation)

        return action  # already correct type

    # ─────────────────────────────────────────────────────────────────────────
    # Collision geometry helper
    # ─────────────────────────────────────────────────────────────────────────

    def _agent_pair_closest_dist(self, i, j, action_i, action_j):
        ai, aj = self.agents[i], self.agents[j]
        px = ai.px - aj.px
        py = ai.py - aj.py

        from crowd_sim.envs.utils.action import ActionXY, ActionRot

        def _vel(action, agent):
            if isinstance(action, ActionXY):
                return action.vx, action.vy
            else:  # ActionRot
                return (action.v * np.cos(action.r + agent.theta),
                        action.v * np.sin(action.r + agent.theta))

        vxi, vyi = _vel(action_i, ai)
        vxj, vyj = _vel(action_j, aj)
        vx = vxi - vxj
        vy = vyi - vyj

        ex = px + vx * self.time_step
        ey = py + vy * self.time_step
        return (
            point_to_segment_dist(px, py, ex, ey, 0, 0)
            - ai.radius - aj.radius
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Placement helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _place_agents(self, rule, rng):
        """
        Dispatcher mirroring generate_random_human_position() in CrowdSim.
        Each agent is a Robot running your policy; placement rules are identical
        to the original human placement so scenarios are directly comparable.

        Supported rules:
            circle_crossing  — evenly spaced on circle, antipodal goals
            square_crossing  — random start/goal on opposite sides of y-axis
            evaluation       — fully random start+goal inside square, seeded
            npz_hard         — agent 0 from npz robot data, rest from obstacles
            mixed            — mirrors parent mixed logic
        """
        if rule == 'circle_crossing':
            self._place_circle_crossing(rng)
        elif rule == 'square_crossing':
            self._place_square_crossing(rng)
        elif rule == 'evaluation':
            self._place_evaluation(rng)
        elif rule == 'npz_hard':
            self._place_agents_npz(rng)
        elif rule == 'mixed':
            self._place_mixed(rng)
        else:
            raise ValueError(f"Unknown placement rule: {rule!r}")

    # ── individual placement strategies ──────────────────────────────────────

    def _place_circle_crossing(self, rng):
        """Evenly space N agents on circle; antipodal goals (deterministic)."""
        n = self.num_agents
        for i, agent in enumerate(self.agents):
            angle  = 2 * np.pi * i / n
            px     = self.circle_radius * np.cos(angle)
            py     = self.circle_radius * np.sin(angle)
            gx, gy = -px, -py
            agent.set(px, py, gx, gy, 0, 0, angle + np.pi)
        for agent in self.agents:
            self._reset_agent_unicycle_state(agent, v0=0.0)

    def _place_agents_circle(self, rng=None):
        """Alias kept for backward compat."""
        self._place_circle_crossing(rng)

    def _place_square_crossing(self, rng):
        """Random start/goal on opposite sides of y-axis — mirrors generate_square_crossing_human."""
        placed = []
        for agent in self.agents:
            sign = -1 if rng.random() > 0.5 else 1
            # start
            while True:
                px = rng.random() * self.square_width * 0.5 * sign
                py = (rng.random() - 0.5) * self.square_width
                if all(norm((px - a.px, py - a.py)) >= agent.radius + a.radius + self.discomfort_dist
                       for a in placed):
                    break
            # goal
            while True:
                gx = rng.random() * self.square_width * 0.5 * -sign
                gy = (rng.random() - 0.5) * self.square_width
                if all(norm((gx - a.gx, gy - a.gy)) >= agent.radius + a.radius + self.discomfort_dist
                       for a in placed):
                    break
            theta = np.arctan2(gy - py, gx - px)
            agent.set(px, py, gx, gy, 0, 0, theta)
            placed.append(agent)
        for agent in self.agents:
            self._reset_agent_unicycle_state(agent, v0=0.0)

    def _place_evaluation(self, rng):
        """Fully random start+goal inside square, seeded — mirrors generate_evaluation_human."""
        half_w = self.square_width / 2
        placed = []
        for agent in self.agents:
            while True:
                px = rng.uniform(-half_w, half_w)
                py = rng.uniform(-half_w, half_w)
                if all(norm((px - a.px, py - a.py)) >= agent.radius + a.radius + self.discomfort_dist
                       for a in placed):
                    break
            while True:
                gx = rng.uniform(-half_w, half_w)
                gy = rng.uniform(-half_w, half_w)
                if all(norm((gx - a.gx, gy - a.gy)) >= agent.radius + a.radius + self.discomfort_dist
                       for a in placed):
                    break
            theta = np.arctan2(gy - py, gx - px)
            agent.set(px, py, gx, gy, 0, 0, theta)
            placed.append(agent)
        for agent in self.agents:
            self._reset_agent_unicycle_state(agent, v0=0.0)

    def _place_mixed(self, rng):
        """Mirrors parent mixed logic: random split of circle/square crossing."""
        for i, agent in enumerate(self.agents):
            if i < 2:
                # use circle crossing for first two
                angle  = 2 * np.pi * i / self.num_agents
                px     = self.circle_radius * np.cos(angle)
                py     = self.circle_radius * np.sin(angle)
                agent.set(px, py, -px, -py, 0, 0, angle + np.pi)
            else:
                # square crossing for remainder
                placed = self.agents[:i]
                sign   = -1 if rng.random() > 0.5 else 1
                while True:
                    px = rng.random() * self.square_width * 0.5 * sign
                    py = (rng.random() - 0.5) * self.square_width
                    if all(norm((px - a.px, py - a.py)) >= agent.radius + a.radius + self.discomfort_dist
                           for a in placed):
                        break
                while True:
                    gx = rng.random() * self.square_width * 0.5 * -sign
                    gy = (rng.random() - 0.5) * self.square_width
                    if all(norm((gx - a.gx, gy - a.gy)) >= agent.radius + a.radius + self.discomfort_dist
                           for a in placed):
                        break
                theta = np.arctan2(gy - py, gx - px)
                agent.set(px, py, gx, gy, 0, 0, theta)
        for agent in self.agents:
            self._reset_agent_unicycle_state(agent, v0=0.0)

    def _place_agents_npz(self, rng):
        """
        NPZ hard mode:
          agent 0 → npz robot start / goal
          agents 1..N → npz obstacle states (as many as available)
        """
        goal = self._npz_goal
        v0   = self._npz_v0
        self.agents[0].set(0, 0, float(goal[0]), float(goal[1]), v0, 0, 0)

        for i, obs_data in enumerate(self._npz_humans_data):
            if i + 1 >= self.num_agents:
                break
            px, py, vx, vy = (
                float(obs_data[0]), float(obs_data[1]),
                float(obs_data[2]), float(obs_data[3]),
            )
            spd   = np.sqrt(vx ** 2 + vy ** 2)
            prop  = rng.uniform(3.0, 10.0)
            if spd > 0.05:
                gx = np.clip(px + vx * prop, -self.square_width / 2, self.square_width / 2)
                gy = np.clip(py + vy * prop, -self.square_width / 2, self.square_width / 2)
            else:
                gx, gy = px, py
            theta = np.arctan2(vy, vx) if spd > 0.05 else 0.0
            self.agents[i + 1].set(px, py, gx, gy, vx, vy, theta)

        # If npz has fewer obstacles than num_agents-1, fill remainder on circle
        n_filled = min(len(self._npz_humans_data), self.num_agents - 1)
        for i in range(n_filled + 1, self.num_agents):
            angle = 2 * np.pi * i / self.num_agents
            px    = self.circle_radius * np.cos(angle)
            py    = self.circle_radius * np.sin(angle)
            self.agents[i].set(px, py, -px, -py, 0, 0, angle + np.pi)
 
        # Reset unicycle state for all agents.
        # Agent 0 may start with a non-zero speed (v0 from the npz), so pass
        # that through so its first feasible-action set is correct.
        npz_v0 = float(self._npz_v0) if hasattr(self, '_npz_v0') else 0.0
        self._reset_agent_unicycle_state(self.agents[0], v0=npz_v0, omega0=0.0)
        for agent in self.agents[1:]:
            self._reset_agent_unicycle_state(agent, v0=0.0, omega0=0.0)

    def _reset_agent_unicycle_state(self, agent, v0=0.0, omega0=0.0):
        """
        Reset the unicycle dynamics state of a single agent's policy.
        Called after every placement method so prev_v / prev_omega are correct
        from the very first step of the episode.
        Uses _reset_unicycle_state() when available (CADRL / MultiHumanRL /
        Diffusion), falls back to direct attribute writes for other policies.
        """
        p = agent.policy
        if hasattr(p, '_reset_unicycle_state'):
            p._reset_unicycle_state(v0, omega0)
        else:
            if hasattr(p, 'prev_v'):
                p.prev_v = v0
            if hasattr(p, 'prev_omega'):
                p.prev_omega = omega0

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def get_aggregate_metrics(self):
        """
        Call after an episode ends.
        Returns a flat dict of episode-level scalars plus per_agent detail.

        Metric groups
        -------------
        OUTCOME        success/collision/timeout rates
        EFFICIENCY     time to goal, path length, path efficiency ratio
        SAFETY         min/avg obstacle distance, danger time, space violations,
                       near misses, head-on events
        SMOOTHNESS     avg speed, acceleration, jerk, angular velocity,
                       path irregularity, direction changes, freeze events
        SOCIAL         yielding, side-passing, cut-ins, disturbance caused
        EFFORT         control effort (cumulative |a|·dt)
        JOINT          simultaneous success rate, avg completion spread,
                       social entropy (how evenly agents share space)
        """
        n   = self.num_agents
        ms  = self.agent_metrics   # shorthand

        def _mean(vals):
            v = [x for x in vals if x is not None]
            return float(np.mean(v)) if v else None

        def _per_agent(key):
            return [m[key] for m in ms]

        # ── OUTCOME ────────────────────────────────────────────────────────────
        results = {
            'success_rate':        sum(m['success']   for m in ms) / n,
            'collision_rate':      sum(m['collision'] for m in ms) / n,
            'timeout_rate':        sum(m['timeout']   for m in ms) / n,
        }

        # ── EFFICIENCY ─────────────────────────────────────────────────────────
        ttg = [m['time_to_goal'] for m in ms if m['time_to_goal'] is not None]
        results['avg_time_to_goal']   = float(np.mean(ttg)) if ttg else None
        results['std_time_to_goal']   = float(np.std(ttg))  if len(ttg) > 1 else None

        results['avg_path_length']    = _mean(_per_agent('path_length'))

        # Path efficiency: straight-line dist / actual path length (1.0 = perfect)
        eff_vals = []
        for i, m in enumerate(ms):
            sx, sy = self._agent_starts[i]
            gx, gy = m['goal']
            sl = np.sqrt((gx - sx)**2 + (gy - sy)**2)
            if m['path_length'] > 0.01 and sl > 0.01:
                eff_vals.append(min(sl / m['path_length'], 1.0))
        results['avg_path_efficiency'] = _mean(eff_vals)

        # ── SAFETY ─────────────────────────────────────────────────────────────
        finite_min = [m['min_obs_dist'] for m in ms if m['min_obs_dist'] < float('inf')]
        results['avg_min_obs_dist']   = _mean(finite_min)
        results['min_min_obs_dist']   = float(min(finite_min)) if finite_min else None

        all_avg_d = [np.mean(m['avg_obs_dist']) for m in ms if m['avg_obs_dist']]
        results['avg_avg_obs_dist']   = _mean(all_avg_d)

        results['avg_time_in_danger'] = _mean(_per_agent('time_in_danger'))
        results['avg_space_violations']= _mean(_per_agent('space_violations'))
        results['avg_near_misses']    = _mean(_per_agent('near_miss_events'))
        results['avg_head_on_events'] = _mean(_per_agent('head_on_events'))

        # ── SMOOTHNESS ─────────────────────────────────────────────────────────
        avg_speeds  = [np.mean(m['speeds'])             for m in ms if m['speeds']]
        avg_accels  = [np.mean(m['accelerations'])      for m in ms if m['accelerations']]
        avg_jerks   = [np.mean(m['jerks'])              for m in ms if m['jerks']]
        avg_omegas  = [np.mean(m['angular_velocities']) for m in ms if m['angular_velocities']]

        results['avg_speed']             = _mean(avg_speeds)
        results['avg_acceleration']      = _mean(avg_accels)
        results['avg_jerk']              = _mean(avg_jerks)
        results['avg_angular_velocity']  = _mean(avg_omegas)
        results['avg_path_irregularity'] = _mean(_per_agent('path_irregularity'))
        results['avg_direction_changes'] = _mean(_per_agent('direction_changes'))
        results['avg_freeze_events']     = _mean(_per_agent('freeze_events'))

        # Smoothness index: inverse of normalised jerk (higher = smoother)
        if results['avg_jerk'] and results['avg_jerk'] > 0:
            results['smoothness_index'] = 1.0 / (1.0 + results['avg_jerk'])
        else:
            results['smoothness_index'] = None

        # ── SOCIAL ─────────────────────────────────────────────────────────────
        results['avg_yielding_events']   = _mean(_per_agent('yielding_events'))
        results['avg_cut_in_events']     = _mean(_per_agent('cut_in_events'))
        results['avg_side_pass_right']   = _mean(_per_agent('side_pass_right'))
        results['avg_side_pass_left']    = _mean(_per_agent('side_pass_left'))

        # Side-pass bias: 0 = perfectly symmetric, 1 = always one side
        total_sp = (results['avg_side_pass_right'] or 0) + (results['avg_side_pass_left'] or 0)
        if total_sp > 0:
            r_frac = (results['avg_side_pass_right'] or 0) / total_sp
            results['side_pass_bias'] = abs(r_frac - 0.5) * 2  # 0=symmetric, 1=all one side
        else:
            results['side_pass_bias'] = None

        avg_dist = [np.mean(m['disturbance_caused']) for m in ms if m['disturbance_caused']]
        results['avg_disturbance_caused'] = _mean(avg_dist)

        # ── EFFORT ─────────────────────────────────────────────────────────────
        results['avg_control_effort']    = _mean(_per_agent('control_effort'))

        # ── JOINT / MULTI-AGENT ────────────────────────────────────────────────
        # Simultaneous success: fraction of episodes where ALL agents succeed
        results['joint_success_rate'] = 1.0 if all(m['success'] for m in ms) else 0.0

        # Completion spread: std of time-to-goal among successful agents
        # (low = agents finish together; high = some much faster than others)
        results['completion_spread'] = float(np.std(ttg)) if len(ttg) > 1 else None

        # Deadlock pairs: pairs that spent >20% of episode within discomfort dist
        # of each other without either finishing — indicates mutual blocking
        deadlock_pairs = 0
        threshold_steps = int(0.2 * (self.time_limit / self.time_step))
        for i in range(n):
            for j in range(i + 1, n):
                shared_danger = min(
                    ms[i]['time_in_danger'],
                    ms[j]['time_in_danger'],
                )
                if (shared_danger > threshold_steps
                        and not ms[i]['success']
                        and not ms[j]['success']):
                    deadlock_pairs += 1
        results['deadlock_pairs'] = deadlock_pairs

        # Social cohesion: mean pairwise distance averaged over episode
        # (computed from trajectories — lower variance = tighter group)
        if all(len(m['trajectory']) > 0 for m in ms) and n > 1:
            traj_len = min(len(m['trajectory']) for m in ms)
            pw_dists = []
            for t in range(traj_len):
                for i in range(n):
                    for j in range(i + 1, n):
                        xi, yi = ms[i]['trajectory'][t]
                        xj, yj = ms[j]['trajectory'][t]
                        pw_dists.append(np.sqrt((xi - xj)**2 + (yi - yj)**2))
            results['avg_pairwise_dist']     = float(np.mean(pw_dists)) if pw_dists else None
            results['std_pairwise_dist']     = float(np.std(pw_dists))  if pw_dists else None
        else:
            results['avg_pairwise_dist'] = None
            results['std_pairwise_dist'] = None

        results['per_agent'] = ms
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Render  (single-agent delegates to parent; multi-agent draws all agents)
    # ─────────────────────────────────────────────────────────────────────────

    def render(self, mode='human', **kwargs):
        # Single-agent: parent handles everything unchanged
        if self.num_agents == 1:
            return super().render(mode=mode, **kwargs)

        # ── Multi-agent video render ──────────────────────────────────────────
        if mode != 'video':
            # traj / human modes: fall back to parent (uses self.robot / self.humans)
            return super().render(mode=mode, **kwargs)

        import matplotlib.pyplot as plt
        from matplotlib import animation

        plt.rcParams['animation.ffmpeg_path'] = (
            '/cvmfs/soft.computecanada.ca/easybuild/software/2023/'
            'x86-64-v3/Compiler/gcccore/ffmpeg/7.1.1/bin/ffmpeg'
        )

        output_file = kwargs.get('output_file', None)
        base_policy = kwargs.get('basepolicy', 'policy')
        test_case   = kwargs.get('test_case', None)
        testnum     = kwargs.get('testnum', test_case)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=14)
        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)
        ax.set_xlabel('x (m)', fontsize=14)
        ax.set_ylabel('y (m)', fontsize=14)
        title = f'{base_policy} – Test {testnum} – {self.num_agents} agents'
        ax.set_title(title, fontsize=14, fontweight='bold')

        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors

        # Custom green→yellow→red colormap matching the cost coloring
        # To (reversed — green=1=best at top, red=0=worst at bottom):
        cost_cmap = mcolors.LinearSegmentedColormap.from_list(
            'cost', ['#d7191c', '#ffffbf', '#1a9641']   # red→yellow→green
        )
        sm = plt.cm.ScalarMappable(
            cmap=cost_cmap,
            norm=mcolors.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('trajectory cost  (0=best, 1=worst)', fontsize=11)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['worst', 'mid', 'best'])

        cmap          = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        x_off, y_off  = 0.11, 0.11

        # Static goal markers
        goal_markers = []
        for i in range(self.num_agents):
            gx, gy = self.agent_metrics[i]['goal']
            gm = mlines.Line2D(
                [gx], [gy], color=cmap[i], marker='*',
                linestyle='None', markersize=14,
                label=f'A{i} goal'
            )
            ax.add_artist(gm)
            goal_markers.append(gm)

        # Agent circles
        agent_circles = []
        for i in range(self.num_agents):
            pos = self.states[0][i].position
            c   = plt.Circle(pos, self.agents[i].radius,
                             fill=True, color=cmap[i], alpha=0.85)
            ax.add_artist(c)
            agent_circles.append(c)

        # Agent number labels
        agent_labels = []
        for i in range(self.num_agents):
            pos = self.states[0][i].position
            lbl = ax.text(
                pos[0] - x_off, pos[1] - y_off,
                str(i), color='white', fontsize=10, fontweight='bold'
            )
            agent_labels.append(lbl)

        # Trajectory tail lines (dashed)
        traj_lines = [
            ax.plot([], [], '--', color=cmap[i], alpha=0.45, linewidth=1)[0]
            for i in range(self.num_agents)
        ]

        # Direction arrows
        arrow_style = patches.ArrowStyle('->', head_length=4, head_width=2)
        arrows = []

        def _make_arrows(frame):
            for arr in arrows:
                arr.remove()
            arrows.clear()
            state = self.states[frame]
            for i in range(self.num_agents):
                s     = state[i]
                r     = self.agents[i].radius
                theta = np.arctan2(s.vy, s.vx) if self.agents[i].kinematics != 'unicycle' \
                        else s.theta
                tip   = (s.px + r * np.cos(theta), s.py + r * np.sin(theta))
                if not hasattr(self, '_render_candidates') or not self._render_candidates[i]:
                    arr   = patches.FancyArrowPatch(
                        (s.px, s.py), tip,
                        color='black', arrowstyle=arrow_style
                    )
                    ax.add_artist(arr)
                    arrows.append(arr)

        _make_arrows(0)

        time_text = ax.text(-1, 6.5, 'Time: 0.00 s', fontsize=13)

        # ax.legend(handles=goal_markers + agent_circles,
        #           labels=[f'Agent {i} goal' for i in range(self.num_agents)]
        #                + [f'Agent {i}' for i in range(self.num_agents)],
        #           fontsize=10, loc='upper right',
        #           ncol=2)

        def update(frame_num):
            for i in range(self.num_agents):
                s = self.states[frame_num][i]
                agent_circles[i].center = s.position
                agent_labels[i].set_position(
                    (s.position[0] - x_off, s.position[1] - y_off)
                )
                xs = [self.states[k][i].px for k in range(frame_num + 1)]
                ys = [self.states[k][i].py for k in range(frame_num + 1)]
                traj_lines[i].set_data(xs, ys)
            _make_arrows(frame_num)
            time_text.set_text(f'Time: {frame_num * self.time_step:.2f} s')

            # ── Candidate trajectory fan ──────────────────────────────────────────────
            for artist in getattr(ax, '_cand_artists', []):
                try:
                    artist.remove()
                except ValueError:
                    pass
            ax._cand_artists = []

            history = getattr(self, '_render_candidates_history', [])
            if history and frame_num < len(history):
                for i in range(self.num_agents):
                    # In update(), just read one frame ahead for candidates:
                    cands = history[frame_num + 1][i] if frame_num + 1 < len(history) and i < len(history[frame_num + 1]) else []
                    if not cands:
                        continue

                    # Origin from states — guaranteed to match when candidates were computed
                    s      = self.states[frame_num][i]
                    origin = np.array([s.px, s.py])

                    scores  = [c['score'] for c in cands]
                    s_min   = min(scores)
                    s_range = (max(scores) - s_min) + 1e-6

                    for c in cands:
                        traj = c['traj']   # (2, H) absolute positions from predict()

                        # Shift traj so it starts from current state position.
                        # ego_coarse is absolute from predict-time origin; replace that
                        # origin with the render-time state position to eliminate drift.
                        traj_origin = np.array([traj[0][0], traj[1][0]])  # first predicted point
                        # The displacement from predict-time pos to first traj point
                        # equals one kinematic step — rebase the whole traj onto s.px/py
                        delta = traj - traj_origin[:, None]   # (2, H) displacements from step 1
                        rebased_x = origin[0] + delta[0]
                        rebased_y = origin[1] + delta[1]

                        xs = np.concatenate([[origin[0]], rebased_x])
                        ys = np.concatenate([[origin[1]], rebased_y])

                        t     = (c['score'] - s_min) / s_range
                        color = cost_cmap(t)[:3]
                        lw    = 2.5 if c['is_best'] else 1.2
                        alpha = 0.95 if c['is_best'] else 0.65

                        ln, = ax.plot(xs, ys, '-', color=color,
                                    linewidth=lw, zorder=4 if c['is_best'] else 3,
                                    alpha=alpha)
                        ax._cand_artists.append(ln)

                        if not c['is_best']:
                            dot, = ax.plot(xs[-1], ys[-1], 'o', color=color,
                                        markersize=3, zorder=3, alpha=0.6)
                            ax._cand_artists.append(dot)

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.states),
            interval=self.time_step * 1000
        )
        anim.running = True

        if output_file is not None:
            writer = animation.writers['ffmpeg'](
                fps=8, metadata=dict(artist='Me'), bitrate=1800
            )
            anim.save(output_file, writer=writer)
        else:
            plt.show()

        plt.close(fig)