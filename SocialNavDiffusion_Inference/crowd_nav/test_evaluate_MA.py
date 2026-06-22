import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
import glob
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout
# ── Multi-agent env ───────────────────────────────────────────────────────────
from crowd_sim.envs.multi_agent_crowd_sim import MultiAgentCrowdSim


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config',    type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy',        type=str, default='orca')
    parser.add_argument('--model_dir',     type=str, default=None)
    parser.add_argument('--il',            default=False, action='store_true')
    parser.add_argument('--gpu',           default=False, action='store_true')
    parser.add_argument('--visualize',     default=False, action='store_true')
    parser.add_argument('--phase',         type=str, default='test')
    parser.add_argument('--test_case',     type=int, default=None)
    parser.add_argument('--square',        default=False, action='store_true')
    parser.add_argument('--circle',        default=False, action='store_true')
    parser.add_argument('--video_file',    type=str, default=None)
    parser.add_argument('--traj',          default=False, action='store_true')
    parser.add_argument('--results_suffix', type=str, default='')
    # ── NPZ hard ─────────────────────────────────────────────────────────────
    parser.add_argument('--npz_hard',      default=False, action='store_true')
    parser.add_argument('--npz_dir',       type=str, default='random_stage_B_examples_HARDEST')
    parser.add_argument('--npz_num_eval',  type=int, default=50)
    # ── Multi-agent ──────────────────────────────────────────────────────────
    parser.add_argument('--multi_agent',   default=False, action='store_true',
                        help='Run multi-agent mode: all N agents use your policy')
    parser.add_argument('--num_agents',    type=int, default=4,
                        help='Number of agents in multi-agent mode (ignored otherwise)')
    args = parser.parse_args()

    # ── Model weights / config paths ─────────────────────────────────────────
    if args.model_dir is not None:
        env_config_file    = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        elif os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
            model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
        else:
            model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file    = args.env_config
        policy_config_file = args.env_config

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Using device: %s', device)

    # ── Policy setup ─────────────────────────────────────────────────────────
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)

    def make_policy():
        """Factory: called once per agent in multi-agent mode."""
        p = policy_factory[args.policy]()
        p.configure(policy_config)
        if p.trainable:
            if args.model_dir is None:
                raise ValueError('Trainable policy requires --model_dir')
            p.get_model().load_state_dict(torch.load(model_weights, map_location=device))
        p.set_phase(args.phase)
        p.set_device(device)
        return p

    # ── Environment ──────────────────────────────────────────────────────────
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)

    if args.multi_agent:
        # Instantiate MultiAgentCrowdSim directly (not via gym.make, which
        # would give you the base CrowdSim).
        env = MultiAgentCrowdSim()
        env.configure(env_config)
        env.num_agents = args.num_agents
        env.set_agent_policy_factory(make_policy)
    else:
        env = gym.make('CrowdSim-v0')
        env.configure(env_config)

    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    if args.npz_hard:
        env.test_sim = 'npz_hard'

    # Single agent still needs env.robot
    robot = Robot(env_config, 'robot')
    robot.set_policy(make_policy())
    env.set_robot(robot)

    # Explorer is used only for the non-visualize path (single-agent batch eval)
    explorer = Explorer(env, robot, device, gamma=0.9)

    if isinstance(robot.policy, ORCA):
        robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    robot.policy.set_env(env)
    robot.print_info()

    # ── Visualize / evaluate ──────────────────────────────────────────────────
    if args.visualize:

        # Build scenario list
        if args.npz_hard:
            all_files  = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
            hard_files = []
            for f in all_files:
                try:
                    d          = np.load(f, allow_pickle=True)
                    difficulty = str(d['difficulty'])  if 'difficulty'  in d else None
                    threat     = str(d['threat_type']) if 'threat_type' in d else None
                    is_hard    = (
                        difficulty == 'hard' or
                        threat in ['frontal', 'side', 'rear', 'mixed', 'pincer', 'group']
                    )
                    if is_hard:
                        hard_files.append(f)
                except Exception as e:
                    logging.warning('Could not load %s: %s', f, e)

            if not hard_files:
                raise ValueError(f'No hard npz files found in {args.npz_dir}')
            hard_files = hard_files[:args.npz_num_eval]
            num_tests  = len(hard_files)
            logging.info('NPZ hard: %d scenarios from %s', num_tests, args.npz_dir)
        else:
            hard_files = None
            num_tests  = 1

        # Per-episode accumulators
        episode_times  = []
        path_lengths   = []
        successes      = []
        min_dists      = []
        avg_min_dists  = []
        threat_types   = []
        rendered_files = []
        base_policy    = None

        # ── Per-agent accumulators (multi-agent only) ─────────────────────────
        if args.multi_agent:
            ma_success_rates   = []   # per-episode aggregate success rate
            ma_collision_rates = []
            ma_ttg             = []   # per-episode avg TTG (successful agents)
            ma_agg_list        = []   # full aggregate dict per episode

        for i in range(num_tests):
            logging.info('Test %d / %d', i + 1, num_tests)

            # Reset
            if args.npz_hard:
                env.load_npz_scenario(hard_files[i])
                ob = env.reset(phase='test', test_case=i)
                try:
                    d = np.load(hard_files[i], allow_pickle=True)
                    threat_types.append(str(d.get('threat_type', 'unknown')))
                except Exception:
                    threat_types.append('unknown')
            else:
                ob = env.reset(phase='test')
                threat_types.append('standard')

            done = False

            # ── Episode loop ─────────────────────────────────────────────────
            if args.multi_agent:
                while not done:
                    # actions=None → each agent self-acts via its policy
                    ob, rewards, done, infos = env.step(actions=None)

                agg   = env.get_aggregate_metrics()
                info  = infos[0]   # agent-0 outcome for backward-compat logging

                ma_agg_list.append(agg)
                ma_success_rates.append(agg['success_rate'])
                ma_collision_rates.append(agg['collision_rate'])
                if agg['avg_time_to_goal'] is not None:
                    ma_ttg.append(agg['avg_time_to_goal'])

                # Single-agent-compatible metrics derived from agent 0
                if env.agent_metrics[0]['success']:
                    episode_times.append(env.agent_metrics[0]['time_to_goal'])
                    path_lengths.append(env.agent_metrics[0]['path_length'])
                successes.append(1 if env.agent_metrics[0]['success'] else 0)
                min_dists.append(env.agent_metrics[0]['min_obs_dist'])
                avg_od = env.agent_metrics[0]['avg_obs_dist']
                avg_min_dists.append(np.mean(avg_od) if avg_od else float('inf'))

            else:
                # ── Single-agent loop (unchanged from original) ───────────────
                last_pos = np.array(robot.get_position())
                while not done:
                    action = robot.act(ob)
                    ob, _, done, info = env.step(action)
                    last_pos = np.array(robot.get_position())

                if isinstance(info, ReachGoal):
                    episode_times.append(env.global_time)
                    path_lengths.append(env.pathlength)
                successes.append(1 if isinstance(info, ReachGoal) else 0)
                min_dists.append(env.minobsdist)
                avg_min_dists.append(
                    np.mean(env.avgobsdist) if env.avgobsdist else float('inf')
                )

            # ── Render ───────────────────────────────────────────────────────
            if args.video_file:
                base, ext = os.path.splitext(args.video_file)
                base_policy = base
                out = f'{base}_{i + 1}{ext}'
                if args.traj:
                    env.render(mode='traj', output_file=out,
                               basepolicy=base, testnum=i)
                else:
                    env.render(mode='video', output_file=out,
                               basepolicy=base, testnum=i)
                    rendered_files.append(out)
            else:
                env.render(mode='human')

            logging.info(
                'Test %d: t=%.2f | %s | SR=%.0f%% (agg) | path=%.2f | minD=%.3f',
                i + 1, env.global_time,
                info,
                (agg['success_rate'] * 100 if args.multi_agent else successes[-1] * 100),
                (env.agent_metrics[0]['path_length'] if args.multi_agent else env.pathlength),
                (env.agent_metrics[0]['min_obs_dist'] if args.multi_agent else env.minobsdist),
            )

        # ── Aggregate and save ────────────────────────────────────────────────
        results_dir = f'results_{args.results_suffix}' if args.results_suffix else 'results'
        os.makedirs(results_dir, exist_ok=True)
        bp          = base_policy or args.policy
        eval_file   = os.path.join(results_dir, f'{bp}_eval.txt')

        with open(eval_file, 'w') as f:
            f.write('========== Evaluation Results ==========\n')
            mode_label = ('NPZ Hard' if args.npz_hard else 'Standard') + (
                ' [MULTI-AGENT]' if args.multi_agent else ' [SINGLE-AGENT]'
            )
            f.write(f'Mode: {mode_label}\n')
            if args.multi_agent:
                f.write(f'Agents per episode: {args.num_agents}\n')
            f.write(f'Episodes: {num_tests}\n\n')

            # Single-agent / agent-0 metrics (always written)
            f.write('--- Agent-0 / Single-Agent Metrics ---\n')
            f.write(f'Success Rate:              {100 * np.mean(successes):.2f} %\n')
            avg_t = np.mean(episode_times) if episode_times else float('nan')
            avg_p = np.mean(path_lengths)  if path_lengths  else float('nan')
            f.write(f'Avg Time to Goal:          {avg_t:.3f} s\n')
            f.write(f'Avg Path Length:           {avg_p:.3f} m\n')
            f.write(f'Avg Min Obs Dist:          {np.mean(min_dists):.3f} m\n')
            f.write(f'Avg Per-Ep Avg Min Dist:   {np.mean(avg_min_dists):.3f} m\n')

            if args.multi_agent and ma_agg_list:
                # Average the aggregate dicts across all episodes
                import collections
                agg_keys = [k for k in ma_agg_list[0] if k != 'per_agent']
                avg_agg  = {}
                for k in agg_keys:
                    vals = [ep[k] for ep in ma_agg_list if ep.get(k) is not None]
                    avg_agg[k] = float(np.mean(vals)) if vals else None

                def _fmt(v, pct=False, dec=3):
                    if v is None: return 'N/A'
                    return f'{v * 100:.1f} %' if pct else f'{v:.{dec}f}'

                f.write('\n--- Multi-Agent Aggregate Metrics ---\n')
                f.write(f'Agents per episode:              {args.num_agents}\n\n')

                f.write('  OUTCOME\n')
                f.write(f'    Joint Success Rate:          {_fmt(avg_agg.get("joint_success_rate"), pct=True)}\n')
                f.write(f'    Avg Episode Success Rate:    {_fmt(avg_agg.get("success_rate"), pct=True)}\n')
                f.write(f'    Avg Episode Collision Rate:  {_fmt(avg_agg.get("collision_rate"), pct=True)}\n')
                f.write(f'    Avg Episode Timeout Rate:    {_fmt(avg_agg.get("timeout_rate"), pct=True)}\n')
                f.write(f'    Deadlock Pairs (avg):        {_fmt(avg_agg.get("deadlock_pairs"), dec=2)}\n')

                f.write('\n  EFFICIENCY\n')
                f.write(f'    Avg Time to Goal:            {_fmt(avg_agg.get("avg_time_to_goal"))} s\n')
                f.write(f'    Std Time to Goal:            {_fmt(avg_agg.get("std_time_to_goal"))} s\n')
                f.write(f'    Completion Spread (std TTG): {_fmt(avg_agg.get("completion_spread"))} s\n')
                f.write(f'    Avg Path Length:             {_fmt(avg_agg.get("avg_path_length"))} m\n')
                f.write(f'    Avg Path Efficiency:         {_fmt(avg_agg.get("avg_path_efficiency"))}\n')

                f.write('\n  SAFETY\n')
                f.write(f'    Avg Min Obs Dist:            {_fmt(avg_agg.get("avg_min_obs_dist"))} m\n')
                f.write(f'    Min Min Obs Dist:            {_fmt(avg_agg.get("min_min_obs_dist"))} m\n')
                f.write(f'    Avg Avg Obs Dist:            {_fmt(avg_agg.get("avg_avg_obs_dist"))} m\n')
                f.write(f'    Avg Time in Danger (steps):  {_fmt(avg_agg.get("avg_time_in_danger"), dec=1)}\n')
                f.write(f'    Avg Space Violations:        {_fmt(avg_agg.get("avg_space_violations"), dec=1)}\n')
                f.write(f'    Avg Near Misses:             {_fmt(avg_agg.get("avg_near_misses"), dec=1)}\n')
                f.write(f'    Avg Head-On Events:          {_fmt(avg_agg.get("avg_head_on_events"), dec=1)}\n')

                f.write('\n  SMOOTHNESS\n')
                f.write(f'    Avg Speed:                   {_fmt(avg_agg.get("avg_speed"))} m/s\n')
                f.write(f'    Avg Acceleration:            {_fmt(avg_agg.get("avg_acceleration"))} m/s²\n')
                f.write(f'    Avg Jerk:                    {_fmt(avg_agg.get("avg_jerk"))} m/s³\n')
                f.write(f'    Smoothness Index:            {_fmt(avg_agg.get("smoothness_index"))}\n')
                f.write(f'    Avg Angular Velocity:        {_fmt(avg_agg.get("avg_angular_velocity"))} rad/s\n')
                f.write(f'    Avg Path Irregularity:       {_fmt(avg_agg.get("avg_path_irregularity"))}\n')
                f.write(f'    Avg Direction Changes:       {_fmt(avg_agg.get("avg_direction_changes"), dec=1)}\n')
                f.write(f'    Avg Freeze Events:           {_fmt(avg_agg.get("avg_freeze_events"), dec=1)}\n')

                f.write('\n  SOCIAL\n')
                f.write(f'    Avg Yielding Events:         {_fmt(avg_agg.get("avg_yielding_events"), dec=1)}\n')
                f.write(f'    Avg Cut-In Events:           {_fmt(avg_agg.get("avg_cut_in_events"), dec=1)}\n')
                f.write(f'    Avg Side-Pass Right:         {_fmt(avg_agg.get("avg_side_pass_right"), dec=1)}\n')
                f.write(f'    Avg Side-Pass Left:          {_fmt(avg_agg.get("avg_side_pass_left"), dec=1)}\n')
                f.write(f'    Side-Pass Bias (0=sym):      {_fmt(avg_agg.get("side_pass_bias"))}\n')
                f.write(f'    Avg Disturbance Caused:      {_fmt(avg_agg.get("avg_disturbance_caused"))}\n')

                f.write('\n  EFFORT\n')
                f.write(f'    Avg Control Effort:          {_fmt(avg_agg.get("avg_control_effort"))}\n')

                f.write('\n  JOINT COHESION\n')
                f.write(f'    Avg Pairwise Distance:       {_fmt(avg_agg.get("avg_pairwise_dist"))} m\n')
                f.write(f'    Std Pairwise Distance:       {_fmt(avg_agg.get("std_pairwise_dist"))} m\n')

            if args.npz_hard:
                f.write('\n--- Per Threat Type (agent-0) ---\n')
                for tt in sorted(set(threat_types)):
                    idxs   = [j for j, t in enumerate(threat_types) if t == tt]
                    tt_sr  = 100 * np.mean([successes[j]     for j in idxs])
                    tt_min = np.mean([min_dists[j]           for j in idxs])
                    f.write(f'  {tt:10s}: n={len(idxs):4d}  '
                            f'SR={tt_sr:.1f}%  minD={tt_min:.3f}\n')

        logging.info('Results saved to %s', eval_file)

        # ── Combine videos ────────────────────────────────────────────────────
        if rendered_files:
            import subprocess
            list_file = 'video_list.txt'
            with open(list_file, 'w') as f:
                for vid in rendered_files:
                    f.write(f"file '{vid}'\n")
            combined = os.path.join(results_dir, f'{bp}_VIDEOS.mp4')
            subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0',
                            '-i', list_file, '-c', 'copy', combined])
            for vf in rendered_files:
                os.remove(vf)
            os.remove(list_file)
            logging.info('Combined video: %s', combined)

    else:
        # Non-visualize path: use explorer (single-agent batch eval, unchanged)
        if args.multi_agent:
            logging.warning(
                '--multi_agent without --visualize falls back to single-agent '
                'explorer.run_k_episodes. Pass --visualize to use multi-agent eval.'
            )
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()