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
from crowd_sim.envs.utils.info import *

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument("--results_suffix", type=str, default="",
                        help="Optional suffix for results folder, e.g. 'stageB'")
    # ── NPZ hard evaluation ───────────────────────────────────────────────────
    parser.add_argument('--npz_hard', default=False, action='store_true',
                        help='Evaluate on pre-generated hard npz scenarios')
    parser.add_argument('--npz_dir', type=str, default='random_stage_B_examples_HARDEST',
                        help='Directory containing hard .npz scenario files')
    parser.add_argument('--npz_num_eval', type=int, default=50,
                        help='Number of hard scenarios to evaluate on')
    # ── Map-eval (structured layout) evaluation ───────────────────────────────
    parser.add_argument('--map_eval', default=False, action='store_true',
                        help='Generate structured map layouts (corridor, room, etc.) '
                             'for evaluation. Only meaningful when use_map=true in policy config.')
    parser.add_argument('--map_eval_num', type=int, default=50,
                        help='Number of map_eval scenarios to evaluate')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    # Use env.unwrapped to reach the underlying CrowdSim directly — gym wrappers
    # do not forward __setattr__ to the wrapped env, so env.test_sim = '...'
    # would only set the attribute on the wrapper, not on the CrowdSim itself.
    _sim = env.unwrapped
    if args.square:
        _sim.test_sim = 'square_crossing'
    if args.circle:
        _sim.test_sim = 'circle_crossing'
    if args.npz_hard:
        _sim.test_sim = 'npz_hard'
    if args.map_eval:
        _sim.test_sim = 'map_eval'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.visualize:

        # ── Build scenario list ───────────────────────────────────────────────
        if args.npz_hard:
            # Load hard-only npz files, sorted for full determinism across policies
            all_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
            hard_files = []
            for f in all_files:
                try:
                    d = np.load(f, allow_pickle=True)
                    difficulty = str(d['difficulty']) if 'difficulty' in d else None
                    threat = str(d['threat_type']) if 'threat_type' in d else None
                    is_hard = (
                        difficulty == 'hard' or
                        threat in ['frontal', 'side', 'rear', 'mixed', 'pincer', 'group']
                    )
                    if is_hard:
                        hard_files.append(f)
                except Exception as e:
                    logging.warning('Could not load %s: %s', f, e)

            if len(hard_files) == 0:
                raise ValueError(f'No hard npz files found in {args.npz_dir}')

            hard_files = hard_files[:args.npz_num_eval]
            num_tests = len(hard_files)
            logging.info('NPZ hard evaluation: %d scenarios from %s', num_tests, args.npz_dir)
        elif args.map_eval:
            # map_eval: scenes are generated on-the-fly from a seeded RNG
            hard_files = None
            num_tests  = args.map_eval_num
            logging.info('Map-eval evaluation: %d generated scenarios', num_tests)
        else:
            hard_files = None
            num_tests = 50

        # ── Metrics ──────────────────────────────────────────────────────────
        episode_times = []
        path_lengths = []
        successes = []
        timeouts = []
        collisions = []           # agent-agent collisions (Collision)
        static_map_collisions = [] # robot-static-map collisions (WallCollision)
        min_dists = []
        avg_min_dists = []
        threat_types = []         # map_type string for map_eval; threat label otherwise
        rendered_files = []
        base_policy = None

        for i in range(num_tests):
            logging.info('Starting test case %d / %d', i + 1, num_tests)

            # ── Reset: npz_hard / map_eval / standard ────────────────────────
            if args.npz_hard:
                npz_path = hard_files[i]
                env.load_npz_scenario(npz_path)
                # test_case=i gives a fixed per-scenario seed — same across policies
                ob = env.reset(phase='test', test_case=i)

                # Record threat type for per-category breakdown
                try:
                    d = np.load(npz_path, allow_pickle=True)
                    threat_types.append(str(d.get('threat_type', 'unknown')))
                except Exception:
                    threat_types.append('unknown')
            elif args.map_eval:
                # Scene is generated inside reset() via the seeded RNG;
                # test_case=i ensures each scenario is repeatable.
                ob = env.reset(phase='test', test_case=i)
                # Record the map type generated for this episode
                map_type = getattr(env, '_npz_map_eval_type', 'unknown')
                threat_types.append(map_type)
            else:
                ob = env.reset(phase='test')
                threat_types.append('standard')

            done = False
            last_pos = np.array(robot.get_position())

            while not done:
                action = robot.act(ob)
                ob, _, done, info = env.step(action)
                current_pos = np.array(robot.get_position())
                logging.debug('Speed: %.2f',
                              np.linalg.norm(current_pos - last_pos) / robot.time_step)
                last_pos = current_pos

            # ── Render ───────────────────────────────────────────────────────
            if args.traj:
                env.render(mode='human')
                base, ext = os.path.splitext(args.video_file)
                base_policy = base
                env.render(mode='traj', output_file=f'{base}_{i+1}{ext}',
                           basepolicy=base, testnum=i)
            else:
                env.render(mode='human')
                base, ext = os.path.splitext(args.video_file)
                base_policy = base
                env.render(mode='video', output_file=f'{base}_{i+1}{ext}',
                           basepolicy=base, testnum=i)
                rendered_files.append(f'{base}_{i+1}{ext}')

            # ── Record metrics ───────────────────────────────────────────────
            if isinstance(info, ReachGoal):
                episode_times.append(env.global_time)
                path_lengths.append(env.pathlength)

            successes.append(1 if isinstance(info, ReachGoal) else 0)
            timeouts.append(1 if isinstance(info, Timeout) else 0)
            collisions.append(1 if isinstance(info, Collision) else 0)
            static_map_collisions.append(1 if isinstance(info, WallCollision) else 0)
            min_dists.append(env.minobsdist)
            avg_min_dists.append(np.mean(env.avgobsdist) if env.avgobsdist else float('inf'))

            logging.info('Test %d: %.2fs | %s | path=%.2f minD=%.3f',
                         i + 1, env.global_time, info,
                         env.pathlength, env.minobsdist)

        # ── Aggregate and save results ────────────────────────────────────────
        avg_time                  = np.mean(episode_times) if episode_times else float('nan')
        avg_path_length           = np.mean(path_lengths)  if path_lengths  else float('nan')
        success_rate              = 100 * np.mean(successes)
        timeout_rate              = 100 * np.mean(timeouts)
        collision_rate            = 100 * np.mean(collisions)
        static_map_collision_rate = 100 * np.mean(static_map_collisions)
        avg_min_dist              = np.mean(min_dists)
        avg_avg_min_dist          = np.mean(avg_min_dists)

        results_dir = f'results_{args.results_suffix}' if args.results_suffix else 'results'
        os.makedirs(results_dir, exist_ok=True)
        eval_file = os.path.join(results_dir, f'{base_policy}_eval.txt')

        with open(eval_file, 'w') as f:
            f.write('========== Evaluation Results ==========\n')
            mode_label = 'NPZ Hard' if args.npz_hard else 'Standard'
            f.write(f'Evaluation mode: {mode_label}\n')
            if args.npz_hard:
                f.write(f'NPZ directory: {args.npz_dir}\n')
            f.write(f'Number of test episodes: {num_tests}\n')
            if args.map_eval:
                f.write('Map types evaluated: corridor, room, doorway, pillars, '
                        'narrow_passage, L_corner, single_wall, open\n')
            f.write(f'Success Rate: {success_rate:.2f} %\n')
            f.write(f'Timeout Rate: {timeout_rate:.2f} %\n')
            f.write(f'Agent Collision Rate: {collision_rate:.2f} %\n')
            f.write(f'Static Map Collision Rate: {static_map_collision_rate:.2f} %\n')
            f.write(f'Total Collision Rate: {collision_rate + static_map_collision_rate:.2f} %\n')
            f.write(f'Average Time to Goal: {avg_time:.3f}\n')
            f.write(f'Average Path Length to Goal: {avg_path_length:.3f}\n')
            f.write(f'Average Minimum Distance to Obstacles: {avg_min_dist:.3f}\n')
            f.write(f'Average Per-Episode Avg Min Distance: {avg_avg_min_dist:.3f}\n')
            f.write(f'Policy: {base_policy}\n')

            # Per-category breakdown (threat type for npz_hard, map type for map_eval)
            if args.npz_hard or args.map_eval:
                label = 'Map Type' if args.map_eval else 'Threat Type'
                f.write(f'\n--- Per {label} ---\n')
            if args.npz_hard or args.map_eval:
                unique_threats = sorted(set(threat_types))
                for tt in unique_threats:
                    idxs = [j for j, t in enumerate(threat_types) if t == tt]
                    tt_sr     = 100 * np.mean([successes[j] for j in idxs])
                    tt_col    = 100 * np.mean([collisions[j] for j in idxs])
                    tt_mapcol = 100 * np.mean([static_map_collisions[j] for j in idxs])
                    tt_min    = np.mean([min_dists[j] for j in idxs])
                    f.write(f'  {tt:10s}: n={len(idxs):4d}  '
                            f'SR={tt_sr:.1f}%  '
                            f'AgentCR={tt_col:.1f}%  '
                            f'MapCR={tt_mapcol:.1f}%  '
                            f'minD={tt_min:.3f}\n')

        logging.info('Results saved to %s', eval_file)

        # ── Combine videos ────────────────────────────────────────────────────
        if rendered_files:
            import subprocess
            list_file = 'video_list.txt'
            with open(list_file, 'w') as f:
                for vid in rendered_files:
                    f.write(f"file '{vid}'\n")
            combined = os.path.join(results_dir, f'{base_policy}_VIDEOS{ext}')
            subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0',
                            '-i', list_file, '-c', 'copy', combined])
            for f in rendered_files:
                os.remove(f)
            os.remove(list_file)
            logging.info('Combined video saved as %s', combined)

    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()