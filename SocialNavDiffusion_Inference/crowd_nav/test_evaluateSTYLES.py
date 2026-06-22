#!/usr/bin/env python3
"""
test_evaluateSTYLES.py

Style-sweep evaluation for DiffusionConditionalUNet1DCFG.
Runs 6 policy variants on every test scene and produces a single 6-panel
video per scene where all panels are time-synchronized:

  Panel 0 — style=[0,0,0,0]  + projection ON   (neutral baseline)
  Panel 1 — style=[1,0,0,0]  + projection ON   (proxemic conservative)
  Panel 2 — style=[0,1,0,0]  + projection ON   (pass-side bias)
  Panel 3 — style=[0,0,1,0]  + projection ON   (yielding)
  Panel 4 — style=[0,0,0,1]  + projection ON   (group-aware)
  Panel 5 — style=[0,0,0,0]  + projection OFF  (neutral, no feasibility layer)

All 6 panels advance frame-by-frame in lockstep.  Shorter episodes hold
their last frame until the longest episode in that scene finishes.  All
per-scene videos are concatenated into one combined output file.

A CSV summary of per-variant, per-scene metrics is written to the results dir.

Usage
-----
  python test_evaluateSTYLES.py \\
      --policy_config  path/to/policy.config \\
      --env_config     path/to/env.config \\
      --policy         diffusion_conditional_unet1dcfg \\
      --video_file     style_sweep.mp4 \\
      [--num_tests 50] \\
      [--npz_hard --npz_dir DIR] [--map_eval] [--square] [--circle] \\
      [--fps 10] [--results_suffix TAG]
"""

import logging
import argparse
import configparser
import copy
import os
import subprocess
import glob
import csv

import numpy as np
import torch
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib import animation

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import *  # ReachGoal, Collision, Timeout, WallCollision …

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# Each entry: (short_label, style_vector, use_projection)
# Axis ordering matches STYLE_AXES in diffusion_CondUNetCFG.py:
#   [prox, pass, yield, group]
STYLE_VARIANTS = [
    ("neutral+proj",   [0.0, 0.0, 0.0, 0.0], True),
    ("prox+proj",      [1.0, 0.0, 0.0, 0.0], True),
    ("pass+proj",      [0.0, 1.0, 0.0, 0.0], True),
    ("yield+proj",     [0.0, 0.0, 1.0, 0.0], True),
    ("group+proj",     [0.0, 0.0, 0.0, 1.0], True),
    ("neutral-noproj", [0.0, 0.0, 0.0, 0.0], False),
    ("prox-proj",      [-1.0, 0.0, 0.0, 0.0], True),
    ("pass-proj",      [0.0, -1.0, 0.0, 0.0], True),
    ("yield-proj",     [0.0, 0.0, -1.0, 0.0], True),
    ("group-proj",     [0.0, 0.0, 0.0, -1.0], True),
]
N_VARIANTS = len(STYLE_VARIANTS)  # 10

# Video layout: 2 rows × 3 cols
GRID_ROWS = 2
GRID_COLS = 5

_ROBOT_COLORS = ['gold', 'tomato', 'salmon',
                 'limegreen', 'darkgreen',
                 'deepskyblue', 'navy',
                 'mediumpurple', 'indigo',
                 'darkorange']

# Predicted/selected/projected trajectories share ONE color scheme across
# all style panels (only the past/trail trajectory is per-style colored via
# _ROBOT_COLORS), so style-vs-style comparisons aren't muddied by color noise.
_PROJ_COLOR   = 'royalblue'

_OUTCOME_COLORS = {
    'success':   'green',
    'collision': 'red',
    'timeout':   'darkorange',
    'other':     'gray',
}




# ---------------------------------------------------------------------------
# Fixed-scale social metric computation for style comparison
#
# All metrics use NEUTRAL weights (no style modulation) so numbers are
# directly comparable across the 6 variants.  Each metric is designed so
# the style axis that should improve it produces a LOWER score.
#
# prox axis  → metrics: prox_violation_rate, mean_min_clearance
# pass axis  → metrics: pass_side_consistency (signed, not a penalty)
# yield axis → metrics: head_on_rate, cutoff_rate
# group axis → metrics: group_split_score
# ---------------------------------------------------------------------------

_EVAL_DT           = 0.05
_EVAL_COLL_R       = 0.65   # same as generation COLLISION_RADIUS
_EVAL_PROX_SIGMA   = 2.0    # same as generation PROX_SIGMA
_EVAL_TTC_TAU      = 2.0
_EVAL_TTC_MAX      = 5.0
_EVAL_GROUP_DIST   = 1.5
_EVAL_V_MAX        = 1.0

# c2 — head-on
_EVAL_C2_SIGMA_MISS = 0.9

# c5 — cutoff
_EVAL_C5_SIGMA_LAT  = 0.6
_EVAL_C5_AHEAD_RAMP = 0.5
_EVAL_C5_T_GAP_MAX  = 50.0

# c7 — group split
_EVAL_C7_SIGMA_SPLIT = 0.45
_EVAL_C7_CAND_FACTOR = 1.5
_EVAL_C7_HEAD_COS    = 0.866
_EVAL_C7_SPEED_SIM   = 0.3


def compute_episode_social_metrics(states, humans_data, time_step):
    """
    Compute fixed-scale social metrics for one episode.

    Parameters
    ----------
    states      : list of [robot_full_state, [human_full_states...]]
                  as stored in sim.states
    humans_data : list of human objects (for radius; use sim.humans at
                  episode end — radii don't change mid-episode)
    time_step   : float

    Returns
    -------
    dict of scalar metrics, all on fixed neutral scale
    """
    T   = len(states)
    dt  = time_step
    M   = len(states[0][1]) if T > 0 else 0

    # ── Extract trajectories ──────────────────────────────────────────────
    rob_xy    = np.array([s[0].position for s in states],  dtype=np.float32)  # (T, 2)
    rob_theta = np.array([s[0].theta    for s in states],  dtype=np.float32)  # (T,)
    rob_vx_fs = np.array([s[0].vx       for s in states],  dtype=np.float32)  # (T,)
    rob_vy_fs = np.array([s[0].vy       for s in states],  dtype=np.float32)  # (T,)

    if M == 0:
        return _zero_metrics()

    hum_xy = np.array(
        [[s[1][j].position for j in range(M)] for s in states],
        dtype=np.float32,
    )  # (T, M, 2)

    hum_vx = np.array(
        [[s[1][j].vx for j in range(M)] for s in states], dtype=np.float32
    )  # (T, M)
    hum_vy = np.array(
        [[s[1][j].vy for j in range(M)] for s in states], dtype=np.float32
    )  # (T, M)
    hum_vel = np.stack([hum_vx, hum_vy], axis=-1)  # (T, M, 2)

    # Use mean velocity per human as heading proxy (more stable than per-frame)
    mean_hum_vel = hum_vel.mean(axis=0)                        # (M, 2)
    hum_spd      = np.linalg.norm(mean_hum_vel, axis=-1)      # (M,)
    hum_vhat     = mean_hum_vel / hum_spd[:, None].clip(min=1e-4)  # (M, 2)

    # ── Pairwise robot-human distances ────────────────────────────────────
    delta = rob_xy[:, None, :] - hum_xy           # (T, M, 2)
    dist  = np.linalg.norm(delta, axis=-1)         # (T, M)
    dist  = dist.clip(min=1e-4)

    # Robot velocity vectors
    rob_vel    = np.stack([rob_vx_fs, rob_vy_fs], axis=-1)    # (T, 2)

    # ── 1. Proximity metrics (prox axis) ──────────────────────────────────
    # prox_violation_frac: fraction of (timestep, human) pairs where robot
    # is within PROX_SIGMA of a human. Lower is better; prox+ should reduce.
    prox_mask = (dist < _EVAL_PROX_SIGMA)          # (T, M) bool
    prox_violation_frac = float(prox_mask.mean())

    # mean_min_clearance: average over timesteps of minimum distance to any
    # human. Higher is better; prox+ should increase.
    min_dist_per_step = dist.min(axis=1)            # (T,)
    mean_min_clearance = float(min_dist_per_step.mean())

    # ── 2. Passing-side consistency (pass axis) ───────────────────────────
    # For each (timestep, human) pair, measure which side of the human the
    # robot is on. Sign convention: +1 = robot is to human's left,
    # -1 = robot is to human's right. We use the human's mean velocity as
    # their forward direction; lateral axis is 90° CCW from that.
    #
    # pass_side_mean: mean signed side across all moving-human encounters
    # within PROX_SIGMA * 2. Values near +1 = always pass left, near -1 =
    # always pass right, near 0 = inconsistent.
    #
    # This is a SIGNED metric, not a penalty.  pass+ (right-hand traffic)
    # should push it toward -1; pass- (left-hand) toward +1.
    hum_left = np.stack([-hum_vhat[:, 1], hum_vhat[:, 0]], axis=-1)  # (M, 2)
    # delta points FROM human TO robot → positive dot = robot is to human's left
    robot_on_left = (delta * hum_left[None]).sum(axis=-1)             # (T, M)
    robot_on_left_sign = np.sign(robot_on_left)                       # (T, M)

    # Only count encounters with moving humans within 2× prox sigma
    moving    = (hum_spd > 0.1)[None, :]                              # (1, M)
    in_range  = dist < (_EVAL_PROX_SIGMA * 2)                         # (T, M)
    encounter = moving & in_range                                      # (T, M)
    if encounter.any():
        pass_side_mean = float(robot_on_left_sign[encounter].mean())
    else:
        pass_side_mean = 0.0

    # pass_side_consistency: how consistently the robot chooses one side.
    # abs(pass_side_mean) near 1 = very consistent, near 0 = erratic.
    pass_side_consistency = abs(pass_side_mean)

    # ── 3. Head-on rate (yield axis) ──────────────────────────────────────
    # Fraction of timesteps where robot is in a head-on approach with ANY
    # human: mutual aim × closing × CPA-miss kernel × TTC urgency (c2).
    rob_vhat_arr = (rob_vel / np.linalg.norm(rob_vel, axis=-1, keepdims=True)
                    .clip(min=1e-4))                                   # (T, 2)

    to_ped     = -delta                                                # rob -> ped, (T, M, 2)
    to_ped_hat = to_ped / dist[..., None]                              # (T, M, 2)
    rel_vel    = rob_vel[:, None, :] - hum_vel                         # (T, M, 2)
    closing    = (rel_vel * to_ped_hat).sum(axis=-1).clip(min=0)       # (T, M)
    closing_norm = (closing / _EVAL_V_MAX).clip(0, 1)                  # (T, M)

    rob_aim = (rob_vhat_arr[:, None, :] * to_ped_hat).sum(axis=-1).clip(min=0)  # (T, M)
    ped_aim = (-(hum_vhat[None] * to_ped_hat).sum(axis=-1)).clip(min=0)         # (T, M)

    v_rel_sq = (rel_vel ** 2).sum(axis=-1).clip(min=1e-6)
    t_cpa    = ((to_ped * rel_vel).sum(axis=-1) / v_rel_sq).clip(0, _EVAL_TTC_MAX)
    miss     = np.linalg.norm(to_ped - rel_vel * t_cpa[..., None], axis=-1)

    c2_step = (rob_aim * ped_aim * closing_norm
               * np.exp(-miss**2 / (2.0 * _EVAL_C2_SIGMA_MISS**2))
               * np.exp(-t_cpa / _EVAL_TTC_TAU)
               * moving.astype(np.float32))                            # (T, M)
    headon_rate = float((c2_step.max(axis=1) > 0.3).mean())           # fraction of timesteps

    # ── 4. Cut-off rate (yield axis) ────────────────────────────────────────
    # Fraction of timesteps where the robot is blocking a moving human's
    # forward corridor (c5).
    hs     = hum_spd.clip(min=1e-4)                                    # (M,)
    s_long = -(to_ped * hum_vhat[None]).sum(axis=-1)                   # (T, M)
    l_lat  = np.abs((to_ped * hum_left[None]).sum(axis=-1))            # (T, M)
    ahead  = (s_long / _EVAL_C5_AHEAD_RAMP).clip(0, 1)
    lat_k  = np.exp(-l_lat**2 / (2.0 * _EVAL_C5_SIGMA_LAT**2))
    t_gap  = (s_long.clip(min=0) / hs[None]).clip(max=_EVAL_C5_T_GAP_MAX)
    urgency  = np.exp(-t_gap / _EVAL_TTC_TAU)
    v_along  = (rob_vel[:, None, :] * hum_vhat[None]).sum(axis=-1)     # (T, M)
    blocking = (1.0 - v_along / hs[None]).clip(0, 1)
    c5_step  = ahead * lat_k * urgency * blocking * moving.astype(np.float32)  # (T, M)
    cutoff_rate = float((c5_step.max(axis=1) > 0.2).mean())           # fraction of timesteps

    # ── 5. Group split score (group axis) ─────────────────────────────────
    # Mean (over time) of the worst per-pair intrusion into the segment
    # between co-walking (or both-stationary) pedestrian pairs (c7).
    group_split_score = 0.0
    if M >= 2:
        moving_vec = (hum_spd > 0.1)                                   # (M,) bool

        seg_p  = hum_xy[:, None, :, :] - hum_xy[:, :, None, :]         # (T,M,M,2): xy_j - xy_i
        sep_p  = np.linalg.norm(seg_p, axis=-1).clip(min=1e-4)         # (T,M,M)
        seg_sq = (sep_p ** 2).clip(min=1e-6)

        robdot = (rob_xy[:, None, None, :] * seg_p).sum(axis=-1)       # (T,M,M)
        pdot   = (hum_xy[:, :, None, :] * seg_p).sum(axis=-1)          # (T,M,M)
        dotri  = robdot - pdot
        u      = dotri / seg_sq
        u_c    = u.clip(0, 1)

        dist_i = dist[:, :, None]                                      # (T,M,1) -> dist[t,i]
        d_sq   = (dist_i**2 - 2.0 * u_c * dotri + u_c**2 * seg_sq).clip(min=0)
        interior = (4.0 * u * (1.0 - u)).clip(0, 1)
        pairfield = interior * np.exp(-d_sq / (2.0 * _EVAL_C7_SIGMA_SPLIT**2))  # (T,M,M)

        cosij     = (hum_vhat[:, None, :] * hum_vhat[None, :, :]).sum(axis=-1)  # (M,M)
        both_mov  = moving_vec[:, None] & moving_vec[None, :]
        both_stat = (~moving_vec)[:, None] & (~moving_vec)[None, :]
        speed_sim = np.abs(hum_spd[:, None] - hum_spd[None, :]) < _EVAL_C7_SPEED_SIM
        co   = ((cosij > _EVAL_C7_HEAD_COS) & both_mov & speed_sim) | both_stat
        triu = np.triu(np.ones((M, M), dtype=bool), k=1)
        sep_min_t = sep_p.min(axis=0)                                  # (M,M)
        cand = co & triu & (sep_min_t < _EVAL_GROUP_DIST * _EVAL_C7_CAND_FACTOR)

        if cand.any():
            final_mask = cand[None] & (sep_p < _EVAL_GROUP_DIST)       # (T,M,M)
            c7_per_t = (pairfield * final_mask).max(axis=(1, 2))       # (T,)
            group_split_score = float(c7_per_t.mean())

    return {
        # prox axis — lower is better for both; prox+ should reduce violation, increase clearance
        'prox_violation_frac':   prox_violation_frac,
        'mean_min_clearance_m':  mean_min_clearance,
        # pass axis — signed; pass+ (RHT) → toward -1, pass- (LHT) → toward +1
        'pass_side_mean':        pass_side_mean,
        'pass_side_consistency': pass_side_consistency,
        # yield axis — lower is better; yield+ should reduce both
        'headon_rate':           headon_rate,
        'cutoff_rate':           cutoff_rate,
        # group axis — lower is better; group+ should reduce
        'group_split_score':     group_split_score,
    }


def _zero_metrics():
    return {
        'prox_violation_frac':   0.0,
        'mean_min_clearance_m':  float('inf'),
        'pass_side_mean':        0.0,
        'pass_side_consistency': 0.0,
        'headon_rate':           0.0,
        'cutoff_rate':           0.0,
        'group_split_score':     0.0,
    }



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_policy_state(policy):
    """Zero out per-episode velocity memory so each run starts cold."""
    if hasattr(policy, '_reset_unicycle_state'):
        policy._reset_unicycle_state()
    else:
        for attr, val in [('prev_v', 0.0), ('prev_omega', 0.0),
                          ('_v0_from_last_step', 0.0),
                          ('_w0_from_last_step', 0.0)]:
            if hasattr(policy, attr):
                setattr(policy, attr, val)


def _outcome_label(info):
    if isinstance(info, ReachGoal):
        return 'success'
    if isinstance(info, Collision):
        return 'collision'
    if isinstance(info, Timeout):
        return 'timeout'
    return 'other'


def run_one_episode(env, robot, policy, style_vec, use_proj,
                    phase, scene_idx, npz_path=None,
                    scene_testoffset=None, scene_counter=None):
    """
    Run one episode for the given (style_vec, use_proj) variant on scene_idx.

    scene_testoffset / scene_counter: if provided, the env's internal RNG
    anchor (sim.testoffset and sim.case_counter[phase]) is restored to these
    values before reset().  This guarantees all 6 variants see the same scene
    regardless of how many reset() calls have accumulated beforehand.

    Returns a dict of everything needed for rendering + metrics.
    """
    policy.style_vector    = np.array(style_vec, dtype=np.float32)
    policy.use_projection  = use_proj
    _reset_policy_state(policy)

    sim = env.unwrapped

    # Restore the RNG anchor so this variant gets the same scene as the first
    # variant for scene_idx.  testoffset increments on every reset(), so
    # without restoring it each subsequent variant would receive a different
    # random seed and thus a different scene.
    if scene_testoffset is not None:
        sim.testoffset = scene_testoffset
    if scene_counter is not None:
        sim.case_counter[phase] = scene_counter

    if npz_path is not None:
        env.load_npz_scenario(npz_path)

    ob = env.reset(phase=phase, test_case=scene_idx)

    done = False
    while not done:
        action = robot.act(ob)
        ob, _, done, info = env.step(action)

    # Compute fixed-scale social metrics for style comparison
    social_metrics = compute_episode_social_metrics(
        copy.deepcopy(sim.states),
        sim.humans,
        sim.time_step,
    )

    return {
        'states':           copy.deepcopy(sim.states),
        'predicted_trajs':  copy.deepcopy(sim.predicted_trajs),
        'info':             info,
        'global_time':      sim.global_time,
        'pathlength':       sim.pathlength,
        'minobsdist':       sim.minobsdist,
        'avgobsdist':       list(sim.avgobsdist),
        'human_radii':      [h.radius for h in sim.humans],
        'n_humans':         len(sim.humans),
        'robot_radius':     robot.radius,
        'robot_kinematics': robot.kinematics,
        'goal_pos':         tuple(robot.get_goal_position()),
        'time_step':        float(sim.time_step),
        'occ_map':  copy.deepcopy(getattr(sim, '_npz_occ_map', None)),
        'has_map':  float(getattr(sim, '_npz_has_map', 0.0)),
        'map_extent': float(getattr(sim, '_npz_map_extent', 10.0)),
        'social_metrics': social_metrics,   # ← new
    }


def _pad_to(lst, target_len):
    """Return a list of length target_len, repeating the last element."""
    if not lst:
        return [None] * target_len
    return list(lst) + [lst[-1]] * max(0, target_len - len(lst))


# ---------------------------------------------------------------------------
# Multi-panel per-scene video
# ---------------------------------------------------------------------------

def build_scene_video(episodes, scene_idx, output_path, fps=10):
    """
    Build one MP4 with GRID_ROWS × GRID_COLS panels from `episodes` list.
    All panels are synchronized: shorter episodes hold their last frame.
    """
    max_frames = max(len(ep['states']) for ep in episodes)

    padded_states = [_pad_to(ep['states'],         max_frames) for ep in episodes]
    padded_trajs  = [_pad_to(ep['predicted_trajs'], max_frames) for ep in episodes]

    cmap_human = plt.cm.get_cmap('hsv', 10)
    x_off = y_off = 0.11
    arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                              figsize=(6 * GRID_COLS, 6 * GRID_ROWS))
    axes = np.array(axes).flatten()
    fig.suptitle(f'Style Sweep — Scene {scene_idx + 1}', fontsize=16,
                 fontweight='bold', y=0.995)
    # Leave headroom at the top for the suptitle (rect) and widen the gap
    # between rows (hspace) so row-2 panel titles don't overlap row-1's
    # x-axis labels/ticks.
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.4)

    panels = []  # per-panel mutable render state

    for p, (ep, (label, style_vec, use_proj)) in enumerate(
            zip(episodes, STYLE_VARIANTS)):

        ax          = axes[p]
        r_color     = _ROBOT_COLORS[p]

        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)

        style_str = "[" + ",".join(f"{v:+.0f}" for v in style_vec) + "]"
        proj_str  = "proj=ON" if use_proj else "proj=OFF"
        ax.set_title(f'{label}\n{style_str}  {proj_str}', fontsize=9,
                     fontweight='bold')

        # Static map underlay
        occ = ep['occ_map']
        if occ is not None and ep['has_map'] > 0.5:
            half = ep['map_extent'] / 2.0
            ax.imshow(occ, extent=[-half, half, -half, half],
                      origin='lower', cmap='gray_r', alpha=0.35, zorder=0,
                      interpolation='nearest')

        # Goal marker
        gx, gy = ep['goal_pos']
        goal_dot = mlines.Line2D([gx], [gy], color='red', marker='*',
                                  linestyle='None', markersize=11, label='Goal',
                                  zorder=5)
        ax.add_artist(goal_dot)

        # Initial positions
        s0 = padded_states[p][0]
        r_pos0 = s0[0].position
        h_pos0 = [s0[1][j].position for j in range(ep['n_humans'])]

        robot_circle = plt.Circle(r_pos0, ep['robot_radius'],
                                   fill=True, color=r_color, zorder=4)
        ax.add_artist(robot_circle)

        robot_trail, = ax.plot([], [], '--', color=r_color, lw=1.5,
                                alpha=0.7, zorder=1)

        h_circles = [
            plt.Circle(h_pos0[j], ep['human_radii'][j],
                       fill=False, color=cmap_human(j), zorder=3)
            for j in range(ep['n_humans'])
        ]
        h_labels = [
            ax.text(h_pos0[j][0] - x_off, h_pos0[j][1] - y_off, str(j),
                    color='black', fontsize=8, zorder=6)
            for j in range(ep['n_humans'])
        ]
        for hc in h_circles:
            ax.add_artist(hc)

        # Initial heading arrow
        arrows = []
        if ep['robot_kinematics'] == 'unicycle':
            th0 = s0[0].theta
            r0  = ep['robot_radius']
            arr = patches.FancyArrowPatch(
                r_pos0,
                (r_pos0[0] + r0 * np.cos(th0),
                 r_pos0[1] + r0 * np.sin(th0)),
                color='red', arrowstyle=arrow_style, zorder=4
            )
            ax.add_artist(arr)
            arrows.append(arr)

        # Trajectory lines
        K = 0
        for td in ep['predicted_trajs']:
            if isinstance(td, dict) and td.get('all_samples'):
                K = len(td['all_samples'])
                break
        bg_lines = [
            ax.plot([], [], '-', color='orange', lw=1.2, alpha=0.55, zorder=1)[0]
            for _ in range(max(K - 1, 0))
        ]
        sel_line,  = ax.plot([], [], '-', color='purple',  lw=2,   alpha=0.8,
                              zorder=2, label='Selected')
        proj_line, = ax.plot([], [], '-', color=_PROJ_COLOR, lw=2.5, alpha=0.9,
                              zorder=3, label='Projected')

        time_txt = ax.text(-7.0, 6.5, 'T: 0.00 s', fontsize=8, zorder=7)

        # Outcome annotation (shown on final frame)
        outcome = _outcome_label(ep['info'])
        if outcome == 'success':
            out_str = f'SUCCESS  {ep["global_time"]:.2f}s'
        else:
            out_str = outcome.upper()
        out_col  = _OUTCOME_COLORS[outcome]
        out_text = ax.text(0, 0, out_str, fontsize=11, fontweight='bold',
                            color=out_col, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.75,
                                      edgecolor=out_col, boxstyle='round'),
                            visible=False, zorder=10)

        ax.legend(handles=[robot_circle, goal_dot, sel_line, proj_line],
                  labels=['Robot', 'Goal', 'Selected', 'Projected'],
                  fontsize=7, loc='upper right')

        panels.append({
            'ax':           ax,
            'robot_circle': robot_circle,
            'robot_trail':  robot_trail,
            'h_circles':    h_circles,
            'h_labels':     h_labels,
            'arrows':       arrows,
            'bg_lines':     bg_lines,
            'sel_line':     sel_line,
            'proj_line':    proj_line,
            'time_txt':     time_txt,
            'out_text':     out_text,
            'ep_len':       len(ep['states']),
            'kinematics':   ep['robot_kinematics'],
            'robot_radius': ep['robot_radius'],
            'n_humans':     ep['n_humans'],
            'time_step':    ep['time_step'],
        })

    def _update(frame_num):
        for p, pd in enumerate(panels):
            # Hold last frame for episodes that ended earlier
            f  = min(frame_num, pd['ep_len'] - 1)
            st = padded_states[p][f]
            td = padded_trajs[p][f]

            ax = pd['ax']

            # Robot
            r_pos = st[0].position
            pd['robot_circle'].center = r_pos

            trail_x = [padded_states[p][k][0].position[0] for k in range(f + 1)]
            trail_y = [padded_states[p][k][0].position[1] for k in range(f + 1)]
            pd['robot_trail'].set_data(trail_x, trail_y)

            # Humans
            for j, hc in enumerate(pd['h_circles']):
                hp = st[1][j].position
                hc.center = hp
                pd['h_labels'][j].set_position((hp[0] - x_off, hp[1] - y_off))

            # Heading arrow
            for arr in pd['arrows']:
                arr.remove()
            pd['arrows'].clear()
            if pd['kinematics'] == 'unicycle':
                th = st[0].theta
                r0 = pd['robot_radius']
                arr = patches.FancyArrowPatch(
                    r_pos,
                    (r_pos[0] + r0 * np.cos(th),
                     r_pos[1] + r0 * np.sin(th)),
                    color='red',
                    arrowstyle=patches.ArrowStyle("->", head_length=4, head_width=2),
                    zorder=4,
                )
                ax.add_artist(arr)
                pd['arrows'].append(arr)

            # Time label
            pd['time_txt'].set_text(f'T: {f * pd["time_step"]:.2f} s')

            # Trajectory lines
            for bl in pd['bg_lines']:
                bl.set_data([], [])
            pd['sel_line'].set_data([], [])
            pd['proj_line'].set_data([], [])

            if isinstance(td, dict):
                sel    = td.get('selected_sample')
                proj   = td.get('projection')
                allsmp = td.get('all_samples') or []
                for i, bl in enumerate(pd['bg_lines']):
                    if i + 1 < len(allsmp):
                        s = np.asarray(allsmp[i + 1])
                        bl.set_data(s[:, 0], s[:, 1])
                if sel is not None:
                    sel = np.asarray(sel)
                    pd['sel_line'].set_data(sel[:, 0], sel[:, 1])
                if proj is not None:
                    proj = np.asarray(proj)
                    pd['proj_line'].set_data(proj[:, 0], proj[:, 1])

            # Show outcome label on the final synchronized frame
            if frame_num >= max_frames - 1:
                pd['out_text'].set_visible(True)

        return []   # blit=False — no need to return artist list

    anim = animation.FuncAnimation(
        fig, _update, frames=max_frames, interval=1000 // fps, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={'title': f'style_sweep_scene{scene_idx + 1}'},
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
    )
    anim.save(output_path, writer=writer)
    plt.close(fig)
    logging.info('  Saved scene video: %s', output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Style-sweep evaluation (6 variants, synchronized video)')
    parser.add_argument('--env_config',    type=str,
                        default='configs/env.config')
    parser.add_argument('--policy_config', type=str,
                        default='configs/policy.config')
    parser.add_argument('--policy',        type=str,
                        default='diffusion_conditional_unet1dcfg')
    parser.add_argument('--gpu',           action='store_true', default=False)
    parser.add_argument('--phase',         type=str, default='test')
    parser.add_argument('--num_tests',     type=int, default=50,
                        help='Number of test scenes to evaluate')
    parser.add_argument('--video_file',    type=str,
                        default='style_sweep.mp4')
    parser.add_argument('--fps',           type=int, default=10)
    parser.add_argument('--results_suffix', type=str, default='',
                        help='Optional suffix for the results directory')
    # Scenario types (same as test_evaluate.py)
    parser.add_argument('--square',    action='store_true', default=False)
    parser.add_argument('--circle',    action='store_true', default=False)
    parser.add_argument('--npz_hard',  action='store_true', default=False)
    parser.add_argument('--npz_dir',   type=str,
                        default='random_stage_B_examples_HARDEST')
    parser.add_argument('--npz_num_eval', type=int, default=50)
    parser.add_argument('--map_eval',  action='store_true', default=False)
    parser.add_argument('--map_eval_num', type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu
                          else 'cpu')
    logging.info('Device: %s', device)

    # ── Build and configure the policy ONCE ──────────────────────────────
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)

    # ── Build env and robot ───────────────────────────────────────────────
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    sim = env.unwrapped

    if args.square:
        sim.test_sim = 'square_crossing'
    if args.circle:
        sim.test_sim = 'circle_crossing'
    if args.npz_hard:
        sim.test_sim = 'npz_hard'
    if args.map_eval:
        sim.test_sim = 'map_eval'

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    policy.set_device(device)
    policy.set_env(env)
    robot.print_info()

    # ── Build the scene list ──────────────────────────────────────────────
    if args.npz_hard:
        all_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
        hard_files = []
        for f in all_files:
            try:
                d = np.load(f, allow_pickle=True)
                difficulty = str(d['difficulty']) if 'difficulty' in d else None
                threat     = str(d['threat_type']) if 'threat_type' in d else None
                # Accept diverseset_v4-style "hard" labels AND any
                # hand-crafted scene that carries a threat_type at all
                # (e.g. test_scene_generator2.py's style-probe scenes use
                # their own threat_type vocabulary: head_on, cutoff,
                # sneak_up, group_head_on, composite, ...).
                if (difficulty == 'hard' or threat is not None):
                    hard_files.append(f)
            except Exception as e:
                logging.warning('Could not load %s: %s', f, e)
        if not hard_files:
            raise ValueError(f'No hard npz files found in {args.npz_dir}')
        hard_files = hard_files[:args.npz_num_eval]
        scene_files = hard_files
        num_scenes  = len(hard_files)
        logging.info('NPZ hard: %d scenes', num_scenes)
    elif args.map_eval:
        scene_files = [None] * args.map_eval_num
        num_scenes  = args.map_eval_num
        logging.info('Map-eval: %d scenes', num_scenes)
    else:
        scene_files = [None] * args.num_tests
        num_scenes  = args.num_tests
        logging.info('Standard: %d scenes', num_scenes)

    # ── Results directory ─────────────────────────────────────────────────
    tag         = f'_styles_{args.results_suffix}' if args.results_suffix \
                  else '_styles'
    base_video, ext = os.path.splitext(args.video_file)
    results_dir = f'results{tag}'
    os.makedirs(results_dir, exist_ok=True)
    tmp_dir = os.path.join(results_dir, 'tmp_scenes')
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Per-variant metric accumulators ──────────────────────────────────
    metrics = {label: {
        'successes':   [], 'timeouts':      [], 'collisions': [],
        'wall_colls':  [], 'ep_times':      [], 'path_lens':  [],
        'min_dists':   [], 'avg_min_dists': [],
        # Fixed-scale social metrics for cross-variant comparison
        'prox_violation_frac':   [],
        'mean_min_clearance_m':  [],
        'pass_side_mean':        [],
        'pass_side_consistency': [],
        'headon_rate':           [],
        'cutoff_rate':           [],
        'group_split_score':     [],
    } for label, _, _ in STYLE_VARIANTS}

    all_scene_videos = []
    csv_rows = []   # one row per (scene, variant)

    # ── Scene loop ────────────────────────────────────────────────────────
    for scene_idx in range(num_scenes):
        npz_path = scene_files[scene_idx] if scene_files[scene_idx] else None
        logging.info('Scene %d / %d  (npz=%s)',
                     scene_idx + 1, num_scenes,
                     os.path.basename(npz_path) if npz_path else 'generated')

        # Snapshot the RNG anchor BEFORE any variant touches the env.
        # sim.testoffset increments on every reset() call regardless of
        # test_case, so without restoring it each variant would receive a
        # different seed and thus a completely different scene.
        scene_testoffset = sim.testoffset
        scene_counter    = sim.case_counter[args.phase]

        episodes = []

        for v_idx, (label, style_vec, use_proj) in enumerate(STYLE_VARIANTS):
            logging.info('  Variant %d/%d: %s  style=%s  proj=%s',
                         v_idx + 1, N_VARIANTS, label, style_vec, use_proj)
            ep = run_one_episode(
                env, robot, policy,
                style_vec=style_vec,
                use_proj=use_proj,
                phase=args.phase,
                scene_idx=scene_idx,
                npz_path=npz_path,
                scene_testoffset=scene_testoffset,
                scene_counter=scene_counter,
            )
            episodes.append(ep)

            # Accumulate metrics
            m = metrics[label]
            info = ep['info']
            m['successes'].append(1 if isinstance(info, ReachGoal)      else 0)
            m['timeouts'].append(1   if isinstance(info, Timeout)        else 0)
            m['collisions'].append(1 if isinstance(info, Collision)      else 0)
            try:
                m['wall_colls'].append(1 if isinstance(info, WallCollision) else 0)
            except NameError:
                m['wall_colls'].append(0)
            if isinstance(info, ReachGoal):
                m['ep_times'].append(ep['global_time'])
                m['path_lens'].append(ep['pathlength'])
            m['min_dists'].append(ep['minobsdist'])
            m['avg_min_dists'].append(
                np.mean(ep['avgobsdist']) if ep['avgobsdist'] else float('nan')
            )
            # Social metrics (fixed scale)
            sm = ep['social_metrics']
            for key in ['prox_violation_frac', 'mean_min_clearance_m',
                        'pass_side_mean', 'pass_side_consistency',
                        'headon_rate', 'cutoff_rate', 'group_split_score']:
                m[key].append(sm[key])

            csv_rows.append({
                'scene':      scene_idx + 1,
                'variant':    label,
                'projection': use_proj,
                'style':      str(style_vec),
                'outcome':    _outcome_label(info),
                'time':       f'{ep["global_time"]:.3f}',
                'path_len':   f'{ep["pathlength"]:.3f}',
                'min_dist':   f'{ep["minobsdist"]:.3f}',
                # Social metrics
                'prox_violation_frac':   f'{sm["prox_violation_frac"]:.4f}',
                'mean_min_clearance_m':  f'{sm["mean_min_clearance_m"]:.3f}',
                'pass_side_mean':        f'{sm["pass_side_mean"]:+.3f}',
                'pass_side_consistency': f'{sm["pass_side_consistency"]:.3f}',
                'headon_rate':           f'{sm["headon_rate"]:.4f}',
                'cutoff_rate':           f'{sm["cutoff_rate"]:.4f}',
                'group_split_score':     f'{sm["group_split_score"]:.4f}',
            })

            logging.info('    → %s  t=%.2fs  path=%.2f  minD=%.3f',
                         _outcome_label(info), ep['global_time'],
                         ep['pathlength'], ep['minobsdist'])

        # ── Build synchronized 6-panel video for this scene ──────────────
        scene_video = os.path.join(tmp_dir, f'scene_{scene_idx + 1:04d}{ext}')
        build_scene_video(episodes, scene_idx, scene_video, fps=args.fps)
        all_scene_videos.append(scene_video)

    # ── Concatenate all scene videos ─────────────────────────────────────
    combined_video = os.path.join(results_dir, os.path.basename(args.video_file))
    if all_scene_videos:
        list_file = os.path.join(tmp_dir, 'video_list.txt')
        with open(list_file, 'w') as fh:
            for vid in all_scene_videos:
                fh.write(f"file '{os.path.abspath(vid)}'\n")
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
             '-i', list_file, '-c', 'copy', combined_video],
            check=False,
        )
        logging.info('Combined video: %s', combined_video)

    # ── Write per-scene CSV ───────────────────────────────────────────────
    csv_path = os.path.join(results_dir, 'per_scene_metrics.csv')
    fieldnames = ['scene', 'variant', 'projection', 'style',
                  'outcome', 'time', 'path_len', 'min_dist',
                  'prox_violation_frac', 'mean_min_clearance_m',
                  'pass_side_mean', 'pass_side_consistency',
                  'headon_rate', 'cutoff_rate', 'group_split_score']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    logging.info('Per-scene CSV: %s', csv_path)

    # ── Write aggregate summary ───────────────────────────────────────────
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as fh:
        fh.write('=' * 60 + '\n')
        fh.write('Style-Sweep Evaluation Summary\n')
        fh.write(f'Scenes evaluated: {num_scenes}\n')
        fh.write('=' * 60 + '\n\n')

        header = (f"{'Variant':<20} {'SR%':>6} {'TO%':>6} {'CR%':>6} "
                  f"{'WallCR%':>8} {'AvgT':>7} {'AvgPL':>7} "
                  f"{'AvgMinD':>8}\n")
        fh.write(header)
        fh.write('-' * len(header) + '\n')

        for label, _, _ in STYLE_VARIANTS:
            m = metrics[label]
            sr   = 100 * np.mean(m['successes'])   if m['successes']   else float('nan')
            to   = 100 * np.mean(m['timeouts'])    if m['timeouts']    else float('nan')
            cr   = 100 * np.mean(m['collisions'])  if m['collisions']  else float('nan')
            wcr  = 100 * np.mean(m['wall_colls'])  if m['wall_colls']  else float('nan')
            avgt = np.mean(m['ep_times'])           if m['ep_times']    else float('nan')
            avpl = np.mean(m['path_lens'])          if m['path_lens']   else float('nan')
            avmd = np.mean(m['min_dists'])          if m['min_dists']   else float('nan')
            fh.write(f"{label:<20} {sr:6.1f} {to:6.1f} {cr:6.1f} "
                     f"{wcr:8.1f} {avgt:7.3f} {avpl:7.3f} {avmd:8.3f}\n")

        fh.write('\nColumn key:\n')
        fh.write('  SR%    = Success Rate\n')
        fh.write('  TO%    = Timeout Rate\n')
        fh.write('  CR%    = Agent Collision Rate\n')
        fh.write('  WallCR%= Static Map Collision Rate\n')
        fh.write('  AvgT   = Average Time to Goal (successful episodes)\n')
        fh.write('  AvgPL  = Average Path Length to Goal (successful episodes)\n')
        fh.write('  AvgMinD= Average Minimum Distance to Obstacles (all episodes)\n')

        fh.write('\n\nSocial Style Metrics (fixed neutral scale — comparable across variants)\n')
        fh.write('Expected direction per axis:\n')
        fh.write('  prox+  → lower prox_violation_frac, higher mean_min_clearance\n')
        fh.write('  prox-  → higher prox_violation_frac, lower mean_min_clearance\n')
        fh.write('  pass+  → pass_side_mean toward -1 (right-hand traffic)\n')
        fh.write('  pass-  → pass_side_mean toward +1 (left-hand traffic)\n')
        fh.write('  yield+ → lower headon_rate, lower cutoff_rate\n')
        fh.write('  yield- → higher headon_rate, higher cutoff_rate\n')
        fh.write('  group+ → lower group_split_score\n')
        fh.write('  group- → higher group_split_score\n\n')

        soc_header = (
            f"{'Variant':<20} "
            f"{'ProxViol':>9} {'MinClear':>9} "
            f"{'PassMean':>9} {'PassCons':>9} "
            f"{'HeadonR':>8} {'CutoffR':>8} "
            f"{'GrpSplit':>9}\n"
        )
        fh.write(soc_header)
        fh.write('-' * len(soc_header) + '\n')

        for label, _, _ in STYLE_VARIANTS:
            m = metrics[label]
            def _avg(k): return np.nanmean(m[k]) if m[k] else float('nan')
            fh.write(
                f"{label:<20} "
                f"{_avg('prox_violation_frac'):9.4f} "
                f"{_avg('mean_min_clearance_m'):9.3f} "
                f"{_avg('pass_side_mean'):+9.3f} "
                f"{_avg('pass_side_consistency'):9.3f} "
                f"{_avg('headon_rate'):8.4f} "
                f"{_avg('cutoff_rate'):8.4f} "
                f"{_avg('group_split_score'):9.4f}\n"
            )

    logging.info('Summary: %s', summary_path)

    # Print to console as well
    with open(summary_path) as fh:
        print(fh.read())


if __name__ == '__main__':
    main()
