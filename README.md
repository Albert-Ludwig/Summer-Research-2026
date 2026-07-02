# SocialNavDiffusion ROS Wrapper for Jackal

## Overview

This repository contains the ROS 2 wrapper used to integrate SocialNavDiffusion as a local policy for Clearpath Jackal/Gazebo/HuNavSim.

The wrapper subscribes to robot odometry, goal poses, HuNav people, and optionally map data, converts those ROS inputs into the SocialNavDiffusion policy state, runs inference, and publishes `TwistStamped` commands for the Jackal command chain.

## Repository Contents

- `social_nav_diffusion_ros/`: ROS 2 Python package for the wrapper.
- `social_nav_diffusion_ros/social_nav_diffusion_ros/policy_cmd_vel_node.py`: main policy control node.
- `social_nav_diffusion_ros/social_nav_diffusion_ros/nav2_goal_to_pose_bridge.py`: persistent `NavigateToPose` action bridge for RViz/Nav2-style goal testing.
- `social_nav_diffusion_ros/config/raw_eval.yaml`: raw model evaluation configuration with optional planner-altering safety layers disabled.
- `social_nav_diffusion_ros/config/guarded_eval.yaml`: guarded evaluation configuration with the sign-conflict guard enabled.
- `docs/`: runtime parameters, model path references, audit notes, and copied SocialNavDiffusion config files.
- `debug_cases/`: available debug CSV traces from wrapper testing.

## External Model Dependency

Model weights are not included in this repository.

Expected SocialNavDiffusion root:

```text
/workspace/SocialNavDiffusion_Inference
```

Expected checkpoint path:

```text
/workspace/SocialNavDiffusion_Inference/SocialGuidedNavPlanner.pt
```

Expected norm stats path:

```text
/workspace/SocialNavDiffusion_Inference/norm_stats_SOCIAL_NORMS8.npy
```

The repository includes config references only:

```text
docs/SocialNavDiffusion_configs/policy.config
docs/SocialNavDiffusion_configs/env.config
docs/model_paths_used.txt
```

## Current Tested Environment

- ROS 2 Jazzy
- Gazebo
- Clearpath Jackal J100 simulation
- HuNavSim
- Current robot namespace: `/cpr_j100_0001`

## Important Runtime Parameters

Key parameters used in the current wrapper tests:

```text
robot_v_pref=1.0
robot_radius=0.25
human_radius=0.25
policy_time_step=0.25
diffusion_inference_period_sec=0.25
cmd_publish_period_sec=0.1
max_linear_speed=0.4
max_angular_speed=0.8
sync_policy_warm_start_from_odom=true
sync_prev_action_from_odom=true
```

`max_angular_speed=0.8` was tested, but the collaborator indicated the original simulator may have used `pi` rad/s. This mismatch should be audited.

Current `ActionRot` conversion:

```text
linear.x = ActionRot.v
angular.z = ActionRot.r / policy.time_step
```

With the observed policy time step:

```text
policy.time_step=0.25
```

## Current Test Conclusions

After simulator-consistency fixes, `raw_eval` improved:

- no-motion front/left/right tests passed
- forward `raw_eval` motion succeeded
- side/back or large-heading-error `raw_eval` motion still showed sign-conflict/spinning issues

Representative success:

```text
distance_to_goal decreased approximately:
2.237 -> 1.862 -> 1.469 -> 1.081 -> 0.698 -> 0.326

final command_source=zero_goal_reached
```

Representative failure examples:

```text
heading_error=0.427, raw_cmd_angular=-0.800
heading_error=1.395, raw_cmd_angular=-0.800
heading_error=-0.650, raw_cmd_angular=0.800
```

These cases suggest that when the target is side/back or the heading error is large, the first raw angular commands can conflict with the geometric heading direction.

## Build Instructions

Clone this repository into a ROS 2 workspace `src/` folder, then build:

```bash
source /opt/ros/jazzy/setup.bash
cd <ros2_workspace>
colcon build --symlink-install --packages-select social_nav_diffusion_ros
source install/setup.bash
```

## Run Example

Example raw evaluation launch command for the policy node:

```bash
ros2 run social_nav_diffusion_ros policy_cmd_vel_node \
  --ros-args \
  --params-file <ros2_workspace>/src/Summer-Research-2026/social_nav_diffusion_ros/config/raw_eval.yaml \
  -p use_sim_time:=true \
  -p use_diffusion_policy:=true
```

The wrapper expects the current Jackal test topics:

```text
/people
/cpr_j100_0001/platform/odom
/cpr_j100_0001/platform/odom/filtered
/goal_pose
/cpr_j100_0001/goal_pose
/cpr_j100_0001/cmd_vel
```

## RoboHub Adaptation Notes

In RoboHub Jackal Docker, clone this repo into a ROS workspace `src/` folder, build it, source `install/setup.bash`, and adapt topic namespaces from `/cpr_j100_0001` to `/jackal1` or whichever robot namespace is active.

The angular velocity limit should be retested with `pi` rad/s if that was the original training/simulation limit.

## Open Audit Questions

Please audit:

- robot/map/goal frame conventions
- goal conversion into model state
- human state conversion
- `ActionRot` conversion
- warm-start / previous-action sync
- angular velocity limit mismatch, `0.8` vs `pi` rad/s
- acados projection output logging availability

