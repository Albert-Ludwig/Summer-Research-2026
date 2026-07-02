# SocialNavDiffusion ROS Wrapper Audit Package

## A. What This Package Is

This package is a ROS 2 wrapper audit bundle for SocialNavDiffusion running with Clearpath Jackal J100 in Gazebo/HuNavSim.

It is intended for inspecting how the ROS wrapper builds model inputs, calls `policy.predict()`, converts `ActionRot` output into `cmd_vel`, and logs raw/debug traces.

## B. Main Files

- `social_nav_diffusion_ros/social_nav_diffusion_ros/policy_cmd_vel_node.py`: builds model input from odom/goal/people/map, calls `policy.predict()`, converts output to `/cpr_j100_0001/cmd_vel`.
- `social_nav_diffusion_ros/social_nav_diffusion_ros/nav2_goal_to_pose_bridge.py`: converts RViz/Nav2 `NavigateToPose` goals into `/goal_pose` and `/cpr_j100_0001/goal_pose`.
- `social_nav_diffusion_ros/config/raw_eval.yaml`: raw model evaluation parameters.
- `social_nav_diffusion_ros/config/guarded_eval.yaml`: guarded evaluation parameters with sign-conflict guard enabled.
- `SocialNavDiffusion_configs/policy.config`: original SocialNavDiffusion policy config.
- `SocialNavDiffusion_configs/env.config`: original SocialNavDiffusion environment config.
- `runtime_params_used.txt`: exact raw_eval runtime parameters used for the current audit context.
- `model_paths_used.txt`: checkpoint/norm/SND root paths. The checkpoint was not copied.

## C. Important Debug Fields

Robot state:

- `robot_map_x`
- `robot_map_y`
- `robot_yaw`
- `odom_linear_velocity_used_for_sync`
- `odom_angular_velocity_used_for_sync`

Goal state:

- `received_goal_x`
- `received_goal_y`
- `goal_x_robot_frame`
- `goal_y_robot_frame`
- `distance_to_goal`
- `heading_error`

Model/action output:

- `raw_action_type`
- `raw_model_v_before_conversion`
- `raw_model_r_or_w_before_conversion`
- `converted_cmd_linear`
- `converted_cmd_angular`
- `raw_cmd_linear`
- `raw_cmd_angular`
- `final_cmd_linear`
- `final_cmd_angular`

Guard/safety fields:

- `enable_sign_conflict_guard`
- `sign_conflict_guard_active`
- `command_source`

## D. Current Test Conclusion

After simulator-consistency fixes, raw_eval improved:

- no-motion front/left/right tests passed
- forward raw_eval motion succeeded
- side/back or large-heading-error raw_eval motion still failed due to sign conflicts

Successful forward case:

- distance_to_goal decreased approximately:
  `2.237 -> 1.862 -> 1.469 -> 1.081 -> 0.698 -> 0.326`
- `command_source` became `zero_goal_reached`

Failure examples:

- `heading_error=0.427`, `raw_cmd_angular=-0.800`
- `heading_error=1.395`, `raw_cmd_angular=-0.800`
- `heading_error=-0.650`, `raw_cmd_angular=0.800`

Debug CSV availability:

Missing requested CSV debug cases:
- /tmp/snd_raw_eval_side_spinning_failure.csv
- /tmp/snd_guarded_eval_consistency_motion.csv

## E. Open Audit Questions

Please help determine whether:

1. The robot state and goal state passed into the model match the original simulation convention.
2. The goal frame / robot frame / map frame conversion is correct.
3. The `ActionRot` output conversion is correct.
4. The previous-action or warm-start sync is consistent with training/simulation.
5. The runtime parameters match training assumptions.
6. The acados projection layer output is exposed somewhere. The wrapper currently logs the returned policy action, but may not expose all internal diffusion/acados intermediate values.
