# ROS wrapper v1 - fake odom success

Date: 2026-06-18

Status:
- social_nav_diffusion_node starts through ros2 run.
- venv trampoline works.
- running python: /workspace/SocialNavDiffusion_Inference/.venv/bin/python
- policy checkpoint loads.
- acados projection solver builds.
- fake /cpr_j100_0001/platform/odom triggers policy.predict().
- /social_nav_diffusion/debug_action publishes ActionRot.
- /social_nav_diffusion/debug_trajectory publishes visualization_msgs/msg/Marker.
- node is debug-only and does not publish /cmd_vel.

Observed debug_action:
- action type: ActionRot
- v: about 1.0
- r: about 0.006-0.007
- people count: 0
- used_projection: True or truncated in debug string

Observed debug_trajectory:
- Marker type: LINE_STRIP
- frame_id: map
- points generated from projected trajectory

Next:
- Test with real Gazebo/Jackal odom publisher.
- Test with /people publisher.
- Only after stable debug validation, add optional /cmd_vel publishing behind a safety parameter.
