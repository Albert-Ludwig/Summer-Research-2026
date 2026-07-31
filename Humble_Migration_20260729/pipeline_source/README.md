# SocialNavDiffusion ROS Wrapper for Jackal

This repo is a small ROS 2 Jazzy overlay for testing SocialNavDiffusion as a local policy on a Clearpath Jackal workflow.

The wrapper does this pipeline:

```text
ROS topics -> model state -> SocialNavDiffusion inference -> acados/projection -> TwistStamped cmd
```

The repo is meant to be cloned inside RoboHub Docker, a local Gazebo workspace, or a Jackal onboard overlay workspace.

## What Is Included

```text
social_nav_diffusion_ros/
  policy_cmd_vel_node.py          # main policy wrapper
  nav2_goal_to_pose_bridge.py     # persistent Nav2 goal bridge
  social_nav_diffusion_node.py    # debug node

config/
  raw_eval.yaml                   # raw model evaluation config
  guarded_eval.yaml               # raw model plus sign-conflict guard
  topics.yaml                     # original local topic config
  topics_sim.yaml                 # Clearpath/Gazebo sim topics
  topics_jackal1.yaml             # Jackal debug topics, publishes /debug_cmd_vel
  safety_limits.yaml              # conservative real-robot debug limits

launch/
  jackal_pipeline.launch.py       # sim/local policy launch
  jackal_onboard_debug.launch.py  # onboard debug launch, safe output topic

scripts/
  run_single_step.py              # offline one-step model check

experiment_setup/hunav/
  scenarios/office_2_agents.yaml
  behavior_trees/office_2_agents__agent_1_bt.xml
  behavior_trees/office_2_agents__agent_2_bt.xml

dependencies.repos                # TODO placeholders for external repos
requirements.txt                  # minimal Python note for gym==0.26.2
```

## What Is Not Included

These are external dependencies. They are not vendored here.

```text
RoboHub Docker / uw_jackal
Clearpath packages
Gazebo installation
HuNavSim source
acados
/workspace/SocialNavDiffusion_Inference
model checkpoint weights
norm stats files
```

Expected model path:

```text
/workspace/SocialNavDiffusion_Inference
```

Expected checkpoint:

```text
/workspace/SocialNavDiffusion_Inference/ckpt_step478000_SOCIAL_NORMS8.pt
```

Expected norm stats:

```text
/workspace/SocialNavDiffusion_Inference/norm_stats_SOCIAL_NORMS8.npy
```

## Main ROS Nodes

```text
policy_cmd_vel_node
nav2_goal_to_pose_bridge
social_nav_diffusion_node
```

`policy_cmd_vel_node` subscribes to goal, odom, people, TF, and map data. It publishes `geometry_msgs/msg/TwistStamped` commands.

`nav2_goal_to_pose_bridge` accepts RViz/Nav2 `NavigateToPose` goals and republishes them as `/goal_pose` topics.

## Build

From a ROS 2 workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select social_nav_diffusion_ros
source install/setup.bash
```

## Offline Single-Step Test

This does not start ROS topics, Gazebo, or the robot.

```bash
cd <workspace>/src/social_nav_diffusion_ros
source /opt/ros/jazzy/setup.bash
python3 scripts/run_single_step.py
```

The script will use `/workspace/SocialNavDiffusion_Inference/.venv/bin/python` if that venv exists. It also sets `ACADOS_SOURCE_DIR=/home/ubuntu/acados` and adds `/home/ubuntu/acados/lib` before acados loads shared libraries.

The CrowdSim code expects the legacy `gym` package. The known working version is:

```text
gym==0.26.2
```

To rebuild the model venv in a RoboHub container:

```bash
cd /workspace/SocialNavDiffusion_Inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r /path/to/social_nav_diffusion_ros/requirements.txt

export ACADOS_SOURCE_DIR=/home/ubuntu/acados
export LD_LIBRARY_PATH=/home/ubuntu/acados/lib:$LD_LIBRARY_PATH
python -m pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
python -m pip install -e /workspace/SocialNavDiffusion_Inference
```

Use this test first to check model loading, config values, raw model output, projected trajectory, and final action conversion.

## Tested SocialNavDiffusion Speedup Update

External inference repo tested:

```text
/workspace/SocialNavDiffusion_Inference
```

Upstream commit applied from colleague repo:

```text
c785225ee545d79ea63fa06ca8a025b3e5a536ed
```

Updated external file:

```text
crowd_nav/policy/diffusion_CondUNetCFG.py
```

Default config used by the single-step test:

```text
crowd_nav/configs/policy.config
num_inference_steps = 5
```

Single-step result on this CPU-only test machine:

```text
conditioning: passed
inference: passed
acados/projection: passed
final action: ActionRot(v=0.374901, r=-0.101623)
final cmd: linear.x=0.374901, angular.z=-0.406493
DDIM sampling: 0:02:06.308479
total shell time: real 3m8.440s
```

Note: this speedup update enables `torch.compile` for the UNet. On this CPU-only single-step smoke test, first-run compile overhead is large. GPU/RoboHub timing should be measured separately. Lower diffusion step counts are for speed testing and may affect policy quality.

## Local / Gazebo Simulation Launch

Build and source the workspace first.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch social_nav_diffusion_ros jackal_pipeline.launch.py \
  params_file:=$(ros2 pkg prefix social_nav_diffusion_ros)/share/social_nav_diffusion_ros/config/raw_eval.yaml \
  topics_file:=$(ros2 pkg prefix social_nav_diffusion_ros)/share/social_nav_diffusion_ros/config/topics_sim.yaml \
  use_sim_time:=true \
  use_diffusion_policy:=true
```

Default sim topics:

```text
people: /people
odom: /cpr_j100_0001/platform/odom/filtered
map: /cpr_j100_0001/map
goal: /goal_pose and /cpr_j100_0001/goal_pose
cmd output: /cpr_j100_0001/cmd_vel
debug: /social_nav_diffusion/policy_debug
marker: /social_nav_diffusion/active_goal_marker
path: /social_nav_diffusion/goal_path
projected path: /social_nav_diffusion/projected_trajectory
raw predicted path: /social_nav_diffusion/predicted_trajectory
```

Open the trajectory debug view with:

```bash
rviz2 -d $(ros2 pkg prefix social_nav_diffusion_ros)/share/social_nav_diffusion_ros/config/social_nav_trajectories.rviz
```

## HuNav Two-Human Scenario

Included scenario files:

```text
experiment_setup/hunav/scenarios/office_2_agents.yaml
experiment_setup/hunav/behavior_trees/office_2_agents__agent_1_bt.xml
experiment_setup/hunav/behavior_trees/office_2_agents__agent_2_bt.xml
```

Copy or symlink them into the matching HuNav package folders if needed:

```text
hunav_gazebo_fortress_wrapper/scenarios
hunav_gazebo_fortress_wrapper/behavior_trees
```

HuNav agent manager should have Groot/ZMQ monitoring disabled by default. This avoids multi-agent BT port conflicts.

## RoboHub Docker Simulation

Follow RoboHub training first.

Typical base flow:

```bash
mkdir -p ~/robohub/jackal
cd ~/robohub/jackal
git clone https://git.uwaterloo.ca/robohub/jackal/uw_jackal.git
./uw_jackal/start.sh
```

Inside Docker:

```bash
ros_local
cp -R uw_jackal/clearpath ./
ros2 launch clearpath_gz simulation.launch.py
```

In another terminal, clone this repo into a ROS workspace:

```bash
mkdir -p ~/jackal_overlay_ws/src
cd ~/jackal_overlay_ws/src
git clone https://git.uwaterloo.ca/Johnson_Ji/jackal_peronal.git social_nav_diffusion_ros
cd ~/jackal_overlay_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select social_nav_diffusion_ros
source install/setup.bash
```

Then launch the policy wrapper with `topics_sim.yaml`.

## Onboard Jackal Debug Mode

Do not run this on hardware until RoboHub training and safety checks are complete.

Default onboard output is:

```text
/debug_cmd_vel
```

It does not publish to the real robot command topic.

Connect and inspect topics first:

```bash
ros_robot jackal1
ros2 topic list
ros2 topic list | grep -E "scan|odom|cmd_vel|tf|people|map|goal"
```

Update `config/topics_jackal1.yaml` after confirming real topic names.

Run debug mode:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch social_nav_diffusion_ros jackal_onboard_debug.launch.py
```

Check output:

```bash
ros2 topic echo /debug_cmd_vel
ros2 topic echo /social_nav_diffusion/policy_debug
```

Only remap to a real command topic after supervisor approval.

## Safety Defaults

`config/safety_limits.yaml` uses conservative limits:

```text
max_linear_speed: 0.05
max_angular_speed: 0.1
cmd_vel_topic: /debug_cmd_vel
```

Before real movement:

```text
confirm real cmd topic
confirm message type
confirm emergency stop
confirm TF frames
confirm odom topic
confirm open test area
confirm human supervision
```

## Useful Checks

```bash
ros2 pkg executables social_nav_diffusion_ros
ros2 topic echo /social_nav_diffusion/policy_debug
ros2 topic echo /people --once
ros2 topic echo /cpr_j100_0001/platform/odom/filtered --once
ros2 topic info /cpr_j100_0001/cmd_vel -v
```

For onboard debug:

```bash
ros2 topic echo /debug_cmd_vel
```

## Known TODOs

```text
Fill TODO versions in dependencies.repos.
Confirm people_msgs in RoboHub Docker.
Confirm Jackal LiDAR, odom, IMU, TF, and cmd_vel topics.
Confirm Jackal namespace and frames.
Confirm real cmd_vel topic with RoboHub staff.
Confirm acados environment setup.
Confirm model checkpoint and norm stats are present.
```

## Notes

Runtime can be slow without GPU.

Model weights are not committed.

Real Jackal tests must start in debug mode.
