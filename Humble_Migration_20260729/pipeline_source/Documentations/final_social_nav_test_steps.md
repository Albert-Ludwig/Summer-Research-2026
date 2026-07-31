# Final SocialNavDiffusion + Jackal Simulation Test Steps

This document describes the final startup and validation procedure for running the SocialNavDiffusion policy on the Clearpath Jackal simulation without Nav2 controlling the robot.

Goal:

```text
HuNav/Gazebo + Jackal + SLAM + TF repair
+ policy_cmd_vel_node
+ nav2_goal_to_pose_bridge
+ RViz Nav2 Goal or manual /goal_pose
-> SocialNavDiffusion planner
-> /cpr_j100_0001/cmd_vel
-> /cpr_j100_0001/platform/cmd_vel
-> Gazebo Jackal moves
```

Important: do **not** start Nav2 controller/planner for this test.

Do not run:

```bash
ros2 launch clearpath_nav2_demos nav2.launch.py
```

Do not run the old debug node:

```bash
ros2 run social_nav_diffusion_ros social_nav_diffusion_node
```

The only control node should be:

```text
policy_cmd_vel_node
```

The RViz goal bridge is only used to convert RViz Nav2 Goal into `/goal_pose`:

```text
nav2_goal_to_pose_bridge
```

---

## 1. Recommended one-command launcher

A Python launcher was created at:

```bash
/home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test.py
```

If it is not there yet, copy it into the repo and make it executable:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
cp /mnt/data/run_final_social_nav_test.py .
chmod +x run_final_social_nav_test.py
```

Run the full stack:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test.py
```

Run the full stack and automatically send a far goal:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test.py --goal 4.0 0.0
```

Run without RViz and send a command-line goal:

```bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
python3 run_final_social_nav_test.py --no-rviz --goal 4.0 0.0
```

The launcher starts:

```text
HuNav + Gazebo
/clock bridge
Jackal spawn
TF repair
SLAM
policy_cmd_vel_node
nav2_goal_to_pose_bridge
RViz, unless --no-rviz is used
```

Logs are written to:

```bash
/tmp/social_nav_final_test_logs
```

Useful log checks:

```bash
tail -n 80 /tmp/social_nav_final_test_logs/policy_wrapper.log
tail -n 80 /tmp/social_nav_final_test_logs/goal_bridge.log
tail -n 80 /tmp/social_nav_final_test_logs/slam.log
```

To stop everything launched by the Python script, press:

```text
Ctrl+C
```

---

## 2. Manual startup procedure

Use this section if you want to start each component manually in separate terminals.

Before starting, use this shared ROS environment block in every terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
```

---

## 3. Clean old wrapper and bridge processes

Run this before a fresh test:

```bash
source /opt/ros/jazzy/setup.bash

pkill -INT -f '[p]olicy_cmd_vel_node' 2>/dev/null || true
pkill -INT -f '[j]ackal_pipeline.launch.py' 2>/dev/null || true
pkill -INT -f '[n]av2_goal_to_pose_bridge' 2>/dev/null || true

sleep 3

pkill -KILL -f '[p]olicy_cmd_vel_node' 2>/dev/null || true
pkill -KILL -f '[j]ackal_pipeline.launch.py' 2>/dev/null || true
pkill -KILL -f '[n]av2_goal_to_pose_bridge' 2>/dev/null || true

ros2 daemon stop
sleep 2
ros2 daemon start
sleep 2

ros2 node list | grep -Ei "policy|goal|bridge|navigate" || true
```

After cleanup, you should not see:

```text
/policy_cmd_vel_node
/nav2_goal_to_pose_bridge
```

---

## 4. Terminal 1: Start HuNav + Gazebo

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

export HUNAV_SHARE="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper"
export HUNAV_LIB="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/lib"

export GZ_SIM_SYSTEM_PLUGIN_PATH="$HUNAV_LIB${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GAZEBO_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"

ros2 launch hunav_gazebo_fortress_wrapper simulation_fortress.launch.py \
  environment_name:=office_no_sensors \
  configuration_file:=office_2_agents.yaml \
  robot_name:=cpr_j100_0001/robot \
  use_gazebo_obs:=false \
  use_navgoal_to_start:=false \
  global_frame_to_publish:=map \
  ignore_models:='ground_plane sun charge_dock office link visual collision surface base_link fixed_joint lump lidar chassis fender bracket sensor wheel cpr_j100_0001 robot' \
  update_rate:=20.0 \
  verbose:=false
```

Keep this terminal running.

---

## 5. Terminal 2: Start `/clock` bridge

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

if gz topic -l | grep -qx "/clock"; then
  ros2 run ros_gz_bridge parameter_bridge \
    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
elif gz topic -l | grep -qx "/world/office/clock"; then
  ros2 run ros_gz_bridge parameter_bridge \
    '/world/office/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' \
    --ros-args -r /world/office/clock:=/clock
else
  echo "FAIL: No Gazebo clock topic found."
  gz topic -l | grep clock || true
fi
```

Keep this terminal running.

---

## 6. Terminal 3: Spawn Jackal

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

ros2 launch clearpath_gz robot_spawn.launch.py \
  setup_path:=$HOME/clearpath \
  world:=office \
  use_sim_time:=true \
  generate:=true \
  rviz:=false \
  x:=0.0 \
  y:=0.0 \
  z:=0.30 \
  yaw:=0.0
```

Keep this terminal running.

---

## 7. Terminal 4: TF repair

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

python3 - <<'PY'
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

ROBOT_NS = "/cpr_j100_0001"
ODOM_TOPIC = f"{ROBOT_NS}/platform/odom"

rclpy.init()
node = rclpy.create_node("jackal_tf_repair_planner_test")

tf_pub_global = node.create_publisher(TFMessage, "/tf", 100)
tf_pub_ns = node.create_publisher(TFMessage, f"{ROBOT_NS}/tf", 100)

static_qos = QoSProfile(depth=10)
static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
static_qos.reliability = ReliabilityPolicy.RELIABLE

tf_static_pub_global = node.create_publisher(TFMessage, "/tf_static", static_qos)
tf_static_pub_ns = node.create_publisher(TFMessage, f"{ROBOT_NS}/tf_static", static_qos)

def make_static(parent, child, x=0.0, y=0.0, z=0.0):
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    t.transform.rotation.w = 1.0
    return t

def publish_static():
    msg = TFMessage()
    msg.transforms.append(make_static("base_link", "chassis_link"))
    msg.transforms.append(make_static("chassis_link", "lidar2d_0_laser", 0.0, 0.0, 0.25))
    tf_static_pub_global.publish(msg)
    tf_static_pub_ns.publish(msg)

def odom_cb(msg):
    t = TransformStamped()
    t.header.stamp = msg.header.stamp
    t.header.frame_id = msg.header.frame_id if msg.header.frame_id else "odom"
    t.child_frame_id = msg.child_frame_id if msg.child_frame_id else "base_link"
    t.transform.translation.x = msg.pose.pose.position.x
    t.transform.translation.y = msg.pose.pose.position.y
    t.transform.translation.z = msg.pose.pose.position.z
    t.transform.rotation = msg.pose.pose.orientation
    out = TFMessage()
    out.transforms.append(t)
    tf_pub_global.publish(out)
    tf_pub_ns.publish(out)

node.create_subscription(Odometry, ODOM_TOPIC, odom_cb, 50)
node.create_timer(1.0, publish_static)
publish_static()

print("jackal_tf_repair_planner_test running.")
print("Keep this terminal running.")

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass

node.destroy_node()
rclpy.shutdown()
PY
```

Keep this terminal running.

---

## 8. Terminal 5: Start SLAM

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

ros2 launch clearpath_nav2_demos slam.launch.py \
  use_sim_time:=true \
  setup_path:=$HOME/clearpath
```

Keep this terminal running.

---

## 9. Test terminal: Basic sensor/topic checks

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

echo "===== /clock ====="
timeout 5s ros2 topic echo /clock rosgraph_msgs/msg/Clock --once >/dev/null \
  && echo "PASS: /clock active" \
  || echo "FAIL: /clock inactive"

echo "===== /people ====="
timeout 10s ros2 topic echo /people people_msgs/msg/People --once >/dev/null \
  && echo "PASS: /people active" \
  || echo "FAIL: /people inactive"

echo "===== odom ====="
timeout 10s ros2 topic echo /cpr_j100_0001/platform/odom/filtered --once >/dev/null \
  && echo "PASS: filtered odom active" \
  || echo "WARN: filtered odom inactive"

echo "===== map ====="
timeout 10s ros2 topic echo /cpr_j100_0001/map nav_msgs/msg/OccupancyGrid --once >/dev/null \
  && echo "PASS: map active" \
  || echo "FAIL: map inactive"

echo "===== lidar ====="
timeout 10s ros2 topic echo /cpr_j100_0001/sensors/lidar2d_0/scan --once >/dev/null \
  && echo "PASS: lidar active" \
  || echo "FAIL: lidar inactive"
```

---

## 10. Test terminal: Check namespaced TF

```bash
timeout 5s ros2 run tf2_ros tf2_echo map base_link \
  --ros-args \
  -r /tf:=/cpr_j100_0001/tf \
  -r /tf_static:=/cpr_j100_0001/tf_static
```

It is acceptable if the first line says the frame is temporarily unavailable, as long as transforms start printing afterwards.

---

## 11. Terminal 6: Start policy wrapper

This is the main control node.

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
export SOCIAL_NAV_DIFFUSION_USE_VENV=true

ros2 launch social_nav_diffusion_ros jackal_pipeline.launch.py \
  params_file:=/home/ubuntu/waterloo_jackal_pipeline_repo/install/social_nav_diffusion_ros/share/social_nav_diffusion_ros/config/raw_eval.yaml \
  topics_file:=/home/ubuntu/waterloo_jackal_pipeline_repo/install/social_nav_diffusion_ros/share/social_nav_diffusion_ros/config/topics_sim.yaml \
  use_sim_time:=true \
  use_diffusion_policy:=true
```

Keep this terminal running.

Expected signs:

```text
/workspace/SocialNavDiffusion_Inference/.venv/bin/python
Device: cuda
policy_cmd_vel_node running
```

---

## 12. Terminal 7: Start RViz goal bridge

This is not Nav2 planner. It only converts RViz Nav2 Goal into `/goal_pose`.

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

ros2 run social_nav_diffusion_ros nav2_goal_to_pose_bridge --ros-args \
  --params-file /home/ubuntu/waterloo_jackal_pipeline_repo/install/social_nav_diffusion_ros/share/social_nav_diffusion_ros/config/topics_sim.yaml
```

Keep this terminal running.

---

## 13. Terminal 8: Start RViz

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

ros2 launch clearpath_viz view_navigation.launch.py \
  namespace:=cpr_j100_0001 \
  use_sim_time:=true
```

In RViz, set:

```text
Global Options -> Fixed Frame = map
```

Recommended displays:

```text
TF
Map: /cpr_j100_0001/map
LaserScan: /cpr_j100_0001/sensors/lidar2d_0/scan
Path: /social_nav_diffusion/goal_path
Marker: /social_nav_diffusion/active_goal_marker
```

---

## 14. Test terminal: Confirm wrapper and bridge are connected

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

echo "===== nodes ====="
ros2 node list | grep -Ei "policy|goal|bridge|navigate"

echo
echo "===== goal endpoints ====="
ros2 topic info /goal_pose --verbose

echo
echo "===== cmd_vel endpoints ====="
ros2 topic info /cpr_j100_0001/cmd_vel --verbose

echo
echo "===== policy_debug endpoints ====="
ros2 topic info /social_nav_diffusion/policy_debug --verbose
```

Expected:

```text
/policy_cmd_vel_node
/nav2_goal_to_pose_bridge
```

Expected `/goal_pose`:

```text
Publisher = nav2_goal_to_pose_bridge
Subscriber = policy_cmd_vel_node
```

Expected `/cpr_j100_0001/cmd_vel`:

```text
Publisher = policy_cmd_vel_node
Subscriber = twist_mux
```

---

## 15. Final test option A: RViz Nav2 Goal

In RViz, click **Nav2 Goal** and choose a goal at least 1 to 2 meters away from the robot.

Do not click too close to the robot. If the goal is too close, the policy may stop because it considers the goal reached or locally unsafe.

---

## 16. Final test option B: Command-line goal

This is recommended for final validation because it is deterministic.

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
"{header: {frame_id: map}, pose: {position: {x: 4.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

---

## 17. Final validation commands

Run after sending a goal:

```bash
echo "===== policy_debug ====="
ros2 topic echo /social_nav_diffusion/policy_debug --once --full-length

echo "===== policy cmd_vel ====="
timeout 30s ros2 topic echo /cpr_j100_0001/cmd_vel --once

echo "===== platform cmd_vel ====="
timeout 30s ros2 topic echo /cpr_j100_0001/platform/cmd_vel --once
```

Success criteria:

```text
command_source=diffusion or diffusion_projected
raw_action_type=ActionRot
raw_model_v_before_conversion nonzero
converted_cmd_linear nonzero
converted_cmd_angular nonzero
final_cmd_linear nonzero
/cpr_j100_0001/cmd_vel nonzero
/cpr_j100_0001/platform/cmd_vel nonzero
```

---

## 18. Confirm Gazebo Jackal movement

```bash
timeout 15s ros2 topic echo /cpr_j100_0001/platform/odom/filtered \
  | grep -E "position:|orientation:|x:|y:|z:|w:" \
  | head -120
```

Check whether `position.x` or `position.y` changes.

Movement confirms the full chain:

```text
SocialNavDiffusion -> cmd_vel -> twist_mux -> Gazebo Jackal odom
```

---

## 19. If RViz Nav2 Goal does not move the robot

First check whether RViz goal reaches `/goal_pose`:

```bash
ros2 topic echo /goal_pose geometry_msgs/msg/PoseStamped
```

Then click Nav2 Goal in RViz.

If no message appears, the issue is in RViz or `nav2_goal_to_pose_bridge`.

Check:

```bash
ros2 node list | grep -Ei "policy|goal|bridge|navigate"
ros2 action list | grep navigate || true
ros2 topic info /goal_pose --verbose
ros2 topic info /cpr_j100_0001/goal_pose --verbose
```

If `/goal_pose` appears but the robot does not move, check policy debug:

```bash
ros2 topic echo /social_nav_diffusion/policy_debug --once --full-length
```

Important fields:

```text
goal_received=
command_source=
distance_to_goal=
raw_action_type=
raw_model_v_before_conversion=
converted_cmd_linear=
final_cmd_linear=
tf_robot_to_target_success=
tf_goal_to_target_success=
```

Common cases:

```text
command_source=zero_goal_reached
and distance_to_goal < 0.050
=> goal is too close. Send a farther goal.
```

```text
tf_robot_to_target_success=False
or tf_goal_to_target_success=False
=> TF issue. Check namespaced TF with tf2_echo.
```

```text
raw_model_v_before_conversion nonzero
converted_cmd_linear nonzero
final_cmd_linear nonzero
but /platform/cmd_vel is zero
=> twist_mux or command bridge issue.
```

```text
/platform/cmd_vel nonzero
but robot does not move
=> Gazebo may be paused or controller issue.
```

Unpause Gazebo:

```bash
gz service -s /world/office/control \
  --reqtype gz.msgs.WorldControl \
  --reptype gz.msgs.Boolean \
  --req "pause: false"
```

---

## 20. Final milestone statement

Use this as the final test conclusion if it passes:

```text
No-Nav2 simulation closed-loop control validated.
SocialNavDiffusion policy produces nonzero ROS cmd_vel from nonzero ActionRot output.
/cpr_j100_0001/platform/cmd_vel is nonzero.
Gazebo Jackal odom changes, confirming robot motion.
```

---

## 21. Start from inside the container

Run this inside `ros_vnc_jazzy_gpu_full`:

```bash
cp /workspace/Documentations/run_final_social_nav_test.py /home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test.py

chmod +x /home/ubuntu/waterloo_jackal_pipeline_repo/run_final_social_nav_test.py

cd /home/ubuntu/waterloo_jackal_pipeline_repo

python3 run_final_social_nav_test.py
```
