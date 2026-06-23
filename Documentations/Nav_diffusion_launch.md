# Nav diffusion launch

## Rule

- Numbered terminals = daemon / long-running processes.
- Test terminal = one-shot checks only.
- Do not run test commands inside daemon terminals.
- SocialNavDiffusion node does **not** publish `/cmd_vel` yet. It only publishes debug topics.

## Current working daemon order

1. Terminal 1: HuNav + Gazebo
2. Terminal 5: `/clock` bridge
3. Terminal 2: Spawn Jackal
4. Terminal 7: TF repair
5. Terminal 4: SLAM
6. Terminal 3: Nav2
7. Terminal 9: SocialNavDiffusion node
8. Terminal 6: RViz
9. Test terminal: checks only

> If Terminal 3 is already used by SocialNavDiffusion, run Nav2 in any free numbered terminal instead.

---

## Common ROS env block

Use this in most terminals after sourcing ROS/workspace:

```bash
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS
```

---

## Terminal 1: HuNav + Gazebo

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

export HUNAV_SHARE="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper"
export HUNAV_LIB="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/lib"

export GZ_SIM_SYSTEM_PLUGIN_PATH="$HUNAV_LIB${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GAZEBO_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"

rm -f run_hunav_cold_start.log

ros2 launch hunav_gazebo_fortress_wrapper simulation_fortress.launch.py \
  environment_name:=office_no_sensors \
  configuration_file:=office_random_3_agents.yaml \
  robot_name:=cpr_j100_0001/robot \
  use_gazebo_obs:=false \
  use_navgoal_to_start:=false \
  global_frame_to_publish:=map \
  ignore_models:='ground_plane sun charge_dock office link visual collision surface base_link fixed_joint lump lidar chassis fender bracket sensor wheel cpr_j100_0001 robot' \
  update_rate:=20.0 \
  verbose:=false \
  2>&1 | tee run_hunav_cold_start.log
```

Keep running.

---

## Terminal 5: `/clock` bridge

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

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

Keep running.

---

## Terminal 2: Spawn Jackal

```bash
source /opt/ros/jazzy/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

rm -f ~/hunav_jazzy_ws/spawn_jackal_cold_start.log

ros2 launch clearpath_gz robot_spawn.launch.py \
  setup_path:=$HOME/clearpath \
  world:=office \
  use_sim_time:=true \
  generate:=true \
  rviz:=false \
  x:=0.0 \
  y:=0.0 \
  z:=0.30 \
  yaw:=0.0 \
  2>&1 | tee ~/hunav_jazzy_ws/spawn_jackal_cold_start.log
```

Keep running.

---

## Terminal 7: TF repair

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

python3 - <<'PY'
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

ROBOT_NS = "/cpr_j100_0001"
ODOM_TOPIC = f"{ROBOT_NS}/platform/odom"

rclpy.init()
node = rclpy.create_node("jackal_tf_repair_v2")

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

print("jackal_tf_repair_v2 running.")
print("Keep this terminal running.")

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass

node.destroy_node()
rclpy.shutdown()
PY
```

Keep running.

---

## Terminal 4: SLAM

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

rm -f slam_toolbox_debug.log

ros2 launch clearpath_nav2_demos slam.launch.py \
  use_sim_time:=true \
  setup_path:=$HOME/clearpath \
  2>&1 | tee slam_toolbox_debug.log
```

Keep running.

---

## Terminal 3: Nav2

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

rm -f nav2_debug.log

stdbuf -oL -eL ros2 launch clearpath_nav2_demos nav2.launch.py \
  use_sim_time:=true \
  setup_path:=$HOME/clearpath \
  2>&1 | tee nav2_debug.log
```

Wait for:

```text
Managed nodes are active
```

Keep running.

---

## Test terminal: required pre-check before Terminal 9

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

echo "===== /clock ====="
timeout 5s ros2 topic echo /clock --once >/dev/null \
  && echo "PASS: /clock active" \
  || echo "FAIL: /clock inactive"

echo
echo "===== /people ====="
timeout 10s ros2 topic echo /people --once | grep -E "name:|^[[:space:]]+x:|^[[:space:]]+y:" \
  || echo "FAIL: /people no sample"

echo
echo "===== odom ====="
timeout 10s ros2 topic echo /cpr_j100_0001/platform/odom --once >/dev/null \
  && echo "PASS: odom active" \
  || echo "FAIL: odom inactive"

echo
echo "===== odom info ====="
ros2 topic info /cpr_j100_0001/platform/odom --verbose | grep -E "Publisher count:|Subscription count:|Node name:|Topic type:|Endpoint type:" -A1 || true
```

Do not start Terminal 9 until `/clock`, `/people`, and `/cpr_j100_0001/platform/odom` are active. If Terminal 9 was started before odom/people were active, restart Terminal 9 after the inputs recover.

---

## Terminal 9: SocialNavDiffusion node

Notes:

- Do **not** manually activate the venv here. The node trampoline switches to:

```text
/workspace/SocialNavDiffusion_Inference/.venv/bin/python
```

- Start Terminal 9 only after the pre-check passes.
- If Terminal 9 only prints `waiting for /cpr_j100_0001/platform/odom`, stop it and fix odom first.
- If Terminal 9 was already running while odom or people were missing, restart Terminal 9 after inputs recover.

```bash
cd /home/ubuntu/hunav_jazzy_ws

deactivate 2>/dev/null || true

source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

ros2 run social_nav_diffusion_ros social_nav_diffusion_node
```

Expected debug output:

```text
people count: 3
used_projection: True
```

Keep running.

---

## Terminal 6: RViz

Use Clearpath RViz instead of blank `rviz2`:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

ros2 launch clearpath_viz view_navigation.launch.py \
  namespace:=cpr_j100_0001 \
  use_sim_time:=true
```

RViz setup:

```text
Global Options -> Fixed Frame = map

Displays:
- TF
- Map        -> /cpr_j100_0001/map
- LaserScan  -> /cpr_j100_0001/sensors/lidar2d_0/scan
- Marker     -> /social_nav_diffusion/debug_trajectory

SocialNavDiffusion Marker:
- Add display type: Marker, not MarkerArray.
- Topic: /social_nav_diffusion/debug_trajectory.
- If the topic does not appear in the dropdown, manually type /social_nav_diffusion/debug_trajectory.
- Fixed Frame must be map because the Marker publishes frame_id: map.

Use Nav2 Goal first.
Do not use Publish Point unless Terminal 8 is running.
```

---

## Test terminal: quick health check

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

echo "===== /clock ====="
timeout 5s ros2 topic echo /clock --once >/dev/null \
  && echo "PASS: /clock active" \
  || echo "FAIL: /clock inactive"

echo
echo "===== /people ====="
timeout 5s ros2 topic echo /people --once | grep -E "name:|^[[:space:]]+x:|^[[:space:]]+y:" || true

echo
echo "===== odom ====="
timeout 5s ros2 topic echo /cpr_j100_0001/platform/odom --once >/dev/null \
  && echo "PASS: odom active" \
  || echo "FAIL: odom inactive"

echo
echo "===== lidar ====="
timeout 5s ros2 topic echo /cpr_j100_0001/sensors/lidar2d_0/scan --once >/dev/null \
  && echo "PASS: lidar active" \
  || echo "FAIL: lidar inactive"

echo
echo "===== map ====="
timeout 8s ros2 topic echo /cpr_j100_0001/map --once >/dev/null \
  && echo "PASS: map active" \
  || echo "FAIL: map inactive"

echo
echo "===== Nav2 lifecycle ====="
ros2 lifecycle get /cpr_j100_0001/controller_server || true
ros2 lifecycle get /cpr_j100_0001/bt_navigator || true

echo
echo "===== SocialNavDiffusion endpoints ====="
ros2 topic info /social_nav_diffusion/debug_action --verbose || true
ros2 topic info /social_nav_diffusion/debug_trajectory --verbose || true

echo
echo "===== SocialNavDiffusion debug action ====="
timeout 120s ros2 topic echo /social_nav_diffusion/debug_action std_msgs/msg/String --once \
  || echo "FAIL: no debug_action message in 120s"

echo
echo "===== SocialNavDiffusion trajectory marker ====="
timeout 120s ros2 topic echo /social_nav_diffusion/debug_trajectory visualization_msgs/msg/Marker --once \
  | grep -E "frame_id:|ns:|id:|type:|action:|points:|- x:|  y:" \
  | head -60
```

Expected successful output:

```text
people count: 3
used_projection: True
frame_id: map
type: 4
points:
```

---

## If RViz does not show the SocialNavDiffusion Marker

### A. Confirm the Marker publisher exists

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

ros2 topic info /social_nav_diffusion/debug_trajectory --verbose || true
ros2 topic info /social_nav_diffusion/debug_action --verbose || true
```

Interpretation:

- Publisher count: 1 means SocialNavDiffusion has created the debug publishers.
- Publisher count: 0 or Unknown topic means Terminal 9 is not publishing or ROS graph discovery is broken.

### B. Confirm SocialNavDiffusion is subscribed to both inputs

```bash
ros2 topic info /cpr_j100_0001/platform/odom --verbose | grep -E "Publisher count:|Subscription count:|Node name:|Topic type:|Endpoint type:" -A1
ros2 topic info /people --verbose | grep -E "Publisher count:|Subscription count:|Node name:|Topic type:|Endpoint type:" -A1
```

Expected:

- `/cpr_j100_0001/platform/odom` has Publisher count: 1.
- `/people` has Publisher count: 1.
- `social_nav_diffusion_node` appears as a SUBSCRIPTION under both topics.

### C. Confirm actual Marker messages

```bash
timeout 120s ros2 topic echo /social_nav_diffusion/debug_action std_msgs/msg/String --once
timeout 120s ros2 topic echo /social_nav_diffusion/debug_trajectory visualization_msgs/msg/Marker --once \
  | grep -E "frame_id:|ns:|id:|type:|action:|points:|- x:|  y:" \
  | head -60
```

### D. If publisher exists but no messages arrive

- Check Terminal 9.
- If it is waiting for `/cpr_j100_0001/platform/odom`, fix odom first.
- Verify:

```bash
timeout 10s ros2 topic echo /cpr_j100_0001/platform/odom --once >/dev/null && echo PASS || echo FAIL
timeout 10s ros2 topic echo /people --once >/dev/null && echo PASS || echo FAIL
```

- Restart Terminal 9 after odom and people are active.

### E. If even a simple ROS publisher is invisible

The ROS graph / DDS discovery is stale or broken. Run this CLI probe:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
unset FASTDDS_BUILTIN_TRANSPORTS

ros2 daemon stop 2>/dev/null || true
sleep 2

ros2 topic pub -r 1 /social_nav_diffusion/cli_probe std_msgs/msg/String "{data: cli_probe}" \
  > /tmp/cli_probe.log 2>&1 &

echo "PUBPID=$!"
sleep 5

cat /tmp/cli_probe.log

ros2 topic list --no-daemon | grep /social_nav_diffusion/cli_probe || echo "FAIL: topic not listed"
timeout 10s ros2 topic echo /social_nav_diffusion/cli_probe std_msgs/msg/String --once || true
```

Interpretation:

- If `cli_probe` is listed and echo receives data, ROS graph is healthy.
- If `cli_probe` publishes but cannot be listed or echoed, restart the container.

### F. Container restart recovery

If the CLI probe fails, restart the container from Windows PowerShell:

```powershell
docker restart fc4c042f675c
```

After restart, do not start Gazebo/Nav2 immediately. First run the CLI probe again. Only continue with the normal launch order after the CLI probe passes.

---

## Test terminal: set Nav2 goal tolerance

Run after Nav2 says `Managed nodes are active`.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

python3 - <<'PY'
import rclpy
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient

NODE_NAME = "/cpr_j100_0001/controller_server"

rclpy.init()
node = rclpy.create_node("set_nav2_goal_tolerance")
client = AsyncParameterClient(node, NODE_NAME)

print(f"Waiting for parameter service: {NODE_NAME}")
if not client.wait_for_services(timeout_sec=8.0):
    print("FAIL: controller_server parameter service not available.")
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

future = client.set_parameters([
    Parameter("general_goal_checker.xy_goal_tolerance", Parameter.Type.DOUBLE, 0.08),
    Parameter("general_goal_checker.yaw_goal_tolerance", Parameter.Type.DOUBLE, 0.25),
])

rclpy.spin_until_future_complete(node, future, timeout_sec=8.0)

if future.result() is not None:
    for result in future.result().results:
        print(f"successful={result.successful}, reason={result.reason}")

node.destroy_node()
rclpy.shutdown()
PY
```

---

## Test terminal: check RViz Nav2 Goal topic

Use this if clicking in RViz has no reaction.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

echo "===== goal_pose info ====="
ros2 topic info /cpr_j100_0001/goal_pose

echo
echo "===== waiting for RViz Nav2 Goal click ====="
timeout 30s ros2 topic echo /cpr_j100_0001/goal_pose --once
```

If this does not print a `PoseStamped`, RViz is not publishing the goal to the correct namespaced topic.

---

## Optional Terminal 8: Publish Point -> Nav2 action

Only use this after `Nav2 Goal` works. This converts RViz `Publish Point` clicks into `/cpr_j100_0001/navigate_to_pose` action goals.
