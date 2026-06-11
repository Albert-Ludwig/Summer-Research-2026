# Minimal Cold-Start Guide: HuNav + Jackal + Gazebo Harmonic

> Goal: start the full HuNav + Jackal + Gazebo Harmonic setup from a clean reboot / cold start, then run a minimal health check.

---

## 0. Terminal roles

| Terminal   | Purpose                                   |
| ---------- | ----------------------------------------- |
| Terminal 1 | Background full HuNav + Gazebo simulation |
| Terminal 2 | Background Jackal spawn / controllers     |
| Terminal 3 | Nav2 launch or tuning commands            |
| Terminal 4 | SLAM launch                               |
| Terminal 5 | `/clock` bridge                           |
| Terminal 6 | RViz                                      |
| Terminal 7 | TF repair daemon                          |

---

## 1. Test terminal: Optional clean-state reset

Use this first if the machine was not cleanly rebooted, or if Gazebo / ROS processes may still be running.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws

echo "===== CLEAN OLD SIMULATION PROCESSES ====="

pkill -INT -f '[s]imulation_fortress.launch.py' 2>/dev/null || true
pkill -INT -f '[r]obot_spawn.launch.py' 2>/dev/null || true
pkill -INT -f '[p]arameter_bridge' 2>/dev/null || true
pkill -INT -f '[g]z sim' 2>/dev/null || true
pkill -INT -f '[c]pr_j100_0001' 2>/dev/null || true
pkill -INT -f '[h]unav_' 2>/dev/null || true

sleep 3

pkill -KILL -f '[s]imulation_fortress.launch.py' 2>/dev/null || true
pkill -KILL -f '[r]obot_spawn.launch.py' 2>/dev/null || true
pkill -KILL -f '[p]arameter_bridge' 2>/dev/null || true
pkill -KILL -f '[g]z sim' 2>/dev/null || true
pkill -KILL -f '[c]pr_j100_0001' 2>/dev/null || true
pkill -KILL -f '[h]unav_' 2>/dev/null || true

ros2 daemon stop
sleep 2
ros2 daemon start
sleep 2

echo "===== CLEAN STATE CHECK ====="
pgrep -af "gz sim|hunav|robot_spawn|parameter_bridge|cpr_j100" \
  || echo "PASS: No previous simulation-related processes remain."
```

---

## 2. Terminal 1: Start HuNav + Gazebo

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

export HUNAV_SHARE="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper"
export HUNAV_LIB="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/lib"

export GZ_SIM_SYSTEM_PLUGIN_PATH="$HUNAV_LIB${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GAZEBO_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"

rm -f run_hunav_cold_start.log

ros2 launch hunav_gazebo_fortress_wrapper simulation_fortress.launch.py \
  environment_name:=office_no_sensors \
  configuration_file:=cafe_agents.yaml \
  robot_name:=cpr_j100_0001/robot \
  use_gazebo_obs:=false \
  use_navgoal_to_start:=false \
  global_frame_to_publish:=map \
  ignore_models:='ground_plane sun charge_dock office link visual collision surface base_link fixed_joint lump lidar chassis fender bracket sensor wheel cpr_j100_0001 robot' \
  update_rate:=20.0 \
  verbose:=false \
  2>&1 | tee run_hunav_cold_start.log
```

Keep this terminal running.

---

## 3. Terminal 5: Start the `/clock` bridge

```bash
source /opt/ros/jazzy/setup.bash

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

## 4. Terminal 2: Spawn Jackal

```bash
source /opt/ros/jazzy/setup.bash

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

Wait until you see messages similar to:

```text
Entity creation successful
Configured and activated platform_velocity_controller
Configured and activated joint_state_broadcaster
```

Keep this terminal running.

---

## 5. Test terminal: Minimal health check

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

LOG="run_hunav_cold_start.log"
GEN="install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper/worlds/generatedWorld.sdf"

echo "===== PEOPLE ====="
timeout 5s ros2 topic echo /people --once \
  | grep -E "name:|^[[:space:]]+x:|^[[:space:]]+y:" \
  || echo "FAIL: /people not publishing."

echo
echo "===== CLOCK ====="
timeout 5s ros2 topic echo /clock --once >/dev/null \
  && echo "PASS: /clock active." \
  || echo "FAIL: /clock not active."

echo
echo "===== LIDAR ====="
timeout 5s ros2 topic echo /cpr_j100_0001/sensors/lidar2d_0/scan --once >/dev/null \
  && echo "PASS: LiDAR publishing." \
  || echo "FAIL: LiDAR not publishing."

echo
echo "===== CONTROLLERS ====="
ros2 service call /cpr_j100_0001/controller_manager/list_controllers \
  controller_manager_msgs/srv/ListControllers "{}" \
  | grep -E "joint_state_broadcaster|platform_velocity_controller|active"

echo
echo "===== ODOM ====="
timeout 5s ros2 topic echo /cpr_j100_0001/platform/odom --once \
  --qos-reliability reliable \
  --qos-durability transient_local \
  | sed -n '1,18p' \
  || echo "FAIL: odom not publishing."

echo
echo "===== GENERATED WORLD LEGACY CHECK ====="
grep -nEi 'libignition-gazebo|ignition::gazebo::systems' "$GEN" \
  && echo "FAIL: Legacy Ignition plugins returned." \
  || echo "PASS: generatedWorld is clean."

echo
echo "===== UNKNOWN GEOMETRY COUNT ====="
grep -c "has an unknown geometry" "$LOG" 2>/dev/null || echo "0"
```

Expected passing results:

```text
/people shows agent2
PASS: /clock active.
PASS: LiDAR publishing.
controllers active
odom publishes a message
PASS: generatedWorld is clean.
UNKNOWN GEOMETRY COUNT = 0
```

---

## 6. Test terminal: Optional minimal motion test

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

cat > /tmp/demo_hunav_motion.py <<'PY'
import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TwistStamped

rclpy.init()
node = Node('demo_hunav_motion')
node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

pub = node.create_publisher(
    TwistStamped,
    '/cpr_j100_0001/platform/cmd_vel',
    10
)

deadline = time.monotonic() + 4.0
while rclpy.ok() and node.get_clock().now().nanoseconds == 0 and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)

if node.get_clock().now().nanoseconds == 0:
    node.destroy_node()
    rclpy.shutdown()
    raise RuntimeError('No simulation clock received.')

def publish_cmd(linear_x, angular_z, duration):
    start = time.monotonic()
    while rclpy.ok() and time.monotonic() - start < duration:
        rclpy.spin_once(node, timeout_sec=0.01)
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        pub.publish(msg)
        time.sleep(0.05)

print('Forward 0.10 m/s for 4 seconds')
publish_cmd(0.10, 0.0, 4.0)

print('Stop')
publish_cmd(0.0, 0.0, 1.0)

node.destroy_node()
rclpy.shutdown()
print('Done.')
PY

python3 /tmp/demo_hunav_motion.py
```

---

## 7. RViz + SLAM + Nav2 add-on

Use this only after Terminal 1, Terminal 5, and Terminal 2 are already running.

### Terminal 7: TF repair daemon

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

python3 - <<'PY'
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

ROBOT_NS = "/cpr_j100_0001"
ODOM_TOPIC = f"{ROBOT_NS}/platform/odom"

LIDAR_X = 0.0
LIDAR_Y = 0.0
LIDAR_Z = 0.25

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
    msg.transforms.append(make_static("chassis_link", "lidar2d_0_laser", LIDAR_X, LIDAR_Y, LIDAR_Z))
    tf_static_pub_global.publish(msg)
    tf_static_pub_ns.publish(msg)

def odom_cb(msg: Odometry):
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
print("Publishing dynamic: odom -> base_link")
print("Publishing static: base_link -> chassis_link -> lidar2d_0_laser")
print("Keep this terminal running.")

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass

node.destroy_node()
rclpy.shutdown()
PY
```

### Terminal 4: Start SLAM

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

rm -f slam_toolbox_debug.log

ros2 launch clearpath_nav2_demos slam.launch.py \
  use_sim_time:=true \
  setup_path:=$HOME/clearpath \
  2>&1 | tee slam_toolbox_debug.log
```

Keep this terminal running.

### Terminal 3: Start Nav2

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

rm -f nav2_debug.log

stdbuf -oL -eL ros2 launch clearpath_nav2_demos nav2.launch.py \
  use_sim_time:=true \
  setup_path:=$HOME/clearpath \
  2>&1 | tee nav2_debug.log
```

Wait until this appears:

```text
Managed nodes are active
```

Keep this terminal running.

### Testing terminal: optional smaller Nav2 goal tolerance

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

python3 - <<'PY'
import rclpy
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient

NODE_NAME = "/cpr_j100_0001/controller_server"

PARAMS_TO_GET = [
    "goal_checker_plugins",
    "general_goal_checker.xy_goal_tolerance",
    "general_goal_checker.yaw_goal_tolerance",
    "general_goal_checker.stateful",
]

rclpy.init()
node = rclpy.create_node("set_nav2_goal_tolerance_debug")
client = AsyncParameterClient(node, NODE_NAME)

print(f"Waiting for parameter service: {NODE_NAME}")
if not client.wait_for_services(timeout_sec=8.0):
    print("FAIL: controller_server parameter service not available.")
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

print("\n===== current params =====")
future = client.get_parameters(PARAMS_TO_GET)
rclpy.spin_until_future_complete(node, future, timeout_sec=8.0)

if future.result() is not None:
    for name, value in zip(PARAMS_TO_GET, future.result().values):
        try:
            print(f"{name}: {parameter_value_to_python(value)}")
        except Exception:
            print(f"{name}: {value}")

print("\n===== setting smaller tolerance =====")
set_future = client.set_parameters([
    Parameter("general_goal_checker.xy_goal_tolerance", Parameter.Type.DOUBLE, 0.08),
    Parameter("general_goal_checker.yaw_goal_tolerance", Parameter.Type.DOUBLE, 0.25),
])

rclpy.spin_until_future_complete(node, set_future, timeout_sec=8.0)

if set_future.result() is not None:
    for result in set_future.result().results:
        print(f"successful={result.successful}, reason={result.reason}")

node.destroy_node()
rclpy.shutdown()
PY
```

### Terminal 6: Start RViz

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

ros2 launch clearpath_viz view_navigation.launch.py \
  namespace:=cpr_j100_0001 \
  use_sim_time:=true
```

In RViz:

```text
Global Options -> Fixed Frame = map
Map -> Topic = map
```
