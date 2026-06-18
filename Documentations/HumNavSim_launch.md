# HumNavSim launch

## Rule

Run daemon commands only in numbered terminals. Run non-daemon checks only in **Test terminal**.

## Startup order

1. Terminal 1: HuNav + Gazebo
2. Terminal 5: `/clock` bridge
3. Terminal 2: Jackal spawn
4. Terminal 7: TF repair daemon
5. Terminal 4: SLAM
6. Terminal 3: Nav2
7. Test terminal: goal tolerance / checks
8. Terminal 6: RViz
9. Terminal 8: optional Publish Point single-click navigation

---

## Terminal 1: HuNav + Gazebo

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

ros2 launch hunav_gazebo_fortress_wrapper simulation_fortress.launch.py   environment_name:=office_no_sensors   configuration_file:=cafe_agents.yaml   robot_name:=cpr_j100_0001/robot   use_gazebo_obs:=false   use_navgoal_to_start:=true   global_frame_to_publish:=map   ignore_models:='ground_plane sun charge_dock office link visual collision surface base_link fixed_joint lump lidar chassis fender bracket sensor wheel cpr_j100_0001 robot'   update_rate:=20.0   verbose:=false   2>&1 | tee run_hunav_cold_start.log
```

Keep this terminal running.

---

## Terminal 5: `/clock` bridge

```bash
source /opt/ros/jazzy/setup.bash

if gz topic -l | grep -qx "/clock"; then
  ros2 run ros_gz_bridge parameter_bridge     '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
elif gz topic -l | grep -qx "/world/office/clock"; then
  ros2 run ros_gz_bridge parameter_bridge     '/world/office/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'     --ros-args -r /world/office/clock:=/clock
else
  echo "FAIL: No Gazebo clock topic found."
  gz topic -l | grep clock || true
fi
```

Keep this terminal running.

---

## Terminal 2: Spawn Jackal

```bash
source /opt/ros/jazzy/setup.bash

rm -f ~/hunav_jazzy_ws/spawn_jackal_cold_start.log

ros2 launch clearpath_gz robot_spawn.launch.py   setup_path:=$HOME/clearpath   world:=office   use_sim_time:=true   generate:=true   rviz:=false   x:=0.0   y:=0.0   z:=0.30   yaw:=0.0   2>&1 | tee ~/hunav_jazzy_ws/spawn_jackal_cold_start.log
```

Keep this terminal running.

---

## Terminal 7: TF repair daemon

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

Keep this terminal running.

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

rm -f slam_toolbox_debug.log

ros2 launch clearpath_nav2_demos slam.launch.py   use_sim_time:=true   setup_path:=$HOME/clearpath   2>&1 | tee slam_toolbox_debug.log
```

Keep this terminal running.

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

rm -f nav2_debug.log

stdbuf -oL -eL ros2 launch clearpath_nav2_demos nav2.launch.py   use_sim_time:=true   setup_path:=$HOME/clearpath   2>&1 | tee nav2_debug.log
```

Wait until:

```text
Managed nodes are active
```

Keep this terminal running.

---

## Test terminal: set Nav2 goal tolerance

Run this after Terminal 3 shows `Managed nodes are active`.

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

## Terminal 6: RViz

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

ros2 launch clearpath_viz view_navigation.launch.py   namespace:=cpr_j100_0001   use_sim_time:=true
```

In RViz:

```text
Global Options -> Fixed Frame = map
Map -> Topic = map
```

Use `Nav2 Goal` first.

---

## Terminal 8: Optional Publish Point single-click navigation

Run this only after `Nav2 Goal` works.

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
import math
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener

ROBOT_NS = "/cpr_j100_0001"
CLICKED_TOPICS = [f"{ROBOT_NS}/clicked_point", "/clicked_point"]
ACTION_NAME = f"{ROBOT_NS}/navigate_to_pose"
TARGET_FRAME = "map"
ROBOT_FRAME = "base_link"
MIN_GOAL_DISTANCE = 0.35

def quat_from_yaw(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)

rclpy.init(args=[
    "--ros-args",
    "-r", "/tf:=/cpr_j100_0001/tf",
    "-r", "/tf_static:=/cpr_j100_0001/tf_static",
])

node = rclpy.create_node("publish_point_to_nav2_action_stable")
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node)
client = ActionClient(node, NavigateToPose, ACTION_NAME)

def goal_response_cb(future):
    goal_handle = future.result()
    if goal_handle is None:
        print("FAIL: no goal handle returned.", flush=True)
        return
    if not goal_handle.accepted:
        print("FAIL: goal rejected by Nav2.", flush=True)
        return
    print("PASS: goal accepted by Nav2.", flush=True)

def make_clicked_cb(source_topic):
    def clicked_cb(msg):
        print()
        print("===== Publish Point clicked =====", flush=True)
        print(f"source topic: {source_topic}", flush=True)
        print(f"clicked frame: {msg.header.frame_id}", flush=True)
        print(f"clicked x={msg.point.x:.3f}, y={msg.point.y:.3f}", flush=True)

        if not client.server_is_ready():
            if not client.wait_for_server(timeout_sec=10.0):
                print("FAIL: NavigateToPose action server unavailable.", flush=True)
                return

        try:
            tf = tf_buffer.lookup_transform(TARGET_FRAME, ROBOT_FRAME, rclpy.time.Time())
            rx = tf.transform.translation.x
            ry = tf.transform.translation.y
        except Exception as e:
            print("FAIL: cannot get map -> base_link TF.", flush=True)
            print(str(e), flush=True)
            return

        gx = msg.point.x
        gy = msg.point.y
        dist = math.hypot(gx - rx, gy - ry)

        print(f"robot map pose x={rx:.3f}, y={ry:.3f}", flush=True)
        print(f"clicked distance from robot = {dist:.3f} m", flush=True)

        if dist < MIN_GOAL_DISTANCE:
            print(f"IGNORE: clicked point is too close (< {MIN_GOAL_DISTANCE:.2f} m).", flush=True)
            return

        yaw = math.atan2(gy - ry, gx - rx)
        z, w = quat_from_yaw(yaw)

        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.header.frame_id = msg.header.frame_id if msg.header.frame_id else "map"
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z = z
        goal.pose.pose.orientation.w = w
        goal.behavior_tree = ""

        print("Sending NavigateToPose goal...", flush=True)
        print(f"goal x={gx:.3f}, y={gy:.3f}, yaw={math.degrees(yaw):.1f}", flush=True)

        future = client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)

    return clicked_cb

print("===== Publish Point -> Nav2 action stable daemon =====", flush=True)

for topic in CLICKED_TOPICS:
    node.create_subscription(PointStamped, topic, make_clicked_cb(topic), 10)
    print(f"Listening: {topic}", flush=True)

print("T8 is now subscribed. Keep this terminal running.", flush=True)
print("In RViz: choose Publish Point, then single-click a reachable point.", flush=True)

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass

node.destroy_node()
rclpy.shutdown()
PY
```

---

## Test terminal: health check

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash

echo "===== /people ====="
timeout 5s ros2 topic echo /people --once | grep -E "name:|^[[:space:]]+x:|^[[:space:]]+y:" || true

echo
echo "===== /clock ====="
timeout 5s ros2 topic echo /clock --once >/dev/null && echo "PASS: /clock active" || echo "FAIL: /clock inactive"

echo
echo "===== lidar ====="
timeout 5s ros2 topic echo /cpr_j100_0001/sensors/lidar2d_0/scan --once >/dev/null && echo "PASS: lidar active" || echo "FAIL: lidar inactive"

echo
echo "===== map ====="
python3 - <<'PY'
import time
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

rclpy.init()
node = rclpy.create_node("map_check")
qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

got = False
def cb(msg):
    global got
    got = True
    print(f"PASS: map received frame={msg.header.frame_id} size={msg.info.width}x{msg.info.height}")

node.create_subscription(OccupancyGrid, "/cpr_j100_0001/map", cb, qos)

start = time.monotonic()
while rclpy.ok() and time.monotonic() - start < 8.0:
    rclpy.spin_once(node, timeout_sec=0.2)
    if got:
        break

if not got:
    print("FAIL: no map received")

node.destroy_node()
rclpy.shutdown()
PY
```

---

## Cleanup terminal: kill daemons and clear caches

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash 2>/dev/null || true

echo "===== STOP ALL BACKGROUND DAEMONS ====="

pkill -INT -f '[p]ublish_point_to_nav2_action' 2>/dev/null || true
pkill -INT -f '[j]ackal_tf_repair' 2>/dev/null || true
pkill -INT -f '[v]iew_navigation.launch.py' 2>/dev/null || true
pkill -INT -f '[r]viz2' 2>/dev/null || true
pkill -INT -f '[n]av2.launch.py' 2>/dev/null || true
pkill -INT -f '[s]lam.launch.py' 2>/dev/null || true
pkill -INT -f '[r]obot_spawn.launch.py' 2>/dev/null || true
pkill -INT -f '[p]arameter_bridge' 2>/dev/null || true
pkill -INT -f '[s]imulation_fortress.launch.py' 2>/dev/null || true
pkill -INT -f '[g]z sim' 2>/dev/null || true
pkill -INT -f '[i]gn gazebo' 2>/dev/null || true
pkill -INT -f '[h]unav' 2>/dev/null || true
pkill -INT -f '[H]uNav' 2>/dev/null || true
pkill -INT -f '[c]pr_j100_0001' 2>/dev/null || true

sleep 3

pkill -KILL -f '[p]ublish_point_to_nav2_action' 2>/dev/null || true
pkill -KILL -f '[j]ackal_tf_repair' 2>/dev/null || true
pkill -KILL -f '[v]iew_navigation.launch.py' 2>/dev/null || true
pkill -KILL -f '[r]viz2' 2>/dev/null || true
pkill -KILL -f '[n]av2.launch.py' 2>/dev/null || true
pkill -KILL -f '[s]lam.launch.py' 2>/dev/null || true
pkill -KILL -f '[r]obot_spawn.launch.py' 2>/dev/null || true
pkill -KILL -f '[p]arameter_bridge' 2>/dev/null || true
pkill -KILL -f '[s]imulation_fortress.launch.py' 2>/dev/null || true
pkill -KILL -f '[g]z sim' 2>/dev/null || true
pkill -KILL -f '[i]gn gazebo' 2>/dev/null || true
pkill -KILL -f '[h]unav' 2>/dev/null || true
pkill -KILL -f '[H]uNav' 2>/dev/null || true
pkill -KILL -f '[c]pr_j100_0001' 2>/dev/null || true

echo "===== CLEAR ROS CLI DAEMON AND LOCAL LOG CACHE ====="
ros2 daemon stop 2>/dev/null || true
rm -rf ~/.ros/log/* 2>/dev/null || true
rm -f run_hunav_cold_start.log slam_toolbox_debug.log nav2_debug.log spawn_jackal_cold_start.log 2>/dev/null || true
rm -f /tmp/launch_params_* 2>/dev/null || true

sleep 2
ros2 daemon start

echo "===== REMAINING PROCESSES ====="
pgrep -af "publish_point_to_nav2_action|jackal_tf_repair|view_navigation|rviz2|nav2.launch|slam.launch|robot_spawn|parameter_bridge|simulation_fortress|gz sim|ign gazebo|hunav|HuNav|cpr_j100"   || echo "PASS: no matching background daemons remain."
```

---

## Cleanup terminal: full daemon stop

Use this when you want the simulation fully stopped and do not want to restart the ROS CLI daemon afterward.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash 2>/dev/null || true

echo "===== FULL STOP: SIMULATION, NAVIGATION, RVIZ, AND ROS DAEMON ====="

for pattern in \
  '[p]ublish_point_to_nav2_action' \
  '[j]ackal_tf_repair' \
  '[v]iew_navigation.launch.py' \
  '[r]viz2' \
  '[n]av2.launch.py' \
  '[s]lam.launch.py' \
  '[r]obot_spawn.launch.py' \
  '[p]arameter_bridge' \
  '[s]imulation_fortress.launch.py' \
  '[g]z sim' \
  '[i]gn gazebo' \
  '[h]unav' \
  '[H]uNav' \
  '[c]pr_j100_0001'
do
  pkill -INT -f "$pattern" 2>/dev/null || true
done

sleep 3

for pattern in \
  '[p]ublish_point_to_nav2_action' \
  '[j]ackal_tf_repair' \
  '[v]iew_navigation.launch.py' \
  '[r]viz2' \
  '[n]av2.launch.py' \
  '[s]lam.launch.py' \
  '[r]obot_spawn.launch.py' \
  '[p]arameter_bridge' \
  '[s]imulation_fortress.launch.py' \
  '[g]z sim' \
  '[i]gn gazebo' \
  '[h]unav' \
  '[H]uNav' \
  '[c]pr_j100_0001'
do
  pkill -KILL -f "$pattern" 2>/dev/null || true
done

echo "===== STOP ROS CLI DAEMON AND CLEAR LOCAL STATE ====="
ros2 daemon stop 2>/dev/null || true
rm -rf ~/.ros/log/* 2>/dev/null || true
rm -f run_hunav_cold_start.log slam_toolbox_debug.log nav2_debug.log spawn_jackal_cold_start.log 2>/dev/null || true
rm -f /tmp/launch_params_* 2>/dev/null || true

echo "===== VERIFY NO MATCHING DAEMONS REMAIN ====="
pgrep -af "publish_point_to_nav2_action|jackal_tf_repair|view_navigation|rviz2|nav2.launch|slam.launch|robot_spawn|parameter_bridge|simulation_fortress|gz sim|ign gazebo|hunav|HuNav|cpr_j100" \
  || echo "PASS: no matching background daemons remain."

ros2 daemon status 2>/dev/null || echo "PASS: ROS CLI daemon is stopped."
```
