# Minimal Cold-Start Guide: HuNav + Jackal + Gazebo Harmonic

> Goal: start the full HuNav + Jackal + Gazebo Harmonic setup from a clean reboot / cold start, then run a minimal health check.

---

## 0. Terminal roles

| Terminal   | Purpose                                   |
| ---------- | ----------------------------------------- |
| Terminal 1 | Background full HuNav + Gazebo simulation |
| Terminal 2 | Background Jackal spawn / controllers     |
| Terminal 3 | Health checks / tests                     |
| Terminal 4 | Cleanup / recovery terminal               |
| Terminal 5 | `/clock` bridge                           |

---

## 1. Terminal 4: Optional clean-state reset

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

echo "===== FIND GAZEBO CLOCK TOPIC ====="
CLOCK_TOPIC=$(gz topic -l | grep -E '^(/clock|/world/.*/clock)$' | head -n 1)

echo "CLOCK_TOPIC=$CLOCK_TOPIC"

if [ -z "$CLOCK_TOPIC" ]; then
  echo "FAIL: No Gazebo clock topic found."
  gz topic -l | grep clock || true
elif [ "$CLOCK_TOPIC" = "/clock" ]; then
  ros2 run ros_gz_bridge parameter_bridge \
    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
else
  ros2 run ros_gz_bridge parameter_bridge \
    "${CLOCK_TOPIC}@rosgraph_msgs/msg/Clock[gz.msgs.Clock" \
    --ros-args -r "${CLOCK_TOPIC}:=/clock"
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

## 5. Terminal 3: Minimal health check

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

## 6. Terminal 3: Optional minimal motion test

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

## 7. Shutdown order

Stop the terminals using `Ctrl+C` in this order:

```text
Terminal 2: Jackal
Terminal 5: /clock bridge
Terminal 1: HuNav + Gazebo
```

Then use Terminal 4 for cleanup if needed:

```bash
source /opt/ros/jazzy/setup.bash

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
```

---

## 8. Last verified successful result

The most recent full validation showed:

```text
/people active
LiDAR active
odom active
controller produced linear.x = 0.10
odom delta = +0.1408 m
wheel delta = +1.4367 rad
unknown geometry = 0
generatedWorld has no legacy Ignition systems
```
