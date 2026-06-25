# Nav Diffusion Launch

## Rules

- Numbered terminals are daemon / long-running processes only.
- Test terminal is for one-shot checks only.
- Do not run test commands inside daemon terminals.
- Use the same ROS discovery profile in every terminal.
- Do not mix old `LOCALHOST` terminals with new `SUBNET` terminals.
- SocialNavDiffusion currently does not publish `/cmd_vel`; it only publishes debug topics:
  - `/social_nav_diffusion/debug_action`
  - `/social_nav_diffusion/debug_trajectory`

---

## Common ROS env block

Use this after sourcing ROS/workspace in every terminal:

```bash
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
```

Use `SUBNET + UDPv4` in this container.

Do not use:

```bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset FASTDDS_BUILTIN_TRANSPORTS
```

`LOCALHOST + unset FASTDDS_BUILTIN_TRANSPORTS` caused ROS graph discovery failures where publishers could publish, but `ros2 topic list`, RViz, and endpoint discovery were unstable.

---

## Launch order

1. Terminal 1: HuNav + Gazebo
2. Terminal 5: `/clock` bridge
3. Terminal 2: Spawn Jackal
4. Terminal 7: TF repair
5. Terminal 4: SLAM
6. Terminal 3: Nav2
7. Test terminal: pre-check before NavDiffusion
8. Terminal 8: SocialNavDiffusion
9. Test terminal: verify debug topics and marker
10. Terminal 6: RViz

---

## Optional cold reset

Use this if ROS graph discovery is broken or after changing the ROS discovery profile.

From Windows PowerShell:

```powershell
docker restart fc4c042f675c
```

After restarting the container, run the CLI probe before launching Gazebo/Nav2/SocialNavDiffusion.

---

## Test terminal: CLI probe before launching anything

Do not launch Gazebo, Nav2, RViz, or SocialNavDiffusion before this passes.

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

ros2 daemon stop 2>/dev/null || true
sleep 2

ros2 topic pub -r 1 /social_nav_diffusion/cli_probe std_msgs/msg/String "{data: cli_probe}" \
  > /tmp/cli_probe.log 2>&1 &

echo "PUBPID=$!"
sleep 5

echo "===== cli probe log ====="
cat /tmp/cli_probe.log

echo
echo "===== cli probe topic ====="
ros2 topic list --no-daemon | grep /social_nav_diffusion/cli_probe || echo "FAIL: topic not listed"

echo
echo "===== cli probe echo ====="
timeout 10s ros2 topic echo /social_nav_diffusion/cli_probe std_msgs/msg/String --once || true
```

Expected:

```text
/social_nav_diffusion/cli_probe
data: cli_probe
```

If the CLI probe fails, restart the container again. Do not continue.

Kill the probe:

```bash
pkill -f "ros2 topic pub.*social_nav_diffusion/cli_probe" || true
sleep 1
ros2 topic list --no-daemon | grep cli_probe || echo "PASS: cli_probe stopped"
```

---

## Terminal 1: HuNav + Gazebo

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

rm -f run_hunav_cold_start.log

ros2 launch hunav_gazebo_fortress_wrapper simulation_fortress.launch.py \
  environment_name:=office_no_sensors \
  configuration_file:=office_2_agents.yaml \
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

Keep running.

---

## Terminal 2: Spawn Jackal

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
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
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

Keep running.

---

## Terminal 3: Nav2

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

## Test terminal: pre-check before Terminal 8

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

echo "===== /clock ====="
timeout 5s ros2 topic echo /clock --once >/dev/null \
  && echo "PASS: /clock active" \
  || echo "FAIL: /clock inactive"

echo
echo "===== /people ====="
timeout 10s ros2 topic echo /people people_msgs/msg/People --once >/dev/null \
  && echo "PASS: /people active" \
  || echo "FAIL: /people inactive"

echo
echo "===== odom ====="
timeout 10s ros2 topic echo /cpr_j100_0001/platform/odom nav_msgs/msg/Odometry --once >/dev/null \
  && echo "PASS: odom active" \
  || echo "FAIL: odom inactive"
```

Do not start Terminal 8 unless all three are PASS.

---

## Test terminal: kill stale SocialNavDiffusion

```bash
pkill -INT -f social_nav_diffusion_node || true
pkill -INT -f social_nav_diffusion_ros || true
sleep 3
pkill -KILL -f social_nav_diffusion_node || true
pkill -KILL -f social_nav_diffusion_ros || true

pgrep -af "social_nav_diffusion_node|social_nav_diffusion_ros" || echo "PASS: no old social node"
```

---

## Terminal 8: SocialNavDiffusion

```bash
cd /home/ubuntu/hunav_jazzy_ws

deactivate 2>/dev/null || true

source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

ros2 run social_nav_diffusion_ros social_nav_diffusion_node
```

Expected Terminal 8 runtime output:

```text
social_nav_diffusion_node ready. Debug only: not publishing /cmd_vel.
DDIM sampling ...
[proj] OK
[predict] total ...
```

Keep running.

---

## Test terminal: verify SocialNavDiffusion endpoints

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

echo "===== subscriptions ====="
ros2 topic info /cpr_j100_0001/platform/odom --verbose | grep -E "Publisher count:|Subscription count:|Node name:|Endpoint type:" -A1
ros2 topic info /people --verbose | grep -E "Publisher count:|Subscription count:|Node name:|Endpoint type:" -A1

echo
echo "===== debug endpoints ====="
ros2 topic info /social_nav_diffusion/debug_action --verbose
ros2 topic info /social_nav_diffusion/debug_trajectory --verbose
```

Expected:
- `social_nav_diffusion_node` appears as a SUBSCRIPTION under `/people`.
- `social_nav_diffusion_node` appears as a SUBSCRIPTION under `/cpr_j100_0001/platform/odom`.
- `/social_nav_diffusion/debug_action` has Publisher count: 1.
- `/social_nav_diffusion/debug_trajectory` has Publisher count: 1.

---

## Test terminal: verify debug action and marker

```bash
timeout 180s ros2 topic echo /social_nav_diffusion/debug_action std_msgs/msg/String \
  --qos-reliability reliable \
  --qos-durability volatile \
  --once \
  || echo "FAIL: no debug_action message in 180s"
```

Then:

```bash
timeout 180s ros2 topic echo /social_nav_diffusion/debug_trajectory visualization_msgs/msg/Marker \
  --qos-reliability reliable \
  --qos-durability volatile \
  --once \
  | grep -E "frame_id:|ns:|id:|type:|action:|points:|- x:|  y:" \
  | head -60
```

Expected:

```text
people count: 3
used_projection: True
frame_id: map
type: 4
points:
```

---

## Terminal 6: RViz

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

ros2 launch clearpath_viz view_navigation.launch.py \
  namespace:=cpr_j100_0001 \
  use_sim_time:=true
```

RViz setup:

```text
Global Options -> Fixed Frame = map

Add:
- TF
- Map        /cpr_j100_0001/map
- LaserScan  /cpr_j100_0001/sensors/lidar2d_0/scan
- Marker     /social_nav_diffusion/debug_trajectory
```

Marker settings:
- Display type: `Marker`, not `MarkerArray`.
- Topic: `/social_nav_diffusion/debug_trajectory`.
- If the topic does not appear in the dropdown, manually type it.
- Fixed Frame must be `map`.

---

# If the SocialNavDiffusion Marker is missing

## Possible causes

1. Wrong ROS discovery profile.
   - Bad profile: `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` with `unset FASTDDS_BUILTIN_TRANSPORTS`.
   - Correct profile: `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET` with `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`.

2. Mixed terminal environments.
   - Some daemon terminals may still use old `LOCALHOST`.
   - All terminals must use `SUBNET + UDPv4`.

3. ROS graph / DDS discovery is broken.
   - A publisher can publish, but `ros2 topic list` cannot see it.
   - RViz cannot discover the Marker topic.
   - CLI probe fails.

4. `/clock` is inactive.
   - Restart Terminal 5.

5. `/people` is inactive.
   - HuNav must publish `/people`.
   - Do not start Terminal 8 before `/people` is active.

6. `/cpr_j100_0001/platform/odom` is inactive.
   - Jackal odom must be active.
   - Do not start Terminal 8 before odom is active.

7. Stale or duplicate SocialNavDiffusion process.
   - Old `social_nav_diffusion_node` processes can remain alive.
   - A process may exist but have no endpoints or no messages.

8. Terminal 8 has endpoints but predict loop is not publishing.
   - Check Terminal 8 output.
   - It should print `DDIM sampling`, `[proj] OK`, and `[predict] total`.

9. RViz display setup is wrong.
   - Use `Marker`, not `MarkerArray`.
   - Topic must be `/social_nav_diffusion/debug_trajectory`.
   - Fixed Frame must be `map`.

---

## Full repair procedure

### Step 1: Restart container if ROS graph is broken

From Windows PowerShell:

```powershell
docker restart fc4c042f675c
```

### Step 2: Run CLI probe before launching anything

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

Expected:

```text
/social_nav_diffusion/cli_probe
data: cli_probe
```

If CLI probe fails, restart the container again. Do not launch Gazebo/Nav2/SocialNavDiffusion until CLI probe passes.

Kill probe:

```bash
pkill -f "ros2 topic pub.*social_nav_diffusion/cli_probe" || true
```

### Step 3: Relaunch all daemon terminals with `SUBNET + UDPv4`

Relaunch in order:

```text
Terminal 1 -> Terminal 5 -> Terminal 2 -> Terminal 7 -> Terminal 4 -> Terminal 3
```

Do not launch Terminal 8 yet.

### Step 4: Run pre-check

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

timeout 5s ros2 topic echo /clock --once >/dev/null && echo "PASS: /clock active" || echo "FAIL: /clock inactive"
timeout 10s ros2 topic echo /people people_msgs/msg/People --once >/dev/null && echo "PASS: /people active" || echo "FAIL: /people inactive"
timeout 10s ros2 topic echo /cpr_j100_0001/platform/odom nav_msgs/msg/Odometry --once >/dev/null && echo "PASS: odom active" || echo "FAIL: odom inactive"
```

All three must be PASS.

### Step 5: Kill stale SocialNavDiffusion

```bash
pkill -INT -f social_nav_diffusion_node || true
pkill -INT -f social_nav_diffusion_ros || true
sleep 3
pkill -KILL -f social_nav_diffusion_node || true
pkill -KILL -f social_nav_diffusion_ros || true

pgrep -af "social_nav_diffusion_node|social_nav_diffusion_ros" || echo "PASS: no old social node"
```

### Step 6: Launch Terminal 8

Run the Terminal 8 command above.

Wait for:

```text
social_nav_diffusion_node ready
DDIM sampling ...
[proj] OK
[predict] total ...
```

### Step 7: Verify endpoints

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

ros2 topic info /cpr_j100_0001/platform/odom --verbose | grep -E "Publisher count:|Subscription count:|Node name:|Endpoint type:" -A1
ros2 topic info /people --verbose | grep -E "Publisher count:|Subscription count:|Node name:|Endpoint type:" -A1
ros2 topic info /social_nav_diffusion/debug_action --verbose
ros2 topic info /social_nav_diffusion/debug_trajectory --verbose
```

Expected:
- `social_nav_diffusion_node` subscribes to `/people`.
- `social_nav_diffusion_node` subscribes to `/cpr_j100_0001/platform/odom`.
- `debug_action` Publisher count: 1.
- `debug_trajectory` Publisher count: 1.

### Step 8: Verify Marker message

```bash
timeout 180s ros2 topic echo /social_nav_diffusion/debug_action std_msgs/msg/String \
  --qos-reliability reliable \
  --qos-durability volatile \
  --once \
  || echo "FAIL: no debug_action message in 180s"

timeout 180s ros2 topic echo /social_nav_diffusion/debug_trajectory visualization_msgs/msg/Marker \
  --qos-reliability reliable \
  --qos-durability volatile \
  --once \
  | grep -E "frame_id:|ns:|id:|type:|action:|points:|- x:|  y:" \
  | head -60
```

Expected:

```text
people count: 3
used_projection: True
frame_id: map
type: 4
points:
```

### Step 9: Open RViz

Use the Terminal 6 command above. Add Marker display manually if needed.
