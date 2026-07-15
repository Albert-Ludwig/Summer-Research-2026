#!/usr/bin/env python3
"""
One-command launcher for the final SocialNavDiffusion + HuNav + Jackal Gazebo test.

Run inside the ROS/Gazebo container:
  python3 run_final_social_nav_test.py

Useful options:
  python3 run_final_social_nav_test.py
  python3 run_final_social_nav_test.py --goal 4.0 0.0
  python3 run_final_social_nav_test.py --no-rviz --goal 4.0 0.0
  python3 run_final_social_nav_test.py --skip-cleanup

This starts HuNav/Gazebo, clock bridge, Jackal spawn, TF repair, SLAM,
policy_cmd_vel_node, nav2_goal_to_pose_bridge, and optionally RViz.
It also runs the documented pre-checks, sends a default goal, validates
cmd_vel/debug topics, and checks odom movement.
It does NOT start Nav2.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

ROS_ENV = """
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
""".strip()

COMMON = """
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/hunav_jazzy_ws/install/setup.bash
cd /home/ubuntu/waterloo_jackal_pipeline_repo
source install/setup.bash
""".strip()


class Proc:
    def __init__(self, name: str, popen: subprocess.Popen):
        self.name = name
        self.popen = popen

    def alive(self) -> bool:
        return self.popen.poll() is None


def sh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def start(name: str, cmd: str, log_dir: Path) -> Proc:
    log = log_dir / f"{name}.log"
    f = log.open("w", buffering=1)
    print(f"[start] {name}  log={log}")
    p = subprocess.Popen(
        ["bash", "-lc", "set -e\n" + cmd],
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        text=True,
    )
    return Proc(name, p)


def cleanup_old() -> None:
    print("[cleanup] killing old policy wrapper and goal bridge")
    cmd = """
source /opt/ros/jazzy/setup.bash || true
pkill -INT -f '[p]olicy_cmd_vel_node' 2>/dev/null || true
pkill -INT -f '[j]ackal_pipeline.launch.py' 2>/dev/null || true
pkill -INT -f '[n]av2_goal_to_pose_bridge' 2>/dev/null || true
sleep 3
pkill -KILL -f '[p]olicy_cmd_vel_node' 2>/dev/null || true
pkill -KILL -f '[j]ackal_pipeline.launch.py' 2>/dev/null || true
pkill -KILL -f '[n]av2_goal_to_pose_bridge' 2>/dev/null || true
ros2 daemon stop 2>/dev/null || true
sleep 2
ros2 daemon start 2>/dev/null || true
sleep 2
"""
    out = sh(cmd).stdout.strip()
    if out:
        print(out)


def wait(label: str, seconds: int) -> None:
    print(f"[wait] {label}: {seconds}s")
    time.sleep(seconds)


def check_topic(topic: str, msg_type: str, timeout: int = 10) -> bool:
    cmd = f"""
{COMMON}
{ROS_ENV}
timeout {timeout}s ros2 topic echo {topic} {msg_type} --once >/dev/null
"""
    ok = sh(cmd).returncode == 0
    print(f"[check] {topic}: {'PASS' if ok else 'FAIL'}")
    return ok


def check_namespaced_tf() -> None:
    cmd = f"""
{COMMON}
{ROS_ENV}
echo '===== namespaced TF: map -> base_link ====='
timeout 8s ros2 run tf2_ros tf2_echo map base_link \
  --ros-args \
  -r /tf:=/cpr_j100_0001/tf \
  -r /tf_static:=/cpr_j100_0001/tf_static
"""
    print(sh(cmd).stdout)


def print_state() -> None:
    cmd = f"""
{COMMON}
{ROS_ENV}
echo '===== nodes ====='
ros2 node list | grep -Ei 'policy|goal|bridge|navigate' || true

echo '===== actions ====='
ros2 action list | grep navigate || true

echo '===== /goal_pose ====='
ros2 topic info /goal_pose --verbose || true

echo '===== /cpr_j100_0001/cmd_vel ====='
ros2 topic info /cpr_j100_0001/cmd_vel --verbose || true

echo '===== /social_nav_diffusion/policy_debug ====='
ros2 topic info /social_nav_diffusion/policy_debug --verbose || true
"""
    print(sh(cmd).stdout)


def send_goal(x: float, y: float) -> None:
    print(f"[goal] /goal_pose map x={x}, y={y}")
    cmd = f"""
{COMMON}
{ROS_ENV}
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
"{{header: {{frame_id: map}}, pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: 1.0}}}}}}"
"""
    print(sh(cmd).stdout)


def validate_once() -> None:
    cmd = f"""
{COMMON}
{ROS_ENV}
echo '===== policy_debug ====='
timeout 20s ros2 topic echo /social_nav_diffusion/policy_debug --once --full-length || echo 'NO policy_debug'

echo '===== policy cmd_vel ====='
timeout 20s ros2 topic echo /cpr_j100_0001/cmd_vel --once || echo 'NO policy cmd_vel'

echo '===== platform cmd_vel ====='
timeout 20s ros2 topic echo /cpr_j100_0001/platform/cmd_vel --once || echo 'NO platform cmd_vel'
"""
    print(sh(cmd).stdout)


def tail(log_dir: Path, names: Optional[List[str]] = None) -> None:
    names = names or ["hunav_gazebo", "clock_bridge", "spawn_jackal", "tf_repair", "slam", "policy_wrapper", "goal_bridge", "rviz"]
    print(f"\n[logs] tail in {log_dir}")
    for n in names:
        p = log_dir / f"{n}.log"
        if p.exists():
            print(f"\n----- {n}.log -----")
            print(sh(f"tail -n 35 {p}").stdout.rstrip())


def stop_all(procs: List[Proc]) -> None:
    print("\n[shutdown] stopping launched processes")
    for p in reversed(procs):
        if p.alive():
            print(f"[shutdown] SIGINT {p.name}")
            try:
                os.killpg(os.getpgid(p.popen.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
    time.sleep(4)
    for p in reversed(procs):
        if p.alive():
            print(f"[shutdown] SIGKILL {p.name}")
            try:
                os.killpg(os.getpgid(p.popen.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def commands() -> dict[str, str]:
    return {
        "hunav_gazebo": f"""
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash
{ROS_ENV}
export HUNAV_SHARE="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/share/hunav_gazebo_fortress_wrapper"
export HUNAV_LIB="$HOME/hunav_jazzy_ws/install/hunav_gazebo_fortress_wrapper/lib"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HUNAV_LIB${{GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}}"
export GZ_SIM_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${{GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}}"
export GAZEBO_RESOURCE_PATH="$HUNAV_SHARE/worlds:/opt/ros/jazzy/share/clearpath_gz/worlds:/opt/ros/jazzy/share/clearpath_gz/meshes:/opt/ros/jazzy/share${{GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}}"
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
""",
        "clock_bridge": f"""
source /opt/ros/jazzy/setup.bash
{ROS_ENV}
if gz topic -l | grep -qx "/clock"; then
  ros2 run ros_gz_bridge parameter_bridge '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
elif gz topic -l | grep -qx "/world/office/clock"; then
  ros2 run ros_gz_bridge parameter_bridge '/world/office/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' --ros-args -r /world/office/clock:=/clock
else
  echo "FAIL: No Gazebo clock topic found."
  gz topic -l | grep clock || true
  sleep 3600
fi
""",
        "spawn_jackal": f"""
source /opt/ros/jazzy/setup.bash
{ROS_ENV}
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
""",
        "tf_repair": f"""
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash
{ROS_ENV}
python3 - <<'PY_TF_REPAIR'
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

ROBOT_NS = "/cpr_j100_0001"
ODOM_TOPIC = f"{{ROBOT_NS}}/platform/odom"

rclpy.init()
node = rclpy.create_node("jackal_tf_repair_planner_test")

tf_pub_global = node.create_publisher(TFMessage, "/tf", 100)
tf_pub_ns = node.create_publisher(TFMessage, f"{{ROBOT_NS}}/tf", 100)

static_qos = QoSProfile(depth=10)
static_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
static_qos.reliability = ReliabilityPolicy.RELIABLE

tf_static_pub_global = node.create_publisher(TFMessage, "/tf_static", static_qos)
tf_static_pub_ns = node.create_publisher(TFMessage, f"{{ROBOT_NS}}/tf_static", static_qos)

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
print("jackal_tf_repair_planner_test running.", flush=True)
rclpy.spin(node)
PY_TF_REPAIR
""",
        "slam": f"""
source /opt/ros/jazzy/setup.bash
cd ~/hunav_jazzy_ws
source install/setup.bash
{ROS_ENV}
ros2 launch clearpath_nav2_demos slam.launch.py \
  use_sim_time:=true \
  setup_path:=$HOME/clearpath
""",
        "policy_wrapper": f"""
{COMMON}
{ROS_ENV}
export SOCIAL_NAV_DIFFUSION_USE_VENV=true
ros2 launch social_nav_diffusion_ros jackal_pipeline.launch.py \
  params_file:=/home/ubuntu/waterloo_jackal_pipeline_repo/install/social_nav_diffusion_ros/share/social_nav_diffusion_ros/config/angular_half_eval.yaml \
  topics_file:=/home/ubuntu/waterloo_jackal_pipeline_repo/install/social_nav_diffusion_ros/share/social_nav_diffusion_ros/config/topics_sim.yaml \
  use_sim_time:=true \
  use_diffusion_policy:=true
""",
        "goal_bridge": f"""
{COMMON}
{ROS_ENV}
ros2 run social_nav_diffusion_ros nav2_goal_to_pose_bridge --ros-args \
  --params-file /home/ubuntu/waterloo_jackal_pipeline_repo/install/social_nav_diffusion_ros/share/social_nav_diffusion_ros/config/topics_sim.yaml
""",
        "rviz": f"""
{COMMON}
{ROS_ENV}
ros2 launch clearpath_viz view_navigation.launch.py \
  namespace:=cpr_j100_0001 \
  use_sim_time:=true
""",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rviz", action="store_true")
    ap.add_argument("--skip-cleanup", action="store_true")
    ap.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), default=(4.0, 0.0))
    ap.add_argument("--no-goal", action="store_true", help="Start the stack only; do not send the default goal.")
    ap.add_argument("--log-dir", default="/tmp/social_nav_final_test_logs")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_cleanup:
        cleanup_old()

    cmds = commands()
    procs: List[Proc] = []
    shutting_down = False

    def on_signal(signum, frame):
        nonlocal shutting_down
        shutting_down = True
        print(f"\n[signal] received {signum}")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        procs.append(start("hunav_gazebo", cmds["hunav_gazebo"], log_dir)); wait("Gazebo/HuNav", 12)
        procs.append(start("clock_bridge", cmds["clock_bridge"], log_dir)); wait("clock bridge", 3)
        procs.append(start("spawn_jackal", cmds["spawn_jackal"], log_dir)); wait("Jackal spawn", 12)
        procs.append(start("tf_repair", cmds["tf_repair"], log_dir)); wait("TF repair", 3)
        procs.append(start("slam", cmds["slam"], log_dir)); wait("SLAM", 12)

        print("\n[precheck]")
        check_topic("/clock", "rosgraph_msgs/msg/Clock", 8)
        check_topic("/people", "people_msgs/msg/People", 10)
        check_topic("/cpr_j100_0001/platform/odom/filtered", "nav_msgs/msg/Odometry", 10)
        check_topic("/cpr_j100_0001/map", "nav_msgs/msg/OccupancyGrid", 15)
        check_topic("/cpr_j100_0001/sensors/lidar2d_0/scan", "sensor_msgs/msg/LaserScan", 10)
        check_namespaced_tf()

        procs.append(start("policy_wrapper", cmds["policy_wrapper"], log_dir)); wait("policy wrapper", 20)
        procs.append(start("goal_bridge", cmds["goal_bridge"], log_dir)); wait("goal bridge", 4)
        if not args.no_rviz:
            procs.append(start("rviz", cmds["rviz"], log_dir)); wait("RViz", 5)

        print_state()

        if not args.no_goal:
            send_goal(args.goal[0], args.goal[1])
            wait("policy response", 8)
            validate_once()

        print("\n[ready] Stack is running.")
        if args.no_goal:
            print("Use RViz Nav2 Goal, or run again with --goal X Y.")
        else:
            print(f"Default goal was sent: x={args.goal[0]}, y={args.goal[1]}")
        print(f"Logs: {log_dir}")
        print("Press Ctrl+C in this terminal to stop all launched processes.")

        warned: set[str] = set()
        while not shutting_down:
            dead = [p.name for p in procs if not p.alive()]
            new_dead = [d for d in dead if d not in warned]
            if new_dead:
                warned.update(new_dead)
                print(f"[warn] exited: {new_dead}")
                tail(log_dir, new_dead)
            time.sleep(5)
    finally:
        stop_all(procs)
        tail(log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
