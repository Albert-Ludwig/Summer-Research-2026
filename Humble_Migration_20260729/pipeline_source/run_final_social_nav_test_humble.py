#!/usr/bin/env python3
"""Safe one-command Humble simulation launcher for SocialNavDiffusion."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


ROS_SETUP = Path("/opt/ros/humble/setup.bash")
HUNAV_WS = Path("/home/ubuntu/hunav_humble_ws")
PIPELINE = Path("/home/ubuntu/waterloo_jackal_pipeline_repo")
VENV = Path("/home/ubuntu/social_nav_diffusion_humble_venv")
INFERENCE = Path("/workspace/SocialNavDiffusion_Inference")
ACADOS = Path("/home/ubuntu/acados")
CLEARPATH_SETUP = Path("/home/ubuntu/clearpath")
CLEARPATH_MESHES = Path(
    "/workspace/Humble_Migration_20260729/"
    "clearpath_gz_jazzy_package/meshes"
)

ROBOT_NS = "/cpr_j100_0001"
ROBOT_MODEL = "cpr_j100_0001/robot"

SIM_ENV = """
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/home/ubuntu/.Xauthority}"
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI
""".strip()

COMMON = f"""
source {ROS_SETUP}
source {HUNAV_WS}/install/setup.bash
source {PIPELINE}/install/setup.bash
{SIM_ENV}
export SOCIAL_NAV_DIFFUSION_VENV={VENV}
export SOCIAL_NAV_DIFFUSION_USE_VENV=true
export ACADOS_SOURCE_DIR={ACADOS}
export LD_LIBRARY_PATH={ACADOS}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
""".strip()


class StackError(RuntimeError):
    pass


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    process_group: int
    log_path: Path
    log_handle: object

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


class StackLauncher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.processes: list[ManagedProcess] = []
        self.stop_requested = False
        self.failed = False
        self.tf_repair_used = False

        if args.log_dir:
            self.log_dir = Path(args.log_dir).expanduser().resolve()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = Path(f"/tmp/social_nav_humble_logs/run_{stamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_shell(
        self, command: str, timeout: Optional[float] = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )

    def ros_shell(
        self, command: str, timeout: Optional[float] = None
    ) -> subprocess.CompletedProcess:
        return self.run_shell(f"{COMMON}\n{command}", timeout=timeout)

    def prepare_runtime_assets(self) -> None:
        source_robot = PIPELINE / "config/clearpath_humble/robot.yaml"
        source_world = (
            PIPELINE
            / "experiment_setup/hunav/worlds/office_no_sensors.sdf"
        )
        source_scenario = (
            PIPELINE
            / "experiment_setup/hunav/scenarios/"
            "office_2_agents_humble.yaml"
        )
        source_bt_dir = PIPELINE / "experiment_setup/hunav/behavior_trees"
        wrapper_source = (
            HUNAV_WS / "src/hunav_gazebo_fortress_wrapper"
        )

        required_sources = [
            source_robot,
            source_world,
            source_scenario,
            source_bt_dir / "office_2_agents__agent_1_bt.xml",
            source_bt_dir / "office_2_agents__agent_2_bt.xml",
        ]
        missing = [str(path) for path in required_sources if not path.exists()]
        if missing:
            raise StackError(
                "Required simulation assets are missing:\n  "
                + "\n  ".join(missing)
            )

        CLEARPATH_SETUP.mkdir(parents=True, exist_ok=True)
        target_robot = CLEARPATH_SETUP / "robot.yaml"
        if target_robot.exists():
            if target_robot.read_bytes() != source_robot.read_bytes():
                raise StackError(
                    f"Refusing to overwrite different file: {target_robot}"
                )
        else:
            shutil.copy2(source_robot, target_robot)

        copies = [
            (source_world, wrapper_source / "worlds/office_no_sensors.sdf"),
            (
                source_scenario,
                wrapper_source
                / "scenarios/office_2_agents_humble.yaml",
            ),
            (
                source_bt_dir / "office_2_agents__agent_1_bt.xml",
                wrapper_source
                / "behavior_trees/office_2_agents__agent_1_bt.xml",
            ),
            (
                source_bt_dir / "office_2_agents__agent_2_bt.xml",
                wrapper_source
                / "behavior_trees/office_2_agents__agent_2_bt.xml",
            ),
        ]
        for source, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copy2(source, target)

    def preflight(self) -> None:
        print("[stage A] Humble environment preflight")
        self.prepare_runtime_assets()

        required_paths = [
            ROS_SETUP,
            HUNAV_WS / "install/setup.bash",
            PIPELINE / "install/setup.bash",
            VENV / "bin/python",
            INFERENCE / "ckpt_step478000_SOCIAL_NORMS8.pt",
            ACADOS / "lib/libacados.so",
            PIPELINE
            / "c_generated_proj/"
            "libacados_ocp_solver_diffusion_projection_unicycle.so",
            CLEARPATH_SETUP / "robot.yaml",
            CLEARPATH_MESHES / "office/office.dae",
            CLEARPATH_MESHES / "accessories/wibotic_tr301.dae",
            HUNAV_WS
            / "install/hunav_gazebo_fortress_wrapper/share/"
            "hunav_gazebo_fortress_wrapper/worlds/office_no_sensors.sdf",
            HUNAV_WS
            / "install/hunav_gazebo_fortress_wrapper/share/"
            "hunav_gazebo_fortress_wrapper/scenarios/"
            "office_2_agents_humble.yaml",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise StackError(
                "Preflight paths are missing. Rebuild/stage assets first:\n  "
                + "\n  ".join(missing)
            )

        packages = [
            "people_msgs",
            "hunav_msgs",
            "hunav_agent_manager",
            "hunav_gazebo_fortress_wrapper",
            "ros_gz_bridge",
            "ros_gz_sim",
            "clearpath_gz",
            "clearpath_nav2_demos",
            "clearpath_viz",
            "slam_toolbox",
            "social_nav_diffusion_ros",
        ]
        for package in packages:
            result = self.ros_shell(f"ros2 pkg prefix {package}")
            if result.returncode != 0:
                raise StackError(
                    f"ROS package not available: {package}\n{result.stdout}"
                )
            print(f"[check] package {package}: PASS")

        version = self.ros_shell(
            'test "$ROS_DISTRO" = humble && '
            'python3 -c "import rclpy, people_msgs; print(\'imports ok\')"'
        )
        if version.returncode != 0:
            raise StackError(f"Humble/import check failed:\n{version.stdout}")

        gpu = self.ros_shell(
            f"{VENV}/bin/python -c \"import torch; "
            "assert torch.cuda.is_available(); "
            "print(torch.__version__); print(torch.version.cuda); "
            "print(torch.cuda.get_device_name(0))\"",
            timeout=30,
        )
        if gpu.returncode != 0:
            raise StackError(f"CUDA preflight failed:\n{gpu.stdout}")
        print("[check] CUDA policy environment: PASS")
        print(gpu.stdout.strip())

        gazebo = self.run_shell(
            "command -v ign && ign gazebo --versions && "
            "ign topic --help >/dev/null"
        )
        if gazebo.returncode != 0 or "6." not in gazebo.stdout:
            raise StackError(
                "Gazebo Fortress CLI check failed:\n" + gazebo.stdout
            )
        print("[check] Gazebo Fortress ign CLI: PASS")

        if not self.args.skip_cleanup:
            stale = self.run_shell(
                "pgrep -af "
                "'[i]gn gazebo|[r]uby .*ign gazebo|"
                "[h]unav_gazebo_world_generator|"
                "[h]unav_agent_manager|[p]olicy_cmd_vel_node'"
            )
            if stale.returncode == 0 and stale.stdout.strip():
                raise StackError(
                    "Stale simulation processes are running. Stop them "
                    "manually or use --skip-cleanup after confirming they "
                    "belong to this isolated simulation:\n"
                    + stale.stdout
                )

    def start(self, name: str, command: str) -> ManagedProcess:
        log_path = self.log_dir / f"{name}.log"
        handle = log_path.open("w", encoding="utf-8", buffering=1)
        print(f"[start] {name}  log={log_path}")
        process = subprocess.Popen(
            ["bash", "-lc", "set -eo pipefail\n" + command],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        managed = ManagedProcess(
            name, process, os.getpgid(process.pid), log_path, handle
        )
        self.processes.append(managed)
        return managed

    def wait_alive(self, process: ManagedProcess, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise KeyboardInterrupt
            if not process.alive:
                self.print_log_tail(process)
                raise StackError(
                    f"Critical process exited: {process.name} "
                    f"(code {process.process.returncode})"
                )
            time.sleep(0.5)
        print(f"[check] {process.name} alive for {seconds:.0f}s: PASS")

    def assert_all_alive(self) -> None:
        dead = [proc for proc in self.processes if not proc.alive]
        if dead:
            for proc in dead:
                self.print_log_tail(proc)
            names = ", ".join(proc.name for proc in dead)
            raise StackError(f"Critical process exited: {names}")

    def wait_for_topic(
        self, topic: str, msg_type: str, timeout_sec: int
    ) -> str:
        self.assert_all_alive()
        result = self.ros_shell(
            f"timeout {timeout_sec}s ros2 topic echo "
            f"{topic} {msg_type} --once",
            timeout=timeout_sec + 5,
        )
        if result.returncode != 0:
            raise StackError(
                f"Topic check failed: {topic}\n{result.stdout[-2000:]}"
            )
        print(f"[check] topic {topic}: PASS")
        return result.stdout

    def check_tf(
        self, parent: str, child: str, timeout_sec: int = 8
    ) -> bool:
        result = self.ros_shell(
            f"timeout {timeout_sec}s ros2 run tf2_ros tf2_echo "
            f"{parent} {child} --ros-args "
            f"-r /tf:={ROBOT_NS}/tf "
            f"-r /tf_static:={ROBOT_NS}/tf_static",
            timeout=timeout_sec + 5,
        )
        passed = (
            "Translation:" in result.stdout and "Rotation:" in result.stdout
        )
        print(
            f"[check] TF {parent} -> {child}: "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed and result.stdout.strip():
            print("\n".join(result.stdout.strip().splitlines()[-10:]))
        return passed

    def check_robot_model(self) -> None:
        result = self.run_shell("timeout 8s ign model --list", timeout=12)
        if result.returncode != 0 or ROBOT_MODEL not in result.stdout:
            raise StackError(
                f"Gazebo model {ROBOT_MODEL!r} not found:\n{result.stdout}"
            )
        print(f"[check] Gazebo model {ROBOT_MODEL}: PASS")

    def ensure_zero_policy_command(self) -> None:
        topic = f"{ROBOT_NS}/cmd_vel"
        result = self.ros_shell(
            f"timeout 4s ros2 topic echo "
            f"{topic} geometry_msgs/msg/TwistStamped",
            timeout=9,
        )
        output = result.stdout
        samples: list[tuple[float, float]] = []
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        for block in output.split("---"):
            linear = re.search(
                rf"(?m)^\s*linear:\s*$\n\s*x:\s*({number})", block
            )
            angular = re.search(
                rf"(?ms)^\s*angular:\s*$.*?^\s*z:\s*({number})", block
            )
            if linear and angular:
                samples.append(
                    (float(linear.group(1)), float(angular.group(1)))
                )
        if not samples:
            raise StackError(
                f"No command samples received from {topic} during "
                "the no-goal safety check."
            )
        nonzero = [
            sample
            for sample in samples
            if abs(sample[0]) > 1e-4 or abs(sample[1]) > 1e-4
        ]
        if nonzero:
            raise StackError(
                "Policy emitted nonzero command before an explicit goal: "
                f"{nonzero[:5]}"
            )
        print(
            f"[check] no-goal command safety: PASS "
            f"({len(samples)} zero samples)"
        )

    def require_cmd_subscriber(self) -> None:
        topic = f"{ROBOT_NS}/cmd_vel"
        result = self.ros_shell(f"ros2 topic info {topic} -v")
        match = re.search(r"Subscription count:\s*(\d+)", result.stdout)
        count = int(match.group(1)) if match else 0
        if result.returncode != 0 or count < 1:
            raise StackError(
                f"No simulated platform subscriber on {topic}:\n"
                + result.stdout
            )
        print(f"[check] simulated cmd_vel subscribers={count}: PASS")

    def send_goal(self, x: float, y: float) -> None:
        self.require_cmd_subscriber()
        result = self.ros_shell(
            "ros2 topic pub --once /goal_pose "
            "geometry_msgs/msg/PoseStamped "
            f"'{{header: {{frame_id: map}}, pose: "
            f"{{position: {{x: {x}, y: {y}, z: 0.0}}, "
            "orientation: {w: 1.0}}}'",
            timeout=15,
        )
        if result.returncode != 0:
            raise StackError(f"Explicit goal publish failed:\n{result.stdout}")
        print(f"[goal] explicit simulation goal sent: x={x}, y={y}")

    def verify_goal_control(self, timeout_sec: int = 60) -> None:
        topic = f"{ROBOT_NS}/cmd_vel"
        policy = next(
            (proc for proc in self.processes if proc.name == "policy"), None
        )
        if policy is None:
            raise StackError("Policy process is not registered.")

        deadline = time.monotonic() + timeout_sec
        nonzero_command: Optional[tuple[float, float]] = None
        acados_ok = False
        warmup_complete = False
        model_load_count = 0
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

        while time.monotonic() < deadline:
            self.assert_all_alive()
            result = self.ros_shell(
                f"timeout 3s ros2 topic echo "
                f"{topic} geometry_msgs/msg/TwistStamped --once",
                timeout=6,
            )
            linear = re.search(
                rf"(?m)^\s*linear:\s*$\n\s*x:\s*({number})",
                result.stdout,
            )
            angular = re.search(
                rf"(?ms)^\s*angular:\s*$.*?^\s*z:\s*({number})",
                result.stdout,
            )
            if linear and angular:
                sample = (float(linear.group(1)), float(angular.group(1)))
                if abs(sample[0]) > 1e-4 or abs(sample[1]) > 1e-4:
                    nonzero_command = sample

            log_text = policy.log_path.read_text(
                encoding="utf-8", errors="replace"
            )
            acados_ok = bool(
                re.search(r"\[proj\]\s+OK\s+status=0", log_text)
            )
            warmup_complete = "Policy warm-up completed" in log_text
            model_load_count = log_text.count("diffusion model loaded:")

            if (
                nonzero_command is not None
                and acados_ok
                and warmup_complete
                and model_load_count == 1
            ):
                print(
                    "[check] explicit-goal policy control: PASS "
                    f"(cmd_vel={nonzero_command}, acados status=0, "
                    "model loads=1)"
                )
                return
            time.sleep(1)

        raise StackError(
            "Explicit-goal control verification timed out: "
            f"nonzero_cmd={nonzero_command}, acados_ok={acados_ok}, "
            f"warmup_complete={warmup_complete}, "
            f"model_load_count={model_load_count}"
        )

    def commands(self) -> dict[str, str]:
        resource_paths = (
            f"{HUNAV_WS}/install/hunav_gazebo_fortress_wrapper/share/"
            "hunav_gazebo_fortress_wrapper/worlds:"
            f"{CLEARPATH_MESHES}:"
            "/opt/ros/humble/share/clearpath_gz/worlds:"
            "/opt/ros/humble/share"
        )
        plugin_path = (
            f"{HUNAV_WS}/install/hunav_gazebo_fortress_wrapper/lib"
        )

        return {
            "hunav_gazebo": f"""
source {ROS_SETUP}
source {HUNAV_WS}/install/setup.bash
{SIM_ENV}
export GZ_SIM_SYSTEM_PLUGIN_PATH="{plugin_path}${{GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}}"
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="{plugin_path}${{IGN_GAZEBO_SYSTEM_PLUGIN_PATH:+:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH}}"
export GZ_SIM_RESOURCE_PATH="{resource_paths}${{GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}}"
export IGN_GAZEBO_RESOURCE_PATH="{resource_paths}${{IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}}"
export GAZEBO_RESOURCE_PATH="{resource_paths}${{GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}}"
export LD_LIBRARY_PATH="{plugin_path}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
exec ros2 launch hunav_gazebo_fortress_wrapper simulation_fortress.launch.py \
  environment_name:=office_no_sensors \
  configuration_file:=office_2_agents_humble.yaml \
  robot_name:={ROBOT_MODEL} \
  use_gazebo_obs:=false \
  use_navgoal_to_start:=false \
  global_frame_to_publish:=map \
  update_rate:=20.0 \
  verbose:=false
""",
            "clock_bridge": f"""
source {ROS_SETUP}
{SIM_ENV}
clock_topic=""
for unused in {{1..30}}; do
  if ign topic -l | grep -qx /clock; then
    clock_topic=/clock
    break
  fi
  if ign topic -l | grep -qx /world/office/clock; then
    clock_topic=/world/office/clock
    break
  fi
  sleep 1
done
if [ "$clock_topic" = /clock ]; then
  exec ros2 run ros_gz_bridge parameter_bridge \
    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'
elif [ "$clock_topic" = /world/office/clock ]; then
  exec ros2 run ros_gz_bridge parameter_bridge \
    '/world/office/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock' \
    --ros-args -r /world/office/clock:=/clock
else
  echo "No Fortress clock topic found."
  ign topic -l | grep clock || true
  exit 2
fi
""",
            "spawn_j100": f"""
source {ROS_SETUP}
{SIM_ENV}
export GZ_SIM_RESOURCE_PATH="{resource_paths}${{GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}}"
export IGN_GAZEBO_RESOURCE_PATH="{resource_paths}${{IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}}"
exec ros2 launch clearpath_gz robot_spawn.launch.py \
  setup_path:={CLEARPATH_SETUP}/ \
  world:=office \
  use_sim_time:=true \
  generate:=true \
  rviz:=false \
  x:=0.0 y:=0.0 z:=0.30 yaw:=0.0
""",
            "tf_repair": f"""
{COMMON}
exec python3 {PIPELINE}/scripts/tf_repair_humble.py
""",
            "slam": f"""
source {ROS_SETUP}
{SIM_ENV}
exec ros2 launch clearpath_nav2_demos slam.launch.py \
  use_sim_time:=true setup_path:={CLEARPATH_SETUP}/
""",
            "policy": f"""
{COMMON}
cd {PIPELINE}
exec ros2 launch social_nav_diffusion_ros jackal_pipeline.launch.py \
  params_file:={PIPELINE}/config/angular_half_eval.yaml \
  topics_file:={PIPELINE}/config/topics_sim.yaml \
  use_sim_time:=true \
  use_diffusion_policy:=true \
  start_goal_bridge:=true \
  tf_topic:={ROBOT_NS}/tf \
  tf_static_topic:={ROBOT_NS}/tf_static
""",
            "rviz": f"""
source {ROS_SETUP}
{SIM_ENV}
exec ros2 launch clearpath_viz view_navigation.launch.py \
  namespace:=cpr_j100_0001 use_sim_time:=true
""",
        }

    def run_stack(self) -> None:
        commands = self.commands()

        print("[stage B] HuNav + Gazebo Fortress")
        gazebo = self.start("hunav_gazebo", commands["hunav_gazebo"])
        self.wait_alive(gazebo, 20)

        clock = self.start("clock_bridge", commands["clock_bridge"])
        self.wait_alive(clock, 3)
        self.wait_for_topic("/clock", "rosgraph_msgs/msg/Clock", 12)

        print("[stage C] Simulated Clearpath J100")
        spawn = self.start("spawn_j100", commands["spawn_j100"])
        self.wait_alive(spawn, 20)
        self.check_robot_model()
        self.wait_for_topic("/people", "people_msgs/msg/People", 20)
        self.wait_for_topic(
            f"{ROBOT_NS}/platform/odom",
            "nav_msgs/msg/Odometry",
            30,
        )
        self.wait_for_topic(
            f"{ROBOT_NS}/platform/odom/filtered",
            "nav_msgs/msg/Odometry",
            30,
        )
        self.wait_for_topic(
            f"{ROBOT_NS}/sensors/lidar2d_0/scan",
            "sensor_msgs/msg/LaserScan",
            30,
        )

        if not self.check_tf("odom", "base_link"):
            repair = self.start("tf_repair", commands["tf_repair"])
            self.wait_alive(repair, 3)
            self.tf_repair_used = True
            if not self.check_tf("odom", "base_link", 12):
                raise StackError("TF odom -> base_link remains unavailable.")

        print("[stage D] SLAM")
        slam = self.start("slam", commands["slam"])
        self.wait_alive(slam, 10)
        self.wait_for_topic(
            f"{ROBOT_NS}/map", "nav_msgs/msg/OccupancyGrid", 45
        )
        if not self.check_tf("map", "base_link", 15):
            raise StackError("TF map -> base_link is unavailable.")

        print("[stage E] SocialNavDiffusion policy")
        policy = self.start("policy", commands["policy"])
        self.wait_alive(policy, 15)
        self.wait_for_topic(
            "/social_nav_diffusion/policy_debug",
            "std_msgs/msg/String",
            150,
        )
        self.ensure_zero_policy_command()
        self.require_cmd_subscriber()

        if not self.args.no_rviz:
            rviz = self.start("rviz", commands["rviz"])
            self.wait_alive(rviz, 5)

        self.assert_all_alive()
        print("\n[ready] Stack is running")
        print("Simulation isolation: ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=73")
        print(f"TF repair used: {self.tf_repair_used}")
        print(f"Logs: {self.log_dir}")

        if self.args.goal is None:
            print("No goal sent. Use RViz Nav2 Goal or --goal X Y.")
        else:
            self.send_goal(self.args.goal[0], self.args.goal[1])
            self.verify_goal_control()

        started = time.monotonic()
        while not self.stop_requested:
            self.assert_all_alive()
            if (
                self.args.validation_seconds is not None
                and time.monotonic() - started
                >= self.args.validation_seconds
            ):
                print(
                    "[validation] requested observation period completed "
                    "with all critical processes alive."
                )
                return
            time.sleep(1)

    def print_log_tail(
        self, process: ManagedProcess, line_count: int = 50
    ) -> None:
        try:
            lines = process.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            print(f"[log] unable to read {process.log_path}: {exc}")
            return
        print(f"\n----- {process.name}.log (last {line_count}) -----")
        print("\n".join(lines[-line_count:]))

    def print_all_log_tails(self) -> None:
        for process in self.processes:
            self.print_log_tail(process, 35)

    def stop_all(self) -> None:
        print("\n[shutdown] stopping launcher-owned process groups")
        for managed in reversed(self.processes):
            try:
                os.killpg(managed.process_group, signal.SIGINT)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + 8
        while (
            any(self.process_group_exists(item) for item in self.processes)
            and time.monotonic() < deadline
        ):
            time.sleep(0.25)

        for managed in reversed(self.processes):
            try:
                os.killpg(managed.process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + 3
        while (
            any(self.process_group_exists(item) for item in self.processes)
            and time.monotonic() < deadline
        ):
            time.sleep(0.25)

        for managed in reversed(self.processes):
            try:
                os.killpg(managed.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            managed.log_handle.close()

    @staticmethod
    def process_group_exists(managed: ManagedProcess) -> bool:
        try:
            os.killpg(managed.process_group, 0)
            return True
        except ProcessLookupError:
            return False

    def handle_signal(self, signum: int, _frame: object) -> None:
        if not self.stop_requested:
            print(f"\n[signal] received {signum}; shutting down")
        self.stop_requested = True


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the isolated Humble HuNav + J100 + "
            "SocialNavDiffusion simulation stack."
        )
    )
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Send one explicit simulation goal after all checks pass.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help=(
            "Skip the stale-process preflight. No broad process cleanup "
            "is performed by this launcher."
        ),
    )
    parser.add_argument(
        "--log-dir",
        help="Log directory. Default: timestamped directory under /tmp.",
    )
    parser.add_argument(
        "--validation-seconds",
        type=int,
        help=(
            "Stop cleanly after N ready-state seconds. Intended for "
            "automated validation."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    launcher = StackLauncher(args)
    signal.signal(signal.SIGINT, launcher.handle_signal)
    signal.signal(signal.SIGTERM, launcher.handle_signal)

    try:
        launcher.preflight()
        launcher.run_stack()
        return 0
    except KeyboardInterrupt:
        return 130
    except (StackError, subprocess.TimeoutExpired) as exc:
        launcher.failed = True
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        launcher.print_all_log_tails()
        return 1
    finally:
        launcher.stop_all()
        print(f"[logs] saved in {launcher.log_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
