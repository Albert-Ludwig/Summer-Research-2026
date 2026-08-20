#!/usr/bin/env python3
"""One-command launcher for the real Jackal SocialNavDiffusion test.

Run from Windows PowerShell:
  python "C:\\Users\\Administrator\\Documents\\Summer Research 2026\\Documentations\\run_jackal_robohub.py"

The Windows side starts Jackal onboard localization over SSH, then starts WSL,
Docker, and the container. The same file owns every process until Ctrl+C.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional


WSL_DISTRO = "Ubuntu-22.04"
CONTAINER = "jackal_robohub"
CONTAINER_SCRIPT = "/workspace/Documentations/run_jackal_robohub.py"
JACKAL_SSH_TARGET = "administrator@192.168.131.1"
JACKAL_WORKSPACE = "/home/administrator/nahl_ws"
JACKAL_SETUP = "/etc/clearpath/setup.bash"
JACKAL_LOCALIZATION_PID = "/tmp/run_jackal_robohub_localization.pid"
JACKAL_CAMERA_NODE = "/jackal1/sensors/camera_0/intel_realsense"
JACKAL_COLOR_TOPIC = "/jackal1/sensors/camera_0/color/image"
JACKAL_DEPTH_TOPIC = "/jackal1/sensors/camera_0/aligned_depth_to_color/image"
JACKAL_CAMERA_INFO_TOPIC = (
    "/jackal1/sensors/camera_0/aligned_depth_to_color/camera_info"
)
JACKAL_RGBD_PROFILE = "424,240,15"
VNC_PORT = 6084
VNC_URL = (
    f"http://127.0.0.1:{VNC_PORT}/vnc.html?autoconnect=1&resize=scale"
)

ROS_SETUP = PurePosixPath("/opt/ros/humble/setup.bash")
HUNAV_SETUP = PurePosixPath("/home/ubuntu/hunav_humble_ws/install/setup.bash")
PIPELINE = PurePosixPath("/home/ubuntu/waterloo_jackal_pipeline_repo")
PIPELINE_SETUP = PIPELINE / "install/setup.bash"
VENV = PurePosixPath("/home/ubuntu/social_nav_diffusion_humble_venv")
ACADOS = PurePosixPath("/home/ubuntu/acados")
DDS_PROFILE = PurePosixPath("/workspace/config_files/fastdds_robot_wired.xml")
RVIZ_CONFIG = PurePosixPath(
    "/workspace/config_files/jackal_robohub_navigation.rviz"
)
CHECKPOINT = PurePosixPath(
    "/workspace/SocialNavDiffusion_Inference/"
    "ckpt_step478000_SOCIAL_NORMS8.pt"
)
TEST_MODE_CHECKPOINT = PurePosixPath(
    "/workspace/SocialNavDiffusion_Inference/"
    "ckpt_step990000_sogudiff_singleaxis_1p5M.pt"
)
TEST_MODE_NORM = PurePosixPath(
    "/workspace/SocialNavDiffusion_Inference/"
    "norm_stats_sogudiff_allarms_1p5M.npy"
)
RUNTIME_MODE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "Humble_Migration_20260729"
    / "pipeline_source"
    / "config"
    / "runtime_mode.yaml"
)
DEFAULT_MAP_FILE = PurePosixPath(
    "/home/administrator/nahl_ws/maps/final.yaml"
)
DEFAULT_MAP_TOPIC = "/jackal1/map"
PID_FILE = Path("/tmp/run_jackal_robohub.pid")
LOG_DIR = Path("/tmp/jackal_robohub_launcher")

ROS_ENV = f"""
source {ROS_SETUP}
source {HUNAV_SETUP}
source {PIPELINE_SETUP}
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset FASTDDS_BUILTIN_TRANSPORTS
unset ROS_DISCOVERY_SERVER
unset CYCLONEDDS_URI
export FASTRTPS_DEFAULT_PROFILES_FILE={DDS_PROFILE}
export SOCIAL_NAV_DIFFUSION_VENV={VENV}
export SOCIAL_NAV_DIFFUSION_USE_VENV=true
export ACADOS_SOURCE_DIR={ACADOS}
export LD_LIBRARY_PATH={ACADOS}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=1
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1
""".strip()


class LaunchError(RuntimeError):
    pass


def configured_test_mode() -> bool:
    """Read the single durable mode switch without requiring PyYAML."""
    try:
        text = RUNTIME_MODE_CONFIG.read_text(encoding="utf-8")
    except OSError:
        return False
    match = re.search(r"^\s*test_mode\s*:\s*(true|false)\s*$", text, re.I | re.M)
    return bool(match and match.group(1).lower() == "true")


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


def printable(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_checked(command: list[str]) -> None:
    print(f"[host] {printable(command)}", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise LaunchError(
            f"Command failed with exit code {result.returncode}: "
            f"{printable(command)}"
        )


def ensure_wsl_jackal_route() -> None:
    script = (
        "ip -o -4 addr show dev eth0 | "
        "grep -q 'inet 192.168.131.101/24'; "
        "ip route replace 192.168.131.1/32 dev eth0 "
        "src 192.168.131.101 metric 5"
    )
    run_checked([
        "wsl",
        "-d",
        WSL_DISTRO,
        "-u",
        "root",
        "--",
        "bash",
        "-lc",
        script,
    ])


def wait_for_vnc(timeout_sec: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", VNC_PORT), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def jackal_password() -> str:
    password = os.environ.get("JACKAL_SSH_PASSWORD", "")
    if password:
        return password
    if not sys.stdin.isatty():
        raise LaunchError(
            "Set JACKAL_SSH_PASSWORD in the current PowerShell session."
        )
    password = getpass.getpass("Jackal SSH password: ")
    if not password:
        raise LaunchError("Jackal SSH password cannot be empty.")
    return password


def jackal_ssh_command(password: str, remote_command: str) -> list[str]:
    payload = base64.b64encode(remote_command.encode("utf-8")).decode("ascii")
    return [
        "wsl",
        "-d",
        WSL_DISTRO,
        "--",
        "env",
        f"SSHPASS={password}",
        "sshpass",
        "-e",
        "ssh",
        "-T",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=accept-new",
        JACKAL_SSH_TARGET,
        f"echo {payload} | base64 -d | bash",
    ]


def run_jackal_checked(
    password: str,
    remote_command: str,
    label: str,
) -> None:
    print(f"[host] {label}", flush=True)
    result = subprocess.run(
        jackal_ssh_command(password, remote_command),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise LaunchError(f"{label} failed: {detail or 'unknown SSH error'}")


def sync_jackal_time(password: str) -> None:
    probe = subprocess.run(
        jackal_ssh_command(password, "date -u +%s.%N"),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        matches = re.findall(r"(?m)^(\d+\.\d+)$", probe.stdout.strip())
        if matches:
            skew = float(matches[-1]) - time.time()
            if abs(skew) <= 0.5:
                print(
                    f"[host] Jackal clock already synchronized "
                    f"(skew={skew:+.3f} s).",
                    flush=True,
                )
                return

    script = f"""set -e
target=$(date -u +%s.%N)
printf '%s\\n' "$SSHPASS" | sshpass -e ssh \\
  -T -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \\
  {shlex.quote(JACKAL_SSH_TARGET)} \\
  sudo -S date -u -s "@$target"
""".strip()
    payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
    command = [
        "wsl",
        "-d",
        WSL_DISTRO,
        "--",
        "env",
        f"SSHPASS={password}",
        "bash",
        "-lc",
        f"echo {payload} | base64 -d | bash",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("[host] Jackal time synchronized over SSH.", flush=True)
        restart_command = (
            f"printf '%s\\n' {shlex.quote(password)} | "
            "sudo -S systemctl restart clearpath-platform.service"
        )
        run_jackal_checked(
            password,
            restart_command,
            "Restart Jackal platform after clock adjustment",
        )
        time.sleep(5)
        return
    detail = (result.stderr or result.stdout).strip()
    print(
        "[warn] Jackal SSH time sync failed; "
        f"the ROS clock gate will decide whether startup is safe: {detail}",
        flush=True,
    )


def preflight_jackal_localization(password: str, map_file: str) -> None:
    quoted_map = shlex.quote(map_file)
    remote_command = (
        "set -e; "
        f"test -r {shlex.quote(JACKAL_SETUP)} || "
        "{ echo 'Missing /etc/clearpath/setup.bash' >&2; exit 20; }; "
        f"source {shlex.quote(JACKAL_SETUP)}; "
        f"cd {shlex.quote(JACKAL_WORKSPACE)}; "
        "test -r install/setup.bash || "
        "{ echo 'Missing nahl_ws/install/setup.bash' >&2; exit 21; }; "
        "source install/setup.bash; "
        "ros2 pkg prefix jackal_nav >/dev/null || "
        "{ echo 'Missing onboard package: jackal_nav' >&2; exit 22; }; "
        f"test -r {quoted_map} || "
        "{ echo 'Missing onboard map file' >&2; exit 23; }"
    )
    run_jackal_checked(
        password,
        remote_command,
        "Jackal onboard localization preflight",
    )


def enable_jackal_rgbd(password: str) -> None:
    node = shlex.quote(JACKAL_CAMERA_NODE)
    color_topic = shlex.quote(JACKAL_COLOR_TOPIC)
    depth_topic = shlex.quote(JACKAL_DEPTH_TOPIC)
    camera_info_topic = shlex.quote(JACKAL_CAMERA_INFO_TOPIC)
    profile = shlex.quote(JACKAL_RGBD_PROFILE)
    remote_command = (
        "set -e; "
        f"source {shlex.quote(JACKAL_SETUP)}; "
        f"cd {shlex.quote(JACKAL_WORKSPACE)}; "
        "source install/setup.bash; "
        "export ROS_SUPER_CLIENT=True; "
        "for attempt in $(seq 1 12); do "
        f"ROS2CLI_NO_DAEMON=1 ros2 param set {node} "
        "align_depth.enable false >/dev/null 2>&1 || true; "
        f"ROS2CLI_NO_DAEMON=1 ros2 param set {node} "
        "enable_sync false >/dev/null 2>&1 || true; "
        f"ROS2CLI_NO_DAEMON=1 ros2 param set {node} "
        "enable_depth false >/dev/null 2>&1 || true; "
        f"ROS2CLI_NO_DAEMON=1 ros2 param set {node} "
        "enable_color false >/dev/null 2>&1 || true; "
        "sleep 1; "
        f"if ROS2CLI_NO_DAEMON=1 ros2 param set {node} "
        f"depth_module.depth_profile {profile} "
        f"&& ROS2CLI_NO_DAEMON=1 ros2 param set {node} "
        f"rgb_camera.color_profile {profile} "
        f"&& ROS2CLI_NO_DAEMON=1 ros2 param set {node} enable_depth true "
        f"&& ROS2CLI_NO_DAEMON=1 ros2 param set {node} enable_color true "
        "&& sleep 1 "
        f"&& ROS2CLI_NO_DAEMON=1 ros2 param set {node} enable_sync true "
        f"&& ROS2CLI_NO_DAEMON=1 ros2 param set {node} "
        "align_depth.enable true "
        f"&& ROS2CLI_NO_DAEMON=1 timeout 6s ros2 topic echo {color_topic} "
        "sensor_msgs/msg/Image --once >/dev/null "
        f"&& ROS2CLI_NO_DAEMON=1 timeout 6s ros2 topic echo {depth_topic} "
        "sensor_msgs/msg/Image --once >/dev/null "
        f"&& ROS2CLI_NO_DAEMON=1 timeout 6s ros2 topic echo {camera_info_topic} "
        "sensor_msgs/msg/CameraInfo --once >/dev/null; then exit 0; fi; "
        "sleep 1; done; "
        f"echo 'Timed out waiting for live RGB-D topics from {node}' >&2; exit 1"
    )
    run_jackal_checked(
        password,
        remote_command,
        "Enable Jackal RGB-D streams for people detection",
    )


def start_jackal_localization(
    password: str,
    map_file: str,
) -> tuple[subprocess.Popen, object, Path]:
    quoted_map = shlex.quote(map_file)
    remote_command = (
        "set -e; "
        f"source {shlex.quote(JACKAL_SETUP)}; "
        f"cd {shlex.quote(JACKAL_WORKSPACE)}; "
        "source install/setup.bash; "
        f"echo $$ > {JACKAL_LOCALIZATION_PID}; "
        "exec ros2 launch jackal_nav localisation.launch.py "
        f"map:={quoted_map}"
    )
    log_path = Path(os.environ.get("TEMP", ".")) / (
        "run_jackal_robohub.localization.log"
    )
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    print(f"[host] Starting Jackal onboard localization. log={log_path}")
    process = subprocess.Popen(
        jackal_ssh_command(password, remote_command),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 30
    detail = ""
    while time.monotonic() < deadline:
        log_handle.flush()
        detail = log_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if "All log files can be found" in detail:
            return process, log_handle, log_path
        if process.poll() is not None:
            log_handle.close()
            log_path.unlink(missing_ok=True)
            raise LaunchError(
                "Jackal onboard localization exited during startup: "
                f"{detail or f'exit code {process.returncode}'}"
            )
        time.sleep(0.5)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    log_handle.close()
    log_path.unlink(missing_ok=True)
    raise LaunchError(
        "Timed out waiting for Jackal onboard localization to start. "
        f"Log: {detail or 'no launch output'}"
    )


def stop_jackal_localization(
    password: str,
    process: Optional[subprocess.Popen],
) -> None:
    remote_command = (
        f"if test -r {JACKAL_LOCALIZATION_PID}; then "
        f"pid=$(cat {JACKAL_LOCALIZATION_PID}); "
        "if ps -p \"$pid\" -o args= 2>/dev/null | "
        "grep -q '[l]ocalisation.launch.py'; then "
        "pgid=$(ps -p \"$pid\" -o pgid= | tr -d ' '); "
        "kill -INT -- \"-$pgid\" 2>/dev/null || true; "
        "for attempt in 1 2 3 4 5; do "
        "kill -0 \"$pid\" 2>/dev/null || break; sleep 1; done; "
        "if kill -0 \"$pid\" 2>/dev/null; then "
        "kill -TERM -- \"-$pgid\" 2>/dev/null || true; sleep 2; fi; "
        "if kill -0 \"$pid\" 2>/dev/null; then "
        "kill -KILL -- \"-$pgid\" 2>/dev/null || true; fi; fi; "
        f"rm -f {JACKAL_LOCALIZATION_PID}; fi"
    )
    try:
        subprocess.run(
            jackal_ssh_command(password, remote_command),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=12,
        )
    except subprocess.TimeoutExpired:
        pass
    if process is None or process.poll() is not None:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def prepare_container_desktop() -> None:
    command = (
        "set -e; "
        "for attempt in $(seq 1 180); do "
        "supervisorctl status >/dev/null 2>&1 && break; sleep 1; done; "
        "supervisorctl status >/dev/null 2>&1; "
        "sed -i 's/127\\.0\\.0\\.1:6084/0.0.0.0:6084/' "
        "/etc/supervisor/conf.d/supervisord.conf; "
        "supervisorctl reread >/dev/null; "
        "supervisorctl update >/dev/null; "
        "supervisorctl restart novnc >/dev/null; "
        "pkill -f '[l]ist-oem-metapackages' 2>/dev/null || true; "
        "pkill -f '[m]ate-screensaver' 2>/dev/null || true"
    )
    payload = base64.b64encode(command.encode("utf-8")).decode("ascii")
    run_checked(
        [
            "wsl",
            "-d",
            WSL_DISTRO,
            "--",
            "docker",
            "exec",
            "-u",
            "root",
            CONTAINER,
            "bash",
            "-lc",
            f"echo {payload} | base64 -d | bash",
        ]
    )


def container_arguments(args: argparse.Namespace) -> list[str]:
    result = [
        "python3",
        CONTAINER_SCRIPT,
        "--inside-container",
        "--max-linear-speed",
        str(args.max_linear_speed),
        "--max-angular-speed",
        str(args.max_angular_speed),
        "--warmup-timeout",
        str(args.warmup_timeout),
        "--clock-skew-limit",
        str(args.clock_skew_limit),
        "--map-timeout",
        str(args.map_timeout),
        "--localization-timeout",
        str(args.localization_timeout),
        "--map-topic",
        args.map_topic,
        "--goal-distance-m",
        str(args.goal_distance_m),
        "--trigger-button-index",
        str(args.trigger_button_index),
        "--style-vector",
        *(str(value) for value in args.style_vector),
    ]
    result.append("--test-mode" if args.test_mode else "--no-test-mode")
    if not args.record_bag:
        result.append("--no-record-bag")
    if args.no_rviz:
        result.append("--no-rviz")
    return result


def run_from_windows(args: argparse.Namespace) -> int:
    start_docker = [
        "wsl",
        "-d",
        WSL_DISTRO,
        "-u",
        "root",
        "--",
        "systemctl",
        "start",
        "docker",
    ]
    start_container = [
        "wsl",
        "-d",
        WSL_DISTRO,
        "--",
        "docker",
        "start",
        CONTAINER,
    ]
    docker_exec = [
        "wsl",
        "-d",
        WSL_DISTRO,
        "--",
        "docker",
        "exec",
        "-i",
    ]
    if sys.stdin.isatty() and sys.stdout.isatty():
        docker_exec[-1] = "-it"
    docker_exec.extend([CONTAINER, *container_arguments(args)])

    if args.dry_run:
        print(f"[dry-run] {printable(start_docker)}")
        print(f"[dry-run] {printable(start_container)}")
        print(
            "[dry-run] Jackal SSH: source "
            f"{JACKAL_SETUP} && source "
            f"{JACKAL_WORKSPACE}/install/setup.bash"
        )
        print(
            "[dry-run] Jackal SSH: enable RealSense depth, sync, "
            "and aligned depth"
        )
        print(
            "[dry-run] Jackal SSH: ros2 launch jackal_nav "
            "localisation.launch.py "
            f"map:={args.map_file}"
        )
        print(f"[dry-run] {printable(docker_exec)}")
        return 0

    password = jackal_password()
    remote_process: Optional[subprocess.Popen] = None
    remote_log_handle: Optional[object] = None
    remote_log_path: Optional[Path] = None
    child: Optional[subprocess.Popen] = None
    run_checked(start_docker)
    ensure_wsl_jackal_route()
    sync_jackal_time(password)
    preflight_jackal_localization(password, args.map_file)
    run_checked(start_container)
    prepare_container_desktop()
    stop_jackal_localization(password, None)
    enable_jackal_rgbd(password)
    remote_process, remote_log_handle, remote_log_path = start_jackal_localization(
        password,
        args.map_file,
    )

    try:
        if args.open_browser and not args.no_browser and not args.no_rviz:
            if wait_for_vnc():
                webbrowser.open(VNC_URL)
            else:
                print(f"[warn] VNC did not answer yet: {VNC_URL}", flush=True)

        print("[host] Attached Docker exec keeps WSL alive during the test.")
        print("[host] Press Ctrl+C once to stop the complete stack cleanly.")
        child = subprocess.Popen(docker_exec)
        try:
            while True:
                child_status = child.poll()
                if child_status is not None:
                    return child_status
                remote_status = remote_process.poll()
                if remote_status is not None:
                    if remote_log_handle is not None:
                        remote_log_handle.flush()
                    remote_log = Path(os.environ.get("TEMP", ".")) / (
                        "run_jackal_robohub.localization.log"
                    )
                    detail = remote_log.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                    raise LaunchError(
                        "Jackal onboard localization stopped unexpectedly: "
                        f"{detail or f'exit code {remote_status}'}"
                    )
                time.sleep(1)
        except KeyboardInterrupt:
            stop_command = [
                "wsl",
                "-d",
                WSL_DISTRO,
                "--",
                "docker",
                "exec",
                CONTAINER,
                "bash",
                "-lc",
                f"test -r {PID_FILE} && kill -INT $(cat {PID_FILE})",
            ]
            subprocess.run(stop_command, check=False)
            try:
                return child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.terminate()
                return 130
        except LaunchError:
            stop_command = [
                "wsl",
                "-d",
                WSL_DISTRO,
                "--",
                "docker",
                "exec",
                CONTAINER,
                "bash",
                "-lc",
                f"test -r {PID_FILE} && kill -INT $(cat {PID_FILE})",
            ]
            subprocess.run(stop_command, check=False)
            if child.poll() is None:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.terminate()
            raise
    finally:
        stop_jackal_localization(password, remote_process)
        if remote_log_handle is not None:
            remote_log_handle.close()
        if remote_log_path is not None:
            for attempt in range(20):
                try:
                    remote_log_path.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 19:
                        print(
                            f"[warn] Could not remove temporary log: "
                            f"{remote_log_path}",
                            flush=True,
                        )
                    else:
                        time.sleep(0.25)


class RealJackalLauncher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.processes: dict[str, ManagedProcess] = {}
        self.stop_requested = False
        self.owns_pid_file = False

    def preflight(self) -> None:
        required = [
            ROS_SETUP,
            HUNAV_SETUP,
            PIPELINE_SETUP,
            VENV / "bin/python",
            ACADOS / "lib/libacados.so",
            DDS_PROFILE,
            TEST_MODE_CHECKPOINT if self.args.test_mode else CHECKPOINT,
        ]
        if self.args.test_mode:
            required.append(TEST_MODE_NORM)
        missing = [str(path) for path in required if not Path(path).exists()]
        if missing:
            raise LaunchError("Missing required paths:\n  " + "\n  ".join(missing))

        if PID_FILE.exists():
            try:
                old_pid = int(PID_FILE.read_text(encoding="ascii").strip())
                if old_pid == os.getpid():
                    raise ProcessLookupError
                os.kill(old_pid, 0)
                old_cmdline = Path(f"/proc/{old_pid}/cmdline").read_bytes()
            except (OSError, ValueError):
                PID_FILE.unlink(missing_ok=True)
            else:
                if b"run_jackal_robohub.py" in old_cmdline:
                    raise LaunchError(
                        "Another real-test launcher is running with "
                        f"PID {old_pid}."
                    )
                PID_FILE.unlink(missing_ok=True)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()), encoding="ascii")
        self.owns_pid_file = True

    def start(
        self,
        name: str,
        command: str,
        run_as_ubuntu: bool = False,
    ) -> ManagedProcess:
        log_path = LOG_DIR / f"{name}.log"
        handle = log_path.open("w", encoding="utf-8", buffering=1)
        argv = ["bash", "-lc", "set -eo pipefail\n" + command]
        if run_as_ubuntu and os.geteuid() == 0:
            argv = ["runuser", "-u", "ubuntu", "--", *argv]
        print(f"[start] {name}  log={log_path}", flush=True)
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        managed = ManagedProcess(
            name=name,
            process=process,
            process_group=os.getpgid(process.pid),
            log_path=log_path,
            log_handle=handle,
        )
        self.processes[name] = managed
        return managed

    def wait_for_markers(
        self,
        process: ManagedProcess,
        markers: Iterable[str],
        timeout_sec: float,
    ) -> None:
        remaining = set(markers)
        deadline = time.monotonic() + timeout_sec
        while remaining and time.monotonic() < deadline:
            if self.stop_requested:
                raise KeyboardInterrupt
            if not process.alive:
                self.print_tail(process)
                raise LaunchError(
                    f"{process.name} exited with code {process.process.returncode}."
                )
            text = process.log_path.read_text(
                encoding="utf-8", errors="replace"
            )
            found = sorted(marker for marker in remaining if marker in text)
            for marker in found:
                print(f"[ready] {marker}", flush=True)
                remaining.remove(marker)
            time.sleep(0.5)
        if remaining:
            self.print_tail(process)
            raise LaunchError(
                f"Timed out waiting for {process.name}: {sorted(remaining)}"
            )

    def assert_alive(self) -> None:
        dead = [item for item in self.processes.values() if not item.alive]
        if not dead:
            return
        for process in dead:
            self.print_tail(process)
        raise LaunchError(
            "Processes exited: " + ", ".join(item.name for item in dead)
        )

    @staticmethod
    def stop_desktop_background_work() -> None:
        for pattern in (
            "/usr/lib/update-notifier/list-oem-metapackages",
            "^update-notifier$",
            "^mate-screensaver$",
            "^caja( |$)",
            "/evolution-",
        ):
            subprocess.run(
                ["pkill", "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def ros_capture(
        self,
        label: str,
        command: str,
        timeout_sec: float,
    ) -> tuple[int, str]:
        print(f"[check] {label}", flush=True)
        try:
            result = subprocess.run(
                ["bash", "-lc", f"set -eo pipefail\n{ROS_ENV}\n{command}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            raise LaunchError(
                f"Timed out during {label} after {timeout_sec:.0f} seconds.\n"
                f"{output.strip()}"
            ) from exc
        output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        return result.returncode, output

    def read_clock_skew(self) -> float:
        returncode, output = self.ros_capture(
            "Jackal clock skew",
            "ros2 topic echo /jackal1/platform/odom/filtered "
            "--once --field header",
            15,
        )
        if returncode != 0:
            raise LaunchError(
                "Could not read a timestamp from Jackal filtered odometry.\n"
                + output
            )
        sec_match = re.search(r"\bsec:\s*(-?\d+)", output)
        nanosec_match = re.search(r"\bnanosec:\s*(\d+)", output)
        if not sec_match or not nanosec_match:
            raise LaunchError(
                "Jackal odometry did not contain a parseable ROS timestamp.\n"
                + output
            )
        robot_time = float(sec_match.group(1)) + (
            float(nanosec_match.group(1)) / 1_000_000_000.0
        )
        skew = robot_time - time.time()
        print(
            f"[check] Jackal clock skew={skew:+.3f} s "
            f"(limit={self.args.clock_skew_limit:.3f} s)",
            flush=True,
        )
        return skew

    def check_clock_skew(self) -> None:
        skew = self.read_clock_skew()
        if abs(skew) > self.args.clock_skew_limit:
            raise LaunchError(
                "Jackal and host clocks are not synchronized "
                f"(skew={skew:+.3f} s). Localization and the GPU policy "
                "were not started."
            )

    def wait_for_valid_map(self) -> None:
        map_topic = shlex.quote(self.args.map_topic)
        returncode, output = self.ros_capture(
            f"first valid {self.args.map_topic} message",
            f"ros2 topic echo {map_topic} nav_msgs/msg/OccupancyGrid --once "
            "--field info --qos-reliability reliable "
            "--qos-durability transient_local --qos-history keep_last "
            "--qos-depth 1",
            self.args.map_timeout,
        )
        if returncode != 0:
            raise LaunchError(
                f"Could not receive {self.args.map_topic}.\n" + output
            )
        width_match = re.search(r"\bwidth:\s*(\d+)", output)
        height_match = re.search(r"\bheight:\s*(\d+)", output)
        if not width_match or not height_match:
            raise LaunchError("Received malformed /map metadata.\n" + output)
        width = int(width_match.group(1))
        height = int(height_match.group(1))
        if width <= 0 or height <= 0:
            raise LaunchError(
                f"Received an empty {self.args.map_topic} "
                f"({width} x {height})."
            )
        print(
            f"[ready] {self.args.map_topic} is valid ({width} x {height})",
            flush=True,
        )

    def wait_for_localization_tf(self) -> None:
        deadline = time.monotonic() + self.args.localization_timeout
        last_output = ""
        while time.monotonic() < deadline:
            returncode, output = self.ros_capture(
                "localization map to base_link TF",
                "timeout 5s ros2 run tf2_ros tf2_echo map base_link "
                "--ros-args -r /tf:=/jackal1/tf "
                "-r /tf_static:=/jackal1/tf_static",
                8,
            )
            last_output = output
            if "Translation:" in output and "Rotation:" in output:
                print(
                    "[ready] localization map -> base_link TF is valid",
                    flush=True,
                )
                return
            if returncode not in (0, 124):
                raise LaunchError("TF validation failed.\n" + output)
        raise LaunchError(
            "No usable map -> base_link transform was received within "
            f"{self.args.localization_timeout:.0f} seconds. Use RViz 2D Pose "
            "Estimate and retry.\n"
            + last_output
        )

    def wait_for_people_detector_output(self, timeout_sec: float = 45.0) -> None:
        deadline = time.monotonic() + timeout_sec
        last_status = "no status received"
        while time.monotonic() < deadline:
            returncode, output = self.ros_capture(
                "active RGB-D people detector status",
                "timeout 4s ros2 topic echo /people_detector/status "
                "std_msgs/msg/String --once --full-length",
                7,
            )
            if output:
                last_status = output
            if returncode == 0 and '"ready": true' in output.lower():
                people_returncode, people_output = self.ros_capture(
                    "first live /people message",
                    "timeout 4s ros2 topic echo /people "
                    "people_msgs/msg/People --once",
                    7,
                )
                if people_returncode == 0:
                    print(
                        "[ready] RGB-D detector is publishing live /people",
                        flush=True,
                    )
                    return
                if people_output:
                    last_status = people_output
            time.sleep(0.5)
        raise LaunchError(
            "RGB-D detector did not publish a ready status and live /people "
            f"within {timeout_sec:.0f} seconds. Vehicle output was not enabled.\n"
            + last_status
        )

    def commands(self) -> tuple[str, str, str, str]:
        map_topic = shlex.quote(self.args.map_topic)
        test_mode = "true" if self.args.test_mode else "false"
        record_bag = "true" if self.args.record_bag else "false"
        style_vector = shlex.quote(
            "[" + ", ".join(str(value) for value in self.args.style_vector) + "]"
        )
        policy = f"""
{ROS_ENV}
cd {PIPELINE}
exec ros2 launch social_nav_diffusion_ros \
  jackal_realtime_social_nav_debug.launch.py \
  start_people_detector:=false \
  start_policy:=true \
  start_goal_bridge:=true \
  map_topic:={map_topic} \
  test_mode:={test_mode} \
  start_ps4_trigger:={test_mode} \
  style_vector:={style_vector} \
  goal_distance_m:={self.args.goal_distance_m} \
  trigger_button_index:={self.args.trigger_button_index} \
  record_bag:={record_bag} \
  start_rviz:=false
""".strip()
        people = f"""
{ROS_ENV}
cd {PIPELINE}
exec ros2 launch social_nav_diffusion_ros \
  jackal_realtime_social_nav_debug.launch.py \
  start_people_detector:=true \
  start_policy:=false \
  start_goal_bridge:=false \
  test_mode:={test_mode} \
  start_ps4_trigger:=false \
  start_rviz:=false
""".strip()
        adapter = f"""
{ROS_ENV}
exec ros2 launch social_nav_diffusion_ros \
  jackal_twist_adapter.launch.py \
  enable_output:=true \
  enable_lidar_safety:=true \
  lidar_topic:=/jackal1/sensors/lidar3d_0/scan \
  max_linear_speed:={self.args.max_linear_speed} \
  max_angular_speed:={self.args.max_angular_speed} \
  test_mode:={test_mode} \
  require_nav_trigger:={test_mode}
""".strip()
        rviz_config = RVIZ_CONFIG
        rviz = f"""
{ROS_ENV}
export DISPLAY=:1
export XAUTHORITY=/home/ubuntu/.Xauthority
export XAUTHLOCALHOSTNAME=AlbertLudwig
exec rviz2 -d {rviz_config} --ros-args \
  -r __ns:=/jackal1 \
  -r /tf:=/jackal1/tf \
  -r /tf_static:=/jackal1/tf_static
""".strip()
        return policy, people, adapter, rviz

    def run(self) -> None:
        self.preflight()
        (
            policy_command,
            people_command,
            adapter_command,
            rviz_command,
        ) = self.commands()
        self.check_clock_skew()
        if wait_for_vnc(5):
            print(
                f"[ready] noVNC is listening on port {VNC_PORT}",
                flush=True,
            )
        else:
            novnc = self.start(
                "novnc",
                "exec websockify --web=/usr/lib/novnc "
                f"127.0.0.1:{VNC_PORT} localhost:5901",
                run_as_ubuntu=True,
            )
            self.wait_for_markers(
                novnc,
                [f"Listen on 127.0.0.1:{VNC_PORT}"],
                15,
            )

        self.wait_for_valid_map()

        if not self.args.no_rviz:
            rviz = self.start("rviz", rviz_command, run_as_ubuntu=True)
            self.wait_for_markers(rviz, ["OpenGl version"], 30)
            self.stop_desktop_background_work()
            print(
                "[action] If localization has no initial pose, use RViz "
                "2D Pose Estimate now.",
                flush=True,
            )

        self.wait_for_localization_tf()

        policy = self.start("policy", policy_command)
        self.wait_for_markers(
            policy,
            ["Policy ready for real goals"],
            self.args.warmup_timeout,
        )

        people = self.start("people", people_command)
        self.wait_for_markers(
            people,
            ["RGB-D people detector ready"],
            45,
        )
        self.wait_for_people_detector_output()

        adapter = self.start("adapter", adapter_command)
        self.wait_for_markers(
            adapter,
            ["Jackal output ENABLED"],
            20,
        )

        print("\n[READY] Real Jackal stack is running.", flush=True)
        print(f"[READY] RViz/VNC: {VNC_URL}", flush=True)
        print("[READY] Use RViz Nav2 Goal. Press Ctrl+C to stop.", flush=True)
        print(
            "[READY] Speed limits: "
            f"linear={self.args.max_linear_speed} m/s, "
            f"angular={self.args.max_angular_speed} rad/s",
            flush=True,
        )

        while not self.stop_requested:
            self.assert_alive()
            time.sleep(1)

    @staticmethod
    def process_group_exists(process: ManagedProcess) -> bool:
        try:
            os.killpg(process.process_group, 0)
            return True
        except ProcessLookupError:
            return False

    def stop_process(self, name: str, timeout_sec: float = 8.0) -> None:
        process = self.processes.get(name)
        if process is None:
            return
        try:
            os.killpg(process.process_group, signal.SIGINT)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout_sec
        while self.process_group_exists(process) and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.process_group_exists(process):
            try:
                os.killpg(process.process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 3
        while self.process_group_exists(process) and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.process_group_exists(process):
            try:
                os.killpg(process.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def stop_all(self) -> None:
        print("\n[shutdown] stopping policy so the adapter watchdog sends zero")
        self.stop_process("policy", timeout_sec=10)
        time.sleep(1)
        self.stop_process("adapter", timeout_sec=5)
        self.stop_process("people", timeout_sec=5)
        self.stop_process("rviz", timeout_sec=5)
        self.stop_process("novnc", timeout_sec=3)
        for process in self.processes.values():
            process.log_handle.close()
        if self.owns_pid_file:
            PID_FILE.unlink(missing_ok=True)
            self.owns_pid_file = False

    @staticmethod
    def print_tail(process: ManagedProcess, lines: int = 50) -> None:
        try:
            content = process.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            print(f"[log] unable to read {process.log_path}: {exc}")
            return
        print(f"\n----- {process.name}.log -----")
        print("\n".join(content[-lines:]))

    def handle_signal(self, signum: int, _frame: object) -> None:
        print(f"\n[signal] received {signum}", flush=True)
        self.stop_requested = True


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the real Jackal SocialNavDiffusion stack."
    )
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open VNC in the default external browser",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-linear-speed", type=float, default=1.0)
    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=3.14,
    )
    parser.add_argument("--warmup-timeout", type=float, default=150.0)
    parser.add_argument("--clock-skew-limit", type=float, default=2.0)
    parser.add_argument("--map-timeout", type=float, default=60.0)
    parser.add_argument("--localization-timeout", type=float, default=600.0)
    parser.add_argument("--map-file", default=str(DEFAULT_MAP_FILE))
    parser.add_argument("--map-topic", default=DEFAULT_MAP_TOPIC)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--test-mode",
        dest="test_mode",
        action="store_true",
        help="enable the teammate JackalUpdate08-18 experiment path",
    )
    mode_group.add_argument(
        "--no-test-mode",
        dest="test_mode",
        action="store_false",
        help="force the stable4 path even when runtime_mode.yaml enables tests",
    )
    parser.set_defaults(test_mode=configured_test_mode())
    parser.add_argument(
        "--style-vector",
        nargs=4,
        type=float,
        metavar=("PROX", "PASS", "YIELD", "GROUP"),
        default=[0.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument("--goal-distance-m", type=float, default=6.0)
    parser.add_argument("--trigger-button-index", type=int, default=7)
    parser.add_argument(
        "--no-record-bag",
        dest="record_bag",
        action="store_false",
        help="disable per-run MCAP recording in test mode",
    )
    parser.set_defaults(record_bag=True)
    args = parser.parse_args(argv)
    if args.max_linear_speed <= 0 or args.max_angular_speed <= 0:
        parser.error("speed limits must be positive")
    if args.warmup_timeout <= 0:
        parser.error("--warmup-timeout must be positive")
    if args.clock_skew_limit <= 0:
        parser.error("--clock-skew-limit must be positive")
    if args.map_timeout <= 0:
        parser.error("--map-timeout must be positive")
    if args.localization_timeout <= 0:
        parser.error("--localization-timeout must be positive")
    if not PurePosixPath(args.map_file).is_absolute():
        parser.error("--map-file must be an absolute path on the Jackal")
    if not args.map_topic.startswith("/"):
        parser.error("--map-topic must be an absolute ROS topic")
    if args.goal_distance_m <= 0:
        parser.error("--goal-distance-m must be positive")
    if args.trigger_button_index < 0:
        parser.error("--trigger-button-index must be non-negative")
    if any(value < -1.0 or value > 1.0 for value in args.style_vector):
        parser.error("--style-vector values must be within [-1, 1]")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if not args.inside_container:
        if os.name != "nt":
            raise LaunchError(
                "Run this file from Windows PowerShell, or pass "
                "--inside-container from jackal_robohub."
            )
        try:
            return run_from_windows(args)
        except LaunchError as exc:
            print(f"\n[FAIL] {exc}", file=sys.stderr)
            return 1

    launcher = RealJackalLauncher(args)
    if args.dry_run:
        for name, command in zip(
            ("policy", "people", "adapter", "rviz"),
            launcher.commands(),
        ):
            print(f"[dry-run:{name}]\n{command}\n")
        return 0
    signal.signal(signal.SIGINT, launcher.handle_signal)
    signal.signal(signal.SIGTERM, launcher.handle_signal)
    try:
        launcher.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except LaunchError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        launcher.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
