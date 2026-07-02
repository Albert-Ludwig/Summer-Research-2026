import configparser
import csv
import copy
import math
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

INFERENCE_REPO_DEFAULT = "/workspace/SocialNavDiffusion_Inference"
VENV_PYTHON_DEFAULT = f"{INFERENCE_REPO_DEFAULT}/.venv/bin/python"


def wants_diffusion_policy(argv: List[str]) -> bool:
    return any("use_diffusion_policy:=true" in arg.lower() for arg in argv)


def maybe_reexec_for_diffusion():
    if not wants_diffusion_policy(sys.argv):
        return
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PYTHON_DEFAULT):
        return
    if sys.prefix == f"{INFERENCE_REPO_DEFAULT}/.venv":
        return
    if not os.path.exists(VENV_PYTHON_DEFAULT):
        print(
            f"[policy_cmd_vel_node] diffusion requested but venv python is missing: {VENV_PYTHON_DEFAULT}",
            flush=True,
        )
        return

    env = os.environ.copy()
    env["ACADOS_SOURCE_DIR"] = "/home/ubuntu/acados"
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":/home/ubuntu/acados/lib"
    extra_pythonpath = [
        INFERENCE_REPO_DEFAULT,
        f"{INFERENCE_REPO_DEFAULT}/diffusers_unet_1d_condition",
        "/home/ubuntu/hunav_jazzy_ws/install/social_nav_diffusion_ros/lib/python3.12/site-packages",
        "/home/ubuntu/hunav_jazzy_ws/install/people_msgs/lib/python3.12/site-packages",
    ]
    env["PYTHONPATH"] = ":".join(extra_pythonpath + [env.get("PYTHONPATH", "")])
    print(f"[policy_cmd_vel_node] re-exec into diffusion venv: {VENV_PYTHON_DEFAULT}", flush=True)
    os.execve(VENV_PYTHON_DEFAULT, [VENV_PYTHON_DEFAULT] + sys.argv, env)


maybe_reexec_for_diffusion()

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from people_msgs.msg import People
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_policy_action(
    robot_state: Dict[str, float],
    people_states: List[Dict[str, float]],
    goal_state: Dict[str, float],
    occupancy: Optional[Dict],
    last_cmd: Dict[str, float],
    limits: Dict[str, float],
) -> Tuple[float, float]:
    """Temporary placeholder policy.

    Replace only this function with the learned/planning policy once it is ready.
    Inputs are deliberately plain dictionaries so the ROS wrapper stays stable.
    """
    del occupancy  # Occupancy is passed through for the real policy interface.

    distance = goal_state["distance"]
    heading_error = goal_state["heading_error"]

    v = min(limits["v_max"], 0.75 * distance)
    w = clamp(1.8 * heading_error, limits["w_min"], limits["w_max"])

    if abs(heading_error) > 1.0:
        v *= 0.35
    elif abs(heading_error) > 0.55:
        v *= 0.65

    for human in people_states:
        hx = human["rel_x"]
        hy = human["rel_y"]
        dist = human["distance"]
        in_front = hx > 0.0 and abs(hy) < limits["front_width"]
        if in_front and dist < limits["stop_distance"]:
            v = 0.0
            break
        if in_front and dist < limits["slow_distance"]:
            scale = clamp((dist - limits["stop_distance"]) / (limits["slow_distance"] - limits["stop_distance"]), 0.15, 1.0)
            v *= scale

    # Smooth the placeholder command a little so handoff to direct control is gentle.
    max_dv = limits["max_linear_accel"] * limits["dt"]
    v = clamp(v, last_cmd["v"] - max_dv, last_cmd["v"] + max_dv)

    return clamp(v, limits["v_min"], limits["v_max"]), clamp(w, limits["w_min"], limits["w_max"])


class SocialNavDiffusionPolicyAdapter:
    def __init__(
        self,
        inference_repo: str,
        config_path: str,
        env_config_path: str,
        checkpoint_path: str,
        norm_path: str,
        logger,
    ):
        self.inference_repo = inference_repo
        self.config_path = config_path
        self.env_config_path = env_config_path
        self.checkpoint_path = checkpoint_path
        self.norm_path = norm_path
        self.logger = logger
        self.policy = None
        self.FullState = None
        self.JointState = None
        self.ObservableState = None
        self.last_action_fields: Any = None
        self.last_action_type = "unknown"
        self.last_raw_model_v = float("nan")
        self.last_raw_model_r_or_w = float("nan")
        self.load_policy()

    def load_policy(self):
        policy_dir = os.path.join(self.inference_repo, "crowd_nav", "policy")
        paths = (
            self.inference_repo,
            policy_dir,
            os.path.join(self.inference_repo, "diffusers_unet_1d_condition"),
        )
        for path in paths:
            if path not in sys.path:
                sys.path.insert(0, path)

        os.environ.setdefault("ACADOS_SOURCE_DIR", "/home/ubuntu/acados")
        os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/home/ubuntu/acados/lib"

        from crowd_nav.policy.diffusion_CondUNetCFG import DiffusionConditionalUNet1DCFG
        from crowd_sim.envs.utils.state import FullState, JointState, ObservableState

        config = configparser.RawConfigParser()
        read_files = config.read(self.config_path)
        if not read_files:
            raise FileNotFoundError(f"policy config not readable: {self.config_path}")

        section = "diffusion_conditional_unet1dcfg"
        config = copy.deepcopy(config)
        config.set(section, "ckpt_path", self.checkpoint_path)
        config.set(section, "norm_file", self.norm_path)

        policy = DiffusionConditionalUNet1DCFG()
        policy.configure(config)

        env_config = configparser.RawConfigParser()
        env_config.read(self.env_config_path)
        policy.time_step = (
            env_config.getfloat("env", "time_step")
            if env_config.has_option("env", "time_step")
            else 0.25
        )
        policy.set_static_map(None, 0.0, policy.map_extent)

        self.policy = policy
        self.FullState = FullState
        self.JointState = JointState
        self.ObservableState = ObservableState
        self.logger.info(
            "diffusion model loaded: "
            f"config={self.config_path}, env_config={self.env_config_path}, "
            f"checkpoint={self.checkpoint_path}, norm={self.norm_path}, "
            f"python={sys.executable}"
        )

    def sync_policy_state_from_odom(self, robot_state: Dict[str, float]):
        v = float(robot_state.get("linear_velocity", 0.0))
        w = float(robot_state.get("angular_velocity", 0.0))
        if robot_state.get("_sync_policy_warm_start_from_odom"):
            if hasattr(self.policy, "_v0_from_last_step"):
                self.policy._v0_from_last_step = v
            if hasattr(self.policy, "_w0_from_last_step"):
                self.policy._w0_from_last_step = w
        if robot_state.get("_sync_prev_action_from_odom"):
            if hasattr(self.policy, "prev_v"):
                self.policy.prev_v = v
            if hasattr(self.policy, "prev_omega"):
                self.policy.prev_omega = w

    def build_state(
        self,
        robot_state: Dict[str, float],
        people_states: List[Dict[str, float]],
        goal_state: Dict[str, float],
        limits: Dict[str, float],
    ):
        yaw = robot_state["yaw"]
        v_body = robot_state.get("linear_velocity", 0.0)
        vx_world = v_body * math.cos(yaw)
        vy_world = v_body * math.sin(yaw)

        robot = self.FullState(
            px=float(robot_state["x"]),
            py=float(robot_state["y"]),
            vx=float(vx_world),
            vy=float(vy_world),
            radius=float(limits["robot_radius"]),
            gx=float(goal_state["x"]),
            gy=float(goal_state["y"]),
            v_pref=float(limits["robot_v_pref"]),
            theta=float(yaw),
            omega=float(robot_state.get("angular_velocity", 0.0)),
        )

        humans = [
            self.ObservableState(
                px=float(human["x"]),
                py=float(human["y"]),
                vx=float(human.get("vx", 0.0)),
                vy=float(human.get("vy", 0.0)),
                radius=float(limits["human_radius"]),
            )
            for human in people_states
        ]
        return self.JointState(robot, humans)

    def action_to_cmd(self, action: Any, robot_state: Dict[str, float], limits: Dict[str, float]) -> Tuple[float, float]:
        if hasattr(action, "_asdict"):
            fields = dict(action._asdict())
        elif hasattr(action, "__dict__"):
            fields = dict(vars(action))
        elif isinstance(action, (tuple, list)) and len(action) >= 2:
            fields = {"v": action[0], "r": action[1]}
        else:
            raise TypeError(f"unsupported diffusion action type: {type(action).__name__}")

        self.last_action_fields = fields
        self.last_action_type = "unknown"
        self.last_raw_model_v = float("nan")
        self.last_raw_model_r_or_w = float("nan")
        if "v" in fields and "r" in fields:
            self.last_action_type = "ActionRot"
            v = float(fields["v"])
            raw_r = float(fields["r"])
            self.last_raw_model_v = v
            self.last_raw_model_r_or_w = raw_r
            model_dt = float(getattr(self.policy, "time_step", limits["dt"])) or limits["dt"]
            w = raw_r / model_dt
        elif "v" in fields and "w" in fields:
            self.last_action_type = "ActionRot"
            v = float(fields["v"])
            w = float(fields["w"])
            self.last_raw_model_v = v
            self.last_raw_model_r_or_w = w
        elif "vx" in fields and "vy" in fields:
            self.last_action_type = "ActionXY"
            vx = float(fields["vx"])
            vy = float(fields["vy"])
            self.last_raw_model_v = math.hypot(vx, vy)
            self.last_raw_model_r_or_w = float("nan")
            yaw = robot_state["yaw"]
            v = math.cos(yaw) * vx + math.sin(yaw) * vy
            speed = math.hypot(vx, vy)
            if speed > 1e-6:
                desired_yaw = math.atan2(vy, vx)
                w = wrap_angle(desired_yaw - yaw) / max(limits["dt"], 1e-3)
            else:
                w = 0.0
        else:
            raise TypeError(f"unsupported diffusion action fields: {fields}")

        return clamp(v, limits["v_min"], limits["v_max"]), clamp(w, limits["w_min"], limits["w_max"])

    def compute_action(
        self,
        robot_state: Dict[str, float],
        people_states: List[Dict[str, float]],
        goal_state: Dict[str, float],
        occupancy: Optional[Dict],
        last_cmd: Dict[str, float],
        limits: Dict[str, float],
    ) -> Tuple[float, float]:
        del occupancy, last_cmd
        self.sync_policy_state_from_odom(robot_state)
        state = self.build_state(robot_state, people_states, goal_state, limits)
        action = self.policy.predict(state)
        return self.action_to_cmd(action, robot_state, limits)


class PolicyCmdVelNode(Node):
    def __init__(self):
        super().__init__("policy_cmd_vel_node")

        self.declare_parameter("control_mode", "raw_eval")
        self.declare_parameter("enable_goal_stop", True)
        self.declare_parameter("enable_speed_clamp", True)
        self.declare_parameter("enable_near_goal_slowdown", False)
        self.declare_parameter("enable_heading_gate", False)
        self.declare_parameter("enable_heading_stop", False)
        self.declare_parameter("enable_heading_align_override", False)
        self.declare_parameter("enable_speed_gain", False)
        self.declare_parameter("debug_verbose_state", False)
        self.declare_parameter("debug_csv_path", "")
        self.declare_parameter("ignore_people_for_policy", False)
        self.declare_parameter("ignore_map_for_policy", False)
        self.declare_parameter("force_zero_humans", False)
        self.declare_parameter("fixed_test_goal_in_robot_frame", False)
        self.declare_parameter("fixed_test_goal_x", 1.0)
        self.declare_parameter("fixed_test_goal_y", 0.0)
        self.declare_parameter("disable_policy_command_publish", False)
        self.declare_parameter("enable_sign_conflict_guard", False)
        self.declare_parameter("sign_conflict_heading_error_rad", 0.35)
        self.declare_parameter("sign_conflict_min_raw_angular", 0.10)
        self.declare_parameter("sign_conflict_linear_scale", 0.0)
        self.declare_parameter("sign_conflict_align_kp", 0.6)
        self.declare_parameter("sign_conflict_min_abs_w", 0.08)
        self.declare_parameter("sign_conflict_max_abs_w", 0.8)
        self.declare_parameter("robot_v_pref", 1.0)
        self.declare_parameter("robot_radius", 0.25)
        self.declare_parameter("human_radius", 0.25)
        self.declare_parameter("sync_policy_warm_start_from_odom", True)
        self.declare_parameter("sync_prev_action_from_odom", True)
        self.declare_parameter("control_period_sec", 0.1)
        self.declare_parameter("cmd_publish_period_sec", 0.1)
        self.declare_parameter("diffusion_inference_period_sec", 0.5)
        self.declare_parameter("command_hold_timeout_sec", 2.0)
        self.declare_parameter("v_max", 1.0)
        self.declare_parameter("w_max", 1.2)
        self.declare_parameter("max_linear_speed", 0.4)
        self.declare_parameter("max_angular_speed", 0.8)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("goal_timeout_sec", 60.0)
        self.declare_parameter("stop_when_goal_reached", True)
        self.declare_parameter("slow_down_radius", 1.2)
        self.declare_parameter("min_approach_speed", 0.04)
        self.declare_parameter("heading_gate_rad", 0.8)
        self.declare_parameter("heading_stop_rad", 1.4)
        self.declare_parameter("near_goal_angular_scale", 0.6)
        self.declare_parameter("heading_align_kp", 0.8)
        self.declare_parameter("heading_align_min_abs_w", 0.10)
        self.declare_parameter("heading_align_override_rad", 0.8)
        self.declare_parameter("heading_align_override_near_goal_only", False)
        self.declare_parameter("stale_timeout_sec", 1.0)
        self.declare_parameter("slow_distance", 1.6)
        self.declare_parameter("stop_distance", 0.75)
        self.declare_parameter("front_width", 0.8)
        self.declare_parameter("max_linear_accel", 0.6)
        self.declare_parameter("cmd_frame_id", "base_link")
        self.declare_parameter("use_diffusion_policy", False)
        self.declare_parameter("diffusion_inference_repo", INFERENCE_REPO_DEFAULT)
        self.declare_parameter("diffusion_config_path", f"{INFERENCE_REPO_DEFAULT}/crowd_nav/configs/policy.config")
        self.declare_parameter("diffusion_env_config_path", f"{INFERENCE_REPO_DEFAULT}/crowd_nav/configs/env.config")
        self.declare_parameter("diffusion_checkpoint_path", f"{INFERENCE_REPO_DEFAULT}/SocialGuidedNavPlanner.pt")
        self.declare_parameter("diffusion_norm_path", f"{INFERENCE_REPO_DEFAULT}/norm_stats_SOCIAL_NORMS8.npy")

        self.control_period = float(self.get_parameter("control_period_sec").value)
        self.cmd_publish_period = float(self.get_parameter("cmd_publish_period_sec").value)
        self.diffusion_inference_period = float(self.get_parameter("diffusion_inference_period_sec").value)
        self.command_hold_timeout = float(self.get_parameter("command_hold_timeout_sec").value)
        self.cmd_frame_id = str(self.get_parameter("cmd_frame_id").value)
        self.use_diffusion_policy = bool(self.get_parameter("use_diffusion_policy").value)
        self.diffusion_adapter: Optional[SocialNavDiffusionPolicyAdapter] = None
        self._last_source_log_time = 0.0
        self._last_hold_log_time = 0.0
        self._inference_lock = threading.Lock()
        self._latest_policy_cmd: Optional[Dict[str, float]] = None
        self._diffusion_thread: Optional[threading.Thread] = None
        self._diffusion_inference_running = False

        if self.use_diffusion_policy:
            try:
                self.diffusion_adapter = SocialNavDiffusionPolicyAdapter(
                    inference_repo=str(self.get_parameter("diffusion_inference_repo").value),
                    config_path=str(self.get_parameter("diffusion_config_path").value),
                    env_config_path=str(self.get_parameter("diffusion_env_config_path").value),
                    checkpoint_path=str(self.get_parameter("diffusion_checkpoint_path").value),
                    norm_path=str(self.get_parameter("diffusion_norm_path").value),
                    logger=self.get_logger(),
                )
            except Exception as exc:
                self.get_logger().error(
                    f"diffusion model failed to load; falling back to placeholder policy: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                self.use_diffusion_policy = False
        else:
            self.get_logger().info("use_diffusion_policy=false: using placeholder goal-seeking policy")

        self.latest_odom: Optional[Odometry] = None
        self.latest_odom_time: Optional[float] = None
        self.latest_people: List = []
        self.latest_people_time: Optional[float] = None
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_map_time: Optional[float] = None
        self.latest_goal: Optional[PoseStamped] = None
        self.latest_goal_time: Optional[float] = None
        self.last_cmd = {"v": 0.0, "w": 0.0}
        self.last_raw_cmd = {"v": 0.0, "w": 0.0}
        self.last_final_cmd = {"v": 0.0, "w": 0.0}
        self.last_cmd_clamped = {"linear": False, "angular": False}
        self.last_command_source = "no_goal"
        self.last_approach_scale = float("nan")
        self.last_allowed_linear_near_goal = float("nan")
        self.last_heading_gate_active = False
        self.last_heading_stop_active = False
        self.last_near_goal_angular_damping_active = False
        self.last_heading_align_override_active = False
        self.last_heading_align_desired_w = float("nan")
        self.last_humans_used = 0
        self.last_policy_goal_x_map = float("nan")
        self.last_policy_goal_y_map = float("nan")
        self.last_goal_dx_map = float("nan")
        self.last_goal_dy_map = float("nan")
        self.last_goal_x_robot_frame = float("nan")
        self.last_goal_y_robot_frame = float("nan")
        self.last_distance_to_goal = float("nan")
        self.last_heading_to_goal = float("nan")
        self.last_heading_error = float("nan")
        self.last_raw_action_type = "unknown"
        self.last_raw_model_v_before_conversion = float("nan")
        self.last_raw_model_r_or_w_before_conversion = float("nan")
        self.last_converted_cmd = {"v": 0.0, "w": 0.0}
        self.last_odom_linear_velocity_used_for_sync = float("nan")
        self.last_odom_angular_velocity_used_for_sync = float("nan")
        self._debug_csv_header_written = False
        self.last_sign_conflict_guard_active = False
        self.last_sign_conflict_desired_w = float("nan")

        self.create_subscription(People, "/people", self.people_callback, 10)
        self.create_subscription(Odometry, "/cpr_j100_0001/platform/odom/filtered", self.odom_callback, 20)
        self.create_subscription(OccupancyGrid, "/cpr_j100_0001/map", self.map_callback, 1)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.create_subscription(PoseStamped, "/cpr_j100_0001/goal_pose", self.goal_callback, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, "/cpr_j100_0001/cmd_vel", 10)
        self.active_goal_marker_pub = self.create_publisher(Marker, "/social_nav_diffusion/active_goal_marker", 10)
        self.goal_path_pub = self.create_publisher(Path, "/social_nav_diffusion/goal_path", 10)
        self.policy_debug_pub = self.create_publisher(String, "/social_nav_diffusion/policy_debug", 10)
        self.cmd_timer = self.create_timer(self.cmd_publish_period, self.command_publish_callback)
        self.policy_debug_timer = self.create_timer(1.0, self.policy_debug_callback)
        self.diffusion_timer = None
        if self.use_diffusion_policy and self.diffusion_adapter is not None:
            self.diffusion_timer = self.create_timer(self.diffusion_inference_period, self.diffusion_inference_callback)

        mode = "diffusion-held-command" if self.use_diffusion_policy and self.diffusion_adapter is not None else "placeholder"
        self.get_logger().info(
            "policy_cmd_vel_node ready: "
            f"mode={mode}, publishing TwistStamped to /cpr_j100_0001/cmd_vel every {self.cmd_publish_period:.3f}s"
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        self.latest_odom_time = self.now_sec()

    def people_callback(self, msg: People):
        self.latest_people = list(msg.people)
        self.latest_people_time = self.now_sec()

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        self.latest_map_time = self.now_sec()

    def goal_callback(self, msg: PoseStamped):
        self.latest_goal = msg
        self.latest_goal_time = self.now_sec()
        self.publish_goal_visualization()
        self.get_logger().info(
            f"received goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}, frame={msg.header.frame_id}"
        )

    def build_robot_state(self) -> Dict[str, float]:
        pose = self.latest_odom.pose.pose
        twist = self.latest_odom.twist.twist
        yaw = quaternion_to_yaw(pose.orientation)
        linear_velocity = float(twist.linear.x)
        angular_velocity = float(twist.angular.z)
        self.last_odom_linear_velocity_used_for_sync = linear_velocity
        self.last_odom_angular_velocity_used_for_sync = angular_velocity
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": yaw,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "_sync_policy_warm_start_from_odom": bool(
                self.get_parameter("sync_policy_warm_start_from_odom").value
            ),
            "_sync_prev_action_from_odom": bool(
                self.get_parameter("sync_prev_action_from_odom").value
            ),
        }

    def build_people_states(self, robot: Dict[str, float]) -> List[Dict[str, float]]:
        states = []
        cos_yaw = math.cos(robot["yaw"])
        sin_yaw = math.sin(robot["yaw"])
        for person in self.latest_people:
            dx = float(person.position.x) - robot["x"]
            dy = float(person.position.y) - robot["y"]
            rel_x = cos_yaw * dx + sin_yaw * dy
            rel_y = -sin_yaw * dx + cos_yaw * dy
            states.append(
                {
                    "x": float(person.position.x),
                    "y": float(person.position.y),
                    "vx": float(person.velocity.x),
                    "vy": float(person.velocity.y),
                    "distance": math.hypot(dx, dy),
                    "rel_x": rel_x,
                    "rel_y": rel_y,
                }
            )
        return states

    def build_goal_state(self, robot: Dict[str, float]) -> Dict[str, float]:
        gx = float(self.latest_goal.pose.position.x)
        gy = float(self.latest_goal.pose.position.y)
        if bool(self.get_parameter("fixed_test_goal_in_robot_frame").value):
            goal_x_robot = float(self.get_parameter("fixed_test_goal_x").value)
            goal_y_robot = float(self.get_parameter("fixed_test_goal_y").value)
            cos_yaw = math.cos(robot["yaw"])
            sin_yaw = math.sin(robot["yaw"])
            gx = robot["x"] + cos_yaw * goal_x_robot - sin_yaw * goal_y_robot
            gy = robot["y"] + sin_yaw * goal_x_robot + cos_yaw * goal_y_robot
        return self.build_goal_state_from_xy(robot, gx, gy)

    def build_goal_state_from_xy(self, robot: Dict[str, float], gx: float, gy: float) -> Dict[str, float]:
        dx = float(gx) - robot["x"]
        dy = float(gy) - robot["y"]
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(target_heading - robot["yaw"])
        cos_yaw = math.cos(robot["yaw"])
        sin_yaw = math.sin(robot["yaw"])
        rel_x = cos_yaw * dx + sin_yaw * dy
        rel_y = -sin_yaw * dx + cos_yaw * dy
        return {
            "x": float(gx),
            "y": float(gy),
            "dx": dx,
            "dy": dy,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "distance": distance,
            "heading_to_goal": target_heading,
            "heading_error": heading_error,
        }

    def remember_goal_debug(self, goal: Dict[str, float]):
        self.last_policy_goal_x_map = float(goal["x"])
        self.last_policy_goal_y_map = float(goal["y"])
        self.last_goal_dx_map = float(goal["dx"])
        self.last_goal_dy_map = float(goal["dy"])
        self.last_goal_x_robot_frame = float(goal["rel_x"])
        self.last_goal_y_robot_frame = float(goal["rel_y"])
        self.last_distance_to_goal = float(goal["distance"])
        self.last_heading_to_goal = float(goal["heading_to_goal"])
        self.last_heading_error = float(goal["heading_error"])

    def build_occupancy_state(self) -> Optional[Dict]:
        if self.latest_map is None:
            return None
        info = self.latest_map.info
        return {
            "msg": self.latest_map,
            "resolution": float(info.resolution),
            "origin_x": float(info.origin.position.x),
            "origin_y": float(info.origin.position.y),
            "width": int(info.width),
            "height": int(info.height),
        }

    def data_is_stale(self, stamp: Optional[float], now: float, timeout: float) -> bool:
        return stamp is None or (now - stamp) > timeout

    def apply_goal_approach_safety(
        self,
        raw_v: float,
        raw_w: float,
        max_linear: float,
        max_angular: float,
    ) -> Tuple[float, float]:
        v = float(raw_v)
        w = float(raw_w)

        self.last_approach_scale = float("nan")
        self.last_allowed_linear_near_goal = float("nan")
        self.last_heading_gate_active = False
        self.last_heading_stop_active = False
        self.last_near_goal_angular_damping_active = False
        self.last_heading_align_override_active = False
        self.last_heading_align_desired_w = float("nan")

        if self.latest_odom is None or self.latest_goal is None:
            return v, w

        robot = self.build_robot_state()
        goal = self.build_goal_state(robot)
        distance = float(goal["distance"])
        heading_error = float(goal["heading_error"])
        abs_heading_error = abs(heading_error)

        slow_down_radius = max(float(self.get_parameter("slow_down_radius").value), 1e-6)
        min_approach_speed = abs(float(self.get_parameter("min_approach_speed").value))
        heading_gate_rad = abs(float(self.get_parameter("heading_gate_rad").value))
        heading_stop_rad = abs(float(self.get_parameter("heading_stop_rad").value))
        near_goal_angular_scale = float(self.get_parameter("near_goal_angular_scale").value)
        heading_align_kp = float(self.get_parameter("heading_align_kp").value)
        heading_align_min_abs_w = abs(float(self.get_parameter("heading_align_min_abs_w").value))
        heading_align_override_rad = abs(float(self.get_parameter("heading_align_override_rad").value))
        heading_align_override_near_goal_only = bool(
            self.get_parameter("heading_align_override_near_goal_only").value
        )

        enable_near_goal_slowdown = bool(self.get_parameter("enable_near_goal_slowdown").value)
        enable_heading_gate = bool(self.get_parameter("enable_heading_gate").value)
        enable_heading_stop = bool(self.get_parameter("enable_heading_stop").value)
        enable_heading_align_override = bool(self.get_parameter("enable_heading_align_override").value)

        near_goal_for_override = distance < slow_down_radius
        if enable_near_goal_slowdown and near_goal_for_override:
            scale = clamp(distance / slow_down_radius, 0.0, 1.0)
            allowed_linear = max(min_approach_speed, max_linear * scale)
            v = clamp(v, -allowed_linear, allowed_linear)
            self.last_approach_scale = scale
            self.last_allowed_linear_near_goal = allowed_linear

            w *= near_goal_angular_scale
            w = clamp(w, -max_angular, max_angular)
            self.last_near_goal_angular_damping_active = True
        else:
            self.last_approach_scale = 1.0
            self.last_allowed_linear_near_goal = max_linear

        if enable_heading_gate and abs_heading_error > heading_gate_rad:
            v *= 0.25
            self.last_heading_gate_active = True

        if enable_heading_stop and abs_heading_error > heading_stop_rad:
            v = 0.0
            self.last_heading_stop_active = True

        override_allowed = (not heading_align_override_near_goal_only) or near_goal_for_override
        if enable_heading_align_override and override_allowed and abs_heading_error > heading_align_override_rad:
            desired_w = heading_align_kp * heading_error
            desired_w = clamp(desired_w, -max_angular, max_angular)
            if abs(desired_w) < heading_align_min_abs_w:
                desired_w = math.copysign(heading_align_min_abs_w, heading_error)
                desired_w = clamp(desired_w, -max_angular, max_angular)
            w = desired_w
            self.last_heading_align_override_active = True
            self.last_heading_align_desired_w = desired_w

        return v, w

    def apply_sign_conflict_guard(self, v: float, w: float, raw_w: float) -> Tuple[float, float]:
        self.last_sign_conflict_guard_active = False
        self.last_sign_conflict_desired_w = float("nan")

        if not bool(self.get_parameter("enable_sign_conflict_guard").value):
            return v, w
        if self.latest_odom is None or self.latest_goal is None:
            return v, w

        robot = self.build_robot_state()
        goal = self.build_goal_state(robot)
        heading_error = float(goal["heading_error"])
        raw_w = float(raw_w)

        heading_threshold = abs(float(self.get_parameter("sign_conflict_heading_error_rad").value))
        min_raw_angular = abs(float(self.get_parameter("sign_conflict_min_raw_angular").value))
        if abs(heading_error) <= heading_threshold:
            return v, w
        if abs(raw_w) <= min_raw_angular:
            return v, w
        if raw_w * heading_error >= 0.0:
            return v, w

        align_kp = float(self.get_parameter("sign_conflict_align_kp").value)
        min_abs_w = abs(float(self.get_parameter("sign_conflict_min_abs_w").value))
        max_abs_w = abs(float(self.get_parameter("sign_conflict_max_abs_w").value))
        linear_scale = float(self.get_parameter("sign_conflict_linear_scale").value)

        desired_w = clamp(align_kp * heading_error, -max_abs_w, max_abs_w)
        if abs(desired_w) < min_abs_w:
            desired_w = math.copysign(min_abs_w, heading_error)
            desired_w = clamp(desired_w, -max_abs_w, max_abs_w)

        self.last_sign_conflict_guard_active = True
        self.last_sign_conflict_desired_w = desired_w
        return float(v) * linear_scale, desired_w

    def publish_cmd(self, v: float, w: float):
        raw_v = float(v)
        raw_w = float(w)
        self.last_converted_cmd = {"v": raw_v, "w": raw_w}
        max_linear = abs(float(self.get_parameter("max_linear_speed").value))
        max_angular = abs(float(self.get_parameter("max_angular_speed").value))
        v, w = self.apply_goal_approach_safety(raw_v, raw_w, max_linear, max_angular)
        v, w = self.apply_sign_conflict_guard(v, w, raw_w)
        if bool(self.get_parameter("enable_speed_clamp").value):
            v = clamp(v, -max_linear, max_linear)
            w = clamp(w, -max_angular, max_angular)
        if bool(self.get_parameter("disable_policy_command_publish").value):
            v = 0.0
            w = 0.0
        linear_clamped = abs(raw_v - v) > 1e-9
        angular_clamped = abs(raw_w - w) > 1e-9
        self.last_raw_cmd = {"v": raw_v, "w": raw_w}
        self.last_final_cmd = {"v": float(v), "w": float(w)}
        self.last_cmd_clamped = {"linear": linear_clamped, "angular": angular_clamped}
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cmd_frame_id
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)
        self.last_cmd = {"v": float(v), "w": float(w)}

    def publish_zero(self, reason: str):
        self.last_command_source = self.command_source_for_stop_reason(reason)
        if abs(self.last_cmd["v"]) > 1e-3 or abs(self.last_cmd["w"]) > 1e-3:
            self.get_logger().info(f"publishing zero command: {reason}")
        self.publish_cmd(0.0, 0.0)

    def command_source_for_stop_reason(self, reason: str) -> str:
        reason_lower = reason.lower()
        if "goal reached" in reason_lower:
            return "zero_goal_reached"
        if "no goal" in reason_lower:
            return "no_goal"
        if "waiting for first diffusion" in reason_lower:
            return "diffusion"
        if "placeholder" in reason_lower:
            return "placeholder"
        if (
            "timed out" in reason_lower
            or "timeout" in reason_lower
            or "stale" in reason_lower
            or "no fresh" in reason_lower
        ):
            return "zero_timeout"
        return "zero_timeout"

    def publish_goal_visualization(self):
        if self.latest_goal is None:
            return

        stamp = self.get_clock().now().to_msg()
        goal = self.latest_goal.pose.position

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.ns = "social_nav_diffusion"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal.x)
        marker.pose.position.y = float(goal.y)
        marker.pose.position.z = float(goal.z) + 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.45
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.9
        self.active_goal_marker_pub.publish(marker)

        if self.latest_odom is None:
            return

        robot = self.latest_odom.pose.pose.position
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"

        start = PoseStamped()
        start.header = path.header
        start.pose.position.x = float(robot.x)
        start.pose.position.y = float(robot.y)
        start.pose.position.z = float(robot.z)
        start.pose.orientation.w = 1.0

        end = PoseStamped()
        end.header = path.header
        end.pose.position.x = float(goal.x)
        end.pose.position.y = float(goal.y)
        end.pose.position.z = float(goal.z)
        end.pose.orientation = self.latest_goal.pose.orientation

        path.poses = [start, end]
        self.goal_path_pub.publish(path)

    def policy_debug_callback(self):
        now = self.now_sec()
        robot_x = robot_y = robot_yaw = float("nan")
        received_goal_x = received_goal_y = float("nan")
        received_goal_frame = ""
        goal_age = float("nan")

        odom_received = self.latest_odom is not None
        goal_received = self.latest_goal is not None
        map_received = self.latest_map is not None

        if odom_received:
            pose = self.latest_odom.pose.pose
            robot_x = float(pose.position.x)
            robot_y = float(pose.position.y)
            robot_yaw = quaternion_to_yaw(pose.orientation)

        if goal_received:
            received_goal_x = float(self.latest_goal.pose.position.x)
            received_goal_y = float(self.latest_goal.pose.position.y)
            received_goal_frame = self.latest_goal.header.frame_id or "map"
            if self.latest_goal_time is not None:
                goal_age = now - self.latest_goal_time

        humans_seen = len(self.latest_people)
        fields = {
            "control_mode": str(self.get_parameter("control_mode").value),
            "enable_goal_stop": bool(self.get_parameter("enable_goal_stop").value),
            "enable_speed_clamp": bool(self.get_parameter("enable_speed_clamp").value),
            "enable_near_goal_slowdown": bool(self.get_parameter("enable_near_goal_slowdown").value),
            "enable_heading_gate": bool(self.get_parameter("enable_heading_gate").value),
            "enable_heading_stop": bool(self.get_parameter("enable_heading_stop").value),
            "enable_heading_align_override": bool(self.get_parameter("enable_heading_align_override").value),
            "enable_speed_gain": bool(self.get_parameter("enable_speed_gain").value),
            "enable_sign_conflict_guard": bool(self.get_parameter("enable_sign_conflict_guard").value),
            "debug_verbose_state": bool(self.get_parameter("debug_verbose_state").value),
            "ignore_people_for_policy": bool(self.get_parameter("ignore_people_for_policy").value),
            "ignore_map_for_policy": bool(self.get_parameter("ignore_map_for_policy").value),
            "force_zero_humans": bool(self.get_parameter("force_zero_humans").value),
            "fixed_test_goal_in_robot_frame": bool(self.get_parameter("fixed_test_goal_in_robot_frame").value),
            "disable_policy_command_publish": bool(self.get_parameter("disable_policy_command_publish").value),
            "sync_policy_warm_start_from_odom": bool(self.get_parameter("sync_policy_warm_start_from_odom").value),
            "sync_prev_action_from_odom": bool(self.get_parameter("sync_prev_action_from_odom").value),
            "command_source": self.last_command_source,
            "robot_v_pref": float(self.get_parameter("robot_v_pref").value),
            "robot_radius": float(self.get_parameter("robot_radius").value),
            "human_radius": float(self.get_parameter("human_radius").value),
            "odom_linear_velocity_used_for_sync": self.last_odom_linear_velocity_used_for_sync,
            "odom_angular_velocity_used_for_sync": self.last_odom_angular_velocity_used_for_sync,
            "policy_time_step": float(getattr(self.diffusion_adapter.policy, "time_step", float("nan"))) if self.diffusion_adapter is not None else float("nan"),
            "diffusion_inference_period_sec": self.diffusion_inference_period,
            "robot_map_x": robot_x,
            "robot_map_y": robot_y,
            "robot_odom_x": robot_x,
            "robot_odom_y": robot_y,
            "robot_yaw": robot_yaw,
            "received_goal_x": received_goal_x,
            "received_goal_y": received_goal_y,
            "received_goal_frame_id": received_goal_frame,
            "goal_x": received_goal_x,
            "goal_y": received_goal_y,
            "goal_frame_id": received_goal_frame,
            "policy_goal_x_map": self.last_policy_goal_x_map,
            "policy_goal_y_map": self.last_policy_goal_y_map,
            "goal_dx_map": self.last_goal_dx_map,
            "goal_dy_map": self.last_goal_dy_map,
            "goal_x_robot_frame": self.last_goal_x_robot_frame,
            "goal_y_robot_frame": self.last_goal_y_robot_frame,
            "distance_to_goal": self.last_distance_to_goal,
            "heading_to_goal": self.last_heading_to_goal,
            "heading_error": self.last_heading_error,
            "humans_seen": humans_seen,
            "humans_used": self.last_humans_used,
            "humans": humans_seen,
            "map_received": map_received,
            "odom_received": odom_received,
            "goal_received": goal_received,
            "raw_action_type": self.last_raw_action_type,
            "raw_model_v_before_conversion": self.last_raw_model_v_before_conversion,
            "raw_model_r_or_w_before_conversion": self.last_raw_model_r_or_w_before_conversion,
            "converted_cmd_linear": self.last_converted_cmd["v"],
            "converted_cmd_angular": self.last_converted_cmd["w"],
            "raw_cmd_linear": self.last_raw_cmd["v"],
            "raw_cmd_angular": self.last_raw_cmd["w"],
            "final_cmd_linear": self.last_final_cmd["v"],
            "final_cmd_angular": self.last_final_cmd["w"],
            "linear_clamped": self.last_cmd_clamped["linear"],
            "angular_clamped": self.last_cmd_clamped["angular"],
            "goal_age_sec": goal_age,
            "stop_when_goal_reached": bool(self.get_parameter("stop_when_goal_reached").value),
            "slow_down_radius": float(self.get_parameter("slow_down_radius").value),
            "approach_scale": self.last_approach_scale,
            "allowed_linear_near_goal": self.last_allowed_linear_near_goal,
            "heading_gate_active": self.last_heading_gate_active,
            "heading_stop_active": self.last_heading_stop_active,
            "near_goal_angular_damping_active": self.last_near_goal_angular_damping_active,
            "heading_align_override_active": self.last_heading_align_override_active,
            "heading_align_desired_w": self.last_heading_align_desired_w,
            "heading_align_kp": float(self.get_parameter("heading_align_kp").value),
            "heading_align_override_rad": float(self.get_parameter("heading_align_override_rad").value),
            "sign_conflict_guard_active": self.last_sign_conflict_guard_active,
            "sign_conflict_heading_error_rad": float(self.get_parameter("sign_conflict_heading_error_rad").value),
            "sign_conflict_min_raw_angular": float(self.get_parameter("sign_conflict_min_raw_angular").value),
            "sign_conflict_desired_w": self.last_sign_conflict_desired_w,
            "sign_conflict_linear_scale": float(self.get_parameter("sign_conflict_linear_scale").value),
        }

        msg = String()
        msg.data = ", ".join(f"{key}={self.format_debug_value(value)}" for key, value in fields.items())
        self.policy_debug_pub.publish(msg)
        if fields["debug_verbose_state"]:
            self.get_logger().info(f"policy_debug: {msg.data}")
        self.append_policy_debug_csv(now, fields)

    def format_debug_value(self, value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def append_policy_debug_csv(self, now: float, fields: Dict[str, Any]):
        csv_path = str(self.get_parameter("debug_csv_path").value)
        if not csv_path:
            return
        header = [
            "sim_time_sec",
            "robot_x",
            "robot_y",
            "robot_yaw",
            "received_goal_x",
            "received_goal_y",
            "policy_goal_x_map",
            "policy_goal_y_map",
            "goal_x_robot_frame",
            "goal_y_robot_frame",
            "distance_to_goal",
            "heading_to_goal",
            "heading_error",
            "humans_seen",
            "humans_used",
            "raw_cmd_linear",
            "raw_cmd_angular",
            "final_cmd_linear",
            "final_cmd_angular",
            "command_source",
            "enable_near_goal_slowdown",
            "enable_heading_gate",
            "enable_heading_stop",
            "enable_heading_align_override",
        ]
        row = {
            "sim_time_sec": now,
            "robot_x": fields["robot_map_x"],
            "robot_y": fields["robot_map_y"],
            "robot_yaw": fields["robot_yaw"],
            "received_goal_x": fields["received_goal_x"],
            "received_goal_y": fields["received_goal_y"],
            "policy_goal_x_map": fields["policy_goal_x_map"],
            "policy_goal_y_map": fields["policy_goal_y_map"],
            "goal_x_robot_frame": fields["goal_x_robot_frame"],
            "goal_y_robot_frame": fields["goal_y_robot_frame"],
            "distance_to_goal": fields["distance_to_goal"],
            "heading_to_goal": fields["heading_to_goal"],
            "heading_error": fields["heading_error"],
            "humans_seen": fields["humans_seen"],
            "humans_used": fields["humans_used"],
            "raw_cmd_linear": fields["raw_cmd_linear"],
            "raw_cmd_angular": fields["raw_cmd_angular"],
            "final_cmd_linear": fields["final_cmd_linear"],
            "final_cmd_angular": fields["final_cmd_angular"],
            "command_source": fields["command_source"],
            "enable_near_goal_slowdown": fields["enable_near_goal_slowdown"],
            "enable_heading_gate": fields["enable_heading_gate"],
            "enable_heading_stop": fields["enable_heading_stop"],
            "enable_heading_align_override": fields["enable_heading_align_override"],
        }
        try:
            write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            self.get_logger().warn(f"failed to append policy debug CSV {csv_path}: {type(exc).__name__}: {exc}")

    def limits(self, dt: Optional[float] = None) -> Dict[str, float]:
        v_max = float(self.get_parameter("v_max").value)
        w_max = float(self.get_parameter("w_max").value)
        if bool(self.get_parameter("enable_speed_clamp").value):
            v_max = min(v_max, abs(float(self.get_parameter("max_linear_speed").value)))
            w_max = min(w_max, abs(float(self.get_parameter("max_angular_speed").value)))
        return {
            "v_min": 0.0,
            "v_max": v_max,
            "w_min": -w_max,
            "w_max": w_max,
            "slow_distance": float(self.get_parameter("slow_distance").value),
            "stop_distance": float(self.get_parameter("stop_distance").value),
            "front_width": float(self.get_parameter("front_width").value),
            "max_linear_accel": float(self.get_parameter("max_linear_accel").value),
            "robot_v_pref": float(self.get_parameter("robot_v_pref").value),
            "robot_radius": float(self.get_parameter("robot_radius").value),
            "human_radius": float(self.get_parameter("human_radius").value),
            "dt": self.cmd_publish_period if dt is None else dt,
        }

    def prepare_policy_inputs(self, now: float):
        stale_timeout = float(self.get_parameter("stale_timeout_sec").value)

        if self.latest_goal is None:
            return None, "no goal received"
        if self.data_is_stale(self.latest_odom_time, now, stale_timeout):
            return None, "no fresh /cpr_j100_0001/platform/odom/filtered"
        goal_timeout = float(self.get_parameter("goal_timeout_sec").value)
        if self.data_is_stale(self.latest_goal_time, now, goal_timeout):
            return None, f"goal timed out: older than {goal_timeout:.1f}s"
        if self.latest_people_time is not None and self.data_is_stale(self.latest_people_time, now, stale_timeout):
            if now - self._last_source_log_time > 1.0:
                self.get_logger().warn("/people is stale; continuing with empty people list")
                self._last_source_log_time = now
            self.latest_people = []

        robot = self.build_robot_state()
        people_seen = self.build_people_states(robot)
        if bool(self.get_parameter("ignore_people_for_policy").value) or bool(self.get_parameter("force_zero_humans").value):
            people = []
        else:
            people = people_seen
        self.last_humans_used = len(people)

        goal = self.build_goal_state(robot)
        self.remember_goal_debug(goal)

        if bool(self.get_parameter("ignore_map_for_policy").value):
            occupancy = None
        else:
            occupancy = self.build_occupancy_state()
        self.publish_goal_visualization()

        stop_when_goal_reached = bool(self.get_parameter("stop_when_goal_reached").value)
        enable_goal_stop = bool(self.get_parameter("enable_goal_stop").value)
        if enable_goal_stop and stop_when_goal_reached and goal["distance"] <= float(self.get_parameter("goal_tolerance").value):
            return None, "goal reached"

        return (robot, people, goal, occupancy), None

    def set_latest_policy_cmd(self, v: float, w: float, stamp: float, duration_sec: float):
        with self._inference_lock:
            self._latest_policy_cmd = {
                "v": float(v),
                "w": float(w),
                "stamp": float(stamp),
                "duration_sec": float(duration_sec),
            }

    def get_latest_policy_cmd(self) -> Optional[Dict[str, float]]:
        with self._inference_lock:
            return dict(self._latest_policy_cmd) if self._latest_policy_cmd is not None else None

    def clear_latest_policy_cmd(self):
        with self._inference_lock:
            self._latest_policy_cmd = None

    def command_publish_callback(self):
        now = self.now_sec()
        inputs, stop_reason = self.prepare_policy_inputs(now)
        if stop_reason is not None:
            self.clear_latest_policy_cmd()
            self.publish_zero(stop_reason)
            return

        robot, people, goal, occupancy = inputs
        limits = self.limits(self.cmd_publish_period)

        if not (self.use_diffusion_policy and self.diffusion_adapter is not None):
            try:
                v, w = compute_policy_action(robot, people, goal, occupancy, self.last_cmd, limits)
            except Exception as exc:
                self.get_logger().error(
                    f"placeholder command failed; publishing zero: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                self.publish_zero("placeholder command failed")
                return

            self.last_raw_action_type = "unknown"
            self.last_raw_model_v_before_conversion = float("nan")
            self.last_raw_model_r_or_w_before_conversion = float("nan")
            if now - self._last_source_log_time > 1.0:
                self.get_logger().info(f"command source=placeholder, v={v:.3f}, w={w:.3f}")
                self._last_source_log_time = now
            self.last_command_source = "placeholder"
            self.publish_cmd(v, w)
            return

        cmd = self.get_latest_policy_cmd()
        if cmd is None:
            self.publish_zero("waiting for first diffusion command")
            return
        age = now - cmd["stamp"]
        effective_timeout = max(
            self.command_hold_timeout,
            cmd["duration_sec"] + self.diffusion_inference_period + self.cmd_publish_period,
        )
        if age > effective_timeout:
            self.publish_zero(f"held diffusion command timeout: {age:.2f}s > {effective_timeout:.2f}s")
            return

        if now - self._last_hold_log_time > 1.0:
            self.get_logger().info(
                f"republishing held diffusion command: v={cmd['v']:.3f}, w={cmd['w']:.3f}, "
                f"age={age:.2f}s, inference duration={cmd['duration_sec']:.3f}s, "
                f"effective timeout={effective_timeout:.2f}s"
            )
            self._last_hold_log_time = now
        self.last_command_source = "diffusion"
        self.publish_cmd(cmd["v"], cmd["w"])

    def diffusion_inference_callback(self):
        if not (self.use_diffusion_policy and self.diffusion_adapter is not None):
            return
        with self._inference_lock:
            if self._diffusion_inference_running:
                return
            self._diffusion_inference_running = True

        now = self.now_sec()
        inputs, stop_reason = self.prepare_policy_inputs(now)
        if stop_reason is not None:
            with self._inference_lock:
                self._diffusion_inference_running = False
            return

        robot, people, goal, occupancy = inputs
        last_cmd = dict(self.last_cmd)
        limits = self.limits(self.diffusion_inference_period)

        self._diffusion_thread = threading.Thread(
            target=self.run_diffusion_inference,
            args=(robot, people, goal, occupancy, last_cmd, limits),
            daemon=True,
        )
        self._diffusion_thread.start()

    def run_diffusion_inference(
        self,
        robot: Dict[str, float],
        people: List[Dict[str, float]],
        goal: Dict[str, float],
        occupancy: Optional[Dict],
        last_cmd: Dict[str, float],
        limits: Dict[str, float],
    ):
        start_wall = time.perf_counter()
        try:
            v, w = self.diffusion_adapter.compute_action(robot, people, goal, occupancy, last_cmd, limits)
            self.last_raw_action_type = self.diffusion_adapter.last_action_type
            self.last_raw_model_v_before_conversion = self.diffusion_adapter.last_raw_model_v
            self.last_raw_model_r_or_w_before_conversion = self.diffusion_adapter.last_raw_model_r_or_w
            duration = time.perf_counter() - start_wall
            self.set_latest_policy_cmd(v, w, self.now_sec(), duration)
            now = self.now_sec()
            if now - self._last_source_log_time > 1.0:
                self.get_logger().info(
                    f"command source=diffusion, v={v:.3f}, w={w:.3f}, "
                    f"inference duration={duration:.3f}s, "
                    f"action={self.diffusion_adapter.last_action_fields}"
                )
                self._last_source_log_time = now
        except Exception as exc:
            self.get_logger().error(
                f"diffusion inference failed; holding previous command until timeout: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )
        finally:
            with self._inference_lock:
                self._diffusion_inference_running = False


def main(args=None):
    rclpy.init(args=args)
    node = PolicyCmdVelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
