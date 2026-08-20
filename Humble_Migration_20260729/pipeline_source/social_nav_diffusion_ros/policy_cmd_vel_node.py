import collections
import configparser
import hashlib
import csv
import copy
import json
import math
import os
import sys
import threading
import time
import traceback
from typing import Any, Deque, Dict, List, Optional, Tuple

INFERENCE_REPO_DEFAULT = "/workspace/SocialNavDiffusion_Inference"
VENV_PYTHON_DEFAULT = f"{INFERENCE_REPO_DEFAULT}/.venv/bin/python"


def wants_diffusion_policy(argv: List[str]) -> bool:
    if os.environ.get("SOCIAL_NAV_DIFFUSION_USE_VENV", "").lower() in ("1", "true", "yes"):
        return True
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
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from people_msgs.msg import People
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rcl_interfaces.msg import SetParametersResult
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_people_detector_status(raw_status: str) -> Tuple[bool, str]:
    try:
        fields = json.loads(raw_status)
    except (TypeError, ValueError):
        return False, "invalid status JSON"
    if not isinstance(fields, dict):
        return False, "status is not a JSON object"
    ready = fields.get("ready") is True
    reason = str(fields.get("reason", "not ready" if not ready else "ready"))
    return ready, reason


def voxelized_laser_points(
    ranges,
    angle_min: float,
    angle_increment: float,
    message_range_min: float,
    message_range_max: float,
    configured_range_min: float,
    configured_range_max: float,
    voxel_size: float,
    max_points: int,
) -> List[Tuple[float, float]]:
    """Return a bounded nearest-first scan snapshot in the scan frame."""
    lower = max(float(message_range_min), float(configured_range_min), 0.0)
    upper_candidates = [float(configured_range_max)]
    if math.isfinite(float(message_range_max)) and float(message_range_max) > 0.0:
        upper_candidates.append(float(message_range_max))
    upper = min(upper_candidates)
    if upper <= lower or max_points <= 0:
        return []

    cell_size = max(float(voxel_size), 1e-3)
    cells: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    angle = float(angle_min)
    increment = float(angle_increment)
    for raw_range in ranges:
        distance = float(raw_range)
        if math.isfinite(distance) and lower <= distance <= upper:
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            key = (int(round(x / cell_size)), int(round(y / cell_size)))
            previous = cells.get(key)
            if previous is None or distance < previous[0]:
                cells[key] = (distance, x, y)
        angle += increment

    nearest = sorted(cells.values(), key=lambda item: item[0])[:max_points]
    return [(float(x), float(y)) for _, x, y in nearest]


def combine_occupancy_points(static_points, live_points):
    """Combine the static map with one bounded LiDAR snapshot."""
    try:
        import numpy as np

        live_array = np.asarray(live_points, dtype=float).reshape((-1, 2))
        if static_points is None or len(static_points) == 0:
            return live_array
        return np.concatenate(
            (np.asarray(static_points, dtype=float).reshape((-1, 2)), live_array),
            axis=0,
        )
    except ImportError:
        static_list = list(static_points) if static_points is not None else []
        return static_list + list(live_points)


def update_obstacle_memory(
    memory,
    points,
    now_sec: float,
    ttl_sec: float,
    voxel_size: float,
    max_points: int,
    origin_xy=(0.0, 0.0),
):
    """Update a bounded, timestamped obstacle voxel cache."""
    now = float(now_sec)
    ttl = max(0.0, float(ttl_sec))
    cell_size = max(float(voxel_size), 1e-3)
    limit = max(0, int(max_points))
    retained = {
        key: value
        for key, value in memory.items()
        if now - float(value[2]) <= ttl
    }
    for x, y in points:
        px = float(x)
        py = float(y)
        key = (
            int(round(px / cell_size)),
            int(round(py / cell_size)),
        )
        retained[key] = (px, py, now)

    origin_x, origin_y = float(origin_xy[0]), float(origin_xy[1])
    ranked = sorted(
        retained.items(),
        key=lambda item: (
            math.hypot(item[1][0] - origin_x, item[1][1] - origin_y),
            -item[1][2],
        ),
    )[:limit]
    bounded = dict(ranked)
    return bounded, [(value[0], value[1]) for _, value in ranked]


def map_subscription_qos() -> QoSProfile:
    """Receive a saved map even when the policy starts after map_server."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


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

def set_quaternion_from_yaw(orientation, yaw: float):
    orientation.x = 0.0
    orientation.y = 0.0
    orientation.z = math.sin(0.5 * float(yaw))
    orientation.w = math.cos(0.5 * float(yaw))





def file_sha256(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_yaw_from_quaternion(q) -> float:
    return quaternion_to_yaw(q)


def apply_transform_xy(x: float, y: float, transform) -> Tuple[float, float]:
    yaw = transform_yaw_from_quaternion(transform.transform.rotation)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    tx = float(transform.transform.translation.x)
    ty = float(transform.transform.translation.y)
    return tx + cos_yaw * float(x) - sin_yaw * float(y), ty + sin_yaw * float(x) + cos_yaw * float(y)


def rotate_vector_xy(vx: float, vy: float, transform) -> Tuple[float, float]:
    yaw = transform_yaw_from_quaternion(transform.transform.rotation)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return cos_yaw * float(vx) - sin_yaw * float(vy), sin_yaw * float(vx) + cos_yaw * float(vy)

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
        test_mode: bool = False,
    ):
        self.inference_repo = inference_repo
        self.config_path = config_path
        self.env_config_path = env_config_path
        self.checkpoint_path = checkpoint_path
        self.norm_path = norm_path
        self.logger = logger
        self.test_mode = bool(test_mode)
        self.policy = None
        self.FullState = None
        self.JointState = None
        self.ObservableState = None
        self.last_action_fields: Any = None
        self.last_action_type = "unknown"
        self.last_raw_model_v = float("nan")
        self.last_raw_model_r_or_w = float("nan")
        self.last_actionxy_angular_dt_used = float("nan")
        self.last_projected_trajectory_available = False
        self.last_projected_x_traj = None
        self.last_projected_u_traj = None
        self.last_projected_info: Dict[str, Any] = {}
        self.last_projected_horizon_sec = float("nan")
        self.last_projected_dt = float("nan")
        self.last_projected_sample_time_sec = float("nan")
        self.policy_config_ckpt_path = ""
        self.policy_config_ckpt_resolved = ""
        self.policy_config_ckpt_exists = False
        self.checkpoint_path_requested = checkpoint_path
        self.checkpoint_path_resolved = checkpoint_path
        self.checkpoint_exists = os.path.exists(checkpoint_path)
        self.checkpoint_consistency_warning = ""
        self.checkpoint_sha256 = None
        self.policy_config_ckpt_sha256 = None
        self.policy_config_max_linear_speed = float("nan")
        self.policy_config_max_angular_speed = float("nan")
        self.policy_config_max_linear_accel = float("nan")
        self.policy_config_max_angular_accel = float("nan")
        self.policy_config_robot_v_pref = float("nan")
        self.policy_config_robot_radius = float("nan")
        self.policy_config_human_radius = float("nan")
        self.last_static_map_has_map_value = 0.0
        self.last_static_map_cells_used = 0
        self.last_policy_state_after_sync = {}
        self.load_policy()

    def resolve_model_path(self, path: str) -> str:
        if not path:
            return path
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return expanded
        candidate = os.path.join(self.inference_repo, expanded)
        if os.path.exists(candidate):
            return candidate
        config_relative = os.path.join(os.path.dirname(self.config_path), expanded)
        if os.path.exists(config_relative):
            return config_relative
        return candidate

    def config_getfloat(self, config, section: str, option: str) -> float:
        try:
            if config.has_option(section, option):
                return config.getfloat(section, option)
        except Exception:
            pass
        return float("nan")

    def reset_policy_state(self) -> Dict[str, float]:
        if self.policy is None:
            return {}
        if hasattr(self.policy, "_reset_unicycle_state"):
            try:
                self.policy._reset_unicycle_state()
            except Exception as exc:
                self.logger.warn(f"policy._reset_unicycle_state() failed: {type(exc).__name__}: {exc}")
        for attr in ("prev_v", "prev_omega", "_v0_from_last_step", "_w0_from_last_step"):
            if hasattr(self.policy, attr):
                try:
                    setattr(self.policy, attr, 0.0)
                except Exception as exc:
                    self.logger.warn(f"failed to reset policy.{attr}: {type(exc).__name__}: {exc}")
        return self.policy_state_snapshot()

    def policy_state_snapshot(self) -> Dict[str, float]:
        if self.policy is None:
            return {}
        return {
            "prev_v": float(getattr(self.policy, "prev_v", float("nan"))),
            "prev_omega": float(getattr(self.policy, "prev_omega", float("nan"))),
            "_v0_from_last_step": float(getattr(self.policy, "_v0_from_last_step", float("nan"))),
            "_w0_from_last_step": float(getattr(self.policy, "_w0_from_last_step", float("nan"))),
        }

    def update_static_map(self, occupancy: Optional[Dict], enable_static_map: bool):
        if self.policy is None:
            return
        if not enable_static_map or occupancy is None or occupancy.get("occ_world_xy") is None:
            self.policy.set_static_map(None, 0.0, self.policy.map_extent)
            self.last_static_map_has_map_value = 0.0
            self.last_static_map_cells_used = 0
            return
        occ_world_xy = occupancy["occ_world_xy"]
        self.policy.set_static_map(occ_world_xy, 1.0, self.policy.map_extent)
        self.last_static_map_has_map_value = 1.0
        try:
            self.last_static_map_cells_used = len(occ_world_xy)
        except TypeError:
            self.last_static_map_cells_used = 0

    def install_projection_cache_hook(self):
        if self.policy is None or not hasattr(self.policy, "_project_trajectory"):
            self.logger.warn("projected trajectory sampling unavailable: policy has no _project_trajectory")
            return
        original_project_trajectory = self.policy._project_trajectory

        def cached_project_trajectory(diffusion_xy_ego, state):
            result = original_project_trajectory(diffusion_xy_ego, state)
            try:
                proj_x, proj_u, info = result
                self.last_projected_x_traj = proj_x.copy() if hasattr(proj_x, "copy") else proj_x
                self.last_projected_u_traj = proj_u.copy() if hasattr(proj_u, "copy") else proj_u
                self.last_projected_info = dict(info) if isinstance(info, dict) else {}
                self.last_projected_trajectory_available = proj_x is not None
                proj_dt = float(getattr(self.policy, "proj_dt", float("nan")))
                self.last_projected_dt = proj_dt
                if proj_x is not None and hasattr(proj_x, "shape") and proj_x.shape[0] > 1 and not math.isnan(proj_dt):
                    self.last_projected_horizon_sec = float((proj_x.shape[0] - 1) * proj_dt)
                else:
                    self.last_projected_horizon_sec = float("nan")
            except Exception as exc:
                self.last_projected_trajectory_available = False
                self.last_projected_x_traj = None
                self.last_projected_u_traj = None
                self.last_projected_info = {"cache_error": f"{type(exc).__name__}: {exc}"}
                self.last_projected_horizon_sec = float("nan")
                self.last_projected_dt = float("nan")
            return result

        self.policy._project_trajectory = cached_project_trajectory
        self.logger.info("projected trajectory cache hook installed on policy._project_trajectory")

    def clear_projected_trajectory_cache(self):
        self.last_projected_trajectory_available = False
        self.last_projected_x_traj = None
        self.last_projected_u_traj = None
        self.last_projected_info = {}
        self.last_projected_horizon_sec = float("nan")
        self.last_projected_dt = float("nan")
        self.last_projected_sample_time_sec = float("nan")

    def projected_trajectory_for_command(self) -> Optional[Dict[str, Any]]:
        if not self.last_projected_trajectory_available or self.last_projected_x_traj is None:
            return None
        return {
            "x_traj": self.last_projected_x_traj.copy() if hasattr(self.last_projected_x_traj, "copy") else self.last_projected_x_traj,
            "u_traj": self.last_projected_u_traj.copy() if hasattr(self.last_projected_u_traj, "copy") else self.last_projected_u_traj,
            "info": dict(self.last_projected_info),
            "dt": self.last_projected_dt,
            "horizon_sec": self.last_projected_horizon_sec,
        }

    def predicted_trajectory_for_command(self) -> Optional[Dict[str, Any]]:
        predicted = getattr(self.policy, "predicted_traj", None)
        if not isinstance(predicted, dict):
            return None
        selected = predicted.get("selected_sample")
        if selected is None:
            return None
        return {
            "xy_world": selected.copy() if hasattr(selected, "copy") else selected,
            "used_projection": bool(predicted.get("used_projection", False)),
        }

    def all_candidate_trajectories_for_command(self) -> Optional[List[Any]]:
        """All K diffusion candidate samples (world-frame), index 0 == selected."""
        predicted = getattr(self.policy, "predicted_traj", None)
        if not isinstance(predicted, dict):
            return None
        samples = predicted.get("all_samples")
        if not samples:
            return None
        return [sample.copy() if hasattr(sample, "copy") else sample for sample in samples]

    def last_planning_timing(self) -> Optional[Dict[str, float]]:
        timing = getattr(self.policy, "last_predict_timing", None)
        if not isinstance(timing, dict):
            return None
        return dict(timing)

    def sample_projected_trajectory(self, trajectory: Dict[str, Any], elapsed_sec: float) -> Optional[Dict[str, float]]:
        if self.policy is None or not hasattr(self.policy, "_interp_projected_state"):
            return None
        x_traj = trajectory.get("x_traj")
        if x_traj is None:
            return None
        try:
            _, _, _, v, w = self.policy._interp_projected_state(x_traj, float(elapsed_sec))
        except Exception as exc:
            self.logger.warn(f"projected trajectory sampling failed: {type(exc).__name__}: {exc}")
            return None
        self.last_projected_sample_time_sec = float(elapsed_sec)
        return {"v": float(v), "w": float(w), "sample_time_sec": float(elapsed_sec)}

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

        if self.test_mode:
            from crowd_nav.policy.diffusion_CondUNetCFG_test_mode import (
                DiffusionConditionalUNet1DCFG,
            )
        else:
            from crowd_nav.policy.diffusion_CondUNetCFG import (
                DiffusionConditionalUNet1DCFG,
            )
        from crowd_sim.envs.utils.state import FullState, JointState, ObservableState

        config = configparser.RawConfigParser()
        read_files = config.read(self.config_path)
        if not read_files:
            raise FileNotFoundError(f"policy config not readable: {self.config_path}")

        section = "diffusion_conditional_unet1dcfg"
        if config.has_option(section, "ckpt_path"):
            self.policy_config_ckpt_path = config.get(section, "ckpt_path")
            self.policy_config_ckpt_resolved = self.resolve_model_path(self.policy_config_ckpt_path)
            self.policy_config_ckpt_exists = os.path.exists(self.policy_config_ckpt_resolved)
        if config.has_section("action_space"):
            self.policy_config_max_linear_speed = self.config_getfloat(config, "action_space", "max_vel")
            self.policy_config_max_angular_speed = self.config_getfloat(config, "action_space", "max_wrot")
            self.policy_config_max_linear_accel = self.config_getfloat(config, "action_space", "max_accel")
            self.policy_config_max_angular_accel = self.config_getfloat(config, "action_space", "max_w_accel")

        self.checkpoint_path_resolved = self.resolve_model_path(self.checkpoint_path)
        self.checkpoint_exists = os.path.exists(self.checkpoint_path_resolved)
        try:
            self.checkpoint_sha256 = file_sha256(self.checkpoint_path_resolved)
            self.policy_config_ckpt_sha256 = file_sha256(self.policy_config_ckpt_resolved)
        except Exception as exc:
            self.logger.warn(f"checkpoint hash check failed: {type(exc).__name__}: {exc}")
        if self.policy_config_ckpt_path and self.policy_config_ckpt_resolved != self.checkpoint_path_resolved:
            if self.policy_config_ckpt_exists and self.checkpoint_exists:
                if self.policy_config_ckpt_sha256 == self.checkpoint_sha256:
                    self.checkpoint_consistency_warning = "wrapper checkpoint and policy.config checkpoint differ by path but hashes match"
                else:
                    self.checkpoint_consistency_warning = (
                        "wrapper checkpoint differs from policy.config ckpt_path; collaborator should confirm "
                        f"whether the configured checkpoint file matches {self.policy_config_ckpt_path}"
                    )
            else:
                self.checkpoint_consistency_warning = "wrapper checkpoint path and policy.config ckpt_path differ; one or both files are missing"

        config = copy.deepcopy(config)
        config.set(section, "ckpt_path", self.checkpoint_path_resolved)
        config.set(section, "norm_file", self.resolve_model_path(self.norm_path))

        policy = DiffusionConditionalUNet1DCFG()
        policy.configure(config)

        env_config = configparser.RawConfigParser()
        env_config.read(self.env_config_path)
        policy.time_step = (
            env_config.getfloat("env", "time_step")
            if env_config.has_option("env", "time_step")
            else 0.25
        )
        self.policy_config_robot_v_pref = self.config_getfloat(env_config, "robot", "v_pref")
        self.policy_config_robot_radius = self.config_getfloat(env_config, "robot", "radius")
        self.policy_config_human_radius = self.config_getfloat(env_config, "humans", "radius")
        policy.set_static_map(None, 0.0, policy.map_extent)

        self.policy = policy
        self.install_projection_cache_hook()
        self.FullState = FullState
        self.JointState = JointState
        self.ObservableState = ObservableState
        self.logger.info(
            "diffusion model loaded: "
            f"config={self.config_path}, env_config={self.env_config_path}, "
            f"checkpoint_requested={self.checkpoint_path_requested}, "
            f"checkpoint_resolved={self.checkpoint_path_resolved}, checkpoint_exists={self.checkpoint_exists}, "
            f"policy_config_ckpt_path={self.policy_config_ckpt_path}, "
            f"policy_config_ckpt_resolved={self.policy_config_ckpt_resolved}, "
            f"policy_config_ckpt_exists={self.policy_config_ckpt_exists}, norm={self.norm_path}, "
            f"python={sys.executable}"
        )
        if self.checkpoint_consistency_warning:
            self.logger.warn(self.checkpoint_consistency_warning)

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
        self.last_policy_state_after_sync = self.policy_state_snapshot()

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
                # ActionXY is rare for this unicycle-trained wrapper, but if it appears,
                # convert the desired heading delta using the model/policy dt, matching ActionRot.
                model_dt = float(getattr(self.policy, "time_step", limits["dt"])) or limits["dt"]
                self.last_actionxy_angular_dt_used = model_dt
                w = wrap_angle(desired_yaw - yaw) / max(model_dt, 1e-3)
            else:
                self.last_actionxy_angular_dt_used = float(getattr(self.policy, "time_step", limits["dt"])) or limits["dt"]
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
        del last_cmd
        self.clear_projected_trajectory_cache()
        self.update_static_map(occupancy, bool(limits.get("enable_static_map_to_policy", True)))
        self.sync_policy_state_from_odom(robot_state)
        state = self.build_state(robot_state, people_states, goal_state, limits)
        action = self.policy.predict(state)
        return self.action_to_cmd(action, robot_state, limits)


class PolicyCmdVelNode(Node):
    def __init__(self):
        super().__init__("policy_cmd_vel_node")

        self.declare_parameter("test_mode", False)
        self.test_mode = bool(self.get_parameter("test_mode").value)
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
        self.declare_parameter("require_people_stream", False)
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
        self.declare_parameter("projected_alignment_max_position_error", 0.45)
        self.declare_parameter("sync_policy_warm_start_from_odom", True)
        self.declare_parameter("sync_prev_action_from_odom", True)
        self.declare_parameter("control_period_sec", 0.1)
        self.declare_parameter("cmd_publish_period_sec", 0.1)
        self.declare_parameter("diffusion_inference_period_sec", 0.1)
        self.declare_parameter("command_hold_timeout_sec", 2.0)
        self.declare_parameter("v_max", 1.0)
        self.declare_parameter("w_max", 3.141592653589793)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_angular_speed", 3.14)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("bridge_goal_tolerance", 0.25)
        self.declare_parameter("goal_timeout_sec", 150.0)
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
        self.declare_parameter("max_linear_accel", 1.5)
        self.declare_parameter("max_angular_accel", 3.14)
        self.declare_parameter("cmd_frame_id", "base_link")
        self.declare_parameter("people_topic", "/people")
        self.declare_parameter(
            "people_detector_status_topic",
            "/people_detector/status",
        )
        self.declare_parameter("odom_topic", "/cpr_j100_0001/platform/odom/filtered")
        self.declare_parameter("map_topic", "/cpr_j100_0001/map")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("robot_goal_topic", "/cpr_j100_0001/goal_pose")
        self.declare_parameter("cmd_vel_topic", "/cpr_j100_0001/cmd_vel")
        self.declare_parameter("active_goal_marker_topic", "/social_nav_diffusion/active_goal_marker")
        self.declare_parameter("projected_trajectory_topic", "/social_nav_diffusion/projected_trajectory")
        self.declare_parameter("predicted_trajectory_topic", "/social_nav_diffusion/predicted_trajectory")
        self.declare_parameter("candidate_trajectories_topic", "/social_nav_diffusion/candidate_trajectories")
        self.declare_parameter("policy_debug_topic", "/social_nav_diffusion/policy_debug")
        self.declare_parameter("style_vector_topic", "/social_nav_diffusion/style_vector")
        # Social style vector, order matches policy.config: prox, pass, yield, group.
        # Live-tunable via `ros2 param set <node> style_vector [...]` without
        # restarting the node or reloading the model (see apply_style_vector()).
        self.declare_parameter("style_vector", [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("use_diffusion_policy", False)
        self.declare_parameter("diffusion_inference_repo", INFERENCE_REPO_DEFAULT)
        default_policy_config = (
            f"{INFERENCE_REPO_DEFAULT}/crowd_nav/configs/policy_test_mode.config"
            if self.test_mode
            else f"{INFERENCE_REPO_DEFAULT}/crowd_nav/configs/policy.config"
        )
        default_checkpoint = (
            f"{INFERENCE_REPO_DEFAULT}/ckpt_step990000_sogudiff_singleaxis_1p5M.pt"
            if self.test_mode
            else f"{INFERENCE_REPO_DEFAULT}/ckpt_step478000_SOCIAL_NORMS8.pt"
        )
        default_norm = (
            f"{INFERENCE_REPO_DEFAULT}/norm_stats_sogudiff_allarms_1p5M.npy"
            if self.test_mode
            else f"{INFERENCE_REPO_DEFAULT}/norm_stats_SOCIAL_NORMS8.npy"
        )
        self.declare_parameter("diffusion_config_path", default_policy_config)
        self.declare_parameter("diffusion_env_config_path", f"{INFERENCE_REPO_DEFAULT}/crowd_nav/configs/env.config")
        self.declare_parameter("diffusion_checkpoint_path", default_checkpoint)
        self.declare_parameter("diffusion_norm_path", default_norm)
        self.declare_parameter("target_policy_frame", "map")
        self.declare_parameter("enable_tf_transforms", True)
        self.declare_parameter("tf_lookup_timeout_sec", 0.1)
        self.declare_parameter("allow_tf_fallback_raw", False)
        self.declare_parameter("people_frame", "map")
        self.declare_parameter("odom_pose_source_frame_override", "")
        self.declare_parameter("goal_frame_override", "")
        self.declare_parameter("map_frame_override", "")
        self.declare_parameter("log_frame_ids", True)
        self.declare_parameter("enable_static_map_to_policy", True)
        self.declare_parameter("static_map_update_on_change_only", True)
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("max_static_map_cells", 20000)
        self.declare_parameter("treat_unknown_as_occupied", False)
        self.declare_parameter("enable_live_lidar_to_policy", False)
        self.declare_parameter("require_live_lidar_for_policy", False)
        self.declare_parameter(
            "lidar_topic",
            "/jackal1/sensors/lidar3d_0/scan",
        )
        self.declare_parameter("lidar_timeout_sec", 0.5)
        self.declare_parameter("lidar_min_range_m", 0.15)
        self.declare_parameter("lidar_max_range_m", 6.0)
        self.declare_parameter("lidar_voxel_size_m", 0.25)
        self.declare_parameter("lidar_max_points", 64)
        self.declare_parameter("lidar_obstacle_memory_sec", 0.8)
        self.declare_parameter("reset_policy_state_on_new_goal", True)
        self.declare_parameter("reset_policy_state_on_goal_reached", True)
        self.declare_parameter("goal_duplicate_position_epsilon", 0.03)
        self.declare_parameter("goal_duplicate_orientation_epsilon", 0.05)
        self.declare_parameter("enable_projected_trajectory_sampling", True)
        self.declare_parameter("enable_projected_trajectory_latency_compensation", False)
        self.declare_parameter("projected_trajectory_fallback_to_hold", True)
        self.declare_parameter("enable_policy_warmup", True)
        self.declare_parameter("warmup_on_startup", True)
        self.declare_parameter("warmup_goal_x_robot_frame", 2.0)
        self.declare_parameter("warmup_goal_y_robot_frame", 0.0)
        self.declare_parameter("warmup_timeout_sec", 240.0)
        self.declare_parameter("publish_cmd_during_warmup", False)

        self.control_period = float(self.get_parameter("control_period_sec").value)
        self.cmd_publish_period = float(self.get_parameter("cmd_publish_period_sec").value)
        self.diffusion_inference_period = float(self.get_parameter("diffusion_inference_period_sec").value)
        self.command_hold_timeout = float(self.get_parameter("command_hold_timeout_sec").value)
        self.cmd_frame_id = str(self.get_parameter("cmd_frame_id").value)
        self.people_topic = str(self.get_parameter("people_topic").value)
        self.people_detector_status_topic = str(
            self.get_parameter("people_detector_status_topic").value
        )
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.robot_goal_topic = str(self.get_parameter("robot_goal_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.active_goal_marker_topic = str(self.get_parameter("active_goal_marker_topic").value)
        self.projected_trajectory_topic = str(self.get_parameter("projected_trajectory_topic").value)
        self.predicted_trajectory_topic = str(self.get_parameter("predicted_trajectory_topic").value)
        self.candidate_trajectories_topic = str(self.get_parameter("candidate_trajectories_topic").value)
        self.policy_debug_topic = str(self.get_parameter("policy_debug_topic").value)
        self.use_diffusion_policy = bool(self.get_parameter("use_diffusion_policy").value)
        self.planning_timing_window: Deque[Dict[str, float]] = collections.deque(maxlen=50)
        self.current_style_vector: List[float] = [0.0, 0.0, 0.0, 0.0]
        if self.test_mode:
            self.add_on_set_parameters_callback(self._on_set_parameters)
        self.diffusion_adapter: Optional[SocialNavDiffusionPolicyAdapter] = None
        self._last_source_log_time = 0.0
        self._last_hold_log_time = 0.0
        self._inference_lock = threading.Lock()
        self._latest_policy_cmd: Optional[Dict[str, Any]] = None
        self._diffusion_thread: Optional[threading.Thread] = None
        self._diffusion_inference_running = False
        self._goal_lock = threading.RLock()
        self._goal_generation = 0
        self._pending_goal_during_warmup: Optional[PoseStamped] = None
        self._pending_goal_received_time: Optional[float] = None
        self.policy_warmup_enabled = bool(self.get_parameter("enable_policy_warmup").value)
        self.policy_warmup_started = False
        self.policy_warmup_complete = False
        self.policy_warmup_success = False
        self.policy_warmup_duration_sec = float("nan")
        self.policy_warmup_error = ""
        self.policy_ready_for_goal = not (
            self.use_diffusion_policy
            and self.policy_warmup_enabled
            and bool(self.get_parameter("warmup_on_startup").value)
        )
        self._policy_warmup_wall_start: Optional[float] = None
        self._policy_warmup_timeout_reported = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_policy_frame = str(self.get_parameter("target_policy_frame").value)
        self.live_lidar_enabled = bool(
            self.get_parameter("enable_live_lidar_to_policy").value
        )
        self.require_live_lidar = bool(
            self.get_parameter("require_live_lidar_for_policy").value
        )
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.last_frame_debug = {
            "latest_odom_header_frame_id": "",
            "latest_odom_child_frame_id": "",
            "latest_goal_header_frame_id": "",
            "latest_map_header_frame_id": "",
            "latest_people_header_frame_id": "",
            "latest_lidar_header_frame_id": "",
            "people_frame_used": "",
            "target_policy_frame": self.target_policy_frame,
            "tf_robot_to_target_success": False,
            "tf_goal_to_target_success": False,
            "tf_people_to_target_success": False,
            "tf_map_to_target_success": False,
            "tf_lidar_to_target_success": False,
            "tf_failure_reason": "",
        }
        self._logged_frame_ids = False
        self._static_map_cache = None
        self._static_map_cache_stamp = None
        self._static_map_last_update_time = float("nan")
        self._static_map_cells_total = 0
        self._static_map_cells_used = 0
        self._static_map_downsampled = False
        self._static_map_frame_id = ""
        self.last_static_map_has_map_value = 0.0
        self._latest_lidar_points_local: List[Tuple[float, float]] = []
        self._latest_lidar_frame_id = ""
        self.latest_lidar_time: Optional[float] = None
        self._latest_lidar_sequence = 0
        self._live_lidar_cache_sequence = -1
        self._live_lidar_points_cache: List[Tuple[float, float]] = []
        self._live_lidar_obstacle_memory = {}
        self._fused_occupancy_cache_key = None
        self._fused_occupancy_cache = None
        self.last_live_lidar_points_used = 0
        self.last_policy_state_reset_on_new_goal = False
        self.last_policy_state_reset_on_goal_reached = False
        self.last_policy_state_after_reset = {}
        self.last_policy_state_after_sync = {}
        self.last_projected_trajectory_available = False
        self.last_projected_trajectory_active = False
        self.last_projected_trajectory_sample_time_sec = float("nan")
        self.last_projected_trajectory_horizon_sec = float("nan")
        self.last_projected_trajectory_dt = float("nan")
        self.last_projected_trajectory_fallback_reason = ""
        self.last_projected_trajectory_point_count = 0
        self.last_projected_trajectory_frame_id = self.target_policy_frame
        self.last_projected_trajectory_publish_time = float("nan")
        self.last_projected_trajectory_input_stamp_sec = float("nan")
        self.last_projected_trajectory_result_stamp_sec = float("nan")
        self.last_projected_trajectory_latency_compensation_sec = float("nan")
        self.last_projected_trajectory_execution_age_sec = float("nan")
        self.last_predicted_trajectory_available = False
        self.last_predicted_trajectory_point_count = 0
        self.last_goal_within_tolerance = False
        self.last_incoming_goal_is_duplicate = False
        self.last_goal_position_delta = float("nan")
        self.last_goal_orientation_delta = float("nan")
        self.last_goal_reset_triggered = False
        self.bridge_goal_republish_count = 0
        self.last_goal_distance_frame = self.target_policy_frame
        self.last_goal_distance_tf_success = False
        self.last_projected_alignment_success = False
        self.last_projected_alignment_offset_sec = 0.0
        self.last_projected_alignment_position_error = float("nan")
        self.last_projected_alignment_heading_error = float("nan")
        self._goal_reached_handled = False

        if self.use_diffusion_policy:
            try:
                self.diffusion_adapter = SocialNavDiffusionPolicyAdapter(
                    inference_repo=str(self.get_parameter("diffusion_inference_repo").value),
                    config_path=str(self.get_parameter("diffusion_config_path").value),
                    env_config_path=str(self.get_parameter("diffusion_env_config_path").value),
                    checkpoint_path=str(self.get_parameter("diffusion_checkpoint_path").value),
                    norm_path=str(self.get_parameter("diffusion_norm_path").value),
                    logger=self.get_logger(),
                    test_mode=self.test_mode,
                )
                if self.test_mode:
                    self.apply_style_vector(list(self.get_parameter("style_vector").value))
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
        self.latest_people_frame_id = ""
        self.latest_people_detector_status_time: Optional[float] = None
        self.latest_people_detector_ready = False
        self.latest_people_detector_reason = "no status received"
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_map_time: Optional[float] = None
        self.latest_goal: Optional[PoseStamped] = None
        self.latest_goal_time: Optional[float] = None
        self.last_cmd = {"v": 0.0, "w": 0.0}
        self.last_raw_cmd = {"v": 0.0, "w": 0.0}
        self.last_final_cmd = {"v": 0.0, "w": 0.0}
        self.last_cmd_clamped = {"linear": False, "angular": False}
        self.last_cmd_publish_time: Optional[float] = None
        self.last_linear_slew_limited = False
        self.last_linear_slew_dt_sec = float("nan")
        self.last_linear_slew_max_delta = float("nan")
        self.last_angular_slew_limited = False
        self.last_angular_slew_dt_sec = float("nan")
        self.last_angular_slew_max_delta = float("nan")
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

        self.create_subscription(People, self.people_topic, self.people_callback, 10)
        self.create_subscription(
            String,
            self.people_detector_status_topic,
            self.people_detector_status_callback,
            10,
        )
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 20)
        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_subscription_qos(),
        )
        if self.live_lidar_enabled:
            self.create_subscription(
                LaserScan,
                self.lidar_topic,
                self.lidar_callback,
                qos_profile_sensor_data,
            )
        self.create_subscription(PoseStamped, self.goal_topic, self.goal_callback, 10)
        self.create_subscription(PoseStamped, self.robot_goal_topic, self.goal_callback, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.active_goal_marker_pub = self.create_publisher(Marker, self.active_goal_marker_topic, 10)
        self.projected_trajectory_pub = self.create_publisher(Path, self.projected_trajectory_topic, 10)
        self.predicted_trajectory_pub = self.create_publisher(Path, self.predicted_trajectory_topic, 10)
        self.candidate_trajectories_pub = None
        self.policy_debug_pub = self.create_publisher(String, self.policy_debug_topic, 10)
        self.style_vector_pub = None
        if self.test_mode:
            self.candidate_trajectories_pub = self.create_publisher(
                MarkerArray,
                self.candidate_trajectories_topic,
                10,
            )
            self.style_vector_pub = self.create_publisher(
                Float32MultiArray,
                str(self.get_parameter("style_vector_topic").value),
                QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
            # The startup apply happens before publisher setup; publish once
            # here so late-starting test tools receive the active vector.
            self.publish_style_vector_topic(self.current_style_vector)
        self.cmd_timer = self.create_timer(self.cmd_publish_period, self.command_publish_callback)
        self.policy_debug_timer = self.create_timer(1.0, self.policy_debug_callback)
        self.diffusion_timer = None
        if self.use_diffusion_policy and self.diffusion_adapter is not None:
            self.diffusion_timer = self.create_timer(self.diffusion_inference_period, self.diffusion_inference_callback)
        self.policy_warmup_watchdog_timer = self.create_timer(1.0, self.policy_warmup_watchdog_callback)

        self.log_startup_audit_state()

        if self.use_diffusion_policy and self.diffusion_adapter is not None:
            mode = (
                "diffusion-projected-trajectory"
                if bool(self.get_parameter("enable_projected_trajectory_sampling").value)
                else "diffusion-held-command"
            )
        else:
            mode = "placeholder"
        self.get_logger().info(
            "policy_cmd_vel_node ready: "
            f"mode={mode}, publishing TwistStamped to {self.cmd_vel_topic} every {self.cmd_publish_period:.3f}s"
        )
        if (
            self.use_diffusion_policy
            and self.diffusion_adapter is not None
            and self.policy_warmup_enabled
            and bool(self.get_parameter("warmup_on_startup").value)
        ):
            self.start_policy_warmup()
        else:
            self.policy_ready_for_goal = True
            self.policy_warmup_complete = not self.policy_warmup_enabled
            self.get_logger().info("Policy ready for real goals (startup warm-up disabled or diffusion inactive)")

    def log_startup_audit_state(self):
        flags = {
            "enable_near_goal_slowdown": bool(self.get_parameter("enable_near_goal_slowdown").value),
            "enable_heading_gate": bool(self.get_parameter("enable_heading_gate").value),
            "enable_heading_stop": bool(self.get_parameter("enable_heading_stop").value),
            "enable_heading_align_override": bool(self.get_parameter("enable_heading_align_override").value),
            "enable_speed_gain": bool(self.get_parameter("enable_speed_gain").value),
            "enable_sign_conflict_guard": bool(self.get_parameter("enable_sign_conflict_guard").value),
        }
        self.get_logger().info(f"heuristic safety flags: {flags}")
        policy_tolerance = float(self.get_parameter("goal_tolerance").value)
        bridge_tolerance = float(self.get_parameter("bridge_goal_tolerance").value)
        self.get_logger().info(
            f"goal tolerance audit: policy_goal_tolerance={policy_tolerance:.3f}, "
            f"bridge_goal_tolerance={bridge_tolerance:.3f}"
        )
        if abs(policy_tolerance - bridge_tolerance) > 1e-9:
            self.get_logger().warn(
                "policy and bridge goal tolerances differ; load the same params file into both nodes"
            )
        policy_dt = self.get_policy_time_step()
        if not math.isnan(policy_dt) and abs(policy_dt - self.diffusion_inference_period) > 1e-6:
            self.get_logger().warn(
                "timing mismatch: "
                f"policy.time_step={policy_dt:.6f}, "
                f"diffusion_inference_period_sec={self.diffusion_inference_period:.6f}. "
                "Projected commands are sampled at cmd_publish_period_sec when available; "
                "the immediate command is held only as an explicit fallback."
            )
        if self.diffusion_adapter is not None:
            self.get_logger().info(
                "policy config limits: "
                f"max_vel={self.diffusion_adapter.policy_config_max_linear_speed}, "
                f"max_wrot={self.diffusion_adapter.policy_config_max_angular_speed}, "
                f"max_accel={self.diffusion_adapter.policy_config_max_linear_accel}, "
                f"max_w_accel={self.diffusion_adapter.policy_config_max_angular_accel}, "
                f"robot_v_pref={self.diffusion_adapter.policy_config_robot_v_pref}, "
                f"robot_radius={self.diffusion_adapter.policy_config_robot_radius}, "
                f"human_radius={self.diffusion_adapter.policy_config_human_radius}"
            )

    def get_policy_time_step(self) -> float:
        if self.diffusion_adapter is None or self.diffusion_adapter.policy is None:
            return float("nan")
        return float(getattr(self.diffusion_adapter.policy, "time_step", float("nan")))

    def start_policy_warmup(self):
        with self._inference_lock:
            if self._diffusion_inference_running or self.policy_warmup_started:
                return
            self._diffusion_inference_running = True
            self.policy_warmup_started = True
            self.policy_warmup_complete = False
            self.policy_warmup_success = False
            self.policy_ready_for_goal = False
            self.policy_warmup_error = ""
            self._policy_warmup_wall_start = time.perf_counter()

        self.get_logger().info("Starting policy warm-up")
        self._diffusion_thread = threading.Thread(
            target=self.run_policy_warmup,
            daemon=True,
            name="social_nav_policy_warmup",
        )
        self._diffusion_thread.start()

    def policy_warmup_watchdog_callback(self):
        if not self.policy_warmup_started or self.policy_warmup_complete:
            return
        if self._policy_warmup_wall_start is None:
            return
        elapsed = time.perf_counter() - self._policy_warmup_wall_start
        self.policy_warmup_duration_sec = elapsed
        timeout = max(0.0, float(self.get_parameter("warmup_timeout_sec").value))
        if timeout > 0.0 and elapsed > timeout and not self._policy_warmup_timeout_reported:
            self._policy_warmup_timeout_reported = True
            self.policy_warmup_error = (
                f"warm-up exceeded timeout {timeout:.1f}s; waiting for the in-process inference to finish safely"
            )
            self.get_logger().error(self.policy_warmup_error)

    def run_policy_warmup(self):
        start_wall = self._policy_warmup_wall_start or time.perf_counter()
        success = False
        error = ""
        try:
            limits = self.limits(self.diffusion_inference_period)
            robot = {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "frame_id": self.target_policy_frame,
                "_sync_policy_warm_start_from_odom": False,
                "_sync_prev_action_from_odom": False,
            }
            goal = self.build_goal_state_from_xy(
                robot,
                float(self.get_parameter("warmup_goal_x_robot_frame").value),
                float(self.get_parameter("warmup_goal_y_robot_frame").value),
            )
            v, w = self.diffusion_adapter.compute_action(
                robot,
                [],
                goal,
                None,
                {"v": 0.0, "w": 0.0},
                limits,
            )
            projected = self.diffusion_adapter.projected_trajectory_for_command()
            if not math.isfinite(v) or not math.isfinite(w):
                raise RuntimeError(f"warm-up returned non-finite action: v={v}, w={w}")
            if bool(getattr(self.diffusion_adapter.policy, "use_projection", False)) and projected is None:
                raise RuntimeError("warm-up projection did not produce a valid projected trajectory")
            success = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(
                f"Policy warm-up failed: {error}\n{traceback.format_exc()}"
            )
        finally:
            duration = time.perf_counter() - start_wall
            self.policy_warmup_duration_sec = duration
            try:
                if self.diffusion_adapter is not None:
                    self.diffusion_adapter.reset_policy_state()
                    self.diffusion_adapter.clear_projected_trajectory_cache()
            except Exception as exc:
                success = False
                cleanup_error = f"warm-up state reset failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {cleanup_error}" if error else cleanup_error
                self.get_logger().error(cleanup_error)

            self.clear_latest_policy_cmd()
            self.clear_trajectory_paths()
            self.last_cmd = {"v": 0.0, "w": 0.0}
            self.last_raw_cmd = {"v": 0.0, "w": 0.0}
            self.last_final_cmd = {"v": 0.0, "w": 0.0}
            self.last_converted_cmd = {"v": 0.0, "w": 0.0}
            self.last_raw_action_type = "unknown"
            self.last_raw_model_v_before_conversion = float("nan")
            self.last_raw_model_r_or_w_before_conversion = float("nan")
            self.last_command_source = "no_goal"
            self.publish_zero_cmd_preserve_model_debug()

            self.policy_warmup_complete = True
            self.policy_warmup_success = success
            self.policy_warmup_error = error
            with self._inference_lock:
                self._diffusion_inference_running = False

            if success:
                self.get_logger().info(f"Policy warm-up completed in {duration:.3f} s")
                pending_goal = None
                with self._goal_lock:
                    self.policy_ready_for_goal = True
                    pending_goal = self._pending_goal_during_warmup
                    self._pending_goal_during_warmup = None
                    self._pending_goal_received_time = None
                    if pending_goal is not None:
                        self._accept_new_goal_locked(pending_goal)
                self.get_logger().info("Policy ready for real goals")
            else:
                self.policy_ready_for_goal = False

    def tf_enabled(self) -> bool:
        return bool(self.get_parameter("enable_tf_transforms").value)

    def allow_tf_fallback_raw(self) -> bool:
        return bool(self.get_parameter("allow_tf_fallback_raw").value)

    def note_tf_failure(self, key: str, source_frame: str, reason: str):
        self.last_frame_debug[key] = False
        message = f"{key}: {source_frame} -> {self.target_policy_frame}: {reason}"
        self.last_frame_debug["tf_failure_reason"] = message
        if self.allow_tf_fallback_raw():
            self.get_logger().warn(f"TF failed; using raw coordinates because allow_tf_fallback_raw=true: {message}")
        else:
            self.get_logger().warn(f"TF failed; refusing inconsistent-frame input: {message}")

    def lookup_transform_to_target(self, source_frame: str, key: str):
        source = source_frame or self.target_policy_frame
        if source == self.target_policy_frame or not self.tf_enabled():
            self.last_frame_debug[key] = True
            return None
        try:
            timeout = Duration(seconds=float(self.get_parameter("tf_lookup_timeout_sec").value))
            transform = self.tf_buffer.lookup_transform(self.target_policy_frame, source, Time(), timeout)
            self.last_frame_debug[key] = True
            return transform
        except TransformException as exc:
            self.note_tf_failure(key, source, str(exc))
            if self.allow_tf_fallback_raw():
                return None
            raise

    def transform_xy_to_target(self, x: float, y: float, source_frame: str, key: str) -> Tuple[float, float]:
        transform = self.lookup_transform_to_target(source_frame, key)
        if transform is None:
            return float(x), float(y)
        return apply_transform_xy(float(x), float(y), transform)

    def rotate_velocity_to_target(self, vx: float, vy: float, source_frame: str, key: str) -> Tuple[float, float]:
        transform = self.lookup_transform_to_target(source_frame, key)
        if transform is None:
            return float(vx), float(vy)
        return rotate_vector_xy(float(vx), float(vy), transform)

    def yaw_to_target(self, yaw: float, source_frame: str, key: str) -> float:
        transform = self.lookup_transform_to_target(source_frame, key)
        if transform is None:
            return float(yaw)
        return wrap_angle(float(yaw) + transform_yaw_from_quaternion(transform.transform.rotation))

    def invalidate_policy_command_state(self):
        self.clear_latest_policy_cmd()
        self.clear_trajectory_paths()
        self.reset_projected_alignment_debug()
        self.last_projected_trajectory_available = False
        self.last_projected_trajectory_active = False
        self.last_projected_trajectory_sample_time_sec = float("nan")
        self.last_projected_trajectory_horizon_sec = float("nan")
        self.last_projected_trajectory_dt = float("nan")
        self.last_projected_trajectory_fallback_reason = ""
        if self.diffusion_adapter is not None:
            self.diffusion_adapter.clear_projected_trajectory_cache()

    def reset_policy_state(self, reset_kind: str):
        self.invalidate_policy_command_state()
        if self.diffusion_adapter is None:
            return
        snapshot = self.diffusion_adapter.reset_policy_state()
        self.last_policy_state_after_reset = snapshot
        if reset_kind == "new_goal":
            self.last_policy_state_reset_on_new_goal = True
        elif reset_kind == "goal_reached":
            self.last_policy_state_reset_on_goal_reached = True
        self.get_logger().info(f"policy state reset on {reset_kind}: {snapshot}")

    # The high-rate publisher samples the latest projected v(t), omega(t) trajectory.
    # It holds the immediate command only when projection sampling is unavailable and
    # projected_trajectory_fallback_to_hold is enabled.

    def policy_occ_world_xy_len(self) -> int:
        if self.diffusion_adapter is None or self.diffusion_adapter.policy is None:
            return 0
        occ = getattr(self.diffusion_adapter.policy, "_cur_occ_world_xy", None)
        if occ is None:
            return 0
        try:
            return int(len(occ))
        except Exception:
            return 0

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        self.latest_odom_time = self.now_sec()

    def people_callback(self, msg: People):
        self.latest_people = list(msg.people)
        self.latest_people_time = self.now_sec()
        header = getattr(msg, "header", None)
        self.latest_people_frame_id = getattr(header, "frame_id", "") if header is not None else ""

    def people_detector_status_callback(self, msg: String):
        ready, reason = parse_people_detector_status(msg.data)
        self.latest_people_detector_status_time = self.now_sec()
        self.latest_people_detector_ready = ready
        self.latest_people_detector_reason = reason

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        self.latest_map_time = self.now_sec()
        self._fused_occupancy_cache_key = None

    def lidar_callback(self, msg: LaserScan):
        self._latest_lidar_points_local = voxelized_laser_points(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            float(self.get_parameter("lidar_min_range_m").value),
            float(self.get_parameter("lidar_max_range_m").value),
            float(self.get_parameter("lidar_voxel_size_m").value),
            max(1, int(self.get_parameter("lidar_max_points").value)),
        )
        self._latest_lidar_frame_id = msg.header.frame_id or self.target_policy_frame
        self.latest_lidar_time = self.now_sec()
        self._latest_lidar_sequence += 1
        self.last_frame_debug["latest_lidar_header_frame_id"] = (
            self._latest_lidar_frame_id
        )
        self._live_lidar_cache_sequence = -1
        self._fused_occupancy_cache_key = None

    def goal_position_delta(self, old_goal: PoseStamped, new_goal: PoseStamped) -> float:
        dx = float(new_goal.pose.position.x) - float(old_goal.pose.position.x)
        dy = float(new_goal.pose.position.y) - float(old_goal.pose.position.y)
        dz = float(new_goal.pose.position.z) - float(old_goal.pose.position.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def goal_orientation_delta(self, old_goal: PoseStamped, new_goal: PoseStamped) -> float:
        return abs(wrap_angle(quaternion_to_yaw(new_goal.pose.orientation) - quaternion_to_yaw(old_goal.pose.orientation)))

    def is_duplicate_goal_against(self, reference: Optional[PoseStamped], msg: PoseStamped) -> bool:
        if reference is None:
            self.last_goal_position_delta = float("inf")
            self.last_goal_orientation_delta = float("inf")
            return False
        old_frame = reference.header.frame_id or "map"
        new_frame = msg.header.frame_id or "map"
        self.last_goal_position_delta = self.goal_position_delta(reference, msg)
        self.last_goal_orientation_delta = self.goal_orientation_delta(reference, msg)
        if old_frame != new_frame:
            return False
        pos_eps = float(self.get_parameter("goal_duplicate_position_epsilon").value)
        yaw_eps = float(self.get_parameter("goal_duplicate_orientation_epsilon").value)
        return self.last_goal_position_delta <= pos_eps and self.last_goal_orientation_delta <= yaw_eps

    def is_duplicate_goal(self, msg: PoseStamped) -> bool:
        return self.is_duplicate_goal_against(self.latest_goal, msg)

    def note_duplicate_goal(self):
        self.bridge_goal_republish_count += 1
        if self.bridge_goal_republish_count <= 3 or self.bridge_goal_republish_count % 50 == 0:
            self.get_logger().debug(
                "ignored duplicate goal republish: "
                f"count={self.bridge_goal_republish_count}, "
                f"position_delta={self.last_goal_position_delta:.4f}, "
                f"orientation_delta={self.last_goal_orientation_delta:.4f}"
            )

    def goal_callback(self, msg: PoseStamped):
        with self._goal_lock:
            self.last_policy_state_reset_on_new_goal = False
            self.last_goal_reset_triggered = False
            if not self.policy_ready_for_goal:
                reference = self._pending_goal_during_warmup or self.latest_goal
                duplicate = self.is_duplicate_goal_against(reference, msg)
                self.last_incoming_goal_is_duplicate = duplicate
                if duplicate:
                    self.note_duplicate_goal()
                    return
                self._pending_goal_during_warmup = copy.deepcopy(msg)
                if not self._pending_goal_during_warmup.header.frame_id:
                    self._pending_goal_during_warmup.header.frame_id = "map"
                self._pending_goal_received_time = self.now_sec()
                self.bridge_goal_republish_count = 0
                self.get_logger().info(
                    "queued latest genuine goal while policy warm-up is incomplete: "
                    f"x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}, "
                    f"frame={self._pending_goal_during_warmup.header.frame_id}"
                )
                return

            duplicate = self.is_duplicate_goal(msg)
            self.last_incoming_goal_is_duplicate = duplicate
            if duplicate:
                self.note_duplicate_goal()
                return
            self._accept_new_goal_locked(msg)

    def _accept_new_goal_locked(self, msg: PoseStamped):
        self._goal_generation += 1
        self.latest_goal = copy.deepcopy(msg)
        if not self.latest_goal.header.frame_id:
            self.latest_goal.header.frame_id = "map"
        self.latest_goal_time = self.now_sec()
        self.bridge_goal_republish_count = 0
        self._goal_reached_handled = False
        self.invalidate_policy_command_state()
        if bool(self.get_parameter("reset_policy_state_on_new_goal").value):
            self.reset_policy_state("new_goal")
            self.last_goal_reset_triggered = True
        self.publish_goal_visualization()
        self.get_logger().info(
            f"accepted new goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}, "
            f"frame={self.latest_goal.header.frame_id}, position_delta={self.last_goal_position_delta:.3f}, "
            f"orientation_delta={self.last_goal_orientation_delta:.3f}"
        )

    def apply_style_vector(self, values: List[float]) -> List[float]:
        """Push a clipped 4-vector (prox, pass, yield, group) into the live
        policy instance. Read fresh every predict() call
        (diffusion_CondUNetCFG.py), so this takes effect on the next
        inference tick with no model reload."""
        vector = [0.0, 0.0, 0.0, 0.0]
        for index in range(min(4, len(values))):
            vector[index] = clamp(float(values[index]), -1.0, 1.0)
        if (
            self.diffusion_adapter is not None
            and self.diffusion_adapter.policy is not None
            and hasattr(self.diffusion_adapter.policy, "style_vector")
        ):
            self.diffusion_adapter.policy.style_vector = list(vector)
            self.get_logger().info(
                f"style_vector applied: prox={vector[0]:.2f}, pass={vector[1]:.2f}, "
                f"yield={vector[2]:.2f}, group={vector[3]:.2f}"
            )
        self.current_style_vector = vector
        # style_vector_pub is created later in __init__ (adapter construction
        # happens before publisher setup); guard so the startup call is a
        # no-op and the republish right after publisher creation covers it.
        if getattr(self, "style_vector_pub", None) is not None:
            self.publish_style_vector_topic(vector)
        return vector

    def publish_style_vector_topic(self, vector: List[float]):
        # TRANSIENT_LOCAL so a late-starting subscriber (e.g. ps4_nav_trigger_node,
        # which uses this to attribute recorded bags to a style) still gets the
        # current value immediately without waiting for the next change.
        if self.style_vector_pub is None:
            return
        msg = Float32MultiArray()
        msg.data = [float(v) for v in vector]
        self.style_vector_pub.publish(msg)

    def _on_set_parameters(self, params) -> SetParametersResult:
        for param in params:
            if param.name != "style_vector":
                continue
            values = list(param.value)
            if len(values) != 4:
                return SetParametersResult(
                    success=False,
                    reason="style_vector must have exactly 4 values: prox, pass, yield, group",
                )
        for param in params:
            if param.name == "style_vector":
                self.apply_style_vector(list(param.value))
        return SetParametersResult(success=True)

    def build_robot_state(self) -> Dict[str, float]:
        pose = self.latest_odom.pose.pose
        twist = self.latest_odom.twist.twist
        odom_frame = str(self.get_parameter("odom_pose_source_frame_override").value) or self.latest_odom.header.frame_id or self.target_policy_frame
        self.last_frame_debug["latest_odom_header_frame_id"] = self.latest_odom.header.frame_id or ""
        self.last_frame_debug["latest_odom_child_frame_id"] = self.latest_odom.child_frame_id or ""
        self.last_frame_debug["target_policy_frame"] = self.target_policy_frame

        x, y = self.transform_xy_to_target(pose.position.x, pose.position.y, odom_frame, "tf_robot_to_target_success")
        yaw = self.yaw_to_target(quaternion_to_yaw(pose.orientation), odom_frame, "tf_robot_to_target_success")
        linear_velocity = float(twist.linear.x)
        angular_velocity = float(twist.angular.z)
        self.last_odom_linear_velocity_used_for_sync = linear_velocity
        self.last_odom_angular_velocity_used_for_sync = angular_velocity
        return {
            "x": float(x),
            "y": float(y),
            "yaw": yaw,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "frame_id": self.target_policy_frame,
            "_sync_policy_warm_start_from_odom": bool(
                self.get_parameter("sync_policy_warm_start_from_odom").value
            ),
            "_sync_prev_action_from_odom": bool(
                self.get_parameter("sync_prev_action_from_odom").value
            ),
        }

    def build_people_states(self, robot: Dict[str, float], source_people=None) -> List[Dict[str, float]]:
        states = []
        cos_yaw = math.cos(robot["yaw"])
        sin_yaw = math.sin(robot["yaw"])
        people_frame = self.latest_people_frame_id or str(self.get_parameter("people_frame").value) or self.target_policy_frame
        self.last_frame_debug["latest_people_header_frame_id"] = self.latest_people_frame_id
        self.last_frame_debug["people_frame_used"] = people_frame
        people_messages = self.latest_people if source_people is None else source_people
        for person in people_messages:
            px, py = self.transform_xy_to_target(
                person.position.x, person.position.y, people_frame, "tf_people_to_target_success"
            )
            vx, vy = self.rotate_velocity_to_target(
                getattr(person.velocity, "x", 0.0), getattr(person.velocity, "y", 0.0), people_frame, "tf_people_to_target_success"
            )
            dx = float(px) - robot["x"]
            dy = float(py) - robot["y"]
            rel_x = cos_yaw * dx + sin_yaw * dy
            rel_y = -sin_yaw * dx + cos_yaw * dy
            states.append(
                {
                    "x": float(px),
                    "y": float(py),
                    "vx": float(vx),
                    "vy": float(vy),
                    "distance": math.hypot(dx, dy),
                    "rel_x": rel_x,
                    "rel_y": rel_y,
                }
            )
        if not people_messages:
            self.last_frame_debug["tf_people_to_target_success"] = True
        return states

    def build_goal_state(self, robot: Dict[str, float]) -> Dict[str, float]:
        goal_frame = str(self.get_parameter("goal_frame_override").value) or self.latest_goal.header.frame_id or self.target_policy_frame
        self.last_frame_debug["latest_goal_header_frame_id"] = self.latest_goal.header.frame_id or ""
        gx, gy = self.transform_xy_to_target(
            self.latest_goal.pose.position.x, self.latest_goal.pose.position.y, goal_frame, "tf_goal_to_target_success"
        )
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

    def reset_projected_alignment_debug(self):
        self.last_projected_alignment_success = False
        self.last_projected_alignment_offset_sec = 0.0
        self.last_projected_alignment_position_error = float("nan")
        self.last_projected_alignment_heading_error = float("nan")

    def remember_goal_debug(self, goal: Dict[str, float]):
        self.last_policy_goal_x_map = float(goal["x"])
        self.last_policy_goal_y_map = float(goal["y"])
        self.last_goal_dx_map = float(goal["dx"])
        self.last_goal_dy_map = float(goal["dy"])
        self.last_goal_x_robot_frame = float(goal["rel_x"])
        self.last_goal_y_robot_frame = float(goal["rel_y"])
        self.last_distance_to_goal = float(goal["distance"])
        self.last_goal_distance_frame = self.target_policy_frame
        self.last_goal_within_tolerance = (
            self.last_distance_to_goal <= float(self.get_parameter("goal_tolerance").value)
        )
        self.last_goal_distance_tf_success = bool(
            self.last_frame_debug.get("tf_robot_to_target_success", False)
            and self.last_frame_debug.get("tf_goal_to_target_success", False)
        )
        self.last_heading_to_goal = float(goal["heading_to_goal"])
        self.last_heading_error = float(goal["heading_error"])

    def _build_static_occupancy_state(self) -> Optional[Dict]:
        if self.latest_map is None or bool(self.get_parameter("ignore_map_for_policy").value):
            self.last_static_map_has_map_value = 0.0
            return None
        if not bool(self.get_parameter("enable_static_map_to_policy").value):
            self.last_static_map_has_map_value = 0.0
            return None

        msg = self.latest_map
        info = msg.info
        map_frame = str(self.get_parameter("map_frame_override").value) or msg.header.frame_id or self.target_policy_frame
        self.last_frame_debug["latest_map_header_frame_id"] = msg.header.frame_id or ""
        self._static_map_frame_id = map_frame
        stamp = (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
            int(info.width),
            int(info.height),
            float(info.resolution),
            float(info.origin.position.x),
            float(info.origin.position.y),
            map_frame,
        )
        if (
            bool(self.get_parameter("static_map_update_on_change_only").value)
            and self._static_map_cache is not None
            and self._static_map_cache_stamp == stamp
        ):
            return self._static_map_cache

        occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        max_cells = max(1, int(self.get_parameter("max_static_map_cells").value))
        treat_unknown = bool(self.get_parameter("treat_unknown_as_occupied").value)
        resolution = float(info.resolution)
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)
        origin_yaw = quaternion_to_yaw(info.origin.orientation)
        cos_origin = math.cos(origin_yaw)
        sin_origin = math.sin(origin_yaw)

        cells = []
        width = int(info.width)
        height = int(info.height)
        data = list(msg.data)
        for index, value in enumerate(data):
            if value < 0 and not treat_unknown:
                continue
            if value < occupied_threshold and not (value < 0 and treat_unknown):
                continue
            cx = index % width
            cy = index // width
            local_x = (cx + 0.5) * resolution
            local_y = (cy + 0.5) * resolution
            wx = origin_x + cos_origin * local_x - sin_origin * local_y
            wy = origin_y + sin_origin * local_x + cos_origin * local_y
            cells.append((wx, wy))

        total = len(cells)
        downsampled = False
        if total > max_cells:
            stride = max(1, math.ceil(total / max_cells))
            cells = cells[::stride][:max_cells]
            downsampled = True
            self.get_logger().warn(
                f"static map occupied cells downsampled: total={total}, used={len(cells)}, max={max_cells}, stride={stride}"
            )

        if cells:
            transformed = []
            transform = self.lookup_transform_to_target(map_frame, "tf_map_to_target_success")
            for wx, wy in cells:
                if transform is None:
                    tx, ty = wx, wy
                else:
                    tx, ty = apply_transform_xy(wx, wy, transform)
                transformed.append((float(tx), float(ty)))
        else:
            self.last_frame_debug["tf_map_to_target_success"] = True
            transformed = []

        try:
            import numpy as np

            occ_world_xy = np.asarray(transformed, dtype=float) if transformed else np.zeros((0, 2), dtype=float)
        except Exception:
            occ_world_xy = transformed

        occupancy = {
            "msg": msg,
            "resolution": resolution,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "width": width,
            "height": height,
            "frame_id": map_frame,
            "occ_world_xy": occ_world_xy,
            "cells_total": total,
            "cells_used": len(transformed),
            "downsampled": downsampled,
        }
        self._static_map_cache = occupancy
        self._static_map_cache_stamp = stamp
        self._static_map_last_update_time = self.now_sec()
        self._static_map_cells_total = total
        self._static_map_cells_used = len(transformed)
        self._static_map_downsampled = downsampled
        self.last_static_map_has_map_value = 1.0 if transformed else 0.0
        return occupancy

    def build_live_lidar_points(self) -> List[Tuple[float, float]]:
        if not self.live_lidar_enabled:
            self.last_live_lidar_points_used = 0
            return []
        timeout = max(0.1, float(self.get_parameter("lidar_timeout_sec").value))
        if self.data_is_stale(self.latest_lidar_time, self.now_sec(), timeout):
            self.last_live_lidar_points_used = 0
            return []
        if self._live_lidar_cache_sequence == self._latest_lidar_sequence:
            return self._live_lidar_points_cache

        transform = self.lookup_transform_to_target(
            self._latest_lidar_frame_id,
            "tf_lidar_to_target_success",
        )
        if transform is None:
            transformed = self._latest_lidar_points_local
            origin_xy = (0.0, 0.0)
        else:
            transformed = [
                apply_transform_xy(x, y, transform)
                for x, y in self._latest_lidar_points_local
            ]
            origin_xy = (
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
            )
        now = self.now_sec()
        (
            self._live_lidar_obstacle_memory,
            self._live_lidar_points_cache,
        ) = update_obstacle_memory(
            self._live_lidar_obstacle_memory,
            transformed,
            now_sec=now,
            ttl_sec=float(
                self.get_parameter("lidar_obstacle_memory_sec").value
            ),
            voxel_size=float(self.get_parameter("lidar_voxel_size_m").value),
            max_points=max(1, int(self.get_parameter("lidar_max_points").value)),
            origin_xy=origin_xy,
        )
        self._live_lidar_cache_sequence = self._latest_lidar_sequence
        self.last_live_lidar_points_used = len(self._live_lidar_points_cache)
        return self._live_lidar_points_cache

    def build_occupancy_state(self) -> Optional[Dict]:
        static_occupancy = self._build_static_occupancy_state()
        live_points = self.build_live_lidar_points()
        if not live_points:
            return static_occupancy

        cache_key = (self._static_map_cache_stamp, self._live_lidar_cache_sequence)
        if (
            self._fused_occupancy_cache_key == cache_key
            and self._fused_occupancy_cache is not None
        ):
            return self._fused_occupancy_cache

        static_points = (
            static_occupancy.get("occ_world_xy")
            if static_occupancy is not None
            else None
        )
        fused_points = combine_occupancy_points(static_points, live_points)
        occupancy = dict(static_occupancy or {})
        occupancy.update({
            "frame_id": self.target_policy_frame,
            "occ_world_xy": fused_points,
            "live_lidar_points_used": len(live_points),
            "cells_total": int(occupancy.get("cells_total", 0))
            + len(live_points),
            "cells_used": int(occupancy.get("cells_used", 0))
            + len(live_points),
        })
        self._fused_occupancy_cache_key = cache_key
        self._fused_occupancy_cache = occupancy
        return occupancy

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

        try:
            robot = self.build_robot_state()
            goal = self.build_goal_state(robot)
        except Exception:
            return v, w
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

        try:
            robot = self.build_robot_state()
            goal = self.build_goal_state(robot)
        except Exception:
            return v, w
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

        command_time = self.now_sec()
        disable_publish = bool(self.get_parameter("disable_policy_command_publish").value)
        if disable_publish:
            v = 0.0
            w = 0.0
            self.last_linear_slew_limited = False
            self.last_linear_slew_dt_sec = float("nan")
            self.last_linear_slew_max_delta = float("nan")
            self.last_angular_slew_limited = False
            self.last_angular_slew_dt_sec = float("nan")
            self.last_angular_slew_max_delta = float("nan")
        else:
            actual_dt = self.cmd_publish_period
            if self.last_cmd_publish_time is not None:
                measured_dt = command_time - self.last_cmd_publish_time
                if measured_dt > 0.0:
                    actual_dt = measured_dt

            max_dv = abs(float(self.get_parameter("max_linear_accel").value)) * actual_dt
            slew_limited_v = clamp(
                v,
                self.last_cmd["v"] - max_dv,
                self.last_cmd["v"] + max_dv,
            )
            self.last_linear_slew_limited = abs(slew_limited_v - v) > 1e-9
            self.last_linear_slew_dt_sec = actual_dt
            self.last_linear_slew_max_delta = max_dv
            v = slew_limited_v

            max_dw = abs(float(self.get_parameter("max_angular_accel").value)) * actual_dt
            slew_limited_w = clamp(
                w,
                self.last_cmd["w"] - max_dw,
                self.last_cmd["w"] + max_dw,
            )
            self.last_angular_slew_limited = abs(slew_limited_w - w) > 1e-9
            self.last_angular_slew_dt_sec = actual_dt
            self.last_angular_slew_max_delta = max_dw
            w = slew_limited_w

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
        self.last_cmd_publish_time = command_time

    def publish_zero(self, reason: str):
        self.last_command_source = self.command_source_for_stop_reason(reason)
        if abs(self.last_cmd["v"]) > 1e-3 or abs(self.last_cmd["w"]) > 1e-3:
            self.get_logger().info(f"publishing zero command: {reason}")
        self.publish_zero_cmd_preserve_model_debug()

    def publish_zero_cmd_preserve_model_debug(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cmd_frame_id
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)
        self.last_final_cmd = {"v": 0.0, "w": 0.0}
        self.last_cmd_clamped = {"linear": False, "angular": False}
        self.last_cmd = {"v": 0.0, "w": 0.0}
        self.last_cmd_publish_time = self.now_sec()
        self.last_linear_slew_limited = False
        self.last_linear_slew_dt_sec = float("nan")
        self.last_linear_slew_max_delta = float("nan")
        self.last_angular_slew_limited = False
        self.last_angular_slew_dt_sec = float("nan")
        self.last_angular_slew_max_delta = float("nan")

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
        goal_frame = str(self.get_parameter("goal_frame_override").value) or self.latest_goal.header.frame_id or self.target_policy_frame
        try:
            goal_x, goal_y = self.transform_xy_to_target(
                self.latest_goal.pose.position.x, self.latest_goal.pose.position.y, goal_frame, "tf_goal_to_target_success"
            )
        except TransformException:
            return

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.target_policy_frame
        marker.ns = "social_nav_diffusion"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_x)
        marker.pose.position.y = float(goal_y)
        marker.pose.position.z = float(self.latest_goal.pose.position.z) + 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.45
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.9
        self.active_goal_marker_pub.publish(marker)

    def time_msg_from_sec(self, seconds: float):
        nanoseconds = max(0, int(float(seconds) * 1e9))
        return Time(nanoseconds=nanoseconds).to_msg()

    def clear_trajectory_paths(self):
        if not hasattr(self, "projected_trajectory_pub"):
            return
        stamp = self.get_clock().now().to_msg()
        for publisher in (self.projected_trajectory_pub, self.predicted_trajectory_pub):
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = self.target_policy_frame
            publisher.publish(path)
        self.last_projected_trajectory_available = False
        self.last_projected_trajectory_active = False
        self.last_projected_trajectory_point_count = 0
        self.last_projected_trajectory_publish_time = float("nan")
        self.last_projected_trajectory_input_stamp_sec = float("nan")
        self.last_projected_trajectory_result_stamp_sec = float("nan")
        self.last_projected_trajectory_latency_compensation_sec = float("nan")
        self.last_projected_trajectory_execution_age_sec = float("nan")
        self.last_predicted_trajectory_available = False
        self.last_predicted_trajectory_point_count = 0

    def publish_projected_trajectory_path(
        self,
        trajectory: Dict[str, Any],
        robot: Dict[str, float],
        result_stamp_sec: float,
    ):
        x_traj = trajectory.get("x_traj")
        if x_traj is None:
            self.clear_trajectory_paths()
            return

        try:
            point_count = len(x_traj)
        except TypeError:
            point_count = 0
        if point_count == 0:
            self.clear_trajectory_paths()
            return

        path = Path()
        path.header.frame_id = self.target_policy_frame
        path.header.stamp = self.time_msg_from_sec(result_stamp_sec)
        dt = float(trajectory.get("dt", 0.0))
        cos_yaw = math.cos(float(robot["yaw"]))
        sin_yaw = math.sin(float(robot["yaw"]))

        for index, state in enumerate(x_traj):
            local_x = float(state[0])
            local_y = float(state[1])
            local_yaw = float(state[2]) if len(state) > 2 else 0.0
            pose = PoseStamped()
            pose.header.frame_id = self.target_policy_frame
            pose.header.stamp = self.time_msg_from_sec(result_stamp_sec + max(0.0, dt) * index)
            pose.pose.position.x = float(robot["x"]) + cos_yaw * local_x - sin_yaw * local_y
            pose.pose.position.y = float(robot["y"]) + sin_yaw * local_x + cos_yaw * local_y
            pose.pose.position.z = 0.0
            set_quaternion_from_yaw(pose.pose.orientation, wrap_angle(float(robot["yaw"]) + local_yaw))
            path.poses.append(pose)

        self.projected_trajectory_pub.publish(path)
        self.last_projected_trajectory_point_count = len(path.poses)
        self.last_projected_trajectory_frame_id = self.target_policy_frame
        self.last_projected_trajectory_publish_time = float(result_stamp_sec)

    def publish_predicted_trajectory_path(
        self,
        trajectory: Optional[Dict[str, Any]],
        result_stamp_sec: float,
        dt: float,
    ):
        if not trajectory or trajectory.get("xy_world") is None:
            path = Path()
            path.header.frame_id = self.target_policy_frame
            path.header.stamp = self.time_msg_from_sec(result_stamp_sec)
            self.predicted_trajectory_pub.publish(path)
            self.last_predicted_trajectory_available = False
            self.last_predicted_trajectory_point_count = 0
            return

        xy_world = trajectory["xy_world"]
        try:
            point_count = len(xy_world)
        except TypeError:
            point_count = 0
        if point_count == 0:
            self.last_predicted_trajectory_available = False
            self.last_predicted_trajectory_point_count = 0
            return

        path = Path()
        path.header.frame_id = self.target_policy_frame
        path.header.stamp = self.time_msg_from_sec(result_stamp_sec)
        for index, point in enumerate(xy_world):
            if point_count == 1:
                yaw = 0.0
            elif index + 1 < point_count:
                yaw = math.atan2(
                    float(xy_world[index + 1][1]) - float(point[1]),
                    float(xy_world[index + 1][0]) - float(point[0]),
                )
            else:
                yaw = math.atan2(
                    float(point[1]) - float(xy_world[index - 1][1]),
                    float(point[0]) - float(xy_world[index - 1][0]),
                )
            pose = PoseStamped()
            pose.header.frame_id = self.target_policy_frame
            pose.header.stamp = self.time_msg_from_sec(result_stamp_sec + max(0.0, dt) * index)
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            set_quaternion_from_yaw(pose.pose.orientation, yaw)
            path.poses.append(pose)

        self.predicted_trajectory_pub.publish(path)
        self.last_predicted_trajectory_available = True
        self.last_predicted_trajectory_point_count = len(path.poses)

    # One uniform, muted color for every candidate so the K samples read as
    # a single "family" distinct from the projected path (green) and the
    # selected raw path (orange) — not a rainbow that implies per-index
    # meaning the samples don't actually have.
    CANDIDATE_TRAJECTORY_COLOR = (0.65, 0.65, 0.70)  # light steel gray

    def publish_candidate_trajectories_markers(
        self,
        samples: Optional[List[Any]],
        result_stamp_sec: float,
    ):
        if self.candidate_trajectories_pub is None:
            return
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        stamp = self.time_msg_from_sec(result_stamp_sec)
        if samples:
            for index, sample in enumerate(samples):
                try:
                    point_count = len(sample)
                except TypeError:
                    point_count = 0
                if point_count < 2:
                    continue
                marker = Marker()
                marker.header.frame_id = self.target_policy_frame
                marker.header.stamp = stamp
                marker.ns = "candidate_trajectory"
                marker.id = index
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.02
                marker.color.r, marker.color.g, marker.color.b = self.CANDIDATE_TRAJECTORY_COLOR
                # Uniform alpha for every candidate, including index 0 (the
                # selected sample) — it's already drawn opaque orange by the
                # Raw Predicted Trajectory Path, so no extra emphasis needed
                # here; that keeps this whole marker set visually "one thing."
                marker.color.a = 0.45
                marker.lifetime = Duration(seconds=0.5).to_msg()
                for point in sample:
                    p = Point()
                    p.x = float(point[0])
                    p.y = float(point[1])
                    p.z = 0.02
                    marker.points.append(p)
                markers.markers.append(marker)

        self.candidate_trajectories_pub.publish(markers)

    def planning_timing_averages(self) -> Dict[str, float]:
        """Rolling averages over the last <=50 predict() calls, for logging."""
        window = self.planning_timing_window
        if not window:
            return {
                "diffusion_ms_avg": float("nan"),
                "projection_ms_avg": float("nan"),
                "planning_ms_avg": float("nan"),
                "planning_timing_samples": 0,
            }
        count = len(window)
        return {
            "diffusion_ms_avg": sum(float(t.get("diffusion_ms", 0.0)) for t in window) / count,
            "projection_ms_avg": sum(float(t.get("projection_ms", 0.0)) for t in window) / count,
            "planning_ms_avg": sum(float(t.get("total_ms", 0.0)) for t in window) / count,
            "planning_timing_samples": count,
        }

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
            try:
                robot_dbg = self.build_robot_state()
                robot_x = float(robot_dbg["x"])
                robot_y = float(robot_dbg["y"])
                robot_yaw = float(robot_dbg["yaw"])
            except Exception:
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
        projected_trajectory_age = (
            now - self.last_projected_trajectory_publish_time
            if math.isfinite(self.last_projected_trajectory_publish_time)
            else float("nan")
        )
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
            "require_people_stream": bool(self.get_parameter("require_people_stream").value),
            "ignore_map_for_policy": bool(self.get_parameter("ignore_map_for_policy").value),
            "force_zero_humans": bool(self.get_parameter("force_zero_humans").value),
            "fixed_test_goal_in_robot_frame": bool(self.get_parameter("fixed_test_goal_in_robot_frame").value),
            "disable_policy_command_publish": bool(self.get_parameter("disable_policy_command_publish").value),
            "sync_policy_warm_start_from_odom": bool(self.get_parameter("sync_policy_warm_start_from_odom").value),
            "sync_prev_action_from_odom": bool(self.get_parameter("sync_prev_action_from_odom").value),
            "policy_warmup_enabled": self.policy_warmup_enabled,
            "policy_warmup_started": self.policy_warmup_started,
            "policy_warmup_complete": self.policy_warmup_complete,
            "policy_warmup_success": self.policy_warmup_success,
            "policy_warmup_duration_sec": self.policy_warmup_duration_sec,
            "policy_ready_for_goal": self.policy_ready_for_goal,
            "policy_warmup_error": self.policy_warmup_error,
            "warmup_timeout_sec": float(self.get_parameter("warmup_timeout_sec").value),
            "publish_cmd_during_warmup": bool(self.get_parameter("publish_cmd_during_warmup").value),
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
            "live_lidar_enabled": self.live_lidar_enabled,
            "require_live_lidar": self.require_live_lidar,
            "live_lidar_points_used": self.last_live_lidar_points_used,
            "live_lidar_max_points": int(self.get_parameter("lidar_max_points").value),
            "live_lidar_obstacle_memory_sec": float(
                self.get_parameter("lidar_obstacle_memory_sec").value
            ),
            "live_lidar_age_sec": (
                now - self.latest_lidar_time
                if self.latest_lidar_time is not None
                else float("nan")
            ),
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
            "linear_slew_limited": self.last_linear_slew_limited,
            "linear_slew_dt_sec": self.last_linear_slew_dt_sec,
            "linear_slew_max_delta": self.last_linear_slew_max_delta,
            "max_linear_accel": float(self.get_parameter("max_linear_accel").value),
            "angular_slew_limited": self.last_angular_slew_limited,
            "angular_slew_dt_sec": self.last_angular_slew_dt_sec,
            "angular_slew_max_delta": self.last_angular_slew_max_delta,
            "max_angular_accel": float(self.get_parameter("max_angular_accel").value),
            "goal_age_sec": goal_age,
            "active_goal_age_sec": goal_age,
            "incoming_goal_is_duplicate": self.last_incoming_goal_is_duplicate,
            "goal_position_delta": self.last_goal_position_delta,
            "goal_orientation_delta": self.last_goal_orientation_delta,
            "goal_reset_triggered": self.last_goal_reset_triggered,
            "goal_distance_frame": self.last_goal_distance_frame,
            "goal_distance_tf_success": self.last_goal_distance_tf_success,
            "policy_goal_tolerance": float(self.get_parameter("goal_tolerance").value),
            "bridge_goal_tolerance": float(self.get_parameter("bridge_goal_tolerance").value),
            "goal_within_tolerance": self.last_goal_within_tolerance,
            "goal_distance_common_frame": self.last_distance_to_goal,
            "bridge_goal_republish_count": self.bridge_goal_republish_count,
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
            "cmd_publish_period_sec": self.cmd_publish_period,
            "timing_match_policy_dt": (not math.isnan(self.get_policy_time_step())) and abs(self.get_policy_time_step() - self.diffusion_inference_period) <= 1e-6,
            "timing_warning": "" if (math.isnan(self.get_policy_time_step()) or abs(self.get_policy_time_step() - self.diffusion_inference_period) <= 1e-6) else "policy_time_step != diffusion_inference_period_sec; verify projected sampling and fallback timing",
            "latest_odom_header_frame_id": self.last_frame_debug.get("latest_odom_header_frame_id", ""),
            "latest_odom_child_frame_id": self.last_frame_debug.get("latest_odom_child_frame_id", ""),
            "latest_goal_header_frame_id": self.last_frame_debug.get("latest_goal_header_frame_id", ""),
            "latest_map_header_frame_id": self.last_frame_debug.get("latest_map_header_frame_id", ""),
            "latest_people_header_frame_id": self.last_frame_debug.get("latest_people_header_frame_id", ""),
            "latest_lidar_header_frame_id": self.last_frame_debug.get("latest_lidar_header_frame_id", ""),
            "people_frame_used": self.last_frame_debug.get("people_frame_used", ""),
            "target_policy_frame": self.target_policy_frame,
            "tf_robot_to_target_success": self.last_frame_debug.get("tf_robot_to_target_success", False),
            "tf_goal_to_target_success": self.last_frame_debug.get("tf_goal_to_target_success", False),
            "tf_people_to_target_success": self.last_frame_debug.get("tf_people_to_target_success", False),
            "tf_map_to_target_success": self.last_frame_debug.get("tf_map_to_target_success", False),
            "tf_lidar_to_target_success": self.last_frame_debug.get("tf_lidar_to_target_success", False),
            "tf_failure_reason": self.last_frame_debug.get("tf_failure_reason", ""),
            "static_map_enabled": bool(self.get_parameter("enable_static_map_to_policy").value),
            "static_map_frame_id": self._static_map_frame_id,
            "static_map_cells_total": self._static_map_cells_total,
            "static_map_cells_used": self._static_map_cells_used,
            "static_map_downsampled": self._static_map_downsampled,
            "static_map_has_map_value": self.last_static_map_has_map_value,
            "static_map_last_update_time": self._static_map_last_update_time,
            "policy_cur_has_map": float(getattr(self.diffusion_adapter.policy, "_cur_has_map", float("nan"))) if self.diffusion_adapter is not None else float("nan"),
            "policy_cur_occ_world_xy_len": self.policy_occ_world_xy_len(),
            "max_linear_speed": float(self.get_parameter("max_linear_speed").value),
            "max_angular_speed": float(self.get_parameter("max_angular_speed").value),
            "goal_tolerance": float(self.get_parameter("goal_tolerance").value),
            "policy_config_max_linear_speed": self.diffusion_adapter.policy_config_max_linear_speed if self.diffusion_adapter is not None else float("nan"),
            "policy_config_max_angular_speed": self.diffusion_adapter.policy_config_max_angular_speed if self.diffusion_adapter is not None else float("nan"),
            "policy_config_max_linear_accel": self.diffusion_adapter.policy_config_max_linear_accel if self.diffusion_adapter is not None else float("nan"),
            "policy_config_max_angular_accel": self.diffusion_adapter.policy_config_max_angular_accel if self.diffusion_adapter is not None else float("nan"),
            "checkpoint_path_requested": self.diffusion_adapter.checkpoint_path_requested if self.diffusion_adapter is not None else "",
            "checkpoint_path_resolved": self.diffusion_adapter.checkpoint_path_resolved if self.diffusion_adapter is not None else "",
            "checkpoint_exists": self.diffusion_adapter.checkpoint_exists if self.diffusion_adapter is not None else False,
            "policy_config_ckpt_path": self.diffusion_adapter.policy_config_ckpt_path if self.diffusion_adapter is not None else "",
            "policy_config_ckpt_resolved": self.diffusion_adapter.policy_config_ckpt_resolved if self.diffusion_adapter is not None else "",
            "policy_config_ckpt_exists": self.diffusion_adapter.policy_config_ckpt_exists if self.diffusion_adapter is not None else False,
            "checkpoint_consistency_warning": self.diffusion_adapter.checkpoint_consistency_warning if self.diffusion_adapter is not None else "",
            "reset_policy_state_on_new_goal": bool(self.get_parameter("reset_policy_state_on_new_goal").value),
            "reset_policy_state_on_goal_reached": bool(self.get_parameter("reset_policy_state_on_goal_reached").value),
            "policy_state_reset_on_new_goal": self.last_policy_state_reset_on_new_goal,
            "policy_state_reset_on_goal_reached": self.last_policy_state_reset_on_goal_reached,
            "policy_prev_v_after_reset": self.last_policy_state_after_reset.get("prev_v", float("nan")),
            "policy_prev_omega_after_reset": self.last_policy_state_after_reset.get("prev_omega", float("nan")),
            "policy_v0_from_last_step_after_reset": self.last_policy_state_after_reset.get("_v0_from_last_step", float("nan")),
            "policy_w0_from_last_step_after_reset": self.last_policy_state_after_reset.get("_w0_from_last_step", float("nan")),
            "policy_prev_v_after_sync": self.last_policy_state_after_sync.get("prev_v", float("nan")),
            "policy_prev_omega_after_sync": self.last_policy_state_after_sync.get("prev_omega", float("nan")),
            "policy_v0_from_last_step_after_sync": self.last_policy_state_after_sync.get("_v0_from_last_step", float("nan")),
            "policy_w0_from_last_step_after_sync": self.last_policy_state_after_sync.get("_w0_from_last_step", float("nan")),
            "enable_projected_trajectory_sampling": bool(self.get_parameter("enable_projected_trajectory_sampling").value),
            "enable_projected_trajectory_latency_compensation": bool(self.get_parameter("enable_projected_trajectory_latency_compensation").value),
            "projected_trajectory_fallback_to_hold": bool(self.get_parameter("projected_trajectory_fallback_to_hold").value),
            "projected_trajectory_available": self.last_projected_trajectory_available,
            "projected_trajectory_active": self.last_projected_trajectory_active,
            "projected_trajectory_sample_time_sec": self.last_projected_trajectory_sample_time_sec,
            "projected_trajectory_horizon_sec": self.last_projected_trajectory_horizon_sec,
            "projected_trajectory_dt": self.last_projected_trajectory_dt,
            "projected_trajectory_fallback_reason": self.last_projected_trajectory_fallback_reason,
            "predicted_trajectory_available": self.last_predicted_trajectory_available,
            "predicted_trajectory_point_count": self.last_predicted_trajectory_point_count,
            "projected_trajectory_point_count": self.last_projected_trajectory_point_count,
            "projected_trajectory_frame_id": self.last_projected_trajectory_frame_id,
            "projected_trajectory_publish_time": self.last_projected_trajectory_publish_time,
            "projected_trajectory_age_sec": projected_trajectory_age,
            "projected_trajectory_input_stamp_sec": self.last_projected_trajectory_input_stamp_sec,
            "projected_trajectory_result_stamp_sec": self.last_projected_trajectory_result_stamp_sec,
            "projected_trajectory_latency_compensation_sec": self.last_projected_trajectory_latency_compensation_sec,
            "projected_trajectory_execution_age_sec": self.last_projected_trajectory_execution_age_sec,
            "projected_alignment_success": self.last_projected_alignment_success,
            "projected_alignment_offset_sec": self.last_projected_alignment_offset_sec,
            "projected_alignment_position_error": self.last_projected_alignment_position_error,
            "projected_alignment_heading_error": self.last_projected_alignment_heading_error,
            "projected_alignment_max_position_error": float(
                self.get_parameter("projected_alignment_max_position_error").value
            ),
            "projected_trajectory_topic": self.projected_trajectory_topic,
            "actionxy_angular_dt_used": self.diffusion_adapter.last_actionxy_angular_dt_used if self.diffusion_adapter is not None else float("nan"),
            "sign_conflict_guard_active": self.last_sign_conflict_guard_active,
            "sign_conflict_heading_error_rad": float(self.get_parameter("sign_conflict_heading_error_rad").value),
            "sign_conflict_min_raw_angular": float(self.get_parameter("sign_conflict_min_raw_angular").value),
            "sign_conflict_desired_w": self.last_sign_conflict_desired_w,
            "sign_conflict_linear_scale": float(self.get_parameter("sign_conflict_linear_scale").value),
        }
        if self.test_mode:
            fields["test_mode"] = True
            fields["style_vector"] = list(self.current_style_vector)
            fields.update(self.planning_timing_averages())

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
            "max_angular_accel": float(self.get_parameter("max_angular_accel").value),
            "enable_static_map_to_policy": bool(self.get_parameter("enable_static_map_to_policy").value),
            "robot_v_pref": float(self.get_parameter("robot_v_pref").value),
            "robot_radius": float(self.get_parameter("robot_radius").value),
            "human_radius": float(self.get_parameter("human_radius").value),
            "dt": self.cmd_publish_period if dt is None else dt,
        }

    def handle_goal_reached(self):
        if self._goal_reached_handled:
            return
        self._goal_generation += 1
        if bool(self.get_parameter("reset_policy_state_on_goal_reached").value):
            self.reset_policy_state("goal_reached")
        else:
            self.invalidate_policy_command_state()
        self.latest_goal = None
        self.latest_goal_time = None
        self._goal_reached_handled = True

    def handle_goal_timeout(self):
        self._goal_generation += 1
        self.invalidate_policy_command_state()
        self.latest_goal = None
        self.latest_goal_time = None
        self._goal_reached_handled = True
        self.last_goal_within_tolerance = False

    def prepare_policy_inputs(self, now: float):
        stale_timeout = float(self.get_parameter("stale_timeout_sec").value)

        if self.latest_goal is None:
            return None, "no goal received"
        if self.data_is_stale(self.latest_odom_time, now, stale_timeout):
            return None, f"no fresh {self.odom_topic}"
        if self.live_lidar_enabled and self.require_live_lidar:
            lidar_timeout = max(
                0.1,
                float(self.get_parameter("lidar_timeout_sec").value),
            )
            if self.data_is_stale(self.latest_lidar_time, now, lidar_timeout):
                return None, f"no fresh {self.lidar_topic}"
        goal_timeout = float(self.get_parameter("goal_timeout_sec").value)
        if self.data_is_stale(self.latest_goal_time, now, goal_timeout):
            return None, f"goal timed out: older than {goal_timeout:.1f}s"
        ignore_people = bool(self.get_parameter("ignore_people_for_policy").value)
        require_people = bool(self.get_parameter("require_people_stream").value)
        people_status_stale = self.data_is_stale(
            self.latest_people_detector_status_time,
            now,
            stale_timeout,
        )
        people_stale = self.data_is_stale(
            self.latest_people_time,
            now,
            stale_timeout,
        )
        if not ignore_people and require_people and people_status_stale:
            return None, "required people detector status is missing or stale"
        if (
            not ignore_people
            and require_people
            and not self.latest_people_detector_ready
        ):
            return None, (
                "people detector is not ready: "
                f"{self.latest_people_detector_reason}"
            )
        if not ignore_people and require_people and people_stale:
            return None, "required /people stream is missing or stale"
        if people_stale and now - self._last_source_log_time > 1.0:
            self.get_logger().warn("/people is stale; continuing with empty people list")
            self._last_source_log_time = now

        try:
            robot = self.build_robot_state()
            people_seen = self.build_people_states(robot)
            if (
                people_stale
                or bool(self.get_parameter("ignore_people_for_policy").value)
                or bool(self.get_parameter("force_zero_humans").value)
            ):
                people = []
            else:
                people = people_seen
            self.last_humans_used = len(people)

            goal = self.build_goal_state(robot)
            self.remember_goal_debug(goal)
            occupancy = self.build_occupancy_state()
        except TransformException as exc:
            return None, f"required TF transform failed: {exc}"
        except Exception as exc:
            return None, f"policy input preparation failed: {type(exc).__name__}: {exc}"

        self.publish_goal_visualization()

        stop_when_goal_reached = bool(self.get_parameter("stop_when_goal_reached").value)
        enable_goal_stop = bool(self.get_parameter("enable_goal_stop").value)
        if enable_goal_stop and stop_when_goal_reached and goal["distance"] <= float(self.get_parameter("goal_tolerance").value):
            self.handle_goal_reached()
            return None, "goal reached"

        return (robot, people, goal, occupancy), None

    def align_projected_trajectory_start(
        self,
        trajectory: Optional[Dict[str, Any]],
        inference_robot: Dict[str, float],
        elapsed_ros_sec: float,
    ) -> float:
        self.reset_projected_alignment_debug()
        if not trajectory or trajectory.get("x_traj") is None:
            return 0.0

        try:
            current_robot = self.build_robot_state()
            x_traj = trajectory["x_traj"]
            dt = float(trajectory.get("dt", float("nan")))
            point_count = len(x_traj)
            if point_count == 0 or not math.isfinite(dt) or dt <= 0.0:
                return 0.0

            origin_yaw = float(inference_robot["yaw"])
            dx = float(current_robot["x"]) - float(inference_robot["x"])
            dy = float(current_robot["y"]) - float(inference_robot["y"])
            cos_yaw = math.cos(origin_yaw)
            sin_yaw = math.sin(origin_yaw)
            current_x_ego = cos_yaw * dx + sin_yaw * dy
            current_y_ego = -sin_yaw * dx + cos_yaw * dy
            current_heading_ego = wrap_angle(float(current_robot["yaw"]) - origin_yaw)
            current_v = float(current_robot.get("linear_velocity", 0.0))
            current_w = float(current_robot.get("angular_velocity", 0.0))

            max_offset = max(0.0, float(elapsed_ros_sec)) + self.cmd_publish_period
            max_index = min(point_count - 1, int(math.floor(max_offset / dt)) + 1)
            best = None
            for index in range(max_index + 1):
                state = x_traj[index]
                position_error = math.hypot(
                    float(state[0]) - current_x_ego,
                    float(state[1]) - current_y_ego,
                )
                heading_error = abs(wrap_angle(float(state[2]) - current_heading_ego))
                velocity_error = abs(float(state[3]) - current_v)
                angular_error = abs(float(state[4]) - current_w)
                cost = (
                    position_error
                    + 0.15 * heading_error
                    + 0.10 * velocity_error
                    + 0.03 * angular_error
                )
                candidate = (cost, index, position_error, heading_error)
                if best is None or candidate < best:
                    best = candidate

            if best is None:
                return 0.0
            _, index, position_error, heading_error = best
            self.last_projected_alignment_position_error = position_error
            self.last_projected_alignment_heading_error = heading_error
            max_position_error = max(
                0.0,
                float(self.get_parameter("projected_alignment_max_position_error").value),
            )
            if position_error > max_position_error:
                return 0.0

            offset = float(index) * dt
            self.last_projected_alignment_success = True
            self.last_projected_alignment_offset_sec = offset
            return offset
        except Exception as exc:
            self.get_logger().warn(
                f"projected trajectory state alignment failed; starting at t=0: "
                f"{type(exc).__name__}: {exc}"
            )
            return 0.0

    def set_latest_policy_cmd(
        self,
        v: float,
        w: float,
        stamp: float,
        duration_sec: float,
        projected_trajectory: Optional[Dict[str, Any]] = None,
        trajectory_origin_stamp: Optional[float] = None,
        trajectory_start_offset_sec: float = 0.0,
        trajectory_alignment_success: bool = False,
    ):
        origin_stamp = stamp if trajectory_origin_stamp is None else trajectory_origin_stamp
        with self._inference_lock:
            self._latest_policy_cmd = {
                "v": float(v),
                "w": float(w),
                "stamp": float(stamp),
                "trajectory_origin_stamp": float(origin_stamp),
                "duration_sec": float(duration_sec),
                "projected_trajectory": projected_trajectory,
                "trajectory_start_offset_sec": float(trajectory_start_offset_sec),
                "trajectory_alignment_success": bool(trajectory_alignment_success),
            }

    def get_latest_policy_cmd(self) -> Optional[Dict[str, Any]]:
        with self._inference_lock:
            return dict(self._latest_policy_cmd) if self._latest_policy_cmd is not None else None

    def clear_latest_policy_cmd(self):
        with self._inference_lock:
            self._latest_policy_cmd = None

    def sample_projected_policy_cmd(self, cmd: Dict[str, Any], age: float) -> Optional[Dict[str, float]]:
        if not bool(self.get_parameter("enable_projected_trajectory_sampling").value):
            self.last_projected_trajectory_active = False
            self.last_projected_trajectory_fallback_reason = "projected trajectory sampling disabled"
            return None
        if self.diffusion_adapter is None:
            self.last_projected_trajectory_active = False
            self.last_projected_trajectory_fallback_reason = "no diffusion adapter"
            return None
        trajectory = cmd.get("projected_trajectory")
        if not trajectory:
            self.last_projected_trajectory_active = False
            self.last_projected_trajectory_fallback_reason = "no projected trajectory available"
            return None
        horizon = float(trajectory.get("horizon_sec", float("nan")))
        self.last_projected_trajectory_available = True
        self.last_projected_trajectory_horizon_sec = horizon
        self.last_projected_trajectory_dt = float(trajectory.get("dt", float("nan")))
        if not math.isnan(horizon) and age > horizon:
            self.last_projected_trajectory_active = False
            self.last_projected_trajectory_fallback_reason = f"projected trajectory expired: age={age:.3f}s > horizon={horizon:.3f}s"
            return None
        sample = self.diffusion_adapter.sample_projected_trajectory(trajectory, age)
        if sample is None:
            self.last_projected_trajectory_active = False
            self.last_projected_trajectory_fallback_reason = "projected trajectory sample failed"
            return None
        self.last_projected_trajectory_active = True
        self.last_projected_trajectory_sample_time_sec = float(sample["sample_time_sec"])
        self.last_projected_trajectory_fallback_reason = ""
        return sample

    def command_publish_callback(self):
        if self.use_diffusion_policy and not self.policy_ready_for_goal:
            self.clear_latest_policy_cmd()
            self.last_command_source = "no_goal"
            self.publish_zero_cmd_preserve_model_debug()
            return

        now = self.now_sec()
        inputs, stop_reason = self.prepare_policy_inputs(now)
        if stop_reason is not None:
            if "goal timed out" in stop_reason.lower():
                self.handle_goal_timeout()
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
        command_age = max(0.0, now - float(cmd["stamp"]))
        trajectory_origin_stamp = float(cmd.get("trajectory_origin_stamp", cmd["stamp"]))
        latency_compensation_enabled = bool(
            self.get_parameter("enable_projected_trajectory_latency_compensation").value
        )
        alignment_success = bool(cmd.get("trajectory_alignment_success", False))
        alignment_offset = max(0.0, float(cmd.get("trajectory_start_offset_sec", 0.0)))
        if alignment_success:
            trajectory_age = alignment_offset + command_age
        elif latency_compensation_enabled:
            trajectory_age = max(0.0, now - trajectory_origin_stamp)
        else:
            trajectory_age = command_age
        self.last_projected_trajectory_execution_age_sec = trajectory_age
        effective_timeout = max(
            self.command_hold_timeout,
            cmd["duration_sec"] + self.diffusion_inference_period + self.cmd_publish_period,
        )
        if command_age > effective_timeout:
            self.publish_zero(f"held diffusion command timeout: {command_age:.2f}s > {effective_timeout:.2f}s")
            return

        sampled_cmd = self.sample_projected_policy_cmd(cmd, trajectory_age)
        if sampled_cmd is not None:
            v = sampled_cmd["v"]
            w = sampled_cmd["w"]
            if now - self._last_hold_log_time > 1.0:
                self.get_logger().info(
                    f"publishing sampled projected trajectory command: v={v:.3f}, w={w:.3f}, "
                    f"sample_t={sampled_cmd['sample_time_sec']:.3f}s, result_age={command_age:.2f}s, "
                    f"trajectory_age={trajectory_age:.2f}s, "
                    f"horizon={self.last_projected_trajectory_horizon_sec:.2f}s, "
                    f"inference duration={cmd['duration_sec']:.3f}s"
                )
                self._last_hold_log_time = now
            self.last_command_source = "diffusion_projected"
            self.publish_cmd(v, w)
            return

        if not bool(self.get_parameter("projected_trajectory_fallback_to_hold").value):
            self.publish_zero(self.last_projected_trajectory_fallback_reason or "projected trajectory unavailable")
            return

        if now - self._last_hold_log_time > 1.0:
            self.get_logger().info(
                f"republishing held diffusion command: v={cmd['v']:.3f}, w={cmd['w']:.3f}, "
                f"result_age={command_age:.2f}s, trajectory_age={trajectory_age:.2f}s, inference duration={cmd['duration_sec']:.3f}s, "
                f"effective timeout={effective_timeout:.2f}s, "
                f"projected fallback={self.last_projected_trajectory_fallback_reason}"
            )
            self._last_hold_log_time = now
        self.last_command_source = "diffusion"
        self.publish_cmd(cmd["v"], cmd["w"])

    def diffusion_inference_callback(self):
        if not (self.use_diffusion_policy and self.diffusion_adapter is not None):
            return
        if not self.policy_ready_for_goal:
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
        with self._goal_lock:
            goal_generation = self._goal_generation

        self._diffusion_thread = threading.Thread(
            target=self.run_diffusion_inference,
            args=(robot, people, goal, occupancy, last_cmd, limits, goal_generation, now),
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
        goal_generation: int,
        trajectory_origin_stamp: float,
    ):
        start_wall = time.perf_counter()
        try:
            v, w = self.diffusion_adapter.compute_action(robot, people, goal, occupancy, last_cmd, limits)
            self.last_raw_action_type = self.diffusion_adapter.last_action_type
            self.last_policy_state_after_sync = dict(self.diffusion_adapter.last_policy_state_after_sync)
            self.last_raw_model_v_before_conversion = self.diffusion_adapter.last_raw_model_v
            self.last_raw_model_r_or_w_before_conversion = self.diffusion_adapter.last_raw_model_r_or_w
            with self._goal_lock:
                result_is_stale = (
                    goal_generation != self._goal_generation
                    or self.latest_goal is None
                )
            if result_is_stale:
                self.diffusion_adapter.reset_policy_state()
                self.diffusion_adapter.clear_projected_trajectory_cache()
                self.clear_trajectory_paths()
                self.get_logger().warn(
                    "discarded diffusion result because its goal is no longer active"
                )
                return
            # Record the immediate ActionRot/ActionXY conversion as soon as inference returns.
            # Zero safety publishes may happen before/after this, but they should not erase
            # the model-conversion debug fields that diagnose the adapter path.
            self.last_converted_cmd = {"v": float(v), "w": float(w)}
            self.last_raw_cmd = {"v": float(v), "w": float(w)}
            duration = time.perf_counter() - start_wall
            projected_trajectory = self.diffusion_adapter.projected_trajectory_for_command()
            predicted_trajectory = self.diffusion_adapter.predicted_trajectory_for_command()
            result_stamp = self.now_sec()
            self.last_projected_trajectory_input_stamp_sec = float(trajectory_origin_stamp)
            self.last_projected_trajectory_result_stamp_sec = float(result_stamp)
            self.last_projected_trajectory_latency_compensation_sec = max(
                0.0, float(result_stamp) - float(trajectory_origin_stamp)
            )
            self.last_projected_trajectory_available = projected_trajectory is not None
            alignment_offset = self.align_projected_trajectory_start(
                projected_trajectory,
                robot,
                self.last_projected_trajectory_latency_compensation_sec,
            )
            alignment_success = self.last_projected_alignment_success
            if projected_trajectory is not None:
                self.last_projected_trajectory_horizon_sec = float(
                    projected_trajectory.get("horizon_sec", float("nan"))
                )
                self.last_projected_trajectory_dt = float(
                    projected_trajectory.get("dt", float("nan"))
                )
                self.publish_projected_trajectory_path(projected_trajectory, robot, result_stamp)
            else:
                empty_path = Path()
                empty_path.header.frame_id = self.target_policy_frame
                empty_path.header.stamp = self.time_msg_from_sec(result_stamp)
                self.projected_trajectory_pub.publish(empty_path)
                self.last_projected_trajectory_point_count = 0
                self.last_projected_trajectory_publish_time = float("nan")
            self.publish_predicted_trajectory_path(
                predicted_trajectory,
                result_stamp,
                self.last_projected_trajectory_dt,
            )
            if self.test_mode:
                candidate_trajectories = (
                    self.diffusion_adapter.all_candidate_trajectories_for_command()
                )
                self.publish_candidate_trajectories_markers(
                    candidate_trajectories,
                    result_stamp,
                )
                planning_timing = self.diffusion_adapter.last_planning_timing()
                if planning_timing is not None:
                    self.planning_timing_window.append(planning_timing)
            self.set_latest_policy_cmd(
                v, w, result_stamp, duration, projected_trajectory,
                trajectory_origin_stamp=trajectory_origin_stamp,
                trajectory_start_offset_sec=alignment_offset,
                trajectory_alignment_success=alignment_success,
            )
            now = result_stamp
            if now - self._last_source_log_time > 1.0:
                self.get_logger().info(
                    f"command source=diffusion, v={v:.3f}, w={w:.3f}, "
                    f"inference duration={duration:.3f}s, "
                    f"action={self.diffusion_adapter.last_action_fields}"
                )
                self._last_source_log_time = now
        except Exception as exc:
            self.clear_trajectory_paths()
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
            node.clear_trajectory_paths()
            node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
