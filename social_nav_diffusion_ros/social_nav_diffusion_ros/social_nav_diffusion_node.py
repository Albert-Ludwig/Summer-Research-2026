import os
import sys

INFERENCE_REPO = "/workspace/SocialNavDiffusion_Inference"
VENV_PYTHON = f"{INFERENCE_REPO}/.venv/bin/python"

if os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON) and sys.prefix != f"{INFERENCE_REPO}/.venv":
    env = os.environ.copy()
    env["ACADOS_SOURCE_DIR"] = "/home/ubuntu/acados"
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":/home/ubuntu/acados/lib"
    extra_pythonpath = [
        INFERENCE_REPO,
        f"{INFERENCE_REPO}/diffusers_unet_1d_condition",
        "/home/ubuntu/hunav_jazzy_ws/install/social_nav_diffusion_ros/lib/python3.12/site-packages",
        "/home/ubuntu/hunav_jazzy_ws/install/people_msgs/lib/python3.12/site-packages",
    ]
    env["PYTHONPATH"] = ":".join(extra_pythonpath + [env.get("PYTHONPATH", "")])
    print(f"[social_nav_diffusion_node] re-exec into venv: {VENV_PYTHON}", flush=True)
    os.execve(VENV_PYTHON, [VENV_PYTHON] + sys.argv, env)

import configparser
import copy
import time
import traceback

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from people_msgs.msg import People
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

print(f"[social_nav_diffusion_node] running python: {sys.executable}", flush=True)

INFERENCE_REPO = "/workspace/SocialNavDiffusion_Inference"
POLICY_DIR = os.path.join(INFERENCE_REPO, "crowd_nav", "policy")
CONFIG_PATH = os.path.join(INFERENCE_REPO, "crowd_nav", "configs", "policy.config")
ENV_CONFIG_PATH = os.path.join(INFERENCE_REPO, "crowd_nav", "configs", "env.config")
CKPT_PATH = os.path.join(INFERENCE_REPO, "SocialGuidedNavPlanner.pt")
NORM_PATH = os.path.join(INFERENCE_REPO, "norm_stats_SOCIAL_NORMS8.npy")

for path in (
    INFERENCE_REPO,
    POLICY_DIR,
    os.path.join(INFERENCE_REPO, "diffusers_unet_1d_condition"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("ACADOS_SOURCE_DIR", "/home/ubuntu/acados")
os.environ["LD_LIBRARY_PATH"] = (
    os.environ.get("LD_LIBRARY_PATH", "") + ":/home/ubuntu/acados/lib"
)

from crowd_nav.policy.diffusion_CondUNetCFG import DiffusionConditionalUNet1DCFG
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def action_fields(action):
    if hasattr(action, "_asdict"):
        return dict(action._asdict())
    return vars(action) if hasattr(action, "__dict__") else str(action)


class SocialNavDiffusionNode(Node):
    def __init__(self):
        super().__init__("social_nav_diffusion_node")

        self.declare_parameter("goal_x", 3.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("planner_period_sec", 5.0)

        self.goal_x = self.get_parameter("goal_x").value
        self.goal_y = self.get_parameter("goal_y").value
        self.planner_period_sec = self.get_parameter("planner_period_sec").value

        self.latest_odom = None
        self.latest_people = []

        self.debug_action_pub = self.create_publisher(
            String, "/social_nav_diffusion/debug_action", 10
        )
        self.debug_traj_pub = self.create_publisher(
            Marker, "/social_nav_diffusion/debug_trajectory", 10
        )

        self.create_subscription(
            Odometry,
            "/cpr_j100_0001/platform/odom",
            self.odom_callback,
            10,
        )
        self.create_subscription(
            People,
            "/people",
            self.people_callback,
            10,
        )

        self.policy = self.load_policy()
        self.timer = self.create_timer(self.planner_period_sec, self.timer_callback)

        self.get_logger().info(
            "social_nav_diffusion_node ready. Debug only: not publishing /cmd_vel."
        )

    def load_policy(self):
        config = configparser.RawConfigParser()
        config.read(CONFIG_PATH)

        section = "diffusion_conditional_unet1dcfg"
        config = copy.deepcopy(config)
        config.set(section, "ckpt_path", CKPT_PATH)
        config.set(section, "norm_file", NORM_PATH)

        policy = DiffusionConditionalUNet1DCFG()
        policy.configure(config)

        env_config = configparser.RawConfigParser()
        env_config.read(ENV_CONFIG_PATH)
        policy.time_step = (
            env_config.getfloat("env", "time_step")
            if env_config.has_option("env", "time_step")
            else 0.25
        )
        policy.set_static_map(None, 0.0, policy.map_extent)
        return policy

    def odom_callback(self, msg):
        self.latest_odom = msg

    def people_callback(self, msg):
        self.latest_people = list(msg.people)

    def build_state(self):
        odom = self.latest_odom
        pose = odom.pose.pose
        twist = odom.twist.twist

        robot_state = FullState(
            px=float(pose.position.x),
            py=float(pose.position.y),
            vx=float(twist.linear.x),
            vy=float(twist.linear.y),
            radius=0.3,
            gx=float(self.goal_x),
            gy=float(self.goal_y),
            v_pref=1.0,
            theta=quaternion_to_yaw(pose.orientation),
        )

        human_states = []
        for person in self.latest_people:
            human_states.append(
                ObservableState(
                    px=float(person.position.x),
                    py=float(person.position.y),
                    vx=float(person.velocity.x),
                    vy=float(person.velocity.y),
                    radius=0.3,
                )
            )

        return JointState(robot_state, human_states)

    def timer_callback(self):
        if self.latest_odom is None:
            msg = "waiting for /cpr_j100_0001/platform/odom"
            self.get_logger().info(msg)
            self.publish_debug_action(msg)
            return

        try:
            state = self.build_state()
            t0 = time.perf_counter()
            action = self.policy.predict(state)
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            predicted = self.policy.predicted_traj or {}
            used_projection = bool(predicted.get("used_projection", False))

            debug = (
                f"action type: {type(action).__name__}; "
                f"action fields: {action_fields(action)}; "
                f"people count: {len(state.human_states)}; "
                f"used_projection: {used_projection}; "
                f"runtime ms: {runtime_ms:.3f}"
            )
            self.publish_debug_action(debug)
            self.publish_debug_trajectory(predicted)
        except Exception as exc:
            err = (
                f"inference error: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            self.get_logger().error(err)
            self.publish_debug_action(err)

    def publish_debug_action(self, text):
        msg = String()
        msg.data = text
        self.debug_action_pub.publish(msg)

    def publish_debug_trajectory(self, predicted):
        traj = predicted.get("projection")
        if traj is None:
            traj = predicted.get("selected_sample")
        if traj is None:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "social_nav_diffusion"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.04
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        for xy in traj:
            point = Point()
            point.x = float(xy[0])
            point.y = float(xy[1])
            point.z = 0.05
            marker.points.append(point)

        self.debug_traj_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SocialNavDiffusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
