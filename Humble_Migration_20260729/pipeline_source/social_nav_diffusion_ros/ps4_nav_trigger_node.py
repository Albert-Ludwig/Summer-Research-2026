#!/usr/bin/env python3
"""PS4-controller start/stop trigger for the diffusion navigation stack.

One button toggles: on press while idle, sends a NavigateToPose goal a fixed
distance ahead of the robot's current heading (computed once via TF, not
continuously) to the same action server nav2_goal_to_pose_bridge already
serves; on press while active, cancels it. A std_msgs/Bool on
`nav_enabled_topic` mirrors that state so jackal_twist_adapter's optional
`require_nav_trigger` gate can use it as a final "commands reach the wheels
or not" switch, independent of the policy's own internal state.

Optionally starts/stops one `ros2 bag record -s mcap ...` subprocess per run
so each test is captured to its own timestamped bag without hand-holding.
"""

import datetime
import math
import os
import signal
import subprocess
import threading
import time
from typing import List, Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

DEFAULT_BAG_TOPICS = [
    '/clock',
    # NOTE: `ros2 bag record` runs as a plain subprocess here, not a launch
    # Node, so it does NOT get jackal_realtime_social_nav_debug.launch.py's
    # /tf -> /jackal1/tf remap. Record the real namespaced topics directly;
    # override `bag_topics` if a different launch file's remapping differs.
    '/jackal1/tf',
    '/jackal1/tf_static',
    '/jackal1/map',
    '/jackal1/robot_description',
    '/jackal1/platform/odom/filtered',
    '/jackal1/joy_teleop/joy',
    '/people',
    '/people_detector/status',
    '/people_detector/markers',
    '/goal_pose',
    '/debug_cmd_vel',
    '/jackal1/cmd_vel',
    '/social_nav_diffusion/policy_debug',
    '/social_nav_diffusion/predicted_trajectory',
    '/social_nav_diffusion/projected_trajectory',
    '/social_nav_diffusion/candidate_trajectories',
    '/social_nav_diffusion/active_goal_marker',
    '/social_nav_diffusion/nav_enabled',
    '/social_nav_diffusion/style_vector',
]


def quaternion_from_yaw(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_process_group_exit(
    process_group_id: int,
    timeout_sec: float,
    process: Optional[subprocess.Popen] = None,
) -> bool:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while process_group_exists(process_group_id):
        if process is not None:
            process.poll()
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


class Ps4NavTriggerNode(Node):
    def __init__(self):
        super().__init__('ps4_nav_trigger_node')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('trigger_button_index', 7)
        self.declare_parameter('goal_distance_m', 6.0)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('tf_lookup_timeout_sec', 0.2)
        self.declare_parameter('navigate_to_pose_action', '/navigate_to_pose')
        self.declare_parameter(
            'robot_navigate_to_pose_action',
            '/cpr_j100_0001/navigate_to_pose',
        )
        self.declare_parameter('nav_enabled_topic', '/social_nav_diffusion/nav_enabled')
        self.declare_parameter('style_vector_topic', '/social_nav_diffusion/style_vector')
        self.declare_parameter('action_wait_timeout_sec', 5.0)
        self.declare_parameter('record_bag', True)
        self.declare_parameter('bag_output_dir', '/workspace/bags')
        self.declare_parameter('bag_topics', DEFAULT_BAG_TOPICS)
        self.declare_parameter('bag_stop_grace_sec', 3.0)

        self.joy_topic = str(self.get_parameter('joy_topic').value)
        self.trigger_button_index = int(self.get_parameter('trigger_button_index').value)
        self.goal_distance_m = float(self.get_parameter('goal_distance_m').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.tf_lookup_timeout_sec = float(self.get_parameter('tf_lookup_timeout_sec').value)
        self.navigate_to_pose_action = str(self.get_parameter('navigate_to_pose_action').value)
        self.robot_navigate_to_pose_action = str(
            self.get_parameter('robot_navigate_to_pose_action').value
        )
        self.nav_enabled_topic = str(self.get_parameter('nav_enabled_topic').value)
        self.style_vector_topic = str(self.get_parameter('style_vector_topic').value)
        self.action_wait_timeout_sec = float(self.get_parameter('action_wait_timeout_sec').value)
        self.record_bag = bool(self.get_parameter('record_bag').value)
        self.bag_output_dir = str(self.get_parameter('bag_output_dir').value)
        self.bag_topics = [str(topic) for topic in self.get_parameter('bag_topics').value]
        self.bag_stop_grace_sec = float(self.get_parameter('bag_stop_grace_sec').value)

        self.callback_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_active = False
        self._goal_pending = False
        self.last_button_state = 0
        self._goal_handle = None
        self._state_lock = threading.Lock()
        self._bag_lock = threading.Lock()
        self._bag_process: Optional[subprocess.Popen] = None
        self._bag_process_group_id: Optional[int] = None
        # Latest style_vector reported by policy_cmd_vel_node (TRANSIENT_LOCAL
        # publisher, so this arrives immediately even if that node started
        # first). Used only to make bag folder names self-attributable; if
        # no message has arrived yet, the folder name says so explicitly
        # rather than silently implying "all zeros".
        self.latest_style_vector: Optional[List[float]] = None

        self.nav_enabled_pub = self.create_publisher(Bool, self.nav_enabled_topic, 10)
        # Fail-safe: announce "not enabled" once at startup so a late-starting
        # jackal_twist_adapter (require_nav_trigger=true) doesn't sit on a
        # stale/unknown default for longer than necessary.
        self.nav_enabled_pub.publish(Bool(data=False))

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            self.robot_navigate_to_pose_action,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            Joy,
            self.joy_topic,
            self.joy_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Float32MultiArray,
            self.style_vector_topic,
            self.style_vector_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f'ps4_nav_trigger_node ready: joy_topic={self.joy_topic}, '
            f'button_index={self.trigger_button_index}, '
            f'goal_distance_m={self.goal_distance_m:.2f}, '
            f'action={self.robot_navigate_to_pose_action}, '
            f'record_bag={self.record_bag}. '
            'Verify the button index with `ros2 topic echo <joy_topic>` and '
            'pressing the intended button once before relying on this in a test.'
        )

    def style_vector_callback(self, msg: Float32MultiArray):
        self.latest_style_vector = [float(v) for v in msg.data]

    def style_vector_suffix(self) -> str:
        """Filesystem-safe folder-name fragment so each bag is attributable
        to its test condition at a glance, without opening the bag."""
        names = ('prox', 'pass', 'yield', 'group')
        vector = self.latest_style_vector
        if vector is None or len(vector) != 4:
            return 'style_unknown'
        # Negative sign spelled as 'n', not '-', to stay unambiguous across
        # tools/shells that treat a leading '-' as an option separator.
        return '_'.join(
            f'{name}{"n" + f"{abs(value):.2f}" if value < 0 else f"{value:.2f}"}'
            for name, value in zip(names, vector)
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def joy_callback(self, msg: Joy):
        if self.trigger_button_index < 0 or self.trigger_button_index >= len(msg.buttons):
            return
        state = int(msg.buttons[self.trigger_button_index])
        if state == 1 and self.last_button_state == 0:
            self.on_button_pressed()
        self.last_button_state = state

    def on_button_pressed(self):
        with self._state_lock:
            active = self.nav_active
            pending = self._goal_pending
        if pending:
            self.get_logger().info(
                'trigger button pressed while goal response is pending; ignoring duplicate start'
            )
            return
        if active:
            self.get_logger().info('trigger button pressed: stopping nav')
            self.stop_nav('button pressed while active')
        else:
            self.get_logger().info('trigger button pressed: starting nav')
            self.start_nav()

    def compute_goal_pose(self) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=RclpyDuration(seconds=self.tf_lookup_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().error(
                f'cannot compute fixed goal: TF lookup {self.map_frame}<-{self.base_frame} '
                f'failed: {exc}'
            )
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        siny_cosp = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
        cosy_cosp = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(translation.x) + self.goal_distance_m * math.cos(yaw)
        goal.pose.position.y = float(translation.y) + self.goal_distance_m * math.sin(yaw)
        goal.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def start_nav(self):
        with self._state_lock:
            if self.nav_active or self._goal_pending:
                return
            self._goal_pending = True

        goal_pose = self.compute_goal_pose()
        if goal_pose is None:
            with self._state_lock:
                self._goal_pending = False
            self.get_logger().error('not starting nav: could not compute fixed goal pose')
            return

        if not self.action_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            with self._state_lock:
                self._goal_pending = False
            self.get_logger().error(
                f'not starting nav: action server {self.robot_navigate_to_pose_action} '
                'unavailable'
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            f'requesting fixed goal: x={goal_pose.pose.position.x:.2f}, '
            f'y={goal_pose.pose.position.y:.2f}, frame={goal_pose.header.frame_id}'
        )
        try:
            send_goal_future = self.action_client.send_goal_async(goal_msg)
        except Exception as exc:
            with self._state_lock:
                self._goal_pending = False
            self.get_logger().error(f'goal send failed: {exc}')
            return
        send_goal_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'goal send failed: {exc}')
            self.stop_nav('goal send failed')
            return
        if not goal_handle.accepted:
            self.get_logger().error('goal rejected by action server')
            self.stop_nav('goal rejected')
            return

        with self._state_lock:
            if not self._goal_pending:
                stale_response = True
            else:
                stale_response = False
                self._goal_pending = False
                self.nav_active = True
                self._goal_handle = goal_handle

        if stale_response:
            self.get_logger().warn('canceling stale accepted goal response')
            goal_handle.cancel_goal_async()
            return

        self.nav_enabled_pub.publish(Bool(data=True))
        if self.record_bag:
            self.start_bag_recording()
        self.get_logger().info('fixed goal accepted; navigation output enabled')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        with self._state_lock:
            was_active = self.nav_active
        if not was_active:
            # Already stopped via a manual button press/cancel; avoid a
            # duplicate stop_nav() call racing the one that already ran.
            return
        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.get_logger().warn(f'goal result future failed: {exc}')
            status = GoalStatus.STATUS_UNKNOWN
        self.get_logger().info(f'nav goal finished on its own, status={status}')
        self.stop_nav(f'goal finished, status={status}')

    def stop_nav(self, reason: str):
        with self._state_lock:
            was_active = self.nav_active
            was_pending = self._goal_pending
            self.nav_active = False
            self._goal_pending = False
            goal_handle = self._goal_handle
            self._goal_handle = None

        self.nav_enabled_pub.publish(Bool(data=False))
        if was_active or was_pending or goal_handle is not None:
            self.get_logger().info(f'nav stopped: {reason}')
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warn(
                    f'cancel_goal_async failed (goal may already be done): {exc}'
                )
        self.stop_bag_recording()

    def start_bag_recording(self):
        try:
            os.makedirs(self.bag_output_dir, exist_ok=True)
        except OSError as exc:
            self.get_logger().error(f'cannot create bag_output_dir {self.bag_output_dir}: {exc}')
            return
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        bag_path = os.path.join(
            self.bag_output_dir,
            f'run_{timestamp}_{self.style_vector_suffix()}',
        )
        command = ['ros2', 'bag', 'record', '-s', 'mcap', '-o', bag_path] + self.bag_topics
        with self._bag_lock:
            if self._bag_process_group_id is not None:
                if process_group_exists(self._bag_process_group_id):
                    self.get_logger().warn(
                        'bag recording already active; '
                        'refusing duplicate recorder'
                    )
                    return
                self._bag_process = None
                self._bag_process_group_id = None
            try:
                process = subprocess.Popen(command, start_new_session=True)
            except OSError as exc:
                self.get_logger().error(f'failed to start ros2 bag record: {exc}')
                return
            self._bag_process = process
            self._bag_process_group_id = process.pid
        self.get_logger().info(f'bag recording started: {bag_path}')

    def stop_bag_recording(self):
        with self._bag_lock:
            process = self._bag_process
            process_group_id = self._bag_process_group_id
            self._bag_process = None
            self._bag_process_group_id = None
        if process_group_id is None:
            return

        try:
            os.killpg(process_group_id, signal.SIGINT)
            if wait_for_process_group_exit(
                process_group_id,
                self.bag_stop_grace_sec,
                process,
            ):
                self.get_logger().info('bag recording stopped cleanly')
                return
            self.get_logger().warn('bag process group did not exit after SIGINT; terminating')
            os.killpg(process_group_id, signal.SIGTERM)
            if wait_for_process_group_exit(process_group_id, 1.0, process):
                return
            self.get_logger().warn('bag process group did not exit after SIGTERM; killing')
            os.killpg(process_group_id, signal.SIGKILL)
            wait_for_process_group_exit(process_group_id, 1.0, process)
        except ProcessLookupError:
            return
        except Exception as exc:
            self.get_logger().warn(f'error stopping bag record: {exc}')
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass

    def shutdown(self):
        self.stop_nav('node shutting down')
        self.stop_bag_recording()


def main(args=None):
    rclpy.init(args=args)
    node = Ps4NavTriggerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
