import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from visualization_msgs.msg import Marker


@dataclass
class ActiveGoal:
    goal_handle: object
    pose: PoseStamped
    start_time: float
    done_event: threading.Event


def seconds_to_duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    msg = Duration()
    msg.sec = int(seconds)
    msg.nanosec = int((seconds - msg.sec) * 1e9)
    return msg


def goal_id_text(goal_handle) -> str:
    goal_id = goal_handle.goal_id.uuid
    return ''.join(f'{byte:02x}' for byte in goal_id)


class Nav2GoalToPoseBridge(Node):
    def __init__(self):
        super().__init__('nav2_goal_to_pose_bridge')

        self.declare_parameter('goal_tolerance', 0.35)
        self.declare_parameter('action_timeout_sec', 60.0)
        self.declare_parameter('republish_rate_hz', 10.0)
        self.declare_parameter('robot_odom_topic', '/cpr_j100_0001/platform/odom')

        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.action_timeout_sec = float(self.get_parameter('action_timeout_sec').value)
        self.republish_rate_hz = float(self.get_parameter('republish_rate_hz').value)
        self.robot_odom_topic = str(self.get_parameter('robot_odom_topic').value)

        self.callback_group = ReentrantCallbackGroup()
        self.active_goal: Optional[ActiveGoal] = None
        self.active_goal_lock = threading.Lock()
        self.latest_odom: Optional[Odometry] = None

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.robot_goal_pub = self.create_publisher(PoseStamped, '/cpr_j100_0001/goal_pose', 10)
        self.active_goal_marker_pub = self.create_publisher(Marker, '/social_nav_diffusion/active_goal_marker', 10)
        self.create_subscription(
            Odometry,
            self.robot_odom_topic,
            self.odom_callback,
            20,
            callback_group=self.callback_group,
        )

        self.global_action_server = ActionServer(
            self,
            NavigateToPose,
            '/navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )
        self.robot_action_server = ActionServer(
            self,
            NavigateToPose,
            '/cpr_j100_0001/navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.republish_period = 1.0 / max(self.republish_rate_hz, 0.1)

        self.get_logger().info(
            'nav2_goal_to_pose_bridge ready: action servers at /navigate_to_pose and '
            '/cpr_j100_0001/navigate_to_pose; publishing goals to /goal_pose and '
            f'/cpr_j100_0001/goal_pose at {self.republish_rate_hz:.1f} Hz; '
            f'odom={self.robot_odom_topic}'
        )

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def goal_callback(self, goal_request: NavigateToPose.Goal):
        pose = goal_request.pose
        frame_id = pose.header.frame_id or 'map'
        self.get_logger().info(
            f'accepted NavigateToPose goal: x={pose.pose.position.x:.2f}, '
            f'y={pose.pose.position.y:.2f}, frame={frame_id}'
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info(f'cancel requested for NavigateToPose goal {goal_id_text(goal_handle)}')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal = self.make_pose_msg(goal_handle.request.pose)
        active = ActiveGoal(
            goal_handle=goal_handle,
            pose=goal,
            start_time=self.now_sec(),
            done_event=threading.Event(),
        )

        old_goal = None
        with self.active_goal_lock:
            old_goal = self.active_goal
            self.active_goal = active

        if old_goal is not None and old_goal.goal_handle is not goal_handle:
            self.get_logger().warn('replacing previous NavigateToPose goal with a new active goal')
            old_goal.goal_handle.abort()
            old_goal.done_event.set()

        self.publish_goal(active.pose)
        last_republish_time = active.start_time
        result = NavigateToPose.Result()

        while rclpy.ok() and not active.done_event.wait(0.05):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                active.done_event.set()
                self.clear_active_goal(active)
                result.error_code = NavigateToPose.Result.NONE
                result.error_msg = 'Goal canceled'
                self.get_logger().info(f'NavigateToPose goal canceled: {goal_id_text(goal_handle)}')
                return result

            now = self.now_sec()
            if now - last_republish_time >= self.republish_period:
                self.publish_goal(active.pose)
                last_republish_time = now

            distance = self.distance_to_goal(active.pose)
            self.publish_feedback(active, now, distance)

            if distance is not None and distance <= self.goal_tolerance:
                goal_handle.succeed()
                active.done_event.set()
                self.clear_active_goal(active)
                result.error_code = NavigateToPose.Result.NONE
                result.error_msg = ''
                self.get_logger().info(
                    f'NavigateToPose goal reached: distance={distance:.3f} <= {self.goal_tolerance:.3f}'
                )
                return result

            elapsed = now - active.start_time
            if elapsed > self.action_timeout_sec:
                goal_handle.abort()
                active.done_event.set()
                self.clear_active_goal(active)
                result.error_code = NavigateToPose.Result.NONE
                result.error_msg = f'Goal timed out after {elapsed:.1f}s'
                self.get_logger().warn(result.error_msg)
                return result

        self.clear_active_goal(active)
        result.error_code = NavigateToPose.Result.NONE
        result.error_msg = 'Bridge shutting down'
        return result

    def republish_callback(self):
        active = self.get_active_goal()
        if active is None or active.done_event.is_set():
            return
        active.pose.header.stamp = self.get_clock().now().to_msg()
        self.publish_goal(active.pose)

    def publish_goal(self, pose: PoseStamped):
        pose.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(pose)
        self.robot_goal_pub.publish(pose)
        self.publish_goal_marker(pose)

    def publish_goal_marker(self, pose: PoseStamped):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "social_nav_diffusion"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(pose.pose.position.x)
        marker.pose.position.y = float(pose.pose.position.y)
        marker.pose.position.z = float(pose.pose.position.z) + 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.45
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.9
        self.active_goal_marker_pub.publish(marker)

    def publish_feedback(self, active: ActiveGoal, now: float, distance: Optional[float]):
        feedback = NavigateToPose.Feedback()
        if self.latest_odom is not None:
            feedback.current_pose.header = self.latest_odom.header
            feedback.current_pose.pose = self.latest_odom.pose.pose
        else:
            feedback.current_pose.header.stamp = self.get_clock().now().to_msg()
            feedback.current_pose.header.frame_id = active.pose.header.frame_id
        feedback.navigation_time = seconds_to_duration(now - active.start_time)
        feedback.distance_remaining = float(distance) if distance is not None else float('nan')
        speed = self.current_speed()
        if distance is not None and speed > 1e-3:
            feedback.estimated_time_remaining = seconds_to_duration(distance / speed)
        else:
            feedback.estimated_time_remaining = seconds_to_duration(0.0)
        feedback.number_of_recoveries = 0
        active.goal_handle.publish_feedback(feedback)

    def make_pose_msg(self, incoming: PoseStamped) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = incoming.header.frame_id or 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose = incoming.pose
        return pose

    def distance_to_goal(self, pose: PoseStamped) -> Optional[float]:
        if self.latest_odom is None:
            return None
        robot = self.latest_odom.pose.pose.position
        goal = pose.pose.position
        return math.hypot(float(goal.x) - float(robot.x), float(goal.y) - float(robot.y))

    def current_speed(self) -> float:
        if self.latest_odom is None:
            return 0.0
        twist = self.latest_odom.twist.twist
        return math.hypot(float(twist.linear.x), float(twist.linear.y))

    def get_active_goal(self) -> Optional[ActiveGoal]:
        with self.active_goal_lock:
            return self.active_goal

    def clear_active_goal(self, active: ActiveGoal):
        with self.active_goal_lock:
            if self.active_goal is active:
                self.active_goal = None

    def stop_active_goal(self):
        active = self.get_active_goal()
        if active is None or active.done_event.is_set():
            return
        try:
            active.goal_handle.abort()
        except Exception:
            pass
        active.done_event.set()
        self.clear_active_goal(active)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = Nav2GoalToPoseBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_active_goal()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
