#!/usr/bin/env python3

import math
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def laser_scan_points_base(
    scan: LaserScan,
    configured_min_range: float,
    configured_max_range: float,
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
) -> List[Tuple[float, float]]:
    """Return valid scan points in the robot base frame."""
    lower = max(float(scan.range_min), float(configured_min_range), 0.0)
    upper_candidates = [float(configured_max_range)]
    if math.isfinite(float(scan.range_max)) and float(scan.range_max) > 0.0:
        upper_candidates.append(float(scan.range_max))
    upper = min(upper_candidates)
    if upper <= lower:
        return []

    points = []
    cos_sensor = math.cos(float(sensor_yaw))
    sin_sensor = math.sin(float(sensor_yaw))
    angle = float(scan.angle_min)
    increment = float(scan.angle_increment)
    for raw_range in scan.ranges:
        distance = float(raw_range)
        if math.isfinite(distance) and lower <= distance <= upper:
            scan_x = distance * math.cos(angle)
            scan_y = distance * math.sin(angle)
            points.append((
                float(sensor_x) + cos_sensor * scan_x - sin_sensor * scan_y,
                float(sensor_y) + sin_sensor * scan_x + cos_sensor * scan_y,
            ))
        angle += increment
    return points


def braking_poses(
    linear: float,
    angular: float,
    reaction_time: float,
    linear_decel: float,
    angular_decel: float,
    step_sec: float,
    max_horizon_sec: float,
) -> List[Tuple[float, float, float, float]]:
    """Predict base poses while reacting and braking from one command."""
    linear_decel = max(float(linear_decel), 1e-3)
    angular_decel = max(float(angular_decel), 1e-3)
    reaction_time = max(0.0, float(reaction_time))
    horizon = max(
        reaction_time + abs(float(linear)) / linear_decel,
        reaction_time + abs(float(angular)) / angular_decel,
    )
    horizon = min(max(0.0, float(max_horizon_sec)), horizon)
    step_sec = max(0.01, float(step_sec))
    steps = max(1, int(math.ceil(horizon / step_sec)))
    dt = horizon / steps if horizon > 0.0 else step_sec

    poses = [(0.0, 0.0, 0.0, 0.0)]
    x = 0.0
    y = 0.0
    yaw = 0.0
    for index in range(steps):
        time_mid = (index + 0.5) * dt
        braking_time = max(0.0, time_mid - reaction_time)
        linear_now = math.copysign(
            max(0.0, abs(float(linear)) - linear_decel * braking_time),
            float(linear),
        )
        angular_now = math.copysign(
            max(0.0, abs(float(angular)) - angular_decel * braking_time),
            float(angular),
        )
        yaw_mid = yaw + 0.5 * angular_now * dt
        x += linear_now * math.cos(yaw_mid) * dt
        y += linear_now * math.sin(yaw_mid) * dt
        yaw = normalize_angle(yaw + angular_now * dt)
        poses.append((x, y, yaw, (index + 1) * dt))
    return poses


def swept_footprint_collision(
    points: Sequence[Tuple[float, float]],
    linear: float,
    angular: float,
    footprint_half_length: float,
    footprint_half_width: float,
    safety_margin: float,
    reaction_time: float,
    linear_decel: float,
    angular_decel: float,
    step_sec: float,
    max_horizon_sec: float,
) -> Optional[Tuple[float, float, float]]:
    """Return the first point/time intersecting the command's stopping sweep."""
    if abs(float(linear)) <= 1e-6 and abs(float(angular)) <= 1e-6:
        return None

    half_length = max(0.0, float(footprint_half_length))
    half_width = max(0.0, float(footprint_half_width))
    inflated_length = half_length + max(0.0, float(safety_margin))
    inflated_width = half_width + max(0.0, float(safety_margin))
    poses = braking_poses(
        linear,
        angular,
        reaction_time,
        linear_decel,
        angular_decel,
        step_sec,
        max_horizon_sec,
    )

    for point_x, point_y in points:
        # Ignore returns from the robot itself, but retain nearby external points.
        if abs(point_x) <= half_length and abs(point_y) <= half_width:
            continue
        for pose_x, pose_y, pose_yaw, pose_time in poses:
            dx = point_x - pose_x
            dy = point_y - pose_y
            cos_yaw = math.cos(pose_yaw)
            sin_yaw = math.sin(pose_yaw)
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            if (
                abs(local_x) <= inflated_length
                and abs(local_y) <= inflated_width
            ):
                return float(point_x), float(point_y), float(pose_time)
    return None


class JackalTwistAdapter(Node):
    def __init__(self):
        super().__init__('jackal_twist_adapter')

        self.declare_parameter('test_mode', False)
        self.declare_parameter('input_topic', '/debug_cmd_vel')
        self.declare_parameter('output_topic', '/jackal1/cmd_vel')
        self.declare_parameter(
            'emergency_stop_topic',
            '/jackal1/platform/emergency_stop',
        )
        self.declare_parameter('enable_output', False)
        self.declare_parameter('require_emergency_stop_clear', True)
        self.declare_parameter('continuous_zero_output', False)
        self.declare_parameter('require_nav_trigger', False)
        self.declare_parameter(
            'nav_enabled_topic',
            '/social_nav_diffusion/nav_enabled',
        )
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter(
            'max_angular_speed',
            3.14,
        )
        self.declare_parameter('input_timeout', 0.5)
        self.declare_parameter('enable_lidar_safety', False)
        self.declare_parameter(
            'lidar_topic',
            '/jackal1/sensors/lidar3d_0/scan',
        )
        self.declare_parameter('lidar_timeout', 0.4)
        self.declare_parameter('lidar_min_range', 0.15)
        self.declare_parameter('lidar_max_range', 6.0)
        self.declare_parameter('lidar_sensor_x', 0.12)
        self.declare_parameter('lidar_sensor_y', 0.0)
        self.declare_parameter('lidar_sensor_yaw', 0.0)
        self.declare_parameter('footprint_length', 0.51)
        self.declare_parameter('footprint_width', 0.43)
        self.declare_parameter('footprint_safety_margin', 0.05)
        self.declare_parameter('collision_reaction_time', 0.15)
        self.declare_parameter('collision_linear_decel', 1.5)
        self.declare_parameter('collision_angular_decel', 3.14)
        self.declare_parameter('collision_step_sec', 0.05)
        self.declare_parameter('collision_max_horizon_sec', 1.5)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.emergency_stop_topic = str(
            self.get_parameter('emergency_stop_topic').value
        )
        self.output_enabled = bool(self.get_parameter('enable_output').value)
        self.require_emergency_stop_clear = bool(
            self.get_parameter('require_emergency_stop_clear').value
        )
        self.continuous_zero_output = bool(
            self.get_parameter('continuous_zero_output').value
        )
        self.test_mode = bool(self.get_parameter('test_mode').value)
        self.require_nav_trigger = self.test_mode and bool(
            self.get_parameter('require_nav_trigger').value
        )
        self.nav_enabled_topic = str(
            self.get_parameter('nav_enabled_topic').value
        )
        # Fail-safe default: blocked until the trigger node publishes True,
        # mirroring require_emergency_stop_clear's default-blocking behavior.
        self.nav_trigger_enabled = not self.require_nav_trigger
        self.max_linear_speed = abs(
            float(self.get_parameter('max_linear_speed').value)
        )
        self.max_angular_speed = abs(
            float(self.get_parameter('max_angular_speed').value)
        )
        self.input_timeout = max(
            0.05,
            float(self.get_parameter('input_timeout').value),
        )
        self.lidar_safety_enabled = bool(
            self.get_parameter('enable_lidar_safety').value
        )
        self.lidar_topic = str(self.get_parameter('lidar_topic').value)
        self.lidar_timeout = max(
            0.1,
            float(self.get_parameter('lidar_timeout').value),
        )
        self.lidar_min_range = max(
            0.0,
            float(self.get_parameter('lidar_min_range').value),
        )
        self.lidar_max_range = max(
            self.lidar_min_range,
            float(self.get_parameter('lidar_max_range').value),
        )
        self.lidar_sensor_x = float(
            self.get_parameter('lidar_sensor_x').value
        )
        self.lidar_sensor_y = float(
            self.get_parameter('lidar_sensor_y').value
        )
        self.lidar_sensor_yaw = float(
            self.get_parameter('lidar_sensor_yaw').value
        )
        self.footprint_half_length = 0.5 * max(
            0.0,
            float(self.get_parameter('footprint_length').value),
        )
        self.footprint_half_width = 0.5 * max(
            0.0,
            float(self.get_parameter('footprint_width').value),
        )
        self.footprint_safety_margin = max(
            0.0,
            float(self.get_parameter('footprint_safety_margin').value),
        )
        self.collision_reaction_time = max(
            0.0,
            float(self.get_parameter('collision_reaction_time').value),
        )
        self.collision_linear_decel = max(
            1e-3,
            float(self.get_parameter('collision_linear_decel').value),
        )
        self.collision_angular_decel = max(
            1e-3,
            float(self.get_parameter('collision_angular_decel').value),
        )
        self.collision_step_sec = max(
            0.01,
            float(self.get_parameter('collision_step_sec').value),
        )
        self.collision_max_horizon_sec = max(
            self.collision_step_sec,
            float(self.get_parameter('collision_max_horizon_sec').value),
        )

        self.output_pub = None
        self.last_input_time: Optional[float] = None
        self.last_lidar_time: Optional[float] = None
        self.latest_lidar_points: List[Tuple[float, float]] = []
        self.emergency_stop_active = self.require_emergency_stop_clear
        self.zero_sent = False
        self.last_forward_log_time = 0.0

        self.create_subscription(
            TwistStamped,
            self.input_topic,
            self.command_callback,
            10,
        )

        if not self.output_enabled:
            self.get_logger().warn(
                'Jackal output disabled; no real cmd_vel publisher was created.'
            )
            return

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.output_pub = self.create_publisher(
            Twist,
            self.output_topic,
            output_qos,
        )
        if self.require_emergency_stop_clear:
            self.create_subscription(
                Bool,
                self.emergency_stop_topic,
                self.emergency_stop_callback,
                10,
            )
        if self.require_nav_trigger:
            self.create_subscription(
                Bool,
                self.nav_enabled_topic,
                self.nav_trigger_callback,
                10,
            )
            self.get_logger().warn(
                'Nav trigger gate ENABLED: blocking output until '
                f'{self.nav_enabled_topic} publishes True.'
            )
        if self.lidar_safety_enabled:
            self.create_subscription(
                LaserScan,
                self.lidar_topic,
                self.lidar_callback,
                qos_profile_sensor_data,
            )
        self.create_timer(0.1, self.watchdog_callback)
        self.get_logger().warn(
            f'Jackal output ENABLED: {self.input_topic} -> {self.output_topic}'
        )
        if self.lidar_safety_enabled:
            self.get_logger().warn(
                'LiDAR safety ENABLED: '
                f'topic={self.lidar_topic}, '
                'mode=transparent-or-full-veto, '
                f'footprint={2.0 * self.footprint_half_length:.2f}x'
                f'{2.0 * self.footprint_half_width:.2f} m, '
                f'margin={self.footprint_safety_margin:.2f} m, '
                f'reaction={self.collision_reaction_time:.2f} s'
            )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def emergency_stop_callback(self, msg: Bool):
        was_active = self.emergency_stop_active
        self.emergency_stop_active = bool(msg.data)
        if self.emergency_stop_active:
            self.publish_zero_once('emergency stop active')
        elif was_active:
            self.get_logger().warn('Emergency stop is clear; commands may pass.')

    def nav_trigger_callback(self, msg: Bool):
        was_enabled = self.nav_trigger_enabled
        self.nav_trigger_enabled = bool(msg.data)
        if not self.nav_trigger_enabled:
            self.publish_zero_once('nav trigger stopped/not started')
        elif not was_enabled:
            self.get_logger().warn('Nav trigger enabled; commands may pass.')

    def lidar_callback(self, msg: LaserScan):
        self.latest_lidar_points = laser_scan_points_base(
            msg,
            self.lidar_min_range,
            self.lidar_max_range,
            self.lidar_sensor_x,
            self.lidar_sensor_y,
            self.lidar_sensor_yaw,
        )
        self.last_lidar_time = self.now_sec()

    def filter_lidar_command(self, linear: float, angular: float):
        if not self.lidar_safety_enabled:
            return linear, angular, []
        if self.last_lidar_time is None:
            return 0.0, 0.0, ['waiting for first lidar scan']
        age = self.now_sec() - self.last_lidar_time
        if age > self.lidar_timeout:
            return 0.0, 0.0, [f'lidar scan timeout ({age:.2f}s)']
        if not self.latest_lidar_points:
            return 0.0, 0.0, ['lidar scan contains no samples']
        collision = swept_footprint_collision(
            self.latest_lidar_points,
            linear,
            angular,
            self.footprint_half_length,
            self.footprint_half_width,
            self.footprint_safety_margin,
            self.collision_reaction_time,
            self.collision_linear_decel,
            self.collision_angular_decel,
            self.collision_step_sec,
            self.collision_max_horizon_sec,
        )
        if collision is None:
            return float(linear), float(angular), []
        point_x, point_y, collision_time = collision
        return 0.0, 0.0, [
            'predicted footprint collision: '
            f'point=({point_x:.2f},{point_y:.2f}) m, '
            f't={collision_time:.2f} s'
        ]

    def command_callback(self, msg: TwistStamped):
        if self.output_pub is None:
            return

        self.last_input_time = self.now_sec()
        if self.emergency_stop_active:
            self.publish_zero_once('waiting for clear emergency-stop state')
            return
        if self.require_nav_trigger and not self.nav_trigger_enabled:
            self.publish_zero_once('waiting for nav trigger start')
            return

        output = Twist()
        linear = clamp(
            float(msg.twist.linear.x),
            -self.max_linear_speed,
            self.max_linear_speed,
        )
        angular = clamp(
            float(msg.twist.angular.z),
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        linear, angular, lidar_reasons = self.filter_lidar_command(
            linear,
            angular,
        )
        if abs(linear) <= 1e-9 and abs(angular) <= 1e-9 and lidar_reasons:
            self.publish_zero_once('; '.join(lidar_reasons))
            return
        output.linear.x = linear
        output.angular.z = angular
        self.output_pub.publish(output)
        now = self.now_sec()
        if self.zero_sent or now - self.last_forward_log_time >= 2.0:
            self.get_logger().info(
                'Forwarding command: '
                f'v={output.linear.x:.3f}, w={output.angular.z:.3f}'
                + (f", lidar_veto={'; '.join(lidar_reasons)}" if lidar_reasons else '')
            )
            self.last_forward_log_time = now
        self.zero_sent = False

    def watchdog_callback(self):
        if self.output_pub is None:
            return
        if self.lidar_safety_enabled:
            if self.last_lidar_time is None:
                self.publish_zero_once('waiting for first lidar scan')
                return
            lidar_age = self.now_sec() - self.last_lidar_time
            if lidar_age > self.lidar_timeout:
                self.publish_zero_once(
                    f'lidar scan timeout ({lidar_age:.2f}s)'
                )
                return
        if self.last_input_time is None:
            if self.continuous_zero_output:
                self.publish_zero_once(
                    'waiting for first input command',
                    repeat=True,
                )
            return
        if self.now_sec() - self.last_input_time > self.input_timeout:
            self.publish_zero_once(
                'input command timeout',
                repeat=self.continuous_zero_output,
            )

    def publish_zero_once(self, reason: str, repeat: bool = False):
        if self.output_pub is None or (self.zero_sent and not repeat):
            return
        self.output_pub.publish(Twist())
        should_log = not self.zero_sent
        self.zero_sent = True
        if should_log:
            self.get_logger().warn(f'Publishing zero Twist: {reason}')


def main(args=None):
    rclpy.init(args=args)
    node = JackalTwistAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
