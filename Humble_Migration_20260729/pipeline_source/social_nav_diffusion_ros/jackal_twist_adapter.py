#!/usr/bin/env python3

import math
from typing import Dict, Optional

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


def directional_scan_clearances(
    scan: LaserScan,
    configured_min_range: float,
    configured_max_range: float,
    front_half_angle: float,
    rear_half_angle: float,
) -> Dict[str, float]:
    lower = max(float(scan.range_min), float(configured_min_range), 0.0)
    upper_candidates = [float(configured_max_range)]
    if math.isfinite(float(scan.range_max)) and float(scan.range_max) > 0.0:
        upper_candidates.append(float(scan.range_max))
    upper = min(upper_candidates)
    clearances = {
        'front': float('inf'),
        'rear': float('inf'),
        'all': float('inf'),
        'samples': float(len(scan.ranges)),
    }
    if upper <= lower:
        return clearances

    angle = float(scan.angle_min)
    increment = float(scan.angle_increment)
    for raw_range in scan.ranges:
        distance = float(raw_range)
        if math.isfinite(distance) and lower <= distance <= upper:
            wrapped = normalize_angle(angle)
            clearances['all'] = min(clearances['all'], distance)
            if abs(wrapped) <= front_half_angle:
                clearances['front'] = min(clearances['front'], distance)
            if abs(abs(wrapped) - math.pi) <= rear_half_angle:
                clearances['rear'] = min(clearances['rear'], distance)
        angle += increment
    return clearances


class JackalTwistAdapter(Node):
    def __init__(self):
        super().__init__('jackal_twist_adapter')

        self.declare_parameter('input_topic', '/debug_cmd_vel')
        self.declare_parameter('output_topic', '/jackal1/cmd_vel')
        self.declare_parameter(
            'emergency_stop_topic',
            '/jackal1/platform/emergency_stop',
        )
        self.declare_parameter('enable_output', False)
        self.declare_parameter('require_emergency_stop_clear', True)
        self.declare_parameter('continuous_zero_output', False)
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter(
            'max_angular_speed',
            1.5707963267948966,
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
        self.declare_parameter('lidar_stop_distance', 0.55)
        self.declare_parameter('lidar_slow_distance', 1.0)
        self.declare_parameter('lidar_rotation_stop_distance', 0.55)
        self.declare_parameter('lidar_front_half_angle', 1.0471975512)
        self.declare_parameter('lidar_rear_half_angle', 1.0471975512)

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
        self.lidar_stop_distance = max(
            self.lidar_min_range,
            float(self.get_parameter('lidar_stop_distance').value),
        )
        self.lidar_slow_distance = max(
            self.lidar_stop_distance + 1e-3,
            float(self.get_parameter('lidar_slow_distance').value),
        )
        self.lidar_rotation_stop_distance = max(
            self.lidar_min_range,
            float(
                self.get_parameter('lidar_rotation_stop_distance').value
            ),
        )
        self.lidar_front_half_angle = abs(
            float(self.get_parameter('lidar_front_half_angle').value)
        )
        self.lidar_rear_half_angle = abs(
            float(self.get_parameter('lidar_rear_half_angle').value)
        )

        self.output_pub = None
        self.last_input_time: Optional[float] = None
        self.last_lidar_time: Optional[float] = None
        self.lidar_clearances = {
            'front': float('inf'),
            'rear': float('inf'),
            'all': float('inf'),
            'samples': 0.0,
        }
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
                f'stop={self.lidar_stop_distance:.2f} m, '
                f'slow={self.lidar_slow_distance:.2f} m'
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

    def lidar_callback(self, msg: LaserScan):
        self.lidar_clearances = directional_scan_clearances(
            msg,
            self.lidar_min_range,
            self.lidar_max_range,
            self.lidar_front_half_angle,
            self.lidar_rear_half_angle,
        )
        self.last_lidar_time = self.now_sec()

    def lidar_command_scale(self, linear: float, angular: float):
        if not self.lidar_safety_enabled:
            return 1.0, ""
        if self.last_lidar_time is None:
            return 0.0, 'waiting for first lidar scan'
        age = self.now_sec() - self.last_lidar_time
        if age > self.lidar_timeout:
            return 0.0, f'lidar scan timeout ({age:.2f}s)'
        if self.lidar_clearances['samples'] <= 0:
            return 0.0, 'lidar scan contains no samples'

        if (
            abs(angular) > 1e-3
            and self.lidar_clearances['all']
            <= self.lidar_rotation_stop_distance
        ):
            return 0.0, 'obstacle inside rotation safety radius'

        if linear > 1e-3:
            clearance = self.lidar_clearances['front']
            direction = 'front'
        elif linear < -1e-3:
            clearance = self.lidar_clearances['rear']
            direction = 'rear'
        else:
            return 1.0, ""

        if clearance <= self.lidar_stop_distance:
            return 0.0, f'{direction} obstacle inside stop distance'
        if clearance >= self.lidar_slow_distance:
            return 1.0, ""
        scale = (
            (clearance - self.lidar_stop_distance)
            / (self.lidar_slow_distance - self.lidar_stop_distance)
        )
        return clamp(scale, 0.0, 1.0), f'{direction} obstacle slowdown'

    def command_callback(self, msg: TwistStamped):
        if self.output_pub is None:
            return

        self.last_input_time = self.now_sec()
        if self.emergency_stop_active:
            self.publish_zero_once('waiting for clear emergency-stop state')
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
        lidar_scale, lidar_reason = self.lidar_command_scale(linear, angular)
        if lidar_scale <= 0.0:
            self.publish_zero_once(lidar_reason)
            return
        output.linear.x = linear * lidar_scale
        output.angular.z = angular
        self.output_pub.publish(output)
        now = self.now_sec()
        if self.zero_sent or now - self.last_forward_log_time >= 2.0:
            self.get_logger().info(
                'Forwarding command: '
                f'v={output.linear.x:.3f}, w={output.angular.z:.3f}'
                + (
                    f', lidar_scale={lidar_scale:.2f}'
                    if lidar_scale < 1.0
                    else ''
                )
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
