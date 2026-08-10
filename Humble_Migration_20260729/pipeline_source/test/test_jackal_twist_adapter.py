import math

import rclpy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from social_nav_diffusion_ros.jackal_twist_adapter import (
    JackalTwistAdapter,
    directional_scan_clearances,
)


class CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def make_node(
    enable_output,
    require_emergency_stop_clear=True,
    continuous_zero_output=False,
    enable_lidar_safety=False,
):
    rclpy.init(args=[
        '--ros-args',
        '-p',
        f'enable_output:={str(enable_output).lower()}',
        '-p',
        'require_emergency_stop_clear:='
        f'{str(require_emergency_stop_clear).lower()}',
        '-p',
        'continuous_zero_output:='
        f'{str(continuous_zero_output).lower()}',
        '-p',
        'enable_lidar_safety:='
        f'{str(enable_lidar_safety).lower()}',
    ])
    return JackalTwistAdapter()


def make_scan(front=float('inf'), left=float('inf'), rear=float('inf')):
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = math.pi / 2.0
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = [rear, float('inf'), front, left, rear]
    return scan


def destroy_node(node):
    node.destroy_node()
    rclpy.shutdown()


def test_output_is_disabled_by_default():
    node = make_node(False)
    try:
        assert node.output_pub is None
    finally:
        destroy_node(node)


def test_enabled_output_is_estop_gated_and_clamped():
    node = make_node(True)
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        command = TwistStamped()
        command.twist.linear.x = 2.0
        command.twist.angular.z = -2.0

        node.command_callback(command)
        assert len(capture.messages) == 1
        assert capture.messages[-1].linear.x == 0.0
        assert capture.messages[-1].angular.z == 0.0

        node.emergency_stop_callback(Bool(data=False))
        node.command_callback(command)
        assert len(capture.messages) == 2
        assert capture.messages[-1].linear.x == 1.0
        assert math.isclose(
            capture.messages[-1].angular.z,
            -math.pi / 2.0,
        )
    finally:
        destroy_node(node)


def test_simulation_output_can_bypass_estop_gate_explicitly():
    node = make_node(True, require_emergency_stop_clear=False)
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        command = TwistStamped()
        command.twist.linear.x = 2.0
        command.twist.angular.z = -2.0

        node.command_callback(command)
        assert len(capture.messages) == 1
        assert capture.messages[-1].linear.x == 1.0
        assert math.isclose(
            capture.messages[-1].angular.z,
            -math.pi / 2.0,
        )
    finally:
        destroy_node(node)


def test_simulation_can_publish_continuous_zero_while_waiting_for_input():
    node = make_node(
        True,
        require_emergency_stop_clear=False,
        continuous_zero_output=True,
    )
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        node.watchdog_callback()
        node.watchdog_callback()

        assert len(capture.messages) == 2
        assert all(msg.linear.x == 0.0 for msg in capture.messages)
        assert all(msg.angular.z == 0.0 for msg in capture.messages)
    finally:
        destroy_node(node)


def test_directional_scan_clearances_find_front_rear_and_side():
    clearances = directional_scan_clearances(
        make_scan(front=0.8, left=0.4, rear=0.6),
        configured_min_range=0.15,
        configured_max_range=6.0,
        front_half_angle=math.pi / 3.0,
        rear_half_angle=math.pi / 3.0,
    )

    assert math.isclose(clearances['front'], 0.8, rel_tol=1e-6)
    assert math.isclose(clearances['rear'], 0.6, rel_tol=1e-6)
    assert math.isclose(clearances['all'], 0.4, rel_tol=1e-6)


def test_lidar_safety_fails_closed_and_scales_forward_motion():
    node = make_node(
        True,
        require_emergency_stop_clear=False,
        enable_lidar_safety=True,
    )
    capture = CapturePublisher()
    node.output_pub = capture
    command = TwistStamped()
    command.twist.linear.x = 0.2
    try:
        node.command_callback(command)
        assert capture.messages[-1].linear.x == 0.0

        node.lidar_callback(make_scan(front=0.775))
        node.command_callback(command)
        assert math.isclose(
            capture.messages[-1].linear.x,
            0.1,
            rel_tol=1e-6,
        )

        node.lidar_callback(make_scan(front=0.4))
        node.command_callback(command)
        assert capture.messages[-1].linear.x == 0.0
        assert capture.messages[-1].angular.z == 0.0
    finally:
        destroy_node(node)


def test_lidar_safety_blocks_rotation_near_side_obstacle():
    node = make_node(
        True,
        require_emergency_stop_clear=False,
        enable_lidar_safety=True,
    )
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        node.lidar_callback(make_scan(left=0.4))
        command = TwistStamped()
        command.twist.angular.z = 0.3
        node.command_callback(command)

        assert capture.messages[-1].linear.x == 0.0
        assert capture.messages[-1].angular.z == 0.0
    finally:
        destroy_node(node)
