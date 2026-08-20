import math

import rclpy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from social_nav_diffusion_ros.jackal_twist_adapter import (
    JackalTwistAdapter,
    braking_poses,
    laser_scan_points_base,
    swept_footprint_collision,
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
    require_nav_trigger=False,
    test_mode=True,
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
        '-p',
        'require_nav_trigger:='
        f'{str(require_nav_trigger).lower()}',
        '-p',
        f'test_mode:={str(test_mode).lower()}',
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
        command.twist.angular.z = -4.0

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
            -3.14,
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
        command.twist.angular.z = -4.0

        node.command_callback(command)
        assert len(capture.messages) == 1
        assert capture.messages[-1].linear.x == 1.0
        assert math.isclose(
            capture.messages[-1].angular.z,
            -3.14,
        )
    finally:
        destroy_node(node)


def test_nav_trigger_gate_blocks_until_enabled_and_off_by_default():
    # Off by default: existing behavior is unaffected when the feature isn't opted into.
    node = make_node(True, require_emergency_stop_clear=False)
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        command = TwistStamped()
        command.twist.linear.x = 0.5
        node.command_callback(command)
        assert capture.messages[-1].linear.x == 0.5
    finally:
        destroy_node(node)

    # Opted in: blocked until the trigger node publishes True, then passes,
    # then blocks again once it publishes False.
    node = make_node(
        True,
        require_emergency_stop_clear=False,
        require_nav_trigger=True,
    )
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        command = TwistStamped()
        command.twist.linear.x = 0.5

        node.command_callback(command)
        assert capture.messages[-1].linear.x == 0.0

        node.nav_trigger_callback(Bool(data=True))
        node.command_callback(command)
        assert capture.messages[-1].linear.x == 0.5

        node.nav_trigger_callback(Bool(data=False))
        node.command_callback(command)
        assert capture.messages[-1].linear.x == 0.0
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


def collision_for(points, linear, angular):
    return swept_footprint_collision(
        points,
        linear,
        angular,
        footprint_half_length=0.255,
        footprint_half_width=0.215,
        safety_margin=0.05,
        reaction_time=0.15,
        linear_decel=1.5,
        angular_decel=3.14,
        step_sec=0.05,
        max_horizon_sec=1.5,
    )


def test_scan_points_are_transformed_from_lidar_to_base():
    points = laser_scan_points_base(
        make_scan(front=0.8, left=0.4, rear=0.6),
        configured_min_range=0.15,
        configured_max_range=6.0,
        sensor_x=0.12,
        sensor_y=0.0,
        sensor_yaw=0.0,
    )

    assert any(
        math.isclose(x, 0.92, abs_tol=1e-6)
        and math.isclose(y, 0.0, abs_tol=1e-6)
        for x, y in points
    )
    assert any(
        math.isclose(x, 0.12, abs_tol=1e-6)
        and math.isclose(y, 0.4, abs_tol=1e-6)
        for x, y in points
    )


def test_braking_sweep_detects_straight_collision_and_ignores_side_point():
    poses = braking_poses(1.0, 0.0, 0.15, 1.5, 3.14, 0.05, 1.5)
    assert poses[-1][0] > 0.4
    assert collision_for([(0.65, 0.0)], 1.0, 0.0) is not None
    assert collision_for([(0.65, 0.4)], 1.0, 0.0) is None


def test_braking_sweep_is_command_direction_aware():
    obstacle = [(0.45, 0.35)]
    assert collision_for(obstacle, 0.8, 1.0) is not None
    assert collision_for(obstacle, 0.8, -1.0) is None


def test_lidar_safety_fails_closed_without_overriding_clear_motion():
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

        node.lidar_callback(make_scan(front=0.8))
        node.command_callback(command)
        assert math.isclose(
            capture.messages[-1].linear.x,
            0.2,
            rel_tol=1e-6,
        )

        node.lidar_callback(make_scan(front=0.2))
        node.command_callback(command)
        assert capture.messages[-1].linear.x == 0.0
        assert capture.messages[-1].angular.z == 0.0
    finally:
        destroy_node(node)


def test_collision_veto_clears_the_whole_command():
    node = make_node(
        True,
        require_emergency_stop_clear=False,
        enable_lidar_safety=True,
    )
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        node.latest_lidar_points = [(0.45, 0.35)]
        node.last_lidar_time = node.now_sec()
        command = TwistStamped()
        command.twist.linear.x = 0.8
        command.twist.angular.z = 1.0
        node.command_callback(command)

        assert capture.messages[-1].linear.x == 0.0
        assert capture.messages[-1].angular.z == 0.0
    finally:
        destroy_node(node)


def test_safe_alternative_command_passes_without_component_changes():
    node = make_node(
        True,
        require_emergency_stop_clear=False,
        enable_lidar_safety=True,
    )
    capture = CapturePublisher()
    node.output_pub = capture
    try:
        node.latest_lidar_points = [(0.45, 0.35)]
        node.last_lidar_time = node.now_sec()
        command = TwistStamped()
        command.twist.linear.x = 0.8
        command.twist.angular.z = -1.0
        node.command_callback(command)

        assert capture.messages[-1].linear.x == 0.8
        assert capture.messages[-1].angular.z == -1.0
    finally:
        destroy_node(node)
