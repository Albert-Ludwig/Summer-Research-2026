from types import SimpleNamespace

from social_nav_diffusion_ros.policy_cmd_vel_node import (
    PolicyCmdVelNode,
    parse_people_detector_status,
)


def test_people_detector_status_accepts_only_explicit_ready_true():
    assert parse_people_detector_status('{"ready": true}') == (True, 'ready')
    assert parse_people_detector_status(
        '{"ready": false, "reason": "rgbd_source_stale"}'
    ) == (False, 'rgbd_source_stale')


def test_people_detector_status_fails_closed_on_invalid_payload():
    ready, reason = parse_people_detector_status('not-json')

    assert ready is False
    assert reason == 'invalid status JSON'


def test_policy_input_gate_stops_when_people_detector_is_not_ready():
    parameters = {
        'stale_timeout_sec': SimpleNamespace(value=1.0),
        'goal_timeout_sec': SimpleNamespace(value=150.0),
        'ignore_people_for_policy': SimpleNamespace(value=False),
        'require_people_stream': SimpleNamespace(value=True),
    }
    node = SimpleNamespace(
        latest_goal=object(),
        latest_odom_time=10.0,
        live_lidar_enabled=False,
        require_live_lidar=False,
        latest_goal_time=10.0,
        latest_people_detector_status_time=10.0,
        latest_people_detector_ready=False,
        latest_people_detector_reason='rgbd_source_stale',
        latest_people_time=10.0,
        odom_topic='/odom',
        get_parameter=lambda name: parameters[name],
        data_is_stale=lambda stamp, now, timeout: (
            stamp is None or now - stamp > timeout
        ),
    )

    inputs, reason = PolicyCmdVelNode.prepare_policy_inputs(node, 10.1)

    assert inputs is None
    assert reason == 'people detector is not ready: rgbd_source_stale'
