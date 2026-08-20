from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import PoseStamped

import social_nav_diffusion_ros.ps4_nav_trigger_node as trigger_module
from social_nav_diffusion_ros.ps4_nav_trigger_node import (
    DEFAULT_BAG_TOPICS,
    Ps4NavTriggerNode,
)


class CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class PendingFuture:
    def __init__(self, result=None):
        self._result = result
        self.callback = None

    def add_done_callback(self, callback):
        self.callback = callback

    def result(self):
        return self._result


class FakeActionClient:
    def __init__(self):
        self.send_future = PendingFuture()

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal_async(self, goal):
        self.goal = goal
        return self.send_future


def make_node():
    rclpy.init(args=['--ros-args', '-p', 'record_bag:=false'])
    return Ps4NavTriggerNode()


def destroy_node(node):
    node.record_bag = False
    node.destroy_node()
    rclpy.shutdown()


def test_default_bag_topics_cover_offline_replay_and_trigger_audit():
    required_topics = {
        '/jackal1/map',
        '/jackal1/robot_description',
        '/jackal1/joy_teleop/joy',
        '/social_nav_diffusion/candidate_trajectories',
        '/people_detector/markers',
        '/social_nav_diffusion/active_goal_marker',
        '/social_nav_diffusion/nav_enabled',
        '/social_nav_diffusion/style_vector',
    }

    assert required_topics.issubset(DEFAULT_BAG_TOPICS)


def test_output_and_bag_wait_for_goal_acceptance():
    node = make_node()
    capture = CapturePublisher()
    bag_starts = []
    action_client = FakeActionClient()
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.pose.orientation.w = 1.0
    try:
        node.record_bag = True
        node.nav_enabled_pub = capture
        node.action_client = action_client
        node.compute_goal_pose = lambda: goal_pose
        node.start_bag_recording = lambda: bag_starts.append(True)

        node.start_nav()

        assert node._goal_pending is True
        assert node.nav_active is False
        assert bag_starts == []
        assert not any(message.data for message in capture.messages)

        result_future = PendingFuture()
        goal_handle = SimpleNamespace(
            accepted=True,
            get_result_async=lambda: result_future,
            cancel_goal_async=lambda: None,
        )
        node._on_goal_response(PendingFuture(goal_handle))

        assert node._goal_pending is False
        assert node.nav_active is True
        assert bag_starts == [True]
        assert capture.messages[-1].data is True
        assert result_future.callback == node._on_goal_result
    finally:
        destroy_node(node)


def test_pending_goal_ignores_duplicate_start_without_button_debounce():
    node = make_node()
    starts = []
    stops = []
    try:
        node._goal_pending = True
        node.start_nav = lambda: starts.append(True)
        node.stop_nav = lambda reason: stops.append(reason)

        node.on_button_pressed()

        assert starts == []
        assert stops == []
        assert node._goal_pending is True
    finally:
        destroy_node(node)


def test_bag_recorder_uses_and_stops_its_process_group(monkeypatch, tmp_path):
    node = make_node()
    popen_calls = []
    signals = []

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(trigger_module.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(trigger_module, 'process_group_exists', lambda group_id: True)
    monkeypatch.setattr(
        trigger_module,
        'wait_for_process_group_exit',
        lambda group_id, timeout_sec, process=None: True,
    )
    monkeypatch.setattr(
        trigger_module.os,
        'killpg',
        lambda group_id, signal_number: signals.append((group_id, signal_number)),
    )
    try:
        node.bag_output_dir = str(tmp_path)
        node.bag_topics = ['/goal_pose']
        node.start_bag_recording()
        node.start_bag_recording()

        assert len(popen_calls) == 1
        assert popen_calls[0][1]['start_new_session'] is True

        node.stop_bag_recording()

        assert signals == [(4321, trigger_module.signal.SIGINT)]
        assert node._bag_process is None
        assert node._bag_process_group_id is None
    finally:
        destroy_node(node)
