import rclpy

from social_nav_diffusion_ros.policy_cmd_vel_node import PolicyCmdVelNode


def make_node():
    rclpy.init(args=[
        '--ros-args',
        '-p',
        'use_diffusion_policy:=false',
        '-p',
        'disable_policy_command_publish:=true',
        '-p',
        'enable_policy_warmup:=false',
    ])
    return PolicyCmdVelNode()


def destroy_node(node):
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_prepare_for_shutdown_cancels_timers_and_keeps_output_disabled():
    node = make_node()
    try:
        node.prepare_for_shutdown(join_timeout_sec=0.0)

        assert node._shutdown_event.is_set()
        assert node.cmd_pub is None
        assert node.cmd_timer.is_canceled()
        assert node.policy_debug_timer.is_canceled()
        assert node.policy_warmup_watchdog_timer.is_canceled()
    finally:
        destroy_node(node)


def test_shutdown_blocks_new_diffusion_thread():
    node = make_node()
    try:
        node.use_diffusion_policy = True
        node.diffusion_adapter = object()
        node.policy_ready_for_goal = True
        node.prepare_for_shutdown(join_timeout_sec=0.0)

        node.diffusion_inference_callback()

        assert node._diffusion_thread is None
        assert not node._diffusion_inference_running
    finally:
        destroy_node(node)
