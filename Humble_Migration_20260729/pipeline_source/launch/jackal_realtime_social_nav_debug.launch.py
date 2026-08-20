import re
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def runtime_test_mode_default():
    config_path = Path(__file__).resolve().parents[1] / 'config' / 'runtime_mode.yaml'
    try:
        text = config_path.read_text(encoding='utf-8')
    except OSError:
        return 'false'
    match = re.search(r'^\s*test_mode\s*:\s*(true|false)\s*$', text, re.I | re.M)
    return match.group(1).lower() if match else 'false'


def generate_launch_description():
    start_people_detector = LaunchConfiguration('start_people_detector')
    start_policy = LaunchConfiguration('start_policy')
    start_goal_bridge = LaunchConfiguration('start_goal_bridge')
    start_rviz = LaunchConfiguration('start_rviz')
    start_ps4_trigger = LaunchConfiguration('start_ps4_trigger')
    test_mode = LaunchConfiguration('test_mode')
    map_topic = LaunchConfiguration('map_topic')
    style_vector = LaunchConfiguration('style_vector')
    goal_distance_m = LaunchConfiguration('goal_distance_m')
    trigger_button_index = LaunchConfiguration('trigger_button_index')
    record_bag = LaunchConfiguration('record_bag')
    package_share = FindPackageShare('social_nav_diffusion_ros')
    tf_remaps = [
        ('/tf', '/jackal1/tf'),
        ('/tf_static', '/jackal1/tf_static'),
    ]

    people_detector = Node(
        package='social_nav_diffusion_ros',
        executable='rgbd_people_detector',
        name='rgbd_people_detector',
        output='screen',
        condition=IfCondition(start_people_detector),
        remappings=tf_remaps,
        parameters=[PathJoinSubstitution([
            package_share,
            'config',
            'rgbd_people_jackal1.yaml',
        ]), {'test_mode': test_mode}],
    )

    policy = Node(
        package='social_nav_diffusion_ros',
        executable='policy_cmd_vel_node',
        name='policy_cmd_vel_node',
        prefix='/home/ubuntu/social_nav_diffusion_humble_venv/bin/python ',
        output='screen',
        condition=IfCondition(start_policy),
        remappings=[*tf_remaps, ('/map', map_topic)],
        parameters=[
            PathJoinSubstitution([
                package_share,
                'config',
                'test_speed_control.yaml',
            ]),
            PathJoinSubstitution([
                package_share,
                'config',
                'topics_jackal1_live.yaml',
            ]),
            {
                'use_sim_time': False,
                'use_diffusion_policy': True,
                'disable_policy_command_publish': False,
                'cmd_vel_topic': '/debug_cmd_vel',
                'test_mode': test_mode,
                'style_vector': style_vector,
            },
        ],
    )

    ps4_trigger = Node(
        package='social_nav_diffusion_ros',
        executable='ps4_nav_trigger_node',
        name='ps4_nav_trigger_node',
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", test_mode, "' == 'true' and '",
            start_ps4_trigger, "' == 'true'",
        ])),
        remappings=tf_remaps,
        parameters=[
            PathJoinSubstitution([
                package_share,
                'config',
                'topics_jackal1_live_test_mode.yaml',
            ]),
            {
                'use_sim_time': False,
                'goal_distance_m': goal_distance_m,
                'trigger_button_index': trigger_button_index,
                'record_bag': record_bag,
            },
        ],
    )

    goal_bridge = Node(
        package='social_nav_diffusion_ros',
        executable='nav2_goal_to_pose_bridge',
        name='nav2_goal_to_pose_bridge',
        output='screen',
        condition=IfCondition(start_goal_bridge),
        remappings=tf_remaps,
        parameters=[
            PathJoinSubstitution([
                package_share,
                'config',
                'test_speed_control.yaml',
            ]),
            PathJoinSubstitution([
                package_share,
                'config',
                'topics_jackal1_live.yaml',
            ]),
            {'use_sim_time': False},
        ],
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('clearpath_viz'),
                'launch',
                'view_navigation.launch.py',
            ])
        ),
        condition=IfCondition(start_rviz),
        launch_arguments={
            'namespace': 'jackal1',
            'use_sim_time': 'false',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('start_people_detector', default_value='true'),
        DeclareLaunchArgument('start_policy', default_value='true'),
        DeclareLaunchArgument('start_goal_bridge', default_value='true'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument(
            'test_mode',
            default_value=runtime_test_mode_default(),
        ),
        DeclareLaunchArgument('start_ps4_trigger', default_value='true'),
        DeclareLaunchArgument(
            'style_vector',
            default_value='[0.0, 0.0, 0.0, 0.0]',
        ),
        DeclareLaunchArgument('goal_distance_m', default_value='6.0'),
        DeclareLaunchArgument('trigger_button_index', default_value='7'),
        DeclareLaunchArgument('record_bag', default_value='true'),
        DeclareLaunchArgument('map_topic', default_value='/map'),
        SetEnvironmentVariable('SOCIAL_NAV_DIFFUSION_USE_VENV', 'false'),
        SetEnvironmentVariable(
            'SOCIAL_NAV_DIFFUSION_VENV',
            '/home/ubuntu/social_nav_diffusion_humble_venv',
        ),
        people_detector,
        policy,
        goal_bridge,
        ps4_trigger,
        rviz,
    ])
