from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_people_detector = LaunchConfiguration('start_people_detector')
    start_policy = LaunchConfiguration('start_policy')
    start_goal_bridge = LaunchConfiguration('start_goal_bridge')
    start_rviz = LaunchConfiguration('start_rviz')
    map_topic = LaunchConfiguration('map_topic')
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
        ])],
    )

    policy = Node(
        package='social_nav_diffusion_ros',
        executable='policy_cmd_vel_node',
        name='policy_cmd_vel_node',
        output='screen',
        condition=IfCondition(start_policy),
        remappings=[*tf_remaps, ('/map', map_topic)],
        parameters=[
            PathJoinSubstitution([
                package_share,
                'config',
                'angular_half_eval.yaml',
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
                'angular_half_eval.yaml',
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
        DeclareLaunchArgument('map_topic', default_value='/map'),
        SetEnvironmentVariable('SOCIAL_NAV_DIFFUSION_USE_VENV', 'true'),
        SetEnvironmentVariable(
            'SOCIAL_NAV_DIFFUSION_VENV',
            '/home/ubuntu/social_nav_diffusion_humble_venv',
        ),
        people_detector,
        policy,
        goal_bridge,
        rviz,
    ])
