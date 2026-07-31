from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    topics_file = LaunchConfiguration('topics_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_diffusion_policy = LaunchConfiguration('use_diffusion_policy')
    start_goal_bridge = LaunchConfiguration('start_goal_bridge')
    tf_topic = LaunchConfiguration('tf_topic')
    tf_static_topic = LaunchConfiguration('tf_static_topic')

    default_params = PathJoinSubstitution([
        FindPackageShare('social_nav_diffusion_ros'),
        'config',
        'raw_eval.yaml',
    ])
    default_topics = PathJoinSubstitution([
        FindPackageShare('social_nav_diffusion_ros'),
        'config',
        'topics.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('topics_file', default_value=default_topics),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_diffusion_policy', default_value='true'),
        DeclareLaunchArgument('start_goal_bridge', default_value='true'),
        DeclareLaunchArgument('tf_topic', default_value='/cpr_j100_0001/tf'),
        DeclareLaunchArgument('tf_static_topic', default_value='/cpr_j100_0001/tf_static'),
        SetEnvironmentVariable('SOCIAL_NAV_DIFFUSION_USE_VENV', use_diffusion_policy),
        Node(
            package='social_nav_diffusion_ros',
            executable='policy_cmd_vel_node',
            name='policy_cmd_vel_node',
            output='screen',
            remappings=[
                ('/tf', tf_topic),
                ('/tf_static', tf_static_topic),
            ],
            parameters=[
                params_file,
                topics_file,
                {
                    'use_sim_time': use_sim_time,
                    'use_diffusion_policy': use_diffusion_policy,
                },
            ],
        ),
        Node(
            package='social_nav_diffusion_ros',
            executable='nav2_goal_to_pose_bridge',
            name='nav2_goal_to_pose_bridge',
            output='screen',
            condition=IfCondition(start_goal_bridge),
            remappings=[
                ('/tf', tf_topic),
                ('/tf_static', tf_static_topic),
            ],
            parameters=[
                params_file,
                topics_file,
                {
                    'use_sim_time': use_sim_time,
                },
            ],
        ),
    ])
