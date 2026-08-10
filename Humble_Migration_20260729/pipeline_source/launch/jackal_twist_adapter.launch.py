from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'input_topic',
            default_value='/debug_cmd_vel',
        ),
        DeclareLaunchArgument('output_topic', default_value='/jackal1/cmd_vel'),
        DeclareLaunchArgument(
            'emergency_stop_topic',
            default_value='/jackal1/platform/emergency_stop',
        ),
        DeclareLaunchArgument('enable_output', default_value='false'),
        DeclareLaunchArgument('max_linear_speed', default_value='1.0'),
        DeclareLaunchArgument(
            'max_angular_speed',
            default_value='1.5707963267948966',
        ),
        DeclareLaunchArgument('input_timeout', default_value='0.5'),
        DeclareLaunchArgument('enable_lidar_safety', default_value='false'),
        DeclareLaunchArgument(
            'lidar_topic',
            default_value='/jackal1/sensors/lidar3d_0/scan',
        ),
        DeclareLaunchArgument('lidar_timeout', default_value='0.4'),
        DeclareLaunchArgument('lidar_stop_distance', default_value='0.55'),
        DeclareLaunchArgument('lidar_slow_distance', default_value='1.0'),
        DeclareLaunchArgument(
            'lidar_rotation_stop_distance',
            default_value='0.55',
        ),
        Node(
            package='social_nav_diffusion_ros',
            executable='jackal_twist_adapter',
            name='jackal_twist_adapter',
            output='screen',
            parameters=[{
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'emergency_stop_topic': LaunchConfiguration(
                    'emergency_stop_topic'
                ),
                'enable_output': LaunchConfiguration('enable_output'),
                'max_linear_speed': LaunchConfiguration('max_linear_speed'),
                'max_angular_speed': LaunchConfiguration('max_angular_speed'),
                'input_timeout': LaunchConfiguration('input_timeout'),
                'enable_lidar_safety': LaunchConfiguration(
                    'enable_lidar_safety'
                ),
                'lidar_topic': LaunchConfiguration('lidar_topic'),
                'lidar_timeout': LaunchConfiguration('lidar_timeout'),
                'lidar_stop_distance': LaunchConfiguration(
                    'lidar_stop_distance'
                ),
                'lidar_slow_distance': LaunchConfiguration(
                    'lidar_slow_distance'
                ),
                'lidar_rotation_stop_distance': LaunchConfiguration(
                    'lidar_rotation_stop_distance'
                ),
            }],
        ),
    ])
