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
            default_value='3.14',
        ),
        DeclareLaunchArgument('input_timeout', default_value='0.5'),
        DeclareLaunchArgument('enable_lidar_safety', default_value='false'),
        DeclareLaunchArgument(
            'lidar_topic',
            default_value='/jackal1/sensors/lidar3d_0/scan',
        ),
        DeclareLaunchArgument('lidar_timeout', default_value='0.4'),
        DeclareLaunchArgument('lidar_sensor_x', default_value='0.12'),
        DeclareLaunchArgument('lidar_sensor_y', default_value='0.0'),
        DeclareLaunchArgument('lidar_sensor_yaw', default_value='0.0'),
        DeclareLaunchArgument('footprint_length', default_value='0.51'),
        DeclareLaunchArgument('footprint_width', default_value='0.43'),
        DeclareLaunchArgument('footprint_safety_margin', default_value='0.05'),
        DeclareLaunchArgument('collision_reaction_time', default_value='0.15'),
        DeclareLaunchArgument('collision_linear_decel', default_value='1.5'),
        DeclareLaunchArgument('collision_angular_decel', default_value='3.14'),
        DeclareLaunchArgument('collision_step_sec', default_value='0.05'),
        DeclareLaunchArgument(
            'collision_max_horizon_sec',
            default_value='1.5',
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
                'lidar_sensor_x': LaunchConfiguration('lidar_sensor_x'),
                'lidar_sensor_y': LaunchConfiguration('lidar_sensor_y'),
                'lidar_sensor_yaw': LaunchConfiguration('lidar_sensor_yaw'),
                'footprint_length': LaunchConfiguration('footprint_length'),
                'footprint_width': LaunchConfiguration('footprint_width'),
                'footprint_safety_margin': LaunchConfiguration(
                    'footprint_safety_margin'
                ),
                'collision_reaction_time': LaunchConfiguration(
                    'collision_reaction_time'
                ),
                'collision_linear_decel': LaunchConfiguration(
                    'collision_linear_decel'
                ),
                'collision_angular_decel': LaunchConfiguration(
                    'collision_angular_decel'
                ),
                'collision_step_sec': LaunchConfiguration(
                    'collision_step_sec'
                ),
                'collision_max_horizon_sec': LaunchConfiguration(
                    'collision_max_horizon_sec'
                ),
            }],
        ),
    ])
