import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'social_nav_diffusion_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config', glob('config/*.rviz')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/scripts', glob('scripts/*.py')),
        ('share/' + package_name + '/experiment_setup/hunav/scenarios',
            glob('experiment_setup/hunav/scenarios/*.yaml')),
        ('share/' + package_name + '/experiment_setup/hunav/behavior_trees',
            glob('experiment_setup/hunav/behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='ROS 2 wrapper for SocialNavDiffusion local policy on Clearpath Jackal.',
    license='TODO: confirm project license before public release',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'social_nav_diffusion_node = social_nav_diffusion_ros.social_nav_diffusion_node:main',
            'social_nav_diffusion_node_test_mode = social_nav_diffusion_ros.social_nav_diffusion_node_test_mode:main',
            'policy_cmd_vel_node = social_nav_diffusion_ros.policy_cmd_vel_node:main',
            'nav2_goal_to_pose_bridge = social_nav_diffusion_ros.nav2_goal_to_pose_bridge:main',
            'jackal_twist_adapter = social_nav_diffusion_ros.jackal_twist_adapter:main',
            'rgbd_people_detector = social_nav_diffusion_ros.rgbd_people_detector:main',
            'ps4_nav_trigger_node = social_nav_diffusion_ros.ps4_nav_trigger_node:main',
        ],
    },
)
